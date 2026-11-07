from __future__ import annotations

import logging
import os
from typing import Optional

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger("nyaya_mitra")


HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "legal_corpus"

ALL_DOC_TYPES = [
    "court_summons",
    "fir",
    "rent_agreement",
    "property_document",
    "legal_notice",
]

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 15  # how many bi-encoder hits to pull before reranking

_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_client = None
_collection = None
_collection_missing_warned = False


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
        logger.info("Cross-encoder reranker loaded | model=%s", RERANK_MODEL_NAME)
    return _reranker


def _get_collection():
    """
    Lazily connect to the persisted ChromaDB collection. Returns None (and
    logs once) if the index hasn't been built yet, so callers can fail soft
    rather than crashing the whole analysis pipeline over a missing RAG
    index — the rest of the app should still work without legal context.
    """
    global _client, _collection, _collection_missing_warned

    if _collection is not None:
        return _collection

    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        _collection = _client.get_collection(COLLECTION_NAME)
        logger.info(
            "Connected to ChromaDB collection '%s' | entries=%d",
            COLLECTION_NAME,
            _collection.count(),
        )
    except Exception:
        if not _collection_missing_warned:
            logger.error(
                "ChromaDB collection '%s' not found at %s — run "
                "'python build_chroma_index.py' once to build it. "
                "Proceeding without legal context in the meantime.",
                COLLECTION_NAME,
                CHROMA_PATH,
            )
            _collection_missing_warned = True
        return None

    return _collection


def _doc_types_from_metadata(metadata: dict) -> list[str]:
    doc_types = [
        doc_type
        for doc_type in ALL_DOC_TYPES
        if metadata.get(f"doc_type_{doc_type}")
    ]
    if doc_types:
        return doc_types

    display = metadata.get("doc_types_display", "")
    return [part.strip() for part in display.split(",") if part.strip()]


def _citation(source: str, section: str) -> str:
    if source and section:
        return f"{source} §{section}"
    return source or section or "Legal source"


def _strip_citation_prefix(text: str, source: str, section: str) -> str:
    """Remove the stored '[Source Section]' prefix before re-citing in prompts."""
    if not text:
        return ""
    if text.startswith("["):
        head, separator, tail = text.partition("]")
        if separator and source and source.lower() in head.lower():
            return tail.strip()
    prefix = f"[{source} {section}]"
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text


def _build_hit(
    doc_id: str,
    text: str,
    metadata: dict,
    score: Optional[float] = None,
) -> dict:
    source = metadata.get("source", "")
    section = metadata.get("section", "")
    return {
        "id": doc_id,
        "text": _strip_citation_prefix(text, source, section),
        "source": source,
        "section": section,
        "citation": _citation(source, section),
        "doc_types": _doc_types_from_metadata(metadata),
        "score": score,
    }


def retrieve_legal_context(
    query: str,
    k: int = 3,
    doc_type: Optional[str] = None,
) -> list[dict]:
    """
    Return the top-k legal corpus entries as structured hits.

    Each hit contains:
        id, text, source, section, citation, doc_types, score

    Retrieval is a two-pass process:
      1. Bi-encoder similarity search in ChromaDB pulls RERANK_CANDIDATES
         candidates.
      2. A cross-encoder reranks those candidates using (query, candidate)
         pairs, and the top-k are returned. If reranking fails, the
         bi-encoder order is used.
    """
    if not query:
        return []

    collection = _get_collection()
    if collection is None:
        return []

    where = None
    if doc_type and doc_type in ALL_DOC_TYPES:
        where = {f"doc_type_{doc_type}": True}

    query_text = query[:2000]

    try:
        model = _get_model()
        query_embedding = model.encode(
            [query_text], normalize_embeddings=True
        )[0].tolist()

        n_candidates = min(RERANK_CANDIDATES, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(n_candidates, k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.warning(
            "ChromaDB query failed, proceeding without legal context | %s", e
        )
        return []

    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    candidates = [
        _build_hit(
            doc_id=doc_id,
            text=document or "",
            metadata=metadata or {},
        )
        for doc_id, document, metadata in zip(ids, documents, metadatas)
        if document
    ]

    if not candidates:
        return []

    # --- Pass 2: cross-encoder reranking of the candidate shortlist ---
    try:
        reranker = _get_reranker()
        pairs = [[query_text, candidate["text"]] for candidate in candidates]
        scores = reranker.predict(pairs)

        # predict() returns a numpy array for normal batches, but a bare
        # scalar for a single pair. Normalise both cases before zipping.
        if not hasattr(scores, "__len__"):
            scores = [scores]
        scores = list(scores)
        if len(scores) != len(candidates):
            raise ValueError(
                f"reranker returned {len(scores)} scores for "
                f"{len(candidates)} candidates"
            )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        top_hits = []
        for candidate, score in ranked[:k]:
            hit = dict(candidate)
            hit["score"] = round(float(score), 4)
            top_hits.append(hit)
        return top_hits
    except Exception as e:
        logger.warning(
            "Cross-encoder reranking failed, falling back to bi-encoder order | %s",
            e,
        )
        return candidates[:k]


def format_hits_for_prompt(hits: list[dict]) -> str:
    """Format structured hits as cited text for an LLM prompt."""
    if not hits:
        return ""

    blocks = []
    for hit in hits:
        blocks.append(f"[{hit['citation']}]\n{hit['text']}")
    return "\n\n".join(blocks)


def get_relevant_context(
    query: str,
    k: int = 3,
    doc_type: Optional[str] = None,
) -> str:
    """Backward-compatible helper that returns cited context as one string."""
    return format_hits_for_prompt(
        retrieve_legal_context(query, k=k, doc_type=doc_type)
    )


def corpus_size() -> int:
    """Convenience helper, e.g. for logging/README stats."""
    collection = _get_collection()
    return collection.count() if collection is not None else 0
