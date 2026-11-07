from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional until requirements are installed
    BM25Okapi = None

logger = logging.getLogger("nyaya_mitra")


HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(HERE, "chroma_db")
CORPUS_PATH = os.path.join(HERE, "legal_corpus.json")
COLLECTION_NAME = "legal_corpus"

ALL_DOC_TYPES = [
    "court_summons",
    "fir",
    "rent_agreement",
    "property_document",
    "legal_notice",
]

# multilingual-e5-small is small, fast, and supports English + Indian scripts.
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)
RERANK_MODEL_NAME = os.environ.get(
    "RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
RERANK_CANDIDATES = 15  # dense/BM25 candidates before reranking
BM25_CANDIDATES = 15

_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_client = None
_collection = None
_collection_missing_warned = False

_corpus_entries: Optional[list[dict]] = None
_corpus_by_id: dict[str, dict] = {}
_bm25_cache: dict[str, tuple] = {}


def _use_e5_prefix() -> bool:
    return "e5" in EMBEDDING_MODEL.lower()


def passage_text(text: str) -> str:
    """Add the E5 passage prefix when using an E5 model."""
    return f"passage: {text}" if _use_e5_prefix() else text


def _query_text(text: str) -> str:
    return f"query: {text}" if _use_e5_prefix() else text


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model | model=%s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
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


# ---------------------------------------------------------------------------
# Corpus + BM25
# ---------------------------------------------------------------------------


def _doc_types_from_entry(entry: dict) -> list[str]:
    return [doc_type for doc_type in entry.get("doc_types", []) if doc_type]


def _entry_display_text(entry: dict) -> str:
    return (
        f"[{entry.get('source', '')} {entry.get('section', '')}] "
        f"{entry.get('plain_language', entry.get('text', ''))}"
    )


def _entry_search_text(entry: dict) -> str:
    return " ".join(
        [
            entry.get("source", ""),
            entry.get("section", ""),
            entry.get("text", ""),
            entry.get("plain_language", ""),
        ]
    )


def _load_corpus() -> None:
    global _corpus_entries, _corpus_by_id
    if _corpus_entries is not None:
        return

    with open(CORPUS_PATH, encoding="utf-8") as handle:
        _corpus_entries = json.load(handle)
    _corpus_by_id = {entry["id"]: entry for entry in _corpus_entries}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _get_bm25(doc_type: Optional[str] = None):
    if BM25Okapi is None:
        return None

    _load_corpus()
    key = doc_type if doc_type in ALL_DOC_TYPES else "all"
    if key in _bm25_cache:
        return _bm25_cache[key]

    entries = _corpus_entries or []
    if key != "all":
        entries = [
            entry for entry in entries if doc_type in entry.get("doc_types", [])
        ]

    if not entries:
        _bm25_cache[key] = ([], None)
        return _bm25_cache[key]

    tokenized = [_tokenize(_entry_search_text(entry)) for entry in entries]
    _bm25_cache[key] = (entries, BM25Okapi(tokenized))
    return _bm25_cache[key]


def _bm25_hits(query: str, doc_type: Optional[str] = None) -> list[dict]:
    cached = _get_bm25(doc_type)
    if not cached:
        return []
    entries, index = cached
    if index is None or not entries:
        return []

    scores = index.get_scores(_tokenize(query))
    ranked = sorted(
        zip(entries, scores), key=lambda pair: float(pair[1]), reverse=True
    )[:BM25_CANDIDATES]

    hits = []
    for entry, score in ranked:
        if float(score) <= 0:
            continue
        hit = _entry_hit(entry)
        hit["score"] = round(float(score), 4)
        hits.append(hit)
    return hits


def _entry_hit(entry: dict) -> dict:
    source = entry.get("source", "")
    section = entry.get("section", "")
    return {
        "id": entry["id"],
        "text": entry.get("plain_language") or entry.get("text", ""),
        "source": source,
        "section": section,
        "citation": _citation(source, section),
        "doc_types": _doc_types_from_entry(entry),
        "score": None,
    }


# ---------------------------------------------------------------------------
# Structured retrieval
# ---------------------------------------------------------------------------


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


def _dense_hits(
    query: str,
    k: int,
    doc_type: Optional[str] = None,
) -> list[dict]:
    collection = _get_collection()
    if collection is None:
        return []

    where = None
    if doc_type and doc_type in ALL_DOC_TYPES:
        where = {f"doc_type_{doc_type}": True}

    query_text = _query_text(query[:2000])

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
        logger.warning("Dense retrieval failed | %s", e)
        return []

    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]

    return [
        _build_hit(
            doc_id=doc_id,
            text=document or "",
            metadata=metadata or {},
        )
        for doc_id, document, metadata in zip(ids, documents, metadatas)
        if document
    ]


def retrieve_legal_context(
    query: str,
    k: int = 3,
    doc_type: Optional[str] = None,
) -> list[dict]:
    """
    Return the top-k legal corpus entries as structured, cited hits.

    Hybrid retrieval:
      1. Dense multilingual embeddings from ChromaDB
      2. BM25 keyword search over the corpus
      3. Cross-encoder reranking of the merged candidate set

    Each hit contains id, text, source, section, citation, doc_types, score.
    """
    if not query:
        return []

    where_type = doc_type if doc_type in ALL_DOC_TYPES else None
    dense = _dense_hits(query, k=k, doc_type=where_type)
    bm25 = _bm25_hits(query, doc_type=where_type)

    for rank, hit in enumerate(dense, start=1):
        hit["_dense_rank"] = rank
    for rank, hit in enumerate(bm25, start=1):
        hit["_bm25_rank"] = rank

    def merge(dense_hits: list[dict], bm25_hits: list[dict]) -> list[dict]:
        combined: dict[str, dict] = {}
        for hit in dense_hits:
            item = combined.setdefault(hit["id"], {**hit, "_rrf": 0.0})
            item["_rrf"] += 1.0 / (60 + hit["_dense_rank"])
        for hit in bm25_hits:
            item = combined.setdefault(hit["id"], {**hit, "_rrf": 0.0})
            item["_rrf"] += 1.0 / (60 + hit["_bm25_rank"])
        return sorted(
            combined.values(), key=lambda item: item["_rrf"], reverse=True
        )[: RERANK_CANDIDATES * 2]

    candidates = merge(dense, bm25)

    if not candidates and where_type:
        # The document-type filter can be too narrow for general questions.
        dense = _dense_hits(query, k=k, doc_type=None)
        bm25 = _bm25_hits(query, doc_type=None)
        for rank, hit in enumerate(dense, start=1):
            hit["_dense_rank"] = rank
        for rank, hit in enumerate(bm25, start=1):
            hit["_bm25_rank"] = rank
        candidates = merge(dense, bm25)

    if not candidates:
        return []

    def public_hit(item: dict, score: float, rerank_score: Optional[float] = None) -> dict:
        hit = {key: value for key, value in item.items() if not key.startswith("_")}
        hit["score"] = round(score, 6)
        if rerank_score is not None:
            hit["rerank_score"] = round(rerank_score, 4)
        return hit

    # --- Cross-encoder reranking, blended with first-stage RRF ---
    try:
        reranker = _get_reranker()
        pairs = [[query, candidate["text"]] for candidate in candidates]
        scores = reranker.predict(pairs)

        if not hasattr(scores, "__len__"):
            scores = [scores]
        scores = list(scores)
        if len(scores) != len(candidates):
            raise ValueError(
                f"reranker returned {len(scores)} scores for "
                f"{len(candidates)} candidates"
            )

        reranked = sorted(
            zip(candidates, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        blended = []
        indic_query = any(
            0x0900 <= ord(char) <= 0x0D7F for char in query
        )
        rrf_weight = 0.5 if indic_query else 0.8
        for rerank_rank, (candidate, rerank_score) in enumerate(reranked, start=1):
            # RRF keeps strong dense/BM25 hits from being buried by one
            # imperfect reranker prediction. For Indic-script queries the
            # multilingual reranker is more reliable, so give it more weight.
            final_score = (
                rrf_weight * candidate["_rrf"]
                + (1 - rrf_weight) / (60 + rerank_rank)
            )
            blended.append(
                public_hit(candidate, final_score, float(rerank_score))
            )

        blended.sort(key=lambda hit: hit["score"], reverse=True)
        return blended[:k]
    except Exception as e:
        logger.warning(
            "Cross-encoder reranking failed, falling back to hybrid order | %s",
            e,
        )
        return [
            public_hit(candidate, candidate["_rrf"])
            for candidate in candidates[:k]
        ]


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
