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

ALL_DOC_TYPES = ["court_summons", "fir", "rent_agreement", "property_document", "legal_notice"]

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
        logger.info("Connected to ChromaDB collection '%s' | entries=%d", COLLECTION_NAME, _collection.count())
    except Exception:
        if not _collection_missing_warned:
            logger.error(
                "ChromaDB collection '%s' not found at %s — run "
                "'python build_chroma_index.py' once to build it. "
                "Proceeding without legal context in the meantime.",
                COLLECTION_NAME, CHROMA_PATH,
            )
            _collection_missing_warned = True
        return None

    return _collection


def get_relevant_context(query: str, k: int = 3, doc_type: Optional[str] = None) -> str:
    """
    Return top-k relevant legal corpus snippets as a single string, for
    injection into the LLM analysis prompt.

    Retrieval happens in two passes:
      1. Bi-encoder similarity search in ChromaDB pulls RERANK_CANDIDATES
         (default 15) candidates — fast but coarse.
      2. A cross-encoder reranks those candidates by scoring (query,
         candidate) pairs jointly, and the top-k by that score are
         returned — slower but much better at judging true relevance.
         Falls back to the bi-encoder's own ranking if the reranker fails
         for any reason, so a reranker hiccup never blocks analysis.

    Args:
        query: the document text (or a summary of it) to search against.
        k: number of snippets to return.
        doc_type: optional classifier output (e.g. "fir", "rent_agreement").
            When provided and recognised, retrieval is pre-filtered to only
            entries tagged for that document type via a ChromaDB metadata
            `where` clause before similarity ranking runs.
    """
    if not query:
        return ""

    collection = _get_collection()
    if collection is None:
        return ""

    where = None
    if doc_type and doc_type in ALL_DOC_TYPES:
        where = {f"doc_type_{doc_type}": True}

    query_text = query[:2000]

    # --- Pass 1: fast bi-encoder candidate retrieval ---
    try:
        model = _get_model()
        query_embedding = model.encode([query_text], normalize_embeddings=True)[0].tolist()

        n_candidates = min(RERANK_CANDIDATES, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(n_candidates, k),
            where=where,
        )
    except Exception as e:
        logger.warning("ChromaDB query failed, proceeding without legal context | %s", e)
        return ""

    documents = results.get("documents") or [[]]
    candidates = documents[0] if documents else []

    if not candidates:
        return ""

    # --- Pass 2: cross-encoder reranking of the candidate shortlist ---
    try:
        reranker = _get_reranker()
        pairs = [[query_text, doc] for doc in candidates]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        top_hits = [doc for doc, _score in ranked[:k]]
    except Exception as e:
        logger.warning("Cross-encoder reranking failed, falling back to bi-encoder ranking | %s", e)
        top_hits = candidates[:k]

    return "\n".join(top_hits)


def corpus_size() -> int:
    """Convenience helper, e.g. for logging/README stats."""
    collection = _get_collection()
    return collection.count() if collection is not None else 0