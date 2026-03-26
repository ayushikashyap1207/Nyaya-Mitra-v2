from __future__ import annotations

from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


_KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "title": "Court Summons Basics",
        "text": (
            "A court summons/notice is a formal direction to appear in court on a specific date. "
            "If a person does not appear, the court can proceed ex parte or issue a warrant depending on the case."
        ),
    },
    {
        "title": "FIR (First Information Report)",
        "text": (
            "An FIR is the first written record of a cognizable offence reported to police. "
            "It starts the criminal investigation process. The accused may seek legal counsel and bail as per law."
        ),
    },
    {
        "title": "Legal Notice",
        "text": (
            "A legal notice is a formal communication, usually by an advocate, asking the other party to comply with a demand. "
            "Ignoring it may lead to a lawsuit or further legal action."
        ),
    },
    {
        "title": "Rent / Lease Agreements",
        "text": (
            "Rent agreements define rent amount, duration, security deposit, and responsibilities. "
            "Eviction and termination clauses must comply with applicable state rent laws."
        ),
    },
    {
        "title": "Property Documents",
        "text": (
            "Property documents like sale deeds record ownership transfer and must be registered. "
            "Stamp duty and registration fees are mandatory under state laws."
        ),
    },
    {
        "title": "Court Dates and Deadlines",
        "text": (
            "Court dates are binding. If a date is missed without sufficient cause, the court may pass orders in absence. "
            "Parties should consult their lawyer and appear or seek adjournment in advance."
        ),
    },
]


_model: SentenceTransformer | None = None
_kb_texts: List[str] | None = None
_kb_embeddings: np.ndarray | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _build_index() -> None:
    global _kb_texts, _kb_embeddings
    if _kb_texts is not None and _kb_embeddings is not None:
        return

    model = _get_model()
    _kb_texts = [f"{item['title']}: {item['text']}" for item in _KNOWLEDGE_BASE]
    embeddings = model.encode(_kb_texts, normalize_embeddings=True)
    _kb_embeddings = np.asarray(embeddings, dtype="float32")


def get_relevant_context(query: str, k: int = 3) -> str:
    """Return top-k relevant knowledge base snippets as a single string."""
    if not query:
        return ""

    _build_index()
    assert _kb_texts is not None
    assert _kb_embeddings is not None

    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    scores = np.dot(_kb_embeddings, query_vec[0])
    top_k = np.argsort(scores)[::-1][:k]
    hits = [_kb_texts[idx] for idx in top_k if 0 <= idx < len(_kb_texts)]

    return "\n".join(hits)
