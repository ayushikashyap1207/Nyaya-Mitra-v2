import pytest

from rag import corpus_size, get_relevant_context, retrieve_legal_context


pytestmark = pytest.mark.skipif(
    corpus_size() == 0,
    reason="Chroma index missing; run python build_chroma_index.py first",
)


def test_retrieval_returns_citation_metadata():
    hits = retrieve_legal_context("punishment for murder under IPC 302", k=3)
    assert hits
    assert all({"id", "text", "citation", "section", "source"} <= hit.keys() for hit in hits)
    assert any("302" in hit["section"] for hit in hits)


def test_relevant_context_is_cited_text():
    context = get_relevant_context("free legal aid", k=2)
    assert context
    assert "§" in context
