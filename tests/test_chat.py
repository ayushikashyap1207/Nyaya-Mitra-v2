import chat


def test_answer_question_returns_sources_and_uses_document(monkeypatch):
    hits = [
        {
            "id": "ipc_302",
            "citation": "Indian Penal Code §302",
            "text": "Punishment for murder.",
            "source": "Indian Penal Code",
            "section": "302",
            "doc_types": ["fir"],
            "score": 1.0,
        }
    ]

    monkeypatch.setattr(chat, "check_llm_available", lambda: None)
    monkeypatch.setattr(chat, "retrieve_legal_context", lambda *a, **k: hits)
    monkeypatch.setattr(
        chat, "format_hits_for_prompt", lambda values: "[IPC §302] Punishment."
    )
    captured = {}

    def fake_call_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "You should appear on 5 February 2025. [Indian Penal Code §302]"

    monkeypatch.setattr(chat, "call_llm", fake_call_llm)

    result = chat.answer_question(
        "What happens if I miss the hearing?",
        document_text="You must appear on 05/02/2025.",
        analysis={"simple_summary": "A court notice.", "critical_dates": []},
        doc_type="court_summons",
        history=[("What is this?", "A court notice.")],
        language="Hindi",
    )

    assert "5 February 2025" in result["answer"]
    assert result["sources"] == hits
    assert "05/02/2025" in captured["prompt"]
    assert "What is this?" in captured["prompt"]
    assert "Hindi" in captured["prompt"]
    assert captured["kwargs"]["json_mode"] is False


def test_answer_question_rejects_empty_question():
    result = chat.answer_question("   ")
    assert result == {"answer": "", "sources": []}
