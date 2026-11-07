from app import format_analysis_html, format_sources_html, is_text_usable


def test_html_escapes_untrusted_content():
    analysis = {
        "simple_summary": "<script>alert(1)</script>",
        "urgency_level": "HIGH",
        "urgency_reason": "Soon.",
        "critical_dates": [
            {"label": "Hearing", "date": "05/02/2025", "importance": "Attend."}
        ],
    }
    html = format_analysis_html(analysis, "Court Notice", ["05/02/2025"])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "Important dates (1)" in html


def test_sources_are_deduplicated():
    hits = [
        {"citation": "IPC §302", "text": "Murder punishment."},
        {"citation": "IPC §302", "text": "Duplicate."},
    ]
    html = format_sources_html(hits)
    assert html.count("IPC §302") == 1
    assert "Murder punishment." in html


def test_text_usability():
    assert is_text_usable("short") is False
    assert is_text_usable(
        "This is a sufficiently long legal notice with readable words."
    ) is True
