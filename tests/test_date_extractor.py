from date_extractor import extract_date_mentions


def test_extracts_english_dates():
    text = "Notice issued on 12th January 2025. Hearing on 05/02/2025."
    assert extract_date_mentions(text) == ["12th January 2025", "05/02/2025"]


def test_extracts_relative_deadlines():
    text = "Reply within 30 days. Produce the record within 24 hours."
    dates = extract_date_mentions(text)
    assert "within 30 days" in dates
    assert "within 24 hours" in dates


def test_extracts_notice_period():
    text = "The tenant must give 30 days notice before vacating."
    assert "30 days notice" in extract_date_mentions(text)


def test_extracts_indic_months():
    assert extract_date_mentions("सुनवाई 12 जनवरी 2025 को है।") == ["12 जनवरी 2025"]
    assert extract_date_mentions("விசாரணை 12 ஜனவரி 2025.") == ["12 ஜனவரி 2025"]
    assert extract_date_mentions("విచారణ 12 జనవరి 2025.") == ["12 జనవరి 2025"]


def test_deduplicates_overlapping_matches():
    dates = extract_date_mentions("Reply within 30 days of receipt.")
    assert dates == ["within 30 days"]


def test_no_dates_returns_empty_list():
    assert extract_date_mentions("This document has no dates.") == []


def test_respects_max_items():
    text = " ".join(f"{day}/01/2025" for day in range(1, 11))
    assert len(extract_date_mentions(text, max_items=3)) == 3
