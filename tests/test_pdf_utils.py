from pdf_extractor import clean_text, is_extraction_poor


def test_clean_text_preserves_indic_scripts():
    text = "नमस्ते दुनिया\n\n\nTamil: வணக்கம்"
    cleaned = clean_text(text)
    assert "नमस्ते" in cleaned
    assert "வணக்கம்" in cleaned


def test_clean_text_collapses_extra_spaces():
    assert clean_text("hello    world") == "hello world"


def test_poor_extraction_for_empty_text():
    assert is_extraction_poor("") is True


def test_good_extraction_for_normal_text():
    text = "This is a readable legal notice with enough alphabetical content."
    assert is_extraction_poor(text) is False
