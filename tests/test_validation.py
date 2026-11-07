import pytest
from pydantic import ValidationError

from analyser import DocumentAnalysis, fallback_analysis


def test_valid_analysis_validates():
    analysis = DocumentAnalysis(
        document_type="Court Notice",
        simple_summary="A court notice about a hearing.",
        critical_dates=[
            {"label": "Hearing", "date": "05/02/2025", "importance": "Appear."}
        ],
        urgency_level="HIGH",
    )
    assert analysis.urgency_level == "HIGH"


def test_invalid_urgency_is_rejected():
    with pytest.raises(ValidationError):
        DocumentAnalysis(urgency_level="VERY HIGH")


def test_invalid_critical_dates_type_is_rejected():
    with pytest.raises(ValidationError):
        DocumentAnalysis(critical_dates="05/02/2025")


def test_fallback_analysis_is_valid():
    fallback = fallback_analysis("test")
    assert fallback.urgency_level in {"HIGH", "MEDIUM", "LOW"}
    assert fallback.next_steps
