"""
Small, dependency-free date/deadline scanner.

The LLM decides which dates are legally important, but it can occasionally
miss one. This module provides a deterministic second pass over the OCR/text
so the UI can show every date-like string found in the document as well.
"""

from __future__ import annotations

import re

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)"
)

# Month names as OCR'd from Hindi/Marathi, Tamil, and Telugu documents.
_INDIC_MONTHS = (
    r"(?:जनवरी|फ़रवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|"
    r"सितंबर|सितम्बर|अक्टूबर|अक्तूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर|"
    r"ஜனவரி|பிப்ரவரி|மார்ச்|ஏப்ரல்|மே|ஜூன்|ஜூலை|ஆகஸ்ட்|"
    r"செப்டம்பர்|அக்டோபர்|நவம்பர்|டிசம்பர்|"
    r"జనవరి|ఫిబ్రవరి|మార్చి|ఏప్రిల్|మే|జూన్|జూలై|ఆగస్టు|"
    r"సెప్టెంబర్|అక్టోబర్|నవంబర్|డిసెంబర్)"
)

_PATTERNS = [
    # 12/01/2025, 12-01-25, 12.01.2025
    re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"),
    # 2025-01-12, 2025/01/12
    re.compile(r"\b\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\b"),
    # 12th January 2025, 5th day of January, 2025
    re.compile(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?{_MONTH},?\s+\d{{2,4}}\b",
        re.IGNORECASE,
    ),
    # January 12, 2025 / Jan. 12 2025
    re.compile(
        rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",
        re.IGNORECASE,
    ),
    # 12 जनवरी 2025 / 12 ஜனவரி 2025 / 12 జనవరి 2025
    re.compile(rf"(?<!\d)\d{{1,2}}\s+{_INDIC_MONTHS}\s+\d{{4}}(?!\d)"),
    # जनवरी 12, 2025 / జనవరి 12, 2025
    re.compile(rf"(?<!\d){_INDIC_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}(?!\d)"),
    # Relative deadlines: within 30 days, after 2 weeks, 90 days from ...
    re.compile(
        r"\b(?:within|after|before)\s+\d{1,3}\s+(?:days?|weeks?|months?|years?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,3}\s+(?:days?|weeks?|months?)\s+(?:from|of|after)\b",
        re.IGNORECASE,
    ),
]


def extract_date_mentions(text: str, max_items: int = 40) -> list[str]:
    """
    Return unique date-like strings in the order they appear in the text.

    Deduplication is case-insensitive; this is deliberately just a scanner,
    not a date parser, so the original wording is preserved.
    """
    if not text:
        return []

    matches: list[tuple[int, int, str]] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            value = re.sub(r"\s+", " ", match.group(0)).strip(" .,;:-")
            if value:
                matches.append((match.start(), match.end(), value))

    # Prefer longer matches at the same position, and drop overlapping ones so
    # "within 30 days of receipt" doesn't also produce a separate "30 days of".
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    seen: set[str] = set()
    accepted_spans: list[tuple[int, int]] = []
    results: list[str] = []
    for start, end, value in matches:
        if any(start < accepted_end and end > accepted_start for accepted_start, accepted_end in accepted_spans):
            continue

        key = value.lower()
        if key in seen:
            continue

        seen.add(key)
        accepted_spans.append((start, end))
        results.append(value)
        if len(results) >= max_items:
            break

    return results
