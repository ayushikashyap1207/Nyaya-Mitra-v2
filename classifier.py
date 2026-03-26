import re


DOCUMENT_PATTERNS = {
    "court_summons": [
        r"summon", r"appear before", r"court notice", r"district court",
        r"high court", r"supreme court", r"magistrate", r"hearing date",
        r"plaintiff", r"defendant", r"civil suit", r"criminal case",
        r"crpc", r"cr\.p\.c", r"ipc", r"section \d+",
    ],
    "fir": [
        r"first information report", r"f\.i\.r", r"\bfir\b",
        r"police station", r"station house officer", r"sho",
        r"complainant", r"accused", r"cognizable offence",
        r"investigation", r"arrested",
    ],
    "rent_agreement": [
        r"rent agreement", r"lease deed", r"rental agreement",
        r"landlord", r"tenant", r"licensee", r"licensor",
        r"monthly rent", r"security deposit", r"eviction",
        r"tenancy", r"premises", r"sq\.? ?ft",
    ],
    "property_document": [
        r"sale deed", r"property", r"plot no", r"survey no",
        r"registration", r"sub-registrar", r"stamp duty",
        r"buyer", r"seller", r"vendor", r"purchaser",
        r"khasra", r"khatauni", r"patta",
    ],
    "legal_notice": [
        r"legal notice", r"notice under", r"take notice",
        r"advocate", r"counsel", r"attorney",
        r"demand notice", r"cease and desist",
        r"within \d+ days", r"failing which",
    ],
}

DOCUMENT_LABELS = {
    "court_summons": "Court Notice / Summons",
    "fir": "FIR (First Information Report)",
    "rent_agreement": "Rent / Lease Agreement",
    "property_document": "Property Document",
    "legal_notice": "Legal Notice",
    "unknown": "Legal Document",
}


def classify_document(text: str) -> str:
    """
    Classify the legal document type based on keyword patterns.
    Returns a document type key string.
    """
    text_lower = text.lower()
    scores = {}

    for doc_type, patterns in DOCUMENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            score += len(matches)
        scores[doc_type] = score

    best_type = max(scores, key=scores.get)

    if scores[best_type] == 0:
        return "unknown"

    return best_type


def get_document_label(doc_type: str) -> str:
    return DOCUMENT_LABELS.get(doc_type, "Legal Document")
