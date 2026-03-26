
import os
import json
from groq import Groq
from rag import get_relevant_context


def safe_text(text: str) -> str:
    if not text:
        return ""

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.encode("utf-8", "ignore").decode("utf-8")


def ascii_safe(text: str) -> str:
    """Strip non-ASCII characters to avoid codec errors in HTTP libraries."""
    if not text:
        return ""
    return text.encode("ascii", "ignore").decode("ascii")

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment variables.")
    return Groq(api_key=api_key)


ANALYSIS_PROMPT = """You are Nyaya Mitra, an Indian legal document assistant that helps rural citizens understand legal documents.

You will be given a legal document text and its type. Your job is to extract and explain the key information in very simple, clear language that a person with no legal education can understand.

Document Type: {doc_type}

Relevant Legal Context:
{legal_context}

Document Text:
{doc_text}

Respond ONLY with a valid JSON object in this exact format (no extra text, no markdown):
{{
  "document_type": "<what kind of document this is>",
  "simple_summary": "<2-3 sentences explaining what this document is about in very simple language>",
  "parties_involved": [
    {{"role": "<their role e.g. Plaintiff, Landlord>", "name": "<their name if mentioned>"}}
  ],
  "critical_dates": [
    {{"label": "<what this date is for>", "date": "<the date>", "importance": "<why this date matters>"}}
  ],
  "key_conditions": [
    "<condition 1 in simple language>",
    "<condition 2 in simple language>"
  ],
  "what_happens_if_ignored": "<clearly explain the consequence of ignoring this document>",
  "urgency_level": "<HIGH or MEDIUM or LOW>",
  "urgency_reason": "<one sentence explaining why this urgency level>",
  "your_rights": [
    "<right 1 that the person has in this situation>",
    "<right 2>"
  ],
  "next_steps": [
    "<step 1 - what the person should do first>",
    "<step 2>",
    "<step 3>"
  ],
  "audio_summary": "<A 4-6 sentence spoken summary in simple conversational Hindi-friendly English that will be translated and read aloud. Use simple words. Avoid legal jargon. Include the most important date, what to do, and what happens if ignored.>"
}}"""


def analyse_document(text: str, doc_type: str, doc_label: str) -> dict:
    """
    Send document to Groq (Llama 3) for structured legal analysis.
    Returns a parsed dictionary of the analysis.
    """
    client = get_groq_client()

    doc_text = text[:6000] if len(text) > 6000 else text
    doc_text = safe_text(doc_text)
    doc_label = safe_text(doc_label)
    
    legal_context = get_relevant_context(f"{doc_label}\n{doc_text}")
    prompt = ANALYSIS_PROMPT.format(
        doc_type=doc_label,
        legal_context=safe_text(legal_context),
        doc_text=doc_text,
    )
    prompt = ascii_safe(safe_text(prompt))

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()
    raw = safe_text(raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            cleaned_json = safe_text(json_match.group())
            result = json.loads(cleaned_json)
        else:
            result = {"error": "Could not parse response", "raw": raw}

    return result


def get_urgency_color(urgency: str) -> str:
    mapping = {
        "HIGH": "#FF4444",
        "MEDIUM": "#FF8C00",
        "LOW": "#28A745",
    }
    return mapping.get(urgency.upper(), "#888888")
