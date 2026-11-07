import os
import json
import logging
from typing import List, Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from rag import get_relevant_context

logger = logging.getLogger("nyaya_mitra")

MAX_RETRIES = 2


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Local inference (especially on CPU, or a first cold-start request) can be
# noticeably slower than a cloud API — give it real headroom rather than
# timing out mid-generation.
OLLAMA_TIMEOUT_SECONDS = 180


def safe_text(text: str) -> str:
    if not text:
        return ""

    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.encode("utf-8", "ignore").decode("utf-8")


def clean_prompt_text(text: str) -> str:
    """
    Remove control characters without stripping non-Latin scripts. Hindi,
    Marathi, Tamil, and Telugu document text must reach the model intact.
    """
    if not text:
        return ""
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def check_ollama_available() -> None:
    """
    Verify Ollama is reachable and the configured model is actually pulled,
    with clear, actionable error messages if not — since a raw connection
    error here isn't obvious to someone new to running local models.
    """
    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
    except httpx.RequestError as e:
        raise ValueError(
            f"Could not connect to Ollama at {OLLAMA_HOST}. "
            "Is Ollama running? Open the Ollama app, or start it from a "
            "terminal with 'ollama serve'."
        ) from e
    except httpx.HTTPStatusError as e:
        raise ValueError(f"Ollama at {OLLAMA_HOST} returned an error: {e}") from e

    models = [m.get("name", "") for m in resp.json().get("models", [])]

    def matches(installed: str, wanted: str) -> bool:
        installed_base = installed.split(":")[0]
        wanted_base = wanted.split(":")[0]
        return installed == wanted or installed_base == wanted_base

    # Model names sometimes include a ":latest" suffix, so compare the base
    # names too — but never accept llama3 when llama3.1 was requested.
    if not any(matches(installed, OLLAMA_MODEL) for installed in models):
        raise ValueError(
            f"Ollama is running, but model '{OLLAMA_MODEL}' isn't pulled yet. "
            f"Run: ollama pull {OLLAMA_MODEL}\n"
            f"(Currently available models: {', '.join(models) if models else 'none'})"
        )


# ---------------------------------------------------------------------------
# Schema — matches the actual fields analyse_document produces
# ---------------------------------------------------------------------------

class Party(BaseModel):
    role: str = ""
    name: str = "Not specified"


class CriticalDate(BaseModel):
    label: str = ""
    date: str = ""
    importance: str = ""


class DocumentAnalysis(BaseModel):
    document_type: str = "Legal Document"
    simple_summary: str = ""
    parties_involved: List[Party] = Field(default_factory=list)
    critical_dates: List[CriticalDate] = Field(default_factory=list)
    key_conditions: List[str] = Field(default_factory=list)
    what_happens_if_ignored: str = ""
    urgency_level: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    urgency_reason: str = ""
    your_rights: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    audio_summary: str = ""


ANALYSIS_PROMPT = """You are Nyaya Mitra, an Indian legal document assistant that helps rural citizens understand legal documents.

You will be given a legal document text and its type. Your job is to extract and explain the key information in very simple, clear language that a person with no legal education can understand.

Document Type: {doc_type}

Relevant Legal Context:
{legal_context}

Dates automatically detected in the document (use this as a checklist — include every genuinely relevant one in critical_dates, and ignore only obvious non-dates):
{date_hints}

Document Text:
{doc_text}

Rules for the case summary:
- Write 3-5 short sentences covering: what this document/case is about, who is involved, what is being alleged or asked, and what the current situation is.
- Explain it like you are talking to someone who has never seen a legal paper before.
- Do not invent facts that are not in the document.

Rules for dates (very important):
- Extract EVERY date, deadline, hearing date, notice date, filing date, payment date, and limitation period mentioned in the document.
- Include dates even if they are old or already passed.
- Preserve the date exactly as written in the document. If the year is clear from context, include it.
- For relative deadlines such as "within 30 days", put the full phrase in the date field and explain what it counts from.
- Give each date a clear label saying what it is, for example "Next court hearing", "Notice issued on", "Last date to reply", or "Rent due date".
- Put the dates in the order they appear in the document. Do not skip or merge any dates.
- Never invent a date. If there are no dates, return an empty list.

Respond ONLY with a valid JSON object in this exact format (no extra text, no markdown):
{{
  "document_type": "<what kind of document this is>",
  "simple_summary": "<3-5 sentences explaining the case/document in very simple language>",
  "parties_involved": [
    {{"role": "<their role e.g. Plaintiff, Landlord>", "name": "<their name if mentioned>"}}
  ],
  "critical_dates": [
    {{"label": "<what this date is for e.g. Next hearing, Notice issued, Last date to reply>", "date": "<the date exactly as written, or the relative deadline>", "importance": "<why this date matters and what happens on/by it>"}}
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


def strip_markdown_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def extract_json_block(text: str) -> Optional[str]:
    """Best-effort extraction of a {...} block from a noisy LLM response."""
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else None


def fallback_analysis(error: str = "") -> DocumentAnalysis:
    """A safe, honest response shown when the LLM output can't be salvaged."""
    logger.warning("Falling back to default analysis | reason=%s", error)
    return DocumentAnalysis(
        document_type="Legal Document",
        simple_summary=(
            "We were unable to fully process this document automatically. "
            "Please have it reviewed by a person."
        ),
        urgency_level="MEDIUM",
        urgency_reason="Urgency could not be determined automatically — treat as important until reviewed.",
        what_happens_if_ignored=(
            "Unknown — because we could not confidently analyse this document, please do not ignore it. "
            "Consult a lawyer or legal aid center."
        ),
        your_rights=["You have the right to consult a lawyer or free legal aid clinic before acting."],
        next_steps=[
            "Contact a local legal aid center or lawyer as soon as possible.",
            "Keep the original document safe and bring it with you.",
        ],
        audio_summary=(
            "We could not fully understand this document automatically. "
            "Please show it to a lawyer or a legal aid center as soon as possible."
        ),
    )


def _call_llm(prompt: str) -> str:
    """
    Call the local Ollama server. Uses format="json" so Ollama constrains
    generation to valid JSON syntax — this doesn't enforce our specific
    schema (that's still validated separately via DocumentAnalysis), but
    it substantially cuts down on malformed/truncated JSON compared to
    free-form generation.
    """
    try:
        response = httpx.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 3000},
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS}s. "
            "Local inference can be slow on first run (model loading into memory) "
            "or on machines without much RAM/GPU — try again, or use a smaller model."
        )
    except httpx.ConnectError:
        raise RuntimeError(f"Lost connection to Ollama at {OLLAMA_HOST} mid-request.")

    data = response.json()
    content = data.get("message", {}).get("content", "")
    return safe_text(content.strip())


def get_validated_analysis(prompt: str, max_retries: int = MAX_RETRIES) -> DocumentAnalysis:
    """
    Call the local LLM, parse the JSON, and validate it against
    DocumentAnalysis. Retries with an error-correction instruction on
    failure, and returns a safe fallback if all retries are exhausted.
    """
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            raw = _call_llm(prompt)
        except Exception as e:
            last_error = f"Ollama call failed: {e}"
            logger.warning("Analysis attempt %d failed | %s", attempt + 1, last_error)
            if attempt == max_retries:
                return fallback_analysis(last_error)
            continue

        cleaned = strip_markdown_fences(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            json_block = extract_json_block(cleaned)
            if not json_block:
                last_error = "Response did not contain a parseable JSON object."
                logger.warning("Analysis attempt %d failed | %s", attempt + 1, last_error)
                if attempt == max_retries:
                    return fallback_analysis(last_error)
                prompt = _retry_prompt(prompt, last_error)
                continue
            try:
                parsed = json.loads(json_block)
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON even after extraction: {e}"
                logger.warning("Analysis attempt %d failed | %s", attempt + 1, last_error)
                if attempt == max_retries:
                    return fallback_analysis(last_error)
                prompt = _retry_prompt(prompt, last_error)
                continue

        try:
            return DocumentAnalysis(**parsed)
        except ValidationError as e:
            last_error = f"Schema validation failed: {e}"
            logger.warning("Analysis attempt %d failed | %s", attempt + 1, last_error)
            if attempt == max_retries:
                return fallback_analysis(last_error)
            prompt = _retry_prompt(prompt, last_error)

    return fallback_analysis(last_error)


def _retry_prompt(original_prompt: str, error: str) -> str:
    return (
        original_prompt
        + f"\n\nYour previous response was invalid: {error}. "
        "Return ONLY valid JSON matching the exact schema described above. No markdown, no extra text."
    )


def analyse_document(
    text: str,
    doc_type: str,
    doc_label: str,
    date_hints: Optional[List[str]] = None,
) -> dict:
    """
    Send document to a local Llama 3.1 model (via Ollama) for structured
    legal analysis. Returns a validated dict (from DocumentAnalysis) —
    always well-formed, even in failure cases, so downstream rendering
    code never has to guard against missing keys.
    """
    check_ollama_available()

    # More context = more dates found. 12k characters is roughly 5-8 typed
    # pages, or 3-4 OCR'd photo pages, which keeps local inference practical.
    doc_text = text[:12000] if len(text) > 12000 else text
    doc_text = safe_text(doc_text)
    doc_label = safe_text(doc_label)

    hints = date_hints or []
    date_hints_text = (
        "\n".join(f"- {safe_text(hint)}" for hint in hints if hint)
        if hints
        else "None detected by the automatic scanner."
    )

    try:
        legal_context = get_relevant_context(
            f"{doc_label}\n{doc_text}", doc_type=doc_type
        )
    except Exception as e:
        logger.warning("RAG context retrieval failed, proceeding without it: %s", e)
        legal_context = ""

    prompt = ANALYSIS_PROMPT.format(
        doc_type=doc_label,
        legal_context=safe_text(legal_context),
        date_hints=date_hints_text,
        doc_text=doc_text,
    )
    prompt = clean_prompt_text(safe_text(prompt))

    analysis = get_validated_analysis(prompt)
    return analysis.model_dump()


def get_urgency_color(urgency: str) -> str:
    mapping = {
        "HIGH": "#FF4444",
        "MEDIUM": "#FF8C00",
        "LOW": "#28A745",
    }
    return mapping.get(urgency.upper(), "#888888")