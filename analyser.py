import copy
import os
import json
import logging
from typing import List, Literal, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

from rag import format_hits_for_prompt, retrieve_legal_context

logger = logging.getLogger("nyaya_mitra")

MAX_RETRIES = 2


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

# Cloud provider used on Hugging Face Spaces (and anywhere Ollama is not
# available). Set LLM_PROVIDER=groq and provide GROQ_API_KEY.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_SECONDS = 120

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


def check_llm_available() -> None:
    """
    Verify the configured LLM provider is usable.

    For Groq this means the API key is present. For Ollama it checks that
    the server is reachable and the configured model is actually pulled.
    """
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it as a Hugging Face Space secret "
                "or a local environment variable, then restart the app."
            )
        return

    if LLM_PROVIDER != "ollama":
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'ollama' or 'groq'."
        )

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

Target language for all explanations: {language}

Relevant Legal Context:
{legal_context}

Dates automatically detected in the document (use this as a checklist — include every genuinely relevant one in critical_dates, and ignore only obvious non-dates):
{date_hints}

Document Text:
{doc_text}

Rules for language:
- Write simple_summary, key_conditions, what_happens_if_ignored, urgency_reason, your_rights, next_steps, audio_summary, party roles, and the label/importance of each date in {language}.
- Keep JSON keys in English and keep urgency_level exactly HIGH, MEDIUM, or LOW.
- Keep names, dates, case numbers, and section numbers exactly as written.
- If the document text is in a different language, still explain everything in {language}.

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
  "audio_summary": "<A 4-6 sentence spoken summary in {language} that will be read aloud. Use simple words. Avoid legal jargon. Include the most important date, what to do, and what happens if ignored.>"
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


def _call_ollama(
    prompt: str,
    json_mode: bool,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"

    try:
        response = httpx.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS}s. "
            "Local inference can be slow on first run (model loading into memory) "
            "or on machines without much RAM/GPU — try again, or use a smaller model."
        ) from None
    except httpx.RequestError as e:
        raise RuntimeError(
            f"Lost connection to Ollama at {OLLAMA_HOST} mid-request: {e}"
        ) from e

    data = response.json()
    content = data.get("message", {}).get("content", "")
    return safe_text(content.strip())


def _call_groq(
    prompt: str,
    json_mode: bool,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None,
) -> str:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it as a Hugging Face Space secret "
            "or a local environment variable."
        )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    try:
        response = httpx.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=GROQ_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Groq did not respond within {GROQ_TIMEOUT_SECONDS}s. "
            "Please try again."
        ) from None
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:400]
        raise RuntimeError(
            f"Groq API error {e.response.status_code}: {detail}"
        ) from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Could not reach the Groq API: {e}") from e

    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return safe_text(content.strip())


def call_llm(
    prompt: str,
    json_mode: bool = True,
    max_tokens: int = 3000,
    temperature: float = 0.1,
    system_prompt: str | None = None,
) -> str:
    """
    Call the configured LLM provider.

    json_mode=True constrains generation to valid JSON syntax for the
    structured analysis. Chat answers use json_mode=False for normal prose.
    """
    if LLM_PROVIDER == "groq":
        return _call_groq(
            prompt, json_mode, max_tokens, temperature, system_prompt
        )
    if LLM_PROVIDER == "ollama":
        return _call_ollama(
            prompt, json_mode, max_tokens, temperature, system_prompt
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'ollama' or 'groq'."
    )


def get_validated_analysis(prompt: str, max_retries: int = MAX_RETRIES) -> DocumentAnalysis:
    """
    Call the local LLM, parse the JSON, and validate it against
    DocumentAnalysis. Retries with an error-correction instruction on
    failure, and returns a safe fallback if all retries are exhausted.
    """
    last_error = ""

    for attempt in range(max_retries + 1):
        try:
            raw = call_llm(prompt, json_mode=True)
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


_SCRIPT_RANGES = {
    "Hindi": (0x0900, 0x097F),     # Devanagari
    "Marathi": (0x0900, 0x097F),   # Devanagari
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
}


def _has_target_script(text: str, language: str) -> bool:
    if not text:
        return False
    script_range = _SCRIPT_RANGES.get(language)
    if not script_range:
        return True
    start, end = script_range
    return any(start <= ord(char) <= end for char in text)


def _analysis_needs_localization(analysis: dict, language: str) -> bool:
    if language == "English" or language not in _SCRIPT_RANGES:
        return False

    summary = analysis.get("simple_summary", "")
    audio_summary = analysis.get("audio_summary", "")
    return not (
        _has_target_script(summary, language)
        and _has_target_script(audio_summary, language)
    )


def _translate_text_plain(text: str, language: str) -> str:
    """Translate one field with a simple prose prompt (more reliable on 3B)."""
    if not text or not str(text).strip():
        return text

    prompt = (
        f"Translate the following legal text into {language}. "
        f"Use the {language} native script. Return only the translation, "
        "with no explanation, no English sentences, and no markdown.\n\n"
        f"{text}"
    )

    for attempt in range(2):
        try:
            result = call_llm(
                prompt,
                json_mode=False,
                max_tokens=500,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("Plain-text translation failed | %s", e)
            return text

        result = strip_markdown_fences(result).strip().strip('"').strip()
        if result and _has_target_script(result, language):
            return result

        prompt += (
            f"\n\nIMPORTANT: The previous answer was not in {language}. "
            f"Translate into {language} using its native script only."
        )

    return text


def localize_analysis(analysis: dict, language: str) -> dict:
    """
    Ensure the analysis values are in the target language.

    Small local models often follow the JSON schema but ignore the
    target-language instruction. When that happens, translate the fields
    one by one with plain-prose prompts. This is slower than Google
    Translate but has no external rate limit and keeps legal facts intact.
    """
    if not _analysis_needs_localization(analysis, language):
        return analysis

    result = copy.deepcopy(analysis)

    for key in (
        "simple_summary",
        "what_happens_if_ignored",
        "urgency_reason",
        "audio_summary",
    ):
        result[key] = _translate_text_plain(result.get(key, ""), language)

    for key in ("key_conditions", "your_rights", "next_steps"):
        result[key] = [
            _translate_text_plain(item, language) if isinstance(item, str) else item
            for item in (result.get(key) or [])
        ]

    result["parties_involved"] = [
        {
            **person,
            "role": _translate_text_plain(person.get("role", ""), language),
        }
        if isinstance(person, dict)
        else person
        for person in (result.get("parties_involved") or [])
    ]

    result["critical_dates"] = [
        {
            **item,
            "label": _translate_text_plain(item.get("label", ""), language),
            "importance": _translate_text_plain(
                item.get("importance", ""), language
            ),
        }
        if isinstance(item, dict)
        else item
        for item in (result.get("critical_dates") or [])
    ]

    result["_sources"] = analysis.get("_sources") or []
    logger.info("Analysis localized field-by-field | language=%s", language)
    return result


def analyse_document(
    text: str,
    doc_type: str,
    doc_label: str,
    date_hints: Optional[List[str]] = None,
    language: str = "English",
) -> dict:
    """
    Send document to a local Llama 3.1 model (via Ollama) for structured
    legal analysis. Returns a validated dict (from DocumentAnalysis) —
    always well-formed, even in failure cases, so downstream rendering
    code never has to guard against missing keys.
    """
    check_llm_available()

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

    hits = []
    try:
        hits = retrieve_legal_context(
            f"{doc_label}\n{doc_text}", k=3, doc_type=doc_type
        )
        legal_context = format_hits_for_prompt(hits)
    except Exception as e:
        logger.warning("RAG context retrieval failed, proceeding without it: %s", e)
        legal_context = ""

    prompt = ANALYSIS_PROMPT.format(
        doc_type=doc_label,
        language=language,
        legal_context=safe_text(legal_context),
        date_hints=date_hints_text,
        doc_text=doc_text,
    )
    prompt = clean_prompt_text(safe_text(prompt))

    result = get_validated_analysis(prompt).model_dump()
    result["_sources"] = hits
    result = localize_analysis(result, language)
    return result


def get_urgency_color(urgency: str) -> str:
    mapping = {
        "HIGH": "#FF4444",
        "MEDIUM": "#FF8C00",
        "LOW": "#28A745",
    }
    return mapping.get(urgency.upper(), "#888888")