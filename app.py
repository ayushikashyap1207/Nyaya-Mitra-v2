import copy
import html as html_lib
import logging
import os
import time
import traceback
import urllib.parse

import gradio as gr

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def ensure_ffmpeg() -> None:
    """Make ffmpeg/ffprobe available for Gradio audio preprocessing."""
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except Exception:
        pass


def safe_text(text) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
        .encode("utf-8", "ignore")
        .decode("utf-8")
    )


def esc(value) -> str:
    """Escape untrusted document/LLM text before inserting it into HTML."""
    return html_lib.escape(safe_text(value), quote=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("nyaya_mitra.log"), logging.StreamHandler()],
)
logger = logging.getLogger("nyaya_mitra")

from pdf_extractor import (  # noqa: E402
    OCRUnavailableError,
    SUPPORTED_UPLOAD_EXTENSIONS,
    extract_text_from_uploads,
)
from date_extractor import extract_date_mentions  # noqa: E402
from classifier import classify_document, get_document_label  # noqa: E402
from analyser import analyse_document, get_urgency_color  # noqa: E402
from audio_generator import (  # noqa: E402
    build_audio_script,
)
from chat import SUGGESTED_QUESTIONS, answer_question  # noqa: E402
from voice import speak_text, transcribe_audio  # noqa: E402


URGENCY_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
SUPPORTED_LANGUAGES = ["Hindi", "Marathi", "Tamil", "Telugu"]

# OCR language packs to load for each UI language. Marathi uses the
# Devanagari script, so the Hindi pack reads it too.
OCR_LANGUAGE_BY_UI = {
    "Hindi": "eng+hin",
    "Marathi": "eng+hin",
    "Tamil": "eng+tam",
    "Telugu": "eng+tel",
}

RESULT_LABELS = {
    "English": {
        "case_summary": "Case summary",
        "important_dates": "Important dates",
        "dates_deadlines": "Dates &amp; deadlines found",
        "all_dates": "All dates &amp; deadlines found in the document",
        "if_ignored": "⚠️ If you ignore this document",
        "what_next": "What to do next",
        "more_details": "More details",
        "sources_used": "Sources used",
        "people_involved": "People involved",
        "key_conditions": "Key conditions",
        "legal_rights": "Your legal rights",
        "high": "High priority",
        "medium": "Medium priority",
        "low": "Low priority",
    },
    "Hindi": {
        "case_summary": "केस सारांश",
        "important_dates": "महत्वपूर्ण तिथियाँ",
        "dates_deadlines": "मिली तिथियाँ और समय-सीमाएँ",
        "all_dates": "दस्तावेज़ में मिली सभी तिथियाँ और समय-सीमाएँ",
        "if_ignored": "⚠️ अगर आप इस दस्तावेज़ को नज़रअंदाज़ करते हैं",
        "what_next": "आगे क्या करें",
        "more_details": "अधिक जानकारी",
        "sources_used": "इस्तेमाल किए गए स्रोत",
        "people_involved": "शामिल लोग",
        "key_conditions": "मुख्य शर्तें",
        "legal_rights": "आपके कानूनी अधिकार",
        "high": "उच्च प्राथमिकता",
        "medium": "मध्यम प्राथमिकता",
        "low": "कम प्राथमिकता",
    },
    "Marathi": {
        "case_summary": "प्रकरण सारांश",
        "important_dates": "महत्त्वाच्या तारखा",
        "dates_deadlines": "सापडलेल्या तारखा आणि मुदती",
        "all_dates": "दस्तऐवजातील सर्व तारखा आणि मुदती",
        "if_ignored": "⚠️ जर तुम्ही हा दस्तऐवज दुर्लक्षित केला तर",
        "what_next": "पुढे काय करावे",
        "more_details": "अधिक माहिती",
        "sources_used": "वापरलेले स्रोत",
        "people_involved": "समाविष्ट व्यक्ती",
        "key_conditions": "मुख्य अटी",
        "legal_rights": "तुमचे कायदेशीर अधिकार",
        "high": "उच्च प्राधान्य",
        "medium": "मध्यम प्राधान्य",
        "low": "कमी प्राधान्य",
    },
    "Tamil": {
        "case_summary": "வழக்கு சுருக்கம்",
        "important_dates": "முக்கிய தேதிகள்",
        "dates_deadlines": "கிடைத்த தேதிகள் மற்றும் காலக்கெடுகள்",
        "all_dates": "ஆவணத்தில் உள்ள அனைத்து தேதிகள் மற்றும் காலக்கெடுகள்",
        "if_ignored": "⚠️ இந்த ஆவணத்தை புறக்கணித்தால்",
        "what_next": "அடுத்து என்ன செய்ய வேண்டும்",
        "more_details": "மேலும் விவரங்கள்",
        "sources_used": "பயன்படுத்தப்பட்ட ஆதாரங்கள்",
        "people_involved": "ஈடுபட்டவர்கள்",
        "key_conditions": "முக்கிய நிபந்தனைகள்",
        "legal_rights": "உங்கள் சட்ட உரிமைகள்",
        "high": "அதிக முன்னுரிமை",
        "medium": "நடுத்தர முன்னுரிமை",
        "low": "குறைந்த முன்னுரிமை",
    },
    "Telugu": {
        "case_summary": "కేసు సారాంశం",
        "important_dates": "ముఖ్యమైన తేదీలు",
        "dates_deadlines": "కనుగొన్న తేదీలు మరియు గడువులు",
        "all_dates": "పత్రంలోని అన్ని తేదీలు మరియు గడువులు",
        "if_ignored": "⚠️ మీరు ఈ పత్రాన్ని నిర్లక్ష్యం చేస్తే",
        "what_next": "తర్వాత ఏమి చేయాలి",
        "more_details": "మరిన్ని వివరాలు",
        "sources_used": "ఉపయోగించిన మూలాలు",
        "people_involved": "పాల్గొన్న వ్యక్తులు",
        "key_conditions": "ముఖ్యమైన షరతులు",
        "legal_rights": "మీ చట్టపరమైన హక్కులు",
        "high": "అధిక ప్రాధాన్యత",
        "medium": "మధ్యస్థ ప్రాధాన్యత",
        "low": "తక్కువ ప్రాధాన్యత",
    },
}


def labels_for(language: str) -> dict:
    return RESULT_LABELS.get(language, RESULT_LABELS["English"])

# These fields must not be translated: dates and names must stay exactly as
# they appear in the document.
KEEP_AS_IS_KEYS = {"date", "name", "urgency_level"}

EMPTY_RESULT_HTML = """
<div class="nm-empty">
  <div class="nm-empty-icon">📄</div>
  <div class="nm-empty-title">Your case summary will appear here</div>
  <div class="nm-empty-sub">Upload a PDF or photos of the scanned pages, then tap Analyse.</div>
</div>
"""


def format_analysis_html(
    analysis: dict,
    doc_label: str,
    all_dates=None,
    language: str = "English",
) -> str:
    """
    Build a simple, phone-first result view:
      1. urgency banner
      2. case summary (the main thing)
      3. important dates (the second main thing)
      4. all dates/deadlines found in the raw text
      5. consequences + next steps
      6. everything else behind a "More details" expander
    """
    if not isinstance(analysis, dict):
        return "<p class='nm-error'>Analysis failed: unexpected response.</p>"

    if "error" in analysis:
        return f"<p class='nm-error'>Analysis failed: {esc(analysis.get('error'))}</p>"

    labels = labels_for(language)
    urgency = safe_text(analysis.get("urgency_level", "MEDIUM")).upper() or "MEDIUM"
    urgency_color = get_urgency_color(urgency)
    urgency_emoji = URGENCY_EMOJI.get(urgency, "🟠")
    urgency_label = labels.get(urgency.lower(), f"{urgency.title()} priority")
    urgency_reason = analysis.get("urgency_reason", "")
    summary = analysis.get("simple_summary", "")

    parts = []

    parts.append(
        f"""
        <div class="nm-urgency" style="background:{urgency_color}12; border:1px solid {urgency_color};">
          <div class="nm-urgency-title" style="color:{urgency_color};">
            {urgency_emoji} {esc(urgency_label)}
          </div>
          {f'<div class="nm-urgency-reason">{esc(urgency_reason)}</div>' if urgency_reason else ''}
        </div>
        """
    )

    parts.append(
        f"""
        <div class="nm-card nm-summary">
          <div class="nm-doctype">{esc(doc_label or 'Legal Document')}</div>
          <div class="nm-label">{labels['case_summary']}</div>
          <div class="nm-summary-text">{esc(summary) or 'No summary could be generated.'}</div>
        </div>
        """
    )

    dates = analysis.get("critical_dates") or []
    all_dates_chips = "".join(
        f'<span class="nm-chip">{esc(value)}</span>' for value in (all_dates or [])
    )

    if dates:
        parts.append(
            f'<div class="nm-card nm-dates-card">'
            f'<div class="nm-label">{labels["important_dates"]} ({len(dates)})</div>'
        )
        for item in dates:
            if not isinstance(item, dict):
                continue
            date_value = item.get("date") or "Date not specified"
            label = item.get("label") or "Important date"
            importance = item.get("importance") or ""
            parts.append(
                f"""
                <div class="nm-date">
                  <div class="nm-date-when">{esc(date_value)}</div>
                  <div class="nm-date-body">
                    <div class="nm-date-label">{esc(label)}</div>
                    {f'<div class="nm-date-importance">{esc(importance)}</div>' if importance else ''}
                  </div>
                </div>
                """
            )
        parts.append("</div>")

        if all_dates:
            parts.append(
                f"""
                <details class="nm-card nm-details">
                  <summary>{labels['all_dates']} ({len(all_dates)})</summary>
                  <div class="nm-chips">{all_dates_chips}</div>
                </details>
                """
            )
    elif all_dates:
        parts.append(
            f"""
            <div class="nm-card nm-dates-card">
              <div class="nm-label">{labels['dates_deadlines']} ({len(all_dates)})</div>
              <div class="nm-muted" style="margin-bottom:8px;">
                The AI did not rank specific dates, so here is every date and deadline
                found in the text.
              </div>
              <div class="nm-chips">{all_dates_chips}</div>
            </div>
            """
        )
    else:
        parts.append(
            f'<div class="nm-card"><div class="nm-label">{labels["important_dates"]}</div>'
            '<div class="nm-muted">No specific dates were found in this document.</div></div>'
        )

    share_lines = []
    if summary:
        share_lines.append(summary)
    if dates:
        share_lines.append(f"{labels['important_dates']}:")
        for item in dates:
            if isinstance(item, dict):
                share_lines.append(
                    f"- {item.get('date', '')}: {item.get('label', '')}"
                )
    if share_lines:
        share_text = urllib.parse.quote("\n".join(share_lines)[:1500])
        parts.append(
            f'<a class="nm-share" href="https://wa.me/?text={share_text}" '
            'target="_blank" rel="noopener">📲 Share this summary on WhatsApp</a>'
        )

    ignored = analysis.get("what_happens_if_ignored", "")
    if ignored:
        parts.append(
            f"""
            <div class="nm-card nm-warning">
              <div class="nm-label">{labels['if_ignored']}</div>
              <div class="nm-warning-text">{esc(ignored)}</div>
            </div>
            """
        )

    steps = analysis.get("next_steps") or []
    if steps:
        parts.append(
            '<div class="nm-card nm-steps">'
            f'<div class="nm-label">{labels["what_next"]}</div>'
        )
        for index, step in enumerate(steps, start=1):
            parts.append(
                f"""
                <div class="nm-step">
                  <span class="nm-step-number">{index}</span>
                  <span class="nm-step-text">{esc(step)}</span>
                </div>
                """
            )
        parts.append("</div>")

    parties = analysis.get("parties_involved") or []
    conditions = analysis.get("key_conditions") or []
    rights = analysis.get("your_rights") or []

    if parties or conditions or rights:
        parts.append(
            '<details class="nm-card nm-details">'
            f'<summary>{labels["more_details"]}</summary>'
        )

        if parties:
            parts.append(f'<div class="nm-subhead">{labels["people_involved"]}</div>')
            for person in parties:
                if not isinstance(person, dict):
                    continue
                role = person.get("role") or "Party"
                name = person.get("name") or "Not specified"
                parts.append(
                    f'<div class="nm-person"><span class="nm-role">{esc(role)}</span>'
                    f'<span>{esc(name)}</span></div>'
                )

        if conditions:
            parts.append(f'<div class="nm-subhead">{labels["key_conditions"]}</div>')
            for condition in conditions:
                parts.append(f'<div class="nm-bullet">• {esc(condition)}</div>')

        if rights:
            parts.append(f'<div class="nm-subhead">{labels["legal_rights"]}</div>')
            for right in rights:
                parts.append(f'<div class="nm-bullet">✓ {esc(right)}</div>')

        parts.append("</details>")

    sources = analysis.get("_sources") or []
    if sources:
        parts.append(format_sources_html(sources, language=language))

    return "".join(parts)


def format_sources_html(hits, language: str = "English") -> str:
    """Render cited RAG hits below a chat answer."""
    if not hits:
        return ""

    labels = labels_for(language)
    parts = [
        '<div class="nm-card nm-sources">'
        f'<div class="nm-label">{labels["sources_used"]}</div>'
    ]
    seen = set()
    for hit in hits:
        citation = hit.get("citation") or hit.get("source") or "Legal source"
        if citation.lower() in seen:
            continue
        seen.add(citation.lower())
        parts.append(
            f'<div class="nm-source">'
            f'<div class="nm-source-citation">{esc(citation)}</div>'
            f'<div class="nm-source-text">{esc(hit.get("text", ""))}</div>'
            f'</div>'
        )
    parts.append("</div>")
    return "".join(parts)


def translate_analysis(analysis: dict, language: str) -> dict:
    """
    Legacy Google-Translate helper kept for compatibility.

    The analysis pipeline no longer calls this; analyser.localize_analysis()
    now writes the fields in the selected language directly.
    """
    from audio_generator import translate_batch

    if language == "English":
        return analysis

    paths = []
    texts = []

    def collect(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in KEEP_AS_IS_KEYS or key in {"audio_summary", "document_type", "_sources"}:
                    continue
                collect(child, path + [key])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, path + [index])
        elif isinstance(value, str):
            if value.strip():
                paths.append(path)
                texts.append(value)

    collect(analysis, [])

    if not texts:
        return analysis

    try:
        translated_values = translate_batch(texts, language)
    except Exception as e:
        logger.warning("Batch translation failed, showing original | %s", e)
        return analysis

    translated = copy.deepcopy(analysis)
    for path, value in zip(paths, translated_values):
        if not value or not str(value).strip():
            continue
        try:
            target = translated
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        except Exception:
            continue

    return translated


def is_text_usable(text: str) -> bool:
    stripped = safe_text(text).strip()
    if len(stripped) < 40:
        return False
    alpha_count = sum(ch.isalpha() for ch in stripped)
    ratio = alpha_count / max(1, len(stripped))
    return ratio >= 0.2


def error_panel(message: str) -> str:
    return f"<div class='nm-error'>{esc(message)}</div>"


def _normalise_upload_paths(files) -> list[str]:
    """
    Gradio returns different shapes depending on version and whether one or
    many files were uploaded: a path string, a NamedString, a temp-file
    wrapper, or a list of those. Normalise them all to existing paths.
    """
    if files is None:
        return []

    if isinstance(files, (str, os.PathLike)):
        items = [files]
    elif isinstance(files, dict):
        items = [files]
    else:
        try:
            items = list(files)
        except TypeError:
            items = [files]

    paths = []
    for item in items:
        if isinstance(item, (str, os.PathLike)):
            path = str(item)
        elif isinstance(item, dict):
            path = item.get("path") or item.get("name")
        else:
            path = getattr(item, "name", None) or getattr(item, "path", None)

        if path:
            paths.append(str(path))

    existing = [path for path in paths if os.path.exists(path)]
    missing = len(paths) - len(existing)
    if missing:
        logger.warning("Ignoring %d uploaded path(s) that no longer exist", missing)

    return existing


def process_document(files, language, progress=None):
    """
    Main pipeline: uploads -> OCR/text extraction -> classify -> analyse ->
    translate -> display. Extraction and analysis are blocking; everything
    else degrades gracefully so the user always gets something useful.
    """
    request_start = time.time()

    def report(value, message):
        if progress is not None:
            try:
                progress(value, desc=message)
            except Exception:
                pass

    paths = _normalise_upload_paths(files)
    if not paths:
        return (
            error_panel("Please upload a PDF or at least one photo of the pages."),
            None,
            None,
        )

    # --- Stage 1: PDF / image extraction (blocking) -------------------------
    report(0.05, "Reading the document…")
    stage_start = time.time()
    try:
        text = safe_text(
            extract_text_from_uploads(
                paths, lang=OCR_LANGUAGE_BY_UI.get(language, "eng+hin")
            )
        )
        logger.info(
            "Extraction complete | files=%d | chars=%d | time=%.2fs",
            len(paths),
            len(text),
            time.time() - stage_start,
        )
    except OCRUnavailableError as e:
        logger.error("OCR unavailable | %s", e)
        return (
            error_panel(
                "We couldn't read the scanned pages because no OCR engine is "
                "available on this computer. Install Tesseract with "
                "'brew install tesseract tesseract-lang poppler', or install "
                "EasyOCR with 'pip install easyocr'."
            ),
            None,
            None,
        )
    except ValueError as e:
        logger.warning("Extraction failed | %s", e)
        return (
            error_panel(
                "We couldn't read this file. Please make sure it is a clear PDF "
                "or a well-lit photo of the page, then try again."
            ),
            None,
            None,
        )
    except Exception:
        logger.error("Unexpected extraction failure\n%s", traceback.format_exc())
        return (
            error_panel("We couldn't read this file. Please try again."),
            None,
            None,
        )
    finally:
        # Privacy: do not keep uploaded documents around after extraction.
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass

    if not is_text_usable(text):
        logger.warning(
            "Extracted text failed the usability check | chars=%d", len(text)
        )
        return (
            error_panel(
                "The text looks unreadable. Please upload a clearer scan or photo, "
                "or a text-based PDF."
            ),
            None,
            None,
        )

    all_dates = extract_date_mentions(text)
    logger.info("Date scan | dates_found=%d", len(all_dates))

    # --- Stage 2: Classification (degrade gracefully) -----------------------
    try:
        doc_type = classify_document(text)
    except Exception as e:
        logger.warning("Classification failed, defaulting to 'unknown' | %s", e)
        doc_type = "unknown"
    doc_label = get_document_label(doc_type)
    logger.info("Classification | type=%s | label=%s", doc_type, doc_label)

    # --- Stage 3: Analysis (blocking) ---------------------------------------
    report(0.35, "Understanding the case and collecting dates…")
    stage_start = time.time()
    try:
        analysis = analyse_document(
            safe_text(text),
            safe_text(doc_type),
            safe_text(doc_label),
            date_hints=all_dates,
            language=language,
        )
        logger.info(
            "LLM analysis | urgency=%s | dates=%d | language=%s | time=%.2fs",
            analysis.get("urgency_level", "MEDIUM"),
            len(analysis.get("critical_dates") or []),
            language,
            time.time() - stage_start,
        )
    except ValueError as e:
        logger.error("Analysis setup failed | %s", e)
        return (
            error_panel(
                "The AI model is not available right now. Check the Ollama "
                "server or the configured API key/provider, then try again."
            ),
            None,
            None,
        )
    except Exception:
        logger.error("Analysis failed\n%s", traceback.format_exc())
        return (
            error_panel(
                "We had trouble analysing this document. Please try again in a moment."
            ),
            None,
            None,
        )

    # --- Stage 4: Render the localized analysis -----------------------------
    report(0.7, "Building the result…")
    try:
        html_output = format_analysis_html(
            analysis, doc_label, all_dates, language=language
        )
    except Exception:
        logger.error("HTML rendering failed\n%s", traceback.format_exc())
        return (
            error_panel(
                "We analysed the document but hit an error displaying the results. "
                "Please try again."
            ),
            None,
            None,
        )

    # --- Stage 5: Audio generation (degrade gracefully) ---------------------
    report(0.9, "Preparing the audio summary…")
    stage_start = time.time()
    audio_path = None
    try:
        # The analysis is already written in the selected language, so speak
        # it directly instead of sending it through Google Translate again.
        audio_script = safe_text(
            build_audio_script(analysis, analysis.get("urgency_level", "MEDIUM"))
        )
        audio_path = speak_text(audio_script, language)
        logger.info(
            "Audio generation | language=%s | time=%.2fs",
            language,
            time.time() - stage_start,
        )
    except Exception as e:
        logger.warning("Audio generation failed, continuing without audio | %s", e)
        audio_path = None

    logger.info("Request complete | total_time=%.2fs", time.time() - request_start)
    report(1.0, "Done")
    document_state = {
        "text": text,
        "analysis": analysis,
        "doc_type": doc_type,
        "doc_label": doc_label,
        "language": language,
    }
    return html_output, audio_path, document_state


def process_document_safe(files, language, progress=gr.Progress()):
    """Top-level catch-all so the user never sees a raw Python traceback."""
    try:
        return process_document(files, language, progress=progress)
    except Exception:
        logger.critical(
            "Unhandled exception in process_document\n%s", traceback.format_exc()
        )
        return (
            error_panel(
                "Something went wrong while processing this document. Please try "
                "again, and if the problem continues, try a different file."
            ),
            None,
            None,
        )


def chat_answer(question, history, document_state, language="Hindi"):
    """Answer a follow-up question using the uploaded document + cited RAG."""
    question = safe_text(question).strip()
    history = list(history or [])

    if not question:
        return history, "", ""

    state = document_state or {}

    try:
        result = answer_question(
            question,
            document_text=state.get("text", ""),
            analysis=state.get("analysis"),
            doc_type=state.get("doc_type"),
            history=history,
            language=language,
        )
        answer = result.get("answer") or (
            "I could not find a clear answer. Please ask a lawyer or your "
            "nearest legal aid centre."
        )
        sources_html = format_sources_html(
            result.get("sources") or [], language=language
        )
    except ValueError as e:
        logger.error("Chat model unavailable | %s", e)
        answer = (
            "The AI model is not available right now. Check the Ollama server "
            "or the configured API key/provider, then try again."
        )
        sources_html = ""
    except Exception:
        logger.error("Chat failed\n%s", traceback.format_exc())
        answer = "Something went wrong while answering. Please try again."
        sources_html = ""

    return history + [(question, answer)], sources_html, ""


def voice_ask(audio_path, history, document_state, language="Hindi"):
    """Transcribe a spoken question, answer it, and speak the answer."""
    history = list(history or [])

    if not audio_path:
        gr.Warning("Record a question first, then tap the button.")
        return history, "", "", None

    try:
        question = transcribe_audio(audio_path, language)
    except Exception as e:
        logger.error("Voice transcription failed | %s", e)
        gr.Warning(f"Could not understand the recording: {e}")
        return history, "", "", None

    new_history, sources_html, _cleared = chat_answer(
        question, history, document_state, language
    )

    answer = ""
    if new_history:
        last = new_history[-1]
        if isinstance(last, (list, tuple)) and len(last) > 1:
            answer = last[1] or ""

    audio_out = None
    if answer and not answer.startswith("The AI model is not available"):
        try:
            audio_out = speak_text(answer, language)
        except Exception as e:
            logger.warning("Voice answer generation failed | %s", e)
            gr.Warning("The answer was generated, but the audio could not be created.")

    return new_history, sources_html, question, audio_out


def play_answer(history, language="Hindi"):
    """Generate speech for the latest assistant message."""
    history = list(history or [])
    if not history:
        gr.Warning("There is no answer to play yet.")
        return None

    last = history[-1]
    answer = ""
    if isinstance(last, (list, tuple)) and len(last) > 1:
        answer = last[1] or ""
    elif isinstance(last, dict):
        answer = last.get("content", "") if last.get("role") == "assistant" else ""

    if not answer:
        gr.Warning("There is no answer to play yet.")
        return None

    try:
        return speak_text(answer, language)
    except Exception as e:
        logger.warning("Play-answer generation failed | %s", e)
        gr.Warning("Could not generate audio for this answer.")
        return None


def clear_chat():
    return [], "", "", None


CSS = """
:root {
    --nm-ink: #111827;
    --nm-muted: #6b7280;
    --nm-line: #e5e7eb;
    --nm-bg: #f8fafc;
    --nm-blue: #1d4ed8;
}

.gradio-container {
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 10px 14px 28px !important;
    background: var(--nm-bg);
}

footer {
    display: none !important;
}

#nm-header {
    text-align: center;
    padding: 14px 0 8px;
}

#nm-header .nm-kicker {
    display: inline-block;
    background: #e0e7ff;
    color: #3730a3;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .03em;
    margin-bottom: 10px;
}

#nm-header h1 {
    font-size: clamp(1.65rem, 7vw, 2.1rem);
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: var(--nm-ink);
    margin: 0;
}

#nm-header p {
    color: #4b5563;
    font-size: 15px;
    line-height: 1.45;
    margin: 8px auto 0;
    max-width: 560px;
}

#nm-header .nm-sub {
    color: var(--nm-muted);
    font-size: 13px;
    margin-top: 6px;
}

#nm-file {
    min-height: 130px;
}

#nm-analyse-btn {
    min-height: 52px;
    font-size: 16px;
    font-weight: 700;
    width: 100%;
}

#nm-audio {
    margin-top: 4px;
}

.nm-card {
    background: #ffffff;
    border: 1px solid var(--nm-line);
    border-radius: 14px;
    padding: 16px;
    margin: 0 0 12px;
    box-sizing: border-box;
    width: 100%;
}

.nm-label {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--nm-muted);
    margin-bottom: 8px;
}

.nm-doctype {
    display: inline-block;
    background: #eef2ff;
    color: #3730a3;
    border-radius: 999px;
    padding: 4px 11px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 10px;
}

.nm-summary {
    border-left: 5px solid var(--nm-blue);
}

.nm-summary-text {
    font-size: 17px;
    line-height: 1.65;
    color: var(--nm-ink);
    overflow-wrap: anywhere;
}

.nm-urgency {
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 12px;
}

.nm-urgency-title {
    font-size: 15px;
    font-weight: 800;
}

.nm-urgency-reason {
    font-size: 13px;
    line-height: 1.45;
    color: #374151;
    margin-top: 3px;
}

.nm-dates-card {
    border-color: #fcd34d;
    background: #fffdf5;
}

.nm-date {
    display: flex;
    gap: 12px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 8px;
}

.nm-date:last-child {
    margin-bottom: 0;
}

.nm-date-when {
    min-width: 130px;
    color: #92400e;
    font-size: 15px;
    font-weight: 800;
    line-height: 1.35;
    overflow-wrap: anywhere;
}

.nm-date-body {
    flex: 1;
    min-width: 0;
}

.nm-date-label {
    font-size: 15px;
    font-weight: 700;
    color: var(--nm-ink);
    line-height: 1.35;
}

.nm-date-importance {
    font-size: 13px;
    line-height: 1.45;
    color: #6b7280;
    margin-top: 3px;
}

.nm-details summary {
    cursor: pointer;
    font-size: 15px;
    font-weight: 700;
    color: #1f2937;
    list-style-position: inside;
}

.nm-chips {
    margin-top: 10px;
}

.nm-chip {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid var(--nm-line);
    border-radius: 999px;
    padding: 5px 10px;
    margin: 0 5px 5px 0;
    font-size: 13px;
    color: #374151;
}

.nm-share {
    display: inline-block;
    background: #ecfdf5;
    border: 1px solid #86efac;
    color: #166534;
    border-radius: 999px;
    padding: 9px 14px;
    margin: 0 0 12px;
    font-size: 14px;
    font-weight: 700;
    text-decoration: none;
}

.nm-share:hover {
    background: #dcfce7;
}

.nm-warning {
    border-color: #fca5a5;
    background: #fff5f5;
}

.nm-warning-text {
    font-size: 15px;
    line-height: 1.55;
    color: #7f1d1d;
}

.nm-step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 9px;
}

.nm-step:last-child {
    margin-bottom: 0;
}

.nm-step-number {
    flex-shrink: 0;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--nm-blue);
    color: #fff;
    font-size: 12px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}

.nm-step-text {
    font-size: 15px;
    line-height: 1.5;
    color: var(--nm-ink);
    overflow-wrap: anywhere;
}

.nm-subhead {
    font-size: 13px;
    font-weight: 800;
    color: #374151;
    margin: 14px 0 7px;
}

.nm-subhead:first-child {
    margin-top: 10px;
}

.nm-person {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 7px;
    font-size: 14px;
}

.nm-role {
    background: #eef2ff;
    color: #3730a3;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 700;
}

.nm-bullet {
    font-size: 14px;
    line-height: 1.5;
    color: #374151;
    margin-bottom: 6px;
}

.nm-muted {
    color: var(--nm-muted);
    font-size: 14px;
}

.nm-sources {
    border-color: #c7d2fe;
    background: #f8faff;
}

.nm-source {
    border-top: 1px solid var(--nm-line);
    padding: 10px 0;
}

.nm-source:first-of-type {
    border-top: none;
    padding-top: 0;
}

.nm-source-citation {
    color: #3730a3;
    font-size: 13px;
    font-weight: 800;
    line-height: 1.35;
}

.nm-source-text {
    color: #4b5563;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 2px;
}

.nm-chat-intro {
    color: #4b5563;
    font-size: 14px;
    line-height: 1.5;
    margin-bottom: 10px;
}

#nm-chat {
    min-height: 300px;
}

.nm-error {
    background: #fff5f5;
    border: 1px solid #fca5a5;
    color: #991b1b;
    border-radius: 12px;
    padding: 14px;
    font-size: 14px;
    line-height: 1.5;
}

.nm-empty {
    text-align: center;
    padding: 34px 18px;
    color: var(--nm-muted);
}

.nm-empty-icon {
    font-size: 38px;
    margin-bottom: 8px;
}

.nm-empty-title {
    font-size: 16px;
    font-weight: 700;
    color: #374151;
}

.nm-empty-sub {
    font-size: 13px;
    margin-top: 5px;
    line-height: 1.4;
}

#nm-footer {
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    line-height: 1.5;
    padding: 10px 4px 0;
}

@media (max-width: 640px) {
    .gradio-container {
        padding: 6px 10px 22px !important;
    }

    #nm-header {
        padding: 10px 0 6px;
    }

    #nm-header p {
        font-size: 14px;
    }

    .nm-card {
        padding: 14px;
        border-radius: 12px;
    }

    .nm-summary-text {
        font-size: 16px;
    }

    .nm-date {
        flex-direction: column;
        gap: 3px;
    }

    .nm-date-when {
        min-width: 0;
        font-size: 16px;
    }

    .nm-date-label {
        font-size: 14px;
    }

    #nm-analyse-btn {
        min-height: 54px;
        font-size: 17px;
    }

    /* Prevent iOS Safari from zooming when a form field is focused. */
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select {
        font-size: 16px !important;
    }
}
"""

SCROLL_JS = """
function () {
  const attach = () => {
    const target = document.querySelector('#nm-results');
    if (!target) {
      setTimeout(attach, 500);
      return;
    }
    let lastText = '';
    const observer = new MutationObserver(() => {
      const text = (target.innerText || '').trim();
      if (!text || text.includes('Your case summary will appear here')) return;
      if (text === lastText) return;
      lastText = text;
      setTimeout(() => {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 200);
    });
    observer.observe(target, { childList: true, subtree: true });
  };
  attach();
}
"""

with gr.Blocks(
    title="Nyaya Mitra — AI Legal Assistant",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=CSS,
    js=SCROLL_JS,
) as app:

    gr.HTML(
        """
        <div id="nm-header">
          <div class="nm-kicker">⚖️ Free legal help</div>
          <h1>Nyaya Mitra</h1>
          <p>Understand any legal document in minutes.</p>
          <p class="nm-sub">
            Upload a PDF or photos of the scanned pages. We will explain the case,
            list every important date, and read the summary aloud in your language.
          </p>
        </div>
        """
    )

    document_state = gr.State(value=None)

    with gr.Tabs():
        with gr.Tab("Analyse document"):
            docs_input = gr.File(
                label="Legal document — PDF or photos of the pages",
                file_types=SUPPORTED_UPLOAD_EXTENSIONS,
                file_count="multiple",
                type="filepath",
                height=160,
                elem_id="nm-file",
            )

            language_input = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="Hindi",
                label="Document and audio language",
                info="Choose the language printed on the document. The summary and audio will use it too.",
            )

            submit_btn = gr.Button(
                "Analyse document",
                variant="primary",
                size="lg",
                elem_id="nm-analyse-btn",
            )

            analysis_output = gr.HTML(
                value=EMPTY_RESULT_HTML,
                elem_id="nm-results",
            )

            audio_output = gr.Audio(
                label="Listen to the summary",
                interactive=False,
                elem_id="nm-audio",
            )

        with gr.Tab("Ask a question"):
            gr.HTML(
                """
                <div class="nm-chat-intro">
                  Ask about the uploaded document, or ask a general legal question.
                  Answers use the document, the legal database, and cited sections.
                </div>
                """
            )

            chat_language = gr.Dropdown(
                choices=SUPPORTED_LANGUAGES,
                value="Hindi",
                label="Chat language",
                info="Used for answers, speech recognition, and spoken replies.",
            )

            chatbot = gr.Chatbot(
                label="Nyaya Mitra",
                height=360,
                placeholder="Ask a question to get started.",
                elem_id="nm-chat",
            )

            question_input = gr.Textbox(
                label="Your question",
                placeholder="For example: What happens if I miss the hearing?",
                lines=2,
            )

            ask_btn = gr.Button(
                "Ask",
                variant="primary",
                size="lg",
            )

            sources_output = gr.HTML(elem_id="nm-chat-sources")

            gr.Examples(
                examples=SUGGESTED_QUESTIONS,
                inputs=question_input,
            )

            gr.HTML(
                '<div class="nm-chat-intro" style="margin-top:16px;">'
                "🎙️ Ask by voice</div>"
            )

            voice_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or upload your question",
                elem_id="nm-voice-input",
            )

            voice_btn = gr.Button(
                "Transcribe and ask",
                variant="secondary",
                size="lg",
            )

            chat_audio = gr.Audio(
                label="Spoken answer",
                interactive=False,
                elem_id="nm-chat-audio",
            )

            play_btn = gr.Button(
                "Play last answer",
                variant="secondary",
                size="sm",
            )

            gr.HTML(
                '<div class="nm-muted" style="font-size:12px; margin-top:6px;">'
                "On a phone, microphone access needs the HTTPS Gradio link. "
                "You can also upload a voice note.</div>"
            )

            clear_btn = gr.Button("Clear chat", variant="secondary", size="sm")

    gr.HTML(
        """
        <div id="nm-footer">
          Nyaya Mitra gives general legal information, not legal advice.
          For urgent matters, contact a lawyer or your nearest free legal aid centre.
        </div>
        """
    )

    submit_btn.click(
        fn=process_document_safe,
        inputs=[docs_input, language_input],
        outputs=[analysis_output, audio_output, document_state],
    ).then(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, sources_output, question_input, chat_audio],
    )

    ask_btn.click(
        fn=chat_answer,
        inputs=[question_input, chatbot, document_state, chat_language],
        outputs=[chatbot, sources_output, question_input],
    )

    question_input.submit(
        fn=chat_answer,
        inputs=[question_input, chatbot, document_state, chat_language],
        outputs=[chatbot, sources_output, question_input],
    )

    voice_btn.click(
        fn=voice_ask,
        inputs=[voice_input, chatbot, document_state, chat_language],
        outputs=[chatbot, sources_output, question_input, chat_audio],
    )

    play_btn.click(
        fn=play_answer,
        inputs=[chatbot, chat_language],
        outputs=[chat_audio],
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, sources_output, question_input, chat_audio],
    )


def ensure_legal_index() -> None:
    """Build the Chroma index on first run (needed on Hugging Face Spaces)."""
    try:
        from rag import corpus_size

        if corpus_size() > 0:
            return

        logger.info("Legal index missing; building it from legal_corpus.json…")
        from build_chroma_index import main as build_index

        build_index()
    except Exception as e:
        logger.warning("Could not build the legal index at startup: %s", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    ensure_ffmpeg()
    ensure_legal_index()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
    )
