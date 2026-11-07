
import logging
import os
import re
import tempfile
from deep_translator import GoogleTranslator
from gtts import gTTS

logger = logging.getLogger("nyaya_mitra")

# A token that translation engines reliably leave untouched. It lets us batch
# many fields into one Google Translate request and split the result back out.
BATCH_SEPARATOR = "\n\n<<<NM_9X7>>>\n\n"
_BATCH_SPLIT_RE = re.compile(r"\s*<{2,4}\s*NM[_\s]?9X7\s*>{2,4}\s*", re.IGNORECASE)
MAX_BATCH_CHARS = 4000


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

    # Remove control characters only. Do NOT ASCII-encode: the translated
    # output is Hindi/Marathi/Tamil/Telugu and must reach gTTS intact.
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
LANGUAGE_CONFIG = {
    "Hindi": {
        "translator_code": "hi",
        "gtts_code": "hi",
        "greeting": "नमस्ते। न्याय मित्र आपके दस्तावेज़ की जानकारी दे रहा है।",
    },
    "Marathi": {
        "translator_code": "mr",
        "gtts_code": "mr",
        "greeting": "नमस्कार। न्याय मित्र तुमच्या कागदपत्राची माहिती सांगत आहे।",
    },
    "Tamil": {
        "translator_code": "ta",
        "gtts_code": "ta",
        "greeting": "வணக்கம். நியாய மித்ரா உங்கள் ஆவணத்தை விளக்குகிறது.",
    },
    "Telugu": {
        "translator_code": "te",
        "gtts_code": "te",
        "greeting": "నమస్కారం. న్యాయ మిత్ర మీ పత్రం గురించి వివరిస్తోంది.",
    },
}


def translate_text(text: str, target_language: str) -> str:
    """Translate text to the target language using Google Translate (free)."""
    config = LANGUAGE_CONFIG.get(target_language)
    if not config:
        return text

    try:
        translator = GoogleTranslator(
            source="auto",
            target=config["translator_code"]
        )
        text = safe_text(text)
        chunks = split_into_chunks(text, max_chars=4500)

        translated_chunks = [translator.translate(chunk) for chunk in chunks]
        return " ".join(translated_chunks)
    except Exception as e:
        return f"Translation error: {str(e)}"


def split_into_chunks(text: str, max_chars: int = 4500) -> list:
    """Split long text into chunks for translation API limits."""
    sentences = text.replace(". ", ".|").split("|")
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence + " "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text]


def translate_batch(texts: list[str], target_language: str) -> list[str]:
    """
    Translate many short strings in as few requests as possible.

    All fields are joined with a marker token, translated in groups under the
    Google character limit, then split back apart. If the marker is lost or a
    request fails, that group falls back to per-field translation, and any
    field that still fails keeps its original English text.
    """
    if not texts:
        return []

    config = LANGUAGE_CONFIG.get(target_language)
    if not config:
        return list(texts)

    cleaned = [safe_text(text) for text in texts]

    try:
        translator = GoogleTranslator(
            source="auto", target=config["translator_code"]
        )
    except Exception as e:
        logger.warning("Could not create translator; keeping English | %s", e)
        return cleaned

    groups: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for text in cleaned:
        extra = len(text) + len(BATCH_SEPARATOR)
        if current and current_chars + extra > MAX_BATCH_CHARS:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += extra
    if current:
        groups.append(current)

    results: list[str] = []

    for group in groups:
        translated = None
        try:
            translated = translator.translate(BATCH_SEPARATOR.join(group))
        except Exception as e:
            logger.warning("Batch translation request failed | %s", e)

        if translated:
            parts = _BATCH_SPLIT_RE.split(translated)
            if len(parts) == len(group):
                results.extend(part.strip() for part in parts)
                continue
            logger.warning(
                "Batch translation changed the field separators; "
                "falling back to per-field translation"
            )

        for text in group:
            try:
                result = translator.translate(text)
                results.append(result if result else text)
            except Exception as e:
                logger.warning("Translation failed for one field | %s", e)
                results.append(text)

    return results


def build_audio_script(analysis: dict, urgency_level: str) -> str:
    """Build a clear spoken script from the analysis for audio generation."""
    parts = []

    summary = analysis.get("audio_summary", analysis.get("simple_summary", ""))
    if summary:
        parts.append(summary)

    dates = analysis.get("critical_dates", [])
    if dates:
        date_parts = []
        for d in dates[:5]:
            date_parts.append(f"{d.get('label', 'Important date')}: {d.get('date', '')}. {d.get('importance', '')}")
        parts.append(" ".join(date_parts))

    ignored = analysis.get("what_happens_if_ignored", "")
    if ignored:
        parts.append(f"Important warning: {ignored}")

    steps = analysis.get("next_steps", [])
    if steps:
        parts.append("What you should do: " + ". ".join(steps[:3]))

    return safe_text(" ".join(parts))


def generate_audio(text: str, language: str, output_dir: str | None = None) -> str:
    """
    Translate text and generate audio file.
    Returns path to the generated MP3 file.
    """
    config = LANGUAGE_CONFIG.get(language)
    if not config:
        raise ValueError(f"Unsupported language: {language}")

    greeting = config["greeting"]
    translated_body = translate_text(text, language)
    if not translated_body or str(translated_body).startswith("Translation error:"):
        raise RuntimeError(
            f"Could not translate the audio summary: {translated_body}"
        )
    translated_body = safe_text(translated_body)
    full_text = safe_text(greeting + " " + translated_body)

    output_dir = output_dir or tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)

    # Unique filename so two phone users don't overwrite each other's audio.
    fd, output_path = tempfile.mkstemp(
        prefix=f"nyaya_mitra_{language.lower()}_",
        suffix=".mp3",
        dir=output_dir,
    )
    os.close(fd)

    tts = gTTS(text=full_text, lang=config["gtts_code"], tld="co.in", slow=False)
    tts.save(output_path)

    return output_path
