
import os
import tempfile
from deep_translator import GoogleTranslator
from gtts import gTTS
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

    # Remove any remaining problematic characters
    return text.encode("ascii", "ignore").decode()
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


def build_audio_script(analysis: dict, urgency_level: str) -> str:
    """Build a clear spoken script from the analysis for audio generation."""
    parts = []

    summary = analysis.get("audio_summary", analysis.get("simple_summary", ""))
    if summary:
        parts.append(summary)

    dates = analysis.get("critical_dates", [])
    if dates:
        date_parts = []
        for d in dates[:2]:
            date_parts.append(f"{d.get('label', 'Important date')}: {d.get('date', '')}. {d.get('importance', '')}")
        parts.append(" ".join(date_parts))

    ignored = analysis.get("what_happens_if_ignored", "")
    if ignored:
        parts.append(f"Important warning: {ignored}")

    steps = analysis.get("next_steps", [])
    if steps:
        parts.append("What you should do: " + ". ".join(steps[:3]))

    return safe_text(" ".join(parts))


def generate_audio(text: str, language: str, output_dir: str = "/tmp") -> str:
    """
    Translate text and generate audio file.
    Returns path to the generated MP3 file.
    """
    config = LANGUAGE_CONFIG.get(language)
    if not config:
        raise ValueError(f"Unsupported language: {language}")

    greeting = config["greeting"]
    translated_body = translate_text(text, language)
    translated_body = safe_text(translated_body)
    full_text = safe_text(greeting + " " + translated_body)

    output_path = os.path.join(output_dir, f"nyaya_mitra_{language.lower()}.mp3")
    print("FINAL TEXT:", repr(full_text))
    tts = gTTS(text=full_text, lang=config["gtts_code"], tld="co.in", slow=False)
    tts.save(output_path)

    return output_path
