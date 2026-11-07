from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

from audio_generator import LANGUAGE_CONFIG, safe_text
from gtts import gTTS

logger = logging.getLogger("nyaya_mitra")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

LANGUAGE_CODES = {
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "English": "en",
}

_model = None


def _get_model():
    """Load and cache the Whisper model. The first call downloads it."""
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "Voice input requires faster-whisper. Install it with: "
                "pip install faster-whisper"
            ) from e

        logger.info(
            "Loading Whisper model '%s' (first run downloads it; this can take "
            "a few minutes)",
            WHISPER_MODEL,
        )
        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe_audio(audio_path: str, language: str = "Hindi") -> str:
    """
    Transcribe a recorded/uploaded audio question.

    Uses Whisper with the language hint from the UI, which is much more
    accurate for short Indian-language questions than auto-detection.
    """
    if not audio_path:
        raise ValueError("No audio recording was provided.")

    model = _get_model()
    language_code = LANGUAGE_CODES.get(language)

    try:
        segments, _info = model.transcribe(
            audio_path,
            language=language_code,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as e:
        raise RuntimeError(f"Could not transcribe the audio: {e}") from e

    if not text:
        raise ValueError(
            "No speech was detected. Please record again in a quieter place."
        )

    return text


def speak_text(
    text: str,
    language: str = "Hindi",
    output_dir: Optional[str] = None,
) -> str:
    """
    Convert a chat answer directly to speech.

    Unlike audio_generator.generate_audio(), this does NOT translate first:
    the chat answer is already in the language selected by the user.
    """
    config = LANGUAGE_CONFIG.get(language)
    if not config:
        raise ValueError(f"Unsupported language: {language}")

    spoken = safe_text(text).strip()
    if not spoken:
        raise ValueError("There is no answer text to speak.")

    output_dir = output_dir or tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)

    fd, output_path = tempfile.mkstemp(
        prefix=f"nyaya_voice_{language.lower()}_",
        suffix=".mp3",
        dir=output_dir,
    )
    os.close(fd)

    try:
        tts = gTTS(
            text=spoken,
            lang=config["gtts_code"],
            tld="co.in",
            slow=False,
        )
        tts.save(output_path)
    except Exception as e:
        raise RuntimeError(f"Could not generate speech: {e}") from e

    return output_path
