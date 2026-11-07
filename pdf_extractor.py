import logging
import os
import re
import sys
import threading

import PyPDF2

logger = logging.getLogger("nyaya_mitra")

MIN_CHARS_FOR_USABLE_TEXT = 50
DEFAULT_OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "eng+hin")
TESSERACT_CONFIG = "--oem 3 --psm 3"

# Image formats accepted from the Gradio uploader. These cover phone photos
# (JPG), screenshots (PNG), and common scanner output (TIFF/BMP). HEIC is
# intentionally not listed: Pillow needs an extra codec for it, and iOS
# converts photos to JPG when uploading through a browser.
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
SUPPORTED_UPLOAD_EXTENSIONS = [".pdf"] + sorted(SUPPORTED_IMAGE_EXTENSIONS)

# Tesseract language code -> EasyOCR language code. EasyOCR is the fallback
# when Tesseract isn't installed; it supports all four app languages.
_TESSERACT_TO_EASYOCR = {
    "eng": "en",
    "hin": "hi",
    "mar": "mr",
    "tam": "ta",
    "tel": "te",
    "ben": "bn",
    "guj": "gu",
    "kan": "kn",
    "mal": "ml",
    "pan": "pa",
    "ori": "or",
    "urd": "ur",
}

_DIGIT_SETS = [
    "०१२३४५६७८९",  # Devanagari (Hindi/Marathi)
    "௦௧௨௩௪௫௬௭௮௯",  # Tamil
    "౦౧౨౩౪౫౬౭౮౯",  # Telugu
    "০১২৩৪৫৬৭৮৯",  # Bengali
    "૦૧૨૩૪૫૬૭૮૯",  # Gujarati
    "೦೧೨೩೪೫೬೭೮೯",  # Kannada
    "൦൧൨൩൪൫൬൭൮൯",  # Malayalam
    "୦୧୨୩୪୫୬୭୮୯",  # Odia
    "੦੧੨੩੪੫੬੭੮੯",  # Punjabi
]
_DIGIT_TRANSLATION = {
    ord(char): str(index)
    for digits in _DIGIT_SETS
    for index, char in enumerate(digits)
}

_EASYOCR_READERS = {}
_EASYOCR_LOCK = threading.Lock()


class OCRUnavailableError(Exception):
    """Raised when no OCR engine is installed/usable."""


def _import_pillow():
    try:
        from PIL import Image, ImageOps

        return Image, ImageOps
    except ImportError as e:
        raise OCRUnavailableError(
            "Image support requires Pillow. Install it with: pip install pillow"
        ) from e


def _normalise_digits(text: str) -> str:
    """Convert Indian-script numerals to ASCII so dates remain machine-usable."""
    if not text:
        return ""
    return text.translate(_DIGIT_TRANSLATION)


def _configure_tesseract(pytesseract):
    """
    Point pytesseract at a Homebrew/MacPorts Tesseract install when the app is
    started from a GUI (Finder, Dock) and /opt/homebrew/bin is not on PATH.
    TESSERACT_CMD can override this.
    """
    if os.environ.get("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
        return

    for candidate in (
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ):
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


def _prepare_image_for_ocr(image):
    """
    Normalise a photo/scan for OCR: honour EXIF rotation, grayscale,
    upscale small images (phone crops), downscale huge photos for speed, and
    stretch contrast so faint print is easier to read.
    """
    Image, ImageOps = _import_pillow()

    image = ImageOps.exif_transpose(image)
    if image.mode != "L":
        image = image.convert("L")

    width, height = image.size
    longest = max(width, height)
    resample = getattr(Image, "Resampling", Image).LANCZOS

    if longest > 3200:
        scale = 3200 / longest
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            resample,
        )
    elif width < 1400:
        scale = min(3.0, 1400 / max(1, width))
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            resample,
        )

    return ImageOps.autocontrast(image, cutoff=1)


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------


def _ocr_with_tesseract(image, lang: str | None = None) -> str:
    """OCR with Tesseract, if it is installed."""
    try:
        import pytesseract
    except ImportError as e:
        raise OCRUnavailableError(
            "pytesseract is not installed. Install it with: pip install pytesseract"
        ) from e

    _configure_tesseract(pytesseract)
    lang = lang or DEFAULT_OCR_LANGUAGES
    prepared = _prepare_image_for_ocr(image)

    try:
        return pytesseract.image_to_string(
            prepared, lang=lang, config=TESSERACT_CONFIG
        )
    except pytesseract.TesseractNotFoundError as e:
        raise OCRUnavailableError(
            "Tesseract OCR is not installed. Install it with: "
            "brew install tesseract tesseract-lang"
        ) from e
    except pytesseract.TesseractError as e:
        # Usually means the Hindi traineddata isn't installed. English OCR is
        # still far better than failing the whole upload.
        if lang != "eng":
            logger.warning(
                "OCR language '%s' unavailable; retrying with 'eng' | %s", lang, e
            )
            try:
                return pytesseract.image_to_string(
                    prepared, lang="eng", config=TESSERACT_CONFIG
                )
            except Exception as fallback_error:
                raise OCRUnavailableError(
                    "Tesseract OCR could not run. Make sure language data is "
                    "installed: brew install tesseract-lang"
                ) from fallback_error
        raise OCRUnavailableError(
            "Tesseract OCR could not run. Make sure language data is installed: "
            "brew install tesseract-lang"
        ) from e


def _easyocr_languages(lang: str | None) -> tuple[str, ...]:
    raw = lang or DEFAULT_OCR_LANGUAGES
    parts = [
        _TESSERACT_TO_EASYOCR.get(part.strip(), part.strip())
        for part in raw.split("+")
        if part.strip()
    ]
    # English is always useful for case numbers, sections, and court names.
    if "en" not in parts:
        parts.insert(0, "en")
    return tuple(dict.fromkeys(parts))


def _get_easyocr_reader(languages: tuple[str, ...]):
    """Load and cache one EasyOCR reader per language set."""
    if languages in _EASYOCR_READERS:
        return _EASYOCR_READERS[languages]

    with _EASYOCR_LOCK:
        if languages in _EASYOCR_READERS:
            return _EASYOCR_READERS[languages]

        try:
            import easyocr
        except ImportError as e:
            raise OCRUnavailableError(
                "EasyOCR is not installed. Install it with: pip install easyocr"
            ) from e

        logger.info(
            "Loading EasyOCR models for %s (first run downloads them; this can "
            "take a few minutes)",
            ", ".join(languages),
        )
        try:
            reader = easyocr.Reader(list(languages), gpu=False, verbose=False)
        except Exception as e:
            raise OCRUnavailableError(
                f"Could not load EasyOCR models for {', '.join(languages)}: {e}"
            ) from e

        _EASYOCR_READERS[languages] = reader
        return reader


def _ocr_with_easyocr(image, lang: str | None = None) -> str:
    """OCR with EasyOCR (multilingual fallback, requires the easyocr package)."""
    try:
        import numpy as np
    except ImportError as e:
        raise OCRUnavailableError("EasyOCR requires numpy.") from e

    reader = _get_easyocr_reader(_easyocr_languages(lang))
    prepared = _prepare_image_for_ocr(image).convert("RGB")

    try:
        results = reader.readtext(np.array(prepared), detail=1, paragraph=False)
    except Exception as e:
        raise OCRUnavailableError(f"EasyOCR failed: {e}") from e

    # Keep reasonably confident boxes and sort them into reading order
    # (top-to-bottom, then left-to-right).
    kept = []
    for bbox, text, confidence in results:
        if confidence < 0.2 or not text.strip():
            continue
        top = min(point[1] for point in bbox)
        left = min(point[0] for point in bbox)
        kept.append((top, left, text.strip()))

    kept.sort(key=lambda item: (item[0], item[1]))
    return _normalise_digits("\n".join(item[2] for item in kept))


def _vision_text_is_usable(text: str) -> bool:
    """
    macOS Vision is fast and excellent for English, but it has no Hindi,
    Marathi, Tamil, or Telugu support. Accept its result only when it looks
    like a real Latin-script page; otherwise let EasyOCR handle the script.
    """
    stripped = text.strip()
    if len(stripped) < 20:
        return False

    alpha_count = sum(ch.isalpha() for ch in stripped)
    if alpha_count < 20:
        return False

    latin_count = sum(
        ("a" <= ch.lower() <= "z") or ch in "'’-" for ch in stripped
    )
    return latin_count / max(1, alpha_count) >= 0.5


def _ocr_with_vision(image) -> str:
    """OCR with Apple's Vision framework (macOS only, English-heavy pages)."""
    if sys.platform != "darwin":
        raise OCRUnavailableError("macOS Vision OCR is only available on macOS.")

    try:
        from ocrmac import ocrmac
    except ImportError as e:
        raise OCRUnavailableError(
            "ocrmac is not installed. Install it with: pip install ocrmac"
        ) from e

    try:
        result = ocrmac.OCR(
            image,
            language_preference=["en-US"],
            recognition_level="accurate",
        ).recognize()
    except Exception as e:
        raise OCRUnavailableError(f"macOS Vision OCR failed: {e}") from e

    return _normalise_digits(
        "\n".join(item[0] for item in result if item and item[0].strip())
    )


def _ocr_image(image, lang: str | None = None) -> str:
    """
    Try the available OCR engines in order of quality/speed:
      1. Tesseract (best multilingual option, if installed)
      2. macOS Vision (fast, English/Latin scripts only)
      3. EasyOCR (multilingual fallback, downloads models on first use)
    """
    errors = []

    try:
        text = _ocr_with_tesseract(image, lang=lang)
        if text.strip():
            return _normalise_digits(text)
    except OCRUnavailableError as e:
        errors.append(str(e))
        logger.info("Tesseract unavailable, trying next OCR engine | %s", e)
    except Exception as e:
        errors.append(str(e))
        logger.warning("Tesseract failed, trying next OCR engine | %s", e)

    if sys.platform == "darwin":
        try:
            text = _ocr_with_vision(image)
            if _vision_text_is_usable(text):
                return text
            logger.info(
                "macOS Vision result did not look usable (likely a non-Latin "
                "script); trying EasyOCR"
            )
        except OCRUnavailableError as e:
            errors.append(str(e))
            logger.info("macOS Vision unavailable | %s", e)
        except Exception as e:
            errors.append(str(e))
            logger.warning("macOS Vision failed | %s", e)

    try:
        text = _ocr_with_easyocr(image, lang=lang)
        if text.strip():
            return text
    except OCRUnavailableError as e:
        errors.append(str(e))
        logger.warning("EasyOCR unavailable | %s", e)
    except Exception as e:
        errors.append(str(e))
        logger.warning("EasyOCR failed | %s", e)

    detail = " | ".join(dict.fromkeys(errors)) if errors else "No OCR engine found."
    raise OCRUnavailableError(
        "No OCR engine could read this image. Install Tesseract (recommended) "
        "with 'brew install tesseract tesseract-lang poppler', or install "
        f"EasyOCR with 'pip install easyocr'. Details: {detail}"
    )


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------


def _render_pdf_page(pdf_path: str, page_number: int, dpi: int = 300):
    """Render one PDF page to a PIL image, preferring PyMuPDF over Poppler."""
    Image, _ = _import_pillow()

    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            fitz = None

    if fitz is not None:
        try:
            document = fitz.open(pdf_path)
            try:
                page = document.load_page(page_number - 1)
                pixmap = page.get_pixmap(dpi=dpi)
                return Image.frombytes(
                    "RGB", (pixmap.width, pixmap.height), pixmap.samples
                )
            finally:
                document.close()
        except Exception as e:
            logger.warning(
                "PyMuPDF could not render page %d: %s; trying pdf2image",
                page_number,
                e,
            )

    try:
        from pdf2image import convert_from_path
    except ImportError as e:
        raise OCRUnavailableError(
            "PDF OCR requires PyMuPDF or pdf2image. Install PyMuPDF with: "
            "pip install pymupdf"
        ) from e

    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
        )
    except Exception as e:
        raise OCRUnavailableError(
            "Could not rasterize the PDF page. Install PyMuPDF with: "
            "pip install pymupdf (or install poppler for pdf2image)."
        ) from e

    if not images:
        raise ValueError(f"PDF page {page_number} could not be rendered.")

    return images[0]


def _pdf_page_count(pdf_path: str) -> int:
    try:
        import pymupdf as fitz

        document = fitz.open(pdf_path)
        try:
            return document.page_count
        finally:
            document.close()
    except Exception:
        pass

    try:
        from pdf2image import pdfinfo_from_path

        info = pdfinfo_from_path(pdf_path)
        return int(info.get("Pages", 0))
    except Exception as e:
        raise OCRUnavailableError(
            "Could not determine the number of PDF pages. Install PyMuPDF with: "
            "pip install pymupdf"
        ) from e


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text page by page. Pages with a usable text layer are read
    normally; only pages that look empty/scan-like are sent to OCR. This
    handles mixed PDFs (some typed pages, some scanned pages) and avoids
    OCR'ing pages that already have text.
    """
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            page_texts = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_texts.append(page.extract_text() or "")
                except Exception as e:
                    # One corrupt page shouldn't discard the rest of the file;
                    # leave it empty so OCR can try it.
                    logger.warning(
                        "Could not read text layer of page %d: %s", page_number, e
                    )
                    page_texts.append("")
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}") from e

    if not page_texts:
        raise ValueError("The PDF has no pages.")

    page_blocks = []
    ocr_failures = 0

    for page_number, page_text in enumerate(page_texts, start=1):
        text = page_text

        if is_extraction_poor(text, min_chars=20):
            try:
                ocr_text = _ocr_pdf_page(pdf_path, page_number)
            except OCRUnavailableError as e:
                ocr_failures += 1
                logger.warning("OCR unavailable for page %d: %s", page_number, e)
                ocr_text = ""
            except Exception as e:
                ocr_failures += 1
                logger.warning("OCR failed for page %d: %s", page_number, e)
                ocr_text = ""

            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text

        if text.strip():
            page_blocks.append(f"--- Page {page_number} ---\n{text.strip()}")

    combined = clean_text("\n\n".join(page_blocks))
    if not combined:
        if ocr_failures:
            raise OCRUnavailableError(
                "The PDF appears to be scanned, but no OCR engine could read it. "
                "Install Tesseract (recommended) or EasyOCR, then try again."
            )
        raise ValueError(
            "No readable text found in PDF, even after OCR. "
            "It may be a very low quality scan or a corrupted file."
        )

    return combined


def extract_with_pypdf2(pdf_path: str) -> str:
    """Extract raw text using PyPDF2's text-layer extraction."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}") from e

    return text


def is_extraction_poor(
    text: str, min_chars: int = MIN_CHARS_FOR_USABLE_TEXT
) -> bool:
    """
    Heuristic: near-empty or mostly non-alphabetic content usually means
    the page has no real text layer (i.e. it's a scanned image).
    """
    cleaned = text.strip()
    if len(cleaned) < min_chars:
        return True

    alpha_count = sum(ch.isalpha() for ch in cleaned)
    ratio = alpha_count / max(1, len(cleaned))
    return ratio < 0.2


def _ocr_pdf_page(pdf_path: str, page_number: int, lang: str | None = None) -> str:
    """Rasterize a single PDF page and OCR it."""
    image = _render_pdf_page(pdf_path, page_number)
    return _ocr_image(image, lang=lang)


def extract_with_ocr(pdf_path: str, lang: str | None = None) -> str:
    """Rasterize every PDF page and OCR it. Useful for forcing OCR."""
    page_count = _pdf_page_count(pdf_path)
    if page_count <= 0:
        raise ValueError("The PDF has no pages.")

    page_blocks = []
    for page_number in range(1, page_count + 1):
        try:
            page_text = _ocr_pdf_page(pdf_path, page_number, lang=lang)
        except OCRUnavailableError:
            raise
        except Exception as e:
            logger.warning("OCR failed on page %d: %s", page_number, e)
            continue

        if page_text.strip():
            page_blocks.append(f"--- Page {page_number} ---\n{page_text.strip()}")

    return clean_text("\n\n".join(page_blocks))


def extract_text_from_image(image_path: str, lang: str | None = None) -> str:
    """OCR a single uploaded photo/screenshot/scan."""
    Image, _ = _import_pillow()

    try:
        with Image.open(image_path) as image:
            text = _ocr_image(image, lang=lang)
    except OCRUnavailableError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read image: {e}") from e

    text = clean_text(text)
    if not text:
        raise ValueError(
            "No readable text was found in this image. Please upload a clearer, "
            "well-lit photo with the page filling most of the frame."
        )
    return text


def extract_text_from_upload(path: str, lang: str | None = None) -> str:
    """Dispatch one uploaded file to the PDF or image extraction path."""
    extension = os.path.splitext(path)[1].lower()

    if extension == ".pdf":
        return extract_text_from_pdf(path)
    if extension in SUPPORTED_IMAGE_EXTENSIONS:
        return extract_text_from_image(path, lang=lang)

    raise ValueError(
        f"Unsupported file type '{extension or 'unknown'}'. "
        "Upload a PDF or a photo (PNG, JPG, WEBP, TIFF, or BMP)."
    )


def extract_text_from_uploads(paths, lang: str | None = None) -> str:
    """
    Extract text from one or more uploaded PDFs/images, preserving the order
    the user uploaded them in. Each upload is labelled so the LLM can tell
    separate pages/documents apart.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    paths = [str(path) for path in (paths or []) if path]
    if not paths:
        raise ValueError("Please upload at least one PDF or photo.")

    parts = []
    ocr_error = None
    for index, path in enumerate(paths, start=1):
        logger.info("Extracting upload %d/%d: %s", index, len(paths), os.path.basename(path))
        try:
            text = extract_text_from_upload(path, lang=lang)
        except OCRUnavailableError as e:
            # Keep going: a text-based PDF in the same upload can still work
            # even when one image cannot be OCR'd.
            logger.warning("OCR unavailable for upload %d: %s", index, e)
            ocr_error = e
            continue
        except Exception as e:
            logger.warning("Could not read upload %d: %s", index, e)
            continue

        if text.strip():
            parts.append(f"--- Part {index} ---\n{text.strip()}")

    combined = clean_text("\n\n".join(parts))
    if not combined:
        if ocr_error is not None:
            raise ocr_error
        raise ValueError(
            "No readable text was found in the uploaded file(s). Please upload "
            "clearer pages or a text-based PDF."
        )

    return combined


def clean_text(text: str) -> str:
    # Fix smart quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")

    # Remove weird spacing and zero-width characters from OCR output.
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Remove NUL/control characters but keep all Unicode scripts (Hindi,
    # Marathi, Tamil, Telugu, ...) so OCR'd regional documents survive.
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)

    return text.strip()
