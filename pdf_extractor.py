
import PyPDF2
import re


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Could not read PDF: {str(e)}")

    if not text.strip():
        raise ValueError("No readable text found in PDF. It may be a scanned image.")

    return clean_text(text)


def clean_text(text: str) -> str:
    # Fix smart quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    # Remove weird spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove problematic characters safely
    text = text.encode("utf-8", "ignore").decode("utf-8")

    return text.strip()
