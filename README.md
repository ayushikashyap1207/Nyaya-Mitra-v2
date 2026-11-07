# ⚖️ Nyaya Mitra v2 — AI Legal Assistant for Rural India

> **"Every Indian citizen deserves to understand the legal document in their hand."**

Nyaya Mitra is a free AI legal-document assistant built for rural Indian citizens. Upload a court summons, FIR, rent agreement, property document, or legal notice — as a **PDF or phone photos of the scanned pages** — and get:

- a plain-language **case summary**,
- **every important date and deadline** found in the document,
- urgency, legal rights, consequences, and next steps,
- an **audio summary** in Hindi, Marathi, Tamil, or Telugu.

The interface is mobile-first: it works comfortably on a phone browser.

---

## ✨ What is new in this version

| Feature | Details |
|---|---|
| 📷 Photo / scanned-page upload | Upload JPG, PNG, WEBP, TIFF, or BMP photos of pages. The app automatically uses Tesseract if installed, then macOS Vision for English, then EasyOCR for Hindi/Marathi/Telugu. Multiple files can be uploaded in page order. |
| 📄 Mixed PDF support | PDFs with some typed pages and some scanned pages are handled page by page: text-layer pages are read normally, scan-like pages are OCR'd. PyMuPDF renders pages without needing Poppler. |
| 📅 Complete date list | The LLM is instructed to extract every hearing date, notice date, filing date, payment date, and relative deadline. A deterministic date scanner also lists every date-like string found in the raw text. |
| 🗂️ Case summary first | The summary is the first thing shown after the urgency banner, followed by a dedicated important-dates card. |
| 📱 Mobile-friendly UI | Single-column layout, large tap targets, 16px inputs (no iOS zoom), stacked date cards, and a max-width that fits small screens. |
| 🎧 Audio | gTTS reads the translated summary aloud. Regional-script text is preserved correctly. |

---

## 🏗️ How it works

```
Upload PDFs / page photos
          │
          ▼
pdf_extractor.py
  PyPDF2 text layer per page
  + OCR fallback: Tesseract → macOS Vision → EasyOCR
          │
          ▼
classifier.py  →  document type (summons, FIR, rent, property, notice)
          │
          ▼
rag.py  →  relevant IPC/CrPC/Consumer Protection context from ChromaDB
          │
          ▼
analyser.py  →  local Llama 3.1 via Ollama
  Pydantic-validated JSON: summary, all critical dates, parties,
  conditions, urgency, rights, next steps, audio script
          │
          ▼
app.py  →  translate → mobile-friendly HTML + gTTS audio
```

---

## 📁 Project structure

```
Nyaya-Mitra-v2/
├── app.py                  # Gradio UI, mobile CSS, pipeline orchestration
├── pdf_extractor.py        # PDF text extraction + image OCR + multi-file input
├── date_extractor.py       # Deterministic date/deadline scanner
├── classifier.py           # Keyword-based document type detection
├── analyser.py             # Ollama LLM call + Pydantic validation + retries
├── rag.py                  # ChromaDB bi-encoder retrieval + cross-encoder rerank
├── build_chroma_index.py   # One-time legal corpus index builder
├── legal_corpus.json       # IPC/CrPC/Consumer Protection entries
├── chroma_db/              # Persisted vector index
├── audio_generator.py      # Translation, script building, gTTS MP3
├── requirements.txt
└── README.md
```

---

## 🔧 Tech stack

| Component | Tool |
|---|---|
| UI | Gradio 4.44 |
| PDF text | PyPDF2 |
| OCR | Tesseract → macOS Vision → EasyOCR fallback chain |
| PDF rendering | PyMuPDF (no Poppler required) |
| Document classification | Regex keyword scoring |
| Legal analysis | Local Llama 3.1 8B via Ollama |
| Legal context | ChromaDB + sentence-transformers + cross-encoder reranker |
| Translation | deep-translator (Google Translate, free) |
| Text-to-speech | gTTS |
| Validation | Pydantic 2 |

Total cost to run: **₹0** (local model + free services).

---

## 🚀 Setup

### 1. Install system OCR dependencies (optional but recommended)

The app works without these because EasyOCR and macOS Vision are included
through pip, but Tesseract gives the best multilingual results, especially
for Tamil.

macOS:
```bash
brew install tesseract tesseract-lang poppler
```

Ubuntu / Debian:
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-hin poppler-utils
```

### 2. Create a virtual environment and install Python packages

```bash
cd Nyaya-Mitra-v2
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The first OCR request may take a few minutes while EasyOCR downloads its
> models. They are cached in `~/.EasyOCR` and reused afterwards.

### 3. Install and start Ollama

```bash
# Install Ollama from https://ollama.com, then:
ollama serve
ollama pull llama3.1:8b
```

The app uses `llama3.1:8b` by default. Override with environment variables:

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.1:8b
```

### 4. Build the legal vector index (first run only)

```bash
python build_chroma_index.py
```

If `chroma_db/` is missing, the app still works but skips legal-context retrieval.

### 5. Run

```bash
python app.py
```

Open **http://localhost:7860**.

---

## 📱 Opening it on a phone

- **Same Wi-Fi:** run the app, then open `http://<your-computer-LAN-IP>:7860` on the phone. The server listens on `0.0.0.0` so LAN access works.
- **Anywhere:** by default `share=True` creates a temporary public Gradio link. Use it on mobile data. Disable with `GRADIO_SHARE=0 python app.py`.
- **Custom port:** `PORT=8080 python app.py`.

On a phone, tap the upload box and choose **Camera** to photograph each page, or select existing photos from the gallery. Upload pages in order.

---

## 📋 Supported inputs and document types

**Inputs:** PDF, PNG, JPG/JPEG, WEBP, TIFF, BMP — one file or many pages at once.

**Document types detected:** Court notice/summons, FIR, rent/lease agreement, property document, legal notice, and generic legal documents.

**Audio languages:** Hindi, Marathi, Tamil, Telugu.

---

## ⚙️ Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model name |
| `OCR_LANGUAGES` | `eng+hin` | Default OCR language string used outside the UI |
| `TESSERACT_CMD` | auto-detected | Path to the tesseract binary |
| `PORT` | `7860` | Gradio port |
| `GRADIO_SHARE` | `1` | Set to `0` to disable the public share link |

---

## ⚠️ Known limitations

- OCR quality depends on photo clarity; well-lit, straight, full-page photos work best.
- The first OCR request can be slow while EasyOCR downloads models.
- Tamil OCR currently needs Tesseract: EasyOCR's bundled Tamil model has a checkpoint compatibility issue on some installs.
- The analysis prompt currently uses the first ~12,000 characters of the extracted text.
- Local Llama 3.1 8B inference can be slow on CPU, especially the first request.
- gTTS and Google Translate require an internet connection.
- Tamil/Telugu TTS quality is good but not perfect for complex legal terms.
- This is general legal information, **not legal advice**.

---

## 📄 License

MIT — free to use, modify, and deploy.
