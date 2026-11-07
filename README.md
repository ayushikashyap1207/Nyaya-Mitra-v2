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
| 🔎 Cited legal sources | RAG results now return the source and section (for example `Indian Penal Code §302`). The analysis page and the Q&A tab show the exact legal sources used. |
| 💬 Ask a question | A new chat tab answers follow-up questions using the uploaded document + cited legal database. Works with a document, or as a general legal question assistant. |
| 🎙️ Voice input and spoken answers | Record or upload a question; Whisper transcribes it, the cited RAG answers it, and gTTS reads the answer aloud. A “Play last answer” button also works for typed questions. |
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
          │
          ▼
chat.py  →  follow-up questions with cited RAG sources
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
├── chat.py                 # Cited RAG Q&A over the document
├── voice.py                # Whisper speech-to-text + gTTS spoken answers
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
| Speech-to-text | faster-whisper (Whisper model, runs locally) |
| Audio preprocessing | static-ffmpeg (bundled ffmpeg/ffprobe) |
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
>
> The first voice request also downloads the Whisper model
> (`small` by default, about 500 MB). Use `WHISPER_MODEL=tiny` or `base` for
> a faster first run.

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

## 💬 Ask a question

After analysing a document, open the **Ask a question** tab. Example questions:

- “What happens if I miss the hearing?”
- “What should I do next?”
- “Can I get free legal aid?”
- “Explain the most important date.”

You can also use the tab without uploading a document for general legal questions. Answers use the legal database and show the cited source sections underneath.

### Voice

In the same tab:

1. Choose the **Chat language**.
2. Tap the microphone and record your question, or upload a voice note.
3. Tap **Transcribe and ask**.
4. The question is transcribed, answered with cited RAG, and the answer is spoken aloud.
5. For typed questions, use **Play last answer** to hear the latest response.

On a phone, browser microphone access requires a secure context — use the HTTPS Gradio share link. The same-Wi-Fi `http://<LAN-IP>:7860` address may block the microphone, so upload a voice note there instead.

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
| `OLLAMA_MODEL` | `llama3.1:8b` | Model name (`llama3.2:3b` for a lighter test model) |
| `OCR_LANGUAGES` | `eng+hin` | Default OCR language string used outside the UI |
| `TESSERACT_CMD` | auto-detected | Path to the tesseract binary |
| `WHISPER_MODEL` | `small` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WHISPER_DEVICE` | `cpu` | faster-whisper device |
| `WHISPER_COMPUTE_TYPE` | `int8` | faster-whisper compute type |
| `PORT` | `7860` | Gradio port |
| `GRADIO_SHARE` | `1` | Set to `0` to disable the public share link |

---

## ⚠️ Known limitations

- OCR quality depends on photo clarity; well-lit, straight, full-page photos work best.
- The first OCR request can be slow while EasyOCR downloads models.
- Tamil OCR currently needs Tesseract: EasyOCR's bundled Tamil model has a checkpoint compatibility issue on some installs.
- The analysis prompt currently uses the first ~12,000 characters of the extracted text.
- Chat answers use the same local model and the current legal corpus; a larger corpus will improve them.
- Voice input needs a microphone; on phones this requires the HTTPS Gradio share link, or upload a voice note.
- Whisper accuracy drops with background noise, strong accents, or very quiet recordings.
- The first voice request downloads the Whisper model (`small` is about 500 MB).
- Local Llama 3.1 8B inference can be slow on CPU, especially the first request.
- gTTS and Google Translate require an internet connection.
- Tamil/Telugu TTS quality is good but not perfect for complex legal terms.
- This is general legal information, **not legal advice**.

---

## 📄 License

MIT — free to use, modify, and deploy.
