# ⚖️ Nyaya Mitra v2 — AI Legal Assistant for Rural India

> **"Every Indian citizen deserves to understand the legal document in their hand."**

Nyaya Mitra is an AI-powered legal document assistant built specifically for rural Indian citizens who receive court notices, FIRs, rent agreements, or property documents — but have no legal background to understand them. It analyses any legal PDF, extracts critical information in structured form, and reads it aloud in the user's regional language.

---

## 🎯 What Problem It Solves

A farmer in MP receives a court summons. A tenant in Mumbai gets an eviction notice. A villager in Tamil Nadu gets an FIR copy.

**They cannot:**
- Read dense English or Hindi legal language
- Understand what sections like "Section 302 IPC" mean for them
- Know what happens if they ignore the document
- Afford a lawyer immediately

**Nyaya Mitra solves this by:**
- Automatically detecting what type of document it is
- Extracting names, dates, conditions, and consequences in plain language
- Explaining their specific legal rights in that situation
- Reading the entire summary aloud in Hindi, Marathi, Tamil, or Telugu

---

## ✨ How It Stands Out from Generic Chatbots

| Feature | ChatGPT / Gemini | Nyaya Mitra |
|---|---|---|
| Document type detection | Generic summary only | Detects: summons, FIR, rent agreement, property doc, legal notice |
| Structured extraction | Unstructured paragraph | Named parties, critical dates, key conditions — all separated |
| Urgency flagging | None | HIGH / MEDIUM / LOW with specific reason and consequence |
| Legal rights | Generic advice | Rights grounded in actual IPC / CrPC / Consumer Protection Act |
| Audio output | None | MP3 in Hindi, Marathi, Tamil, Telugu |
| Target user | English-literate users | Rural citizens, semi-literate users, anyone without legal background |
| Cost | Paid subscription | 100% free — Groq free tier + gTTS + HuggingFace Spaces |

---

## 🏗️ Architecture
![alt text](image-2.png)
![analyser.py architecture diagram](image-1.png)
```
┌─────────────────────────────────────────────────────────────┐
│                    USER (Browser / Phone)                    │
│                   Uploads PDF + Selects Language             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   GRADIO FRONTEND (app.py)                   │
│              Hosted on HuggingFace Spaces (free)             │
└──────┬────────────────────────────────────────┬─────────────┘
       │                                        │
       ▼                                        ▼
┌──────────────────┐                 ┌──────────────────────┐
│  pdf_extractor   │                 │     classifier.py    │
│  (PyPDF2)        │                 │                      │
│                  │                 │  Keyword pattern     │
│  Extracts raw    │                 │  scoring → detects   │
│  text from PDF   │                 │  document type       │
└──────┬───────────┘                 └──────────┬───────────┘
       │                                        │
       └──────────────────┬─────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     analyser.py                              │
│                                                             │
│   Groq API (free tier) + Llama 3.1 8B Instant               │
│                                                             │
│   Structured JSON prompt → extracts:                        │
│   • Parties involved       • Critical dates                 │
│   • Key conditions         • Urgency level (HIGH/MED/LOW)   │
│   • Your legal rights      • Next steps to take            │
│   • What happens if ignored                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   audio_generator.py                         │
│                                                             │
│   deep-translator (Google Translate, free)                  │
│   Translates summary → Hindi / Marathi / Tamil / Telugu     │
│              ↓                                              │
│   gTTS (Google Text-to-Speech, free)                        │
│   Converts translated text → .mp3                           │
│              ↓                                              │
│   Audio plays back directly in the browser                  │
└─────────────────────────────────────────────────────────────┘
```

---
## 📸 snapshots
![alt text](<Screenshot 2026-03-26 at 1.15.09 AM.png>)

![alt text](<Screenshot 2026-03-26 at 1.15.17 AM.png>)

![alt text](<Screenshot 2026-03-26 at 1.15.27 AM.png>)


## 📁 Project Structure

```
nyaya-mitra/
│
├── app.py                  # Main Gradio UI + pipeline orchestrator
│
├── src/
│   ├── __init__.py
│   ├── pdf_extractor.py    # Extracts and cleans text from any PDF
│   ├── classifier.py       # Detects document type via keyword patterns
│   ├── analyser.py         # Groq + Llama 3 structured legal analysis
│   └── audio_generator.py  # Translation + gTTS audio generation
│
├── data/
│   └── legal_docs/         # Store sample legal PDFs here for testing
│
├── requirements.txt        # All Python dependencies
├── README.md               # This file
└── .env.example            # Environment variable template
```

---


## 🔧 Tech Stack

| Component | Tool | Cost |
|---|---|---|
| PDF text extraction | PyPDF2 | Free |
| Document classification | Regex keyword scoring | Free, no API |
| Legal analysis (LLM) | Groq API + Llama 3.1 8B | Free tier |
| Translation | deep-translator (Google Translate) | Free |
| Text-to-speech | gTTS | Free |
| UI framework | Gradio 4.x | Free |
| Deployment | HuggingFace Spaces | Free |

**Total cost to run: ₹0**

---

## 🚀 Local Setup — Step by Step

### Prerequisites
- Python 3.9 or higher installed
- pip package manager
- A free Groq API key (see Step 4)

---

### Step 1 — Get the project files

Download all files and place them in a folder called `nyaya-mitra`, keeping the `src/` subfolder structure intact.

---

### Step 2 — Create a virtual environment

```bash
# Navigate into the project
cd nyaya-mitra

# Create a virtual environment
python -m venv .venv

# Activate it — Mac/Linux:
source .venv/bin/activate

# Activate it — Windows:
.venv\Scripts\activate
```

You should see `(.venv)` appear in your terminal prompt.

---

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

> First install takes 3–5 minutes due to torch and transformers packages.

---

### Step 4 — Get your free Groq API key

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account (no credit card needed)
3. Go to **API Keys** in the left sidebar
4. Click **Create API Key**
5. Copy the key — it starts with `gsk_`

---

### Step 5 — Set the environment variable

**Mac / Linux:**
```bash
export GROQ_API_KEY=gsk_your_key_here
```

**Windows (Command Prompt):**
```cmd
set GROQ_API_KEY=gsk_your_key_here
```

**Using a .env file (recommended):**
```bash
cp .env.example .env
# Open .env in any text editor and paste your key
```

---

### Step 6 — Run the app

```bash
python app.py
```

Open your browser and go to: **http://localhost:7860**

Upload any legal PDF, select a language, and click **Analyse Document**.

---

## ☁️ Deploy to HuggingFace Spaces (Free)

### Step 1 — Create a new Space

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Name it: `nyaya-mitra`
3. SDK: select **Gradio**
4. Visibility: **Public**
5. Click **Create Space**

---

### Step 2 — Push your code

```bash
# Initialise git in your project folder
git init
git add .
git commit -m "Nyaya Mitra v2 - initial deploy"

# Add HuggingFace remote (replace YOUR_USERNAME)
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/nyaya-mitra

# Push
git push origin main
```

---

### Step 3 — Add your API key as a Secret

1. Open your Space on HuggingFace
2. Click the **Settings** tab
3. Scroll to **Repository Secrets**
4. Click **New Secret**
   - Name: `GROQ_API_KEY`
   - Value: your Groq API key
5. Click **Add Secret**

HuggingFace restarts the Space automatically. It goes live in about 2 minutes. Share the URL with anyone — it works from any phone browser.

---

## 📋 Supported Document Types

| Document Type | Key Fields Extracted |
|---|---|
| Court Notice / Summons | Hearing date, court name, parties, sections charged, response deadline |
| FIR | Complainant, accused, offence, police station, next steps |
| Rent / Lease Agreement | Landlord, tenant, rent amount, deposit, tenure, eviction clauses |
| Property Document | Buyer, seller, property details, registration info |
| Legal Notice | Sender, recipient, demand, deadline, consequence of non-compliance |
| Any other legal PDF | Generic structured analysis with urgency, dates, rights, next steps |

---

## 🌐 Supported Audio Languages

| Language | Best For |
|---|---|
| Hindi | UP, MP, Bihar, Rajasthan, Delhi, pan-India |
| Marathi | Maharashtra |
| Tamil | Tamil Nadu |
| Telugu | Andhra Pradesh, Telangana |

---

## 🧠 ML Components Explained

### Groq + Llama 3.1 8B Instant
The core intelligence. A carefully engineered prompt instructs the model to act as a legal assistant for rural India and return a strict JSON schema containing parties, dates, urgency level, legal rights, and next steps. The prompt enforces JSON-only output so parsing is reliable. Groq's free tier delivers responses in under 2 seconds.

### Document Classifier
A regex-based keyword scorer — no API needed, runs instantly. Each document type (summons, FIR, rent agreement, etc.) has 10–15 domain-specific patterns. The classifier scores the extracted text against all pattern sets and returns the highest-scoring document type.

### Translation + TTS Pipeline
`deep-translator` wraps Google Translate's free web API. Long text is automatically chunked to stay within limits. `gTTS` then converts the translated summary to an MP3 which is served directly in the Gradio audio player.

---

## ⚠️ Known Limitations

- Scanned PDFs (image-only, no selectable text) are not supported yet
- Documents longer than ~15 pages are analysed from the first 6000 characters only
- gTTS requires an internet connection
- Telugu and Tamil TTS quality is good but not perfect for complex legal terms

---

## 🛣️ Roadmap

- [ ] OCR support for scanned documents using Tesseract
- [ ] RAG pipeline — ChromaDB vector store of IPC, CrPC, Consumer Protection Act
- [ ] WhatsApp bot for users without smartphones (Twilio integration)
- [ ] More languages: Bengali, Gujarati, Punjabi, Kannada
- [ ] Urgency-based SMS alert system

---

## 📄 License

MIT — free to use, modify, and deploy.
