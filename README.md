# Multi-Source RAG

An end-to-end Multi-Source Retrieval-Augmented Generation (RAG) system with a FastAPI backend, ChromaDB vector store, and a React + Vite frontend. Ask questions and retrieve context across research PDFs, structured CSV data, and web sources.

---

## 🚀 Features

- **Multi-Source Ingestion**: Ingests arXiv research papers (PDFs), structured question-answer pairs (CSV), and Wikipedia/web content.
- **Vector Search**: ChromaDB vector store using HuggingFace embeddings (`BAAI/bge-small-en-v1.5`).
- **LLM Reasoning**: Fast inference powered by Groq (e.g. LLaMA models) and Google Gemini.
- **RESTful API**: FastAPI backend with health monitoring, ingestion endpoints, and Swagger UI docs.
- **Interactive Web UI**: Modern React frontend with chat interface, knowledge base browser, and real-time source citations.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, LlamaIndex, ChromaDB, Groq SDK, python-dotenv
- **Frontend**: React, Vite, TailwindCSS / CSS modules
- **Embeddings**: HuggingFace (`BAAI/bge-small-en-v1.5`)
- **LLMs**: Groq / Gemini

---

## 📁 Project Structure

```text
├── data/                    # Sample data (PDFs, CSVs)
│   ├── csv/
│   └── pdfs/
├── frontend/                # React + Vite application
│   ├── src/
│   ├── index.html
│   └── package.json
├── chroma_db/               # (Local) ChromaDB persistence (gitignored)
├── evaluate.py              # RAG evaluation script
├── ingest_csv.py            # Ingestion script for CSV datasets
├── ingest_pdfs.py           # Ingestion script for PDF documents
├── ingest_wikipedia.py      # Ingestion script for Wikipedia / web articles
├── main.py                  # FastAPI server entry point
├── query.py                 # Interactive CLI query script
├── rag_engine.py            # Core RAG retrieval and generation engine
├── requirements.txt         # Python dependencies
├── run.py                   # Single-command orchestrator (starts backend + frontend)
├── start.bat                # Windows quick launcher
└── .env.example             # Template for API keys
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Multisource Rag"
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Backend Setup
Create and activate a virtual environment, then install dependencies:
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Ingest Knowledge Sources
Build the Chroma vector index from the documents:
```bash
# Ingest PDF papers
python ingest_pdfs.py

# Ingest CSV datasets
python ingest_csv.py

# Ingest Wikipedia / web articles
python ingest_wikipedia.py
```

### 5. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Running the Application

### Option 1: Quick Launcher (All-in-One)
Run both backend (port 8000) and frontend (port 5173) simultaneously:
```bash
python run.py
```
*(On Windows, you can also double-click `start.bat`)*

### Option 2: Manual Start
- **Backend**:
  ```bash
  uvicorn main:app --reload --port 8000
  ```
  API documentation is available at `http://localhost:8000/docs`.

- **Frontend**:
  ```bash
  cd frontend
  npm run dev
  ```
  Open `http://localhost:5173` in your browser.
