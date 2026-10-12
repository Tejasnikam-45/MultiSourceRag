# main.py — FastAPI backend for Multi-Source RAG
#
# Install:  pip install fastapi uvicorn python-multipart
# Run:      uvicorn main:app --reload --port 8000
# Docs:     http://localhost:8000/docs  (auto-generated Swagger UI)

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_engine   # our reusable RAG module

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Source RAG API",
    description="Q&A over PDFs, websites, and CSV using LlamaIndex + Groq",
    version="1.0.0"
)

# Allow React frontend (localhost and production deployments)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class URLRequest(BaseModel):
    url: str

# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — visit this to confirm the API is running."""
    return {"status": "ok", "message": "Multi-Source RAG API is running"}


@app.get("/health")
def health():
    """Returns collection stats — useful for the React dashboard."""
    stats = rag_engine.get_stats()
    return {"status": "ok", **stats}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Ask a question — retrieves relevant chunks and generates an answer.

    Returns:
        answer: LLM-generated response
        sources: list of documents used (with file name, type, relevance score)
        query: the original question
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag_engine.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and ingest it into the knowledge base.
    Accepts: .pdf files only
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save upload to a temp file, then ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = rag_engine.ingest_pdf(tmp_path, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)   # clean up temp file


@app.post("/upload/url")
def upload_url(request: URLRequest):
    """
    Scrape a URL and ingest its content into the knowledge base.
    Accepts: any public URL
    """
    if not request.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        result = rag_engine.ingest_url(request.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
def get_sources():
    """
    Returns all unique sources currently in the knowledge base.
    Used by React to show the user what documents are available.
    """
    sources = rag_engine.get_sources()
    return {"sources": sources, "count": len(sources)}