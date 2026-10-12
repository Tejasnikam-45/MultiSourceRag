import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from llama_index.core import Document, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import requests

# Auto-load .env if present
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# ── Config ────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_papers"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

# ── Setup models once at import time ─────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def get_groq_llm(api_key: str) -> Optional[Groq]:
    if not api_key:
        return None

    candidate_models = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "allam-2-7b",
    ]

    try:
        from groq import Groq as GroqClient

        client = GroqClient(api_key=api_key)
        available = {m.id for m in client.models.list().data}
        for candidate in candidate_models:
            if candidate in available:
                print(f"[INFO] Connected to Groq using model: {candidate}")
                return Groq(model=candidate, api_key=api_key, temperature=0.1, max_tokens=256)
    except Exception as e:
        print(f"[WARNING] Groq model detection: {e}")

    # Fallback
    return Groq(model="qwen/qwen3.8-27b", api_key=api_key, temperature=0.1, max_tokens=256)


try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm_instance = get_groq_llm(GROQ_API_KEY)
if llm_instance is not None:
    Settings.llm = llm_instance


# ── Load index from existing ChromaDB ────────────────────────
def load_index() -> Tuple[VectorStoreIndex, Collection]:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index, chroma_collection


# Load once when the module is imported
_index, _collection = load_index()
_query_engine = _index.as_query_engine(
    similarity_top_k=TOP_K,
    response_mode="compact"
)


# ── Query function ────────────────────────────────────────────
def query(question: str) -> Dict[str, Any]:
    """Run a question through RAG and return answer + sources."""
    response = _query_engine.query(question)

    # Build deduplicated source list
    seen = set()
    sources = []
    source_nodes = getattr(response, "source_nodes", []) or []
    for node in source_nodes:
        node_meta = getattr(node, "metadata", {}) or {}
        file_name = node_meta.get("file_name", "unknown")
        source_type = node_meta.get("source_type", "pdf")
        score_val = getattr(node, "score", 0) or 0
        score = round(float(score_val), 4)
        url = node_meta.get("url", "")

        if file_name in seen:
            continue
        seen.add(file_name)

        sources.append({
            "file_name": file_name,
            "source_type": source_type,
            "relevance": score,
            "url": url,
        })

    return {
        "answer": str(response),
        "sources": sources,
        "query": question,
    }


# ── Ingest a PDF file at runtime ─────────────────────────────
def ingest_pdf(file_path: str, original_filename: str) -> Dict[str, Any]:
    """Add a new PDF to the existing ChromaDB collection."""
    tmp_dir = tempfile.mkdtemp()
    dest = os.path.join(tmp_dir, original_filename)
    shutil.copy(file_path, dest)

    try:
        documents = SimpleDirectoryReader(
            input_dir=tmp_dir,
            required_exts=[".pdf"],
            filename_as_id=True
        ).load_data()

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(documents)

        # Tag each node with metadata
        for node in nodes:
            node.metadata["source_type"] = "pdf"
            node.metadata["file_name"] = original_filename

        _index.insert_nodes(nodes)

        return {
            "status": "success",
            "filename": original_filename,
            "chunks": len(nodes),
            "total_chunks": _collection.count(),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Ingest a URL at runtime ───────────────────────────────────
def ingest_url(url: str) -> Dict[str, Any]:
    """Scrape a URL and add it to ChromaDB."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    # Remove nav/footer noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    doc = Document(
        text=text,
        metadata={
            "source_type": "website",
            "file_name": f"Web: {title[:60]}",
            "url": url,
            "title": title,
        },
        id_=f"url_{abs(hash(url))}"
    )

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents([doc])
    _index.insert_nodes(nodes)

    return {
        "status": "success",
        "url": url,
        "title": title,
        "chunks": len(nodes),
        "total_chunks": _collection.count(),
    }


# ── Get all ingested sources ──────────────────────────────────
def get_sources() -> List[Dict[str, Any]]:
    """Return a unique list of all sources in ChromaDB."""
    results = _collection.get(include=["metadatas"])  # type: ignore
    seen = set()
    sources = []

    metadatas = results.get("metadatas") or []
    for meta in metadatas:
        if not meta:
            continue
        file_name = meta.get("file_name", "unknown")
        source_type = meta.get("source_type", "pdf")
        url = meta.get("url", "")

        if file_name in seen:
            continue
        seen.add(file_name)
        sources.append({
            "file_name": file_name,
            "source_type": source_type,
            "url": url,
        })

    return sorted(sources, key=lambda x: str(x.get("source_type", "")))


# ── Collection stats ──────────────────────────────────────────
def get_stats() -> Dict[str, Any]:
    total = _collection.count()
    results = _collection.get(include=["metadatas"])  # type: ignore
    metas = results.get("metadatas") or []
    by_type: Dict[str, int] = {}
    for m in metas:
        if not m:
            continue
        t = str(m.get("source_type", "pdf"))
        by_type[t] = by_type.get(t, 0) + 1

    return {"total_chunks": total, "by_source_type": by_type}