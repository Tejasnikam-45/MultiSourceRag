# ─────────────────────────────────────────────────────────────
# STEP 1: Install Groq integration
# pip install llama-index-llms-groq groq
#
# STEP 2: Get free Groq API key (no credit card needed)
# Go to: https://console.groq.com → Sign up → API Keys → Create
#
# STEP 3: Set key in PowerShell and run
#   $env:GROQ_API_KEY="your-key-here"
#   python query.py
# ─────────────────────────────────────────────────────────────

import os
import sys
from pathlib import Path
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# Auto-load .env if present
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.replace("$env:", "").strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)

# ── Config ────────────────────────────────────────────────────
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "rag_papers"
TOP_K       = 5      # chunks retrieved before sending to LLM

# ── Step 1: Check API key ──────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("❌  GROQ_API_KEY not found in environment or .env file.")
    print("    1. Get a free key at: https://console.groq.com")
    print("    2. Add GROQ_API_KEY=your_key in .env file")
    print("    3. Then run this script again.")
    sys.exit(1)

def get_groq_llm(api_key: str):
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
    return Groq(model="qwen/qwen3.8-27b", api_key=api_key, temperature=0.1, max_tokens=256)

# ── Step 2: Setup embedding model + LLM ───────────────────────
print("Loading models...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = get_groq_llm(GROQ_API_KEY)
print("✅ Embedding model + Groq LLM ready\n")

# ── Step 3: Connect to ChromaDB ────────────────────────────────
print("Loading ChromaDB...")
chroma_client     = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_collection(COLLECTION)
vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
print(f"✅ Connected — {chroma_collection.count()} chunks available\n")

# ── Step 4: Load index + build query engine ────────────────────
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(
    similarity_top_k=TOP_K,
    response_mode="compact",
    verbose=False
)

# ── Helper: ask a question and show answer + citations ─────────
def ask(question: str):
    print("\n" + "═" * 60)
    print(f"❓  {question}")
    print("═" * 60)
    print("🔍  Retrieving relevant chunks...")

    response = query_engine.query(question)

    print(f"\n💬  Answer:\n{str(response)}\n")

    print("📚  Sources used:")
    seen = set()
    for node in response.source_nodes:
        source_type = node.metadata.get("source_type", "pdf")
        file_name   = node.metadata.get("file_name", "unknown")
        score       = node.score or 0

        if file_name in seen:
            continue
        seen.add(file_name)

        badge = {"wikipedia": "🌐", "csv": "📊"}.get(source_type, "📄")
        print(f"   {badge}  {file_name}  (relevance: {score:.3f})")
    print()

# ── Step 5: Run 3 test queries ─────────────────────────────────
print("─" * 60)
print("Running 3 test queries...")
print("─" * 60)

ask("What is retrieval augmented generation and how does it work?")
ask("What are the main challenges with long context in language models?")
ask("How are vector databases used in machine learning applications?")

# ── Step 6: Interactive mode ───────────────────────────────────
print("─" * 60)
print("✅  Entering interactive mode. Type 'quit' to exit.")
print("─" * 60)

while True:
    try:
        user_input = input("\n❓  Your question: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        ask(user_input)
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break