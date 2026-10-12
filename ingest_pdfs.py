# ─────────────────────────────────────────────────────────────
# STEP 1: Run this in your terminal first
# pip install llama-index-core llama-index-vector-stores-chroma llama-index-embeddings-huggingface llama-index-readers-file
# ─────────────────────────────────────────────────────────────

import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# ── Config (change these paths if needed) ────────────────────
PDFS_DIR       = "./data/pdfs"      # put all 9 arXiv PDFs here
CHROMA_DIR     = "./chroma_db"      # ChromaDB saves here automatically
COLLECTION     = "rag_papers"       # name of the collection inside ChromaDB
CHUNK_SIZE     = 512                # characters per chunk (512 is a good start)
CHUNK_OVERLAP  = 50                 # overlap keeps context across chunks

# ── Step 1: Load embedding model ─────────────────────────────
# BAAI/bge-small-en-v1.5 is free, fast, and downloads once (~25MB)
print("Loading embedding model (downloads once, ~25MB)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None           # no LLM needed during ingestion
Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = CHUNK_OVERLAP
print("✅ Embedding model ready\n")

# ── Step 2: Load all PDFs ─────────────────────────────────────
# SimpleDirectoryReader reads every PDF in the folder automatically
print(f"Loading PDFs from '{PDFS_DIR}'...")

if not os.path.exists(PDFS_DIR):
    raise FileNotFoundError(f"Folder not found: {PDFS_DIR}. Create it and add your PDFs.")

pdf_files = [f for f in os.listdir(PDFS_DIR) if f.endswith(".pdf")]
if not pdf_files:
    raise FileNotFoundError(f"No PDFs found in {PDFS_DIR}. Add your arXiv PDFs there.")

print(f"   Found {len(pdf_files)} PDFs: {pdf_files}")

documents = SimpleDirectoryReader(
    input_dir=PDFS_DIR,
    required_exts=[".pdf"],
    filename_as_id=True       # uses filename as doc ID, helps with citations later
).load_data()

print(f"✅ Loaded {len(documents)} pages total across all PDFs\n")

# ── Step 3: Split into chunks ──────────────────────────────────
# SentenceSplitter respects sentence boundaries (cleaner than fixed-size splits)
print("Splitting documents into chunks...")
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
print(f"✅ Created {len(nodes)} chunks from {len(documents)} pages\n")

# ── Step 4: Set up ChromaDB ───────────────────────────────────
# PersistentClient saves to disk so you don't re-embed every time
print("Setting up ChromaDB...")
chroma_client    = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION)
vector_store     = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context  = StorageContext.from_defaults(vector_store=vector_store)
print("✅ ChromaDB ready\n")

# ── Step 5: Embed and store ───────────────────────────────────
# This is the slow step — embeds every chunk and stores in ChromaDB
# First run: ~2–5 minutes depending on your machine
# Subsequent runs: skips already-stored chunks
print("Embedding chunks and storing in ChromaDB...")
print("(First run takes 2–5 min. Grab a coffee ☕)\n")

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True
)

print(f"\n✅ Done! {len(nodes)} chunks embedded and stored in ChromaDB")
print(f"   Location: {os.path.abspath(CHROMA_DIR)}\n")

# ── Step 6: Test retrieval ────────────────────────────────────
# Quick sanity check — retrieves top 3 matching chunks for a test query
print("─" * 50)
print("Running a test query to verify everything works...")
print("─" * 50)

TEST_QUERY = "What is retrieval augmented generation?"

retriever   = index.as_retriever(similarity_top_k=3)
test_nodes  = retriever.retrieve(TEST_QUERY)

print(f'\nQuery: "{TEST_QUERY}"\n')

for i, result in enumerate(test_nodes, 1):
    file_name   = result.metadata.get("file_name", "unknown")
    page_label  = result.metadata.get("page_label", "?")
    score       = result.score or 0
    preview     = result.text[:300].replace("\n", " ")

    print(f"Result {i}:")
    print(f"  📄 File   : {file_name}  (page {page_label})")
    print(f"  📊 Score  : {score:.4f}")
    print(f"  📝 Preview: {preview}...")
    print()

print("─" * 50)
print("✅ PDF ingestion complete and verified!")
print("   Next step: run ingest_wikipedia.py to add web sources")
print("─" * 50)