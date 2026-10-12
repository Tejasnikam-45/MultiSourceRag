# ─────────────────────────────────────────────────────────────
# STEP 1: Run this in your terminal first
# pip install datasets
# ─────────────────────────────────────────────────────────────

import os
import pandas as pd
from datasets import load_dataset
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# ── Config — MUST match previous ingest scripts ───────────────
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "rag_papers"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
CSV_SAVE_PATH = "./data/csv/squad.csv"     # saved locally for reference
MAX_ROWS      = 200                        # keep it small for demo purposes

# ── Step 1: Load embedding model ─────────────────────────────
print("Loading embedding model...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
print("✅ Embedding model ready\n")

# ── Step 2: Download dataset from HuggingFace ─────────────────
# rajpurkar/squad = Stanford Q&A dataset, standard Parquet format
# Has: title, context passage, question, answer — perfect for RAG demo
print("Downloading dataset from HuggingFace (SQuAD)...")
print("(Downloads ~30MB once, cached after that)\n")

dataset = load_dataset("rajpurkar/squad", split="train")
print(f"✅ Dataset loaded — {len(dataset)} rows total")
print(f"   Using first {MAX_ROWS} rows for this demo\n")

# ── Step 3: Convert to a clean CSV ────────────────────────────
print("Converting to CSV format...")
rows = []

for item in dataset.select(range(MAX_ROWS)):
    title    = item.get("title", "").strip().replace("_", " ")
    context  = item.get("context", "").strip()
    question = item.get("question", "").strip()

    # answers is a dict with "text" list — grab first answer
    answers = item.get("answers", {})
    answer  = answers["text"][0] if answers.get("text") else ""

    if not title or not context:
        continue

    rows.append({
        "title":    title,
        "context":  context,
        "question": question,
        "answer":   answer,
    })

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(CSV_SAVE_PATH), exist_ok=True)
df.to_csv(CSV_SAVE_PATH, index=False)

print(f"✅ CSV saved to: {CSV_SAVE_PATH}")
print(f"   Rows: {len(df)}")
print(f"\nSample row:")
print(f"  Title   : {df['title'].iloc[0][:80]}...")
print(f"  Context : {df['context'].iloc[0][:120]}...")
print(f"  Question: {df['question'].iloc[0][:80]}")
print(f"  Answer  : {df['answer'].iloc[0][:80]}\n")

# ── Step 4: Convert CSV rows → LlamaIndex Documents ───────────
print("Converting CSV rows to documents...")
documents = []

for _, row in df.iterrows():
    text = (
        f"Title: {row['title']}\n\n"
        f"Context: {row['context']}\n\n"
        f"Question: {row['question']}\n\n"
        f"Answer: {row['answer']}"
    ).strip()

    doc = Document(
        text=text,
        metadata={
            "source_type": "csv",
            "file_name":   f"CSV: {row['title'][:60]}",
            "title":       row["title"],
            "dataset":     "rajpurkar/squad",
        },
        id_=f"csv_{hash(row['title'])}"
    )
    documents.append(doc)

print(f"✅ Converted {len(documents)} CSV rows to documents\n")

# ── Step 5: Split into chunks ─────────────────────────────────
print("Splitting into chunks...")
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
print(f"✅ Created {len(nodes)} chunks from CSV\n")

# ── Step 6: Add to EXISTING ChromaDB collection ───────────────
print("Connecting to existing ChromaDB collection...")
chroma_client     = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION)

count_before = chroma_collection.count()
print(f"   Chunks already in collection (PDFs + Wikipedia): {count_before}\n")

vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ── Step 7: Embed and store ────────────────────────────────────
print("Embedding CSV chunks and storing in ChromaDB...")
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True
)

count_after = chroma_collection.count()
print(f"\n✅ CSV ingestion complete!")
print(f"   Before : {count_before} chunks  (PDFs + Wikipedia)")
print(f"   After  : {count_after} chunks  (PDFs + Wikipedia + CSV)")
print(f"   Added  : {count_after - count_before} new chunks\n")

# ── Step 8: Final 3-source retrieval test ─────────────────────
print("─" * 55)
print("Final test — all 3 sources should appear below...")
print("─" * 55)

TEST_QUERY = "What methods improve retrieval accuracy in question answering?"

retriever = index.as_retriever(similarity_top_k=5)
results   = retriever.retrieve(TEST_QUERY)

print(f'\nQuery: "{TEST_QUERY}"\n')

source_types_seen = set()
for i, result in enumerate(results, 1):
    source_type = result.metadata.get("source_type", "pdf")
    file_name   = result.metadata.get("file_name", "unknown")
    score       = result.score or 0
    preview     = result.text[:220].replace("\n", " ")
    source_types_seen.add(source_type)

    badge = {"wikipedia": "🌐 WIKI", "csv": "📊 CSV"}.get(source_type, "📄 PDF")

    print(f"Result {i}  {badge}")
    print(f"  Source : {file_name}")
    print(f"  Score  : {score:.4f}")
    print(f"  Preview: {preview}...")
    print()

print("─" * 55)
print(f"✅ Sources seen in results: {source_types_seen}")
if len(source_types_seen) >= 2:
    print("🎉 Multi-source RAG is fully working!")
else:
    print("ℹ️  Only one source appeared — try a broader query.")

print("\n   TOTAL chunks in ChromaDB:", chroma_collection.count())
print("   Next step: build query.py (the main Q&A engine)")
print("─" * 55)