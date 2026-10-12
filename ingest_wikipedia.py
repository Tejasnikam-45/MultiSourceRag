# ─────────────────────────────────────────────────────────────
# STEP 1: Run this in your terminal first (if not already done)
# pip install wikipedia
# ─────────────────────────────────────────────────────────────

import wikipedia
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# ── Config — MUST match ingest_pdfs.py exactly ───────────────
CHROMA_DIR  = "./chroma_db"     # same folder as before
COLLECTION  = "rag_papers"      # same collection name as before
CHUNK_SIZE  = 512
CHUNK_OVERLAP = 50

# Add or remove pages here as you like
WIKI_PAGES = [
    "Retrieval-augmented generation",
    "Large language model",
    "Vector database"
]

# ── Step 1: Load embedding model ─────────────────────────────
# Same model as ingest_pdfs.py — must match for vectors to be comparable
print("Loading embedding model...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
print("✅ Embedding model ready\n")

# ── Step 2: Fetch Wikipedia pages ────────────────────────────
print("Fetching Wikipedia pages...")
documents = []

for title in WIKI_PAGES:
    try:
        # auto_suggest=False prevents Wikipedia from guessing a different page
        page = wikipedia.page(title, auto_suggest=False)

        doc = Document(
            text=page.content,
            metadata={
                "source_type": "wikipedia",
                "title":       page.title,
                "url":         page.url,
                # file_name is used for citations later — keep it descriptive
                "file_name":   f"Wikipedia: {page.title}",
            },
            # unique ID prevents duplicate entries if you run this script twice
            id_=f"wiki_{page.title.replace(' ', '_')}"
        )
        documents.append(doc)
        print(f"  ✅ {page.title}  ({len(page.content):,} characters, {page.url})")

    except wikipedia.exceptions.DisambiguationError as e:
        # Happens when the title matches multiple pages
        print(f"  ⚠️  Ambiguous title '{title}'. Try one of: {e.options[:4]}")

    except wikipedia.exceptions.PageError:
        print(f"  ❌ Page not found: '{title}'. Check the spelling.")

if not documents:
    raise RuntimeError("No Wikipedia pages were fetched. Check titles above.")

print(f"\n✅ Fetched {len(documents)} Wikipedia pages\n")

# ── Step 3: Split into chunks ─────────────────────────────────
print("Splitting into chunks...")
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
print(f"✅ Created {len(nodes)} chunks from Wikipedia pages\n")

# ── Step 4: Connect to EXISTING ChromaDB collection ──────────
# This is the key step — we open the same collection PDFs were stored in
# so all sources end up in one searchable place
print("Connecting to existing ChromaDB collection...")
chroma_client     = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION)

count_before = chroma_collection.count()
print(f"   Chunks already in collection (from PDFs): {count_before}\n")

vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ── Step 5: Embed and add to collection ──────────────────────
print("Embedding Wikipedia chunks and adding to ChromaDB...")
print("(Should be faster than PDFs — Wikipedia text is cleaner)\n")

index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    show_progress=True
)

count_after = chroma_collection.count()
print(f"\n✅ Wikipedia chunks added!")
print(f"   Before : {count_before} chunks  (PDFs only)")
print(f"   After  : {count_after} chunks  (PDFs + Wikipedia)")
print(f"   Added  : {count_after - count_before} new chunks\n")

# ── Step 6: Test — verify BOTH sources are retrieved ─────────
# A good multi-source test query should pull from both PDFs and Wikipedia
print("─" * 55)
print("Testing multi-source retrieval (should see both sources)...")
print("─" * 55)

TEST_QUERY = "How do vector databases store and retrieve embeddings?"

retriever = index.as_retriever(similarity_top_k=4)
results   = retriever.retrieve(TEST_QUERY)

print(f'\nQuery: "{TEST_QUERY}"\n')

for i, result in enumerate(results, 1):
    source_type = result.metadata.get("source_type", "pdf")
    file_name   = result.metadata.get("file_name", "unknown")
    score       = result.score or 0
    preview     = result.text[:250].replace("\n", " ")

    # Show source type badge clearly so you can confirm both appear
    badge = "🌐 WIKI" if source_type == "wikipedia" else "📄 PDF"

    print(f"Result {i}  {badge}")
    print(f"  Source : {file_name}")
    print(f"  Score  : {score:.4f}")
    print(f"  Preview: {preview}...")
    print()

print("─" * 55)
print("✅ Wikipedia ingestion complete!")
print("   If you see both 📄 PDF and 🌐 WIKI results above,")
print("   multi-source retrieval is working correctly.")
print("\n   Next step: run ingest_csv.py")
print("─" * 55)