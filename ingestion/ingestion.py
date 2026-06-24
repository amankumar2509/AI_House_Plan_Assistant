import chromadb

from sentence_transformers import SentenceTransformer

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text


# -------------------------
# ChromaDB Setup
# -------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="house_plans"
)


# -------------------------
# Embedding Model
# -------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -------------------------
# PDF Path
# -------------------------

pdf_path = "data/uploads/house1.pdf"


# -------------------------
# Load PDF
# -------------------------

print("Loading PDF...")

text = load_pdf(pdf_path)

print(f"Text Length: {len(text)}")


# -------------------------
# Chunk Text
# -------------------------

chunks = chunk_text(text)

print(f"Created {len(chunks)} chunks")


# -------------------------
# Generate Embeddings
# -------------------------

print("Generating embeddings...")

embeddings = model.encode(
    chunks
).tolist()


# -------------------------
# Store in ChromaDB
# -------------------------

ids = []

metadatas = []

for i in range(len(chunks)):

    ids.append(f"house1_chunk_{i}")

    metadatas.append(
        {
            "source": "house1.pdf"
        }
    )


collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Stored in ChromaDB")


print(
    f"Collection Count: {collection.count()}"
)

print("Ingestion Complete")