import chromadb
from sentence_transformers import SentenceTransformer

# ChromaDB Connection
# -------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="house_plans"
)

# Embedding Model
# -------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Search Function
# -------------------------

def search_house_plans(query, top_k=3):

    query_embedding = model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results

# Main Program
# -------------------------

if __name__ == "__main__":

    query = input("\nAsk a question: ")

    results = search_house_plans(query)

    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(documents)):

        print(f"\nResult #{i+1}")

        print(f"Source : {metadatas[i]['source']}")

        print(f"Distance : {distances[i]:.4f}")

        print("-" * 60)

        print(documents[i])

        print("-" * 60)