from src.data_loader import load_all_documents
from src.vectorstore import ChromaVectorStore
from src.search import RAGSearch


# Example Usage
if __name__ == "__main__":

    # Load all documents
    docs = load_all_documents("data")

    # Initialize Chroma Vector Store
    store = ChromaVectorStore(
        persist_dir="data/vector_store"
    )

    # Build vector database only if it's empty
    if store.collection.count() == 0:
        print("[INFO] Building Vector Store...")
        store.build_from_documents(docs)
    else:
        print(
            f"[INFO] Existing Vector Store found with "
            f"{store.collection.count()} documents."
        )

    # Example Retrieval
    results = store.query(
        "What is attention mechanism?",
        top_k=3
    )

    print("\nRetrieved Chunks\n")

    for i, result in enumerate(results, 1):
        print("=" * 80)
        print(f"Result {i}")
        print(f"Distance : {result['distance']:.4f}")
        print(result["content"][:300])
        print()

    # ----------------------------
    # RAG Search
    # ----------------------------

    rag = RAGSearch()

    query = "What is attention mechanism?"

    answer = rag.search_and_summarize(
        query=query,
        top_k=3
    )

    print("\nFinal Answer\n")
    print(answer)