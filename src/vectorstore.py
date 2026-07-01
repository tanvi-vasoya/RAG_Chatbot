import os
import uuid
import chromadb
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer

from src.embeddings import EmbeddingPipeline


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str = "data/vector_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Embedding model
        self.model = SentenceTransformer(embedding_model)

        # Chroma Client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Create / Load collection
        self.collection = self.client.get_or_create_collection(
            name="rag_documents"
        )

        print(f"[INFO] Loaded embedding model: {embedding_model}")
        print(f"[INFO] ChromaDB initialized at: {persist_dir}")

    def build_from_documents(self, documents: List[Any]):
        """
        Build vector database from raw LangChain documents
        """

        print(f"[INFO] Building vector store from {len(documents)} raw documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        self.add_documents(chunks, embeddings)

        print("[INFO] Vector Store created successfully.")

    def add_documents(self, chunks: List[Any], embeddings: np.ndarray):

        ids = []
        texts = []
        metadatas = []

        print(f"[INFO] Adding {len(chunks)} chunks to ChromaDB...")

        for i, chunk in enumerate(chunks):

            ids.append(str(uuid.uuid4()))

            texts.append(chunk.page_content)

            metadata = dict(chunk.metadata)
            metadata["chunk_id"] = i

            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        print(f"[INFO] Total documents in collection: {self.collection.count()}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )

        retrieved = []

        for i in range(len(results["ids"][0])):

            retrieved.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return retrieved

    def query(self, query_text: str, top_k: int = 5):

        print(f"[INFO] Query: {query_text}")

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
        )

        return self.search(query_embedding, top_k)


# ------------------------------
# Example Usage
# ------------------------------

if __name__ == "__main__":

    from src.data_loader import load_all_documents

    docs = load_all_documents("data")

    vector_store = ChromaVectorStore()

    vector_store.build_from_documents(docs)

    results = vector_store.query(
        "What is Attention Mechanism?",
        top_k=3,
    )

    print("\nRetrieved Results:\n")

    for i, result in enumerate(results, 1):

        print(f"\nResult {i}")
        print("-" * 40)
        print(result["content"][:300])
        print("\nDistance:", result["distance"])