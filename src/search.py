from src.vectorstore import ChromaVectorStore
from langchain_ollama import ChatOllama


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "data/vector_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.2"
    ):

        # Initialize Chroma Vector Store
        self.vectorstore = ChromaVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model
        )

        # If database is empty, build it
        if self.vectorstore.collection.count() == 0:

            print("[INFO] No existing vector database found.")

            from src.data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        else:

            print(
                f"[INFO] Existing vector database found with "
                f"{self.vectorstore.collection.count()} documents."
            )

        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.1,
        )

        print(f"[INFO] Ollama Model Loaded : {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        # Retrieve similar chunks
        results = self.vectorstore.query(
            query_text=query,
            top_k=top_k
        )

        if len(results) == 0:
            return "No relevant documents found."

        # Create context
        context = "\n\n".join(
            [result["content"] for result in results]
        )

        prompt = f"""
            You are a helpful AI assistant.
        
            Answer ONLY using the context below.

            Context:
                {context}

            Question:
                {query}

                Answer:
            """

        response = self.llm.invoke(prompt)

        return response.content



# Example Usage

if __name__ == "__main__":

    rag = RAGSearch()

    query = "What is attention mechanism?"

    answer = rag.search_and_summarize(
        query=query,
        top_k=3
    )

    print("\nAnswer:\n")
    print(answer)
