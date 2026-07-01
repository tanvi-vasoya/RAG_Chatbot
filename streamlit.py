import streamlit as st
from src.search import RAGSearch

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Cache the RAG pipeline
# -----------------------------
@st.cache_resource
def load_rag():
    return RAGSearch()

rag = load_rag()

# -----------------------------
# Title
# -----------------------------
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions...")

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Display Chat History
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Input
# -----------------------------
if prompt := st.chat_input("Ask anything about your documents..."):

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            response = rag.search_and_summarize(
                query=prompt,
                top_k=3
            )

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )