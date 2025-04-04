import streamlit as st
from PyPDF2 import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama
import re

# ========== CONFIG ==========
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 100

# ========== PAGE SETUP ==========
st.set_page_config(page_title="📚 AI PDF Chat with FAISS", page_icon="🤖")
st.title("📚 Ask Your PDF – Now with FAISS RAG 💬")

# ========== STATE ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.chunks = []

# ========== PDF UPLOAD ==========
uploaded_file = st.file_uploader("📂 Upload your study material (PDF)", type="pdf")

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def create_faiss_index(chunks):
    embeddings = EMBEDDING_MODEL.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings

if uploaded_file:
    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)

    start_page = st.number_input("📄 Start Page", min_value=1, max_value=total_pages, value=1)
    end_page = st.number_input("📄 End Page", min_value=start_page, max_value=total_pages, value=total_pages)

    # Extract text
    text = "\n".join([
        reader.pages[i - 1].extract_text()
        for i in range(start_page, end_page + 1)
        if reader.pages[i - 1].extract_text()
    ])

    if text.strip():
        st.success("✅ Document loaded. You can now ask questions.")

        # Chunk + embed + store in FAISS
        with st.spinner("🔎 Indexing with FAISS..."):
            chunks = chunk_text(text)
            faiss_index, embeddings = create_faiss_index(chunks)
            st.session_state.faiss_index = faiss_index
            st.session_state.chunks = chunks
    else:
        st.error("⚠️ No readable text found in selected pages.")

# ========== CHAT UI ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("💬 Ask something about the PDF...")

if user_query and st.session_state.faiss_index:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Embed the question and retrieve top chunks
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            query_embedding = EMBEDDING_MODEL.encode([user_query])
            D, I = st.session_state.faiss_index.search(np.array(query_embedding), k=5)
            retrieved_chunks = [st.session_state.chunks[i] for i in I[0]]

            context = "\n\n".join(retrieved_chunks)

            # Send to LLM
            response = ollama.chat(
                model="deepseek-r1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. ONLY use the provided context from the document to answer questions. "
                                   "Avoid assumptions or hallucinations."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"
                    }
                ]
            )

            answer = response["message"]["content"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
