# 📚 AI PDF Chatbot with FAISS (RAG-powered)

An intelligent chatbot that lets you **chat with your PDFs**, powered by **FAISS + DeepSeek + Ollama**. Upload any document and ask questions in natural language — get answers grounded in your file with Retrieval-Augmented Generation (RAG).

![screenshot](preview.png)

---

## ✨ Features

- 🧠 **DeepSeek RAG**: Powerful responses from your PDF content only
- 📁 **PDF Upload**: Select page ranges to narrow focus
- 🧩 **Smart Chunking**: Document split into overlapping chunks for better context
- 🔍 **FAISS Semantic Search**: Efficient retrieval of relevant chunks
- 💬 **Chat Interface**: Conversation history maintained across questions
- ⚡ **Local & Fast**: Runs with Ollama locally – no OpenAI key needed

---

## 🚀 Setup
### 1. Install Requirements
pip install streamlit PyPDF2 faiss-cpu sentence-transformers
### 2. Install and Run Ollama
### 3. Run the streamlit app
---

## 📌 Use Cases
- Study notes assistant
- Academic paper Q&A
- Chat with policies, manuals, or reports
- Foundation for flashcard generation or document summarization

---

## 🔧 Roadmap Ideas
- 📝 Flashcard generation toggle
- 💾 Multi-file persistent vector DB
- 📄 Support DOCX and TXT
- 🗃️ Save/load conversation history

---

## 🧠 Credits
- Ollama
- DeepSeek
- FAISS
- SentenceTransformers
- Streamlit
