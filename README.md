# 🤖 GitHub Repository Q&A Assistant

A Streamlit-powered RAG (Retrieval-Augmented Generation) application that allows you to paste any public GitHub repository link, clone & index it automatically, and chat with an AI assistant powered by **Google Gemini** and **LlamaIndex** to get simple explanations, architectural breakdowns, file references, and code snippets.

---

## 🚀 Features

1. **Repo Ingestion**: Clones public GitHub repositories and indexes code files using LlamaIndex and ChromaDB.
2. **Gemini Integration**: Utilizes Google Gemini (`gemini-1.5-pro` and `embedding-001`) for high-fidelity code understanding and embedding generation.
3. **Interactive Chat**: Streamlit UI with conversation history, architectural breakdowns, code snippets, and direct source file citations.

---

## 🛠️ Installation & Setup

1. **Clone or Download this repository**.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API Key (Optional)**:
   Copy `.env.example` to `.env` and add your Gemini API Key:
   ```bash
   cp .env.example .env
   ```
   *(Or you can enter your API key directly in the Streamlit sidebar).*

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Project Structure

- `app.py`: Main Streamlit frontend and chat interface.
- `indexer.py`: Handles repository cloning, code file filtering, and vector embedding storage in ChromaDB.
- `query_engine.py`: Configures the LlamaIndex retriever and Gemini QA prompt template for grounded answers.
- `requirements.txt`: Python package dependencies.
