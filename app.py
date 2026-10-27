import os
import json
import streamlit as st
from dotenv import load_dotenv
from indexer import index_repository
from query_engine import get_query_engine
from config import Config

@st.cache_resource(show_spinner=False)
def load_engine(persist_dir, mode="hybrid_rerank"):
    return get_query_engine(persist_dir=persist_dir, mode=mode)

load_dotenv()

st.set_page_config(
    page_title="GitHub Repo Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GitHub Repository Q&A Assistant")
st.markdown("Paste a public GitHub repository link below, index it, and ask questions about its architecture, code, and file structure powered by **Gemini** & **LlamaIndex**.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

repo_url = st.sidebar.text_input(
    "GitHub Repository URL",
    value="https://github.com/octocat/Hello-World",
    placeholder="https://github.com/owner/repo",
    help="Enter the public GitHub repository URL."
)

retrieval_mode = st.sidebar.selectbox(
    "Retrieval Method",
    options=["hybrid_rerank", "hybrid", "vector"],
    format_func=lambda x: {
        "vector": "Vector (Dense Embeddings only)",
        "hybrid": "Hybrid (Dense + BM25 Search)",
        "hybrid_rerank": "Hybrid + Cross-Encoder Rerank (Recommended)"
    }[x],
    help="User-side control: Change how code chunks are retrieved and ranked."
)

persist_dir = "./chroma_db"

if st.sidebar.button("🚀 Clone & Index Repository", type="primary"):
    if not repo_url:
        st.sidebar.error("Please enter a GitHub repository URL.")
    else:
        with st.spinner(f"Cloning and indexing {repo_url}... This may take a few seconds."):
            try:
                # Validate server configuration before indexing
                Config.validate()
                index_repository(repo_url, persist_dir=persist_dir)
                st.cache_resource.clear()
                st.sidebar.success("Repository successfully indexed!")
                st.session_state["indexed"] = True
                st.session_state["messages"] = []
            except Exception as e:
                st.sidebar.error(f"Error indexing repository: {e}")

# Initialize session state for indexed state (default to False on fresh launch)
if "indexed" not in st.session_state:
    st.session_state["indexed"] = False

# Sidebar utilities for managing previous index
has_saved_index = os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0
if has_saved_index and not st.session_state["indexed"]:
    st.sidebar.divider()
    if st.sidebar.button("📂 Load Previous Index"):
        st.session_state["indexed"] = True
        st.rerun()

if st.session_state["indexed"]:
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Reset / Clear Repository"):
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
        st.session_state["indexed"] = False
        st.session_state["messages"] = []
        st.cache_resource.clear()
        st.rerun()

# Main Chat Interface
if not st.session_state["indexed"]:
    st.info("👈 Please enter a GitHub Repository URL in the sidebar, then click **Clone & Index Repository** to get started.")
else:
    # Display Repository Metadata Header
    repo_info_path = os.path.join(persist_dir, "repo_info.json")
    if os.path.exists(repo_info_path):
        try:
            with open(repo_info_path, "r", encoding="utf-8") as f:
                repo_info = json.load(f)
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6; margin-bottom: 1rem; border: 1px solid #d1d5db;">
                <h3 style="margin-top: 0; color: #1f2937;">📂 <a href="{repo_info.get('repo_url')}" target="_blank" style="text-decoration: none; color: #2563eb;">{repo_info.get('owner')}/{repo_info.get('repo_name')}</a></h3>
                <p style="margin: 0.25rem 0; color: #4b5563;"><b>Owner/Organization:</b> {repo_info.get('owner')} | <b>Created (First Commit):</b> {repo_info.get('created_date')} | <b>Latest Commit:</b> {repo_info.get('latest_commit')}</p>
                <p style="margin: 0.25rem 0; color: #4b5563;"><b>Files Indexed:</b> {repo_info.get('files_indexed')} | <b>Languages:</b> {', '.join(repo_info.get('languages', []))}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the repository (e.g., 'How is authentication structured?', 'Where is main entry point?')"):
        try:
            Config.validate()
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing repository code..."):
                    try:
                        query_engine = load_engine(persist_dir, mode=retrieval_mode)
                        response = query_engine.query(prompt)
                        
                        response_text = str(response)
                        
                        # Extract source nodes/files with line ranges and construct GitHub permalinks
                        citations = {}
                        base_repo_url = "https://github.com"
                        commit_sha = "main"
                        if os.path.exists(repo_info_path):
                            try:
                                with open(repo_info_path, "r", encoding="utf-8") as f:
                                    r_info = json.load(f)
                                base_repo_url = r_info.get("repo_url", "https://github.com").rstrip("/")
                                commit_sha = r_info.get("commit_sha", "main")
                            except Exception:
                                pass

                        if hasattr(response, "source_nodes"):
                            for node in response.source_nodes:
                                file_path = node.node.metadata.get("file_path") or node.node.metadata.get("path")
                                if file_path:
                                    start = node.node.metadata.get("start_line")
                                    end = node.node.metadata.get("end_line")
                                    if file_path not in citations:
                                        citations[file_path] = []
                                    if start and end:
                                        citations[file_path].append((start, end))
                        
                        full_response = response_text
                        if citations:
                            full_response += "\n\n**📁 Clickable Citations:**\n"
                            for fp, ranges in sorted(citations.items()):
                                if ranges:
                                    start, end = ranges[0]
                                    link = f"{base_repo_url}/blob/{commit_sha}/{fp}#L{start}-L{end}"
                                    full_response += f"- [`{fp}:{start}-{end}`]({link})\n"
                                else:
                                    link = f"{base_repo_url}/blob/{commit_sha}/{fp}"
                                    full_response += f"- [`{fp}`]({link})\n"
                                
                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        err_msg = f"An error occurred while querying the repository: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
        except Exception as e:
            st.error(f"Configuration error: {e}")
