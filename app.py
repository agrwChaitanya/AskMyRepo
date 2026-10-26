import os
import streamlit as st
from dotenv import load_dotenv
from indexer import index_repository
from query_engine import get_query_engine
from config import Config

@st.cache_resource(show_spinner=False)
def load_engine(persist_dir):
    return get_query_engine(persist_dir=persist_dir)

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

# Check if index exists or was created in session
index_exists = os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0
if "indexed" not in st.session_state:
    st.session_state["indexed"] = index_exists

# Main Chat Interface
if not st.session_state["indexed"]:
    st.info("👈 Please enter a GitHub Repository URL in the sidebar, then click **Clone & Index Repository** to get started.")
else:
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
                        query_engine = load_engine(persist_dir)
                        response = query_engine.query(prompt)
                        
                        response_text = str(response)
                        
                        # Extract source nodes/files
                        sources = set()
                        if hasattr(response, "source_nodes"):
                            for node in response.source_nodes:
                                file_path = node.node.metadata.get("file_path") or node.node.metadata.get("path")
                                if file_path:
                                    sources.add(file_path)
                        
                        full_response = response_text
                        if sources:
                            full_response += "\n\n**📁 Referenced Files:**\n"
                            for src in sorted(list(sources)):
                                full_response += f"- `{src}`\n"
                                
                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        err_msg = f"An error occurred while querying the repository: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
        except Exception as e:
            st.error(f"Configuration error: {e}")
