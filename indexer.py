import os
import shutil
import tempfile
from pathlib import Path
from git import Repo
import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from model_provider import get_llm_and_embeddings

# Common directories and files to exclude during repository parsing
EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

EXCLUDED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf",
    ".zip", ".tar", ".gz", ".rar", ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".lock", ".ipynb"
}

def clone_repo(repo_url: str, target_dir: str) -> str:
    """Clones a public GitHub repository to the target directory."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    try:
        Repo.clone_from(repo_url, target_dir, depth=1)
        return target_dir
    except Exception as e:
        raise RuntimeError(f"Failed to clone repository {repo_url}: {e}")

def setup_llamaindex_settings():
    """Configures global LlamaIndex settings with server-configured LLM and Embeddings."""
    llm, embed_model = get_llm_and_embeddings()
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 200

def index_repository(repo_url: str, persist_dir: str = "./chroma_db") -> VectorStoreIndex:
    """Clones a repository, reads its code files, and indexes them into ChromaDB using LlamaIndex."""
    setup_llamaindex_settings()
    
    # Clear existing vector store database if it exists to ensure old repo data is deleted and replaced
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)
    
    # Create temp directory for cloning
    temp_clone_dir = tempfile.mkdtemp(prefix="github_repo_")
    try:
        print(f"Cloning {repo_url} into {temp_clone_dir}...")
        clone_repo(repo_url, temp_clone_dir)
        
        print("Reading repository files...")
        reader = SimpleDirectoryReader(
            input_dir=temp_clone_dir,
            recursive=True,
            exclude_hidden=True,
            required_exts=None
        )
        
        documents = reader.load_data()
        
        # Filter out documents from excluded directories or binary extensions
        filtered_docs = []
        for doc in documents:
            file_path = doc.metadata.get("file_path", "")
            path_parts = Path(file_path).parts
            
            # Check if any excluded directory is in the path parts
            if any(exc in path_parts for exc in EXCLUDED_DIRS):
                continue
            
            # Check extension
            if Path(file_path).suffix.lower() in EXCLUDED_EXTENSIONS:
                continue
                
            filtered_docs.append(doc)
            
        print(f"Indexed {len(filtered_docs)} document chunks from repository.")
        
        # Initialize Chroma client and collection
        db = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = db.get_or_create_collection("github_repo_qa")
        
        # Assign Chroma as Vector Store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        index = VectorStoreIndex.from_documents(
            filtered_docs, storage_context=storage_context
        )
        
        return index
        
    finally:
        # Cleanup cloned temporary directory
        if os.path.exists(temp_clone_dir):
            shutil.rmtree(temp_clone_dir, ignore_errors=True)
