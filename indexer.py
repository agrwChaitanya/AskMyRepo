import os
import shutil
import tempfile
import json
from datetime import datetime
from pathlib import Path
 
from git import Repo
import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from model_provider import get_llm_and_embeddings
 
# Directories never worth indexing (hidden dirs are skipped too)
EXCLUDED_DIRS = {
    "node_modules", "venv", ".venv", "env", "__pycache__",
    "dist", "build", "site-packages", "vendor", "target",
}
 
# Extensions we parse with a language-aware splitter (tree-sitter language names)
CODE_EXTS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust", ".cpp": "cpp", ".c": "c",
}
# Other text files worth indexing (docs, config, frontend, scripts)
TEXT_EXTS = {
    ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".html", ".css", ".sh", ".tsx", ".jsx", ".sql",
}
ALLOWED_EXTS = set(CODE_EXTS) | TEXT_EXTS
SKIP_FILES = {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock"}
MAX_FILE_BYTES = 200_000  # skip huge generated / data files
 
 
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
 
 
def collect_files(root: str) -> list[str]:
    """Walks the repo, pruning excluded dirs BEFORE reading anything."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for name in filenames:
            path = Path(dirpath) / name
            if name in SKIP_FILES or path.suffix.lower() not in ALLOWED_EXTS:
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            files.append(str(path))
    return files
 
 
def make_metadata_fn(root: str):
    """Metadata with repo-relative paths (not temp-dir paths)."""
    def fn(path: str) -> dict:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        ext = Path(rel).suffix.lower()
        return {
            "file_path": rel,
            "file_name": os.path.basename(rel),
            "language": CODE_EXTS.get(ext, "text"),
        }
    return fn
 
 
def _annotate(doc, nodes):
    """Adds line ranges to chunks and keeps noisy metadata out of the embedding text."""
    text = doc.text
    is_md = doc.metadata["file_path"].lower().endswith(".md")  # md readers split docs, so lines would be wrong
    for n in nodes:
        n.excluded_embed_metadata_keys = ["file_name", "start_line", "end_line"]
        n.excluded_llm_metadata_keys = ["file_name"]
        if is_md:
            continue
        chunk = n.get_content(metadata_mode=MetadataMode.NONE)
        start = text.find(chunk)
        if start < 0:
            continue
        n.metadata["start_line"] = text.count("\n", 0, start) + 1
        n.metadata["end_line"] = n.metadata["start_line"] + chunk.count("\n")
 
 
def build_nodes(documents, chunking: str = "code"):
    """chunking='default' -> plain sentence splitter (baseline).
    chunking='code'    -> tree-sitter splitter for known languages, sentence splitter for the rest."""
    if chunking not in ("default", "code"):
        raise ValueError("chunking must be 'default' or 'code'")
 
    default_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    code_splitters = {}
    if chunking == "code":
        # Built up front so a missing tree-sitter install fails loudly instead of silently
        # falling back to the baseline (which would make your eval numbers meaningless).
        code_splitters = {
            lang: CodeSplitter(language=lang, chunk_lines=40, chunk_lines_overlap=10, max_chars=1500)
            for lang in set(CODE_EXTS.values())
        }
 
    nodes, fallbacks = [], 0
    for doc in documents:
        ext = Path(doc.metadata["file_path"]).suffix.lower()
        splitter = code_splitters.get(CODE_EXTS.get(ext))
        try:
            doc_nodes = (splitter or default_splitter).get_nodes_from_documents([doc])
        except Exception:  # e.g. a file with a syntax error
            doc_nodes = default_splitter.get_nodes_from_documents([doc])
            fallbacks += 1
        _annotate(doc, doc_nodes)
        nodes.extend(doc_nodes)
 
    print(f"Built {len(nodes)} chunks from {len(documents)} documents "
          f"(chunking={chunking}, parse fallbacks={fallbacks}).")
    return nodes
 
 
def index_repository(repo_url: str, persist_dir: str = "./chroma_db", chunking: str = "code") -> VectorStoreIndex:
    """Clones a repo, chunks it, embeds into ChromaDB, and saves a docstore (needed for BM25)."""
    setup_llamaindex_settings()
 
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)
 
    temp_clone_dir = tempfile.mkdtemp(prefix="github_repo_")
    try:
        print(f"Cloning {repo_url} into {temp_clone_dir}...")
        clone_repo(repo_url, temp_clone_dir)
 
        files = collect_files(temp_clone_dir)
        if not files:
            raise RuntimeError("No indexable files found in this repository.")
 
        print(f"Reading {len(files)} files...")
        reader = SimpleDirectoryReader(
            input_files=files,
            file_metadata=make_metadata_fn(temp_clone_dir),
        )
        documents = reader.load_data()
        nodes = build_nodes(documents, chunking=chunking)
 
        db = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = db.get_or_create_collection("github_repo_qa")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
 
        # Chroma stores text, so LlamaIndex won't fill the docstore by itself.
        # We keep our own copy so BM25 can be rebuilt from the saved chunks.
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
 
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
        docstore.persist(os.path.join(persist_dir, "docstore.json"))

        # Build and persist BM25 retriever for instant hybrid search loading
        from llama_index.retrievers.bm25 import BM25Retriever
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)
        bm25_retriever.persist(persist_dir)

        # Extract and persist repository metadata
        parts = repo_url.rstrip("/").split("/")
        owner = parts[-2] if len(parts) >= 2 else "Unknown"
        repo_name = parts[-1].replace(".git", "") if len(parts) >= 1 else "Unknown"

        created_date = "Unknown"
        latest_commit = "Unknown"
        try:
            repo = Repo(temp_clone_dir)
            commits = list(repo.iter_commits(max_count=2000))
            if commits:
                oldest = commits[-1]
                created_date = datetime.fromtimestamp(oldest.committed_date).strftime("%Y-%m-%d")
                latest = commits[0]
                latest_commit = datetime.fromtimestamp(latest.committed_date).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        languages = set(doc.metadata.get("language") for doc in documents if doc.metadata.get("language"))

        repo_info = {
            "repo_url": repo_url,
            "owner": owner,
            "repo_name": repo_name,
            "created_date": created_date,
            "latest_commit": latest_commit,
            "files_indexed": len(files),
            "languages": sorted(list(languages)),
        }
        with open(os.path.join(persist_dir, "repo_info.json"), "w", encoding="utf-8") as f:
            json.dump(repo_info, f, indent=2)

        return index
 
    finally:
        if os.path.exists(temp_clone_dir):
            shutil.rmtree(temp_clone_dir, ignore_errors=True)