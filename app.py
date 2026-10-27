
import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote, urlparse

import streamlit as st
from dotenv import load_dotenv

from config import Config
from indexer import index_repository
from query_engine import get_query_engine


# ============================================================
# Configuration
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AskMyRepo",
    page_icon="⌘",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDEX_ROOT = Path("./chroma_db")
DEFAULT_REPO = "https://github.com/octocat/Hello-World"

RETRIEVAL_MODES = {
    "vector": "Vector search",
    "hybrid": "Hybrid search",
    "hybrid_rerank": "Hybrid + reranking",
}


# ============================================================
# UI styling
# ============================================================

st.markdown(
    """
    <style>
    /* Overall application */
    .stApp {
        background: #0b0d12;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 13, 18, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #10131a;
        border-right: 1px solid #252a35;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    /* Typography */
    h1, h2, h3 {
        color: #f3f4f6 !important;
        letter-spacing: -0.035em;
    }

    p, li, label {
        color: #c3c9d4;
    }

    [data-testid="stCaptionContainer"] {
        color: #929baa;
    }

    /* Inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stChatInput"] textarea,
    [data-testid="stSelectbox"] > div > div {
        background: #151923;
        border-color: #303746;
        border-radius: 9px;
        color: #f3f4f6;
    }

    /* Buttons */
    .stButton > button {
        border: 1px solid #303746;
        border-radius: 9px;
        min-height: 2.55rem;
        font-weight: 500;
        transition: border-color 0.15s ease, background 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #8294b5;
        color: #ffffff;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #11151d;
        border: 1px solid #242a36;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stChatInput"] {
        border-color: #303746;
    }

    /* Metrics and separators */
    [data-testid="stMetric"] {
        background: #11151d;
        border: 1px solid #242a36;
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }

    [data-testid="stMetricValue"] {
        color: #f3f4f6;
    }

    hr {
        border-color: #252a35;
    }

    /* Product-style header and metadata */
    .app-eyebrow {
        font-size: 0.72rem;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: #8b9bb6;
        font-weight: 650;
        margin-bottom: 0.45rem;
    }

    .app-subtitle {
        color: #9ba4b4;
        font-size: 0.98rem;
        line-height: 1.65;
        max-width: 740px;
    }

    .repo-panel {
        background: #11151d;
        border: 1px solid #293140;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: 1rem 0 1.25rem 0;
    }

    .repo-label {
        font-size: 0.7rem;
        color: #8f9aae;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .repo-name {
        color: #f3f4f6;
        font-size: 1.15rem;
        font-weight: 650;
        overflow-wrap: anywhere;
    }

    .repo-url {
        color: #91a9d4;
        font-size: 0.82rem;
        overflow-wrap: anywhere;
    }

    .section-label {
        color: #8f9aae;
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0.8rem 0;
    }

    .empty-state {
        background: #10141c;
        border: 1px dashed #333b4a;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
    }

    /* Keep the app comfortable on narrower screens */
    @media (max-width: 768px) {
        .repo-panel {
            padding: 0.85rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Repository and retrieval helpers
# ============================================================

def normalize_repo_url(url: str) -> str:
    """Accept a GitHub repository root URL only."""
    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Use an HTTPS GitHub repository URL.")

    if parsed.netloc.lower() not in {
        "github.com",
        "www.github.com",
    }:
        raise ValueError("Only github.com repository URLs are supported.")

    match = re.fullmatch(
        r"/([A-Za-z0-9-]+)/([A-Za-z0-9_.-]+)/?",
        parsed.path,
    )

    if not match:
        raise ValueError(
            "Use the format https://github.com/owner/repository"
        )

    owner, repo = match.groups()
    repo = repo.removesuffix(".git")

    if not repo or repo in {".", ".."}:
        raise ValueError("Invalid repository name.")

    return f"https://github.com/{owner}/{repo}"


def get_repo_paths(repo_url: str):
    """Give each repository a separate persistence directory."""
    repo_id = hashlib.sha256(
        repo_url.encode("utf-8")
    ).hexdigest()[:16]

    persist_dir = INDEX_ROOT / repo_id

    return persist_dir, persist_dir / "repo_info.json"


def read_repo_info(info_path: Path):
    """Load valid repository metadata."""
    if not info_path.is_file():
        return None

    try:
        with info_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else None

    except (OSError, json.JSONDecodeError):
        logger.warning("Invalid repository metadata at %s", info_path)
        return None


@st.cache_resource(show_spinner=False)
def load_engine(persist_dir: str, mode: str):
    """Cache the query engine by index and retrieval mode."""
    return get_query_engine(
        persist_dir=persist_dir,
        mode=mode,
    )


def clear_engine_cache():
    load_engine.clear()


def make_github_link(repo_url, commit_sha, file_path, start=None, end=None):
    """Create a GitHub file permalink from source metadata."""
    if not isinstance(file_path, str):
        return None

    file_path = file_path.replace("\\", "/").lstrip("/")
    parts = file_path.split("/")

    if (
        not file_path
        or any(part in {"", ".", ".."} for part in parts)
        or "\x00" in file_path
    ):
        return None

    encoded_path = quote(file_path, safe="/")
    encoded_ref = quote(str(commit_sha or "main"), safe="")

    link = f"{repo_url}/blob/{encoded_ref}/{encoded_path}"

    if isinstance(start, int) and not isinstance(start, bool) and start > 0:
        link += f"#L{start}"

        if (
            isinstance(end, int)
            and not isinstance(end, bool)
            and end >= start
        ):
            link += f"-L{end}"

    return link


def get_citations(response, repo_url, commit_sha):
    """Extract and deduplicate citations from retrieved source nodes."""
    citations = {}

    for source in getattr(response, "source_nodes", []) or []:
        node = getattr(source, "node", None)

        if node is None:
            continue

        metadata = node.metadata or {}

        file_path = (
            metadata.get("file_path")
            or metadata.get("path")
        )

        if not file_path:
            continue

        start = metadata.get("start_line")
        end = metadata.get("end_line")

        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or start <= 0
        ):
            start = None

        if (
            not isinstance(end, int)
            or isinstance(end, bool)
            or end < (start or 1)
        ):
            end = None

        link = make_github_link(
            repo_url,
            commit_sha,
            file_path,
            start,
            end,
        )

        if not link:
            continue

        label = str(file_path).replace("`", "'")

        if start:
            label += f":{start}"

            if end:
                label += f"-{end}"

        citations[link] = label

    return list(citations.items())


def render_citations(citations):
    if not citations:
        return

    with st.expander(
        f"Sources · {len(citations)}",
        expanded=False,
    ):
        for link, label in citations:
            st.markdown(f"- [`{label}`]({link})")


def request_index(repo_url: str):
    """Index a repository and update the active session."""
    persist_dir, info_path = get_repo_paths(repo_url)

    try:
        Config.validate()

        with st.spinner("Cloning repository and building index..."):
            persist_dir.mkdir(parents=True, exist_ok=True)

            index_repository(
                repo_url,
                persist_dir=str(persist_dir),
            )

        info = read_repo_info(info_path)

        if not info:
            raise RuntimeError(
                "Indexing completed without valid repo_info.json metadata."
            )

        # Avoid retaining an old cached engine after re-indexing.
        clear_engine_cache()

        st.session_state.active_repo_url = repo_url
        st.session_state.indexed_persist_dir = str(persist_dir)
        st.session_state.messages = []

        st.success("Repository indexed successfully.")
        st.rerun()

    except Exception:
        logger.exception("Repository indexing failed")

        st.error(
            "Indexing failed. Check the repository URL, API settings, "
            "network connection, and application logs."
        )


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_repo_url" not in st.session_state:
    st.session_state.active_repo_url = None

if "indexed_persist_dir" not in st.session_state:
    st.session_state.indexed_persist_dir = None

if "suggested_prompt" not in st.session_state:
    st.session_state.suggested_prompt = None


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## ⌘ AskMyRepo")
    st.caption("Repository intelligence, grounded in source code.")

    st.divider()

    st.markdown("#### REPOSITORY")

    repo_input = st.text_input(
        "GitHub URL",
        value=st.session_state.active_repo_url or DEFAULT_REPO,
        placeholder="https://github.com/owner/repo",
        label_visibility="collapsed",
        help="Enter the root URL of a public GitHub repository.",
    )

    try:
        normalized_url = normalize_repo_url(repo_input)
        candidate_dir, candidate_info_path = get_repo_paths(
            normalized_url
        )
        candidate_info = read_repo_info(candidate_info_path)
        url_error = None

    except ValueError as exc:
        normalized_url = None
        candidate_dir = None
        candidate_info_path = None
        candidate_info = None
        url_error = str(exc)

    if url_error:
        st.caption(url_error)

    st.markdown("#### RETRIEVAL")

    retrieval_mode = st.selectbox(
        "Retrieval strategy",
        options=list(RETRIEVAL_MODES),
        format_func=lambda mode: RETRIEVAL_MODES[mode],
        index=0,
        label_visibility="collapsed",
        help=(
            "Compare retrieval methods based on answer quality "
            "and response latency."
        ),
    )

    if st.button(
        "Index repository  ↗",
        type="primary",
        use_container_width=True,
        disabled=normalized_url is None,
    ):
        request_index(normalized_url)

    if candidate_info and normalized_url:
        if st.button(
            "Load existing index",
            use_container_width=True,
        ):
            st.session_state.active_repo_url = normalized_url
            st.session_state.indexed_persist_dir = str(candidate_dir)
            st.session_state.messages = []
            st.rerun()

    st.divider()

    st.markdown("#### ACTIVE SESSION")

    if (
        st.session_state.active_repo_url
        and st.session_state.indexed_persist_dir
    ):
        st.caption(st.session_state.active_repo_url)

        if st.button("Clear active index", use_container_width=True):
            try:
                root = INDEX_ROOT.resolve()
                target = Path(
                    st.session_state.indexed_persist_dir
                ).resolve()

                if target != root and root in target.parents:
                    clear_engine_cache()

                    if target.exists():
                        shutil.rmtree(target)

                    st.session_state.active_repo_url = None
                    st.session_state.indexed_persist_dir = None
                    st.session_state.messages = []
                    st.session_state.suggested_prompt = None

                    st.rerun()

                else:
                    st.error("Unexpected index directory; nothing was deleted.")

            except OSError:
                logger.exception("Could not clear active index")
                st.error("Could not clear the index. Check application logs.")

    else:
        st.caption("No repository loaded.")

    st.divider()

    st.caption(
        "Powered by Streamlit · LlamaIndex · Gemini"
    )


# ============================================================
# Main header
# ============================================================

header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        '<div class="app-eyebrow">DEVELOPER TOOL / REPOSITORY EXPLORER</div>',
        unsafe_allow_html=True,
    )
    st.title("Understand any codebase.")
    st.markdown(
        '<div class="app-subtitle">'
        "Ask questions about architecture, implementation, and file "
        "relationships. Answers include source links when retrieval "
        "metadata is available."
        "</div>",
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown("")
    st.markdown("")

    if st.session_state.active_repo_url:
        st.markdown("**● Ready**")
    else:
        st.caption("Awaiting repository")


# ============================================================
# Active repository metadata
# ============================================================

active_url = st.session_state.active_repo_url
active_dir_string = st.session_state.indexed_persist_dir
active_info = None

if active_url and active_dir_string:
    active_dir = Path(active_dir_string)
    active_info = read_repo_info(active_dir / "repo_info.json")

    # Verify that session state and the active index agree.
    expected_dir, _ = get_repo_paths(active_url)

    if active_dir.resolve() != expected_dir.resolve():
        active_info = None

    if not active_info:
        st.warning(
            "The active index is missing valid metadata. "
            "Re-index the repository before continuing."
        )

        active_url = None


# ============================================================
# Repository overview
# ============================================================

if active_url and active_info:
    owner = active_info.get("owner", "")
    repo_name = active_info.get("repo_name", "")

    st.markdown(
        f"""
        <div class="repo-panel">
            <div class="repo-label">CURRENT REPOSITORY</div>
            <div class="repo-name">{owner}/{repo_name}</div>
            <div class="repo-url">{active_url}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    languages = active_info.get("languages", [])

    if not isinstance(languages, list):
        languages = []

    files_indexed = active_info.get("files_indexed", "—")
    latest_commit = str(
        active_info.get("latest_commit") or "—"
    )[:12]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Files indexed", files_indexed)

    with col2:
        st.metric("Languages detected", len(languages))

    with col3:
        st.metric("Latest commit", latest_commit)

    if languages:
        st.caption(" · ".join(str(lang) for lang in languages))

    st.divider()


# ============================================================
# Chat interface
# ============================================================

if not active_url or not active_dir_string or not active_info:
    st.markdown(
        """
        <div class="empty-state">
            <div class="section-label">GET STARTED</div>
            <h3>Bring a repository into context.</h3>
            <p>
                Add a public GitHub URL in the sidebar, index the code,
                and start exploring its implementation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Questions you can explore")

    examples = [
        "Where is the application entry point?",
        "Explain the main architecture.",
        "How does data flow through the application?",
        "Which files implement the core functionality?",
    ]

    for i in range(0, len(examples), 2):
        columns = st.columns(2)

        for j, column in enumerate(columns):
            idx = i + j

            if idx < len(examples):
                with column:
                    if st.button(
                        examples[idx],
                        key=f"empty_example_{idx}",
                        use_container_width=True,
                    ):
                        st.info(
                            "Index a repository first, then ask this question."
                        )

else:
    st.markdown("#### Conversation")

    if not st.session_state.messages:
        st.markdown(
            '<div class="section-label">SUGGESTED QUESTIONS</div>',
            unsafe_allow_html=True,
        )

        examples = [
            "Where is the application entry point?",
            "Explain the main architecture.",
            "How are errors handled?",
            "Which files are most important?",
        ]

        cols = st.columns(2)

        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(
                    example,
                    key=f"chat_example_{i}",
                    use_container_width=True,
                ):
                    st.session_state.suggested_prompt = example
                    st.rerun()

    # Replay the conversation on each Streamlit rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("citations"):
                render_citations(message["citations"])

    typed_prompt = st.chat_input(
        "Ask about architecture, functions, files, or implementation..."
    )

    prompt = typed_prompt

    if not prompt and st.session_state.suggested_prompt:
        prompt = st.session_state.suggested_prompt
        st.session_state.suggested_prompt = None

    if prompt and prompt.strip():
        prompt = prompt.strip()

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                Config.validate()

                with st.spinner("Tracing relevant code..."):
                    engine = load_engine(
                        active_dir_string,
                        retrieval_mode,
                    )

                    response = engine.query(prompt)
                    answer = str(response).strip()

                if not answer:
                    answer = (
                        "No answer was generated. Try asking a more "
                        "specific question about the repository."
                    )

                commit_sha = active_info.get("commit_sha") or "main"

                citations = get_citations(
                    response,
                    active_url,
                    commit_sha,
                )

                st.markdown(answer)
                render_citations(citations)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                })

            except Exception:
                logger.exception("Query failed")

                error_message = (
                    "I couldn't complete that query. Check the API "
                    "configuration and application logs, then try again."
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                })

