import os

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

MODES = ("vector", "hybrid", "hybrid_rerank")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_index_and_nodes(persist_dir: str):
    """Loads the Chroma-backed index and the saved chunks (the chunks feed BM25)."""
    db = chromadb.PersistentClient(path=persist_dir)
    collection = db.get_or_create_collection("github_repo_qa")
    index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection))

    docstore = SimpleDocumentStore.from_persist_path(os.path.join(persist_dir, "docstore.json"))
    nodes = list(docstore.docs.values())
    return index, nodes


def build_retriever(index, nodes, mode: str = "hybrid_rerank", top_k: int = 5, candidates: int = 20):
    """Returns (retriever, postprocessors).

    vector         : dense embeddings only (your original behaviour)
    hybrid         : dense + BM25, merged with reciprocal rank fusion
    hybrid_rerank  : hybrid pulls `candidates` chunks, a cross-encoder keeps the best `top_k`

    Call setup_llamaindex_settings() before this (the fusion retriever resolves Settings.llm).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    if mode == "vector":
        return VectorIndexRetriever(index=index, similarity_top_k=top_k), []

    vector = VectorIndexRetriever(index=index, similarity_top_k=candidates)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=candidates)

    rerank = mode == "hybrid_rerank"
    fusion = QueryFusionRetriever(
        [vector, bm25],
        similarity_top_k=candidates if rerank else top_k,
        num_queries=1,            # 1 = no extra LLM calls to rewrite the query
        mode="reciprocal_rerank",
        use_async=False,
    )

    postprocessors = []
    if rerank:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        postprocessors.append(SentenceTransformerRerank(model=RERANK_MODEL, top_n=top_k))
    return fusion, postprocessors


def retrieve(question: str, retriever, postprocessors):
    """Retrieve + postprocess. Used by the eval script so it tests exactly what the app uses."""
    nodes = retriever.retrieve(question)
    for p in postprocessors:
        nodes = p.postprocess_nodes(nodes, query_str=question)
    return nodes