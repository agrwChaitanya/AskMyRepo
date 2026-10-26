from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from indexer import setup_llamaindex_settings
from retrieval import build_retriever, load_index_and_nodes

TEXT_QA_TEMPLATE = (
    "You are an expert GitHub repository assistant. Use the provided context to answer the user's question.\n\n"
    "Guidelines:\n"
    "- Match the structure and depth of your answer directly to the user's query.\n"
    "- For simple or conceptual questions, provide a clear, conversational explanation without forcing unnecessary sections.\n"
    "- Only include architectural breakdowns, file references, or code snippets when they are specifically relevant or requested.\n"
    "- Never use a rigid, repeating 4-part template for every answer.\n\n"
    "Context:\n"
    "{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer: "
)


def get_query_engine(persist_dir: str = "./chroma_db", mode: str = "hybrid_rerank", top_k: int = 5):
    """Loads the saved index and returns a query engine using the chosen retrieval mode."""
    setup_llamaindex_settings()

    try:
        index, nodes = load_index_and_nodes(persist_dir)
        retriever, postprocessors = build_retriever(index, nodes, mode=mode, top_k=top_k, persist_dir=persist_dir)

        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=PromptTemplate(TEXT_QA_TEMPLATE),
        )

        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load query engine: {e}")
