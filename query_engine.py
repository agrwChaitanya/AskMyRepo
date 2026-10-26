import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore
from indexer import setup_llamaindex_settings

def get_query_engine(persist_dir: str = "./chroma_db"):
    """Loads the ChromaDB vector store and returns a LlamaIndex query engine configured for GitHub repo Q&A."""
    setup_llamaindex_settings()
    
    try:
        db = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = db.get_or_create_collection("github_repo_qa")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )
        
        # Configure retriever (top 5 relevant chunks)
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
        )
        
        # Configure response synthesizer with custom prompt
        text_qa_template_str = (
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
        
        from llama_index.core import PromptTemplate
        qa_prompt = PromptTemplate(text_qa_template_str)
        
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=qa_prompt
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        return query_engine
        
    except Exception as e:
        raise RuntimeError(f"Failed to load query engine from ChromaDB: {e}")
