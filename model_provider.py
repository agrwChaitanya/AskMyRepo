from config import Config
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

def get_llm_and_embeddings():
    """
    Factory function to initialize and return the LLM and Embedding models
    based on the configured provider in server settings.
    Provider-agnostic interface supporting Gemini and extensible to others.
    """
    Config.validate()
    
    provider = Config.LLM_PROVIDER
    
    if provider == "gemini":
        llm = Gemini(model=Config.LLM_MODEL, api_key=Config.GEMINI_API_KEY)
        embed_model = GeminiEmbedding(model_name=Config.EMBEDDING_MODEL, api_key=Config.GEMINI_API_KEY)
        return llm, embed_model
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Please check LLM_PROVIDER in .env.")
