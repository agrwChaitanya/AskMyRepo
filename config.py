import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Server-side configuration for AskMyRepo."""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Provider & Model Settings (Provider Agnostic)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-3.6-flash")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    
    @classmethod
    def validate(cls):
        """Validates that necessary server configurations are present."""
        if cls.LLM_PROVIDER == "gemini":
            if not cls.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set in the server environment (.env).")
        # Future providers can add their validation here
        return True
