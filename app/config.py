"""Configuration for the AI-Powered Healthcare Management System."""
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    LLM_PROVIDER: str       = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str     = os.getenv("OPENAI_API_KEY", "")
    ORCHESTRATOR_MODEL: str = os.getenv("ORCHESTRATOR_MODEL", "gpt-4o")
    WORKER_MODEL: str       = os.getenv("WORKER_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str    = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    CHROMA_PATH: str        = os.getenv("CHROMA_PATH", "./chroma_db")
    DOCS_PATH: str          = os.getenv("DOCS_PATH", "./sample_docs")

    CHUNK_SIZE: int         = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int      = int(os.getenv("CHUNK_OVERLAP", "64"))
    TOP_K: int              = int(os.getenv("TOP_K", "4"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.60"))

    MAX_AGENT_STEPS: int    = int(os.getenv("MAX_AGENT_STEPS", "12"))
    SANDBOX_MODE: bool      = os.getenv("SANDBOX_MODE", "true").lower() == "true"
    APP_PORT: int           = int(os.getenv("APP_PORT", "8000"))
    APP_HOST: str           = os.getenv("APP_HOST", "0.0.0.0")
    DEBUG: bool             = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()
