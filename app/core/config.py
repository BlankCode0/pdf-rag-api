import os
from pydantic_settings import BaseSettings

# Set this before anything imports tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Settings(BaseSettings):
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    max_file_size_mb: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"    # ← this ignores any extra env vars not in Settings


settings = Settings()