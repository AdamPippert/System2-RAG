from __future__ import annotations

import os
from pydantic import BaseModel


class Settings(BaseModel):
    # Index storage
    data_dir: str = os.getenv("DATA_DIR", "data")

    # LLM provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()

    # Ollama
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # OpenAI (optional)
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # RAG
    top_k: int = int(os.getenv("TOP_K", "6"))
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # System-2 loops
    max_revision_rounds: int = int(os.getenv("MAX_REVISION_ROUNDS", "2"))
    self_consistency_n: int = int(os.getenv("SELF_CONSISTENCY_N", "5"))
    uncertainty_cv_threshold: float = float(os.getenv("UNCERTAINTY_CV_THRESHOLD", "0.15"))


settings = Settings()
