from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_mvp_chunks")
    qdrant_path: str = os.getenv("QDRANT_PATH", "")
    embedding_model_name: str = os.getenv(
        "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    retrieval_fetch_k: int = int(os.getenv("RETRIEVAL_FETCH_K", "24"))
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8108/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "/model")
    vllm_temperature: float = float(os.getenv("VLLM_TEMPERATURE", "0.0"))
    vllm_max_tokens: int = int(os.getenv("VLLM_MAX_TOKENS", "4096"))
    vllm_timeout_s: int = int(os.getenv("VLLM_TIMEOUT_S", "120"))
    # --- Multi-doc / ranking extensions ---
    # Cross-encoder rerank model (empty = disabled)
    rerank_model: str = os.getenv("RERANK_MODEL", "")
    # Maximal Marginal Relevance diversity selection
    use_mmr: bool = os.getenv("USE_MMR", "false").lower() in ("1", "true", "yes")
    # MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.6"))
    # Max chunks to scroll globally for lexical search (multi-doc)
    lexical_global_limit: int = int(os.getenv("LEXICAL_GLOBAL_LIMIT", "2000"))


settings = Settings()
