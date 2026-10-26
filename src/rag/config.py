"""All settings in one place, read from environment variables (or a .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless model-cache warning on Windows


def _default_llm() -> str:
    """Use whichever provider has a key set; LLM_MODEL in .env always wins."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic:claude-opus-5-5"
    if os.getenv("OPENAI_API_KEY"):
        return "openai:gpt-5-mini"
    return "anthropic:claude-opus-5-5"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data" / "raw"

    # Qdrant: no QDRANT_URL -> embedded local mode (a folder on disk, no server needed).
    qdrant_url: str | None = os.getenv("QDRANT_URL") or None
    qdrant_path: Path = ROOT / "qdrant_data"
    collection: str = os.getenv("QDRANT_COLLECTION", "docs")

    # Local ONNX models via fastembed, so ingest and retrieval cost nothing to run.
    dense_model: str = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    sparse_model: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
    rerank_model: str = os.getenv("RERANK_MODEL", "jinaai/jina-reranker-v1-tiny-en")
    model_cache: Path = ROOT / ".cache" / "fastembed"
    # Texts per ONNX forward pass. A batch is padded to its longest text, and on a laptop CPU
    # that padding cost more than batching saved: embedding ran at ~9 chunks/s with 1 vs ~6
    # with 16 (256 used 7 GB of RAM), reranking 30 chunks took 1.5 s with 1 vs 4.8 s with 64.
    # On a GPU or a big server CPU, try 32-64.
    encoder_batch: int = int(os.getenv("ENCODER_BATCH", "1"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))  # characters, not tokens
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    candidates: int = int(os.getenv("CANDIDATES", "30"))  # hybrid hits passed to the reranker
    top_k: int = int(os.getenv("TOP_K", "5"))  # reranked hits passed to the LLM

    # "provider:model", anything langchain's init_chat_model accepts.
    llm_model: str = os.getenv("LLM_MODEL") or _default_llm()
    judge_model: str = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL") or _default_llm()


settings = Settings()
