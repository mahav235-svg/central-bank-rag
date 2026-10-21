"""Runtime settings, read from environment variables (prefix ``CBRAG_``) or a local ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "ollama", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CBRAG_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")

    # corpus
    start_year: int = 2021
    user_agent: str = "central-bank-rag/0.1 (research project; contact via GitHub)"
    request_interval: float = Field(default=1.0, description="seconds between requests to the same host")

    # chunking
    chunk_words: int = 220
    chunk_overlap_words: int = 40

    # retrieval
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True  # best on 4 of 5 metrics once it sees the header; see docs/evaluation.md
    bge_query_instruction: bool = False
    collection: str = "policy_docs"
    qdrant_url: str | None = Field(
        default=None, description="use a Qdrant server; local on-disk mode when unset"
    )
    top_k: int = 6
    candidates: int = 30

    # generation
    provider: Provider = "anthropic"
    anthropic_model: str = "claude-opus-5-5"
    openai_model: str = "gpt-5"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_url: str = "http://localhost:11434"
    max_output_tokens: int = 2000

    # agent
    max_rewrites: int = 1
    review_mode: Literal["interrupt", "queue"] = "queue"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.duckdb"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def review_queue_path(self) -> Path:
        return self.data_dir / "review_queue.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
