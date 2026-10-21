"""Dense and sparse text encoders (fastembed, ONNX on CPU).

* Dense: ``BAAI/bge-small-en-v1.5`` (384 dimensions). Small enough to embed the whole corpus on a
  laptop CPU in a few minutes, and strong on the MTEB retrieval benchmarks for its size.
* Sparse: ``Qdrant/bm25``. Classic BM25 term weights computed client-side; Qdrant applies the IDF
  part at query time (``Modifier.IDF``), so document vectors stay valid as the corpus grows.

Central bank text is full of exact phrases ("25 basis points", "standing deposit facility",
"withdrawal of accommodation") that dense models blur together, which is why both are used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

# BGE v1.5 works without an instruction, but its model card recommends this prefix for short
# retrieval queries; the evaluation compares both.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


class Encoder:
    def __init__(
        self,
        dense_model: str,
        sparse_model: str,
        query_instruction: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.dense_model_name = dense_model
        self.sparse_model_name = sparse_model
        self.query_instruction = query_instruction if "bge" in dense_model.lower() else None
        self.batch_size = batch_size

    @cached_property
    def _dense(self) -> Any:
        from fastembed import TextEmbedding

        return TextEmbedding(self.dense_model_name)

    @cached_property
    def _sparse(self) -> Any:
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(self.sparse_model_name)

    @property
    def dense_dim(self) -> int:
        return len(self.embed_dense(["probe"])[0])

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._dense.embed(list(texts), batch_size=self.batch_size)]

    def embed_sparse(self, texts: Sequence[str]) -> list[SparseVec]:
        return [
            SparseVec(e.indices.tolist(), e.values.tolist())
            for e in self._sparse.embed(list(texts), batch_size=self.batch_size)
        ]

    def query_dense(self, query: str) -> list[float]:
        text = f"{self.query_instruction}{query}" if self.query_instruction else query
        return self.embed_dense([text])[0]

    def query_sparse(self, query: str) -> SparseVec:
        e = next(iter(self._sparse.query_embed(query)))
        return SparseVec(e.indices.tolist(), e.values.tolist())


class Reranker:
    """Cross-encoder reranker. Scores (query, passage) pairs jointly, which is slower than the
    bi-encoder but much better at telling a passage that answers the question from one that merely
    shares its vocabulary."""

    def __init__(self, model: str) -> None:
        self.model_name = model

    @cached_property
    def _model(self) -> Any:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(self.model_name)

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, list(passages))]
