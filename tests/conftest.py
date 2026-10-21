from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from policyrag.index.embeddings import SparseVec
from policyrag.models import Bank, Chunk, DocType, Hit

FIXTURES = Path(__file__).parent / "fixtures"


class HashEncoder:
    """Deterministic stand-in for the fastembed encoders: bag-of-words hashed into a small vector.

    Similar texts get similar vectors, which is enough to test the store and the retriever
    without downloading models.
    """

    dense_dim = 64

    def _tokens(self, text: str) -> list[str]:
        return [t.strip(".,;:()[]\"'").lower() for t in text.split() if t.strip(".,;:()[]\"'")]

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dense_dim

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dense_dim
            for tok in self._tokens(t):
                v[self._bucket(tok)] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    def embed_sparse(self, texts: Sequence[str]) -> list[SparseVec]:
        out = []
        for t in texts:
            counts: dict[int, float] = {}
            for tok in self._tokens(t):
                idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 100_000
                counts[idx] = counts.get(idx, 0.0) + 1.0
            items = sorted(counts.items())
            out.append(SparseVec([i for i, _ in items], [v for _, v in items]))
        return out

    def query_dense(self, query: str) -> list[float]:
        return self.embed_dense([query])[0]

    def query_sparse(self, query: str) -> SparseVec:
        return self.embed_sparse([query])[0]


def make_chunk(
    text: str,
    doc_id: str = "fed-statement-20240131",
    bank: Bank = Bank.FED,
    doc_type: DocType = DocType.STATEMENT,
    meeting: date = date(2024, 1, 31),
    position: int = 0,
    speaker: str | None = None,
    heading: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#{position:04d}",
        doc_id=doc_id,
        bank=bank,
        doc_type=doc_type,
        meeting_date=meeting,
        title=f"Title of {doc_id}",
        url=f"https://example.org/{doc_id}",
        position=position,
        heading=heading,
        speaker=speaker,
        text=text,
    )


def make_hit(text: str, rank: int = 1, **kw: object) -> Hit:
    return Hit(chunk=make_chunk(text, **kw), score=1.0 / rank, rank=rank)  # type: ignore[arg-type]


@pytest.fixture
def encoder() -> HashEncoder:
    return HashEncoder()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        make_chunk(
            "The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to "
            "5-1/2 percent.",
            "fed-statement-20240131",
            position=0,
        ),
        make_chunk(
            "Inflation has eased over the past year but remains elevated.",
            "fed-statement-20240131",
            position=1,
        ),
        make_chunk(
            "The MPC decided to keep the policy repo rate unchanged at 6.50 per cent.",
            "rbi-statement-20240208",
            Bank.RBI,
            DocType.STATEMENT,
            date(2024, 2, 8),
        ),
        make_chunk(
            "Food inflation remained high, driven by vegetables and pulses.",
            "rbi-minutes-20240208",
            Bank.RBI,
            DocType.MINUTES,
            date(2024, 2, 8),
        ),
        make_chunk(
            "Asked about rate cuts, the Chair said the Committee needs greater confidence on inflation.",
            "fed-press_conference-20240131",
            Bank.FED,
            DocType.PRESS_CONFERENCE,
            date(2024, 1, 31),
            speaker="Chair Powell",
        ),
    ]
