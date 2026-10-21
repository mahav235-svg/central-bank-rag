"""Qdrant collection holding one dense and one sparse vector per chunk.

Runs in Qdrant's embedded local mode (a folder on disk) by default, or against a Qdrant server
when ``CBRAG_QDRANT_URL`` is set (see docker-compose.yml). Hybrid search uses Qdrant's Query API:
both retrievers run as prefetches and their rankings are merged with reciprocal rank fusion.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from qdrant_client import QdrantClient, models

from policyrag.index.embeddings import Encoder
from policyrag.models import Bank, Chunk, DocType, Hit

SearchMode = Literal["dense", "sparse", "hybrid"]
DENSE, SPARSE = "dense", "bm25"
_NAMESPACE = uuid.UUID("5d0f8a64-3b8c-4d1e-9a51-2c7e0f3b9a10")


@dataclass
class Filters:
    banks: list[Bank] = field(default_factory=list)
    doc_types: list[DocType] = field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None

    def is_empty(self) -> bool:
        return not (self.banks or self.doc_types or self.date_from or self.date_to)

    def to_qdrant(self) -> models.Filter | None:
        must: list[models.Condition] = []
        if self.banks:
            must.append(
                models.FieldCondition(key="bank", match=models.MatchAny(any=[b.value for b in self.banks]))
            )
        if self.doc_types:
            must.append(
                models.FieldCondition(
                    key="doc_type", match=models.MatchAny(any=[d.value for d in self.doc_types])
                )
            )
        if self.date_from or self.date_to:
            must.append(
                models.FieldCondition(
                    key="meeting_ymd",
                    range=models.Range(
                        gte=_ymd(self.date_from) if self.date_from else None,
                        lte=_ymd(self.date_to) if self.date_to else None,
                    ),
                )
            )
        return models.Filter(must=must) if must else None


def _ymd(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def embedding_text(chunk: Chunk) -> str:
    return f"{chunk.header()}\n{chunk.text}"


def content_hash(chunk: Chunk) -> str:
    return hashlib.sha1(embedding_text(chunk).encode("utf-8")).hexdigest()


class VectorStore:
    def __init__(
        self, encoder: Encoder, collection: str, path: Path | None = None, url: str | None = None
    ) -> None:
        self.encoder = encoder
        self.collection = collection
        self.is_server = bool(url)
        if url:
            self.client = QdrantClient(url=url)
        else:
            assert path is not None, "either a local path or a server url is required"
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

    def close(self) -> None:
        self.client.close()

    def recreate(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=self.encoder.dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        if not self.is_server:
            return  # payload indexes are a server feature; local mode scans payloads directly
        for key, schema in (
            ("bank", models.PayloadSchemaType.KEYWORD),
            ("doc_type", models.PayloadSchemaType.KEYWORD),
            ("meeting_ymd", models.PayloadSchemaType.INTEGER),
        ):
            self.client.create_payload_index(self.collection, key, schema)

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count

    def upsert(self, chunks: Sequence[Chunk], batch_size: int = 64) -> int:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [embedding_text(c) for c in batch]
            dense = self.encoder.embed_dense(texts)
            sparse = self.encoder.embed_sparse(texts)
            points = [
                models.PointStruct(
                    id=point_id(c.chunk_id),
                    vector={DENSE: d, SPARSE: models.SparseVector(indices=s.indices, values=s.values)},
                    payload={
                        **c.model_dump(mode="json"),
                        "meeting_ymd": _ymd(c.meeting_date),
                        "content_hash": content_hash(c),
                    },
                )
                for c, d, s in zip(batch, dense, sparse, strict=True)
            ]
            self.client.upsert(self.collection, points=points)
        return len(chunks)

    def stored_hashes(self) -> dict[str, str]:
        """Map of point id to content hash for everything in the collection."""
        out: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=1000, offset=offset, with_payload=["content_hash"], with_vectors=False
            )
            out.update({str(p.id): str((p.payload or {}).get("content_hash", "")) for p in points})
            if offset is None:
                return out

    def delete(self, point_ids: Sequence[str]) -> None:
        if point_ids:
            self.client.delete(self.collection, points_selector=models.PointIdsList(points=list(point_ids)))

    def sync(self, chunks: Sequence[Chunk]) -> tuple[int, int]:
        """Bring the collection in line with ``chunks``: embed new or changed chunks, drop the rest.

        Returns (upserted, deleted). Unchanged chunks are not re-embedded, which turns a routine
        update after a new meeting from minutes of CPU into seconds.
        """
        stored = self.stored_hashes()
        wanted = {point_id(c.chunk_id): c for c in chunks}
        changed = [c for pid, c in wanted.items() if stored.get(pid) != content_hash(c)]
        stale = [pid for pid in stored if pid not in wanted]
        self.upsert(changed)
        self.delete(stale)
        return len(changed), len(stale)

    def search(
        self,
        query: str,
        limit: int,
        mode: SearchMode = "hybrid",
        filters: Filters | None = None,
        candidates: int = 30,
    ) -> list[Hit]:
        qfilter = filters.to_qdrant() if filters else None
        if mode == "dense":
            res = self.client.query_points(
                self.collection,
                query=self.encoder.query_dense(query),
                using=DENSE,
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
            )
        elif mode == "sparse":
            s = self.encoder.query_sparse(query)
            res = self.client.query_points(
                self.collection,
                query=models.SparseVector(indices=s.indices, values=s.values),
                using=SPARSE,
                query_filter=qfilter,
                limit=limit,
                with_payload=True,
            )
        else:
            s = self.encoder.query_sparse(query)
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(
                        query=self.encoder.query_dense(query), using=DENSE, filter=qfilter, limit=candidates
                    ),
                    models.Prefetch(
                        query=models.SparseVector(indices=s.indices, values=s.values),
                        using=SPARSE,
                        filter=qfilter,
                        limit=candidates,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        hits = []
        for rank, p in enumerate(res.points, start=1):
            payload = dict(p.payload or {})
            payload.pop("meeting_ymd", None)
            payload.pop("content_hash", None)
            hits.append(
                Hit(chunk=Chunk.model_validate(payload), score=float(p.score), rank=rank, sources=[mode])
            )
        return hits
