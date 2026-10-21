"""DuckDB-backed catalog of parsed documents, chunks and evaluation results.

Parsed documents are also written as JSON (one file per document) so the corpus can be re-chunked
without re-parsing, and chunks can be exported to Parquet for analysis outside Python.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import date
from importlib import resources
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb

from policyrag.models import Bank, Chunk, DocType, Document


class Catalog:
    def __init__(self, path: Path, read_only: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = duckdb.connect(str(path), read_only=read_only)
        if not read_only:
            self.con.execute(resources.files("policyrag").joinpath("catalog.sql").read_text(encoding="utf-8"))

    def __enter__(self) -> Catalog:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        self.con.close()

    # ---- corpus -------------------------------------------------------------------------------

    def upsert(self, doc: Document, chunks: Sequence[Chunk]) -> None:
        self.con.execute("BEGIN")
        try:
            self.con.execute("DELETE FROM chunks WHERE doc_id = ?", [doc.doc_id])
            self.con.execute("DELETE FROM documents WHERE doc_id = ?", [doc.doc_id])
            self.con.execute(
                "INSERT INTO documents (doc_id, bank, doc_type, meeting_date, published, title, url, "
                "n_sections, n_words) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    doc.doc_id,
                    doc.bank.value,
                    doc.doc_type.value,
                    doc.meeting_date,
                    doc.published,
                    doc.title,
                    doc.url,
                    len(doc.sections),
                    len(doc.text.split()),
                ],
            )
            if chunks:
                self.con.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, position, heading, speaker, n_words, text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        [c.chunk_id, c.doc_id, c.position, c.heading, c.speaker, len(c.text.split()), c.text]
                        for c in chunks
                    ],
                )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def known_doc_ids(self) -> set[str]:
        return {row[0] for row in self.con.execute("SELECT doc_id FROM documents").fetchall()}

    def chunks(self, doc_ids: Iterable[str] | None = None) -> list[Chunk]:
        sql = (
            "SELECT c.chunk_id, c.doc_id, d.bank, d.doc_type, d.meeting_date, d.title, d.url, c.position, "
            "c.heading, c.speaker, c.text FROM chunks c JOIN documents d USING (doc_id)"
        )
        params: list[Any] = []
        if doc_ids is not None:
            ids = list(doc_ids)
            sql += f" WHERE c.doc_id IN ({', '.join('?' for _ in ids)})"
            params = ids
        sql += " ORDER BY d.meeting_date, c.doc_id, c.position"
        return [
            Chunk(
                chunk_id=r[0],
                doc_id=r[1],
                bank=Bank(r[2]),
                doc_type=DocType(r[3]),
                meeting_date=r[4],
                title=r[5],
                url=r[6],
                position=r[7],
                heading=r[8],
                speaker=r[9],
                text=r[10],
            )
            for r in self.con.execute(sql, params).fetchall()
        ]

    def summary(self) -> list[tuple[Any, ...]]:
        return self.con.execute("SELECT * FROM corpus_summary").fetchall()

    def meeting_dates(self, bank: Bank) -> list[date]:
        rows = self.con.execute(
            "SELECT DISTINCT meeting_date FROM documents WHERE bank = ? ORDER BY 1", [bank.value]
        ).fetchall()
        return [r[0] for r in rows]

    def export_parquet(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "chunks.parquet"
        self.con.execute(
            "COPY (SELECT c.*, d.bank, d.doc_type, d.meeting_date, d.title, d.url "
            "FROM chunks c JOIN documents d USING (doc_id)) TO ? (FORMAT parquet)",
            [str(target)],
        )
        return target

    # ---- evaluation ---------------------------------------------------------------------------

    def record_eval(
        self,
        run_id: str,
        kind: str,
        label: str,
        config: dict[str, Any],
        rows: Iterable[tuple[str, str, float | None, str | None]],
    ) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO eval_runs (run_id, kind, label, config) VALUES (?, ?, ?, ?)",
            [run_id, kind, label, json.dumps(config, default=str)],
        )
        self.con.executemany(
            "INSERT OR REPLACE INTO eval_results (run_id, question_id, metric, value, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            [[run_id, q, m, v, d] for q, m, v, d in rows],
        )

    def eval_summary(self) -> list[tuple[Any, ...]]:
        return self.con.execute("SELECT * FROM eval_summary").fetchall()


def save_document(doc: Document, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{doc.doc_id}.json"
    path.write_text(doc.model_dump_json(indent=1), encoding="utf-8")
    return path


def load_documents(directory: Path) -> list[Document]:
    return [
        Document.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))
    ]
