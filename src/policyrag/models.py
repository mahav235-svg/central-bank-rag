"""Core data types shared across ingestion, indexing, retrieval and the agent."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class Bank(StrEnum):
    FED = "fed"
    RBI = "rbi"


class DocType(StrEnum):
    STATEMENT = "statement"  # FOMC statement / MPC resolution
    MINUTES = "minutes"
    PRESS_CONFERENCE = "press_conference"
    GOVERNOR_STATEMENT = "governor_statement"  # RBI only


class DocRef(BaseModel):
    """A document we know about but have not necessarily downloaded."""

    doc_id: str
    bank: Bank
    doc_type: DocType
    meeting_date: date
    url: str
    title: str = ""


class Section(BaseModel):
    """A contiguous block of a document: a headed section, or one speaker's turn in a transcript."""

    heading: str | None = None
    speaker: str | None = None
    text: str


class Document(BaseModel):
    doc_id: str
    bank: Bank
    doc_type: DocType
    meeting_date: date
    published: date | None = None
    title: str
    url: str
    sections: list[Section]

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    bank: Bank
    doc_type: DocType
    meeting_date: date
    title: str
    url: str
    position: int
    heading: str | None = None
    speaker: str | None = None
    text: str

    def header(self) -> str:
        """A short provenance line prepended to the text before embedding.

        Embedding the bank, document type and date with the passage gives the dense model a
        chance to separate otherwise similar boilerplate from different meetings.
        """
        parts = [
            self.bank.value.upper(),
            self.doc_type.value.replace("_", " "),
            self.meeting_date.isoformat(),
        ]
        if self.heading:
            parts.append(self.heading)
        if self.speaker:
            parts.append(f"speaker: {self.speaker}")
        return " | ".join(parts)


class Hit(BaseModel):
    chunk: Chunk
    score: float
    rank: int = 0
    sources: list[str] = Field(default_factory=list, description="retrievers that returned this chunk")
