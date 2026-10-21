"""Structure-aware chunking.

Chunks never cross a section heading or a change of speaker. Inside a section, whole paragraphs
are packed up to ``max_words``; a paragraph longer than that is split on sentence boundaries with
a small overlap so a sentence near the cut appears in both neighbours.

Word counts stand in for tokens. For English prose, bge-small's WordPiece tokenizer produces
roughly 1.3 tokens per word, so 220 words stays comfortably under the model's 512-token limit even
after the provenance header is prepended.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from policyrag.models import Chunk, DocType, Document, Section

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE.split(text) if s]


def _split_long(text: str, max_words: int, overlap: int) -> list[str]:
    sentences = _sentences(text)
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sent in sentences:
        n = len(sent.split())
        if current and count + n > max_words:
            pieces.append(" ".join(current))
            # carry the tail of the previous piece forward, whole sentences only
            carried: list[str] = []
            carried_words = 0
            for prev in reversed(current):
                w = len(prev.split())
                if carried_words + w > overlap:
                    break
                carried.insert(0, prev)
                carried_words += w
            current, count = carried, carried_words
        current.append(sent)
        count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def _groups(sections: Iterable[Section]) -> Iterator[list[Section]]:
    """Yield runs of consecutive sections that share a heading and speaker."""
    run: list[Section] = []
    for s in sections:
        if run and (s.heading != run[-1].heading or s.speaker != run[-1].speaker):
            yield run
            run = []
        run.append(s)
    if run:
        yield run


def merge_short_turns(turns: list[Section], min_words: int = 40) -> list[Section]:
    """Fold short transcript turns into the turn that follows them.

    A reporter's one-line question and the moderator's "Next question, please" are useless on their
    own; prefixed to the answer that follows, the question becomes the best retrieval key for it.
    The merged section keeps the answering speaker, and the question stays visible in the text as
    "Name: ...".
    """
    merged: list[Section] = []
    pending: list[str] = []
    for t in turns:
        if len(t.text.split()) < min_words:
            pending.append(f"{t.speaker}: {t.text}" if t.speaker else t.text)
            continue
        if pending:
            t = Section(heading=t.heading, speaker=t.speaker, text=" ".join([*pending, t.text]))
            pending = []
        merged.append(t)
    if pending:
        if merged:
            last = merged[-1]
            merged[-1] = Section(
                heading=last.heading, speaker=last.speaker, text=" ".join([last.text, *pending])
            )
        else:
            merged.append(Section(text=" ".join(pending)))
    return merged


def chunk_document(doc: Document, max_words: int = 220, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = merge_short_turns(doc.sections) if doc.doc_type is DocType.PRESS_CONFERENCE else doc.sections

    def emit(text: str, heading: str | None, speaker: str | None) -> None:
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks):04d}",
                doc_id=doc.doc_id,
                bank=doc.bank,
                doc_type=doc.doc_type,
                meeting_date=doc.meeting_date,
                title=doc.title,
                url=doc.url,
                position=len(chunks),
                heading=heading,
                speaker=speaker,
                text=text,
            )
        )

    for run in _groups(sections):
        heading, speaker = run[0].heading, run[0].speaker
        buf: list[str] = []
        words = 0
        for section in run:
            n = len(section.text.split())
            if n > max_words:
                if buf:
                    emit(" ".join(buf), heading, speaker)
                    buf, words = [], 0
                for piece in _split_long(section.text, max_words, overlap):
                    emit(piece, heading, speaker)
                continue
            if buf and words + n > max_words:
                emit(" ".join(buf), heading, speaker)
                buf, words = [], 0
            buf.append(section.text)
            words += n
        if buf:
            emit(" ".join(buf), heading, speaker)
    return chunks
