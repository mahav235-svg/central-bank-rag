"""Split press-conference transcripts into speaker turns.

The two banks format transcripts differently:

* Fed (PDF): ``CHAIR POWELL.  Good afternoon...`` with the name in capitals followed by a full stop.
* RBI (HTML): the speaker on its own line ending in a colon, ``Sanjay Malhotra:``, sometimes inline.

Keeping turns intact matters for retrieval: a reporter's question and the Governor's answer are
different evidence, and a chunk that mixes the end of one answer with the next question tends to
match both and answer neither.
"""

from __future__ import annotations

import re

import pymupdf

from policyrag.models import Section
from policyrag.parsing.text import clean

# Two or more capitalised words (each at least two letters) ending in a full stop, at line start.
_FED_SPEAKER = re.compile(r"^((?:[A-Z][A-Z'\-]+\.?\s){1,4}[A-Z][A-Z'\-]+)\.\s+(.*)$")
_FED_HEADER = re.compile(
    r"^(Page \d+ of \d+|FINAL|[A-Z][a-z]+ \d{1,2}, \d{4}"
    r"|(Transcript of )?Chair(man)? \w+'s Press Conference)$"
)
# "Sanjay Malhotra:", "Dr. Poonam Gupta:", "Latha Venkatesh, CNBC TV18:" (reporters add their outlet)
_RBI_SPEAKER = re.compile(r"^([A-Z][A-Za-z0-9.,'()&\- ]{1,80}?)\s*:\s*(.*)$")
_LOWER_OK = {"of", "and", "the", "de", "da"}
_RBI_NOT_SPEAKER = {
    "note",
    "participants from the reserve bank of india",
    "participants from rbi",
    "moderator",
    "date",
    "q",
    "a",
}


def pdf_lines(pdf_bytes: bytes) -> list[str]:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
        lines: list[str] = []
        for page in doc:
            for raw in page.get_text().splitlines():
                line = clean(raw)
                if line and not _FED_HEADER.match(line):
                    lines.append(line)
    return lines


def fed_turns(lines: list[str]) -> list[Section]:
    turns: list[Section] = []
    speaker: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if buf:
            turns.append(Section(speaker=speaker, text=" ".join(buf)))

    for line in lines:
        m = _FED_SPEAKER.match(line)
        if m:
            flush()
            speaker = m.group(1).title()
            buf = [m.group(2)] if m.group(2) else []
        else:
            buf.append(line)
    flush()
    return [t for t in turns if t.text.strip()]


def rbi_turns(lines: list[str], known_speakers: set[str] | None = None) -> list[Section]:
    """Segment RBI transcript lines. A line is a speaker tag when it looks like ``Name:`` and the
    name is short; ``known_speakers`` (from the participants list) makes this stricter when given."""
    turns: list[Section] = []
    speaker: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if buf:
            turns.append(Section(speaker=speaker, text=" ".join(buf)))

    for line in lines:
        m = _RBI_SPEAKER.match(line)
        is_tag = False
        if m:
            name = m.group(1).strip()
            words = name.replace(",", " ").split()
            title_case = all(w[0].isupper() or not w[0].isalpha() or w in _LOWER_OK for w in words)
            looks_like_name = 1 <= len(words) <= 9 and title_case and name.lower() not in _RBI_NOT_SPEAKER
            if known_speakers:
                looks_like_name = looks_like_name and any(k in name for k in known_speakers)
            is_tag = looks_like_name and (not m.group(2) or len(words) <= 5)
        if is_tag and m:
            flush()
            speaker = m.group(1).strip()
            buf = [m.group(2)] if m.group(2) else []
        else:
            buf.append(line)
    flush()
    return [t for t in turns if t.text.strip()]
