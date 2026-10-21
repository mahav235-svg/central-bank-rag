"""Federal Reserve: FOMC statements, minutes and press-conference transcripts.

Everything is discovered from the FOMC calendar page, which links each meeting's statement
(``/newsevents/pressreleases/monetaryYYYYMMDDa.htm``), minutes (``fomcminutesYYYYMMDD.htm``) and
press-conference page (``fomcpresconfYYYYMMDD.htm``). The transcript itself is a PDF at a
predictable path under ``/mediacenter/files/``. Dates in all three file names are the last day of
the meeting, which we use as the shared ``meeting_date`` key.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from datetime import date, datetime

from bs4 import BeautifulSoup, Tag

from policyrag.models import Bank, DocRef, DocType, Document, Section
from policyrag.parsing.text import clean, meeting_end_date, parse_long_date
from policyrag.parsing.transcripts import fed_turns, pdf_lines
from policyrag.sources.http import Fetcher, FetchError

log = logging.getLogger(__name__)

BASE = "https://www.federalreserve.gov"
CALENDAR = f"{BASE}/monetarypolicy/fomccalendars.htm"

_PATTERNS = {
    DocType.STATEMENT: re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm$"),
    DocType.MINUTES: re.compile(r"/monetarypolicy/fomcminutes(\d{8})\.htm$"),
    DocType.PRESS_CONFERENCE: re.compile(r"/monetarypolicy/fomcpresconf(\d{8})\.htm$"),
}
_SKIP_PREFIXES = (
    "For media inquiries",
    "Implementation Note issued",
    "For release at",
    "Last Update:",
)
# Minutes sections that carry no policy content and only add noise to retrieval.
_DROP_MINUTES_SECTIONS = {"Attendance"}


def doc_id(doc_type: DocType, d: date) -> str:
    return f"fed-{doc_type.value}-{d:%Y%m%d}"


def list_refs(fetcher: Fetcher, start_year: int) -> list[DocRef]:
    soup = BeautifulSoup(fetcher.get(CALENDAR, cache=False).decode("utf-8", "replace"), "html.parser")
    refs: dict[str, DocRef] = {}
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        for doc_type, pattern in _PATTERNS.items():
            m = pattern.search(href)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%Y%m%d").date()
            if d.year < start_year:
                continue
            url = BASE + href if href.startswith("/") else href
            if doc_type is DocType.PRESS_CONFERENCE:
                url = f"{BASE}/mediacenter/files/FOMCpresconf{m.group(1)}.pdf"
            ref = DocRef(
                doc_id=doc_id(doc_type, d), bank=Bank.FED, doc_type=doc_type, meeting_date=d, url=url
            )
            refs[ref.doc_id] = ref
    return sorted(refs.values(), key=lambda r: (r.meeting_date, r.doc_type.value))


def fetch_document(fetcher: Fetcher, ref: DocRef) -> Document:
    body = fetcher.get(ref.url)
    if ref.doc_type is DocType.STATEMENT:
        return parse_statement(body.decode("utf-8", "replace"), ref)
    if ref.doc_type is DocType.MINUTES:
        return parse_minutes(body.decode("utf-8", "replace"), ref)
    if ref.doc_type is DocType.PRESS_CONFERENCE:
        return parse_press_conference(body, ref)
    raise ValueError(f"unsupported Fed document type {ref.doc_type}")


def _article(html: str, url: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find(id="article")
    if not isinstance(article, Tag):
        raise ValueError(f"no #article element in {url}")
    return article


def parse_statement(html: str, ref: DocRef) -> Document:
    article = _article(html, ref.url)
    time_tag = article.find("p", class_="article__time")
    published = parse_long_date(time_tag.get_text()) if isinstance(time_tag, Tag) else ref.meeting_date
    title_tag = article.find(class_="title")
    title = clean(title_tag.get_text()) if isinstance(title_tag, Tag) else "FOMC statement"
    paragraphs = []
    for div in article.find_all("div", recursive=False):
        if "heading" in (div.get("class") or []):
            continue
        for p in div.find_all("p"):
            text = clean(p.get_text(" "))
            if text and not text.startswith(_SKIP_PREFIXES):
                paragraphs.append(text)
    if not paragraphs:
        raise ValueError(f"no paragraphs parsed from {ref.url}")
    return Document(
        doc_id=ref.doc_id,
        bank=Bank.FED,
        doc_type=DocType.STATEMENT,
        meeting_date=ref.meeting_date,
        published=published,
        title=title,
        url=ref.url,
        sections=[Section(heading=None, text=p) for p in paragraphs],
    )


def _leading_heading(p: Tag) -> tuple[str | None, str]:
    """Split a minutes paragraph into (heading, body).

    Two layouts are in use: a paragraph holding only ``<strong>Heading</strong>`` (2023 onward),
    and ``<strong>Heading</strong><br/>First paragraph...`` with the heading inlined (up to 2022).
    """
    first = next((c for c in p.children if not (isinstance(c, str) and not c.strip())), None)
    full = clean(p.get_text(" "))
    if isinstance(first, Tag) and first.name in ("strong", "b"):
        heading = clean(first.get_text(" "))
        if heading and len(heading) < 120:
            return heading.rstrip(":"), clean(full[len(heading) :]).lstrip(": ")
    return None, full


def parse_minutes(html: str, ref: DocRef) -> Document:
    article = _article(html, ref.url)
    sections: list[Section] = []
    heading: str | None = None
    for p in article.find_all("p"):
        new_heading, text = _leading_heading(p)
        if new_heading:
            if set(new_heading) == {"_"}:
                break  # a rule of underscores precedes the Secretary's signature and the footnotes
            heading = new_heading
            # the opening block sits under the meeting-date heading and is the attendance roll
            if meeting_end_date(heading) is not None:
                heading = "Attendance"
        if not text or heading in _DROP_MINUTES_SECTIONS or text.startswith(_SKIP_PREFIXES):
            continue
        sections.append(Section(heading=heading, text=text))
    if not sections:
        raise ValueError(f"no paragraphs parsed from {ref.url}")
    return Document(
        doc_id=ref.doc_id,
        bank=Bank.FED,
        doc_type=DocType.MINUTES,
        meeting_date=ref.meeting_date,
        title=f"Minutes of the Federal Open Market Committee, {ref.meeting_date:%B %d, %Y}",
        url=ref.url,
        sections=sections,
    )


def parse_press_conference(pdf_bytes: bytes, ref: DocRef) -> Document:
    turns = fed_turns(pdf_lines(pdf_bytes))
    if not turns:
        raise ValueError(f"no speaker turns parsed from {ref.url}")
    return Document(
        doc_id=ref.doc_id,
        bank=Bank.FED,
        doc_type=DocType.PRESS_CONFERENCE,
        meeting_date=ref.meeting_date,
        title=f"Transcript of the Chair's press conference, {ref.meeting_date:%B %d, %Y}",
        url=ref.url,
        sections=turns,
    )


def iter_documents(fetcher: Fetcher, refs: list[DocRef]) -> Iterator[Document]:
    for ref in refs:
        try:
            yield fetch_document(fetcher, ref)
        except (FetchError, ValueError) as exc:
            # a meeting without a press conference, or a page whose layout changed; log and move on
            log.warning("skipping %s: %s", ref.doc_id, exc)
