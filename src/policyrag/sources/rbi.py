"""Reserve Bank of India: MPC resolutions, minutes, Governor's statements and press-conference transcripts.

The RBI's monetary policy page (``Annualpolicy.aspx``) is an ASP.NET WebForms page. It shows the
current financial year on a plain GET; earlier years appear only after a form post-back that sets
the hidden ``hdnYear`` field and replays the page's ``__VIEWSTATE``. We do exactly what the year
links in the browser do, one financial year at a time.

Each listing gives:

* "Minutes of the Monetary Policy Committee Meeting, <dates>" press releases,
* "Governor's Statement: <date>" press releases,
* two "Full Document" press releases per meeting, one of which is the MPC resolution (the other is
  the Statement on Developmental and Regulatory Policies, which we skip), and
* a link to the edited press-conference transcript (a content page until mid-2022, the
  speeches page since).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from datetime import date
from html import unescape

from bs4 import BeautifulSoup, Tag

from policyrag.models import Bank, DocRef, DocType, Document, Section
from policyrag.parsing.text import clean, meeting_end_date, parse_long_date
from policyrag.parsing.transcripts import rbi_turns
from policyrag.sources.http import Fetcher, FetchError

log = logging.getLogger(__name__)

BASE = "https://www.rbi.org.in"
POLICY_PAGE = f"{BASE}/Scripts/Annualpolicy.aspx"

_HIDDEN = re.compile(r'<input type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"')
_FOOTER = re.compile(
    r"^(Press Release: \d{4}-\d{4}/\d+|\(.{3,60}\)|(Chief )?General Manager|Director|"
    r"Executive Director|\(\s*\d+ kb\s*\)|Date\s*:.*)$",
    re.IGNORECASE,
)


def doc_id(doc_type: DocType, d: date) -> str:
    return f"rbi-{doc_type.value}-{d:%Y%m%d}"


def _normalise(href: str) -> str:
    href = unescape(href).strip()
    href = re.sub(r"^https?://(www\.)?rbi\.org\.in", BASE, href, flags=re.IGNORECASE)
    if href.startswith("/"):
        href = BASE + href
    return href.replace("/scripts/", "/Scripts/")


def _year_listing(fetcher: Fetcher, fy_end_year: int) -> str:
    # The post-back body carries a fresh __VIEWSTATE on every visit, so the fetcher's request-hash
    # cache would never hit. Listings for closed financial years never change; cache them by year.
    cached = fetcher.cache_dir / "rbi_listings" / f"fy{fy_end_year}.html"
    closed = fy_end_year < date.today().year
    if closed and cached.exists():
        return cached.read_text(encoding="utf-8")
    landing = fetcher.get(POLICY_PAGE, cache=False).decode("utf-8", "replace")
    form = {name: unescape(value) for name, value in _HIDDEN.findall(landing)}
    form["hdnYear"] = str(fy_end_year)
    form["UsrFontCntr$btn"] = ""
    html = fetcher.post(POLICY_PAGE, form, cache=False).decode("utf-8", "replace")
    if closed:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(html, encoding="utf-8")
    return html


def _press_release_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for td in soup.select("table.tablebg td.tableheader"):
        text = clean(td.get_text(" "))
        if text and not _FOOTER.match(text):
            return text
    return ""


def classify(title: str) -> DocType | None:
    t = title.lower()
    if "resolution of the monetary policy committee" in t:
        return DocType.STATEMENT
    if t.startswith("minutes of the monetary policy committee"):
        return DocType.MINUTES
    if "governor" in t and "statement" in t:
        return DocType.GOVERNOR_STATEMENT
    if "transcript" in t and "press conference" in t:
        return DocType.PRESS_CONFERENCE
    return None


def list_refs(fetcher: Fetcher, start_year: int) -> list[DocRef]:
    refs: dict[str, DocRef] = {}
    for fy_end in range(start_year, date.today().year + 2):
        try:
            html = _year_listing(fetcher, fy_end)
        except FetchError as exc:
            log.warning("could not list RBI FY ending %d: %s", fy_end, exc)
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = str(a["href"])
            # transcripts moved from bs_viewcontent.aspx (to mid-2022) to BS_SpeechesView.aspx
            if "prid=" not in href and not re.search(r"BS_SpeechesView|bs_viewcontent", href, re.IGNORECASE):
                continue
            url = _normalise(href)
            title = clean(a.get_text(" "))
            if title.lower() == "full document":
                title = _press_release_title(fetcher.get(url).decode("utf-8", "replace"))
            doc_type = classify(title)
            meeting = meeting_end_date(title) if doc_type else None
            if doc_type and meeting is None:
                # some links are captioned just "Governor's Statement"; the page title carries the date
                page = fetcher.get(url).decode("utf-8", "replace")
                title = _press_release_title(page) or title
                meeting = meeting_end_date(title)
                if meeting is None and doc_type is DocType.GOVERNOR_STATEMENT:
                    # the statement is released on the decision day, so its release date is the key
                    meeting = _published(_content_table(page, url))
                    title = f"{title}: {meeting:%B %d, %Y}" if meeting else title
            if doc_type is None or meeting is None or meeting.year < start_year:
                continue
            ref = DocRef(
                doc_id=doc_id(doc_type, meeting),
                bank=Bank.RBI,
                doc_type=doc_type,
                meeting_date=meeting,
                url=url,
                title=title,
            )
            refs.setdefault(ref.doc_id, ref)
    return sorted(refs.values(), key=lambda r: (r.meeting_date, r.doc_type.value))


def _content_table(html: str, url: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.tablebg")
    if table is None:
        raise ValueError(f"no content table in {url}")
    return table


def _published(table: Tag) -> date | None:
    for td in table.select("td.tableheader"):
        text = clean(td.get_text(" "))
        if text.lower().startswith("date"):
            try:
                return parse_long_date(text.split(":", 1)[1])
            except ValueError:
                return None
    return None


def parse_press_release(html: str, ref: DocRef) -> Document:
    table = _content_table(html, ref.url)
    title = _press_release_title(html) or ref.title
    sections: list[Section] = []
    heading: str | None = None
    # Decisions in MPC resolutions are bullet points ("the MPC decided to: <ul><li>..."), so list
    # items are read alongside paragraphs, in document order, without counting nested text twice.
    for p in table.find_all(["p", "li"]):
        if p.name == "li" and p.find("p") is not None:
            continue  # its paragraphs are visited on their own
        text = clean(p.get_text(" "))
        if not text or _FOOTER.match(text):
            continue
        if "head" in (p.get("class") or []):
            heading = text
            continue
        sections.append(Section(heading=heading, text=text))
    if not sections:
        raise ValueError(f"no paragraphs parsed from {ref.url}")
    return Document(
        doc_id=ref.doc_id,
        bank=Bank.RBI,
        doc_type=ref.doc_type,
        meeting_date=ref.meeting_date,
        published=_published(table),
        title=title,
        url=ref.url,
        sections=sections,
    )


def parse_transcript(html: str, ref: DocRef) -> Document:
    table = _content_table(html, ref.url)
    lines = [clean(x) for x in table.get_text("\n").splitlines()]
    lines = [x for x in lines if x and not _FOOTER.match(x)]
    # drop the title line(s) and the participants list that precede the first spoken turn
    title = next((x for x in lines if "transcript" in x.lower()), ref.title)
    turns = rbi_turns(lines[lines.index(title) + 1 :] if title in lines else lines)
    turns = [t for t in turns if t.speaker]
    if not turns:
        raise ValueError(f"no speaker turns parsed from {ref.url}")
    return Document(
        doc_id=ref.doc_id,
        bank=Bank.RBI,
        doc_type=DocType.PRESS_CONFERENCE,
        meeting_date=ref.meeting_date,
        published=_published(table),
        title=title,
        url=ref.url,
        sections=turns,
    )


def fetch_document(fetcher: Fetcher, ref: DocRef) -> Document:
    html = fetcher.get(ref.url).decode("utf-8", "replace")
    if ref.doc_type is DocType.PRESS_CONFERENCE:
        return parse_transcript(html, ref)
    return parse_press_release(html, ref)


def iter_documents(fetcher: Fetcher, refs: list[DocRef]) -> Iterator[Document]:
    for ref in refs:
        try:
            yield fetch_document(fetcher, ref)
        except (FetchError, ValueError) as exc:
            log.warning("skipping %s: %s", ref.doc_id, exc)
