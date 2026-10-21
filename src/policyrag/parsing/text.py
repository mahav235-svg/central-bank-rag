"""Text clean-up and date helpers used by both source parsers."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

_WS = re.compile(r"\s+")
_MONTHS = {
    m.lower(): i
    for i, names in enumerate(
        [
            ("January", "Jan"),
            ("February", "Feb"),
            ("March", "Mar"),
            ("April", "Apr"),
            ("May",),
            ("June", "Jun"),
            ("July", "Jul"),
            ("August", "Aug"),
            ("September", "Sep", "Sept"),
            ("October", "Oct"),
            ("November", "Nov"),
            ("December", "Dec"),
        ],
        start=1,
    )
    for m in names
}
_MONTH = (
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?"
)


def clean(text: str) -> str:
    """Normalise unicode (NBSP, ligatures), straighten quotes and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return _WS.sub(" ", text).strip()


def parse_long_date(text: str) -> date:
    """Parse 'October 8, 2021', 'Oct 08, 2021' or '8 October 2021'."""
    t = clean(text).replace(",", "")
    for fmt in ("%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(t, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {text!r}")


# 'October 6-8, 2021', 'August 3 to 5, 2026', 'September 29 to October 1, 2025', 'May 2 and 4, 2022',
# 'March 24, 26 and 27, 2020', 'June 05, 2026'
_SEP = r"\s*(?:,|-|–|to|and)\s*"
_RANGE = re.compile(
    rf"(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?P<tail>(?:{_SEP}(?:{_MONTH}\s+)?\d{{1,2}}(?!\d))*),?\s+(?P<year>\d{{4}})",
    re.IGNORECASE,
)
_TAIL = re.compile(rf"(?P<month>{_MONTH})?\s*(?P<day>\d{{1,2}})\s*$", re.IGNORECASE)


def meeting_end_date(text: str) -> date | None:
    """Return the last day of the first date or date range mentioned in ``text``."""
    m = _RANGE.search(clean(text))
    if not m:
        return None
    month_name, day, tail = m["month"], m["day"], m["tail"]
    if tail and (t := _TAIL.search(tail)):
        day = t["day"]
        months_in_tail = re.findall(_MONTH, tail, flags=re.IGNORECASE)
        if months_in_tail:
            month_name = months_in_tail[-1]
    return date(int(m["year"]), _MONTHS[month_name.lower().rstrip(".")], int(day))
