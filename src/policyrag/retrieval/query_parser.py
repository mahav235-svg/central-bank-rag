"""Turn the constraints a user states in plain words into metadata filters.

"What did the RBI say about food inflation in the June 2024 minutes?" should only search RBI
minutes from June 2024. Embeddings are poor at honouring dates and institutions, so these are
pulled out by rules and applied as hard filters before ranking. Rules are deterministic and easy
to test; anything they miss is left to the ranker rather than guessed.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

from policyrag.index.store import Filters
from policyrag.models import Bank, DocType

_FED = re.compile(
    r"\b(fed|fomc|federal reserve|powell|federal funds|united states|america)\b|(?<!\w)u\.s\.", re.I
)
_RBI = re.compile(
    r"\b(rbi|reserve bank of india|mpc|monetary policy committee|das|malhotra|repo rate|"
    r"india|indian|rupee|crr|slr|liquidity adjustment facility|laf)\b",
    re.I,
)
_DOC_TYPES = [
    (DocType.MINUTES, re.compile(r"\bminutes\b", re.I)),
    (
        DocType.PRESS_CONFERENCE,
        re.compile(
            r"\b(press conference|presser|transcript|reporters?|journalists?|q\s?&\s?a|asked)\b", re.I
        ),
    ),
    (DocType.GOVERNOR_STATEMENT, re.compile(r"\bgovernor'?s statement\b", re.I)),
    (
        DocType.STATEMENT,
        re.compile(r"\b(policy statement|fomc statement|post-meeting statement|resolution)\b", re.I),
    ),
]
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | {
    m.lower(): i for i, m in enumerate(calendar.month_abbr) if m
}
_MONTH_YEAR = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?,?\s+(20\d\d)\b", re.I
)
_YEAR_RANGE = re.compile(r"\b(?:between|from)\s+(20\d\d)\s+(?:and|to|-)\s+(20\d\d)\b", re.I)
_SINCE = re.compile(r"\b(?:since|after)\s+(20\d\d)\b", re.I)
_YEAR = re.compile(r"\b(20\d\d)\b")


@dataclass
class ParsedQuery:
    text: str
    filters: Filters


def parse(query: str, today: date | None = None) -> ParsedQuery:
    today = today or date.today()
    f = Filters()

    fed, rbi = bool(_FED.search(query)), bool(_RBI.search(query))
    if fed != rbi:  # mentioning both (a comparison) or neither leaves the bank open
        f.banks = [Bank.FED if fed else Bank.RBI]

    for doc_type, pattern in _DOC_TYPES:
        if pattern.search(query):
            f.doc_types.append(doc_type)

    if m := _MONTH_YEAR.search(query):
        month, year = _MONTHS[m.group(1).lower()], int(m.group(2))
        f.date_from = date(year, month, 1)
        f.date_to = date(year, month, calendar.monthrange(year, month)[1])
    elif m := _YEAR_RANGE.search(query):
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        f.date_from, f.date_to = date(a, 1, 1), date(b, 12, 31)
    elif m := _SINCE.search(query):
        f.date_from, f.date_to = date(int(m.group(1)), 1, 1), today
    else:
        years = sorted({int(y) for y in _YEAR.findall(query)})
        if len(years) == 1:
            f.date_from, f.date_to = date(years[0], 1, 1), date(years[0], 12, 31)

    return ParsedQuery(text=query, filters=f)
