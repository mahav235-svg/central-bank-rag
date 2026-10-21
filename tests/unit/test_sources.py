from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from policyrag.models import Bank, DocRef, DocType
from policyrag.sources import fed, rbi
from tests.conftest import FIXTURES


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _ref(bank: Bank, doc_type: DocType, d: date, url: str = "https://example.org/doc") -> DocRef:
    return DocRef(
        doc_id=f"{bank.value}-{doc_type.value}-{d:%Y%m%d}",
        bank=bank,
        doc_type=doc_type,
        meeting_date=d,
        url=url,
    )


# ---- Fed -------------------------------------------------------------------------------------


def test_fed_statement_drops_boilerplate() -> None:
    doc = fed.parse_statement(
        _read("fed_statement.html"), _ref(Bank.FED, DocType.STATEMENT, date(2024, 1, 31))
    )
    assert doc.title == "Federal Reserve issues FOMC statement"
    assert doc.published == date(2024, 1, 31)
    texts = [s.text for s in doc.sections]
    assert len(texts) == 2
    assert "5-1/4 to 5-1/2 percent" in texts[1]
    assert not any(t.startswith(("For release", "For media")) for t in texts)


@pytest.mark.parametrize("fixture", ["fed_minutes_2025.html", "fed_minutes_2022.html"])
def test_fed_minutes_both_layouts(fixture: str) -> None:
    doc = fed.parse_minutes(_read(fixture), _ref(Bank.FED, DocType.MINUTES, date(2025, 9, 17)))
    headings = [s.heading for s in doc.sections]
    assert "Attendance" not in headings  # the roll call under the date heading is dropped
    assert headings[0] == "Developments in Financial Markets and Open Market Operations"
    assert any("target range" in s.text for s in doc.sections)
    assert not any("Secretary" in s.text or "Powell, Chair" in s.text for s in doc.sections)


def test_fed_list_refs_from_calendar(tmp_path: Path) -> None:
    calendar = (
        '<a href="/newsevents/pressreleases/monetary20240131a.htm">Statement</a>'
        '<a href="/monetarypolicy/fomcminutes20240131.htm">HTML</a>'
        '<a href="/monetarypolicy/fomcpresconf20240131.htm">Press Conference</a>'
        '<a href="/newsevents/pressreleases/monetary20191030a.htm">Old</a>'
    )

    class Stub:
        def get(self, url: str, cache: bool = True) -> bytes:
            return calendar.encode()

    refs = fed.list_refs(Stub(), start_year=2021)  # type: ignore[arg-type]
    assert [r.doc_id for r in refs] == [
        "fed-minutes-20240131",
        "fed-press_conference-20240131",
        "fed-statement-20240131",
    ]
    presser = next(r for r in refs if r.doc_type is DocType.PRESS_CONFERENCE)
    assert presser.url.endswith("/mediacenter/files/FOMCpresconf20240131.pdf")


# ---- RBI -------------------------------------------------------------------------------------


def test_rbi_classify() -> None:
    assert (
        rbi.classify("Monetary Policy Statement, 2021-22 Resolution of the Monetary Policy Committee")
        is DocType.STATEMENT
    )
    assert (
        rbi.classify("Minutes of the Monetary Policy Committee Meeting, June 3 to 5, 2026") is DocType.MINUTES
    )
    assert rbi.classify("Governor's Statement: June 05, 2026") is DocType.GOVERNOR_STATEMENT
    assert (
        rbi.classify("Edited Transcript of the RBI's Post-Monetary Policy Press Conference")
        is DocType.PRESS_CONFERENCE
    )
    assert rbi.classify("Statement on Developmental and Regulatory Policies") is None


def test_rbi_press_release_sections_and_footer() -> None:
    doc = rbi.parse_press_release(
        _read("rbi_press_release.html"), _ref(Bank.RBI, DocType.STATEMENT, date(2021, 10, 8))
    )
    assert doc.title.startswith("Monetary Policy Statement, 2021-22")
    assert doc.published == date(2021, 10, 8)
    assert [s.heading for s in doc.sections] == [None, None, None, "Assessment"]
    assert doc.sections[1].text == "Keep the reverse repo rate unchanged at 3.35 per cent."
    assert doc.sections[2].text == "Continue with the accommodative stance."  # nested <p> read once
    assert "unchanged at 4 per cent" in doc.sections[0].text
    assert not any("Yogesh" in s.text or "Press Release:" in s.text for s in doc.sections)


def test_rbi_transcript_turns() -> None:
    doc = rbi.parse_transcript(
        _read("rbi_transcript.html"), _ref(Bank.RBI, DocType.PRESS_CONFERENCE, date(2026, 10, 7))
    )
    assert [s.speaker for s in doc.sections] == ["Brij Raj", "Sanjay Malhotra", "Reporter (Mint)"]
    assert "voted unanimously" in doc.sections[1].text


class FakeRbiFetcher:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.posts = 0

    def get(self, url: str, cache: bool = True) -> bytes:
        if url == rbi.POLICY_PAGE:
            return b'<input type="hidden" name="__VIEWSTATE" id="__VIEWSTATE" value="abc" />'
        if url.endswith("prid=63742"):
            return (
                (FIXTURES / "rbi_press_release.html")
                .read_bytes()
                .replace(b"October 6-8, 2021", b"October 5 to 7, 2026")
            )
        if url.endswith("prid=63743"):
            return (
                b'<table class="tablebg"><tr><td class="tableheader">Statement on Developmental and '
                b"Regulatory Policies</td></tr></table>"
            )
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, data: dict[str, str], cache: bool = True) -> bytes:
        assert data["__VIEWSTATE"] == "abc" and data["hdnYear"]
        self.posts += 1
        return (FIXTURES / "rbi_listing.html").read_bytes()


def test_rbi_list_refs_classifies_full_documents(tmp_path: Path) -> None:
    fetcher = FakeRbiFetcher(tmp_path)
    refs = rbi.list_refs(fetcher, start_year=2026)  # type: ignore[arg-type]
    by_id = {r.doc_id: r for r in refs}
    assert set(by_id) == {
        "rbi-press_conference-20261007",
        "rbi-governor_statement-20261007",
        "rbi-statement-20261007",
        "rbi-minutes-20260805",
    }
    assert by_id["rbi-press_conference-20261007"].url.startswith("https://www.rbi.org.in/Scripts/")
    assert by_id["rbi-statement-20261007"].url.endswith("prid=63742")
