from datetime import date

import pytest

from policyrag.parsing.text import clean, meeting_end_date, parse_long_date


def test_clean_normalises_whitespace_and_quotes() -> None:
    assert clean("  Governor’s Statement \n\n ") == "Governor's Statement"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("October 8, 2021", date(2021, 10, 8)),
        ("Oct 08, 2021", date(2021, 10, 8)),
        ("8 October 2021", date(2021, 10, 8)),
    ],
)
def test_parse_long_date(text: str, expected: date) -> None:
    assert parse_long_date(text) == expected


def test_parse_long_date_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_long_date("sometime in autumn")


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Minutes of the Monetary Policy Committee Meeting, August 3 to 5, 2026", date(2026, 8, 5)),
        ("Resolution of the Monetary Policy Committee (MPC) October 6-8, 2021", date(2021, 10, 8)),
        (
            "Minutes of the Monetary Policy Committee Meeting, September 29 to October 1, 2025",
            date(2025, 10, 1),
        ),
        ("Governor's Statement: June 05, 2026", date(2026, 6, 5)),
        ("Governor's Statement - May 5, 2021", date(2021, 5, 5)),
        ("Edited Transcript ... Press Conference: October 7, 2026 (Wednesday)", date(2026, 10, 7)),
        ("Minutes of the Monetary Policy Committee Meeting April 5-7, 2021", date(2021, 4, 7)),
        ("Resolution of the Monetary Policy Committee (MPC) May 2 and 4, 2022", date(2022, 5, 4)),
        ("Minutes of the Monetary Policy Committee Meeting March 24, 26 and 27, 2020", date(2020, 3, 27)),
    ],
)
def test_meeting_end_date(title: str, expected: date) -> None:
    assert meeting_end_date(title) == expected


def test_meeting_end_date_none_when_absent() -> None:
    assert meeting_end_date("Meeting of the Monetary Policy Committee - Rescheduled") is None
