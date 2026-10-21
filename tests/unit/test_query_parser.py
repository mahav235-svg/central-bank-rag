from datetime import date

from policyrag.models import Bank, DocType
from policyrag.retrieval.query_parser import parse


def test_bank_month_and_doc_type() -> None:
    f = parse("What did the RBI minutes for June 2024 say about food inflation?").filters
    assert f.banks == [Bank.RBI]
    assert f.doc_types == [DocType.MINUTES]
    assert (f.date_from, f.date_to) == (date(2024, 6, 1), date(2024, 6, 30))


def test_comparison_leaves_bank_open() -> None:
    f = parse("How did the Fed and the RBI describe inflation in 2023?").filters
    assert f.banks == []
    assert (f.date_from, f.date_to) == (date(2023, 1, 1), date(2023, 12, 31))


def test_press_conference_words() -> None:
    f = parse("What did Powell tell reporters about tariffs?").filters
    assert f.banks == [Bank.FED]
    assert DocType.PRESS_CONFERENCE in f.doc_types


def test_year_range_and_since() -> None:
    f = parse("FOMC balance sheet runoff between 2022 and 2024").filters
    assert (f.date_from, f.date_to) == (date(2022, 1, 1), date(2024, 12, 31))
    g = parse("RBI liquidity measures since 2025", today=date(2026, 10, 18)).filters
    assert (g.date_from, g.date_to) == (date(2025, 1, 1), date(2026, 10, 18))


def test_pronoun_us_is_not_the_united_states() -> None:
    assert parse("Can you tell us what the MPC decided?").filters.banks == [Bank.RBI]


def test_no_constraints() -> None:
    assert parse("What is forward guidance?").filters.is_empty()
