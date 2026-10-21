from datetime import date

from policyrag.chunking import chunk_document
from policyrag.models import Bank, DocType, Document, Section


def _doc(sections: list[Section]) -> Document:
    return Document(
        doc_id="fed-minutes-20240131",
        bank=Bank.FED,
        doc_type=DocType.MINUTES,
        meeting_date=date(2024, 1, 31),
        title="Minutes",
        url="https://example.org",
        sections=sections,
    )


def test_chunks_never_cross_headings_or_speakers() -> None:
    doc = _doc(
        [
            Section(heading="Staff Economic Outlook", text="Growth was solid."),
            Section(heading="Staff Economic Outlook", text="Inflation eased."),
            Section(heading="Committee Policy Action", text="The Committee decided to hold."),
            Section(speaker="Chair Powell", text="Good afternoon."),
            Section(speaker="Reporter", text="A question."),
        ]
    )
    chunks = chunk_document(doc, max_words=100)
    assert [(c.heading, c.speaker) for c in chunks] == [
        ("Staff Economic Outlook", None),
        ("Committee Policy Action", None),
        (None, "Chair Powell"),
        (None, "Reporter"),
    ]
    assert chunks[0].text == "Growth was solid. Inflation eased."
    assert [c.position for c in chunks] == [0, 1, 2, 3]
    assert chunks[0].chunk_id == "fed-minutes-20240131#0000"


def test_paragraphs_are_packed_up_to_the_word_budget() -> None:
    para = " ".join(["word"] * 30) + "."
    chunks = chunk_document(_doc([Section(heading="A", text=para) for _ in range(5)]), max_words=70)
    assert [len(c.text.split()) for c in chunks] == [60, 60, 30]


def test_long_paragraph_is_split_on_sentences_with_overlap() -> None:
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(30)]
    chunks = chunk_document(_doc([Section(heading="A", text=" ".join(sentences))]), max_words=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 50 for c in chunks)
    # the last sentence of one chunk opens the next
    first_tail = chunks[0].text.split(". ")[-1]
    assert chunks[1].text.startswith(first_tail.rstrip("."))


def test_short_transcript_turns_are_folded_into_the_answer() -> None:
    doc = Document(
        doc_id="fed-press_conference-20240131",
        bank=Bank.FED,
        doc_type=DocType.PRESS_CONFERENCE,
        meeting_date=date(2024, 1, 31),
        title="Presser",
        url="https://example.org",
        sections=[
            Section(speaker="Michelle Smith", text="Nick."),
            Section(speaker="Nick Timiraos", text="Would a larger cut be warranted?"),
            Section(speaker="Chair Powell", text=" ".join(["answer"] * 60)),
        ],
    )
    chunks = chunk_document(doc, max_words=200)
    assert len(chunks) == 1
    assert chunks[0].speaker == "Chair Powell"
    assert chunks[0].text.startswith("Michelle Smith: Nick. Nick Timiraos: Would a larger cut be warranted?")
