import pymupdf

from policyrag.parsing.transcripts import fed_turns, pdf_lines, rbi_turns


def test_fed_turns_split_on_capitalised_speaker_names() -> None:
    lines = [
        "CHAIR POWELL.  Good afternoon. My colleagues and I remain focused",
        "on our dual mandate.",
        "NICK TIMIRAOS.  Under what circumstances would a larger cut be warranted?",
        "CHAIR POWELL.  There wasn't widespread support for a 50 basis point cut.",
        "The U.S. economy is in a good place.",
    ]
    turns = fed_turns(lines)
    assert [t.speaker for t in turns] == ["Chair Powell", "Nick Timiraos", "Chair Powell"]
    assert turns[0].text == "Good afternoon. My colleagues and I remain focused on our dual mandate."
    # "U.S." at the start of a line must not be mistaken for a speaker
    assert turns[2].text.endswith("The U.S. economy is in a good place.")


def test_pdf_lines_drop_running_headers() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in [
        "September 17, 2025",
        "Chair Powell's Press Conference",
        "FINAL",
        "Page 1 of 26",
        "CHAIR POWELL.  Good afternoon.",
    ]:
        page.insert_text((72, y), line)
        y += 16
    lines = pdf_lines(doc.tobytes())
    assert lines == ["CHAIR POWELL. Good afternoon."]


def test_rbi_turns_handle_tag_lines_and_inline_tags() -> None:
    lines = [
        "Brij Raj:",
        "Good afternoon, everyone. Welcome to this post-policy press conference.",
        "Sanjay Malhotra: Thank you. Let me begin with the decision.",
        "The repo rate is unchanged.",
        "Note: figures are provisional.",
    ]
    turns = rbi_turns(lines)
    assert [t.speaker for t in turns] == ["Brij Raj", "Sanjay Malhotra"]
    assert "Note: figures are provisional." in turns[1].text


def test_pdf_lines_drop_chairman_headers() -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(
        [
            "September 16, 2026",
            "Chairman Warsh's Press Conference",
            "FINAL",
            "Page 1 of 15",
            "Transcript of Chairman Warsh's Press Conference",
            "CHAIRMAN WARSH.  Good day.",
        ]
    ):
        page.insert_text((72, 72 + 16 * i), line)
    assert pdf_lines(doc.tobytes()) == ["CHAIRMAN WARSH. Good day."]
