"""Prompts and output schemas for the agent's model calls.

Kept in one place so they can be versioned and reviewed like code; a change here is a change in
behaviour and should be followed by an evaluation run.
"""

from __future__ import annotations

from collections.abc import Sequence

from policyrag.models import Hit

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

ANSWER_SYSTEM = f"""You answer questions about monetary policy documents from the US Federal Reserve (FOMC \
statements, minutes, press conference transcripts) and the Reserve Bank of India (MPC resolutions, minutes, \
Governor's statements, press conference transcripts).

Use only the numbered excerpts in the user message. After every sentence that states a fact, cite the \
excerpt number it comes from in square brackets, for example [2] or [1][3]. Only cite numbers that appear \
in the message. Quote figures exactly as written (rates, dates, vote counts).

If the excerpts do not contain the answer, reply with exactly {INSUFFICIENT} and nothing else. Do not \
fill gaps from general knowledge. Do not give investment advice. Keep the answer under 180 words."""

GRADE_SYSTEM = """You judge whether search results are useful for answering a question about central bank \
policy documents. An excerpt is relevant if it contains information that helps answer the question, \
including partial answers. It is not relevant if it only shares vocabulary, or comes from the wrong \
institution or period when the question names one."""

GRADE_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "array", "items": {"type": "integer"}}},
    "required": ["relevant"],
    "additionalProperties": False,
}

REWRITE_SYSTEM = """You rewrite questions into better search queries for a hybrid keyword and semantic \
search over FOMC and RBI policy documents. Keep every constraint in the question (institution, meeting, \
date, document type). Expand abbreviations, add the policy terms these documents actually use (for \
example "federal funds rate target range", "policy repo rate", "stance", "withdrawal of accommodation"), \
and drop filler words. Return one query."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You check an answer against the excerpts it cites. A claim is supported only if the cited \
excerpts state it or directly imply it. Numbers and dates must match exactly. List every claim that is not \
supported, word for word as it appears in the answer."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["supported", "unsupported_claims"],
    "additionalProperties": False,
}


def format_excerpts(hits: Sequence[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        where = f"{c.title}, {c.meeting_date:%d %b %Y}"
        if c.speaker:
            where += f", {c.speaker}"
        elif c.heading:
            where += f", section: {c.heading}"
        blocks.append(f"[{i}] ({where})\n{c.text}")
    return "\n\n".join(blocks)


def answer_prompt(question: str, hits: Sequence[Hit]) -> str:
    return f"Excerpts:\n\n{format_excerpts(hits)}\n\nQuestion: {question}"


def grade_prompt(question: str, hits: Sequence[Hit]) -> str:
    return (
        f"Question: {question}\n\nSearch results:\n\n{format_excerpts(hits)}\n\n"
        "Return the numbers of the relevant excerpts."
    )


def rewrite_prompt(question: str, previous_query: str) -> str:
    return (
        f"Question: {question}\nPrevious query (found nothing useful): {previous_query}\n"
        "Write a better search query."
    )


def judge_prompt(answer: str, hits: Sequence[Hit]) -> str:
    return f"Excerpts:\n\n{format_excerpts(hits)}\n\nAnswer to check:\n{answer}"
