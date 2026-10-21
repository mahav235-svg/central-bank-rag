"""MCP server exposing the corpus to MCP clients (Claude Desktop, Claude Code, IDE agents).

Two tools:

* ``search_policy_documents``: retrieval only. The client's own model reads the passages and
  writes the answer, so no API key is needed on this side.
* ``ask_policy_question``: runs the full agent (grading, citation and groundedness checks) and
  returns a checked answer. Needs a configured LLM provider.

Run over stdio with ``cbrag mcp``; see README for client configuration.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any

from mcp.server.mcpserver import MCPServer

from policyrag.agent.service import PolicyAgent
from policyrag.config import get_settings
from policyrag.index.store import Filters
from policyrag.models import Bank, DocType
from policyrag.pipeline import make_retriever
from policyrag.retrieval.retriever import Retriever

server = MCPServer(
    name="central-bank-rag",
    instructions=(
        "Search and cite Federal Reserve (FOMC) and Reserve Bank of India (MPC) monetary policy documents "
        "from 2021 onward: statements, minutes, Governor's statements and press conference transcripts."
    ),
)


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    return make_retriever(get_settings())


@lru_cache(maxsize=1)
def _agent() -> PolicyAgent:
    return PolicyAgent(get_settings(), retriever=_retriever())


@server.tool()
def search_policy_documents(
    query: str,
    bank: str | None = None,
    doc_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Search FOMC and RBI policy documents.

    Args:
        query: what to look for, in plain English.
        bank: "fed" or "rbi" to restrict the search.
        doc_type: one of "statement", "minutes", "press_conference", "governor_statement".
        date_from: earliest meeting date, YYYY-MM-DD.
        date_to: latest meeting date, YYYY-MM-DD.
        k: number of passages to return (1 to 20).
    """
    filters = Filters(
        banks=[Bank(bank)] if bank else [],
        doc_types=[DocType(doc_type)] if doc_type else [],
        date_from=date.fromisoformat(date_from) if date_from else None,
        date_to=date.fromisoformat(date_to) if date_to else None,
    )
    hits = _retriever().retrieve(
        query, filters=None if filters.is_empty() else filters, top_k=max(1, min(k, 20))
    )
    return [
        {
            "title": h.chunk.title,
            "date": h.chunk.meeting_date.isoformat(),
            "bank": h.chunk.bank.value,
            "doc_type": h.chunk.doc_type.value,
            "speaker": h.chunk.speaker,
            "section": h.chunk.heading,
            "url": h.chunk.url,
            "text": h.chunk.text,
        }
        for h in hits
    ]


@server.tool()
def ask_policy_question(question: str) -> dict[str, Any]:
    """Answer a question from FOMC and RBI documents with numbered citations.

    Answers that fail the citation or groundedness checks are withheld and marked for human review.
    """
    res = _agent().ask(question)
    return {"status": res.status, "answer": res.answer, "sources": res.sources}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
