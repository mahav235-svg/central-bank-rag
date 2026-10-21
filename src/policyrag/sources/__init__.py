"""Document sources.

Each module exposes ``list_refs(fetcher, start_year)`` and ``iter_documents(fetcher, refs)``.
"""

from policyrag.sources import fed, rbi

SOURCES = {"fed": fed, "rbi": rbi}

__all__ = ["SOURCES", "fed", "rbi"]
