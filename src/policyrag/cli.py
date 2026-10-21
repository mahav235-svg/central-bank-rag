"""Command line interface: ``cbrag --help``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from policyrag.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Fed and RBI policy document QA.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation suites.")
app.add_typer(eval_app, name="eval")


@app.callback()
def _setup(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def ingest(
    bank: Annotated[list[str] | None, typer.Option(help="fed and/or rbi (default both)")] = None,
    refresh: Annotated[bool, typer.Option(help="re-parse documents already in the catalog")] = False,
    offline: Annotated[bool, typer.Option(help="use only the local download cache")] = False,
) -> None:
    """Download new documents, parse and chunk them into the catalog."""
    from policyrag.pipeline import ingest as run

    report = run(get_settings(), bank or ["fed", "rbi"], refresh=refresh, offline=offline)
    typer.echo(f"listed {report.listed}, stored {report.fetched}, already had {report.skipped_existing}")
    if report.failed:
        typer.echo(f"failed ({len(report.failed)}): {', '.join(report.failed)}", err=True)


@app.command()
def rechunk() -> None:
    """Re-chunk stored documents with the current chunk settings (no network)."""
    from policyrag.pipeline import rechunk as run

    typer.echo(f"re-chunked {run(get_settings())} documents; run `cbrag index` next")


@app.command()
def index(full: Annotated[bool, typer.Option(help="drop and rebuild the collection")] = False) -> None:
    """Sync the vector index with the catalog (only new or changed chunks are embedded)."""
    from policyrag.pipeline import build_index

    embedded, deleted = build_index(get_settings(), full=full)
    typer.echo(f"embedded {embedded} chunks, removed {deleted} stale points")


@app.command()
def stats() -> None:
    """Corpus summary from the catalog."""
    from policyrag.catalog import Catalog

    with Catalog(get_settings().catalog_path) as cat:
        typer.echo(
            f"{'bank':5} {'type':20} {'docs':>5} {'first':>11} {'last':>11} {'words':>9} {'chunks':>7}"
        )
        for bank, dtype, docs, first, last, words, chunks in cat.summary():
            typer.echo(f"{bank:5} {dtype:20} {docs:5d} {first!s:>11} {last!s:>11} {words:9d} {chunks:7d}")


@app.command()
def search(
    query: str,
    k: int = 5,
    mode: str = "hybrid",
    rerank: Annotated[
        bool | None, typer.Option("--rerank/--no-rerank", help="default: CBRAG_USE_RERANKER")
    ] = None,
) -> None:
    """Retrieval only; prints the top passages."""
    from policyrag.pipeline import make_retriever

    settings = get_settings()
    use_rerank = settings.use_reranker if rerank is None else rerank
    for h in make_retriever(settings, mode=mode, rerank=use_rerank).retrieve(query, top_k=k):
        c = h.chunk
        who = c.speaker or c.heading or ""
        typer.echo(f"\n#{h.rank} {h.score:.3f} {c.doc_id} {who}\n{c.text[:500]}")


@app.command()
def ask(question: str, show_trace: bool = False) -> None:
    """Ask a question. In interrupt review mode, a flagged answer is reviewed here before it is shown."""
    from policyrag.agent.service import PolicyAgent

    agent = PolicyAgent(get_settings())
    res = agent.ask(question)
    if res.status == "awaiting_review" and res.review:
        typer.echo("\nDraft needs review. Issues: " + "; ".join(res.review.get("issues", [])))
        typer.echo(f"\nDraft:\n{res.review.get('draft_answer', '')}\n")
        action = typer.prompt("approve / edit / reject", default="reject")
        edited = typer.prompt("Edited answer") if action == "edit" else None
        res = agent.resume(res.thread_id, action, edited)
    typer.echo(f"\n[{res.status}] {res.answer}\n")
    for s in res.sources:
        typer.echo(f"  [{s['n']}] {s['title']} ({s['date']}) {s['url']}")
    if show_trace:
        typer.echo("\ntrace:\n  " + "\n  ".join(res.trace) + f"\ntokens: {res.tokens}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("policyrag.api.app:app", host=host, port=port)


@app.command()
def mcp() -> None:
    """Run the MCP server over stdio."""
    from policyrag.mcp_server import main

    main()


@app.command()
def export(out: Path = Path("data/processed")) -> None:
    """Export chunks to Parquet."""
    from policyrag.catalog import Catalog

    with Catalog(get_settings().catalog_path) as cat:
        typer.echo(str(cat.export_parquet(out)))


@eval_app.command("retrieval")
def eval_retrieval(
    golden: Path = Path("eval/golden.jsonl"), k: int = 5, report: Path = Path("docs/results/retrieval.md")
) -> None:
    """Compare retrieval variants on the golden set (no LLM calls)."""
    from policyrag.catalog import Catalog
    from policyrag.eval.run import load_golden, markdown_table, run_retrieval_suite
    from policyrag.pipeline import make_retriever, open_store

    settings = get_settings()
    items = load_golden(golden)
    store = open_store(settings)
    with Catalog(settings.catalog_path) as cat:
        summary = run_retrieval_suite(lambda **o: make_retriever(settings, store=store, **o), items, cat, k=k)
    table = markdown_table(
        summary, f"Retrieval, {len([i for i in items if i.answerable])} answerable questions, k={k}"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(table, encoding="utf-8")
    typer.echo(table)


@eval_app.command("answer")
def eval_answer(
    golden: Path = Path("eval/golden.jsonl"),
    label: str = "",
    report: Path = Path("docs/results/answers.md"),
    limit: int = 0,
) -> None:
    """Run the full agent on the golden set and score the answers (uses the configured LLM)."""
    from policyrag.agent.service import PolicyAgent
    from policyrag.catalog import Catalog
    from policyrag.eval.run import load_golden, markdown_table, run_answer_suite

    settings = get_settings()
    items = load_golden(golden)[: limit or None]
    agent = PolicyAgent(settings, durable=False)
    label = label or f"{settings.provider}"
    with Catalog(settings.catalog_path) as cat:
        summary = run_answer_suite(agent, items, cat, label)
    table = markdown_table({label: summary}, f"Answers, {len(items)} questions")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(table, encoding="utf-8")
    typer.echo(table)


@app.command("review-queue")
def review_queue() -> None:
    """List answers queued for human review (queue mode)."""
    path = get_settings().review_queue_path
    if not path.exists():
        typer.echo("queue is empty")
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        typer.echo(f"- {item['queued_at']}  {item['question']}\n  issues: {'; '.join(item['issues'])}")


if __name__ == "__main__":
    app()
