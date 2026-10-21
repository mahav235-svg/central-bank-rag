"""Graph wiring.

    analyze -> retrieve -> grade_documents --relevant--> generate -> check_answer --ok--> finalize
                  ^               |                                       |
                  |          none relevant                             issues
                  |               v                                       v
                  +------ rewrite_query (bounded)                    human_review
                                  |
                           still none -> no_answer

The loop through ``rewrite_query`` is the corrective step: if grading finds nothing useful, the
question is rewritten once and searched again before the agent gives up. Every answer that fails
a check goes to a person instead of the user.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from policyrag.agent.nodes import AgentConfig, Nodes
from policyrag.agent.state import AgentState
from policyrag.llm.base import ChatModel
from policyrag.retrieval.retriever import Retriever


def build_graph(
    retriever: Retriever,
    llm: ChatModel,
    config: AgentConfig | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    nodes = Nodes(retriever, llm, config or AgentConfig())
    g = StateGraph(AgentState)
    g.add_node("analyze", nodes.analyze)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("generate", nodes.generate)
    g.add_node("check_answer", nodes.check_answer)
    g.add_node("finalize", nodes.finalize)
    g.add_node("no_answer", nodes.no_answer)
    g.add_node("human_review", nodes.human_review)

    g.add_edge(START, "analyze")
    g.add_edge("analyze", "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents", nodes.route_after_grade, ["generate", "rewrite_query", "no_answer"]
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "check_answer")
    g.add_conditional_edges("check_answer", nodes.route_after_check, ["finalize", "human_review"])
    for terminal in ("finalize", "no_answer", "human_review"):
        g.add_edge(terminal, END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def sqlite_checkpointer(path: Path) -> SqliteSaver:
    """A durable checkpointer, so a run paused for human review survives a process restart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
