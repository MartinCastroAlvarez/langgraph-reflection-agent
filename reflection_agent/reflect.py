"""Variant 1, basic reflection: generate -> reflect -> revise, until the critic is satisfied.

The generator drafts an answer, the reflector scores it on a fixed rubric
(:data:`~reflection_agent.rubric.RUBRIC`) and lists issues, and the generator revises
with that critique. The loop ends when the weighted score reaches ``threshold``
(``status="passed"``) or after ``max_iterations`` drafts (``status="max_iterations"``).
The answer is the best-scoring draft, which is not always the last one.
"""

from __future__ import annotations

import operator
from typing import Annotated
from typing import Any
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from reflection_agent.config import Settings
from reflection_agent.progress import emit
from reflection_agent.prompts import GENERATOR
from reflection_agent.prompts import REFLECTOR
from reflection_agent.prompts import data_block
from reflection_agent.rubric import model_text
from reflection_agent.rubric import parse_critique

__all__ = ["MAX_ANSWER_CHARS", "ReflectState", "Round", "best_round", "build_reflect"]

MAX_ANSWER_CHARS = 6000


class Round(TypedDict):
    iteration: int
    draft: str
    score: float
    scores: dict[str, float]
    issues: list[str]
    suggestions: list[str]


class ReflectState(TypedDict, total=False):
    task: str
    draft: str
    iteration: int
    rounds: Annotated[list[Round], operator.add]
    status: str
    """running | passed | max_iterations"""
    answer: str
    score: float
    best_iteration: int


def best_round(rounds: list[Round]) -> Round:
    """The highest score; on a tie, the later draft."""
    return max(rounds, key=lambda r: (r["score"], r["iteration"]))


def build_reflect(
    model: BaseChatModel,
    settings: Settings,
    *,
    critic: BaseChatModel | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the loop. ``critic`` defaults to ``model`` (ollama's JSON mode is better)."""
    limits = settings.limits
    reviewer = critic or model

    async def generate(state: ReflectState) -> dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        parts = [data_block("task", state["task"]), data_block("iteration", str(iteration))]
        rounds = state.get("rounds", [])
        if rounds:
            last = rounds[-1]
            parts.append(data_block("draft", last["draft"]))
            parts.append(
                data_block(
                    "critique",
                    {
                        "score": last["score"],
                        "issues": last["issues"],
                        "suggestions": last["suggestions"],
                    },
                )
            )
        reply = await model.ainvoke([SystemMessage(GENERATOR), HumanMessage("\n".join(parts))])
        draft = model_text(reply)[:MAX_ANSWER_CHARS]
        emit({"event": "draft", "iteration": iteration, "chars": len(draft)})
        return {"draft": draft, "iteration": iteration, "status": "running"}

    async def reflect(state: ReflectState) -> dict[str, Any]:
        content = data_block("task", state["task"]) + "\n" + data_block("draft", state["draft"])
        reply = await reviewer.ainvoke([SystemMessage(REFLECTOR), HumanMessage(content)])
        critique = parse_critique(model_text(reply))
        round_: Round = {
            "iteration": state["iteration"],
            "draft": state["draft"],
            "score": critique.score,
            "scores": dict(critique.scores),
            "issues": list(critique.issues),
            "suggestions": list(critique.suggestions),
        }
        emit({"event": "critique", "iteration": state["iteration"], "score": critique.score})
        return {"rounds": [round_]}

    def route(state: ReflectState) -> str:
        last = state["rounds"][-1]
        if last["score"] >= limits.threshold:
            return "finish"
        if state["iteration"] >= limits.max_iterations:
            return "finish"
        return "generate"

    def finish(state: ReflectState) -> dict[str, Any]:
        best = best_round(state["rounds"])
        status = "passed" if best["score"] >= limits.threshold else "max_iterations"
        emit({"event": "finish", "status": status, "best_iteration": best["iteration"]})
        return {
            "status": status,
            "answer": best["draft"],
            "score": best["score"],
            "best_iteration": best["iteration"],
        }

    graph = StateGraph(ReflectState)
    graph.add_node("generate", generate)
    graph.add_node("reflect", reflect)
    graph.add_node("finish", finish)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges("reflect", route, ["generate", "finish"])
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer, name="reflect")
