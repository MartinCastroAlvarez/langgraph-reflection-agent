"""Variant 2, Reflexion (Shinn et al., 2023): act, ground, evaluate, reflect, remember.

One *attempt* (trial) is: the **actor** answers with a self-critique, search queries
and calculations; the **tools** node runs them locally (corpus search, calculator); the
**revisor** rewrites the answer with that evidence and cites document ids; an external
**evaluator** judges it (unit tests in the sandbox, or the rubric critic). On a failure
**self_reflection** turns the evaluator's feedback into a verbal lesson, which goes into
episodic memory, and the next attempt starts with every lesson so far. With a LangGraph
store, lessons also outlive the run (see :mod:`~reflection_agent.memory`).

The loop ends on a pass (``status="passed"``) or after ``max_attempts`` attempts
(``status="max_attempts"``); the answer is the passing attempt, else the best one.
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
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from reflection_agent.config import Settings
from reflection_agent.evaluators import Evaluator
from reflection_agent.memory import MAX_LESSON_CHARS
from reflection_agent.memory import LessonMemory
from reflection_agent.memory import clean
from reflection_agent.memory import task_key
from reflection_agent.progress import emit
from reflection_agent.prompts import ACTOR
from reflection_agent.prompts import REVISOR
from reflection_agent.prompts import SELF_REFLECTION
from reflection_agent.prompts import data_block
from reflection_agent.rubric import ParseError
from reflection_agent.rubric import extract_json
from reflection_agent.rubric import model_text
from reflection_agent.tools import MAX_INPUT_CHARS
from reflection_agent.tools import Evidence
from reflection_agent.tools import Toolbox

__all__ = ["ReflexionState", "Trial", "build_reflexion", "parse_draft", "parse_revision"]

MAX_ANSWER_CHARS = 8000


class Trial(TypedDict):
    attempt: int
    answer: str
    references: list[str]
    passed: bool
    score: float
    feedback: str
    lessons_seen: int
    details: dict[str, Any]


class ReflexionState(TypedDict, total=False):
    task: str
    task_key: str
    """Episodic-memory key (a safe name, or a hash of the task by default)."""
    attempt: int
    lessons: list[str]
    recalled: int
    """How many lessons came from the store at the start of the run."""
    draft: dict[str, Any]
    evidence: list[Evidence]
    answer: str
    references: list[str]
    tool_calls: Annotated[int, operator.add]
    trials: Annotated[list[Trial], operator.add]
    status: str
    """running | passed | max_attempts"""
    score: float
    best_attempt: int


def _strings(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip()[:MAX_INPUT_CHARS] for v in value if str(v).strip()][:limit]


def parse_draft(text: str, *, max_queries: int) -> dict[str, Any]:
    """The actor's JSON; a reply without JSON becomes an answer with no tool requests."""
    try:
        data = extract_json(text)
    except ParseError:
        return {"answer": text[:MAX_ANSWER_CHARS], "critique": {}, "queries": [], "calcs": []}
    critique = data.get("critique")
    return {
        "answer": str(data.get("answer", ""))[:MAX_ANSWER_CHARS],
        "critique": {
            k: str(v)[:500]
            for k, v in (critique.items() if isinstance(critique, dict) else [])
            if k in {"missing", "superfluous"}
        },
        "queries": _strings(data.get("search_queries"), max_queries),
        "calcs": _strings(data.get("calculations"), max_queries),
    }


def parse_revision(text: str, *, draft: str, cited: set[str]) -> tuple[str, list[str]]:
    """The revisor's answer and its references, keeping only ids that were retrieved."""
    try:
        data = extract_json(text)
    except ParseError:
        return (text or draft)[:MAX_ANSWER_CHARS], []
    answer = str(data.get("answer") or draft)[:MAX_ANSWER_CHARS]
    refs = data.get("references")
    kept = [str(r) for r in refs if str(r) in cited] if isinstance(refs, list) else []
    return answer, list(dict.fromkeys(kept))


def _lesson(text: str) -> str:
    body = text[text.find("Lesson") :] if "Lesson" in text else text
    return " ".join(body.split())[:MAX_LESSON_CHARS]


def build_reflexion(
    model: BaseChatModel,
    settings: Settings,
    *,
    evaluator: Evaluator,
    toolbox: Toolbox,
    json_model: BaseChatModel | None = None,
    store: BaseStore | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile Reflexion. ``store`` makes lessons persist across runs of the same task."""
    limits = settings.limits
    structured = json_model or model
    memory = LessonMemory(store)

    async def recall(state: ReflexionState) -> dict[str, Any]:
        key = task_key(state["task"], state.get("task_key"))
        stored = await memory.recall(key)
        lessons = clean(stored + list(state.get("lessons", [])))
        emit({"event": "recall", "key": key, "lessons": len(stored)})
        return {
            "task_key": key,
            "lessons": lessons,
            "recalled": len(stored),
            "attempt": 0,
            "status": "running",
        }

    async def actor(state: ReflexionState) -> dict[str, Any]:
        attempt = state.get("attempt", 0) + 1
        parts = [
            data_block("task", state["task"]),
            data_block("attempt", str(attempt)),
            data_block("lessons", state.get("lessons", [])),
        ]
        trials = state.get("trials", [])
        if trials:
            parts.append(
                data_block(
                    "previous_attempt",
                    {"answer": trials[-1]["answer"], "feedback": trials[-1]["feedback"]},
                )
            )
        reply = await structured.ainvoke([SystemMessage(ACTOR), HumanMessage("\n".join(parts))])
        draft = parse_draft(model_text(reply), max_queries=limits.max_queries)
        emit({"event": "draft", "attempt": attempt, "queries": len(draft["queries"])})
        return {"attempt": attempt, "draft": draft}

    def tools(state: ReflexionState) -> dict[str, Any]:
        draft = state["draft"]
        evidence, calls = toolbox.run(draft["queries"], draft["calcs"], limit=limits.max_queries)
        emit({"event": "tools", "calls": calls, "evidence": len(evidence)})
        return {"evidence": evidence, "tool_calls": calls}

    async def revisor(state: ReflexionState) -> dict[str, Any]:
        draft, evidence = state["draft"], state.get("evidence", [])
        content = "\n".join(
            [
                data_block("task", state["task"]),
                data_block("draft", {"answer": draft["answer"], "critique": draft["critique"]}),
                data_block("evidence", evidence),
                data_block("lessons", state.get("lessons", [])),
            ]
        )
        reply = await structured.ainvoke([SystemMessage(REVISOR), HumanMessage(content)])
        cited = {e["id"] for e in evidence if not e["id"].startswith("calc-")}
        answer, references = parse_revision(model_text(reply), draft=draft["answer"], cited=cited)
        emit({"event": "revise", "attempt": state["attempt"], "references": references})
        return {"answer": answer, "references": references}

    async def evaluate(state: ReflexionState) -> dict[str, Any]:
        result = await evaluator.evaluate(state["task"], state["answer"])
        trial: Trial = {
            "attempt": state["attempt"],
            "answer": state["answer"],
            "references": list(state.get("references", [])),
            "passed": result.passed,
            "score": result.score,
            "feedback": result.feedback,
            "lessons_seen": len(state.get("lessons", [])),
            "details": dict(result.details),
        }
        emit(
            {
                "event": "evaluate",
                "attempt": state["attempt"],
                "passed": result.passed,
                "score": result.score,
            }
        )
        return {"trials": [trial]}

    def route(state: ReflexionState) -> str:
        last = state["trials"][-1]
        if last["passed"] or state["attempt"] >= limits.max_attempts:
            return "finish"
        return "reflect"

    async def reflect(state: ReflexionState) -> dict[str, Any]:
        last = state["trials"][-1]
        content = "\n".join(
            [
                data_block("task", state["task"]),
                data_block("answer", last["answer"]),
                data_block("feedback", last["feedback"]),
                data_block("lessons", state.get("lessons", [])),
            ]
        )
        reply = await model.ainvoke([SystemMessage(SELF_REFLECTION), HumanMessage(content)])
        lesson = _lesson(model_text(reply))
        lessons = clean([*state.get("lessons", []), lesson])
        await memory.remember(state["task_key"], lessons)
        emit({"event": "lesson", "attempt": state["attempt"], "lesson": lesson[:120]})
        return {"lessons": lessons}

    def finish(state: ReflexionState) -> dict[str, Any]:
        trials = state["trials"]
        passing = [t for t in trials if t["passed"]]
        best = passing[0] if passing else max(trials, key=lambda t: (t["score"], t["attempt"]))
        status = "passed" if passing else "max_attempts"
        emit({"event": "finish", "status": status, "best_attempt": best["attempt"]})
        return {
            "status": status,
            "answer": best["answer"],
            "references": best["references"],
            "score": best["score"],
            "best_attempt": best["attempt"],
        }

    graph = StateGraph(ReflexionState)
    graph.add_node("recall", recall)
    graph.add_node("actor", actor)
    graph.add_node("tools", tools)
    graph.add_node("revisor", revisor)
    graph.add_node("evaluate", evaluate)
    graph.add_node("reflect", reflect)
    graph.add_node("finish", finish)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "actor")
    graph.add_edge("actor", "tools")
    graph.add_edge("tools", "revisor")
    graph.add_edge("revisor", "evaluate")
    graph.add_conditional_edges("evaluate", route, ["reflect", "finish"])
    graph.add_edge("reflect", "actor")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer, store=store, name="reflexion")
