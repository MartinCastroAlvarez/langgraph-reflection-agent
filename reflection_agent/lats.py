"""Variant 3, LATS-lite (Zhou et al., 2023): tree search over reflected candidates.

Each round **selects** a leaf by UCT from the root, **expands** it with ``candidates``
new answers sampled in parallel (each sees the path of answers and critiques that led
to it), **reflects** on every child with the rubric critic (reward = score / 10), and
**backpropagates** the reward to all ancestors (``visits += 1``, ``value += reward``).
The search stops when a child reaches ``threshold`` (``status="passed"``), after
``max_expansions`` rounds, or when every leaf is at ``max_depth``. The answer is the
node with the highest reflection score in the whole tree (ties: higher mean value).

*Lite*: rewards come only from the critic (no environment rollouts or simulation
step), and there is no value model beyond the backed-up means.
"""

from __future__ import annotations

import asyncio
import copy
import math
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

from reflection_agent.config import Limits
from reflection_agent.config import Settings
from reflection_agent.progress import emit
from reflection_agent.prompts import EXPANDER
from reflection_agent.prompts import REFLECTOR
from reflection_agent.prompts import data_block
from reflection_agent.rubric import model_text
from reflection_agent.rubric import parse_critique

__all__ = [
    "LatsState",
    "TreeNode",
    "backpropagate",
    "best_node",
    "build_lats",
    "expandable",
    "select",
    "uct",
]

MAX_ANSWER_CHARS = 6000


class TreeNode(TypedDict):
    id: int
    parent: int | None
    depth: int
    answer: str
    score: float | None
    """The critic's 0-10 score (``None`` for the root and before reflection)."""
    issues: list[str]
    suggestions: list[str]
    visits: int
    value: float
    """Sum of backed-up rewards (0-1 each); the mean is ``value / visits``."""


class LatsState(TypedDict, total=False):
    task: str
    tree: list[TreeNode]
    frontier: list[int]
    """The children created by the latest expansion, awaiting reflection."""
    expansions: int
    rounds: list[dict[str, Any]]
    """Per expansion: the expanded node, the children's scores, the best score so far."""
    status: str
    """running | passed | max_expansions | exhausted"""
    answer: str
    score: float
    best_id: int


def _root() -> TreeNode:
    return {
        "id": 0,
        "parent": None,
        "depth": 0,
        "answer": "",
        "score": None,
        "issues": [],
        "suggestions": [],
        "visits": 0,
        "value": 0.0,
    }


def _children(tree: list[TreeNode], node_id: int) -> list[TreeNode]:
    return [n for n in tree if n["parent"] == node_id]


def expandable(tree: list[TreeNode], node: TreeNode, max_depth: int) -> bool:
    """A node can still grow: below ``max_depth`` and unexpanded, or with such a descendant."""
    if node["depth"] >= max_depth:
        return False
    kids = _children(tree, node["id"])
    return not kids or any(expandable(tree, k, max_depth) for k in kids)


def uct(node: TreeNode, parent_visits: int, exploration: float) -> float:
    """Upper confidence bound for trees: mean value + exploration bonus."""
    if node["visits"] == 0:
        return math.inf
    mean = node["value"] / node["visits"]
    return mean + exploration * math.sqrt(math.log(max(parent_visits, 1)) / node["visits"])


def select(tree: list[TreeNode], limits: Limits) -> TreeNode | None:
    """Descend from the root by UCT to an unexpanded node; ``None`` when nothing can grow."""
    node = tree[0]
    if not expandable(tree, node, limits.max_depth):
        return None
    while True:
        kids = [k for k in _children(tree, node["id"]) if expandable(tree, k, limits.max_depth)]
        if not _children(tree, node["id"]):
            return node
        node = max(kids, key=lambda k: (uct(k, node["visits"], limits.exploration), -k["id"]))


def backpropagate(tree: list[TreeNode], node_id: int, reward: float) -> None:
    """Add ``reward`` to the node and every ancestor, in place."""
    by_id = {n["id"]: n for n in tree}
    current: int | None = node_id
    while current is not None:
        node = by_id[current]
        node["visits"] += 1
        node["value"] = round(node["value"] + reward, 6)
        current = node["parent"]


def best_node(tree: list[TreeNode]) -> TreeNode:
    """The reflected node with the highest score; ties go to the higher mean value."""
    scored = [n for n in tree if n["score"] is not None]
    return max(
        scored,
        key=lambda n: (n["score"], n["value"] / max(n["visits"], 1), -n["id"]),
    )


def _path(tree: list[TreeNode], node: TreeNode) -> list[dict[str, Any]]:
    by_id = {n["id"]: n for n in tree}
    path: list[dict[str, Any]] = []
    current: TreeNode | None = node
    while current is not None and current["parent"] is not None:
        path.append(
            {
                "answer": current["answer"],
                "score": current["score"],
                "issues": current["issues"],
                "suggestions": current["suggestions"],
            }
        )
        parent = current["parent"]
        current = by_id.get(parent) if parent is not None else None
    return list(reversed(path))


def _copy(tree: list[TreeNode]) -> list[TreeNode]:
    """Nodes return new state, never mutate the old one (checkpoints keep each version)."""
    return copy.deepcopy(tree)


def build_lats(
    model: BaseChatModel,
    settings: Settings,
    *,
    critic: BaseChatModel | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile LATS-lite. ``model`` samples candidates; ``critic`` (default ``model``) scores."""
    limits = settings.limits
    reviewer = critic or model

    async def expand(state: LatsState) -> dict[str, Any]:
        tree = _copy(state.get("tree") or [_root()])
        leaf = select(tree, limits)
        if leaf is None:  # pragma: no cover - route() never sends an exhausted tree here
            return {"tree": tree, "frontier": []}
        depth = leaf["depth"] + 1
        base = [
            data_block("task", state["task"]),
            data_block("depth", str(depth)),
            data_block("path", _path(tree, leaf)),
        ]

        async def sample(k: int) -> str:
            content = "\n".join([*base, data_block("candidate", str(k))])
            reply = await model.ainvoke([SystemMessage(EXPANDER), HumanMessage(content)])
            return model_text(reply)[:MAX_ANSWER_CHARS]

        answers = await asyncio.gather(*(sample(k) for k in range(1, limits.candidates + 1)))
        next_id = max(n["id"] for n in tree) + 1
        frontier = []
        for offset, answer in enumerate(answers):
            node: TreeNode = {
                **_root(),
                "id": next_id + offset,
                "parent": leaf["id"],
                "depth": depth,
                "answer": answer,
            }
            tree.append(node)
            frontier.append(node["id"])
        expansions = state.get("expansions", 0) + 1
        emit({"event": "expand", "round": expansions, "node": leaf["id"], "depth": depth})
        return {"tree": tree, "frontier": frontier, "expansions": expansions, "status": "running"}

    async def reflect(state: LatsState) -> dict[str, Any]:
        tree = _copy(state["tree"])
        by_id = {n["id"]: n for n in tree}

        async def score(node_id: int) -> tuple[int, Any]:
            content = data_block("task", state["task"]) + "\n"
            content += data_block("draft", by_id[node_id]["answer"])
            reply = await reviewer.ainvoke([SystemMessage(REFLECTOR), HumanMessage(content)])
            return node_id, parse_critique(model_text(reply))

        results = await asyncio.gather(*(score(i) for i in state["frontier"]))
        for node_id, critique in results:
            node = by_id[node_id]
            node["score"] = critique.score
            node["issues"] = list(critique.issues)
            node["suggestions"] = list(critique.suggestions)
            backpropagate(tree, node_id, critique.score / 10)
        children = {str(i): by_id[i]["score"] for i in state["frontier"]}
        parent = by_id[state["frontier"][0]]["parent"] if state["frontier"] else None
        round_ = {
            "round": state.get("expansions", 0),
            "expanded": parent,
            "depth": by_id[state["frontier"][0]]["depth"] if state["frontier"] else 0,
            "children": children,
            "best_so_far": best_node(tree)["score"],
        }
        emit({"event": "reflect", "round": round_["round"], "scores": children})
        return {"tree": tree, "rounds": [*state.get("rounds", []), round_]}

    def route(state: LatsState) -> str:
        tree = state["tree"]
        if any((tree[i]["score"] or 0) >= limits.threshold for i in state["frontier"]):
            return "finish"
        if state.get("expansions", 0) >= limits.max_expansions:
            return "finish"
        if select(tree, limits) is None:
            return "finish"
        return "expand"

    def finish(state: LatsState) -> dict[str, Any]:
        tree = state["tree"]
        best = best_node(tree)
        best_score = best["score"] or 0.0
        if best_score >= limits.threshold:
            status = "passed"
        elif state.get("expansions", 0) >= limits.max_expansions:
            status = "max_expansions"
        else:
            status = "exhausted"
        emit({"event": "finish", "status": status, "best_id": best["id"]})
        return {
            "status": status,
            "answer": best["answer"],
            "score": best_score,
            "best_id": best["id"],
        }

    graph = StateGraph(LatsState)
    graph.add_node("expand", expand)
    graph.add_node("reflect", reflect)
    graph.add_node("finish", finish)
    graph.add_edge(START, "expand")
    graph.add_edge("expand", "reflect")
    graph.add_conditional_edges("reflect", route, ["expand", "finish"])
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer, name="lats")
