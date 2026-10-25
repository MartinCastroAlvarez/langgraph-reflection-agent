"""One entry point for the three variants."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from reflection_agent.config import VARIANTS
from reflection_agent.config import Settings
from reflection_agent.corpus import Corpus
from reflection_agent.evaluators import CriticEvaluator
from reflection_agent.evaluators import Evaluator
from reflection_agent.evaluators import UnitTestEvaluator
from reflection_agent.lats import build_lats
from reflection_agent.llm import make_model
from reflection_agent.reflect import build_reflect
from reflection_agent.reflexion import build_reflexion
from reflection_agent.sandbox import Sandbox
from reflection_agent.tasks import CodeTask
from reflection_agent.tools import Toolbox

__all__ = ["LATS_TEMPERATURE", "build_graph", "make_evaluator", "make_toolbox"]

LATS_TEMPERATURE = 0.8
"""ollama/Anthropic sampling temperature for LATS candidates, so siblings differ."""


def make_toolbox(settings: Settings) -> Toolbox:
    return Toolbox(corpus=Corpus.from_dir(settings.corpus_dir))


def make_evaluator(
    settings: Settings, critic: BaseChatModel, code_task: CodeTask | None = None
) -> Evaluator:
    """Unit tests in the sandbox for a code task, else the rubric critic."""
    if code_task is not None:
        return UnitTestEvaluator(code_task, Sandbox(timeout=settings.sandbox_timeout))
    return CriticEvaluator(critic, threshold=settings.limits.threshold)


def build_graph(
    variant: str,
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    json_model: BaseChatModel | None = None,
    code_task: CodeTask | None = None,
    evaluator: Evaluator | None = None,
    toolbox: Toolbox | None = None,
    store: BaseStore | None = None,
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile ``reflect``, ``reflexion`` or ``lats``.

    Without ``model`` the provider in ``settings`` is used, and the critic, actor and
    revisor get ollama's JSON mode. A given ``model`` serves every role. ``code_task``
    (Reflexion) switches the evaluator to its unit tests in the sandbox.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    sampler = model
    if model is None:
        model = make_model(settings)
        json_model = json_model or make_model(settings, json_mode=True)
        sampler = make_model(settings, temperature=max(settings.temperature, LATS_TEMPERATURE))
    critic = json_model or model
    if variant == "reflect":
        return build_reflect(model, settings, critic=critic, checkpointer=checkpointer)
    if variant == "lats":
        return build_lats(sampler or model, settings, critic=critic, checkpointer=checkpointer)
    return build_reflexion(
        model,
        settings,
        evaluator=evaluator or make_evaluator(settings, critic, code_task),
        toolbox=toolbox or make_toolbox(settings),
        json_model=critic,
        store=store,
        checkpointer=checkpointer,
    )
