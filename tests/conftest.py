"""Shared helpers: fake-provider settings, the fake model, a graph runner."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from reflection_agent.config import Limits
from reflection_agent.config import Settings
from reflection_agent.fake import FakeReflectionModel

IDEMPOTENCY = "Explain idempotency keys for a payments API in at most three sentences."
VARY = "Explain the HTTP Vary header to a junior developer in two sentences."
GENERIC = "Write a haiku about flaky tests."


def make_settings(**limits: Any) -> Settings:
    return Settings(provider="fake", limits=Limits(**limits))


def run(graph: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = asyncio.run(graph.ainvoke(inputs, {"recursion_limit": 200}))
    return result


def critique(*scores: float, issue: str = "") -> str:
    """A reflector reply with four criterion scores (all equal when one is given)."""
    values = list(scores) * 4 if len(scores) == 1 else list(scores)
    names = ("accuracy", "completeness", "clarity", "concision")
    return json.dumps(
        {"scores": dict(zip(names, values, strict=True)), "issues": [issue] if issue else []}
    )


def fake(**script: list[Any]) -> FakeReflectionModel:
    return FakeReflectionModel(script=dict(script))
