"""The critic's structured rubric, and tolerant parsing of model JSON.

The reflector scores each criterion from 0 to 10; the overall score is the weighted mean,
computed here and never taken from the model, so a critic can't "decide" to pass a
draft by writing ``"overall": 10``. A reply that can't be parsed scores 0 with an issue
that says so: the loop keeps going, bounded by its limits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator

__all__ = [
    "RUBRIC",
    "Criterion",
    "Critique",
    "ParseError",
    "extract_json",
    "model_text",
    "parse_critique",
]

MAX_ITEMS = 8
MAX_ITEM_CHARS = 400
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


class ParseError(ValueError):
    """A model reply did not contain the JSON it was asked for."""


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str
    weight: float
    question: str


RUBRIC: tuple[Criterion, ...] = (
    Criterion("accuracy", 0.4, "Is every claim correct and does it answer the task asked?"),
    Criterion("completeness", 0.3, "Does it cover everything the task requires?"),
    Criterion("clarity", 0.2, "Is it easy to follow, with concrete examples where useful?"),
    Criterion("concision", 0.1, "Does it respect length limits and avoid filler?"),
)
_WEIGHTS = {c.name: c.weight for c in RUBRIC}


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return [str(v)[:MAX_ITEM_CHARS] for v in value[:MAX_ITEMS] if str(v).strip()]


class Critique(BaseModel):
    """One reflection: per-criterion scores, the issues found and how to fix them."""

    scores: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    @field_validator("scores", mode="before")
    @classmethod
    def _known_scores(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError("scores must be an object")
        out: dict[str, float] = {}
        for name in _WEIGHTS:
            raw = value.get(name)
            if raw is None:
                raise ValueError(f"missing score for {name!r}")
            out[name] = min(10.0, max(0.0, float(raw)))
        return out

    @field_validator("issues", "suggestions", mode="before")
    @classmethod
    def _strings(cls, value: Any) -> list[str]:
        return _items(value)

    @property
    def score(self) -> float:
        """The weighted mean over the rubric, 0-10, rounded to 2 decimals."""
        if not self.scores:
            return 0.0
        return round(sum(_WEIGHTS[k] * v for k, v in self.scores.items()), 2)

    def summary(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "scores": dict(self.scores),
            "issues": list(self.issues),
            "suggestions": list(self.suggestions),
        }


def model_text(message: BaseMessage) -> str:
    """The text of a model reply, without a ``<think>`` block (qwen3, deepseek-r1)."""
    text = str(message.text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> Any:
    """The first JSON object in ``text``: tolerates prose, code fences and ``<think>``."""
    cleaned = _THINK.sub("", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ParseError("no JSON object in the model output")


def parse_critique(text: str) -> Critique:
    """Parse a reflector reply; an unusable one becomes a zero score with a clear issue."""
    try:
        return Critique.model_validate(extract_json(text))
    except (ParseError, ValidationError) as exc:
        reason = str(exc).splitlines()[0][:200]
        return Critique(
            scores=dict.fromkeys(_WEIGHTS, 0.0),
            issues=[f"the critique could not be parsed ({reason})"],
        )
