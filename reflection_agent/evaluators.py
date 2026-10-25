"""External evaluators for Reflexion: unit tests in the sandbox, or the rubric critic.

Reflexion needs a signal that doesn't come from the actor itself. For coding tasks it's
the unit tests (:class:`UnitTestEvaluator`, run in the :class:`~reflection_agent.sandbox.
Sandbox`). For open-ended text tasks the fallback is the rubric critic
(:class:`CriticEvaluator`), a separate model call with a fixed rubric and a threshold.
That is weaker: a model grading itself can agree with its own mistakes.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from reflection_agent.prompts import REFLECTOR
from reflection_agent.prompts import data_block
from reflection_agent.rubric import model_text
from reflection_agent.rubric import parse_critique
from reflection_agent.sandbox import Sandbox
from reflection_agent.tasks import CodeTask

__all__ = ["CriticEvaluator", "Evaluation", "Evaluator", "UnitTestEvaluator", "extract_code"]

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(?P<code>.*?)```", re.S | re.I)


@dataclass(frozen=True, slots=True)
class Evaluation:
    passed: bool
    score: float
    """0-1: the fraction of tests passed, or the critic score / 10."""
    feedback: str
    details: dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    @property
    def kind(self) -> str: ...

    async def evaluate(self, task: str, answer: str) -> Evaluation: ...


def extract_code(answer: str) -> str:
    """The longest fenced code block in ``answer``, or the answer itself when there is none."""
    blocks = [m["code"] for m in _FENCE.finditer(answer)]
    return max(blocks, key=len).strip() + "\n" if blocks else answer.strip() + "\n"


@dataclass(frozen=True)
class UnitTestEvaluator:
    """Runs the task's hidden tests against the answer's code, in the sandbox only."""

    task: CodeTask
    sandbox: Sandbox = field(default_factory=Sandbox)
    kind: str = "unit-tests"

    async def evaluate(self, task: str, answer: str) -> Evaluation:
        code = extract_code(answer)
        result = await asyncio.to_thread(self.sandbox.run, code, list(self.task.tests))
        return Evaluation(
            passed=result.passed,
            score=round(result.score, 3),
            feedback=result.feedback(),
            details={
                "tests_passed": sum(t.ok for t in result.tests),
                "tests_total": len(self.task.tests),
                "blocked": list(result.blocked),
                "seconds": result.seconds,
            },
        )


@dataclass(frozen=True)
class CriticEvaluator:
    """The rubric critic as the evaluator, for tasks without tests."""

    model: BaseChatModel
    threshold: float = 8.0
    kind: str = "critic"

    async def evaluate(self, task: str, answer: str) -> Evaluation:
        reply = await self.model.ainvoke(
            [
                SystemMessage(REFLECTOR),
                HumanMessage(data_block("task", task) + "\n" + data_block("draft", answer)),
            ]
        )
        critique = parse_critique(model_text(reply))
        issues = "; ".join(critique.issues) or "no issues listed"
        return Evaluation(
            passed=critique.score >= self.threshold,
            score=round(critique.score / 10, 3),
            feedback=f"critic score {critique.score:g}/10 (threshold {self.threshold:g}): {issues}",
            details={"critique": critique.summary()},
        )
