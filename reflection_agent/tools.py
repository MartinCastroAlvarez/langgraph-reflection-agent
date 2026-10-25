"""The two local tools of the Reflexion actor: document search and a safe calculator.

* ``calculate`` walks the AST of an arithmetic expression. No ``eval``: only numbers,
  ``+ - * / // % **``, unary signs, ``pi``/``e`` and a few math functions, with caps on
  length, node count and exponent size.
* :class:`Toolbox` runs the actor's search queries against the local :class:`Corpus`
  and its calculations, and returns evidence with ids the revisor can cite. Neither
  tool touches the network or the filesystem.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing import TypedDict

from reflection_agent.corpus import Corpus

__all__ = ["MAX_INPUT_CHARS", "Evidence", "ToolError", "Toolbox", "calculate"]

MAX_INPUT_CHARS = 300


class ToolError(ValueError):
    """A tool could not handle its input. The message is safe to show to the model."""


# --------------------------------------------------------------------------- calculator
_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_MAX_NODES = 64
_MAX_EXPONENT = 64
_MAX_MAGNITUDE = 1e18
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _number(value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ToolError("only numbers are allowed")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("the result is not a finite number")
    if abs(value) > _MAX_MAGNITUDE:
        raise ToolError("the result is too large")
    return value


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return _number(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _number(_UNARY[type(node.op)](_evaluate(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > _MAX_EXPONENT or abs(left) > 1e6):
            raise ToolError("exponent or base too large")
        try:
            return _number(_BINARY[type(node.op)](left, right))
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
        except OverflowError:
            raise ToolError("the result is too large") from None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
        and 1 <= len(node.args) <= 4
    ):
        args = [_evaluate(a) for a in node.args]
        try:
            return _number(_FUNCTIONS[node.func.id](*args))
        except (ValueError, TypeError):
            raise ToolError(f"invalid arguments for {node.func.id}()") from None
    raise ToolError(f"unsupported expression element: {type(node).__name__}")


def _format(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}".rstrip("0").rstrip(".")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression safely. ``^`` means power, ``x`` and ``×`` multiply."""
    text = expression.strip()
    if not text or len(text) > MAX_INPUT_CHARS:
        raise ToolError(f"the expression must be 1-{MAX_INPUT_CHARS} characters")
    text = _THOUSANDS.sub("", text).replace("×", "*").replace("^", "**")
    text = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", text)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ToolError("not an arithmetic expression") from None
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise ToolError("the expression is too long")
    return _format(_evaluate(tree))


# --------------------------------------------------------------------------- toolbox
class Evidence(TypedDict):
    id: str
    """A document id (cite it) or ``calc-<n>`` for a calculation."""
    query: str
    text: str


@dataclass(frozen=True)
class Toolbox:
    corpus: Corpus
    hits_per_query: int = 2

    def run(
        self, queries: list[str], calculations: list[str], *, limit: int
    ) -> tuple[list[Evidence], int]:
        """Run up to ``limit`` queries + calculations; returns evidence and the calls made."""
        evidence: list[Evidence] = []
        seen: set[tuple[str, str]] = set()
        calls = 0
        for query in queries:
            if calls >= limit:
                break
            calls += 1
            text = query.strip()[:MAX_INPUT_CHARS]
            for fact in self.corpus.search(text, k=self.hits_per_query):
                if (fact.note, fact.text) not in seen:
                    seen.add((fact.note, fact.text))
                    evidence.append({"id": fact.note, "query": text, "text": fact.text})
        for n, expression in enumerate(calculations, start=1):
            if calls >= limit:
                break
            calls += 1
            text = expression.strip()[:MAX_INPUT_CHARS]
            try:
                result = f"{text} = {calculate(text)}"
            except ToolError as exc:
                result = f"error: {exc}"
            evidence.append({"id": f"calc-{n}", "query": text, "text": result})
        return evidence, calls
