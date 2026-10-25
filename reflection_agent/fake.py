"""A deterministic, role-aware fake chat model whose answers improve on a script.

``FakeReflectionModel`` is a real LangChain ``BaseChatModel``, so the graphs call it
exactly as they call ``ChatOllama``. It reads the ``Role: <name>.`` line of the system
prompt and answers like that node would, following a scenario that matches the task:

* **generator** (basic reflection) - the scenario's draft for this iteration;
* **reflector** (all variants) - the scripted rubric scores of the draft it is shown, so
  later drafts score higher;
* **expander** (LATS) - the scenario's candidate for this depth and sibling index;
* **actor** / **revisor** (Reflexion) - code whose quality depends on **how many lessons
  are in the prompt**: with no lessons the first attempt has the scripted bug, and only
  the lessons written by **self_reflection** fix it. Drop the lessons and it never
  improves, which is what the tests use to prove that memory is carried.

Unknown tasks get a generic scenario (``Draft 2: <task>``, scores rising by 1.5).
``script={"role": [reply, ...]}`` overrides any role with canned replies (strings,
``AIMessage``s, or exceptions to raise), consumed in order. Every call is logged with
its prompt, and sleeps ``latency`` seconds.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from pydantic import Field
from pydantic import PrivateAttr

from reflection_agent.prompts import parse_blocks
from reflection_agent.prompts import role_of

__all__ = [
    "CODE_SCENARIOS",
    "TEXT_SCENARIOS",
    "CodeAttempt",
    "CodeScenario",
    "FakeReflectionModel",
    "TextScenario",
]


@dataclass(frozen=True)
class TextScenario:
    """Drafts of rising quality and the rubric scores the fake critic gives each one."""

    name: str
    keywords: tuple[str, ...]
    drafts: tuple[str, ...]
    scores: tuple[tuple[float, float, float, float], ...]
    """(accuracy, completeness, clarity, concision) per draft."""
    issues: tuple[str, ...]
    """The critic's main complaint per draft."""
    reflect_path: tuple[int, ...]
    """Basic reflection: the draft index written at iteration 1, 2, ..."""
    lats_children: tuple[tuple[int, ...], ...]
    """LATS: draft indices of candidates 1..N at depth 1, 2, ..."""
    queries: tuple[str, ...] = ()
    references: tuple[str, ...] = ()

    def matches(self, task: str) -> bool:
        lowered = task.lower()
        return all(k in lowered for k in self.keywords)


@dataclass(frozen=True)
class CodeAttempt:
    revised: str
    draft: str = ""
    """The actor's first answer before the tools ran (defaults to ``revised``)."""
    missing: str = ""


@dataclass(frozen=True)
class CodeScenario:
    """Reflexion on a coding task: attempt *n* is what the actor writes after n-1 lessons."""

    name: str
    keywords: tuple[str, ...]
    attempts: tuple[CodeAttempt, ...]
    lessons: tuple[str, ...]
    queries: tuple[str, ...]
    calculations: tuple[str, ...]
    references: tuple[str, ...]

    def matches(self, task: str) -> bool:
        lowered = task.lower()
        return all(k in lowered for k in self.keywords)


TEXT_SCENARIOS: tuple[TextScenario, ...] = (
    TextScenario(
        name="idempotency",
        keywords=("idempotency",),
        drafts=(
            "An idempotency key makes requests safe.",
            "An idempotency key is a unique value the client sends with a payment request. "
            "If the request is retried, the server sees the same key and does not charge "
            "twice.",
            "An idempotency key is a unique value the client sends with each payment "
            "request, usually in a header. The server stores the key with its first "
            "response and returns that same response to any retry, so a timeout followed "
            "by a retry never charges the customer twice. Keys expire after a retention "
            "window, and a retry that reuses a key with a different body is rejected.",
            "A client attaches a unique idempotency key to every payment request. The "
            "server saves the first response under that key and replays it for every "
            "retry, so a retried charge is never executed twice; a reused key with a "
            "different body is a conflict, and keys expire after about a day.",
            "Send a UUID header so the payment endpoint can deduplicate retries.",
        ),
        scores=((6, 3, 4, 9), (8, 5, 7, 9), (9, 9, 8, 7), (9, 8, 9, 9), (7, 4, 6, 10)),
        issues=(
            "It never says what the key is or what the server does with it.",
            "It doesn't say the server stores and replays the first response, or when keys expire.",
            "The third sentence crams two ideas together; it is a little long.",
            "Could name the header, but nothing important is missing.",
            "It names a mechanism (a UUID header) without saying why it stops double charges.",
        ),
        reflect_path=(0, 1, 2),
        lats_children=((0, 4, 1), (2, 3, 4), (3, 2, 1)),
        queries=("idempotency key retries payments",),
        references=("idempotency",),
    ),
    TextScenario(
        name="vary",
        keywords=("vary",),
        drafts=(
            "Vary tells caches which headers matter.",
            "The Vary header lists request headers that change the response, like "
            "Accept-Language. Caches use it to store separate copies.",
            "The Vary response header lists the request headers, such as Origin or "
            "Accept-Language, that change what the server returns. A shared cache keys its "
            "stored copies on those headers too, so without Vary it could hand one "
            "client's response, for example a CORS answer meant for another site, to the "
            "wrong client.",
            "The Vary header lists the request headers that affect the response, so caches "
            "keep one copy per header value.",
        ),
        scores=((6, 3, 3, 10), (8, 6, 6, 9), (9, 8, 8, 7), (8, 7, 7, 9)),
        issues=(
            "Too vague for a junior developer: which headers, and what goes wrong without it?",
            "It doesn't explain the failure mode Vary prevents.",
            "Slightly over two sentences' worth of content, but accurate and concrete.",
            "Correct, but there is no example of what breaks without it.",
        ),
        reflect_path=(0, 1, 3, 2),
        lats_children=((0, 3, 1), (1, 2, 3), (2, 3, 1)),
        queries=("Vary header shared cache",),
        references=("caching",),
    ),
)

_LUHN_LEFT = """def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number.replace(" ", "")]
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
"""
_LUHN_RIGHT = _LUHN_LEFT.replace("enumerate(digits)", "enumerate(reversed(digits))")
_LUHN_OK = """def luhn_valid(number: str) -> bool:
    text = number.replace(" ", "")
    if len(text) < 2 or not text.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(text)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
"""
_ROMAN_SUM = """VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(numeral: str) -> int:
    return sum(VALUES[ch] for ch in numeral)
"""
_ROMAN_SUB = """VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(numeral: str) -> int:
    total = 0
    for i, ch in enumerate(numeral):
        value = VALUES[ch]
        if i + 1 < len(numeral) and value < VALUES[numeral[i + 1]]:
            total -= value
        else:
            total += value
    return total
"""
_ROMAN_OK = _ROMAN_SUB.replace(
    "    total = 0\n", "    numeral = numeral.strip().upper()\n    total = 0\n", 1
)
_RLE_OK = """import re


def rle_encode(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == text[i]:
            j += 1
        out.append(f"{j - i}{text[i]}")
        i = j
    return "".join(out)


def rle_decode(code: str) -> str:
    return "".join(ch * int(n) for n, ch in re.findall(r"(\\d+)(\\D)", code))
"""

CODE_SCENARIOS: tuple[CodeScenario, ...] = (
    CodeScenario(
        name="luhn",
        keywords=("luhn_valid",),
        attempts=(
            CodeAttempt(
                draft=_LUHN_LEFT,
                revised=_LUHN_RIGHT,
                missing="I'm not sure which end the doubling starts from.",
            ),
            CodeAttempt(revised=_LUHN_OK),
        ),
        lessons=(
            "Lesson: validate the input before the checksum: remove spaces, then return "
            "False for any other non-digit character and for fewer than two digits.",
        ),
        queries=("Luhn double every second digit", "Luhn input edge cases"),
        calculations=("7 * 2 - 9",),
        references=("luhn",),
    ),
    CodeScenario(
        name="roman",
        keywords=("roman_to_int",),
        attempts=(
            CodeAttempt(revised=_ROMAN_SUM, missing="Subtractive pairs such as IV."),
            CodeAttempt(revised=_ROMAN_SUB, missing="Input normalisation."),
            CodeAttempt(revised=_ROMAN_OK),
        ),
        lessons=(
            "Lesson: apply the subtractive rule: when a symbol is worth less than the next "
            "one, subtract it instead of adding it.",
            "Lesson: normalise the input with upper() before looking symbols up, so "
            "lowercase numerals work.",
        ),
        queries=("Roman numeral subtractive notation",),
        calculations=("1000 + 900 + 90 + 4",),
        references=("roman-numerals",),
    ),
    CodeScenario(
        name="rle",
        keywords=("rle_encode",),
        attempts=(CodeAttempt(revised=_RLE_OK),),
        lessons=(),
        queries=("run-length encoding example",),
        calculations=("3 + 1 + 2",),
        references=("run-length-encoding",),
    ),
)

_GENERIC_DRAFT = re.compile(r"^Draft (\d+): ")
_GENERIC_CANDIDATE = re.compile(r"^Candidate (\d+)\.(\d+): ")
_SIBLING_BONUS = (0.0, 1.0, -0.5)


def _fence(code: str) -> str:
    return f"```python\n{code}```"


def _critique(scores: Sequence[float], issue: str, suggestion: str) -> str:
    names = ("accuracy", "completeness", "clarity", "concision")
    return json.dumps(
        {
            "scores": dict(zip(names, scores, strict=True)),
            "issues": [issue] if issue else [],
            "suggestions": [suggestion] if suggestion else [],
        }
    )


def _int(data: dict[str, str], key: str, default: int) -> int:
    try:
        return int(data.get(key, default))
    except ValueError:
        return default


def _lessons(data: dict[str, str]) -> list[str]:
    try:
        value = json.loads(data.get("lessons", "[]"))
    except json.JSONDecodeError:
        return []
    return [str(x) for x in value] if isinstance(value, list) else []


class FakeReflectionModel(BaseChatModel):
    """Role-aware, deterministic fake; see the module docstring."""

    latency: float = 0.0
    script: dict[str, list[Any]] = Field(default_factory=dict)
    text_scenarios: tuple[TextScenario, ...] = TEXT_SCENARIOS
    code_scenarios: tuple[CodeScenario, ...] = CODE_SCENARIOS

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _log: list[tuple[str, str]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-reflection"

    @property
    def calls(self) -> list[str]:
        """The role of every call, in start order."""
        with self._lock:
            return [role for role, _ in self._log]

    def prompts(self, role: str) -> list[str]:
        """The human-message text of every call made with ``role``."""
        with self._lock:
            return [text for r, text in self._log if r == role]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        role, prompt = self._enter(messages)
        time.sleep(self.latency)
        return self._result(role, prompt)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        role, prompt = self._enter(messages)
        await asyncio.sleep(self.latency)
        return self._result(role, prompt)

    def _enter(self, messages: list[BaseMessage]) -> tuple[str, str]:
        system = next((m.text for m in messages if isinstance(m, SystemMessage)), "")
        prompt = "\n".join(m.text for m in messages if isinstance(m, HumanMessage))
        role = role_of(system)
        with self._lock:
            self._log.append((role, prompt))
        return role, prompt

    def _result(self, role: str, prompt: str) -> ChatResult:
        with self._lock:
            queue = self.script.get(role)
            canned = queue.pop(0) if queue else None
        if isinstance(canned, BaseException):
            raise canned
        if canned is not None:
            message = canned if isinstance(canned, AIMessage) else AIMessage(content=str(canned))
        else:
            message = AIMessage(content=self._respond(role, parse_blocks(prompt)))
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ------------------------------------------------------------------ scenarios
    def _text(self, task: str) -> TextScenario | None:
        return next((s for s in self.text_scenarios if s.matches(task)), None)

    def _code(self, task: str) -> CodeScenario | None:
        return next((s for s in self.code_scenarios if s.matches(task)), None)

    def _respond(self, role: str, data: dict[str, str]) -> str:
        task = data.get("task", "")
        if role == "generator":
            n = _int(data, "iteration", 1)
            text = self._text(task)
            if text is None:
                return f"Draft {n}: {task}"
            return text.drafts[text.reflect_path[min(n, len(text.reflect_path)) - 1]]
        if role == "expander":
            depth, k = _int(data, "depth", 1), _int(data, "candidate", 1)
            text = self._text(task)
            if text is None:
                return f"Candidate {depth}.{k}: {task}"
            row = text.lats_children[min(depth, len(text.lats_children)) - 1]
            return text.drafts[row[(k - 1) % len(row)]]
        if role == "reflector":
            return self._reflect(task, data.get("draft", ""))
        if role in {"actor", "revisor"}:
            return self._act(role, task, data)
        if role == "self_reflection":
            return self._lesson(task, data)
        return "ok"

    def _reflect(self, task: str, draft: str) -> str:
        text = self._text(task)
        if text is not None and draft.strip() in text.drafts:
            i = text.drafts.index(draft.strip())
            return _critique(text.scores[i], text.issues[i], "Address: " + text.issues[i])
        if m := _GENERIC_DRAFT.match(draft):
            score = min(10.0, 4.0 + 1.5 * (int(m[1]) - 1))
            return _critique([score] * 4, f"Draft {m[1]} is still generic.", "Be concrete.")
        if m := _GENERIC_CANDIDATE.match(draft):
            depth, k = int(m[1]), int(m[2])
            score = 4.0 + 1.5 * (depth - 1) + _SIBLING_BONUS[(k - 1) % len(_SIBLING_BONUS)]
            score = min(10.0, max(0.0, score))
            return _critique([score] * 4, f"Candidate {depth}.{k} lacks detail.", "Add detail.")
        return _critique([5.0] * 4, "An answer the fake critic has no script for.", "")

    def _act(self, role: str, task: str, data: dict[str, str]) -> str:
        n_lessons = len(_lessons(data))
        code = self._code(task)
        if code is not None:
            attempt = code.attempts[min(n_lessons, len(code.attempts) - 1)]
            if role == "actor":
                return json.dumps(
                    {
                        "answer": _fence(attempt.draft or attempt.revised),
                        "critique": {"missing": attempt.missing, "superfluous": ""},
                        "search_queries": list(code.queries),
                        "calculations": list(code.calculations),
                    }
                )
            return json.dumps(
                {"answer": _fence(attempt.revised), "references": list(code.references)}
            )
        text = self._text(task)
        if text is not None:
            path = text.reflect_path
            answer = text.drafts[path[min(n_lessons, len(path) - 1)]]
            queries, references = list(text.queries), list(text.references)
        else:
            answer, queries, references = f"Draft {n_lessons + 1}: {task}", [task[:80]], []
        if role == "actor":
            return json.dumps(
                {
                    "answer": answer,
                    "critique": {"missing": "", "superfluous": ""},
                    "search_queries": queries,
                    "calculations": [],
                }
            )
        return json.dumps({"answer": answer, "references": references})

    def _lesson(self, task: str, data: dict[str, str]) -> str:
        n_lessons = len(_lessons(data))
        code = self._code(task)
        if code is not None and n_lessons < len(code.lessons):
            return "The tests show the bug. " + code.lessons[n_lessons]
        feedback = " ".join(data.get("feedback", "").split())[:160]
        return f"Lesson {n_lessons + 1}: fix what the evaluator reported: {feedback}"
