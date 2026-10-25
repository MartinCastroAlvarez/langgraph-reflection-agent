"""Reflexion: tools, cited revisions, external evaluation, lessons carried across attempts."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import pytest
from langgraph.store.memory import InMemoryStore

from reflection_agent.corpus import Corpus
from reflection_agent.evaluators import CriticEvaluator
from reflection_agent.evaluators import Evaluation
from reflection_agent.evaluators import UnitTestEvaluator
from reflection_agent.evaluators import extract_code
from reflection_agent.fake import FakeReflectionModel
from reflection_agent.memory import MAX_LESSONS
from reflection_agent.memory import LessonMemory
from reflection_agent.memory import clean
from reflection_agent.memory import task_key
from reflection_agent.reflexion import build_reflexion
from reflection_agent.reflexion import parse_draft
from reflection_agent.reflexion import parse_revision
from reflection_agent.sandbox import Sandbox
from reflection_agent.sandbox import SandboxResult
from reflection_agent.tasks import CODE_TASKS
from reflection_agent.tasks import get_task
from reflection_agent.tools import Toolbox
from tests.conftest import IDEMPOTENCY
from tests.conftest import VARY
from tests.conftest import fake
from tests.conftest import make_settings
from tests.conftest import run

TOOLBOX = Toolbox(Corpus.from_dir())


def graph_for(
    name: str,
    model: FakeReflectionModel | None = None,
    store: InMemoryStore | None = None,
    **limits: Any,
) -> Any:
    return build_reflexion(
        model or fake(),
        make_settings(**limits),
        evaluator=UnitTestEvaluator(CODE_TASKS[name]),
        toolbox=TOOLBOX,
        store=store,
    )


@pytest.mark.parametrize(
    ("name", "attempts", "curve"),
    [("rle", 1, [1.0]), ("luhn", 2, [0.667, 1.0]), ("roman", 3, [0.333, 0.833, 1.0])],
)
def test_code_tasks_pass_after_the_scripted_number_of_attempts(
    name: str, attempts: int, curve: list[float]
) -> None:
    model = fake()
    final = run(graph_for(name, model), {"task": CODE_TASKS[name].prompt})
    assert final["status"] == "passed"
    assert [t["score"] for t in final["trials"]] == curve
    assert final["best_attempt"] == attempts
    expected = ["actor", "revisor"] * attempts
    for i in range(attempts - 1):
        expected.insert(3 * i + 2, "self_reflection")
    assert model.calls == expected


def test_lessons_are_carried_into_every_later_attempt() -> None:
    model = fake()
    final = run(graph_for("roman", model), {"task": CODE_TASKS["roman"].prompt})
    assert [t["lessons_seen"] for t in final["trials"]] == [0, 1, 2]
    assert len(final["lessons"]) == 2
    actor_prompts = model.prompts("actor")
    assert "subtractive rule" not in actor_prompts[0]
    assert "subtractive rule" in actor_prompts[1]
    assert "subtractive rule" in actor_prompts[2] and "upper()" in actor_prompts[2]
    assert "subtractive rule" in model.prompts("revisor")[1]
    # the evaluator's feedback from attempt 1 reaches the self-reflection and attempt 2
    assert 'roman_to_int(\\"IV\\") == 4' in actor_prompts[1]
    assert "roman_to_int" in model.prompts("self_reflection")[0]


def test_without_lessons_the_actor_never_improves_and_the_cap_ends_the_run() -> None:
    model = fake(self_reflection=["", "", ""])  # empty lessons are dropped
    final = run(graph_for("luhn", model), {"task": CODE_TASKS["luhn"].prompt})
    assert final["status"] == "max_attempts"
    assert [t["score"] for t in final["trials"]] == [0.667, 0.667, 0.667]
    assert final["lessons"] == []


def test_max_attempts_caps_the_loop() -> None:
    model = fake()
    final = run(graph_for("roman", model, max_attempts=2), {"task": CODE_TASKS["roman"].prompt})
    assert final["status"] == "max_attempts"
    assert len(final["trials"]) == 2
    assert final["best_attempt"] == 2 and final["score"] == 0.833
    assert model.calls.count("self_reflection") == 1


def test_a_store_keeps_lessons_for_the_next_run_of_the_same_task() -> None:
    store = InMemoryStore()
    task = CODE_TASKS["roman"].prompt
    first = run(graph_for("roman", store=store), {"task": task, "task_key": "roman"})
    assert first["recalled"] == 0 and len(first["trials"]) == 3
    second = run(graph_for("roman", store=store), {"task": task, "task_key": "roman"})
    assert second["recalled"] == 2
    assert len(second["trials"]) == 1 and second["status"] == "passed"
    other = run(graph_for("roman", store=store), {"task": task, "task_key": "roman-v2"})
    assert other["recalled"] == 0


def test_tools_run_and_the_revision_cites_only_retrieved_documents() -> None:
    reply = json.dumps(
        {"answer": "```python\ndef rle_encode(t):\n    return t\n```", "references": ["luhn"]}
    )
    model = fake(revisor=[reply])
    final = run(graph_for("rle", model, max_attempts=1), {"task": CODE_TASKS["rle"].prompt})
    evidence_ids = {e["id"] for e in final["evidence"]}
    assert "run-length-encoding" in evidence_ids and "calc-1" in evidence_ids
    assert any(e["text"] == "3 + 1 + 2 = 6" for e in final["evidence"])
    assert final["references"] == []  # "luhn" was never retrieved: a hallucinated citation
    assert final["tool_calls"] == 2


def test_max_queries_bounds_the_tool_calls() -> None:
    final = run(graph_for("luhn", max_queries=1), {"task": CODE_TASKS["luhn"].prompt})
    assert final["tool_calls"] == len(final["trials"])


def test_code_that_breaks_the_policy_is_refused_not_run() -> None:
    evil = "```python\nimport os\ndef rle_encode(t):\n    return os.getcwd()\n```"
    model = fake(revisor=[json.dumps({"answer": evil, "references": []})])
    final = run(graph_for("rle", model, max_attempts=1), {"task": CODE_TASKS["rle"].prompt})
    trial = final["trials"][0]
    assert not trial["passed"] and trial["score"] == 0.0
    assert "refused by the sandbox policy" in trial["feedback"]


def test_text_tasks_use_the_critic_as_the_evaluator() -> None:
    model = fake()
    graph = build_reflexion(
        model,
        make_settings(),
        evaluator=CriticEvaluator(model, threshold=8.0),
        toolbox=TOOLBOX,
    )
    final = run(graph, {"task": IDEMPOTENCY})
    assert final["status"] == "passed"
    assert [t["score"] for t in final["trials"]] == [0.5, 0.7, 0.86]
    assert final["references"] == ["idempotency"]
    assert "critic score 5/10" in final["trials"][0]["feedback"]
    failed = run(graph, {"task": VARY})
    assert failed["status"] == "max_attempts" and failed["best_attempt"] == 3


@dataclass
class _Countdown:
    """An evaluator that passes on the n-th call."""

    passes_on: int
    kind: str = "countdown"
    seen: list[str] = field(default_factory=list)

    async def evaluate(self, task: str, answer: str) -> Evaluation:
        self.seen.append(answer)
        ok = len(self.seen) >= self.passes_on
        return Evaluation(passed=ok, score=1.0 if ok else 0.0, feedback="nope")


def test_any_evaluator_plugs_in() -> None:
    evaluator = _Countdown(passes_on=2)
    graph = build_reflexion(fake(), make_settings(), evaluator=evaluator, toolbox=TOOLBOX)
    final = run(graph, {"task": "Name a prime."})
    assert final["status"] == "passed" and len(evaluator.seen) == 2
    assert final["lessons"][0].startswith("Lesson 1: fix what the evaluator reported: nope")


def test_parse_draft_tolerates_prose_and_caps_queries() -> None:
    text = "<think>hm</think>Sure: " + json.dumps(
        {"answer": "x", "search_queries": ["a", "b", "c", "d"], "critique": "bad"}
    )
    draft = parse_draft(text, max_queries=2)
    assert draft["queries"] == ["a", "b"] and draft["critique"] == {} and draft["calcs"] == []
    plain = parse_draft("just an answer", max_queries=3)
    assert plain == {"answer": "just an answer", "critique": {}, "queries": [], "calcs": []}


def test_parse_revision_falls_back_to_the_draft() -> None:
    assert parse_revision("", draft="d", cited=set()) == ("d", [])
    assert parse_revision('{"references": ["a", "a", "b"]}', draft="d", cited={"a"}) == ("d", ["a"])


def test_extract_code_picks_the_longest_fence() -> None:
    answer = "```\nx = 1\n```\ntext\n```python\ndef f():\n    return 2\n```"
    assert extract_code(answer) == "def f():\n    return 2\n"
    assert extract_code("def g(): pass") == "def g(): pass\n"


def test_unit_test_evaluator_runs_code_only_through_the_sandbox() -> None:
    calls: list[str] = []

    class Spy(Sandbox):
        def run(self, code: str, tests: list[str]) -> SandboxResult:
            calls.append(code)
            return super().run(code, tests)

    evaluator = UnitTestEvaluator(CODE_TASKS["rle"], Spy())
    result = asyncio.run(
        evaluator.evaluate("t", "```python\ndef rle_encode(t):\n    return t\n```")
    )
    assert calls == ["def rle_encode(t):\n    return t\n"]
    assert not result.passed and result.details["tests_total"] == 5


def test_memory_helpers() -> None:
    assert task_key("Some  task", None) == task_key("some task", None)
    assert task_key("x", "luhn") == "luhn"
    with pytest.raises(ValueError, match="task key"):
        task_key("x", "../etc")
    assert clean(["a", " a ", "", "b"]) == ["a", "b"]
    assert len(clean([f"l{i}" for i in range(30)])) == MAX_LESSONS
    assert asyncio.run(LessonMemory().recall("k")) == []
    asyncio.run(LessonMemory().remember("k", ["x"]))  # no store: a no-op
    store = InMemoryStore()
    store.put(("reflexion", "lessons"), "bad", {"lessons": "not a list"})
    assert asyncio.run(LessonMemory(store).recall("bad")) == []


def test_get_task() -> None:
    assert get_task("luhn").entry_point == "luhn_valid"
    with pytest.raises(ValueError, match="unknown code task"):
        get_task("nope")
