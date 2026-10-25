"""Run the three reflection variants offline, then show lessons outliving a run.

    python -m examples.quickstart

Uses the fake model, so it needs no ollama and no key. Swap ``provider="fake"`` for
``"ollama"`` (and ``ollama pull qwen3:4b``) to run it on a local model.
"""

from __future__ import annotations

import asyncio

from langgraph.store.memory import InMemoryStore

from reflection_agent import CODE_TASKS
from reflection_agent import Settings
from reflection_agent import build_graph

TASK = "Explain idempotency keys for a payments API in at most three sentences."


async def compare() -> None:
    settings = Settings(provider="fake")
    final = await build_graph("reflect", settings).ainvoke({"task": TASK})
    print("reflect:  ", [r["score"] for r in final["rounds"]], final["status"])
    final = await build_graph("lats", settings).ainvoke({"task": TASK})
    print("lats:     ", [r["best_so_far"] for r in final["rounds"]], final["status"])
    final = await build_graph("reflexion", settings).ainvoke({"task": TASK})
    print("reflexion:", [t["score"] for t in final["trials"]], final["references"])
    assert final["status"] == "passed"


async def lessons() -> None:
    settings = Settings(provider="fake")
    store = InMemoryStore()  # use SqliteStore/PostgresStore to keep lessons on disk
    task = CODE_TASKS["roman"]
    for run in (1, 2):
        graph = build_graph("reflexion", settings, code_task=task, store=store)
        final = await graph.ainvoke({"task": task.prompt, "task_key": "roman"})
        tests = [t["details"]["tests_passed"] for t in final["trials"]]
        print(f"roman run {run}: {final['recalled']} lessons recalled, tests passed: {tests}")
    assert len(final["trials"]) == 1  # the second run starts with the lessons and passes at once


if __name__ == "__main__":
    asyncio.run(compare())
    asyncio.run(lessons())
