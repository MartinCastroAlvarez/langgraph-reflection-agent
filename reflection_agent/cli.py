"""``python -m reflection_agent "<task>"``: draft, critique, revise; or Reflexion, or LATS.

python -m reflection_agent "Explain idempotency keys in three sentences."        # reflect, ollama
python -m reflection_agent --variant lats "..."                                   # tree search
python -m reflection_agent --variant reflexion --code-task luhn                   # unit tests
python -m reflection_agent --variant reflexion --code-task roman --memory m.db    # keep lessons
python -m reflection_agent --provider fake --variant all "..."                    # offline
python -m reflection_agent --mermaid reflexion                                    # print a graph
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import sys
import time
from collections.abc import AsyncIterator
from collections.abc import Sequence
from typing import Any
from typing import TextIO

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

from reflection_agent.config import DEFAULT_MODELS
from reflection_agent.config import PROVIDERS
from reflection_agent.config import VARIANTS
from reflection_agent.config import Settings
from reflection_agent.graphs import build_graph
from reflection_agent.llm import ModelConfigError
from reflection_agent.progress import CallCounter
from reflection_agent.progress import Progress
from reflection_agent.progress import format_progress
from reflection_agent.progress import stream_run
from reflection_agent.tasks import CODE_TASKS
from reflection_agent.tasks import get_task

__all__ = ["main", "run_variant"]

MAX_TASK_CHARS = 4000
_LIMIT_FLAGS = (
    "threshold",
    "max_iterations",
    "max_attempts",
    "candidates",
    "max_depth",
    "max_expansions",
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m reflection_agent", description=(__doc__ or "").split("\n")[0]
    )
    p.add_argument("task", nargs="?", help="the task (omit with --code-task)")
    p.add_argument("--variant", choices=(*VARIANTS, "all"), default="reflect")
    p.add_argument("--provider", choices=PROVIDERS)
    p.add_argument("--model", help="model id for the provider")
    p.add_argument("--latency", type=float, help="seconds per call for --provider fake")
    p.add_argument(
        "--code-task", choices=sorted(CODE_TASKS), help="a bundled coding task with hidden tests"
    )
    p.add_argument("--memory", metavar="FILE", help="SQLite store that keeps Reflexion lessons")
    p.add_argument("--task-key", help="memory key for the task (default: a hash of it)")
    p.add_argument("--threshold", type=float)
    p.add_argument("--max-iterations", type=int)
    p.add_argument("--max-attempts", type=int)
    p.add_argument("--candidates", type=int)
    p.add_argument("--max-depth", type=int)
    p.add_argument("--max-expansions", type=int)
    p.add_argument("--mermaid", choices=VARIANTS, help="print a graph and exit")
    p.add_argument("--json", action="store_true", help="print the reports as JSON")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    return p


def _settings(args: argparse.Namespace) -> Settings:
    base = Settings.from_env()
    changes: dict[str, Any] = {}
    if args.provider:
        changes["provider"] = args.provider
        if not args.model:
            changes["model"] = DEFAULT_MODELS[args.provider]
    if args.model:
        changes["model"] = args.model
    if args.latency is not None:
        changes["fake_latency"] = args.latency
    limits = {k: getattr(args, k) for k in _LIMIT_FLAGS if getattr(args, k) is not None}
    if limits:
        changes["limits"] = dataclasses.replace(base.limits, **limits)
    return dataclasses.replace(base, **changes)


@contextlib.asynccontextmanager
async def _store(path: str | None) -> AsyncIterator[BaseStore | None]:
    if path is None:
        yield None
        return
    async with AsyncSqliteStore.from_conn_string(path) as store:
        await store.setup()
        yield store


def summarize(variant: str, final: dict[str, Any]) -> dict[str, Any]:
    """The comparable numbers of one run: how many loops it took, and how it ended."""
    report: dict[str, Any] = {
        "variant": variant,
        "status": final.get("status"),
        "score": final.get("score"),
        "answer": final.get("answer", ""),
    }
    if variant == "reflect":
        report["iterations"] = final.get("iteration", 0)
        report["curve"] = [r["score"] for r in final.get("rounds", [])]
    elif variant == "reflexion":
        trials = final.get("trials", [])
        report["attempts"] = len(trials)
        report["curve"] = [t["score"] for t in trials]
        report["references"] = final.get("references", [])
        report["lessons"] = final.get("lessons", [])
        report["recalled"] = final.get("recalled", 0)
        report["tool_calls"] = final.get("tool_calls", 0)
    else:
        rounds = final.get("rounds", [])
        report["expansions"] = final.get("expansions", 0)
        report["nodes"] = len(final.get("tree", [])) - 1
        report["curve"] = [r["best_so_far"] for r in rounds]
        report["children"] = [list(r["children"].values()) for r in rounds]
        report["best_id"] = final.get("best_id")
    return report


async def run_variant(
    variant: str,
    task: str,
    settings: Settings,
    *,
    code_task: str | None = None,
    memory: str | None = None,
    task_key: str | None = None,
    out: TextIO | None = None,
    model: Any = None,
) -> dict[str, Any]:
    counter = CallCounter()
    config: RunnableConfig = {"recursion_limit": settings.recursion_limit, "callbacks": [counter]}
    inputs: dict[str, Any] = {"task": task}
    if task_key:
        inputs["task_key"] = task_key
    ct = get_task(code_task) if code_task else None
    started = time.perf_counter()
    final: dict[str, Any] = {}
    async with _store(memory if variant == "reflexion" else None) as store:
        graph = build_graph(variant, settings, model=model, code_task=ct, store=store)
        async for item in stream_run(graph, inputs, config):
            if isinstance(item, Progress):
                if out is not None:
                    print(format_progress(item), file=out)
            else:
                final = item
    report = summarize(variant, final)
    report["llm_calls"] = counter.total
    report["llm_calls_by_role"] = dict(counter.by_role)
    report["seconds"] = round(time.perf_counter() - started, 3)
    return report


def _mermaid(name: str) -> str:
    return build_graph(name, Settings(provider="fake")).get_graph().draw_mermaid()


def _loops(r: dict[str, Any]) -> str:
    if r["variant"] == "reflect":
        return f"iterations={r['iterations']}"
    if r["variant"] == "reflexion":
        return f"attempts={r['attempts']} tool_calls={r['tool_calls']} recalled={r['recalled']}"
    return f"expansions={r['expansions']} nodes={r['nodes']}"


def main(argv: Sequence[str] | None = None, out: TextIO = sys.stdout) -> int:
    args = _parser().parse_args(argv)
    if args.mermaid:
        print(_mermaid(args.mermaid), file=out)
        return 0
    task = args.task
    if args.code_task:
        if task:
            print("give either a task or --code-task, not both", file=sys.stderr)
            return 2
        task = get_task(args.code_task).prompt
    if not task:
        _parser().print_usage(sys.stderr)
        return 2
    if len(task) > MAX_TASK_CHARS:
        print(f"the task is longer than {MAX_TASK_CHARS} characters", file=sys.stderr)
        return 2
    try:
        settings = _settings(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    reports = []
    try:
        for variant in variants:
            if len(variants) > 1 and not args.quiet:
                print(f"\n== {variant} ==", file=out)
            report = asyncio.run(
                run_variant(
                    variant,
                    task,
                    settings,
                    code_task=args.code_task,
                    memory=args.memory,
                    task_key=args.task_key,
                    out=None if args.quiet or args.json else out,
                )
            )
            reports.append(report)
    except ModelConfigError as exc:
        print(f"model error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {str(exc)[:300]}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2), file=out)
        return 0
    for r in reports:
        print(f"\n[{r['variant']}] {r['answer']}", file=out)
        if r.get("references"):
            print(f"  references: {', '.join(r['references'])}", file=out)
        print(
            f"  status={r['status']} score={r['score']} {_loops(r)} "
            f"llm_calls={r['llm_calls']} seconds={r['seconds']}",
            file=out,
        )
    return 0
