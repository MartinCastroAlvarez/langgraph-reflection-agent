"""``python -m reflection_agent.bench``: iterations-to-pass for every variant and task.

Runs the real graphs, the real corpus search and calculator, and the real sandbox
running the real unit tests. Only the model is replaced, by
:class:`~reflection_agent.fake.FakeReflectionModel`, whose answers improve on a script
(and, for Reflexion, only when it is given the lessons). So the numbers show how each
loop behaves and what it costs in model calls, not how good any model is.

``--plot docs/convergence.png`` draws the convergence curves (needs the ``plot``
extra: matplotlib). ``--json`` writes the raw reports.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from collections.abc import Sequence
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

from reflection_agent.cli import run_variant
from reflection_agent.config import Settings
from reflection_agent.tasks import CODE_TASKS

__all__ = ["CASES", "main", "markdown_table", "run_bench"]

IDEMPOTENCY = "Explain idempotency keys for a payments API in at most three sentences."
VARY = "Explain the HTTP Vary header to a junior developer in two sentences."

CASES: tuple[tuple[str, str, str | None, str], ...] = (
    # (label, variant, code task, task text)
    ("idempotency (text)", "reflect", None, IDEMPOTENCY),
    ("idempotency (text)", "reflexion", None, IDEMPOTENCY),
    ("idempotency (text)", "lats", None, IDEMPOTENCY),
    ("vary header (text)", "reflect", None, VARY),
    ("vary header (text)", "reflexion", None, VARY),
    ("vary header (text)", "lats", None, VARY),
    ("luhn (code)", "reflexion", "luhn", CODE_TASKS["luhn"].prompt),
    ("roman (code)", "reflexion", "roman", CODE_TASKS["roman"].prompt),
    ("rle (code)", "reflexion", "rle", CODE_TASKS["rle"].prompt),
)


def _loops(r: dict[str, Any]) -> int:
    return int(r.get("iterations") or r.get("attempts") or r.get("expansions") or 0)


async def run_bench(settings: Settings | None = None) -> list[dict[str, Any]]:
    base = settings or Settings(provider="fake")
    rows = []
    for label, variant, code_task, task in CASES:
        report = await run_variant(variant, task, base, code_task=code_task)
        report["task"] = label
        report["evaluator"] = (
            "unit tests (sandbox)"
            if code_task
            else ("critic" if variant == "reflexion" else "rubric critic")
        )
        report.pop("answer", None)
        report.pop("lessons", None)
        rows.append(report)
    return rows


def _unit(r: dict[str, Any]) -> str:
    return {"reflect": "iterations", "reflexion": "attempts", "lats": "expansions"}[r["variant"]]


def _fmt(r: dict[str, Any], value: float) -> str:
    if r["evaluator"].startswith("unit"):
        return f"{value:.0%} tests"
    return f"{value * 10 if r['variant'] == 'reflexion' else value:.1f}"


def markdown_table(rows: list[dict[str, Any]], threshold: float) -> str:
    lines = [
        "| Task | Variant | Evaluator | Loops to pass | First | Final | Model calls | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        curve = r["curve"]
        n = _loops(r)
        loops = (
            f"{n} {_unit(r) if n != 1 else _unit(r)[:-1]}"
            if r["status"] == "passed"
            else f"- ({n} max)"
        )
        lines.append(
            f"| {r['task']} | {r['variant']} | {r['evaluator']} | {loops} | "
            f"{_fmt(r, curve[0])} | {_fmt(r, curve[-1])} | {r['llm_calls']} | {r['status']} |"
        )
    lines.append("")
    lines.append(
        f"Text scores are the rubric's 0-10 weighted mean (pass at {threshold:g}); "
        "LATS columns are the best score in the tree after each expansion."
    )
    return "\n".join(lines)


def plot(rows: list[dict[str, Any]], threshold: float, path: Path) -> None:  # pragma: no cover
    """Three panels: reflection scores, Reflexion test pass rate, LATS best-so-far."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    series = ("#2a78d6", "#eb6834", "#1baf7a")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": grid,
            "axes.labelcolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "text.color": ink,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=surface)
    panels = (
        ("reflect", False, "Basic reflection", "draft", "critic score (0-10)"),
        ("reflexion", True, "Reflexion on code tasks", "attempt", "unit tests passed"),
        (
            "lats",
            False,
            "LATS-lite (rings: every candidate)",
            "expansion",
            "best score in the tree",
        ),
    )
    for ax, (variant, code, title, xlabel, ylabel) in zip(axes, panels, strict=True):
        ax.set_facecolor(surface)
        picked = [
            r for r in rows if r["variant"] == variant and r["evaluator"].startswith("unit") == code
        ]
        line = 1.0 if code else threshold
        ax.axhline(line, color=muted, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            "all tests pass" if code else f"threshold {threshold:g}",
            (0.02, line),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            color=muted,
            fontsize=9,
        )
        longest = 1
        for color, r in zip(series, picked, strict=False):
            ys = r["curve"]
            xs = list(range(1, len(ys) + 1))
            longest = max(longest, len(xs))
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, label=r["task"])
            if variant == "lats":  # every candidate's score, hollow, behind the best-so-far
                for x, children in enumerate(r["children"], start=1):
                    ax.plot(
                        [x] * len(children),
                        children,
                        linestyle="none",
                        marker="o",
                        markersize=8,
                        markerfacecolor="none",
                        markeredgecolor=color,
                        alpha=0.6,
                    )
        ax.set_xticks(range(1, longest + 1))
        ax.set_xlim(0.7, longest + 0.3)
        if code:
            ax.set_ylim(0, 1.08)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.set_ylim(0, 10.5)
        ax.set_title(title, loc="left", fontsize=13, color=ink, pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=muted)
    fig.suptitle(
        "Convergence with the scripted fake model (real graphs, tools and sandbox)",
        x=0.01,
        ha="left",
        color=muted,
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=surface)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m reflection_agent.bench")
    p.add_argument("--json", type=Path, help="write the reports to this file")
    p.add_argument("--plot", type=Path, help="write the convergence chart (PNG) here")
    args = p.parse_args(argv)
    settings = Settings(provider="fake")
    rows = asyncio.run(run_bench(settings))
    print(markdown_table(rows, settings.limits.threshold))
    if args.json:
        meta = {
            "date": date.today().isoformat(),
            "python": platform.python_version(),
            "langgraph": version("langgraph"),
            "model": "FakeReflectionModel (scripted)",
            "limits": {
                "threshold": settings.limits.threshold,
                "max_iterations": settings.limits.max_iterations,
                "max_attempts": settings.limits.max_attempts,
                "candidates": settings.limits.candidates,
                "max_expansions": settings.limits.max_expansions,
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n", "utf-8")
    if args.plot:
        plot(rows, settings.limits.threshold, args.plot)
        print(f"wrote {args.plot}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
