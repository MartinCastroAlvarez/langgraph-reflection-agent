# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`langgraph-reflection-agent` shows the reflection agent architecture in current LangGraph
(basic reflection with a rubric critic, Reflexion with tools, citations, an external
evaluator and episodic memory, LATS-lite tree search) as small, correct and tested code.

## The rules

1. **Local by default.** ollama on loopback is the default; hosted providers stay
   optional adapters. No telemetry, no LangSmith upload, no web access.
2. **Generated code runs only in `sandbox.Sandbox`.** Never `exec`, `eval`, import or
   `runpy` model output anywhere else. `subprocess` lives only in `sandbox.py`, `runpy`
   only in `sandbox_child.py`. Don't weaken a sandbox layer; add a hostile test with any
   change to it.
3. **Model output is untrusted.** Critiques go through `rubric.parse_critique` (the score
   is computed in code), drafts and revisions through `reflexion.parse_draft` /
   `parse_revision`; untrusted text enters prompts only through `prompts.data_block`.
4. **The graph enforces every limit, not the model.** Each loop ends with a status and an
   answer, never an exception.
5. **Offline tests.** `FakeReflectionModel` only. Tests check control flow (termination,
   caps, lessons, tree choice, sandbox refusals), not prose.
6. **README numbers come from real runs.** Rerun
   `python -m reflection_agent.bench --json docs/benchmark/results.json --plot docs/convergence.png`
   and paste its table. A test compares the JSON to a fresh run.
7. Never add `# noqa`, `# nosec` or `# type: ignore` (the two documented ones in
   `sandbox.py` are the only exceptions), skip a test, or loosen lint/type/coverage config.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`, `mypy`
(strict), `bandit` and `pip-audit` all green; `python -m examples.quickstart` runs;
README and CHANGELOG are updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets or `.env`.
