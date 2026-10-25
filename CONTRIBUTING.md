# Contributing to langgraph-reflection-agent

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install                       # add -E anthropic for the optional adapter
poetry run pytest
poetry run ruff check reflection_agent tests examples && poetry run ruff format --check reflection_agent tests examples
poetry run mypy reflection_agent
poetry run bandit -c pyproject.toml -q -r reflection_agent
poetry run pip-audit
poetry run python -m examples.quickstart
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests that call a real model or touch the network.
- Running model-written code anywhere but the sandbox, or an in-process "fast path".
- Hosted providers, web tools or tracing turned on by default.
- A loop that the limits don't bound.
- Hand-edited benchmark numbers or plots.

## What it does accept

- New reflection variants (self-refine with multiple critics, debate-style critique,
  full LATS with rollouts), reusing the rubric and the evaluator protocol, with tests
  for termination and caps.
- New evaluators (linters, type checkers, schema validators) and bundled tasks.
- Sandbox hardening, each with a test that attacks the new layer with the others off.

## Updating the benchmark

```bash
poetry run python -m reflection_agent.bench --json docs/benchmark/results.json --plot docs/convergence.png
```

Paste the printed table into the README together with the machine, versions and date.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
