# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-25

### Added

- Basic reflection: generator -> rubric reflector -> revision loop that stops at a score
  threshold or `max_iterations` and returns the best draft. The rubric's weighted score
  is computed in code; unparseable critiques score 0.
- Reflexion: actor with self-critique, search queries and calculations; local tools
  (BM25 corpus search, AST calculator); a revisor whose citations are filtered to
  retrieved documents; pluggable external evaluators (unit tests in the sandbox, or the
  rubric critic); self-reflection lessons in episodic memory, optionally persisted in a
  LangGraph store (`InMemoryStore`, `AsyncSqliteStore` via `--memory`).
- LATS-lite: UCT selection, parallel expansion of `candidates` children, rubric
  reflection, reward backpropagation, depth/expansion bounds, best-node choice.
- A hardened subprocess sandbox: static AST policy, isolated interpreter with an empty
  environment, `resource` limits, a PEP 578 audit hook, a wall-clock timeout, and a
  nonce-protected report.
- Three bundled coding tasks (Luhn, Roman numerals, run-length encoding) with hidden
  tests, and a corpus of short technical documents written for the repository.
- `FakeReflectionModel`, a role-aware scripted fake chat model whose answers improve
  only when given the critique or the lessons.
- `python -m reflection_agent` CLI (ollama by default, Anthropic optional, fake
  offline), `--mermaid` diagrams, and `python -m reflection_agent.bench`, the
  iterations-to-pass table and convergence plot.
