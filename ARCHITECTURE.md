# Architecture

```
reflection_agent/
  config.py         Settings + Limits from REFLECTION_* env vars; recursion_limit backstop
  rubric.py         RUBRIC (4 weighted criteria), Critique (score computed in code), extract_json
  prompts.py        system prompts ("Role: <name>." first) and data_block() for untrusted text
  reflect.py        variant 1: generate -> reflect -> (generate | finish)
  reflexion.py      variant 2: recall -> actor -> tools -> revisor -> evaluate -> (reflect -> actor | finish)
  lats.py           variant 3: expand (UCT select + N samples) -> reflect (+ backprop) -> (expand | finish)
  evaluators.py     Evaluator protocol, UnitTestEvaluator (sandbox), CriticEvaluator, extract_code
  sandbox.py        check_policy (AST) + Sandbox.run: subprocess, rlimits, timeout, nonce report
  sandbox_child.py  runs INSIDE the sandbox: rlimits, pre-imports, audit hook, runpy, report
  memory.py         LessonMemory over a LangGraph BaseStore; task_key(); clean()
  tools.py          AST calculator + Toolbox (corpus search + calculations -> cited evidence)
  corpus.py         Markdown bullets -> facts, in-memory BM25, confined loader
  tasks.py          bundled coding tasks: prompt (seen by the model) + tests (never seen)
  graphs.py         build_graph(variant, settings, ...), make_evaluator(), make_toolbox()
  fake.py           FakeReflectionModel: role-aware scripted fake whose answers improve
  progress.py       CallCounter (model calls by role), stream_run(), emit()
  bench.py          iterations-to-pass table + convergence plot (matplotlib, optional)
  cli.py            python -m reflection_agent "<task>" [--variant ...] [--code-task ...] [--memory ...]
  data/corpus/      short technical documents (one fact per bullet)
```

## One rubric, three loops

```
reflect    generate ─ reflect ─┬─ score ≥ threshold or iteration = max ─ finish (best draft)
                ▲              │
                └──────────────┘ critique (issues, suggestions) into the next draft

reflexion  recall ─ actor ─ tools ─ revisor ─ evaluate ─┬─ passed or attempt = max ─ finish
           (store)   ▲                     (tests in    │
                     └─────── reflect ◄────  sandbox)   ┘ lesson -> state + store

lats       expand (UCT select, N samples) ─ reflect (score, backprop) ─┬─ child ≥ threshold,
              ▲                                                         │  expansions = max,
              └─────────────────────────────────────────────────────────┘  or tree exhausted
```

- The reflector is shared: basic reflection and LATS use it as their critic, Reflexion
  uses it as the `CriticEvaluator` for text tasks. `Critique.score` is the weighted
  mean of the four criterion scores, clamped 0-10.
- Every loop's exit is a conditional edge that reads only state the graph wrote
  (scores, counters, test results), never a model's claim that it is done.

## State

All state is JSON-native (plain dicts and lists), so any checkpointer can store it.

| Graph | Key | Reducer | Notes |
| --- | --- | --- | --- |
| reflect | `rounds` | `operator.add` | one per draft: iteration, draft, score, scores, issues, suggestions |
| reflexion | `trials` | `operator.add` | one per attempt: answer, references, passed, score, feedback, lessons_seen |
| reflexion | `tool_calls` | `operator.add` | searches + calculations |
| reflexion | `lessons` | last write | the episodic memory of this run (recalled + new) |
| lats | `tree` | last write | nodes: id, parent, depth, answer, score, visits, value; copied, never mutated |
| lats | `frontier` | last write | the children of the latest expansion |

## Reflexion memory

`recall` reads `("reflexion", "lessons") / <task_key>` from the store, `reflect` writes
the updated list after every lesson. Without a store, memory lives only in state.
`--memory FILE` uses `AsyncSqliteStore`; any `BaseStore` works.

## LATS search

`select` walks from the root, choosing among *expandable* children (below `max_depth`,
or with an expandable descendant) the one with the highest UCT, until it reaches a node
with no children. `expand` samples `candidates` answers concurrently; `reflect` scores
them concurrently and calls `backpropagate(tree, child, score / 10)` for each, which adds
one visit and the reward to the child and every ancestor. `best_node` picks the highest
score in the tree, breaking ties by mean value.

## Sandbox protocol

```
parent (sandbox.py)                           child (sandbox_child.py, python -I -S -B)
  check_policy(code)  ── refuse? ──► result
  write solution.py, check.py to a temp dir
  Popen(argv, env={}, cwd=tmp, new session) ─► read {"nonce", "limits"} from stdin, close it
                                               setrlimit(CPU, AS, FSIZE=0, NPROC=0, NOFILE, CORE)
                                               pre-import allowlisted modules (+ runpy's)
                                               addaudithook(refuse network/process/fs/frames)
                                               runpy solution.py  (may read: solution.py only)
                                               runpy check.py     (may read: check.py only)
                                               call test_00.. with stdout captured
  communicate(timeout) ◄─────────────────────  print nonce + JSON report on the real stdout
  timeout? killpg(SIGKILL)
  keep only the nonce line and the test names it generated
```
