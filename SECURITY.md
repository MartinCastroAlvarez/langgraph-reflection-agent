# Security

langgraph-reflection-agent runs LLM agents that critique their own output, search a
local document collection, and **execute code the model wrote** to test it. People
install it and point it at their own models, documents and tasks. Design goals: model
code never runs in the agent's process, nothing leaves the machine unless you opt in,
text in documents, tasks or model replies can't take over the agent, and no loop runs
without bound.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/langgraph-reflection-agent/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise. A sandbox escape is in scope and very welcome.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The local machine and filesystem, environment variables (API keys), the
network position of the host, the document folder, and your model endpoint.

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| The model | drafts, critiques, code, search queries, calculations, lessons | run code outside the sandbox, read or write files, reach the network, loop forever, pass a draft by fiat |
| A document author | the text of a file in the corpus folder | give the agent instructions, make the loader read files outside the folder |
| Whoever writes the task | the task text | escape the prompt structure, run code, exhaust the host |
| Whoever controls `OLLAMA_HOST` | where prompts are sent | receive your documents and tasks without your explicit opt-in |

| Threat | Control |
| --- | --- |
| Model-written code attacking the host | Runs only through `Sandbox.run`, in a separate `python -I -S -B` process with an empty environment, a fresh temp directory, a new session and no inherited fds. Four layers: (1) a static AST policy (import allowlist; no `open`/`eval`/`exec`/`compile`/`getattr`/`globals`/`vars`/...; no `_private`, dunder or frame attributes; no dunder method definitions); (2) `resource` limits: CPU 3 s, address space 512 MB, file size 0, no new processes, 32 fds, no core dumps; (3) a PEP 578 audit hook refusing `socket.*`, `subprocess.*`, `os.*` (except listing the temp dir), `shutil.*`, `ctypes.*`, `urllib.*`, frame/gc introspection, non-allowlisted imports and any open except the single file of the current phase; (4) a wall-clock timeout that kills the process group. Each runtime layer is tested with the policy switched off. |
| Reward hacking by the candidate | The tests are trusted code the model never sees; the candidate's top-level code can't read them (open is refused, the hook only allows the current phase's file). Results come back on a line prefixed with a random nonce sent over stdin; candidate stdout is captured separately; only the test names the parent generated are counted; dunder methods (an `__eq__` that always returns `True`) are refused. |
| Prompt injection (documents, task, drafts, critiques, test output, lessons) | Every system prompt says `<data>` content is untrusted and never instructions. Untrusted text is wrapped in `<data name=...>` after escaping `& < >`, so it can't close the element (tested). Escaping reduces injection but can't remove it: a poisoned document can still bias an answer, and a model can still write a misleading lesson. Treat answers about untrusted documents as untrusted. |
| A critic "passing" a draft by fiat | The overall score is the weighted mean of per-criterion scores computed in code; any `overall` field the model writes is ignored, and scores are clamped to 0-10. Unparseable critiques score 0. |
| Hallucinated citations | The revisor's references are filtered to document ids that the tools actually returned in that attempt. |
| Poisoned episodic memory | Lessons are model output: capped at 10 × 500 characters, de-duplicated, stored under a validated key (`[a-z0-9._-]{1,64}` or a hash of the task), and only ever placed inside `<data>` blocks. They can still be wrong; delete the store (or use a new `--task-key`) to reset. |
| Code execution through the calculator | No `eval`: `ast.parse(mode="eval")` walked against an allowlist (numbers, arithmetic operators, `pi`, `e`, nine math functions), ≤ 300 chars, ≤ 64 nodes, bounded exponents and results. |
| Path traversal / symlink escape in the corpus loader | The root is resolved first; only regular `.md`/`.txt` files inside it are read; symlinks and hidden entries are skipped; size and count are capped. Tested with a symlink pointing outside. |
| Runaway loops | `max_iterations`, `max_attempts`, `max_queries`, `candidates`, `max_depth`, `max_expansions` (each 1-20) are enforced by the graph; `recursion_limit` is derived from them as a backstop. The CLI refuses tasks over 4,000 characters. |
| Data leaving the machine | `OLLAMA_HOST` must be loopback unless `REFLECTION_ALLOW_REMOTE_OLLAMA=1`. Hosted providers are opt-in and need your own key. LangSmith tracing stays off unless you set `LANGSMITH_TRACING`. |
| Secrets | Only `ANTHROPIC_API_KEY` is read, only by the optional adapter. Keys never enter prompts, state, stream events, the memory store or logs, and the sandbox child starts with an empty environment (tested). `.env` is git-ignored. |

### Limits of the sandbox (read this before running untrusted code at scale)

It is defence in depth inside one OS user, not a security boundary against a
determined attacker. Known gaps: `RLIMIT_NPROC` is not enforced for root; `RLIMIT_AS`
is not enforced on macOS; the audit hook can't see everything a C extension does (the
policy refuses importing anything but a few pure stdlib modules, but with the policy
off, modules already loaded by the child, such as `os` and `sys`, are importable); and
there is no network namespace, so the network block relies on the hook and the
policy. For untrusted code at scale, run the whole agent in OS isolation: a container
with `--network none`, gVisor, nsjail, or a VM. The in-process alternative (`exec` with
an AST filter) is deliberately not offered: one filter gap would compromise the agent
process itself.

### The two silenced findings

`reflection_agent/sandbox.py` carries the package's only `# nosec` markers (bandit
B404: importing `subprocess`; B603: `Popen` without a shell) and its only ruff
per-file ignore (S404, S603). That module exists to start exactly one process: a fixed
argv (`sys.executable -I -S -B <bundled sandbox_child.py> <temp dir>`), no shell, an
empty environment. A test fails if any other marker appears in the package.

## 2. Dependencies

CI runs `bandit` on the package and `pip-audit` on the locked environment. Dependabot
alerts, security updates and secret scanning are enabled on the repository.

## 3. Running it safely

- Keep ollama on `127.0.0.1`. If you point `OLLAMA_HOST` elsewhere, use TLS and trust
  that host with your documents and tasks.
- Run Reflexion on code tasks inside a container or VM when the model or the tasks are
  not yours.
- Treat answers, citations and lessons as drafts.
