"""Run model-written code against trusted unit tests, in a hardened subprocess.

Generated code never runs in the agent's own process. :class:`Sandbox` runs it in four
layers, each of which would stop most escapes on its own:

1. **A static policy** (:func:`check_policy`): the code is parsed, never executed, and
   refused if it imports anything outside a small allowlist (``math``, ``re``,
   ``collections``...), names ``open``/``eval``/``exec``/``getattr``/``globals``..., touches
   any ``_private`` or frame attribute, or defines dunder methods (an ``__eq__`` that is
   always ``True`` would game ``assert f(x) == y``).
2. **A separate interpreter**: ``python -I -S -B`` (isolated, no site, no bytecode
   files) with an empty environment, a fresh temporary directory as its working
   directory, ``start_new_session=True`` and no inherited file descriptors.
3. **Resource limits** set by the child before the candidate loads: CPU seconds,
   address space, file size 0 (no writes), no new processes, few open files, no core
   dumps.
4. **A PEP 578 audit hook** in the child that refuses network (``socket.*``), processes
   (``subprocess.*``, ``os.system``, ``os.fork``...), filesystem events outside the two
   files it runs, frame introspection, native code (``ctypes``) and imports outside the
   allowlist.

On top, the parent enforces a wall-clock timeout and kills the whole process group.
The report comes back on a line prefixed by a random nonce that the candidate never
sees, and only the test names the parent wrote are counted.

Trade-off (see README): the in-process alternative, ``exec`` of AST-restricted code
with stripped builtins, is faster (~0.1 ms vs ~30 ms per run) and portable, but a
single gap in the AST filter is a full compromise of the agent process, and it has no
memory/CPU limits. Python sandboxes in the same process have a long history of
escapes, so this package only offers the subprocess. Neither is a boundary against a
determined attacker on a shared host: for that, add OS isolation (a container, gVisor,
nsjail, a VM) around the whole agent. Needs POSIX (``resource``); elsewhere
:class:`SandboxUnavailable` is raised.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import secrets
import signal

# The sandbox is the one module that starts a process (bandit B404, ruff S404 allowed here).
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

__all__ = [
    "ALLOWED_IMPORTS",
    "PolicyViolation",
    "Sandbox",
    "SandboxResult",
    "SandboxUnavailable",
    "TestOutcome",
    "check_policy",
]

ALLOWED_IMPORTS = frozenset(
    {
        "bisect",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "math",
        "operator",
        "re",
        "statistics",
        "string",
        "typing",
    }
)
BANNED_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "__builtins__",
        "__loader__",
        "__spec__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "input",
        "breakpoint",
        "help",
        "memoryview",
        "exit",
        "quit",
        "copyright",
        "credits",
        "license",
    }
)
FRAME_ATTRIBUTES = frozenset(
    {
        "gi_frame",
        "gi_code",
        "ag_frame",
        "ag_code",
        "cr_frame",
        "cr_code",
        "tb_frame",
        "tb_next",
        "f_back",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_code",
        "func_globals",
        "mro",
    }
)
MAX_CODE_CHARS = 20_000
MAX_AST_NODES = 5_000
CHILD = Path(__file__).with_name("sandbox_child.py")


class PolicyViolation(ValueError):
    """The code was refused before it ran. The message is safe to show to the model."""


class SandboxUnavailable(RuntimeError):
    """This platform has no ``resource`` module (POSIX only)."""


def check_policy(code: str) -> None:
    """Refuse code that steps outside the sandbox's static policy. Never runs it."""
    if len(code) > MAX_CODE_CHARS:
        raise PolicyViolation(f"the code is longer than {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PolicyViolation(f"syntax error on line {exc.lineno}: {exc.msg}") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise PolicyViolation("the code is too large")
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                raise PolicyViolation("relative imports are not allowed")
            _check_import(node.module)
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise PolicyViolation(f"the name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in FRAME_ATTRIBUTES
        ):
            raise PolicyViolation(f"the attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
            "__"
        ):
            raise PolicyViolation(f"defining {node.name!r} is not allowed")
        elif isinstance(node, ast.Global | ast.Nonlocal) and any(
            n.startswith("__") for n in node.names
        ):
            raise PolicyViolation("dunder globals are not allowed")


def _check_import(module: str) -> None:
    if module.split(".")[0] not in ALLOWED_IMPORTS:
        raise PolicyViolation(
            f"import {module!r} is not allowed; allowed: {', '.join(sorted(ALLOWED_IMPORTS))}"
        )


@dataclass(frozen=True, slots=True)
class TestOutcome:
    name: str
    ok: bool
    error: str = ""
    source: str = ""
    """The test's own code, e.g. ``assert f(2) == 4`` (trusted: written by the task)."""


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """What one run produced. ``passed`` is true only when every expected test passed."""

    tests: tuple[TestOutcome, ...] = ()
    error: str = ""
    """Why the run as a whole failed: policy, load error, timeout, crash."""
    blocked: tuple[str, ...] = ()
    """Audit events the hook refused (``socket.connect``, ``open('/etc/passwd')``...)."""
    stdout: str = ""
    seconds: float = 0.0
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.tests) and not self.error and all(t.ok for t in self.tests)

    @property
    def score(self) -> float:
        """The fraction of tests that passed (0 when the run failed as a whole)."""
        if not self.tests:
            return 0.0
        return sum(t.ok for t in self.tests) / len(self.tests)

    def feedback(self, limit: int = 5) -> str:
        """A short, model-facing summary of what failed."""
        if self.error:
            return self.error
        failed = [t for t in self.tests if not t.ok]
        if not failed:
            return f"all {len(self.tests)} tests passed"
        lines = [f"{len(failed)} of {len(self.tests)} tests failed:"]
        lines += [f"- {t.source or t.name} -> {t.error or 'failed'}" for t in failed[:limit]]
        return "\n".join(lines)


@dataclass(frozen=True)
class Sandbox:
    """Runs ``code`` + trusted ``tests`` (assert statements) in a hardened subprocess."""

    timeout: float = 5.0
    cpu_seconds: int = 3
    memory_bytes: int = 512 * 1024 * 1024
    open_files: int = 32
    enforce_policy: bool = True
    """Turn the static policy off only to test the runtime layers behind it."""
    python: str = field(default=sys.executable)

    def run(self, code: str, tests: list[str]) -> SandboxResult:
        if os.name != "posix":  # pragma: no cover - CI is Linux
            raise SandboxUnavailable("the sandbox needs POSIX resource limits")
        if not tests:
            raise ValueError("at least one test is needed")
        if self.enforce_policy:
            try:
                check_policy(code)
            except PolicyViolation as exc:
                return SandboxResult(error=f"refused by the sandbox policy: {exc}")
        names = [f"test_{i:02d}" for i in range(len(tests))]
        check = "".join(
            f"def {name}():\n" + "".join(f"    {line}\n" for line in test.splitlines()) + "\n"
            for name, test in zip(names, tests, strict=True)
        )
        nonce = secrets.token_hex(16)
        header = {
            "nonce": nonce,
            "limits": {
                "cpu_seconds": self.cpu_seconds,
                "memory_bytes": self.memory_bytes,
                "open_files": self.open_files,
            },
        }
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="reflection-sandbox-") as workdir:
            Path(workdir, "solution.py").write_text(code, "utf-8")
            Path(workdir, "check.py").write_text(check, "utf-8")
            stdout, timed_out, code_ = self._spawn(workdir, json.dumps(header))
        seconds = round(time.perf_counter() - started, 4)
        if timed_out:
            return SandboxResult(
                error=f"timed out after {self.timeout:g} s", seconds=seconds, timed_out=True
            )
        line = next((ln for ln in stdout.splitlines() if ln.startswith(nonce)), None)
        if line is None:
            why = "it was killed by a resource limit" if code_ < 0 else f"exit code {code_}"
            return SandboxResult(error=f"the sandbox produced no report ({why})", seconds=seconds)
        report = json.loads(line[len(nonce) :])
        by_name = {t["name"]: t for t in report["tests"] if t["name"] in names}
        outcomes = tuple(
            TestOutcome(n, bool(by_name[n]["ok"]), str(by_name[n]["error"]), source)
            if n in by_name
            else TestOutcome(n, False, "not run", source)
            for n, source in zip(names, tests, strict=True)
        )
        load_error = str(report.get("load_error", ""))
        return SandboxResult(
            tests=outcomes,
            error=f"the code failed to load: {load_error}" if load_error else "",
            blocked=tuple(str(b) for b in report.get("blocked", [])),
            stdout=str(report.get("stdout", "")),
            seconds=seconds,
        )

    def _spawn(self, workdir: str, header: str) -> tuple[str, bool, int]:
        argv = [self.python, "-I", "-S", "-B", str(CHILD), workdir]
        # A fixed argv (this interpreter + the bundled child), no shell, an empty env.
        proc = subprocess.Popen(  # nosec B603
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=workdir,
            env={},
            close_fds=True,
            start_new_session=True,
            text=True,
        )
        try:
            out, _ = proc.communicate(header + "\n", timeout=self.timeout)
            return out, False, proc.returncode
        except subprocess.TimeoutExpired:
            _kill(proc.pid)
            proc.communicate()
            return "", True, -9


def _kill(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
