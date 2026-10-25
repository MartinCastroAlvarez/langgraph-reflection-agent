"""The program that runs *inside* the sandbox process. Never import it from the package.

The parent starts it as ``python -I -S -B sandbox_child.py <workdir>`` with an empty
environment, the temporary ``workdir`` as its current directory and a JSON header on
stdin (``{"nonce": ..., "limits": {...}}``). In this order it:

1. applies the ``resource`` limits (CPU seconds, address space, file size 0, no new
   processes, few file descriptors, no core dumps);
2. pre-imports the modules candidate code may import, so no import later needs the
   filesystem;
3. installs an audit hook (PEP 578) that refuses network, process, filesystem, frame
   and native-code events, and every import that isn't on the allowlist;
4. runs ``solution.py`` (the candidate) and then ``check.py`` (the trusted tests) with
   ``runpy``, with the candidate's stdout captured in memory;
5. calls each ``test_*`` function and prints one line, ``<nonce><json report>``, on the
   real stdout. Candidate code can't import ``sys``/``os``/``io``, so it can't reach that
   stream, and it never sees the nonce.

The parent also enforces a wall-clock timeout and kills the process group. Only the
standard library is used, and nothing here trusts the code it runs.
"""

import contextlib
import importlib
import io
import json
import os
import resource
import runpy
import sys
from collections.abc import Callable
from typing import Any

ALLOWED_IMPORTS = (
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
)

_BLOCKED_PREFIXES = (
    "socket.",
    "subprocess.",
    "os.",
    "shutil.",
    "ctypes.",
    "urllib.",
    "http.",
    "ftplib.",
    "smtplib.",
    "imaplib.",
    "poplib.",
    "nntplib.",
    "telnetlib.",
    "webbrowser.",
    "pty.",
    "fcntl.",
    "resource.",
    "signal.",
    "mmap.",
    "sqlite3.",
    "marshal.",
    "pickle.",
    "winreg.",
    "msvcrt.",
    "_winapi.",
)
_BLOCKED_EVENTS = frozenset(
    {
        "sys._getframe",
        "sys._current_frames",
        "sys.settrace",
        "sys.setprofile",
        "sys.addaudithook",
        "sys.monitoring.register_callback",
        "gc.get_objects",
        "gc.get_referrers",
        "gc.get_referents",
        "code.__new__",
        "function.__new__",
        "builtins.input",
        "builtins.breakpoint",
        "cpython.run_command",
        "cpython.run_file",
        "cpython.run_module",
        "cpython.run_startup",
        "cpython.run_interactivehook",
        "cpython.remote_debugger_script",
    }
)
_OUTPUT_LIMIT = 4000
# runpy imports these lazily; load them before the hook so it can refuse every other import.
_RUNPY_IMPORTS = ("pkgutil", "importlib.machinery", "importlib.util", "zipimport")


def _limits(spec: dict[str, int]) -> None:
    def cap(name: str, value: int) -> None:
        limit = getattr(resource, name, None)
        if limit is None:
            return
        _soft, hard = resource.getrlimit(limit)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(limit, (value, value))

    cap("RLIMIT_CPU", spec["cpu_seconds"])
    cap("RLIMIT_AS", spec["memory_bytes"])
    cap("RLIMIT_FSIZE", 0)
    cap("RLIMIT_NPROC", 0)
    cap("RLIMIT_CORE", 0)
    cap("RLIMIT_NOFILE", spec["open_files"])


def _hook(
    workdir: str, readable: set[str], blocked: list[str]
) -> Callable[[str, tuple[Any, ...]], None]:
    def refuse(event: str) -> None:
        if len(blocked) < 50:
            blocked.append(event)
        raise PermissionError(f"sandbox: {event} is not allowed")

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event == "open":
            path, mode = args[0], args[1]
            if isinstance(path, int):
                refuse("open(fd)")
            real = os.path.realpath(os.fsdecode(path))
            writing = isinstance(mode, str) and any(c in mode for c in "wax+")
            if writing or (isinstance(mode, int) and mode & (os.O_WRONLY | os.O_RDWR)):
                refuse(f"open({os.path.basename(real)!r}, write)")
            if real not in readable:
                refuse(f"open({real!r})")
            return
        if event == "import":
            name = str(args[0]).split(".")[0]
            if name not in ALLOWED_IMPORTS:
                refuse(f"import {args[0]}")
            return
        if event in {"os.listdir", "os.scandir"}:
            target = args[0] if args and args[0] is not None else "."
            if isinstance(target, str | bytes):
                real = os.path.realpath(os.fsdecode(target))
                if real == workdir or real.startswith(workdir + os.sep):
                    return
            refuse(event)
        if event in _BLOCKED_EVENTS or event.startswith(_BLOCKED_PREFIXES):
            refuse(event)

    return hook


def main() -> None:
    workdir = os.path.realpath(sys.argv[1])
    header = json.loads(sys.stdin.readline())
    sys.stdin.close()
    nonce = str(header["nonce"])
    _limits(header["limits"])
    for name in (*ALLOWED_IMPORTS, *_RUNPY_IMPORTS):
        importlib.import_module(name)
    real_stdout = sys.stdout
    solution = os.path.join(workdir, "solution.py")
    check = os.path.join(workdir, "check.py")
    blocked: list[str] = []
    report: dict[str, object] = {"tests": [], "load_error": "", "stdout": "", "blocked": blocked}
    captured = io.StringIO()
    # Each phase may read only its own file: the candidate's top-level code can't read the
    # tests, and once both are loaded nothing is readable at all.
    readable = {solution}
    sys.addaudithook(_hook(workdir, readable, blocked))
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        try:
            namespace = runpy.run_path(solution, run_name="solution")
            readable.clear()
            readable.add(check)
            checks = runpy.run_path(check, init_globals=namespace, run_name="check")
            readable.clear()
        except BaseException as exc:  # the candidate failed to load: every test fails
            readable.clear()
            report["load_error"] = f"{type(exc).__name__}: {exc}"[:500]
            checks = {}
        results = []
        for name in sorted(k for k in checks if k.startswith("test_")):
            try:
                checks[name]()
                results.append({"name": name, "ok": True, "error": ""})
            except BaseException as exc:
                text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                results.append({"name": name, "ok": False, "error": text[:300]})
        report["tests"] = results
    report["stdout"] = captured.getvalue()[:_OUTPUT_LIMIT]
    real_stdout.write(nonce + json.dumps(report) + "\n")
    real_stdout.flush()


if __name__ == "__main__":
    main()
