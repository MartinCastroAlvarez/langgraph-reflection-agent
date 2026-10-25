"""Package-wide guards: generated code runs only in the sandbox; no silenced findings."""

from __future__ import annotations

import ast
from pathlib import Path

import reflection_agent

PACKAGE = Path(reflection_agent.__file__).parent
SOURCES = sorted(PACKAGE.rglob("*.py"))
BANNED_CALLS = {"eval", "exec", "compile", "__import__"}
BANNED_MODULES = {"pickle", "marshal", "shelve", "yaml", "requests", "langsmith", "ctypes"}
# The only modules allowed to start a process or run a file of code, and why.
ALLOWED = {"subprocess": {"sandbox.py"}, "runpy": {"sandbox_child.py"}}


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_dynamic_execution_anywhere_and_processes_only_in_the_sandbox() -> None:
    assert SOURCES
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in BANNED_CALLS, f"{path.name}: {node.func.id}()"
            if isinstance(node, ast.keyword) and node.arg == "shell":
                raise AssertionError(f"{path.name}: shell=")
        imported = _imports(tree)
        assert not imported & BANNED_MODULES, f"{path.name}: {imported & BANNED_MODULES}"
        for module, files in ALLOWED.items():
            if module in imported:
                assert path.name in files, f"{path.name} imports {module}"


def test_the_sandbox_child_is_never_imported_by_the_package() -> None:
    for path in SOURCES:
        assert "sandbox_child" not in _imports(ast.parse(path.read_text("utf-8"))) | {
            n.module or ""
            for n in ast.walk(ast.parse(path.read_text("utf-8")))
            if isinstance(n, ast.ImportFrom)
        } - {"reflection_agent"}, path.name
    assert (PACKAGE / "sandbox_child.py").exists()


def test_silenced_findings_are_only_the_two_documented_ones() -> None:
    found = []
    for path in SOURCES:
        for line in path.read_text(encoding="utf-8").splitlines():
            for marker in ("# noqa", "# type: ignore", "# nosec"):
                if marker in line:
                    found.append((path.name, line.split("#", 1)[1].strip()))
    assert sorted(found) == [("sandbox.py", "nosec B404"), ("sandbox.py", "nosec B603")]


def test_bundled_corpus_ships_with_the_package() -> None:
    files = sorted(p.name for p in (PACKAGE / "data" / "corpus").iterdir())
    assert files == [
        "caching.md",
        "idempotency.md",
        "luhn.md",
        "reflection.md",
        "roman-numerals.md",
        "run-length-encoding.md",
    ]
