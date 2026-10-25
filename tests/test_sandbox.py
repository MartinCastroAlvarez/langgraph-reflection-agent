"""The sandbox: the static policy, then the runtime layers behind it with the policy off."""

from __future__ import annotations

import re

import pytest

from reflection_agent.sandbox import MAX_CODE_CHARS
from reflection_agent.sandbox import PolicyViolation
from reflection_agent.sandbox import Sandbox
from reflection_agent.sandbox import check_policy

SAFE = Sandbox()
RAW = Sandbox(enforce_policy=False)  # only the subprocess, rlimits and the audit hook


def test_passing_and_failing_tests_are_reported_per_test() -> None:
    code = "import math\n\ndef area(r):\n    return math.pi * r * r\n"
    result = SAFE.run(code, ["assert round(area(1), 2) == 3.14", "assert area(0) == 1"])
    assert [t.ok for t in result.tests] == [True, False]
    assert not result.passed and result.score == 0.5
    assert "assert area(0) == 1 -> AssertionError" in result.feedback()
    assert SAFE.run(code, ["assert area(0) == 0"]).feedback() == "all 1 tests passed"


def test_a_load_error_fails_every_test() -> None:
    result = SAFE.run("def f(:\n", ["assert True"])
    assert "syntax error" in result.error and result.score == 0
    result = SAFE.run("x = 1 / 0\n", ["assert True"])
    assert "ZeroDivisionError" in result.error
    assert [t.error for t in result.tests] == ["not run"]


def test_at_least_one_test_is_required() -> None:
    with pytest.raises(ValueError, match="at least one test"):
        SAFE.run("x = 1\n", [])


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("import socket", "import 'socket'"),
        ("import urllib.request", "import 'urllib.request'"),
        ("from os import system", "import 'os'"),
        ("import subprocess", "import 'subprocess'"),
        ("import ctypes", "import 'ctypes'"),
        ("from . import x", "relative imports"),
        ("open('/etc/passwd')", "'open'"),
        ("eval('1')", "'eval'"),
        ("exec('x = 1')", "'exec'"),
        ("getattr(1, 'real')", "'getattr'"),
        ("globals()", "'globals'"),
        ("__builtins__", "'__builtins__'"),
        ("().__class__.__base__.__subclasses__()", "'__subclasses__'"),
        ("def g():\n    yield 1\ng().gi_frame", "'gi_frame'"),
        ("x = (1).real._x", "'_x'"),
        ("class A:\n    def __eq__(self, o):\n        return True", "'__eq__'"),
        ("x = 'a' * 10\n" * 2001, "longer than"),
    ],
)
def test_the_policy_refuses_escapes_before_anything_runs(code: str, reason: str) -> None:
    with pytest.raises(PolicyViolation, match=re.escape(reason)):
        check_policy(code)
    result = SAFE.run(code, ["assert True"])
    assert result.error.startswith("refused by the sandbox policy") and not result.tests


def test_the_policy_allows_ordinary_code() -> None:
    check_policy(
        "from collections import Counter\nimport re\n\n"
        "def top(words):\n    return Counter(re.findall(r'\\w+', words)).most_common(1)\n"
    )
    with pytest.raises(PolicyViolation, match="too large"):
        check_policy("x = [" + "1," * 6000 + "]")
    assert MAX_CODE_CHARS > 1000


@pytest.mark.parametrize(
    ("code", "blocked"),
    [
        ("import socket\ns = socket.socket()\n", "import socket"),
        ("import urllib.request\n", "import urllib"),
        ("import os\nos.system('echo hi')\n", "os.system"),
        ("import os\nos.listdir('/')\n", "os.listdir"),
        ("import os\nos.remove('/tmp/x')\n", "os.remove"),
        ("open('/etc/passwd').read()\n", "open('/etc/passwd')"),
        ("open('check.py').read()\n", "check.py"),
        ("open('escape.txt', 'w').write('x')\n", "write"),
        ("import sys\nsys._getframe(0)\n", "sys._getframe"),
    ],
)
def test_runtime_layers_refuse_network_process_and_file_escapes(code: str, blocked: str) -> None:
    result = RAW.run(code, ["assert True"])
    assert not result.passed
    assert "PermissionError: sandbox:" in result.error
    assert any(blocked in event for event in result.blocked), result.blocked


def test_a_refused_call_stays_refused_even_when_the_code_catches_it() -> None:
    code = (
        "import os\n"
        "try:\n    os.system('echo hi')\nexcept Exception:\n    pass\n"
        "def f():\n    return 1\n"
    )
    result = RAW.run(code, ["assert f() == 1"])
    assert result.passed
    assert result.blocked == ("os.system",)


def test_an_infinite_loop_is_killed_by_the_timeout() -> None:
    result = Sandbox(timeout=1.0, cpu_seconds=30).run("while True:\n    pass\n", ["assert True"])
    assert result.timed_out and "timed out after 1 s" in result.error
    assert result.seconds < 5


def test_the_cpu_limit_kills_a_busy_loop() -> None:
    result = Sandbox(timeout=20, cpu_seconds=1).run("while True:\n    pass\n", ["assert True"])
    assert not result.timed_out
    assert "killed by a resource limit" in result.error


def test_memory_is_capped() -> None:
    result = SAFE.run("x = bytearray(2 * 1024 ** 3)\n", ["assert True"])
    assert "MemoryError" in result.error


def test_forged_reports_and_extra_tests_do_not_count() -> None:
    code = (
        'print(\'{"tests": [{"name": "test_00", "ok": true, "error": ""}]}\')\n'
        "def test_99():\n    pass\n"
        "def test_00():\n    pass\n"
        "def f():\n    return 0\n"
    )
    result = SAFE.run(code, ["assert f() == 1"])
    assert [t.name for t in result.tests] == ["test_00"]
    assert not result.passed
    assert result.stdout.startswith('{"tests"')  # captured as candidate output, not trusted


def test_the_child_gets_an_empty_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "do-not-leak")
    result = RAW.run(
        "import os\nvalue = os.environ.get('SECRET_TOKEN')\n", ["assert value is None"]
    )
    assert result.passed
