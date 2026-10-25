"""Settings from the environment, the model factory, the CLI and the benchmark."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from reflection_agent import Settings
from reflection_agent import build_graph
from reflection_agent.bench import CASES
from reflection_agent.bench import markdown_table
from reflection_agent.bench import run_bench
from reflection_agent.cli import main
from reflection_agent.fake import FakeReflectionModel
from reflection_agent.llm import ModelConfigError
from reflection_agent.llm import is_loopback_url
from reflection_agent.llm import make_model
from tests.conftest import IDEMPOTENCY


def test_settings_from_env() -> None:
    s = Settings.from_env(
        {
            "REFLECTION_PROVIDER": "fake",
            "REFLECTION_THRESHOLD": "7.5",
            "REFLECTION_MAX_ITERATIONS": "6",
            "REFLECTION_CANDIDATES": "2",
            "REFLECTION_SANDBOX_TIMEOUT": "2",
        }
    )
    assert s.provider == "fake" and s.model == "fake-reflection"
    assert s.limits.threshold == 7.5 and s.limits.max_iterations == 6 and s.limits.candidates == 2
    assert s.sandbox_timeout == 2.0 and s.recursion_limit > 25
    assert Settings.from_env({}).model == "qwen3:4b"
    with pytest.raises(ValueError, match="REFLECTION_PROVIDER"):
        Settings.from_env({"REFLECTION_PROVIDER": "openai"})


def test_make_model_is_local_by_default() -> None:
    assert isinstance(make_model(Settings(provider="fake")), FakeReflectionModel)
    model = make_model(Settings(), json_mode=True, temperature=0.8)
    assert type(model).__name__ == "ChatOllama"
    assert is_loopback_url("http://127.0.0.1:11434") and is_loopback_url("localhost:1")
    assert not is_loopback_url("http://10.0.0.5:11434")
    with pytest.raises(ModelConfigError, match="not loopback"):
        make_model(Settings(ollama_host="http://example.org:11434"))
    remote = Settings(ollama_host="http://10.0.0.5:11434", allow_remote_ollama=True)
    assert type(make_model(remote)).__name__ == "ChatOllama"


def test_build_graph_validates_the_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        build_graph("debate", Settings(provider="fake"))


def _cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


def test_cli_runs_every_variant_offline() -> None:
    code, text = _cli("--provider", "fake", "--variant", "all", IDEMPOTENCY)
    assert code == 0
    assert "[reflect]" in text and "[reflexion]" in text and "[lats]" in text
    assert "status=passed score=8.6 iterations=3" in text
    assert "references: idempotency" in text
    assert "expansions=2 nodes=6" in text
    assert "critique       iteration=1 score=5.0" in text


def test_cli_code_task_with_persistent_memory(tmp_path: Path) -> None:
    db = str(tmp_path / "lessons.db")
    args = ("--provider", "fake", "--variant", "reflexion", "--code-task", "roman")
    code, text = _cli(*args, "--memory", db, "--json")
    first = json.loads(text)
    assert code == 0 and first["attempts"] == 3 and first["recalled"] == 0
    code, text = _cli(*args, "--memory", db, "--json")
    second = json.loads(text)
    assert second["attempts"] == 1 and second["recalled"] == 2 and second["status"] == "passed"


def test_cli_limits_and_quiet_output() -> None:
    code, text = _cli(
        "--provider", "fake", "--quiet", "--max-iterations", "2", "--threshold", "9", IDEMPOTENCY
    )
    assert code == 0 and "status=max_iterations" in text and "iteration=1" not in text


def test_cli_mermaid() -> None:
    code, text = _cli("--mermaid", "reflexion")
    assert code == 0 and "self_reflection" not in text and "reflect" in text and "graph TD" in text


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--code-task", "luhn", "also a task"],
        ["x" * 5000],
        ["--provider", "fake", "--threshold", "42", "task"],
    ],
)
def test_cli_rejects_bad_input(argv: list[str]) -> None:
    assert _cli(*argv)[0] == 2


def test_cli_reports_a_remote_ollama_as_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://example.org:11434")
    assert _cli("--provider", "ollama", "a task")[0] == 2


def test_bench_rows_and_table() -> None:
    import asyncio

    rows = asyncio.run(run_bench())
    assert len(rows) == len(CASES)
    by = {(r["task"], r["variant"]): r for r in rows}
    assert by[("roman (code)", "reflexion")]["attempts"] == 3
    assert by[("rle (code)", "reflexion")]["llm_calls"] == 2
    assert by[("vary header (text)", "reflexion")]["status"] == "max_attempts"
    table = markdown_table(rows, 8.0)
    assert "| roman (code) | reflexion | unit tests (sandbox) | 3 attempts | 33% tests |" in table
    assert "| rle (code) | reflexion | unit tests (sandbox) | 1 attempt |" in table
    assert "- (3 max)" in table


def test_committed_results_match_a_fresh_run() -> None:
    import asyncio

    committed = json.loads(Path("docs/benchmark/results.json").read_text())["rows"]
    fresh = asyncio.run(run_bench())
    keys = ("task", "variant", "status", "curve", "llm_calls")
    assert [{k: r[k] for k in keys} for r in committed] == [{k: r[k] for k in keys} for r in fresh]


def test_bench_main_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from reflection_agent.bench import main as bench_main

    target = tmp_path / "results.json"
    assert bench_main(["--json", str(target)]) == 0
    data = json.loads(target.read_text())
    assert data["meta"]["model"].startswith("FakeReflectionModel") and len(data["rows"]) == 9
    assert "| Task | Variant |" in capsys.readouterr().out
