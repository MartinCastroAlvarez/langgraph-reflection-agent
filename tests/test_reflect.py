"""Basic reflection: the loop stops at the threshold, at the cap, and keeps the best draft."""

from __future__ import annotations

from reflection_agent.reflect import Round
from reflection_agent.reflect import best_round
from reflection_agent.reflect import build_reflect
from tests.conftest import GENERIC
from tests.conftest import IDEMPOTENCY
from tests.conftest import VARY
from tests.conftest import critique
from tests.conftest import fake
from tests.conftest import make_settings
from tests.conftest import run


def test_terminates_as_soon_as_the_score_clears_the_threshold() -> None:
    model = fake()
    final = run(build_reflect(model, make_settings()), {"task": IDEMPOTENCY})
    assert final["status"] == "passed"
    assert [r["score"] for r in final["rounds"]] == [5.0, 7.0, 8.6]
    assert final["iteration"] == 3 and final["best_iteration"] == 3
    assert model.calls == ["generator", "reflector"] * 3
    assert final["answer"].startswith("An idempotency key is a unique value")


def test_a_lower_threshold_stops_earlier() -> None:
    final = run(build_reflect(fake(), make_settings(threshold=7.0)), {"task": IDEMPOTENCY})
    assert final["status"] == "passed" and final["iteration"] == 2


def test_max_iterations_caps_the_loop() -> None:
    model = fake()
    final = run(build_reflect(model, make_settings(max_iterations=2)), {"task": VARY})
    assert final["status"] == "max_iterations"
    assert final["iteration"] == 2
    assert len(model.calls) == 4
    assert final["score"] == 7.1


def test_an_unreachable_threshold_still_ends_at_the_cap() -> None:
    final = run(
        build_reflect(fake(), make_settings(threshold=10.0, max_iterations=4)), {"task": GENERIC}
    )
    assert final["status"] == "max_iterations"
    assert [r["score"] for r in final["rounds"]] == [4.0, 5.5, 7.0, 8.5]


def test_the_best_draft_wins_even_when_a_later_one_is_worse() -> None:
    model = fake(
        generator=["first", "second", "third"],
        reflector=[critique(5.0), critique(7.5), critique(6.0)],
    )
    final = run(build_reflect(model, make_settings(max_iterations=3)), {"task": GENERIC})
    assert final["status"] == "max_iterations"
    assert final["answer"] == "second" and final["best_iteration"] == 2


def test_the_critique_reaches_the_next_draft() -> None:
    model = fake(reflector=[critique(3.0, issue="Name the header.")])
    run(build_reflect(model, make_settings(max_iterations=2)), {"task": GENERIC})
    second = model.prompts("generator")[1]
    assert "Name the header." in second
    assert '<data name="draft">' in second and '<data name="iteration">\n2' in second


def test_an_unparseable_critique_scores_zero_and_the_loop_goes_on() -> None:
    model = fake(reflector=["looks great!", critique(9.0)])
    final = run(build_reflect(model, make_settings()), {"task": GENERIC})
    assert [r["score"] for r in final["rounds"]] == [0.0, 9.0]
    assert "could not be parsed" in final["rounds"][0]["issues"][0]
    assert final["status"] == "passed"


def test_the_critic_cannot_pass_a_draft_by_claiming_an_overall_score() -> None:
    reply = '{"overall": 10, "scores": {"accuracy": 2, "completeness": 2, "clarity": 2, "concision": 2}}'
    final = run(
        build_reflect(fake(reflector=[reply]), make_settings(max_iterations=1)), {"task": GENERIC}
    )
    assert final["score"] == 2.0 and final["status"] == "max_iterations"


def test_best_round_prefers_the_later_draft_on_a_tie() -> None:
    rounds: list[Round] = [
        {"iteration": 1, "draft": "a", "score": 8.0, "scores": {}, "issues": [], "suggestions": []},
        {"iteration": 2, "draft": "b", "score": 8.0, "scores": {}, "issues": [], "suggestions": []},
    ]
    assert best_round(rounds)["draft"] == "b"
