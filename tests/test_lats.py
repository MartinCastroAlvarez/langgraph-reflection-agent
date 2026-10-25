"""LATS-lite: expansion breadth, backpropagation, UCT selection, best-node choice, bounds."""

from __future__ import annotations

import math

import pytest

from reflection_agent.config import Limits
from reflection_agent.lats import TreeNode
from reflection_agent.lats import backpropagate
from reflection_agent.lats import best_node
from reflection_agent.lats import build_lats
from reflection_agent.lats import expandable
from reflection_agent.lats import select
from reflection_agent.lats import uct
from tests.conftest import GENERIC
from tests.conftest import IDEMPOTENCY
from tests.conftest import critique
from tests.conftest import fake
from tests.conftest import make_settings
from tests.conftest import run


def node(id: int, parent: int | None, depth: int, score: float | None = None) -> TreeNode:
    return {
        "id": id,
        "parent": parent,
        "depth": depth,
        "answer": f"n{id}",
        "score": score,
        "issues": [],
        "suggestions": [],
        "visits": 0,
        "value": 0.0,
    }


def test_passes_when_a_child_clears_the_threshold() -> None:
    model = fake()
    final = run(build_lats(model, make_settings()), {"task": IDEMPOTENCY})
    assert final["status"] == "passed"
    assert final["expansions"] == 2
    assert len(final["tree"]) == 1 + 2 * 3
    assert model.calls.count("expander") == 6 and model.calls.count("reflector") == 6
    assert final["best_id"] == 5 and final["score"] == 8.7
    # round 2 expanded the best child of round 1 (node 3, score 7.0)
    assert final["rounds"][1]["expanded"] == 3
    assert [r["best_so_far"] for r in final["rounds"]] == [7.0, 8.7]


def test_tree_search_picks_the_highest_value_node_not_the_last_one() -> None:
    model = fake(
        reflector=[
            critique(4.0),
            critique(7.5),
            critique(5.0),  # round 1: node 2 is the best
            critique(6.0),
            critique(6.5),
            critique(5.5),  # round 2: all worse than node 2
        ]
    )
    final = run(
        build_lats(model, make_settings(threshold=9.5, max_expansions=2)), {"task": GENERIC}
    )
    assert final["status"] == "max_expansions"
    assert final["best_id"] == 2 and final["score"] == 7.5
    assert final["answer"] == "Candidate 1.2: " + GENERIC


def test_breadth_is_the_number_of_candidates() -> None:
    model = fake()
    final = run(
        build_lats(model, make_settings(candidates=5, max_expansions=1, threshold=10)),
        {"task": GENERIC},
    )
    assert len(final["tree"]) == 6
    assert final["tree"][0]["visits"] == 5
    assert {n["depth"] for n in final["tree"][1:]} == {1}


def test_max_depth_one_exhausts_the_tree_after_one_expansion() -> None:
    final = run(
        build_lats(fake(), make_settings(max_depth=1, threshold=10, max_expansions=5)),
        {"task": GENERIC},
    )
    assert final["status"] == "exhausted"
    assert final["expansions"] == 1


def test_expansions_cap_the_search() -> None:
    model = fake()
    final = run(build_lats(model, make_settings(threshold=10, max_expansions=3)), {"task": GENERIC})
    assert final["status"] == "max_expansions" and final["expansions"] == 3
    assert model.calls.count("expander") == 9


def test_children_see_the_path_that_led_to_them() -> None:
    model = fake()
    run(build_lats(model, make_settings()), {"task": IDEMPOTENCY})
    first, later = model.prompts("expander")[0], model.prompts("expander")[3]
    assert '<data name="path">\n[]' in first
    assert "Send a UUID header" not in later  # the path holds node 3 (draft 1), not node 2
    assert "unique value the client sends" in later and '<data name="depth">\n2' in later


def test_backpropagate_updates_every_ancestor() -> None:
    tree = [node(0, None, 0), node(1, 0, 1), node(2, 1, 2), node(3, 0, 1)]
    backpropagate(tree, 2, 0.8)
    backpropagate(tree, 3, 0.2)
    assert [(n["visits"], n["value"]) for n in tree] == [(2, 1.0), (1, 0.8), (1, 0.8), (1, 0.2)]


def test_uct_balances_value_and_exploration() -> None:
    a = {**node(1, 0, 1), "visits": 1, "value": 0.9}
    b = {**node(2, 0, 1), "visits": 10, "value": 9.5}
    assert uct(node(3, 0, 1), 5, 1.0) == math.inf
    assert uct(a, 11, 0.0) < uct(b, 11, 0.0)  # pure exploitation: b's mean is higher
    assert uct(a, 11, 1.0) > uct(b, 11, 1.0)  # exploration favours the rarely visited a


def test_select_descends_by_uct_and_skips_exhausted_subtrees() -> None:
    limits = Limits(max_depth=2, exploration=0.0)
    tree = [node(0, None, 0), node(1, 0, 1), node(2, 0, 1)]
    backpropagate(tree, 1, 0.9)
    backpropagate(tree, 2, 0.3)
    chosen = select(tree, limits)
    assert chosen is not None and chosen["id"] == 1
    tree += [node(3, 1, 2), node(4, 1, 2)]  # node 1's children sit at max depth
    backpropagate(tree, 3, 0.9)
    backpropagate(tree, 4, 0.9)
    assert not expandable(tree, tree[1], 2)
    chosen = select(tree, limits)
    assert chosen is not None and chosen["id"] == 2
    tree.append(node(5, 2, 2))
    assert select(tree, limits) is None


def test_best_node_breaks_ties_by_mean_value() -> None:
    tree = [node(0, None, 0), node(1, 0, 1, 8.0), node(2, 0, 1, 8.0)]
    tree[1].update(visits=2, value=1.0)
    tree[2].update(visits=1, value=0.8)
    assert best_node(tree)["id"] == 2


def test_limits_are_validated() -> None:
    with pytest.raises(ValueError, match="candidates"):
        Limits(candidates=0)
    with pytest.raises(ValueError, match="threshold"):
        Limits(threshold=11)
    with pytest.raises(ValueError, match="max_queries"):
        Limits(max_queries=11)
    with pytest.raises(ValueError, match="exploration"):
        Limits(exploration=-1)
