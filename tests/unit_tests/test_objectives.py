"""Tests for opto.trainer.objectives — ObjectiveConfig and selection utilities."""
import pytest
import numpy as np
from opto.trainer.objectives import (
    ObjectiveConfig, to_score_dict, apply_minimize, weighted_scalarize,
    dominates, pareto_rank, select_best, select_top_k,
    score_dict_to_scalar, to_scalar_score, aggregate_score_dicts,
)


# ---------------------------------------------------------------------------
# to_score_dict (alias normalize_score kept for backwards-compat)
# ---------------------------------------------------------------------------

def test_to_score_dict_float():
    assert to_score_dict(0.85) == {"score": 0.85}


def test_to_score_dict_zero():
    assert to_score_dict(0.0) == {"score": 0.0}


def test_to_score_dict_int():
    assert to_score_dict(1) == {"score": 1.0}


def test_to_score_dict_int_zero():
    assert to_score_dict(0) == {"score": 0.0}


def test_to_score_dict_bool_true():
    assert to_score_dict(True) == {"score": 1.0}


def test_to_score_dict_bool_false():
    assert to_score_dict(False) == {"score": 0.0}


def test_to_score_dict_dict():
    result = to_score_dict({"acc": 0.9, "lat": 50.0})
    assert result == {"acc": 0.9, "lat": 50.0}


def test_to_score_dict_dict_with_int_values():
    result = to_score_dict({"acc": 1, "lat": 0})
    assert result == {"acc": 1.0, "lat": 0.0}


def test_to_score_dict_empty_dict_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        to_score_dict({})


def test_to_score_dict_nan_raises():
    with pytest.raises(ValueError, match="finite"):
        to_score_dict({"a": float("nan")})


def test_to_score_dict_inf_raises():
    with pytest.raises(ValueError, match="finite"):
        to_score_dict(float("inf"))


def test_to_score_dict_neg_inf_raises():
    with pytest.raises(ValueError, match="finite"):
        to_score_dict(float("-inf"))


def test_to_score_dict_string_raises():
    with pytest.raises(TypeError, match="str"):
        to_score_dict("bad")


def test_to_score_dict_none_raises():
    with pytest.raises(TypeError):
        to_score_dict(None)


def test_backward_compat_alias():
    """normalize_score still works as alias."""
    from opto.trainer.objectives import normalize_score
    assert normalize_score(0.5) == {"score": 0.5}


# ---------------------------------------------------------------------------
# score_dict_to_scalar / to_scalar_score
# ---------------------------------------------------------------------------

def test_score_dict_to_scalar_score_key():
    config = ObjectiveConfig(scalarize_dict="score")
    assert score_dict_to_scalar({"score": 0.9, "extra": 0.1}, config) == pytest.approx(0.9)


def test_score_dict_to_scalar_score_key_missing_raises():
    config = ObjectiveConfig(scalarize_dict="score")
    with pytest.raises(ValueError, match="missing key"):
        score_dict_to_scalar({"acc": 0.9}, config)


def test_score_dict_to_scalar_mean():
    config = ObjectiveConfig(scalarize_dict="mean")
    result = score_dict_to_scalar({"a": 0.8, "b": 0.2}, config)
    assert result == pytest.approx(0.5)


def test_score_dict_to_scalar_weighted():
    config = ObjectiveConfig(scalarize_dict="weighted", weights={"a": 0.7, "b": 0.3})
    result = score_dict_to_scalar({"a": 1.0, "b": 0.5}, config)
    assert result == pytest.approx(0.7 * 1.0 + 0.3 * 0.5)


def test_to_scalar_score_float_passthrough():
    assert to_scalar_score(0.75, None) == pytest.approx(0.75)


def test_to_scalar_score_dict_none_config_raises():
    with pytest.raises(ValueError, match="ObjectiveConfig is None"):
        to_scalar_score({"a": 0.5}, None)


def test_to_scalar_score_dict_with_config():
    config = ObjectiveConfig(scalarize_dict="mean")
    assert to_scalar_score({"a": 0.8, "b": 0.2}, config) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# aggregate_score_dicts
# ---------------------------------------------------------------------------

def test_aggregate_score_dicts_basic():
    result = aggregate_score_dicts([{"a": 1.0, "b": 0.5}, {"a": 0.0, "b": 1.0}])
    assert result["a"] == pytest.approx(0.5)
    assert result["b"] == pytest.approx(0.75)


def test_aggregate_score_dicts_empty():
    assert aggregate_score_dicts([]) == {}


# ---------------------------------------------------------------------------
# apply_minimize
# ---------------------------------------------------------------------------

def test_apply_minimize_negates():
    result = apply_minimize({"acc": 0.9, "lat": 100.0}, frozenset({"lat"}))
    assert result == {"acc": 0.9, "lat": -100.0}


def test_apply_minimize_empty_set():
    result = apply_minimize({"acc": 0.9, "lat": 100.0}, frozenset())
    assert result == {"acc": 0.9, "lat": 100.0}


def test_apply_minimize_all():
    result = apply_minimize({"a": 1.0, "b": 2.0}, frozenset({"a", "b"}))
    assert result == {"a": -1.0, "b": -2.0}


# ---------------------------------------------------------------------------
# weighted_scalarize
# ---------------------------------------------------------------------------

def test_weighted_scalarize_basic():
    result = weighted_scalarize({"a": 0.8, "b": 0.2}, {"a": 0.7, "b": 0.3})
    assert result == pytest.approx(0.7 * 0.8 + 0.3 * 0.2)


def test_weighted_scalarize_empty_weights():
    result = weighted_scalarize({"a": 1.0, "b": 2.0}, {})
    assert result == pytest.approx(3.0)  # equal weight 1.0 each


def test_weighted_scalarize_missing_metric():
    result = weighted_scalarize({"a": 1.0}, {"a": 0.5, "b": 0.5}, missing_value=0.0)
    assert result == pytest.approx(0.5)  # 0.5*1.0 + 0.5*0.0


def test_weighted_scalarize_ignores_extra_metrics():
    result = weighted_scalarize({"a": 1.0, "b": 2.0, "c": 99.0}, {"a": 1.0})
    assert result == pytest.approx(1.0)  # only "a" is weighted


# ---------------------------------------------------------------------------
# dominates
# ---------------------------------------------------------------------------

def test_dominates_yes():
    assert dominates({"a": 2.0, "b": 2.0}, {"a": 1.0, "b": 1.0}) is True


def test_dominates_yes_one_equal():
    assert dominates({"a": 2.0, "b": 1.0}, {"a": 1.0, "b": 1.0}) is True


def test_dominates_no_equal():
    assert dominates({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0}) is False


def test_dominates_no_tradeoff():
    assert dominates({"a": 2.0, "b": 0.5}, {"a": 1.0, "b": 1.0}) is False


def test_dominates_with_metric_subset():
    assert dominates({"a": 2.0, "b": 0.5}, {"a": 1.0, "b": 1.0},
                      metrics=("a",)) is True


# ---------------------------------------------------------------------------
# pareto_rank
# ---------------------------------------------------------------------------

def test_pareto_rank_clear_hierarchy():
    candidates = [
        {"a": 3.0, "b": 3.0},  # dominates everything -> rank 0
        {"a": 2.0, "b": 2.0},  # dominated by [0] -> rank 1
        {"a": 1.0, "b": 1.0},  # dominated by [0],[1] -> rank 2
    ]
    ranks = pareto_rank(candidates)
    assert ranks == [0, 1, 2]


def test_pareto_rank_all_nondominated():
    candidates = [
        {"a": 3.0, "b": 1.0},
        {"a": 1.0, "b": 3.0},
        {"a": 2.0, "b": 2.0},
    ]
    ranks = pareto_rank(candidates)
    # All are tradeoffs — none dominates another
    assert ranks == [0, 0, 0]


def test_pareto_rank_mixed():
    candidates = [
        {"a": 3.0, "b": 1.0},  # front 0
        {"a": 1.0, "b": 3.0},  # front 0
        {"a": 0.5, "b": 0.5},  # dominated by both -> rank 1
    ]
    ranks = pareto_rank(candidates)
    assert ranks[0] == 0
    assert ranks[1] == 0
    assert ranks[2] == 1


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------

def test_select_best_none_config():
    candidates = [(0.5, "A"), (0.9, "B"), (0.7, "C")]
    assert select_best(candidates, None) == 1


def test_select_best_scalar_mode():
    config = ObjectiveConfig(mode="scalar")
    candidates = [(0.5, "A"), (0.9, "B"), (0.7, "C")]
    assert select_best(candidates, config) == 1


def test_select_best_scalar_with_dict_scores_requires_config():
    """Dict scores require explicit scalarization config (no hidden hard-coded mean)."""
    candidates = [({"a": 0.5, "b": 0.3}, "X")]
    with pytest.raises(ValueError, match="ObjectiveConfig is None"):
        select_best(candidates, None)


def test_select_best_scalar_with_dict_scores_score_key_default():
    """Default scalarize_dict='score' uses the 'score' key."""
    config = ObjectiveConfig(mode="scalar")  # scalarize_dict="score" by default
    candidates = [
        ({"score": 0.4, "a": 0.5}, "X"),
        ({"score": 0.7, "a": 0.8}, "Y"),
    ]
    assert select_best(candidates, config) == 1


def test_select_best_scalar_with_dict_scores_mean_configurable():
    """Explicit scalarize_dict='mean' uses mean of all values."""
    config = ObjectiveConfig(mode="scalar", scalarize_dict="mean")
    candidates = [
        ({"a": 0.5, "b": 0.3}, "X"),  # mean 0.4
        ({"a": 0.8, "b": 0.6}, "Y"),  # mean 0.7
    ]
    assert select_best(candidates, config) == 1


def test_select_best_weighted():
    config = ObjectiveConfig(
        mode="weighted",
        weights={"accuracy": 0.8, "latency_s": 0.2},
        minimize=frozenset({"latency_s"}),
    )
    candidates = [
        ({"accuracy": 0.95, "latency_s": 0.200}, "A"),  # 0.8*0.95 + 0.2*(-0.2) = 0.72
        ({"accuracy": 0.70, "latency_s": 0.030}, "B"),  # 0.8*0.70 + 0.2*(-0.03) = 0.554
    ]
    assert select_best(candidates, config) == 0


def test_select_best_weighted_latency_heavy():
    config = ObjectiveConfig(
        mode="weighted",
        weights={"accuracy": 0.2, "latency_s": 0.8},
        minimize=frozenset({"latency_s"}),
    )
    candidates = [
        ({"accuracy": 0.95, "latency_s": 0.200}, "A"),  # 0.2*0.95 + 0.8*(-0.2) = 0.03
        ({"accuracy": 0.70, "latency_s": 0.030}, "B"),  # 0.2*0.70 + 0.8*(-0.03) = 0.116
    ]
    assert select_best(candidates, config) == 1


def test_select_best_pareto_tiebreak_weighted():
    config = ObjectiveConfig(
        mode="pareto",
        weights={"a": 0.5, "b": 0.5},
        tie_break="weighted",
    )
    candidates = [
        ({"a": 0.9, "b": 0.1}, "X"),  # front 0, weighted = 0.5
        ({"a": 0.1, "b": 0.9}, "Y"),  # front 0, weighted = 0.5
        ({"a": 0.6, "b": 0.6}, "Z"),  # front 0, weighted = 0.6 -> winner
    ]
    assert select_best(candidates, config) == 2


def test_select_best_pareto_deterministic():
    config = ObjectiveConfig(
        mode="pareto",
        weights={"a": 0.5, "b": 0.5},
        tie_break="weighted",
        seed=42,
    )
    candidates = [
        ({"a": 0.9, "b": 0.1}, "X"),
        ({"a": 0.1, "b": 0.9}, "Y"),
    ]
    results = [select_best(candidates, config) for _ in range(10)]
    assert len(set(results)) == 1  # same result every time


def test_select_best_pareto_random_seeded_deterministic():
    config = ObjectiveConfig(
        mode="pareto",
        tie_break="random_seeded",
        seed=42,
    )
    candidates = [
        ({"a": 0.9, "b": 0.1}, "X"),
        ({"a": 0.1, "b": 0.9}, "Y"),
    ]
    results = [select_best(candidates, config) for _ in range(20)]
    assert len(set(results)) == 1


def test_select_best_pareto_different_seeds_may_differ():
    results = set()
    for seed in range(50):
        config = ObjectiveConfig(
            mode="pareto",
            tie_break="random_seeded",
            seed=seed,
        )
        candidates = [
            ({"a": 0.9, "b": 0.1}, "X"),
            ({"a": 0.1, "b": 0.9}, "Y"),
        ]
        results.add(select_best(candidates, config))
    # With 50 different seeds across 2 candidates, we expect both to appear
    assert len(results) == 2


# ---------------------------------------------------------------------------
# select_top_k
# ---------------------------------------------------------------------------

def test_select_top_k_scalar_none_config():
    candidates = [(0.5, "A"), (0.9, "B"), (0.7, "C")]
    indices = select_top_k(candidates, None, k=2)
    assert len(indices) == 2
    assert indices[0] == 1  # B is best
    assert indices[1] == 2  # C is second


@pytest.mark.parametrize("k", [1, 2, 3])
def test_select_top_k_scalar_k(k):
    candidates = [(0.5, "A"), (0.9, "B"), (0.7, "C")]
    indices = select_top_k(candidates, None, k=k)
    assert len(indices) == k
    assert indices[0] == 1  # B always best


def test_select_top_k_weighted():
    config = ObjectiveConfig(
        mode="weighted",
        weights={"a": 1.0, "b": 1.0},
    )
    candidates = [
        ({"a": 0.5, "b": 0.5}, "X"),  # weighted = 1.0
        ({"a": 0.9, "b": 0.1}, "Y"),  # weighted = 1.0
        ({"a": 0.8, "b": 0.8}, "Z"),  # weighted = 1.6
    ]
    indices = select_top_k(candidates, config, k=2)
    assert indices[0] == 2  # Z is best


def test_select_top_k_pareto():
    config = ObjectiveConfig(
        mode="pareto",
        weights={"a": 0.5, "b": 0.5},
        tie_break="weighted",
    )
    candidates = [
        ({"a": 0.9, "b": 0.1}, "X"),  # front 0
        ({"a": 0.1, "b": 0.9}, "Y"),  # front 0
        ({"a": 0.05, "b": 0.05}, "Z"),  # front 1 (dominated)
    ]
    indices = select_top_k(candidates, config, k=2)
    assert set(indices) == {0, 1}  # both front-0 candidates


# ---------------------------------------------------------------------------
# ObjectiveConfig validation
# ---------------------------------------------------------------------------

def test_config_default():
    config = ObjectiveConfig()
    assert config.mode == "scalar"
    assert config.weights == {}
    assert config.minimize == frozenset()
    assert config.scalarize_dict == "score"
    assert config.score_key == "score"


def test_config_set_to_frozenset():
    config = ObjectiveConfig(minimize={"lat"})
    assert isinstance(config.minimize, frozenset)
    assert "lat" in config.minimize


def test_config_negative_weight_raises():
    with pytest.raises(ValueError, match="non-negative"):
        ObjectiveConfig(weights={"a": -1.0})


def test_config_bad_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        ObjectiveConfig(mode="unknown")


def test_config_bad_tie_break_raises():
    with pytest.raises(ValueError, match="tie_break"):
        ObjectiveConfig(tie_break="bad")


def test_config_empty_pareto_metrics_raises():
    with pytest.raises(ValueError, match="non-empty"):
        ObjectiveConfig(pareto_metrics=())


def test_config_bad_scalarize_dict_raises():
    with pytest.raises(ValueError, match="scalarize_dict"):
        ObjectiveConfig(scalarize_dict="bad")


def test_config_empty_score_key_raises():
    with pytest.raises(ValueError, match="score_key"):
        ObjectiveConfig(score_key="")


def test_config_frozen():
    config = ObjectiveConfig()
    with pytest.raises(AttributeError):
        config.mode = "weighted"


# ---------------------------------------------------------------------------
# Objective scalar consistency (Fix 1 verification)
# ---------------------------------------------------------------------------

def test_weighted_objective_not_mean():
    """Weighted objective uses weighted_scalarize, not mean(values).

    Xavier's example: score_dict={'accuracy':1.0,'brevity':0.5} with
    weights={'accuracy':0.7,'brevity':0.3} should be 0.85, not 0.75 (mean).
    """
    score_dict = {"accuracy": 1.0, "brevity": 0.5}
    weights = {"accuracy": 0.7, "brevity": 0.3}

    objective = weighted_scalarize(apply_minimize(score_dict, frozenset()), weights)
    assert objective == pytest.approx(0.85)  # 0.7*1.0 + 0.3*0.5

    naive_mean = float(np.mean(list(score_dict.values())))
    assert naive_mean == pytest.approx(0.75)
    assert objective != pytest.approx(naive_mean)


def test_weighted_objective_stub_example():
    """StubLLM example: weighted objective differs from naive mean.

    score_dict={'accuracy':0.0,'brevity':0.01639...} with
    weights={'accuracy':0.7,'brevity':0.3} should be ~0.00492, not ~0.00820.
    """
    score_dict = {"accuracy": 0.0, "brevity": 0.01639344262295082}
    weights = {"accuracy": 0.7, "brevity": 0.3}

    objective = weighted_scalarize(apply_minimize(score_dict, frozenset()), weights)
    expected = 0.7 * 0.0 + 0.3 * 0.01639344262295082
    assert objective == pytest.approx(expected)  # ~0.004918

    naive_mean = float(np.mean(list(score_dict.values())))
    assert naive_mean == pytest.approx(0.00819672131147541)
    assert objective != pytest.approx(naive_mean)


def test_weighted_objective_with_minimize():
    """Minimize metrics are negated before scalarization."""
    score_dict = {"accuracy": 0.95, "latency_s": 0.200}
    config = ObjectiveConfig(
        mode="weighted",
        weights={"accuracy": 0.8, "latency_s": 0.2},
        minimize=frozenset({"latency_s"}),
    )

    minimized = apply_minimize(score_dict, config.minimize)
    assert minimized == {"accuracy": 0.95, "latency_s": -0.200}

    objective = weighted_scalarize(minimized, config.weights, config.missing_value)
    assert objective == pytest.approx(0.8 * 0.95 + 0.2 * (-0.200))  # 0.72


def test_weight_sensitivity_flips_winner():
    """Changing weights flips which candidate wins."""
    candidates = [
        ({"accuracy": 0.95, "brevity": 0.3}, "A"),  # high acc, low brev
        ({"accuracy": 0.70, "brevity": 0.9}, "B"),  # low acc, high brev
    ]

    config_acc = ObjectiveConfig(mode="weighted", weights={"accuracy": 0.9, "brevity": 0.1})
    assert select_best(candidates, config_acc) == 0  # A wins

    config_brev = ObjectiveConfig(mode="weighted", weights={"accuracy": 0.1, "brevity": 0.9})
    assert select_best(candidates, config_brev) == 1  # B wins
