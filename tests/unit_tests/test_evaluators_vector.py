"""Tests for evaluate_vector and aggregate_vector_scores in opto.trainer.evaluators."""
import pytest
import numpy as np
from opto import trace
from opto.trainer.evaluators import evaluate_vector, aggregate_vector_scores
from opto.trainer.guide import Guide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@trace.model
class SimpleAgent:
    """Deterministic agent: returns input + param."""
    def __init__(self, param):
        self.param = trace.node(param, trainable=True)

    def forward(self, x):
        return x + self.param


class MultiMetricGuide(Guide):
    """Guide returning multi-metric score dict."""
    def __init__(self, target):
        super().__init__()
        self.target = target

    def get_feedback(self, query, response, reference=None, **kwargs):
        accuracy = float(response == self.target)
        brevity = 1.0 / max(abs(response - self.target) + 1, 1)
        feedback = f"response={response}, target={self.target}"
        return accuracy, feedback

    def get_score_dict(self, query, response, reference=None, **kwargs):
        accuracy = float(response == self.target)
        brevity = 1.0 / max(abs(response - self.target) + 1, 1)
        return {"accuracy": accuracy, "brevity": brevity}


class ScalarGuide(Guide):
    """Guide using only scalar get_feedback (no get_score_dict override)."""
    def __init__(self, target):
        super().__init__()
        self.target = target

    def get_feedback(self, query, response, reference=None, **kwargs):
        score = float(response == self.target)
        feedback = f"response={response}"
        return score, feedback


# ---------------------------------------------------------------------------
# evaluate_vector
# ---------------------------------------------------------------------------

def test_evaluate_vector_basic():
    """evaluate_vector returns a list of dicts with correct metric values."""
    agent = SimpleAgent(10)
    guide = MultiMetricGuide(target=11)
    inputs = [1, 2, 3]
    infos = [None, None, None]

    results = evaluate_vector(agent, guide, inputs, infos, num_threads=1)

    assert len(results) == 3
    assert isinstance(results[0], dict)
    # input=1 + param=10 = 11 == target=11 -> accuracy=1.0, brevity=1.0
    assert results[0]["accuracy"] == 1.0
    assert results[0]["brevity"] == 1.0
    # input=2 + param=10 = 12 != target=11 -> accuracy=0.0
    assert results[1]["accuracy"] == 0.0
    assert results[1]["brevity"] == pytest.approx(0.5)  # 1/(|12-11|+1) = 0.5
    # input=3 + param=10 = 13 != target=11 -> accuracy=0.0
    assert results[2]["accuracy"] == 0.0
    assert results[2]["brevity"] == pytest.approx(1.0 / 3.0)  # 1/(|13-11|+1)


def test_evaluate_vector_all_keys_present():
    """Every result dict contains the same set of metric keys."""
    agent = SimpleAgent(5)
    guide = MultiMetricGuide(target=10)
    inputs = [1, 2, 3, 4, 5]
    infos = [None] * 5

    results = evaluate_vector(agent, guide, inputs, infos, num_threads=1)

    expected_keys = {"accuracy", "brevity"}
    for rd in results:
        assert set(rd.keys()) == expected_keys


def test_evaluate_vector_scalar_guide_fallback():
    """Guide without get_score_dict override returns {"score": float}."""
    agent = SimpleAgent(10)
    guide = ScalarGuide(target=11)
    inputs = [1, 2]
    infos = [None, None]

    results = evaluate_vector(agent, guide, inputs, infos, num_threads=1)

    assert len(results) == 2
    # input=1 + param=10 = 11 == target=11 -> score=1.0
    assert results[0] == {"score": 1.0}
    # input=2 + param=10 = 12 != target=11 -> score=0.0
    assert results[1] == {"score": 0.0}


def test_evaluate_vector_empty_inputs():
    """Empty inputs produce empty results."""
    agent = SimpleAgent(0)
    guide = MultiMetricGuide(target=0)

    results = evaluate_vector(agent, guide, [], [], num_threads=1)
    assert results == []


# ---------------------------------------------------------------------------
# aggregate_vector_scores
# ---------------------------------------------------------------------------

def test_aggregate_basic():
    """Per-metric mean is computed correctly."""
    score_dicts = [
        {"accuracy": 1.0, "brevity": 0.5},
        {"accuracy": 0.0, "brevity": 1.0},
    ]
    agg = aggregate_vector_scores(score_dicts)
    assert agg["accuracy"] == pytest.approx(0.5)
    assert agg["brevity"] == pytest.approx(0.75)


def test_aggregate_empty():
    """Empty input returns empty dict."""
    assert aggregate_vector_scores([]) == {}


def test_aggregate_single():
    """Single dict returns the same values."""
    score_dicts = [{"a": 0.42, "b": 0.99}]
    agg = aggregate_vector_scores(score_dicts)
    assert agg == {"a": pytest.approx(0.42), "b": pytest.approx(0.99)}


def test_aggregate_missing_keys():
    """Handles dicts with partially overlapping keys."""
    score_dicts = [
        {"accuracy": 1.0},
        {"accuracy": 0.0, "brevity": 0.8},
    ]
    agg = aggregate_vector_scores(score_dicts)
    assert agg["accuracy"] == pytest.approx(0.5)
    # brevity only present in one dict
    assert agg["brevity"] == pytest.approx(0.8)
