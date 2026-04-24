"""Tests for M2 multi-objective support in BeamsearchAlgorithm and PrioritySearch.

Uses DummyLLM and deterministic guides — no API keys required.
"""
import pytest
import re
import numpy as np
import heapq

from opto import trace
from opto.trainer.guide import Guide
from opto.trainer.objectives import ObjectiveConfig
from examples.trainers.beamsearch_algorithm import (
    BeamsearchAlgorithm,
    BeamsearchHistoryAlgorithm,
)
from opto.trainer.algorithms.priority_search import (
    PrioritySearch,
    ModuleCandidate,
    HeapMemory,
    ParetoHeapMemory,
)
from opto.optimizers import OptoPrimeV2
from opto.utils.llm import DummyLLM


# ---------------------------------------------------------------------------
# Fixtures: Guide, Agent, LLM, Dataset
# ---------------------------------------------------------------------------

class ScalarGuide(Guide):
    """Simple scalar guide: exact-match returns 1.0/0.0."""

    def get_feedback(self, query, response, reference=None, **kwargs):
        score = float(str(response).strip() == str(reference).strip())
        feedback = "Correct" if score == 1.0 else "Incorrect"
        return score, feedback


class MultiMetricGuide(Guide):
    """Multi-metric guide: scalar accuracy from get_feedback,
    accuracy + brevity from get_score_dict."""

    def get_feedback(self, query, response, reference=None, **kwargs):
        accuracy = float(str(response).strip() == str(reference).strip())
        feedback = "Correct" if accuracy == 1.0 else "Incorrect"
        return accuracy, feedback

    def get_score_dict(self, query, response, reference=None, **kwargs):
        accuracy = float(str(response).strip() == str(reference).strip())
        brevity = max(0.0, 1.0 - len(str(response)) / 100.0)
        return {"accuracy": accuracy, "brevity": brevity}


@trace.model
class StubAgent:
    def __init__(self):
        self.param = trace.node("default answer", trainable=True)

    def forward(self, x):
        return self.param


# Simple dataset
DATASET = {
    "inputs": ["What is 2+2?", "Capital of France?", "Color of sky?"],
    "infos": ["4", "Paris", "blue"],
}

SUGGESTED_VALUE = "4"


def _llm_callable(messages, **kwargs):
    """Dummy LLM callable returning a fixed value."""
    problem = messages[1]["content"]
    name = re.findall(r'<variable name="\s*(.*?)" type=.*>', problem)
    name = name[0] if name else "unknown"
    return f"""
    <reasoning> Dummy reasoning. </reasoning>
    <variable>
    <name> {name} </name>
    <value> {SUGGESTED_VALUE} </value>
    </variable>
    """


def _make_beamsearch():
    """Create a BeamsearchAlgorithm instance with DummyLLM."""
    agent = StubAgent()
    llm = DummyLLM(_llm_callable)
    optimizer = OptoPrimeV2(agent.parameters(), llm=llm)
    algo = BeamsearchAlgorithm(agent, optimizer)
    return algo


def _make_priority_search():
    """Create a PrioritySearch instance with DummyLLM."""
    agent = StubAgent()
    llm = DummyLLM(_llm_callable)
    optimizer = OptoPrimeV2(agent.parameters(), llm=llm)
    algo = PrioritySearch(agent, optimizer)
    return algo


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_beamsearch_scalar_baseline(self):
        """BeamsearchAlgorithm with objective_config=None trains and returns scores."""
        algo = _make_beamsearch()
        metrics, final_score = algo.train(
            guide=ScalarGuide(),
            train_dataset=DATASET,
            objective_config=None,
            beam_width=2,
            num_proposals=1,
            max_depth=1,
            batch_size=1,
            num_threads=2,
        )
        assert "best_validation_scores" in metrics
        assert isinstance(final_score, (int, float))

    def test_priority_search_scalar_baseline(self):
        """PrioritySearch with objective_config=None trains without error."""
        algo = _make_priority_search()
        algo.train(
            guide=ScalarGuide(),
            train_dataset=DATASET,
            objective_config=None,
            batch_size=1,
            num_batches=1,
            num_epochs=1,
            num_candidates=2,
            num_proposals=1,
            num_threads=2,
            long_term_memory_size=3,
            memory_update_frequency=0,
            verbose=False,
        )
        # If we get here without exception, backward compat is maintained


# ---------------------------------------------------------------------------
# Beamsearch selection
# ---------------------------------------------------------------------------

class TestBeamsearchSelection:

    def test_beamsearch_weighted_mode(self):
        """Weighted mode trains and populates _last_selected_score_dicts."""
        algo = _make_beamsearch()
        config = ObjectiveConfig(
            mode="weighted",
            weights={"accuracy": 0.7, "brevity": 0.3},
        )
        metrics, final_score = algo.train(
            guide=MultiMetricGuide(),
            train_dataset=DATASET,
            objective_config=config,
            beam_width=2,
            num_proposals=1,
            max_depth=1,
            batch_size=1,
            num_threads=2,
        )
        assert isinstance(final_score, (int, float))
        # _last_selected_score_dicts should have been populated by select()
        assert hasattr(algo, "_last_selected_score_dicts")
        if algo._last_selected_score_dicts:
            sd = algo._last_selected_score_dicts[0]
            assert isinstance(sd, dict)
            assert "accuracy" in sd
            assert "brevity" in sd

    def test_beamsearch_pareto_mode(self):
        """Pareto mode trains without error."""
        algo = _make_beamsearch()
        config = ObjectiveConfig(mode="pareto")
        metrics, final_score = algo.train(
            guide=MultiMetricGuide(),
            train_dataset=DATASET,
            objective_config=config,
            beam_width=2,
            num_proposals=1,
            max_depth=1,
            batch_size=1,
            num_threads=2,
        )
        assert isinstance(final_score, (int, float))

    def test_beamsearch_history_forwards_config(self):
        """BeamsearchHistoryAlgorithm accepts and stores objective_config."""
        agent = StubAgent()
        llm = DummyLLM(_llm_callable)
        optimizer = OptoPrimeV2(agent.parameters(), llm=llm)
        algo = BeamsearchHistoryAlgorithm(agent, optimizer)
        config = ObjectiveConfig(
            mode="weighted",
            weights={"accuracy": 1.0},
        )
        algo.train(
            guide=MultiMetricGuide(),
            train_dataset=DATASET,
            objective_config=config,
            beam_width=2,
            num_proposals=1,
            max_depth=1,
            batch_size=1,
            num_threads=2,
        )
        assert algo.objective_config is config


# ---------------------------------------------------------------------------
# ParetoHeapMemory unit tests
# ---------------------------------------------------------------------------

class TestParetoHeapMemory:

    def test_fallback_when_no_config(self):
        """When config is None, pop() behaves like standard heappop."""
        phm = ParetoHeapMemory(size=10, pareto_k=5)
        phm.memory = [(-3.0, "c3"), (-2.0, "c2"), (-1.0, "c1")]
        heapq.heapify(phm.memory)
        neg_score, data = phm.pop()
        # heappop returns the smallest (most negative = highest score)
        assert neg_score == -3.0
        assert data == "c3"

    def test_pareto_pop_selects_from_front(self):
        """When config mode='pareto', pop() selects from Pareto front."""
        # Candidate A: good accuracy, bad brevity
        # Candidate B: bad accuracy, good brevity
        # Candidate C: dominated by both A and B
        score_dicts = {
            "A": {"accuracy": 0.9, "brevity": 0.1},
            "B": {"accuracy": 0.1, "brevity": 0.9},
            "C": {"accuracy": 0.05, "brevity": 0.05},
        }

        config = ObjectiveConfig(
            mode="pareto",
            weights={"accuracy": 0.7, "brevity": 0.3},
        )
        phm = ParetoHeapMemory(
            size=10,
            pareto_k=10,
            score_dict_fn=lambda c: score_dicts[c],
            objective_config=config,
        )
        # Push all three (scalar priority doesn't matter for pareto pop)
        phm.memory = [(-0.5, "A"), (-0.5, "B"), (-0.1, "C")]
        heapq.heapify(phm.memory)

        neg_score, chosen = phm.pop()
        # C is dominated, so chosen must be A or B
        assert chosen in ("A", "B"), f"Expected A or B from Pareto front, got {chosen}"
        # With weights accuracy=0.7, brevity=0.3:
        # A: 0.7*0.9 + 0.3*0.1 = 0.66
        # B: 0.7*0.1 + 0.3*0.9 = 0.34
        # Tie-break by weighted scalarize → A wins
        assert chosen == "A"

    def test_missing_score_dict_fallback(self):
        """When score_dict_fn returns None, falls back to heappop."""
        config = ObjectiveConfig(mode="pareto")
        phm = ParetoHeapMemory(
            size=10,
            pareto_k=10,
            score_dict_fn=lambda c: None,  # always returns None
            objective_config=config,
        )
        phm.memory = [(-5.0, "best"), (-2.0, "mid"), (-1.0, "worst")]
        heapq.heapify(phm.memory)

        neg_score, data = phm.pop()
        # Falls back to heappop → highest priority (most negative)
        assert neg_score == -5.0
        assert data == "best"


# ---------------------------------------------------------------------------
# PrioritySearch multi-objective
# ---------------------------------------------------------------------------

class TestPrioritySearchMultiObjective:

    def test_weighted_priority(self):
        """With weighted config, compute_exploration_priority uses weighted scalarization."""
        algo = _make_priority_search()
        config = ObjectiveConfig(
            mode="weighted",
            weights={"accuracy": 0.6, "brevity": 0.4},
        )
        # Initialize search params to set self.objective_config and self.score_function
        algo._initialize_search_parameters(
            num_candidates=2,
            num_proposals=1,
            validate_exploration_candidates=True,
            use_best_candidate_to_explore=True,
            score_function="mean",
            score_range=(0, 1),
            ucb_exploration_constant=1.0,
            long_term_memory_size=None,
            short_term_memory_size=None,
            memory_update_frequency=0,
            decouple_optimizers=True,
            objective_config=config,
        )

        candidate = ModuleCandidate(algo.agent)
        # Add rollouts with score_dict
        candidate.rollouts = [
            {"module": None, "x": "q", "info": "a", "target": "a",
             "score": 0.8, "feedback": "ok",
             "score_dict": {"accuracy": 0.9, "brevity": 0.7}},
            {"module": None, "x": "q", "info": "a", "target": "a",
             "score": 0.6, "feedback": "ok",
             "score_dict": {"accuracy": 0.7, "brevity": 0.5}},
        ]

        priority = algo.compute_exploration_priority(candidate)
        # Mean score_dict: accuracy=0.8, brevity=0.6
        # Weighted: 0.6*0.8 + 0.4*0.6 = 0.48 + 0.24 = 0.72
        assert isinstance(priority, float)
        assert abs(priority - 0.72) < 1e-6, f"Expected ~0.72, got {priority}"

    def test_score_dict_in_rollouts(self):
        """After validate(), rollouts contain score_dict entries when multi-objective is active."""
        algo = _make_priority_search()
        config = ObjectiveConfig(
            mode="weighted",
            weights={"accuracy": 0.7, "brevity": 0.3},
        )
        algo.train(
            guide=MultiMetricGuide(),
            train_dataset=DATASET,
            objective_config=config,
            batch_size=1,
            num_batches=1,
            num_epochs=1,
            num_candidates=2,
            num_proposals=1,
            num_threads=2,
            long_term_memory_size=3,
            memory_update_frequency=0,
            verbose=False,
        )
        # Check that at least some candidates in memory have score_dict in rollouts
        found_score_dict = False
        for neg_priority, candidate in algo.long_term_memory:
            for rollout in candidate.rollouts:
                if "score_dict" in rollout and rollout["score_dict"] is not None:
                    found_score_dict = True
                    sd = rollout["score_dict"]
                    assert "accuracy" in sd
                    assert "brevity" in sd
                    break
            if found_score_dict:
                break
        assert found_score_dict, "Expected at least one rollout with score_dict in memory"


# ---------------------------------------------------------------------------
# ModuleCandidate.mean_score_dict
# ---------------------------------------------------------------------------

class TestModuleCandidateMeanScoreDict:

    def test_mean_score_dict(self):
        """Returns correct per-metric mean when rollouts have score_dict."""
        agent = StubAgent()
        candidate = ModuleCandidate(agent)
        candidate.rollouts = [
            {"module": None, "x": "q", "info": "a", "target": "a",
             "score": 0.9, "feedback": "ok",
             "score_dict": {"accuracy": 1.0, "brevity": 0.8}},
            {"module": None, "x": "q", "info": "a", "target": "a",
             "score": 0.7, "feedback": "ok",
             "score_dict": {"accuracy": 0.6, "brevity": 0.4}},
        ]
        sd = candidate.mean_score_dict()
        assert sd is not None
        assert abs(sd["accuracy"] - 0.8) < 1e-6
        assert abs(sd["brevity"] - 0.6) < 1e-6

    def test_mean_score_dict_none_when_no_score_dict(self):
        """Returns None when rollouts lack score_dict."""
        agent = StubAgent()
        candidate = ModuleCandidate(agent)
        candidate.rollouts = [
            {"module": None, "x": "q", "info": "a", "target": "a",
             "score": 0.9, "feedback": "ok"},
            {"module": None, "x": "q", "info": "a", "target": "a",
             "score": 0.7, "feedback": "ok"},
        ]
        sd = candidate.mean_score_dict()
        assert sd is None
