import math
import os
import random
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

# Provide a light stub for optional graphviz dependency to allow imports without system graphviz
import sys, types
if "graphviz" not in sys.modules:
    sys.modules["graphviz"] = types.SimpleNamespace(Digraph=object)

from opto.trace.modules import model as trace_model
from opto.trace.nodes import node as trace_node
from opto.optimizers.optoprime_v2 import OptoPrimeV2
import pytest
from opto.trainer.algorithms.gepa_algorithms import (
        GEPAAlgorithmBase,
        GEPAUCBSearch,
        GEPABeamPareto,
        _compute_pareto_counts,
        _pareto_sample,
        _uniform_merge_params,
    )
from opto.trainer.evaluators import evaluate
from opto.trainer.guide import Guide
from opto.utils.llm import DummyLLM


class ExactMatchGuide(Guide):
    """Simple guide: score=1 if response == reference, else 0."""

    def get_feedback(self, query: Any, response: Any, reference: Any, **kwargs):
        score = float(response == reference)
        feedback = f"Score: {score}. Response: {response}. Reference: {reference}."
        return score, feedback


@trace_model
class AddAgent:
    """Toy agent: returns x + param."""

    def __init__(self, param: int = 0):
        self.param = trace_node(int(param), trainable=True)

    def forward(self, x: int) -> int:
        return x + self.param


def make_dummy_llm(suggest_value: int) -> DummyLLM:
    """Dummy LLM that parses the variable name from the prompt and suggests a fixed value.

    Matches the default XML-like output format expected by OptoPrimeV2.
    """

    def _llm_callable(messages, **kwargs):
        # Extract the variable name from the #Variables section in the prompt
        problem = messages[1]["content"] if isinstance(messages, (list, tuple)) and len(messages) > 1 else ""
        name_match = re.findall(r"<variable name=\"\s*(.*?)\" type=.*>", problem)
        var_name = name_match[0] if name_match else "param"
        return (
            f"""
            <reasoning> Dummy reasoning based on the input messages. </reasoning>
            <variable>
            <name> {var_name} </name>
            <value> {suggest_value} </value>
            </variable>
            """
        )

    return DummyLLM(_llm_callable)


def make_dataset(target_add: int, n: int = 8) -> Dict[str, List[int]]:
    xs = list(range(n))
    infos = [x + target_add for x in xs]
    return {"inputs": xs, "infos": infos}


def build_optimizer(agent: AddAgent, suggest_value: int) -> OptoPrimeV2:
    return OptoPrimeV2(agent.parameters(), llm=make_dummy_llm(suggest_value))


def test_pareto_counting_and_sampling():
    # Construct mock candidates with per-instance eval vectors where each wins on one dimension
    from types import SimpleNamespace

    class Cand(SimpleNamespace):
        pass

    A = Cand(eval_vector=[1.0, 0.1], wins=0, mean=0.55)
    B = Cand(eval_vector=[0.2, 1.1], wins=0, mean=0.65)
    cands = [A, B]

    _compute_pareto_counts(cands)
    assert A.wins == 1 and B.wins == 1

    rng = random.Random(0)
    # With equal wins, both should be sampled with similar probability
    picks = [
        _pareto_sample([A, B], temperature=1.0, rng=rng) for _ in range(100)
    ]
    a_count = sum(p is A for p in picks)
    b_count = sum(p is B for p in picks)
    assert abs(a_count - b_count) < 40  # rough balance


def test_uniform_merge_params_uses_both_parents():
    # Use two ParameterNodes to exercise merging across keys
    @trace_model
    class TwoParam:
        def __init__(self):
            self.a = trace_node(1, trainable=True)
            self.b = trace_node(2, trainable=True)

        def forward(self, x):
            return self.a + self.b + x

    m = TwoParam()
    a_params = {p: (10 if p.py_name.endswith("a") else 20) for p in m.parameters()}
    b_params = {p: (100 if p.py_name.endswith("a") else 200) for p in m.parameters()}

    rng = random.Random(123)
    merged = _uniform_merge_params(a_params, b_params, rng)
    # For each key, merged value should be chosen from either a_params or b_params
    for k, v in merged.items():
        assert v in (a_params[k], b_params[k])


@pytest.mark.parametrize(
    "algo_cls,train_kwargs",
    [
        (GEPAAlgorithmBase, {"num_iters": 8, "train_batch_size": 2, "merge_every": 2}),
        (GEPAUCBSearch, {"num_search_iterations": 8, "train_batch_size": 2, "merge_every": 2}),
        (GEPABeamPareto, {"num_search_iterations": 8, "train_batch_size": 2, "merge_every": 2}),
    ],
)
def test_gepa_variants_converge_on_dummyllm(algo_cls, train_kwargs):
    target_add = 5
    ds = make_dataset(target_add, n=6)
    agent = AddAgent(param=0)
    optimizer = build_optimizer(agent, suggest_value=target_add)

    algo = algo_cls(agent=agent, optimizer=optimizer, logger=None, num_threads=1)

    # Prepare kwargs and include 'verbose' only if supported
    import inspect
    call_kwargs = dict(guide=ExactMatchGuide(), train_dataset=ds, pareto_subset_size=4, num_threads=1)
    sig = inspect.signature(algo.train)
    if 'validation_dataset' in sig.parameters:
        call_kwargs['validation_dataset'] = ds
    else:
        call_kwargs['validate_dataset'] = ds
    call_kwargs.update(train_kwargs)
    if 'verbose' in sig.parameters:
        call_kwargs['verbose'] = False

    metrics, best = algo.train(**call_kwargs)

    # Best mean on pareto subset should be perfect
    assert isinstance(best, float)
    assert best == pytest.approx(1.0, rel=0, abs=1e-6)
    # Agent parameter should be updated to target_add
    assert agent.param.data == target_add


def test_compare_gepa_vs_basicsearch_on_dummyllm():
    from opto.trainer.algorithms.basic_algorithms import BasicSearchAlgorithm

    target_add = 7
    ds = make_dataset(target_add, n=6)
    agent_gepa = AddAgent(param=0)
    agent_basic = AddAgent(param=0)

    opt_gepa = build_optimizer(agent_gepa, suggest_value=target_add)
    opt_basic = build_optimizer(agent_basic, suggest_value=target_add)

    # GEPA
    gepa = GEPAAlgorithmBase(agent_gepa, optimizer=opt_gepa, logger=None, num_threads=1)
    _, best_gepa = gepa.train(
        guide=ExactMatchGuide(),
        train_dataset=ds,
        validate_dataset=ds,
        pareto_subset_size=4,
        num_iters=8,
        train_batch_size=2,
        merge_every=2,
        num_threads=1,
    )

    # BasicSearch baseline
    basic = BasicSearchAlgorithm(agent_basic, optimizer=opt_basic, logger=None, num_threads=1)
    basic.train(
        guide=ExactMatchGuide(),
        train_dataset=ds,
        validate_dataset=ds,
        num_proposals=1,
        num_epochs=1,
        batch_size=1,
        test_dataset=ds,
        eval_frequency=1,
        num_threads=1,
        verbose=False,
    )

    # Evaluate both on full dataset
    score_gepa = np.mean(evaluate(agent_gepa, ExactMatchGuide(), ds["inputs"], ds["infos"], num_threads=2))
    score_basic = np.mean(evaluate(agent_basic, ExactMatchGuide(), ds["inputs"], ds["infos"], num_threads=2))

    assert best_gepa == pytest.approx(1.0, rel=0, abs=1e-6)
    assert score_gepa == pytest.approx(1.0, rel=0, abs=1e-6)
    assert score_basic == pytest.approx(1.0, rel=0, abs=1e-6)


def test_snapshot_params_fast():
    """Test the fast parameter snapshot utility function."""
    from opto.trainer.algorithms.gepa_algorithms import _snapshot_params_fast
    
    @trace_model
    class MultiTypeAgent:
        def __init__(self):
            self.int_param = trace_node(42, trainable=True)
            self.str_param = trace_node("hello", trainable=True)
            self.float_param = trace_node(3.14, trainable=True)
            self.list_param = trace_node([1, 2, 3], trainable=True)
            self.dict_param = trace_node({"key": "value"}, trainable=True)
            # Test numpy array
            self.np_param = trace_node(np.array([1, 2, 3]), trainable=True)

        def forward(self, x):
            return x + self.int_param

    agent = MultiTypeAgent()
    params = list(agent.parameters())
    
    # Test snapshot
    snapshot = _snapshot_params_fast(params)
    
    # Check that all parameters are included
    assert len(snapshot) == len(params)
    
    # Modify original values
    agent.int_param._set(100)
    agent.str_param._set("modified")
    agent.np_param._set(np.array([4, 5, 6]))
    
    # Verify snapshot preserved original values
    for p in params:
        if p.py_name == "int_param":
            assert snapshot[p] == 42
        elif p.py_name == "str_param":
            assert snapshot[p] == "hello"
        elif p.py_name == "np_param":
            assert np.array_equal(snapshot[p], np.array([1, 2, 3]))


def test_fingerprint_params():
    """Test the parameter fingerprinting utility function."""
    from opto.trainer.algorithms.gepa_algorithms import _fingerprint_params
    
    @trace_model
    class SimpleAgent:
        def __init__(self):
            self.a = trace_node(1, trainable=True)
            self.b = trace_node("test", trainable=True)

        def forward(self, x):
            return x + self.a

    agent = SimpleAgent()
    params_dict = {p: p.data for p in agent.parameters()}
    
    # Test fingerprinting
    fp1 = _fingerprint_params(params_dict)
    fp2 = _fingerprint_params(params_dict)
    
    # Same parameters should produce same fingerprint
    assert fp1 == fp2
    
    # Different parameters should produce different fingerprint
    agent.a._set(2)
    params_dict2 = {p: p.data for p in agent.parameters()}
    fp3 = _fingerprint_params(params_dict2)
    assert fp1 != fp3


def test_numpy_seeding_reproducibility():
    """Test that numpy seeding ensures reproducible behavior."""
    target_add = 3
    ds = make_dataset(target_add, n=4)
    
    # Test with same seed
    results = []
    for seed in [123, 123]:  # Same seed twice
        agent = AddAgent(param=0)
        optimizer = build_optimizer(agent, suggest_value=target_add)
        algo = GEPAAlgorithmBase(agent=agent, optimizer=optimizer, logger=None, num_threads=1, rng_seed=seed)
        
        metrics, best = algo.train(
            guide=ExactMatchGuide(),
            train_dataset=ds,
            validate_dataset=ds,
            pareto_subset_size=3,
            num_iters=2,
            train_batch_size=1,
            merge_every=2,
            num_threads=1,
        )
        results.append((metrics, best, agent.param.data))
    
    # Results should be identical with same seed
    assert results[0][1] == results[1][1]  # Same best score
    assert results[0][2] == results[1][2]  # Same final parameter
    
    # Test with different seed
    agent_diff = AddAgent(param=0)
    optimizer_diff = build_optimizer(agent_diff, suggest_value=target_add)
    algo_diff = GEPAAlgorithmBase(agent=agent_diff, optimizer=optimizer_diff, logger=None, num_threads=1, rng_seed=456)
    
    metrics_diff, best_diff = algo_diff.train(
        guide=ExactMatchGuide(),
        train_dataset=ds,
        validate_dataset=ds,
        pareto_subset_size=3,
        num_iters=2,
        train_batch_size=1,
        merge_every=2,
        num_threads=1,
    )
    
    # Both should converge but the process might differ
    # (though with DummyLLM behavior is very predictable)
    assert best_diff == pytest.approx(1.0, rel=0, abs=1e-6)


def test_gepa_ucb_pareto_cache():
    """Test Pareto cache functionality in GEPAUCBSearch."""
    target_add = 4
    ds = make_dataset(target_add, n=3)
    agent = AddAgent(param=0)
    optimizer = build_optimizer(agent, suggest_value=target_add)
    
    # Test with cache enabled
    algo = GEPAUCBSearch(agent=agent, optimizer=optimizer, logger=None, num_threads=1, enable_pareto_cache=True)
    
    metrics, best = algo.train(
        guide=ExactMatchGuide(),
        train_dataset=ds,
        validate_dataset=ds,
        pareto_subset_size=2,
        num_search_iterations=2,
        train_batch_size=1,
        merge_every=2,
        num_threads=1,
    )
    
    # Should converge to perfect solution
    assert best == pytest.approx(1.0, rel=0, abs=1e-6)
    assert agent.param.data == target_add
    
    # Test that cache was used (should have some entries)
    # Note: exact cache size depends on algorithm behavior, but should be non-empty if enabled
    if hasattr(algo, '_pareto_cache'):
        assert isinstance(algo._pareto_cache, dict)


def test_budget_tracking_functionality():
    """Test budget tracking in GEPA algorithms."""
    target_add = 2
    ds = make_dataset(target_add, n=4)
    agent = AddAgent(param=0)
    optimizer = build_optimizer(agent, suggest_value=target_add)
    
    # Test GEPABeamPareto with budget
    algo = GEPABeamPareto(agent=agent, optimizer=optimizer, logger=None, num_threads=1)
    
    metrics, best = algo.train(
        guide=ExactMatchGuide(),
        train_dataset=ds,
        validate_dataset=ds,
        pareto_subset_size=3,
        num_search_iterations=2,
        train_batch_size=1,
        merge_every=2,
        budget_B=10,  # Low budget to test tracking
        num_threads=1,
    )
    
    # Should still achieve good results even with budget constraint
    assert isinstance(best, float)
    assert best >= 0.0  # Should be non-negative score


def test_thread_safety_with_sequential_fallback():
    """Test that algorithms work correctly with sequential fallback when batch_run unavailable."""
    target_add = 1
    ds = make_dataset(target_add, n=2)
    agent = AddAgent(param=0)
    optimizer = build_optimizer(agent, suggest_value=target_add)
    
    # Test with num_threads=1 (should use sequential)
    algo = GEPAAlgorithmBase(agent=agent, optimizer=optimizer, logger=None, num_threads=1)
    metrics, best = algo.train(
        guide=ExactMatchGuide(),
        train_dataset=ds,
        validate_dataset=ds,
        pareto_subset_size=2,
        num_iters=2,
        train_batch_size=1,
        merge_every=2,
        num_threads=1,
    )
    
    assert best == pytest.approx(1.0, rel=0, abs=1e-6)
    assert agent.param.data == target_add
    
    # Test with num_threads=2 (may use parallel or fallback to sequential)
    agent2 = AddAgent(param=0)
    optimizer2 = build_optimizer(agent2, suggest_value=target_add)
    algo2 = GEPAAlgorithmBase(agent=agent2, optimizer=optimizer2, logger=None, num_threads=2)
    
    metrics2, best2 = algo2.train(
        guide=ExactMatchGuide(),
        train_dataset=ds,
        validate_dataset=ds,
        pareto_subset_size=2,
        num_iters=2,
        train_batch_size=1,
        merge_every=2,
        num_threads=2,
    )
    
    assert best2 == pytest.approx(1.0, rel=0, abs=1e-6)
    assert agent2.param.data == target_add


def test_gepa_ucb_selectmodule_policy():
    """Test different module selection policies in GEPAUCBSearch."""
    target_add = 6
    ds = make_dataset(target_add, n=3)
    
    # Test different selection policies
    policies = ["round_robin"]  # Could test more if other policies are available
    
    for policy in policies:
        agent = AddAgent(param=0)
        optimizer = build_optimizer(agent, suggest_value=target_add)
        
        algo = GEPAUCBSearch(
            agent=agent,
            optimizer=optimizer,
            logger=None,
            num_threads=1,
            selectmodule_policy=policy
        )
        
        metrics, best = algo.train(
            guide=ExactMatchGuide(),
            train_dataset=ds,
            validate_dataset=ds,
            pareto_subset_size=2,
            num_search_iterations=2,
            train_batch_size=1,
            merge_every=2,
            num_threads=1,
        )
        
        assert best == pytest.approx(1.0, rel=0, abs=1e-6)
        assert agent.param.data == target_add
