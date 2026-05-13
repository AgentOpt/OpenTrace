import os

import pytest

from opto.optimizers.optimizer import Optimizer
from opto.trace import GRAPH
from opto.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything
from opto.optimize_anything.trace_backend import TraceOptimizerBackend, resolve_optimizer_cls


@pytest.fixture(autouse=True)
def clear_graph():
    GRAPH.clear()
    yield
    GRAPH.clear()


class SuffixOptimizer(Optimizer):
    def __init__(self, parameters, suffix="!", **kwargs):
        super().__init__(parameters)
        self.suffix = suffix
        self.seen_kwargs = kwargs

    def _step(self, *args, **kwargs):
        return {parameter: f"{parameter.data}{self.suffix}" for parameter in self.parameters}


class NoBypassOptimizer(Optimizer):
    def _step(self, *args, **kwargs):
        return {parameter: f"{parameter.data}*" for parameter in self.parameters}

    def step(self):
        update_dict = self.propose()
        self.update(update_dict)
        return update_dict


class ProposeOnlyOptimizer:
    def __init__(self, parameters, **kwargs):
        self.parameters = parameters

    def zero_feedback(self):
        pass

    def backward(self, output, feedback):
        self.feedback = feedback

    def propose(self):
        return {parameter: f"{parameter.data}?" for parameter in self.parameters}


def test_trace_backend_updates_string_candidate_with_optimizer_protocol():
    backend = TraceOptimizerBackend(optimizer_cls=SuffixOptimizer, optimizer_kwargs={"suffix": " improved"})
    assert backend(candidate="seed", feedback="make it better", objective="improve candidate") == "seed improved"


def test_trace_backend_does_not_mutate_original_candidate():
    original = {"prompt": "seed"}
    backend = TraceOptimizerBackend(optimizer_cls=SuffixOptimizer, optimizer_kwargs={"suffix": " v2"})
    updated = backend(candidate=original, feedback="make it better")
    assert original == {"prompt": "seed"}
    assert updated is not original


def test_trace_backend_preserves_single_key_dict_candidate_shape():
    backend = TraceOptimizerBackend(optimizer_cls=SuffixOptimizer, optimizer_kwargs={"suffix": " v2"})
    assert backend(candidate={"prompt": "seed"}, feedback="make it better") == {"prompt": "seed v2"}


def test_trace_backend_can_roundtrip_json_dict_candidates_when_configured():
    class JsonOptimizer(Optimizer):
        def _step(self, *args, **kwargs):
            return {parameter: '{"x": 2, "nested": [1]}' for parameter in self.parameters}

    backend = TraceOptimizerBackend(
        optimizer_cls=JsonOptimizer,
        candidate_serializer=lambda candidate: '{"x": 1, "nested": []}',
    )
    assert backend(candidate={"x": 1, "nested": []}, feedback="increase x") == {"x": 2, "nested": [1]}


def test_trace_backend_falls_back_when_step_has_no_bypassing_kwarg():
    backend = TraceOptimizerBackend(optimizer_cls=NoBypassOptimizer)
    assert backend(candidate="a", feedback="change") == "a*"


def test_trace_backend_can_use_propose_only_optimizer():
    backend = TraceOptimizerBackend(optimizer_cls=ProposeOnlyOptimizer)
    assert backend(candidate="a", feedback="change") == "a?"


def test_trace_backend_preserves_single_key_dict_for_scalar_non_string_proposal():
    class ScalarOptimizer(Optimizer):
        def _step(self, *args, **kwargs):
            return {parameter: 2 for parameter in self.parameters}

    backend = TraceOptimizerBackend(optimizer_cls=ScalarOptimizer)
    assert backend(candidate={"x": 1}, feedback="increase") == {"x": 2}


def test_trace_backend_rejects_non_class_optimizer_cls():
    with pytest.raises(ValueError, match="optimizer_cls must be a class or string name"):
        TraceOptimizerBackend(optimizer_cls=object())


def test_trace_backend_inside_optimize_anything_loop():
    backend = TraceOptimizerBackend(optimizer_cls=SuffixOptimizer, optimizer_kwargs={"suffix": "x"})
    result = optimize_anything(
        seed_candidate="a",
        evaluator=lambda candidate: float(len(candidate)),
        objective="make longer",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=3, max_steps=2),
            reflection=ReflectionConfig(custom_candidate_proposer=backend),
        ),
    )
    assert result.best_candidate == "axx"
    assert result.best_score == pytest.approx(3.0)


def test_trace_backend_custom_deserializer():
    backend = TraceOptimizerBackend(
        optimizer_cls=SuffixOptimizer,
        optimizer_kwargs={"suffix": "!"},
        candidate_deserializer=lambda original, proposed: {"old": original, "new": proposed},
    )
    assert backend(candidate="x", feedback="change") == {"old": "x", "new": "x!"}


def test_resolve_optimizer_cls_supports_default_and_string_names_for_available_optimizers():
    assert resolve_optimizer_cls(SuffixOptimizer) is SuffixOptimizer
    assert resolve_optimizer_cls().__name__ in {"OptoPrimeV2", "OptoPrime"}
    for name in ["OptoPrimeV2", "OptoPrime", "OptoPrimeMulti", "OPROv2", "TextGrad"]:
        assert resolve_optimizer_cls(name).__name__ == name


def test_unknown_optimizer_name_has_clear_error():
    with pytest.raises(ValueError, match="Unknown Trace optimizer"):
        resolve_optimizer_cls("DefinitelyMissingOptimizer")


requires_openai = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not available")


@requires_openai
def test_live_gpt5_nano_litellm_helper_smoke():
    pytest.importorskip("litellm")
    os.environ.setdefault("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")
    os.environ.setdefault("TRACE_LITELLM_MODEL", "gpt-5-nano")
    from opto.optimize_anything import make_litellm_lm

    lm = make_litellm_lm(model=os.environ.get("TRACE_LITELLM_MODEL", "gpt-5-nano"), max_retries=1)
    assert callable(lm.model)
    assert getattr(lm, "model_name", None) == os.environ.get("TRACE_LITELLM_MODEL", "gpt-5-nano")


@requires_openai
def test_live_gpt5_nano_trace_backend_protocol_smoke():
    pytest.importorskip("litellm")
    os.environ.setdefault("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")
    os.environ.setdefault("TRACE_LITELLM_MODEL", "gpt-5-nano")
    backend = TraceOptimizerBackend(
        optimizer_cls="OPROv2",
        optimizer_kwargs={"max_tokens": 128, "temperature": 0.0, "llm": None},
    )
    result = optimize_anything(
        seed_candidate="Answer with one short word.",
        evaluator=lambda candidate: 1.0 if isinstance(candidate, str) and candidate else 0.0,
        objective="Keep the instruction concise.",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=2, max_steps=1, capture_stdio=True),
            reflection=ReflectionConfig(custom_candidate_proposer=backend),
        ),
    )
    assert isinstance(result.best_candidate, str)
    assert result.total_metric_calls <= 2
