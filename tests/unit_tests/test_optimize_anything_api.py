import io
import sys
from contextlib import redirect_stdout

import pytest

import opto.optimize_anything as oa


def test_public_api_exports_expected_symbols():
    for name in [
        "optimize_anything", "EngineConfig", "ReflectionConfig", "RefinerConfig",
        "MergeConfig", "TrackingConfig", "GEPAConfig", "OptimizationState",
        "GEPAResult", "EvaluationRecord", "log", "get_log_context",
        "set_log_context", "make_litellm_lm", "TraceOptimizerBackend",
    ]:
        assert hasattr(oa, name)


def test_log_context_does_not_write_to_stdout():
    captured_stdout = io.StringIO()
    captured_logs = []
    token = oa.set_log_context(captured_logs)
    try:
        with redirect_stdout(captured_stdout):
            oa.log("hidden", 1, sep="-")
    finally:
        oa.reset_log_context(token)
    assert captured_stdout.getvalue() == ""
    assert captured_logs == ["hidden-1"]


def test_log_falls_back_to_print_without_context(capsys):
    oa.log("visible", 2)
    assert capsys.readouterr().out == "visible 2\n"


def test_evaluator_supports_stdout_stderr_oa_log_cache_and_opt_state():
    calls = {"n": 0}

    def evaluator(candidate, example, opt_state):
        assert opt_state.candidate == candidate
        print(f"stdout:{example}")
        sys.stderr.write(f"stderr:{example}\n")
        oa.log("structured", example)
        calls["n"] += 1
        return candidate["x"] + example, {"scores": {"x": candidate["x"]}}

    def proposer(candidate, feedback, **kwargs):
        assert "structured" in feedback
        return {"x": candidate["x"] + 1}

    result = oa.optimize_anything(
        seed_candidate={"x": 0},
        evaluator=evaluator,
        dataset=[1, 1],
        objective="increase x",
        config=oa.GEPAConfig(
            engine=oa.EngineConfig(max_metric_calls=10, max_steps=1, capture_stdio=True, cache_evaluation=True),
            reflection=oa.ReflectionConfig(custom_candidate_proposer=proposer),
        ),
    )
    assert calls["n"] == 2
    assert result.best_candidate == {"x": 1}
    assert result.best_score == pytest.approx(2.0)
    assert result.total_metric_calls == 2
    assert any("stdout:1" in r.stdout for r in result.history)
    assert any("stderr:1" in r.stderr for r in result.history)
    assert any("structured" in "\n".join(r.logs) for r in result.history)


@pytest.mark.parametrize("returned,expected", [(1, 1.0), (True, 1.0), (0.25, 0.25), ((0.7, {"a": 1}), 0.7), ((None, {"scores": [0.25, 0.75]}), 0.5)])
def test_evaluator_return_forms(returned, expected):
    result = oa.optimize_anything(seed_candidate="seed", evaluator=lambda candidate: returned, max_metric_calls=1)
    assert result.best_score == pytest.approx(expected)


@pytest.mark.parametrize(
    "evaluator",
    [
        lambda candidate: float(candidate),
        lambda candidate, example: float(candidate + example),
        lambda candidate, example, opt_state: float(candidate + example + opt_state.step),
        lambda c, e, s: float(c + e + s.step),
        lambda candidate, example, *, opt_state: float(candidate + example + opt_state.step),
    ],
)
def test_evaluator_signature_variants_and_opt_state_injection(evaluator):
    result = oa.optimize_anything(seed_candidate=1, evaluator=evaluator, dataset=[2], max_metric_calls=1)
    assert result.best_score >= 1.0


def test_candidate_proposer_can_return_multiple_candidates_and_budget_is_respected():
    def proposer(candidate, **kwargs):
        return [candidate + 1, candidate + 2, candidate + 3]

    result = oa.optimize_anything(
        seed_candidate=0,
        evaluator=lambda candidate: float(candidate),
        config=oa.GEPAConfig(
            engine=oa.EngineConfig(max_metric_calls=3, max_steps=3),
            reflection=oa.ReflectionConfig(custom_candidate_proposer=proposer),
        ),
    )
    assert result.total_metric_calls == 3
    assert result.best_candidate == 2
    assert [r.candidate for r in result.history] == [0, 1, 2]


def test_direct_kwargs_patch_config_for_gepa_style_callsite():
    result = oa.optimize_anything(
        seed_candidate=3,
        evaluator=lambda candidate: float(candidate),
        max_metric_calls=1,
        capture_stdio=True,
        cache_evaluation=False,
    )
    assert result.best_score == pytest.approx(3.0)
    assert result.config.engine.capture_stdio is True
    assert result.config.engine.cache_evaluation is False


def test_stable_cache_handles_unhashable_nested_candidates_and_examples():
    calls = {"n": 0}

    def evaluator(candidate, example):
        calls["n"] += 1
        return float(candidate["values"][0] + example["bias"])

    result = oa.optimize_anything(
        seed_candidate={"values": [1, 2]},
        evaluator=evaluator,
        dataset=[{"bias": 3}, {"bias": 3}],
        config=oa.GEPAConfig(engine=oa.EngineConfig(max_metric_calls=10, max_steps=0, cache_evaluation=True)),
    )
    assert calls["n"] == 1
    assert result.best_score == pytest.approx(4.0)


def test_lower_is_better_selection():
    result = oa.optimize_anything(
        seed_candidate=10,
        evaluator=lambda candidate: float(candidate),
        config=oa.GEPAConfig(
            engine=oa.EngineConfig(max_metric_calls=3, max_steps=2, higher_is_better=False),
            reflection=oa.ReflectionConfig(custom_candidate_proposer=lambda candidate, **kwargs: candidate - 1),
        ),
    )
    assert result.best_candidate == 8
    assert result.best_score == pytest.approx(8.0)


def test_result_to_dict_is_json_like_and_has_validation_aliases():
    result = oa.optimize_anything(
        seed_candidate="x",
        evaluator=lambda candidate, example=None: (1.0, {"scores": {"ok": 1}}),
        valset=[{"heldout": True}],
        config=oa.GEPAConfig(engine=oa.EngineConfig(max_metric_calls=2, max_steps=0)),
    )
    as_dict = result.to_dict()
    assert as_dict["best_candidate"] == "x"
    assert as_dict["best_score"] == 1.0
    assert as_dict["candidate_scores"] == [{"candidate": "x", "score": 1.0}]
    assert as_dict["history"][0]["side_info"] == {"scores": {"ok": 1}}
    assert "validation_records" in as_dict


def test_to_dict_converts_non_json_objects_and_config_callables_to_repr():
    class CustomObject:
        pass

    def proposer(candidate, **kwargs):
        return candidate

    obj = CustomObject()
    result = oa.optimize_anything(
        seed_candidate={"obj": obj},
        evaluator=lambda candidate: (1.0, {"obj": obj}),
        config=oa.GEPAConfig(
            engine=oa.EngineConfig(max_metric_calls=1),
            reflection=oa.ReflectionConfig(custom_candidate_proposer=proposer),
        ),
    )
    data = result.to_dict()
    assert isinstance(data["best_candidate"]["obj"], str)
    assert isinstance(data["history"][0]["side_info"]["obj"], str)
    assert isinstance(data["config"]["reflection"]["custom_candidate_proposer"], str)


def test_cache_key_is_stable_for_sets_and_nested_unhashables():
    calls = {"n": 0}

    def evaluator(candidate, example):
        calls["n"] += 1
        return float(len(candidate["items"]) + len(example["items"]))

    result = oa.optimize_anything(
        seed_candidate={"items": {3, 1, 2}},
        evaluator=evaluator,
        dataset=[{"items": {2, 1}}, {"items": {1, 2}}],
        config=oa.GEPAConfig(
            engine=oa.EngineConfig(max_metric_calls=10, max_steps=0, cache_evaluation=True),
        ),
    )
    assert calls["n"] == 1
    assert len(result.history) == 2
    assert result.total_metric_calls == 1


def test_cache_hits_do_not_consume_budget_after_budget_is_reached():
    calls = {"n": 0}

    def evaluator(candidate, example):
        calls["n"] += 1
        return 1.0

    result = oa.optimize_anything(
        seed_candidate="x",
        evaluator=evaluator,
        dataset=[{"same": [1, 2]}, {"same": [1, 2]}],
        config=oa.GEPAConfig(
            engine=oa.EngineConfig(max_metric_calls=1, max_steps=0, cache_evaluation=True),
        ),
    )
    assert calls["n"] == 1
    assert result.total_metric_calls == 1
    assert len(result.history) == 2
    assert result.history[1].cached is True


def test_evaluator_mixed_known_and_unknown_positional_names():
    def evaluator(candidate, e):
        return float(candidate + e)

    result = oa.optimize_anything(
        seed_candidate=2,
        evaluator=evaluator,
        dataset=[3],
        max_metric_calls=1,
    )
    assert result.best_score == pytest.approx(5.0)


def test_evaluator_keyword_only_opt_state_with_mixed_positional_names():
    seen_steps = []

    def evaluator(candidate, e, *, opt_state):
        seen_steps.append(opt_state.step)
        return float(candidate + e + opt_state.step)

    result = oa.optimize_anything(
        seed_candidate=2,
        evaluator=evaluator,
        dataset=[3],
        max_metric_calls=1,
    )
    assert result.best_score == pytest.approx(5.0)
    assert seen_steps == [0]


def test_proposer_mixed_known_and_unknown_positional_names():
    def evaluator(candidate):
        return float(candidate)

    def proposer(candidate, fb):
        assert "Aggregate score" in fb
        return candidate + 1

    result = oa.optimize_anything(
        seed_candidate=1,
        evaluator=evaluator,
        config=oa.GEPAConfig(
            engine=oa.EngineConfig(max_metric_calls=2, max_steps=1),
            reflection=oa.ReflectionConfig(custom_candidate_proposer=proposer),
        ),
    )
    assert result.best_candidate == 2
    assert result.best_score == pytest.approx(2.0)
