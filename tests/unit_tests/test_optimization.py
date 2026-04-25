"""Tests for opto.trace.io.optimization."""
import pytest
from opto.trace.io.optimization import (
    EvalResult,
    _normalise_eval,
    RunResult,
    OptimizationResult,
)


class TestEvalResult:
    def test_defaults(self):
        er = EvalResult()
        assert er.score is None
        assert er.feedback == ""
        assert er.metrics == {}

    def test_with_values(self):
        er = EvalResult(score=0.8, feedback="good", metrics={"acc": 0.9})
        assert er.score == 0.8


class TestNormaliseEval:
    def test_from_float(self):
        er = _normalise_eval(0.75)
        assert er.score == 0.75
        assert er.feedback == ""

    def test_from_int(self):
        er = _normalise_eval(1)
        assert er.score == 1.0

    def test_from_string_feedback(self):
        er = _normalise_eval("needs improvement")
        assert er.score is None
        assert er.feedback == "needs improvement"

    def test_from_json_string(self):
        import json
        raw = json.dumps({"score": 0.9, "reasons": "well done"})
        er = _normalise_eval(raw)
        assert er.score == 0.9
        assert "well done" in er.feedback

    def test_from_dict(self):
        er = _normalise_eval({"score": 0.6, "feedback": "ok", "extra": 1})
        assert er.score == 0.6
        assert er.feedback == "ok"

    def test_from_eval_result(self):
        original = EvalResult(score=0.5, feedback="test")
        er = _normalise_eval(original)
        assert er is original

    def test_from_unknown(self):
        er = _normalise_eval(42.0)
        assert er.score == 42.0


class TestRunResult:
    def test_fields(self):
        rr = RunResult(
            answer="hello",
            score=0.8,
            feedback="good",
            metrics={"acc": 0.9},
            otlp={"resourceSpans": []},
        )
        assert rr.answer == "hello"
        assert rr.score == 0.8
        assert rr.artifacts == {}


class TestOptimizationResult:
    def test_fields(self):
        result = OptimizationResult(
            baseline_score=0.5,
            best_score=0.8,
            best_iteration=2,
            best_parameters={"prompt": "best"},
            best_updates={"prompt": "new"},
            final_parameters={"prompt": "new"},
            score_history=[0.5, 0.6, 0.8],
            all_runs=[],
        )
        assert result.best_score == 0.8
        assert result.best_iteration == 2
        assert result.best_parameters == {"prompt": "best"}
