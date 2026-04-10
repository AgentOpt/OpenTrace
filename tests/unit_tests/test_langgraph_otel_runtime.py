import pytest

from opto.trace.io.langgraph_otel_runtime import (
    init_otel_runtime,
    TracingLLM,
    flush_otlp,
    extract_eval_metrics_from_otlp,
)


class FakeLLM:
    """
    Minimal LLM stub compatible with the TracingLLM expectations.
    """

    class _Message:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = FakeLLM._Message(content)

    class _Response:
        def __init__(self, content: str) -> None:
            self.choices = [FakeLLM._Choice(content)]

    def __init__(self, content: str = "OK") -> None:
        self.content = content
        self.calls = []

    def __call__(self, messages=None, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return FakeLLM._Response(self.content)


def _attrs_to_dict(attrs):
    return {a["key"]: a["value"]["stringValue"] for a in attrs}


def test_tracing_llm_records_prompt_and_user_query():
    tracer, exporter = init_otel_runtime("test-llm")
    llm = FakeLLM("ANSWER")
    tllm = TracingLLM(
        llm=llm, tracer=tracer, trainable_keys={"planner"},
        emit_llm_child_span=False,  # test focuses on the node span only
    )

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What is 2+2?"},
    ]

    result = tllm.node_call(
        span_name="planner",
        template_name="planner_prompt",
        template="Plan for: {query}",
        optimizable_key="planner",
        code_key=None,
        code_fn=None,
        user_query="What is 2+2?",
        messages=messages,
    )

    assert result == "ANSWER"
    assert len(llm.calls) == 1

    otlp = flush_otlp(exporter, scope_name="test-llm")
    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1
    span = spans[0]
    assert span["name"] == "planner"
    attrs = _attrs_to_dict(span["attributes"])

    # prompt + trainable flag
    assert attrs["param.planner_prompt"] == "Plan for: {query}"
    # trainable flag is a bool string; be tolerant to case
    assert attrs["param.planner_prompt.trainable"].lower() in ("true", "1")

    # inputs.*
    assert attrs["inputs.user_query"] == "What is 2+2?"
    assert attrs["inputs.gen_ai.prompt"] == "What is 2+2?"


def test_tracing_llm_trainable_flag_respects_keys():
    tracer, exporter = init_otel_runtime("test-llm-trainable")
    llm = FakeLLM("OK")
    tllm = TracingLLM(llm=llm, tracer=tracer, trainable_keys=set())

    messages = [{"role": "user", "content": "check"}]
    _ = tllm.node_call(
        span_name="planner",
        template_name="planner_prompt",
        template="Plan for: {query}",
        optimizable_key="planner",  # NOT in trainable_keys
        code_key=None,
        code_fn=None,
        user_query="check",
        messages=messages,
    )

    otlp = flush_otlp(exporter, scope_name="test-llm-trainable")
    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    attrs = _attrs_to_dict(spans[0]["attributes"])

    # Either missing or explicitly false; both are acceptable
    value = attrs.get("param.planner_prompt.trainable")
    assert value is None or value.lower() in ("false", "0")


def test_flush_otlp_clears_exporter():
    tracer, exporter = init_otel_runtime("test-flush")
    llm = FakeLLM("OK")
    tllm = TracingLLM(llm=llm, tracer=tracer)

    messages = [{"role": "user", "content": "ping"}]
    _ = tllm.node_call(span_name="planner", messages=messages)

    # We should have spans before flush
    assert exporter.get_finished_spans()

    _ = flush_otlp(exporter, scope_name="test-flush")
    assert exporter.get_finished_spans() == []


def test_extract_eval_metrics_from_otlp_happy_path():
    # Synthetic OTLP payload with a single evaluator span
    otlp = {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "scope": {"name": "demo"},
                        "spans": [
                            {
                                "name": "evaluator",
                                "attributes": [
                                    {"key": "eval.score", "value": {"stringValue": "0.9"}},
                                    {"key": "eval.answer_relevance", "value": {"stringValue": "0.8"}},
                                    {"key": "eval.groundedness", "value": {"stringValue": "0.7"}},
                                    {"key": "eval.plan_quality", "value": {"stringValue": "0.6"}},
                                    {"key": "eval.reasons", "value": {"stringValue": "good"}},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    # Pass explicit metric_keys matching the synthetic payload
    custom_keys = {
        "answer_relevance": "eval.answer_relevance",
        "groundedness": "eval.groundedness",
        "plan_quality": "eval.plan_quality",
    }
    score, metrics, reasons = extract_eval_metrics_from_otlp(
        otlp, metric_keys=custom_keys
    )
    assert score == 0.9
    assert metrics["answer_relevance"] == 0.8
    assert metrics["groundedness"] == 0.7
    assert metrics["plan_quality"] == 0.6
    assert reasons == "good"


def test_extract_eval_metrics_from_otlp_defaults_when_missing():
    # No evaluator span at all -> fall back to defaults (still usable)
    otlp = {"resourceSpans": []}

    score, metrics, reasons = extract_eval_metrics_from_otlp(otlp)

    # Default score is in [0,1] and we get non-empty metric dict.
    assert 0.0 <= score <= 1.0
    assert metrics
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
    assert reasons == ""
