"""Tests for opto.trace.io.otel_semconv."""
import json
import pytest
from opto.trace.io.otel_semconv import (
    set_span_attributes,
    record_genai_chat,
    emit_reward,
    emit_trace,
)
from opto.trace.io.telemetry_session import TelemetrySession


class TestSetSpanAttributes:
    def test_skips_none(self):
        session = TelemetrySession("test-semconv")
        with session.tracer.start_as_current_span("test") as sp:
            set_span_attributes(sp, {"key1": "val1", "key2": None})
        otlp = session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        attrs = {a["key"]: a["value"]["stringValue"] for a in spans[0]["attributes"]}
        assert "key1" in attrs
        assert "key2" not in attrs

    def test_serializes_dict(self):
        session = TelemetrySession("test-semconv")
        with session.tracer.start_as_current_span("test") as sp:
            set_span_attributes(sp, {"data": {"nested": True}})
        otlp = session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        attrs = {a["key"]: a["value"]["stringValue"] for a in spans[0]["attributes"]}
        parsed = json.loads(attrs["data"])
        assert parsed == {"nested": True}


class TestRecordGenaiChat:
    def test_emits_genai_attributes(self):
        session = TelemetrySession("test-genai")
        with session.tracer.start_as_current_span("llm_call") as sp:
            record_genai_chat(
                sp,
                provider="openrouter",
                model="llama-3.1",
                input_messages=[{"role": "user", "content": "hello"}],
                output_text="world",
            )
        otlp = session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        attrs = {a["key"]: a["value"]["stringValue"] for a in spans[0]["attributes"]}
        assert attrs["gen_ai.operation.name"] == "chat.completion"
        assert attrs["gen_ai.provider.name"] == "openrouter"
        assert attrs["gen_ai.request.model"] == "llama-3.1"
        assert "gen_ai.input.messages" in attrs
        assert "gen_ai.output.messages" in attrs


class TestEmitReward:
    def test_creates_reward_span(self):
        session = TelemetrySession("test-reward")
        emit_reward(session, value=0.85, name="accuracy")
        otlp = session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 1
        attrs = {a["key"]: a["value"]["stringValue"] for a in spans[0]["attributes"]}
        assert attrs["trace.temporal_ignore"] == "true"
        assert attrs["agentlightning.reward.0.name"] == "accuracy"
        assert attrs["agentlightning.reward.0.value"] == "0.85"


class TestEmitTrace:
    def test_creates_custom_span(self):
        session = TelemetrySession("test-trace")
        emit_trace(session, name="my_signal", attrs={"custom_key": "custom_val"})
        otlp = session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 1
        assert spans[0]["name"] == "my_signal"
        attrs = {a["key"]: a["value"]["stringValue"] for a in spans[0]["attributes"]}
        assert attrs["custom_key"] == "custom_val"
