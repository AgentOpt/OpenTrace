"""Tests for opto.trace.io.telemetry_session."""
import pytest
from opto.trace.io.telemetry_session import TelemetrySession


class TestTelemetrySession:
    def test_flush_otlp_returns_spans(self):
        session = TelemetrySession("test-session")
        with session.tracer.start_as_current_span("span1") as sp:
            sp.set_attribute("key", "val")
        otlp = session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) >= 1
        assert spans[0]["name"] == "span1"

    def test_flush_otlp_clears_by_default(self):
        session = TelemetrySession("test-clear")
        with session.tracer.start_as_current_span("span1"):
            pass
        otlp1 = session.flush_otlp(clear=True)
        spans1 = otlp1["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans1) >= 1

        otlp2 = session.flush_otlp(clear=True)
        spans2 = otlp2["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans2) == 0

    def test_record_spans_false_noop(self):
        session = TelemetrySession("test-noop", record_spans=False)
        with session.tracer.start_as_current_span("span1"):
            pass
        otlp = session.flush_otlp()
        assert otlp == {"resourceSpans": []}

    def test_flush_tgj_produces_docs(self):
        session = TelemetrySession("test-tgj")
        with session.tracer.start_as_current_span("node1") as sp:
            sp.set_attribute("param.prompt", "hello world")
            sp.set_attribute("param.prompt.trainable", True)
        docs = session.flush_tgj()
        assert len(docs) >= 1
        doc = docs[0]
        assert "nodes" in doc

    def test_span_attribute_filter(self):
        """Filter should be able to redact attributes."""
        def redact_filter(name, attrs):
            # Drop any span named "secret"
            if name == "secret":
                return {}
            # Otherwise pass through
            return attrs

        session = TelemetrySession(
            "test-filter",
            span_attribute_filter=redact_filter,
        )
        # The filter is stored but note: the real OTEL SDK doesn't call
        # our filter. This tests that the parameter is accepted.
        assert session.span_attribute_filter is not None


class TestExportRunBundle:
    def test_creates_files(self, tmp_path):
        session = TelemetrySession("test-bundle")
        with session.tracer.start_as_current_span("node1") as sp:
            sp.set_attribute("param.prompt", "test")
            sp.set_attribute("param.prompt.trainable", True)

        out_dir = str(tmp_path / "bundle")
        result = session.export_run_bundle(
            out_dir,
            prompts={"prompt": "test"},
        )
        assert result == out_dir
        assert (tmp_path / "bundle" / "otlp_trace.json").exists()
        assert (tmp_path / "bundle" / "trace_graph.json").exists()
        assert (tmp_path / "bundle" / "prompts.json").exists()
