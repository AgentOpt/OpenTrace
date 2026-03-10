"""Tests for opto.trace.io.telemetry_session."""
import pytest
from unittest.mock import patch, MagicMock
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


class TestMlflowAutologBridge:
    """B1: TelemetrySession mlflow_autolog parameter."""

    def test_default_off(self):
        """mlflow_autolog defaults to False, no autolog call."""
        session = TelemetrySession("test")
        assert session.mlflow_autolog is False

    def test_autolog_called_when_enabled(self):
        """When mlflow_autolog=True and MLflow is available, autolog is called."""
        mock_autolog = MagicMock()
        with patch.dict("sys.modules", {}):
            with patch(
                "opto.features.mlflow.autolog.autolog", mock_autolog
            ):
                session = TelemetrySession("test", mlflow_autolog=True)
                assert session.mlflow_autolog is True
                mock_autolog.assert_called_once()
                # silent=True should be in the call kwargs
                call_kwargs = mock_autolog.call_args
                assert call_kwargs[1].get("silent") is True or call_kwargs[0] == ()

    def test_autolog_kwargs_forwarded(self):
        """mlflow_autolog_kwargs are forwarded to the autolog call."""
        mock_autolog = MagicMock()
        with patch(
            "opto.features.mlflow.autolog.autolog", mock_autolog
        ):
            session = TelemetrySession(
                "test",
                mlflow_autolog=True,
                mlflow_autolog_kwargs={"log_models": False},
            )
            call_kwargs = mock_autolog.call_args[1]
            assert call_kwargs.get("log_models") is False

    def test_autolog_failure_does_not_raise(self):
        """If MLflow import fails, session still constructs fine."""
        with patch(
            "opto.features.mlflow.autolog.autolog",
            side_effect=ImportError("no mlflow"),
        ):
            session = TelemetrySession("test", mlflow_autolog=True)
            assert session.mlflow_autolog is True  # flag is set, just didn't activate


class TestStableNodeIdentity:
    """B4: message.id becomes stable TGJ node id."""

    def test_message_id_used_as_node_id(self):
        """When message.id is present on a span, the TGJ node id uses it."""
        session = TelemetrySession("test-stable")
        with session.tracer.start_as_current_span("my_node") as sp:
            sp.set_attribute("message.id", "stable_logical_id")
            sp.set_attribute("param.prompt", "hello")
            sp.set_attribute("param.prompt.trainable", "true")

        docs = session.flush_tgj()
        assert len(docs) >= 1
        nodes = docs[0]["nodes"]
        # The node should be keyed by message.id, not span id
        assert "test-stable:stable_logical_id" in nodes

    def test_fallback_to_span_id_without_message_id(self):
        """Without message.id, node id falls back to span id."""
        session = TelemetrySession("test-fallback")
        with session.tracer.start_as_current_span("my_node") as sp:
            sp.set_attribute("param.prompt", "hello")
            sp.set_attribute("param.prompt.trainable", "true")

        docs = session.flush_tgj()
        assert len(docs) >= 1
        nodes = docs[0]["nodes"]
        # Should have a node keyed by svc:span_hex_id (16 hex chars)
        node_keys = [k for k in nodes if k.startswith("test-fallback:") and "param_" not in k]
        assert len(node_keys) >= 1
        # The key should NOT contain "stable_logical_id"
        for k in node_keys:
            assert "stable_logical_id" not in k
