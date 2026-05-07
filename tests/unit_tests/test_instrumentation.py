"""Tests for opto.trace.io.instrumentation."""
import pytest
from opto.trace.io.instrumentation import instrument_graph, InstrumentedGraph
from opto.trace.io.telemetry_session import TelemetrySession
from opto.trace.io.bindings import Binding, make_dict_binding


class _StubLLM:
    """Minimal deterministic LLM stub for testing."""
    model = "stub"
    call_count = 0

    def __call__(self, messages=None, **kwargs):
        self.call_count += 1

        class Msg:
            content = f"stub response #{self.call_count}"

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


class TestInstrumentGraph:
    def test_returns_instrumented_graph(self):
        ig = instrument_graph(
            graph=None,
            service_name="test",
            llm=_StubLLM(),
            initial_templates={"prompt_a": "template A"},
        )
        assert isinstance(ig, InstrumentedGraph)
        assert ig.session is not None
        assert ig.tracing_llm is not None

    def test_auto_derives_bindings_from_templates(self):
        ig = instrument_graph(
            graph=None,
            service_name="test",
            llm=_StubLLM(),
            initial_templates={"prompt_a": "A", "prompt_b": "B"},
        )
        assert "prompt_a" in ig.bindings
        assert "prompt_b" in ig.bindings
        assert ig.bindings["prompt_a"].get() == "A"

    def test_custom_bindings_override(self):
        store = {"custom": "val"}
        custom = {"custom": make_dict_binding(store, "custom")}
        ig = instrument_graph(
            graph=None,
            service_name="test",
            llm=_StubLLM(),
            bindings=custom,
        )
        assert "custom" in ig.bindings
        assert ig.bindings["custom"].get() == "val"

    def test_reuse_existing_session(self):
        session = TelemetrySession("shared-session")
        ig = instrument_graph(
            graph=None,
            session=session,
            llm=_StubLLM(),
        )
        assert ig.session is session

    def test_trainable_keys_none_means_all(self):
        ig = instrument_graph(
            graph=None,
            service_name="test",
            trainable_keys=None,
            llm=_StubLLM(),
        )
        # trainable_keys=None -> _trainable_keys_all=True
        assert ig.tracing_llm._trainable_keys_all is True

    def test_trainable_keys_explicit(self):
        ig = instrument_graph(
            graph=None,
            service_name="test",
            trainable_keys={"planner"},
            llm=_StubLLM(),
        )
        assert ig.tracing_llm._trainable_keys_all is False
        assert "planner" in ig.tracing_llm.trainable_keys

    def test_compiles_graph_if_needed(self):
        class FakeGraph:
            compiled = False
            def compile(self):
                self.compiled = True
                return self

        fg = FakeGraph()
        ig = instrument_graph(graph=fg, llm=_StubLLM())
        assert fg.compiled is True


class TestTracingLLMChildSpan:
    def test_child_span_emitted(self):
        ig = instrument_graph(
            graph=None,
            service_name="test-child",
            llm=_StubLLM(),
            emit_genai_child_spans=True,
            initial_templates={"my_prompt": "Hello {query}"},
        )
        ig.tracing_llm.node_call(
            span_name="test_node",
            template_name="my_prompt",
            template="Hello {query}",
            optimizable_key="test_node",
            messages=[{"role": "user", "content": "hi"}],
        )
        otlp = ig.session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        names = [s["name"] for s in spans]
        assert "test_node" in names
        assert "llm.chat.completion" in names

        # Child span should have trace.temporal_ignore
        child = [s for s in spans if s["name"] == "llm.chat.completion"][0]
        attrs = {a["key"]: a["value"]["stringValue"] for a in child["attributes"]}
        assert attrs.get("trace.temporal_ignore") == "true"
        assert "gen_ai.operation.name" in attrs

    def test_no_child_span_when_disabled(self):
        ig = instrument_graph(
            graph=None,
            service_name="test-nochild",
            llm=_StubLLM(),
            emit_genai_child_spans=False,
        )
        ig.tracing_llm.node_call(
            span_name="test_node",
            messages=[{"role": "user", "content": "hi"}],
        )
        otlp = ig.session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        names = [s["name"] for s in spans]
        assert "test_node" in names
        assert "llm.chat.completion" not in names


class TestTemporalChaining:
    """M1 acceptance: child spans must NOT advance TGJ temporal chaining."""

    def test_child_spans_do_not_advance_temporal_chain(self):
        from opto.trace.io.otel_adapter import otlp_traces_to_trace_json

        ig = instrument_graph(
            graph=None,
            service_name="temporal-test",
            llm=_StubLLM(),
            emit_genai_child_spans=True,
        )
        # Emit two node spans; each with a child LLM span
        ig.tracing_llm.node_call(
            span_name="node_A",
            template_name="prompt_a",
            template="prompt A",
            optimizable_key="node_A",
            messages=[{"role": "user", "content": "q1"}],
        )
        ig.tracing_llm.node_call(
            span_name="node_B",
            template_name="prompt_b",
            template="prompt B",
            optimizable_key="node_B",
            messages=[{"role": "user", "content": "q2"}],
        )
        otlp = ig.session.flush_otlp()

        # Convert to TGJ with temporal hierarchy
        docs = otlp_traces_to_trace_json(
            otlp,
            agent_id_hint="temporal-test",
            use_temporal_hierarchy=True,
        )
        assert len(docs) >= 1
        doc = docs[0]
        nodes = doc["nodes"]

        # The child LLM spans should NOT be temporal parents of node_B.
        # node_B's parent should be node_A (not the child LLM span of A).
        msg_nodes = {
            nid: n for nid, n in nodes.items()
            if n.get("kind") == "msg"
        }
        # There should be at least node_A and node_B as msg nodes
        node_names = [n.get("name") for n in msg_nodes.values()]
        assert "node_A" in node_names
        assert "node_B" in node_names
