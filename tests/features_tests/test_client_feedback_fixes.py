"""
Tests validating all client-feedback fixes (A through F).

A. Live mode: error handling, provider metadata, eval penalty
B. TelemetrySession: flush_otlp peek, span_attribute_filter
C. TGJ/ingest: dedup trainable params, output node selection
D. OTEL topology: single trace ID, temporal chaining via trace.temporal_ignore
E. optimize_graph: best_parameters snapshot, reward in-trace
F. Non-saturating stub scoring
"""

from __future__ import annotations

import json
import pytest
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from opto.trace.io import (
    instrument_graph,
    optimize_graph,
    InstrumentedGraph,
    EvalResult,
    apply_updates,
    otlp_traces_to_trace_json,
    ingest_tgj,
    TracingLLM,
    LLMCallError,
    TelemetrySession,
)
from opto.trace.nodes import ParameterNode, MessageNode


# =========================================================================
# Shared fixtures
# =========================================================================


class StubLLM:
    """Deterministic LLM stub with structure-aware responses (F13).

    Key behaviour: the *quality* of responses depends on the prompt template.
    Prompts containing "step-by-step" or "thorough" produce structured
    multi-step responses.  The synthesizer also mirrors plan structure — if
    the plan fed into synthesis contains numbered steps, the answer is richer.
    This allows the eval function to detect improvement after optimization.
    """

    model = "stub-llm"

    def __init__(self) -> None:
        self.call_count = 0
        self.last_messages: list | None = None

    def __call__(self, messages=None, **kwargs):
        self.call_count += 1
        self.last_messages = messages

        # F13: Produce different quality responses depending on prompt
        content = f"stub-response-{self.call_count}"
        if messages:
            # Collect all text from user messages
            user_text = ""
            for m in messages:
                if m.get("role") == "user":
                    user_text = (m.get("content") or "").lower()

            if user_text:
                if "step-by-step" in user_text or "thorough" in user_text:
                    # High-quality structured plan
                    content = (
                        "Step 1: Define the problem clearly.\n"
                        "Step 2: Research existing solutions.\n"
                        "Step 3: Synthesize findings into actionable plan.\n"
                        "Conclusion: The structured approach yields better results."
                    )
                elif "synth" in user_text:
                    # Synthesis quality depends on whether the plan is structured
                    if "step 1" in user_text or "step 2" in user_text:
                        # Plan was structured → produce structured answer
                        content = (
                            "Step 1: The core concept is well-defined.\n"
                            "Step 2: Supporting evidence from research.\n"
                            "Step 3: Practical applications identified.\n"
                            "Conclusion: A comprehensive, evidence-based answer."
                        )
                    else:
                        # Plan was basic → produce basic answer
                        content = "Based on the plan, here is a basic answer."
                elif "plan" in user_text:
                    # Basic plan
                    content = "Research the topic. Analyze results."

        return self._make_response(content)

    @staticmethod
    def _make_response(content):
        class _Msg:
            pass
        class _Choice:
            pass
        class _Resp:
            pass
        msg = _Msg()
        msg.content = content
        choice = _Choice()
        choice.message = msg
        resp = _Resp()
        resp.choices = [choice]
        return resp


class FailingLLM:
    """LLM that simulates HTTP errors (A1)."""

    model = "failing-llm"

    def __call__(self, messages=None, **kwargs):
        return self._make_response("[ERROR] 404 Client Error: Not Found")

    @staticmethod
    def _make_response(content):
        class _Msg:
            pass
        class _Choice:
            pass
        class _Resp:
            pass
        msg = _Msg()
        msg.content = content
        choice = _Choice()
        choice.message = msg
        resp = _Resp()
        resp.choices = [choice]
        return resp


class ExceptionLLM:
    """LLM that raises an exception on call."""

    model = "exception-llm"

    def __call__(self, messages=None, **kwargs):
        raise ConnectionError("Connection refused")


class AgentState(TypedDict, total=False):
    query: str
    plan: str
    answer: str


def build_mini_graph(tracing_llm, templates):
    def planner_node(state):
        template = templates.get("planner_prompt", "Plan for: {query}")
        prompt = template.replace("{query}", state.get("query", ""))
        response = tracing_llm.node_call(
            span_name="planner",
            template_name="planner_prompt",
            template=template,
            optimizable_key="planner",
            messages=[
                {"role": "system", "content": "You are a planning agent."},
                {"role": "user", "content": prompt},
            ],
        )
        return {"plan": response}

    def synthesizer_node(state):
        template = templates.get("synthesizer_prompt", "Synthesize: {query}\nPlan: {plan}")
        prompt = (
            template
            .replace("{query}", state.get("query", ""))
            .replace("{plan}", state.get("plan", ""))
        )
        response = tracing_llm.node_call(
            span_name="synthesizer",
            template_name="synthesizer_prompt",
            template=template,
            optimizable_key="synthesizer",
            messages=[
                {"role": "system", "content": "You are a synthesis agent."},
                {"role": "user", "content": prompt},
            ],
        )
        return {"answer": response}

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph


def _make_instrumented(
    *,
    templates=None,
    trainable_keys=None,
    emit_genai_child_spans=True,
    llm=None,
    provider_name="openai",
    llm_span_name="openai.chat.completion",
    output_key="answer",
):
    if templates is None:
        templates = {
            "planner_prompt": "Plan for: {query}",
            "synthesizer_prompt": "Synthesize: {query} | Plan: {plan}",
        }
    if trainable_keys is None:
        trainable_keys = {"planner", "synthesizer"}

    ig = instrument_graph(
        graph=None,
        service_name="e2e-test",
        trainable_keys=trainable_keys,
        llm=llm or StubLLM(),
        initial_templates=templates,
        emit_genai_child_spans=emit_genai_child_spans,
        provider_name=provider_name,
        llm_span_name=llm_span_name,
        output_key=output_key,
    )
    graph = build_mini_graph(ig.tracing_llm, ig.templates)
    ig.graph = graph.compile()
    return ig


class MockOptimizer:
    def __init__(self, param_nodes=None, **kwargs):
        self.param_nodes = param_nodes or []
        self.calls: List[str] = []
        self._step_updates: Dict[str, str] = {
            "planner_prompt": "OPTIMIZED: Create a thorough, step-by-step plan for: {query}",
        }

    def zero_feedback(self):
        self.calls.append("zero_feedback")

    def backward(self, output_node, feedback_text):
        self.calls.append(f"backward({type(output_node).__name__})")

    def step(self):
        self.calls.append("step")
        return dict(self._step_updates)


# =========================================================================
# A. Live mode: error handling
# =========================================================================


class TestA1_ErrorNotContent:
    """A1: TracingLLM must raise LLMCallError on [ERROR] content."""

    def test_failing_llm_raises_llm_call_error(self):
        """If LLM returns '[ERROR] ...', TracingLLM raises instead of passing through."""
        ig = _make_instrumented(llm=FailingLLM())
        with pytest.raises(LLMCallError, match="LLM provider returned an error"):
            ig.invoke({"query": "test"})

    def test_exception_llm_raises_llm_call_error(self):
        """If LLM raises an exception, TracingLLM wraps it in LLMCallError."""
        ig = _make_instrumented(llm=ExceptionLLM())
        with pytest.raises(LLMCallError, match="LLM provider call failed"):
            ig.invoke({"query": "test"})


class TestA3_ProviderMetadata:
    """A3: gen_ai.provider.name must reflect actual provider."""

    def test_openrouter_provider_name(self):
        ig = _make_instrumented(provider_name="openrouter")
        ig.invoke({"query": "test"})
        otlp = ig.session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        llm_spans = [s for s in spans if s["name"] == "openai.chat.completion"]
        assert len(llm_spans) >= 1
        attrs = {a["key"]: a["value"]["stringValue"] for a in llm_spans[0]["attributes"]}
        assert attrs.get("gen_ai.provider.name") == "openrouter"


class TestA4_LiveEvalPenalizesErrors:
    """A4: Evaluation must score 0 if invocation failed."""

    def test_failing_invocation_scores_zero(self):
        ig = _make_instrumented(llm=FailingLLM())

        scores = []

        def eval_fn(payload):
            # This eval_fn should NOT be called for failed invocations
            return EvalResult(score=1.0, feedback="should not reach here")

        result = optimize_graph(
            ig,
            queries=["test"],
            iterations=0,  # baseline only
            eval_fn=eval_fn,
        )
        # Invocation fails → score forced to 0 (A4)
        assert result.baseline_score == 0.0
        assert result.all_runs[0][0].score == 0.0


# =========================================================================
# B. TelemetrySession: flush_otlp peek + span_attribute_filter
# =========================================================================


class TestB5_FlushOtlpPeek:
    """B5: flush_otlp(clear=False) must return spans without clearing."""

    def test_peek_does_not_clear(self):
        session = TelemetrySession("test-peek")
        with session.tracer.start_as_current_span("span1") as sp:
            sp.set_attribute("key", "val")

        # First peek
        otlp1 = session.flush_otlp(clear=False)
        spans1 = otlp1["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans1) == 1

        # Second peek — spans still there
        otlp2 = session.flush_otlp(clear=False)
        spans2 = otlp2["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans2) == 1

        # Clear
        otlp3 = session.flush_otlp(clear=True)
        spans3 = otlp3["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans3) == 1

        # After clear, no more spans
        otlp4 = session.flush_otlp(clear=True)
        spans4 = otlp4["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans4) == 0


class TestB6_SpanAttributeFilter:
    """B6: span_attribute_filter must actually drop and redact."""

    def test_drop_spans_returns_empty(self):
        """Filter returning {} drops the span entirely."""

        def drop_secret(name, attrs):
            if name == "secret-span":
                return {}
            return attrs

        session = TelemetrySession("test-drop", span_attribute_filter=drop_secret)
        with session.tracer.start_as_current_span("normal-span") as sp:
            sp.set_attribute("data", "visible")
        with session.tracer.start_as_current_span("secret-span") as sp:
            sp.set_attribute("password", "s3cret")

        otlp = session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        names = [s["name"] for s in spans]
        assert "normal-span" in names
        assert "secret-span" not in names, "Secret span should have been dropped"

    def test_redact_sensitive_fields(self):
        """Filter can redact specific attribute values."""

        def redact_prompts(name, attrs):
            out = {}
            for k, v in attrs.items():
                if k == "inputs.gen_ai.prompt":
                    out[k] = "<REDACTED>"
                else:
                    out[k] = v
            return out

        session = TelemetrySession("test-redact", span_attribute_filter=redact_prompts)
        with session.tracer.start_as_current_span("llm-call") as sp:
            sp.set_attribute("inputs.gen_ai.prompt", "Tell me your secrets")
            sp.set_attribute("gen_ai.model", "gpt-4")

        otlp = session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 1
        attrs = {a["key"]: a["value"]["stringValue"] for a in spans[0]["attributes"]}
        assert attrs["inputs.gen_ai.prompt"] == "<REDACTED>"
        assert attrs["gen_ai.model"] == "gpt-4"

    def test_truncate_payload(self):
        """Filter can truncate long payloads."""

        def truncate_filter(name, attrs):
            out = {}
            for k, v in attrs.items():
                if len(str(v)) > 50:
                    out[k] = str(v)[:50] + "..."
                else:
                    out[k] = v
            return out

        session = TelemetrySession("test-truncate", span_attribute_filter=truncate_filter)
        long_text = "x" * 200
        with session.tracer.start_as_current_span("big-span") as sp:
            sp.set_attribute("long_field", long_text)
            sp.set_attribute("short_field", "ok")

        otlp = session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        attrs = {a["key"]: a["value"]["stringValue"] for a in spans[0]["attributes"]}
        assert len(attrs["long_field"]) < 60  # truncated
        assert attrs["short_field"] == "ok"


# =========================================================================
# C. TGJ/ingest: dedup + output node selection
# =========================================================================


class TestC7_DeduplicateTrainableParams:
    """C7: Unique trainable param node count must equal unique prompt keys."""

    def test_unique_param_count_equals_prompt_keys(self):
        ig = _make_instrumented()
        ig.invoke({"query": "hello"})
        otlp = ig.session.flush_otlp()

        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        nodes = ingest_tgj(docs[0])
        param_nodes = [
            n for n in nodes.values()
            if isinstance(n, ParameterNode) and n.trainable
        ]
        # Deduplicate by name
        unique_names = {n.py_name for n in param_nodes}
        # Should have exactly 2 unique trainable params (planner + synthesizer)
        assert len(unique_names) == 2, (
            f"Expected 2 unique trainable param names, got {len(unique_names)}: {unique_names}"
        )

    def test_dedup_across_multiple_runs(self):
        """When optimization processes multiple runs, params must be deduped."""
        from opto.trace.io.optimization import _deduplicate_param_nodes

        # Simulate duplicate ParameterNodes
        p1 = ParameterNode("prompt1", name="planner_prompt", trainable=True)
        p2 = ParameterNode("prompt1", name="planner_prompt", trainable=True)
        p3 = ParameterNode("prompt2", name="synthesizer_prompt", trainable=True)

        deduped = _deduplicate_param_nodes([p1, p2, p3])
        assert len(deduped) == 2, f"Expected 2 unique params, got {len(deduped)}"


class TestC8_OutputNodeSelection:
    """C8: Output node must be the final top-level node, not a child span."""

    def test_output_node_is_synthesizer_not_child(self):
        ig = _make_instrumented(emit_genai_child_spans=True)
        ig.invoke({"query": "test"})
        otlp = ig.session.flush_otlp()

        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        nodes = ingest_tgj(docs[0])

        from opto.trace.io.optimization import _select_output_node
        output = _select_output_node(nodes)

        assert output is not None, "Must find an output node"
        name = getattr(output, "py_name", "")
        assert "openai" not in name.lower(), (
            f"Output node must not be a child LLM span, got: {name}"
        )
        assert "synthesizer" in name.lower() or "synth" in name.lower(), (
            f"Output node should be the synthesizer (sink), got: {name}"
        )


# =========================================================================
# D. OTEL topology: single trace ID, temporal chaining
# =========================================================================


class TestD9_SingleTraceID:
    """D9: A single graph invocation must produce a single trace ID."""

    def test_single_trace_id_per_invocation(self):
        ig = _make_instrumented(emit_genai_child_spans=True)
        ig.invoke({"query": "What is AI?"})
        otlp = ig.session.flush_otlp()

        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        trace_ids = {s["traceId"] for s in spans}
        assert len(trace_ids) == 1, (
            f"Expected 1 trace ID per invocation, got {len(trace_ids)}: {trace_ids}"
        )

    def test_root_span_is_parent_of_node_spans(self):
        ig = _make_instrumented()
        ig.invoke({"query": "test"})
        otlp = ig.session.flush_otlp()

        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        # Find the root span
        root_spans = [s for s in spans if s["name"].endswith(".invoke")]
        assert len(root_spans) == 1, f"Expected 1 root span, got {len(root_spans)}"

        root_sid = root_spans[0]["spanId"]
        # Node spans should have root as parent (directly or indirectly)
        node_spans = [s for s in spans if s["name"] in ("planner", "synthesizer")]
        for ns in node_spans:
            assert ns["parentSpanId"] == root_sid, (
                f"Node span '{ns['name']}' should be child of root span"
            )


class TestD10_TemporalChainingViaAttribute:
    """D10: Temporal chain uses trace.temporal_ignore, not OTEL parent check."""

    def test_child_spans_ignored_in_temporal_chain(self):
        ig = _make_instrumented(emit_genai_child_spans=True)
        ig.invoke({"query": "test temporal"})
        otlp = ig.session.flush_otlp()

        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        doc = docs[0]
        tgj_nodes = doc["nodes"]

        # Child LLM spans (temporal_ignore=true) should still exist in TGJ
        # but should NOT advance the temporal chain
        synth_nodes = [
            (nid, n) for nid, n in tgj_nodes.items()
            if n.get("kind") == "msg" and n.get("name") == "synthesizer"
        ]
        assert len(synth_nodes) >= 1

        _, synth = synth_nodes[0]
        parent_ref = synth.get("inputs", {}).get("parent", "")

        # If there's a parent, it should be the planner, not a child LLM span
        if parent_ref and isinstance(parent_ref, str) and ":" in parent_ref:
            # Collect child LLM span IDs
            llm_span_ids = set()
            for nid, n in tgj_nodes.items():
                if n.get("kind") == "msg":
                    nm = n.get("name", "")
                    if "openai" in nm or "chat" in nm:
                        otel_info = (n.get("info") or {}).get("otel", {})
                        llm_span_ids.add(otel_info.get("span_id"))

            _, ref_span_id = parent_ref.rsplit(":", 1)
            assert ref_span_id not in llm_span_ids

    def test_temporal_integrity_preserved_with_root_span(self):
        """With root invocation span, temporal chaining still works correctly."""
        ig = _make_instrumented(emit_genai_child_spans=True)
        ig.invoke({"query": "chain test"})
        otlp = ig.session.flush_otlp()

        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        nodes = ingest_tgj(docs[0])

        # Find planner and synthesizer MessageNodes (excluding child spans)
        planner_nodes = [
            n for n in nodes.values()
            if isinstance(n, MessageNode) and "planner" in (getattr(n, "py_name", "") or "")
            and "openai" not in (getattr(n, "py_name", "") or "")
        ]
        synth_nodes = [
            n for n in nodes.values()
            if isinstance(n, MessageNode) and "synthesizer" in (getattr(n, "py_name", "") or "")
            and "openai" not in (getattr(n, "py_name", "") or "")
        ]

        if planner_nodes and synth_nodes:
            synth = synth_nodes[0]
            # Walk ancestors
            visited, stack = set(), list(synth.parents)
            found = False
            while stack:
                node = stack.pop()
                if id(node) in visited:
                    continue
                visited.add(id(node))
                if node in planner_nodes:
                    found = True
                    break
                stack.extend(getattr(node, "parents", []))
            assert found, "Synthesizer must have planner as ancestor"


# =========================================================================
# E. optimize_graph: best_parameters + reward in-trace
# =========================================================================


class TestE11_BestParametersSnapshot:
    """E11: best_parameters must be a snapshot from the best-scoring iteration."""

    def test_best_parameters_tracked(self):
        ig = _make_instrumented(
            templates={
                "planner_prompt": "ORIGINAL plan for: {query}",
                "synthesizer_prompt": "ORIGINAL synth: {query} | {plan}",
            }
        )
        mock = MockOptimizer()

        result = optimize_graph(
            ig,
            queries=["test"],
            iterations=1,
            optimizer=mock,
            eval_fn=lambda p: EvalResult(score=0.6, feedback="ok"),
        )

        # best_parameters should be a dict snapshot
        assert isinstance(result.best_parameters, dict)
        assert "planner_prompt" in result.best_parameters
        # final_parameters should differ from best if updates were applied after best
        assert isinstance(result.final_parameters, dict)

    def test_best_parameters_reflects_best_score(self):
        """If baseline is best, best_parameters should be the initial values."""
        ig = _make_instrumented(
            templates={
                "planner_prompt": "INITIAL: {query}",
                "synthesizer_prompt": "INITIAL synth: {query} | {plan}",
            }
        )
        mock = MockOptimizer()

        call_count = [0]

        def declining_eval(payload):
            call_count[0] += 1
            # Baseline scores high, iterations score low
            if payload.get("iteration", 0) == 0:
                return EvalResult(score=0.9, feedback="great baseline")
            return EvalResult(score=0.3, feedback="poor after update")

        result = optimize_graph(
            ig,
            queries=["test"],
            iterations=1,
            optimizer=mock,
            eval_fn=declining_eval,
        )

        assert result.best_score == 0.9
        assert result.best_iteration == 0
        # best_parameters should reflect the initial (baseline) state
        assert "INITIAL" in result.best_parameters.get("planner_prompt", "")


class TestE12_RewardInTrace:
    """E12: A single run's OTLP must contain the evaluation score."""

    def test_eval_score_in_otlp_spans(self):
        ig = _make_instrumented()

        result = optimize_graph(
            ig,
            queries=["test"],
            iterations=0,  # baseline only
            eval_fn=lambda p: EvalResult(score=0.85, feedback="good"),
        )

        # Check the OTLP from the run
        run_otlp = result.all_runs[0][0].otlp
        spans = run_otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]

        # Find a span that contains eval.score
        found_score = False
        for sp in spans:
            attrs = {a["key"]: a["value"]["stringValue"] for a in sp.get("attributes", [])}
            if "eval.score" in attrs:
                assert float(attrs["eval.score"]) == 0.85
                found_score = True
                break

        assert found_score, (
            "eval.score must be present in the run's OTLP spans "
            "(either on root span or as a reward span)"
        )


# =========================================================================
# F. Non-saturating stub scoring
# =========================================================================


def _structure_aware_eval(payload):
    """F13: Score based on response structure, not just length.

    Responses with "Step 1:", "Step 2:", etc. score higher than flat text.
    This makes stub optimization demonstrable.
    """
    answer = payload.get("answer", "")
    if isinstance(answer, dict):
        answer = str(answer.get("answer", ""))
    answer = str(answer)

    score = 0.2  # base score

    # Reward structured responses
    step_count = answer.lower().count("step ")
    if step_count >= 3:
        score += 0.4
    elif step_count >= 1:
        score += 0.2

    # Reward conclusion/summary
    if "conclusion" in answer.lower() or "summary" in answer.lower():
        score += 0.2

    # Reward reasonable length (but cap)
    if len(answer) > 50:
        score += 0.1
    if len(answer) > 100:
        score += 0.1

    return EvalResult(
        score=min(score, 1.0),
        feedback=f"Structure: {step_count} steps, {len(answer)} chars",
    )


class TestF13_NonSaturatingStubScoring:
    """F13: Stub optimization must show score improvement when optimizer updates prompts."""

    def test_score_improves_after_optimization(self):
        """With structure-aware eval, OPTIMIZED prompts must score higher.

        Note: the optimizer applies updates *after* eval in each iteration,
        so we need >=2 iterations to see the effect of iteration-1 updates
        in iteration-2's score.
        """
        ig = _make_instrumented(
            templates={
                "planner_prompt": "Plan for: {query}",
                "synthesizer_prompt": "Synthesize: {query} | Plan: {plan}",
            }
        )
        mock = MockOptimizer()

        result = optimize_graph(
            ig,
            queries=["What is machine learning?"],
            iterations=2,  # baseline + 2 iters; iter-2 uses optimized template
            optimizer=mock,
            eval_fn=_structure_aware_eval,
        )

        baseline = result.score_history[0]
        # Iteration 2 (index 2) is the first to use the OPTIMIZED template
        after_opt = result.score_history[2]

        assert after_opt > baseline, (
            f"Score should improve after optimization: "
            f"baseline={baseline:.4f}, after_opt={after_opt:.4f}. "
            f"Full history: {result.score_history}"
        )

    def test_baseline_does_not_saturate_at_one(self):
        """Baseline score must NOT be 1.0 (the issue was saturation)."""
        ig = _make_instrumented()

        result = optimize_graph(
            ig,
            queries=["What is AI?"],
            iterations=0,
            eval_fn=_structure_aware_eval,
        )

        assert result.baseline_score < 1.0, (
            f"Baseline should NOT saturate at 1.0, got {result.baseline_score}"
        )
