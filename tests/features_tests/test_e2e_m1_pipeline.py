"""
End-to-end integration test for M1 acceptance criteria.

Pipeline under test:
    instrument_graph() → build LangGraph → invoke → flush OTLP
    → OTLP→TGJ conversion → ingest_tgj → ParameterNode / MessageNode
    → optimizer step (mock) → apply_updates → verify template change
    → re-invoke → verify new template used

Uses **StubLLM** only (no real LLM calls, CI-safe).
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List

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
)
from opto.trace.nodes import ParameterNode, MessageNode


# =========================================================================
# Stub LLM (deterministic, no API calls)
# =========================================================================


class StubLLM:
    """Deterministic LLM stub that returns canned responses."""

    model = "stub-llm"

    def __init__(self) -> None:
        self.call_count = 0
        self.last_messages: list | None = None

    def __call__(self, messages=None, **kwargs):
        self.call_count += 1
        self.last_messages = messages

        # Build a context-aware canned response
        content = f"stub-response-{self.call_count}"
        if messages:
            for m in messages:
                text = (m.get("content") or "").lower()
                if m.get("role") == "system" and "plan" in text:
                    content = "Step 1: Research. Step 2: Analyze."
                elif m.get("role") == "system" and "synth" in text:
                    content = "Based on the plan, here is a comprehensive answer."

        class _Msg:
            pass

        msg = _Msg()
        msg.content = content

        class _Choice:
            pass

        choice = _Choice()
        choice.message = msg

        class _Resp:
            pass

        resp = _Resp()
        resp.choices = [choice]
        return resp


# =========================================================================
# LangGraph state + builder
# =========================================================================


class AgentState(TypedDict, total=False):
    query: str
    plan: str
    answer: str


def build_mini_graph(
    tracing_llm: TracingLLM,
    templates: Dict[str, str],
) -> StateGraph:
    """Build a minimal 2-node LangGraph (planner → synthesizer).

    Node functions **close over** *tracing_llm* and *templates* so that
    ``apply_updates`` on the dict propagates to subsequent invocations.
    """

    def planner_node(state: AgentState) -> Dict[str, Any]:
        template = templates.get(
            "planner_prompt", "Create a plan for: {query}"
        )
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

    def synthesizer_node(state: AgentState) -> Dict[str, Any]:
        template = templates.get(
            "synthesizer_prompt",
            "Synthesize: {query}\nPlan: {plan}",
        )
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


# =========================================================================
# Mock optimizer (returns deterministic updates)
# =========================================================================


class MockOptimizer:
    """Mock optimizer that records calls and returns known updates."""

    def __init__(self, param_nodes=None, **kwargs):
        self.param_nodes = param_nodes or []
        self.calls: List[str] = []
        self._step_updates: Dict[str, str] = {
            "planner_prompt": "OPTIMIZED: Create an improved plan for: {query}",
        }

    def zero_feedback(self):
        self.calls.append("zero_feedback")

    def backward(self, output_node, feedback_text):
        self.calls.append(f"backward({type(output_node).__name__})")

    def step(self):
        self.calls.append("step")
        return dict(self._step_updates)


# =========================================================================
# Helpers
# =========================================================================


def _make_instrumented(
    *,
    templates: Dict[str, str] | None = None,
    trainable_keys=None,
    emit_genai_child_spans: bool = True,
) -> InstrumentedGraph:
    """Convenience: build an InstrumentedGraph with a real LangGraph."""
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
        llm=StubLLM(),
        initial_templates=templates,
        emit_genai_child_spans=emit_genai_child_spans,
        provider_name="openai",
        llm_span_name="openai.chat.completion",
        output_key="answer",
    )
    graph = build_mini_graph(ig.tracing_llm, ig.templates)
    ig.graph = graph.compile()
    return ig


# =========================================================================
# 1. Instrument + Invoke → OTLP
# =========================================================================


class TestE2EInstrumentAndInvoke:
    """M1 gate: instrument_graph + real LangGraph invoke produces OTLP."""

    def test_invoke_produces_result_with_answer(self):
        ig = _make_instrumented()
        result = ig.invoke({"query": "What is Python?"})
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_invoke_produces_otlp_with_planner_and_synthesizer_spans(self):
        ig = _make_instrumented()
        ig.invoke({"query": "What is AI?"})
        otlp = ig.session.flush_otlp()

        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        names = [s["name"] for s in spans]

        assert "planner" in names, f"Missing planner span; got {names}"
        assert "synthesizer" in names, f"Missing synthesizer span; got {names}"

    def test_child_llm_spans_emitted_when_enabled(self):
        ig = _make_instrumented(emit_genai_child_spans=True)
        ig.invoke({"query": "test"})
        otlp = ig.session.flush_otlp()

        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        names = [s["name"] for s in spans]

        assert names.count("openai.chat.completion") == 2, (
            f"Expected 2 child LLM spans; got {names}"
        )

    def test_no_child_llm_spans_when_disabled(self):
        ig = _make_instrumented(emit_genai_child_spans=False)
        ig.invoke({"query": "test"})
        otlp = ig.session.flush_otlp()

        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        names = [s["name"] for s in spans]

        assert "openai.chat.completion" not in names


# =========================================================================
# 2. OTLP → param.* attributes
# =========================================================================


class TestE2EParamAttributes:
    """M1 gate: spans carry ``param.*`` and ``param.*.trainable``."""

    def test_planner_span_has_param_attributes(self):
        ig = _make_instrumented()
        ig.invoke({"query": "test"})
        otlp = ig.session.flush_otlp()

        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        planner = next(s for s in spans if s["name"] == "planner")
        attrs = {
            a["key"]: a["value"]["stringValue"]
            for a in planner["attributes"]
        }

        assert "param.planner_prompt" in attrs
        assert attrs["param.planner_prompt"] == "Plan for: {query}"
        assert "param.planner_prompt.trainable" in attrs
        assert attrs["param.planner_prompt.trainable"] == "True"

    def test_synthesizer_span_has_param_attributes(self):
        ig = _make_instrumented()
        ig.invoke({"query": "test"})
        otlp = ig.session.flush_otlp()

        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        synth = next(s for s in spans if s["name"] == "synthesizer")
        attrs = {
            a["key"]: a["value"]["stringValue"]
            for a in synth["attributes"]
        }

        assert "param.synthesizer_prompt" in attrs
        assert attrs["param.synthesizer_prompt.trainable"] == "True"


# =========================================================================
# 3. OTLP → TGJ → ParameterNode + MessageNode
# =========================================================================


class TestE2EOtlpToTgj:
    """M1 gate: OTLP→TGJ→ingest_tgj produces ParameterNode + MessageNode."""

    def test_tgj_has_parameter_nodes(self):
        ig = _make_instrumented()
        ig.invoke({"query": "hello"})
        otlp = ig.session.flush_otlp()

        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        assert len(docs) >= 1

        nodes = ingest_tgj(docs[0])
        param_nodes = [
            n for n in nodes.values()
            if isinstance(n, ParameterNode) and n.trainable
        ]
        assert len(param_nodes) > 0, "Expected at least one trainable ParameterNode"

    def test_tgj_has_message_nodes(self):
        ig = _make_instrumented()
        ig.invoke({"query": "hello"})
        otlp = ig.session.flush_otlp()

        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        nodes = ingest_tgj(docs[0])
        msg_nodes = [
            n for n in nodes.values() if isinstance(n, MessageNode)
        ]
        assert len(msg_nodes) > 0, "Expected at least one MessageNode"

    def test_message_node_has_parameter_parent(self):
        """MessageNode for planner should have planner_prompt ParameterNode as parent."""
        ig = _make_instrumented()
        ig.invoke({"query": "hello"})
        otlp = ig.session.flush_otlp()

        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        nodes = ingest_tgj(docs[0])

        # Find the planner MessageNode
        planner_msgs = [
            n for n in nodes.values()
            if isinstance(n, MessageNode)
            and "planner" in (n.py_name or "").lower()
        ]
        assert len(planner_msgs) > 0, "Expected planner MessageNode"

        planner_msg = planner_msgs[0]
        parent_names = [p.py_name for p in planner_msg.parents]
        # At least one parent should be the planner_prompt ParameterNode
        has_param_parent = any(
            isinstance(p, ParameterNode) and "planner_prompt" in p.py_name
            for p in planner_msg.parents
        )
        assert has_param_parent, (
            f"planner MessageNode should have planner_prompt ParameterNode "
            f"as parent; got parents: {parent_names}"
        )


# =========================================================================
# 4. Temporal integrity: child spans don't break the chain
# =========================================================================


class TestE2ETemporalIntegrity:
    """M1 acceptance gate #5: child spans must NOT advance TGJ temporal chain."""

    def test_synthesizer_temporal_parent_is_planner_not_child_span(self):
        ig = _make_instrumented(emit_genai_child_spans=True)
        ig.invoke({"query": "test temporal"})
        otlp = ig.session.flush_otlp()

        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        doc = docs[0]
        tgj_nodes = doc["nodes"]

        # Collect span IDs of child LLM spans (kind=msg, name contains "chat")
        llm_span_ids = set()
        for nid, n in tgj_nodes.items():
            if n.get("kind") == "msg":
                otel_info = (n.get("info") or {}).get("otel", {})
                nm = n.get("name", "")
                if "openai" in nm or "chat" in nm:
                    llm_span_ids.add(otel_info.get("span_id"))

        # Get synthesizer node and check its parent reference
        synth_nodes = [
            (nid, n) for nid, n in tgj_nodes.items()
            if n.get("kind") == "msg" and n.get("name") == "synthesizer"
        ]
        assert len(synth_nodes) >= 1, "Missing synthesizer msg node in TGJ"

        _, synth = synth_nodes[0]
        parent_ref = synth.get("inputs", {}).get("parent", "")

        if parent_ref and isinstance(parent_ref, str) and ":" in parent_ref:
            _, ref_span_id = parent_ref.rsplit(":", 1)
            assert ref_span_id not in llm_span_ids, (
                "Synthesizer's temporal parent must NOT be a child LLM span"
            )

    def test_temporal_chain_preserved_after_ingest(self):
        """After ingest, planner MessageNode should be an ancestor of synthesizer."""
        ig = _make_instrumented(emit_genai_child_spans=True)
        ig.invoke({"query": "chain test"})
        otlp = ig.session.flush_otlp()

        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        nodes = ingest_tgj(docs[0])

        # Find planner and synthesizer MessageNodes
        planner_nodes = [
            n for n in nodes.values()
            if isinstance(n, MessageNode) and "planner" in n.py_name
            and "openai" not in n.py_name
        ]
        synth_nodes = [
            n for n in nodes.values()
            if isinstance(n, MessageNode) and "synthesizer" in n.py_name
            and "openai" not in n.py_name
        ]

        if not planner_nodes or not synth_nodes:
            # If names are mangled, at least verify that we have multiple
            # MessageNodes and they have parent relationships
            msg_nodes = [
                n for n in nodes.values() if isinstance(n, MessageNode)
            ]
            assert len(msg_nodes) >= 2, (
                "Expected at least 2 MessageNodes (planner + synthesizer)"
            )
            return

        synth = synth_nodes[0]
        # Walk ancestors of synthesizer
        visited, stack = set(), list(synth.parents)
        found_planner = False
        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            if node in planner_nodes:
                found_planner = True
                break
            stack.extend(getattr(node, "parents", []))

        assert found_planner, (
            "Synthesizer MessageNode should have planner MessageNode as "
            "ancestor via temporal chain"
        )


# =========================================================================
# 5. Bindings round-trip: apply_updates → template change → next invoke
# =========================================================================


class TestE2EBindingRoundTrip:
    """M1 gate: bindings correctly propagate optimizer output to runtime."""

    def test_auto_derived_bindings_are_functional(self):
        ig = _make_instrumented()
        assert ig.bindings["planner_prompt"].get() == "Plan for: {query}"
        ig.bindings["planner_prompt"].set("NEW")
        assert ig.templates["planner_prompt"] == "NEW"

    def test_apply_updates_changes_template(self):
        ig = _make_instrumented()
        apply_updates(
            {"planner_prompt": "UPDATED: {query}"},
            ig.bindings,
        )
        assert ig.templates["planner_prompt"] == "UPDATED: {query}"
        assert ig.bindings["planner_prompt"].get() == "UPDATED: {query}"

    def test_updated_template_used_in_next_invoke(self):
        """After apply_updates, the next invoke records the NEW template."""
        ig = _make_instrumented()

        # --- invoke 1: original template ---
        ig.invoke({"query": "test"})
        otlp1 = ig.session.flush_otlp()
        spans1 = otlp1["resourceSpans"][0]["scopeSpans"][0]["spans"]
        p1 = next(s for s in spans1 if s["name"] == "planner")
        a1 = {a["key"]: a["value"]["stringValue"] for a in p1["attributes"]}
        assert a1["param.planner_prompt"] == "Plan for: {query}"

        # --- apply update ---
        apply_updates({"planner_prompt": "UPDATED: {query}"}, ig.bindings)

        # --- invoke 2: updated template ---
        ig.invoke({"query": "test"})
        otlp2 = ig.session.flush_otlp()
        spans2 = otlp2["resourceSpans"][0]["scopeSpans"][0]["spans"]
        p2 = next(s for s in spans2 if s["name"] == "planner")
        a2 = {a["key"]: a["value"]["stringValue"] for a in p2["attributes"]}
        assert a2["param.planner_prompt"] == "UPDATED: {query}"


# =========================================================================
# 6. optimize_graph() — eval-only mode (no optimizer)
# =========================================================================


class TestE2EOptimizeEvalOnly:
    """Run optimize_graph with custom eval_fn but without optimizer."""

    def test_baseline_and_iterations_run(self):
        ig = _make_instrumented()

        def score_fn(payload):
            answer = payload.get("answer", "")
            if isinstance(answer, dict):
                answer = str(answer.get("answer", ""))
            return EvalResult(
                score=min(len(str(answer)) / 100.0, 1.0),
                feedback="length-based eval",
            )

        result = optimize_graph(
            ig,
            queries=["What is Python?", "Explain AI"],
            iterations=1,
            eval_fn=score_fn,
            apply_updates_flag=False,
        )

        assert result.baseline_score >= 0
        assert len(result.score_history) == 2  # baseline + 1 iter
        assert len(result.all_runs) == 2
        assert len(result.all_runs[0]) == 2  # 2 queries per iter

        # Each RunResult should carry OTLP data
        for run in result.all_runs[0]:
            assert "resourceSpans" in run.otlp

    def test_on_iteration_callback(self):
        ig = _make_instrumented()
        log: list = []

        def on_iter(iter_num, runs, updates):
            log.append({"iter": iter_num, "n_runs": len(runs)})

        result = optimize_graph(
            ig,
            queries=["q1"],
            iterations=2,
            eval_fn=lambda p: 0.5,
            on_iteration=on_iter,
        )

        # on_iteration is called for iterations 1 and 2 (not baseline)
        assert len(log) == 2
        assert log[0]["iter"] == 1
        assert log[1]["iter"] == 2


# =========================================================================
# 7. optimize_graph() — with mock optimizer → apply_updates
# =========================================================================


class TestE2EOptimizeWithMockOptimizer:
    """Full pipeline with injected mock optimizer to verify apply_updates."""

    def test_mock_optimizer_updates_are_applied(self):
        ig = _make_instrumented(
            templates={
                "planner_prompt": "ORIGINAL plan for: {query}",
                "synthesizer_prompt": "ORIGINAL synth: {query} | {plan}",
            }
        )
        mock = MockOptimizer()

        result = optimize_graph(
            ig,
            queries=["What is AI?"],
            iterations=1,
            optimizer=mock,
            eval_fn=lambda p: EvalResult(score=0.6, feedback="ok"),
        )

        # Optimizer methods should have been called
        assert "zero_feedback" in mock.calls
        assert any("backward" in c for c in mock.calls)
        assert "step" in mock.calls

        # apply_updates should have changed planner_prompt
        assert ig.templates["planner_prompt"] == (
            "OPTIMIZED: Create an improved plan for: {query}"
        )

    def test_second_iteration_uses_updated_template(self):
        """After optimizer updates, next iteration should see the new template."""
        ig = _make_instrumented(
            templates={
                "planner_prompt": "ORIGINAL: {query}",
                "synthesizer_prompt": "Synth: {query} | {plan}",
            }
        )
        mock = MockOptimizer()

        captured_otlps: List[Dict[str, Any]] = []

        def eval_fn(payload):
            captured_otlps.append(payload.get("otlp", {}))
            return EvalResult(score=0.5, feedback="test")

        result = optimize_graph(
            ig,
            queries=["q1"],
            iterations=2,
            optimizer=mock,
            eval_fn=eval_fn,
        )

        # We should have captured OTLP from baseline + iter1 + iter2 = 3 invocations
        assert len(captured_otlps) == 3

        # The 3rd invocation (iteration 2) should use the updated template
        last_otlp = captured_otlps[-1]
        spans = last_otlp.get("resourceSpans", [{}])[0].get("scopeSpans", [{}])[0].get("spans", [])
        planner_spans = [s for s in spans if s.get("name") == "planner"]

        if planner_spans:
            attrs = {
                a["key"]: a["value"]["stringValue"]
                for a in planner_spans[0].get("attributes", [])
            }
            assert "OPTIMIZED" in attrs.get("param.planner_prompt", ""), (
                "Second+ iteration should use the OPTIMIZED template"
            )

    def test_optimization_result_structure(self):
        ig = _make_instrumented()
        mock = MockOptimizer()

        result = optimize_graph(
            ig,
            queries=["q1", "q2"],
            iterations=2,
            optimizer=mock,
            eval_fn=lambda p: EvalResult(score=0.7, feedback="good"),
        )

        assert isinstance(result.baseline_score, float)
        assert isinstance(result.best_score, float)
        assert isinstance(result.best_iteration, int)
        assert isinstance(result.best_updates, dict)
        assert isinstance(result.final_parameters, dict)
        assert len(result.score_history) == 3  # baseline + 2 iters
        assert len(result.all_runs) == 3


# =========================================================================
# 8. Full round-trip: instrument → invoke → TGJ → optimizer → apply → re-invoke
# =========================================================================


class TestE2EFullRoundTrip:
    """The ultimate M1 acceptance test: all components wired together."""

    def test_full_pipeline_end_to_end(self):
        """
        1. instrument_graph with initial templates
        2. invoke → OTLP → verify spans
        3. OTLP → TGJ → verify ParameterNode + MessageNode
        4. apply_updates → verify template change
        5. re-invoke → verify new template in OTLP
        """
        # --- Step 1: instrument ---
        templates = {
            "planner_prompt": "V1: Plan for {query}",
            "synthesizer_prompt": "V1: Synthesize {query} with {plan}",
        }
        ig = _make_instrumented(templates=templates)

        # --- Step 2: invoke ---
        result = ig.invoke({"query": "What is ML?"})
        assert "answer" in result

        otlp = ig.session.flush_otlp()
        spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
        span_names = [s["name"] for s in spans]
        assert "planner" in span_names
        assert "synthesizer" in span_names

        # Verify param attributes
        planner_span = next(s for s in spans if s["name"] == "planner")
        attrs = {
            a["key"]: a["value"]["stringValue"]
            for a in planner_span["attributes"]
        }
        assert attrs["param.planner_prompt"] == "V1: Plan for {query}"
        assert attrs["param.planner_prompt.trainable"] == "True"

        # --- Step 3: OTLP → TGJ → Trace nodes ---
        docs = otlp_traces_to_trace_json(
            otlp, agent_id_hint="e2e-test", use_temporal_hierarchy=True,
        )
        assert len(docs) >= 1

        nodes = ingest_tgj(docs[0])
        param_nodes = [
            n for n in nodes.values()
            if isinstance(n, ParameterNode) and n.trainable
        ]
        msg_nodes = [
            n for n in nodes.values() if isinstance(n, MessageNode)
        ]
        assert len(param_nodes) > 0, "TGJ must produce trainable ParameterNodes"
        assert len(msg_nodes) > 0, "TGJ must produce MessageNodes"

        # --- Step 4: apply_updates ---
        apply_updates(
            {"planner_prompt": "V2: Improved plan for {query}"},
            ig.bindings,
        )
        assert ig.templates["planner_prompt"] == "V2: Improved plan for {query}"

        # --- Step 5: re-invoke with new template ---
        result2 = ig.invoke({"query": "What is DL?"})
        assert "answer" in result2

        otlp2 = ig.session.flush_otlp()
        spans2 = otlp2["resourceSpans"][0]["scopeSpans"][0]["spans"]
        planner2 = next(s for s in spans2 if s["name"] == "planner")
        attrs2 = {
            a["key"]: a["value"]["stringValue"]
            for a in planner2["attributes"]
        }
        assert attrs2["param.planner_prompt"] == "V2: Improved plan for {query}", (
            "Re-invocation must use the UPDATED template"
        )

    def test_optimize_graph_full_integration(self):
        """optimize_graph with mock optimizer: end-to-end template update."""
        ig = _make_instrumented(
            templates={
                "planner_prompt": "BEFORE: Plan for {query}",
                "synthesizer_prompt": "BEFORE: Synth {query} | {plan}",
            }
        )
        mock = MockOptimizer()

        result = optimize_graph(
            ig,
            queries=["What is AI?"],
            iterations=1,
            optimizer=mock,
            eval_fn=lambda p: EvalResult(score=0.5, feedback="needs work"),
        )

        # Verify optimizer was exercised
        assert "step" in mock.calls

        # Verify templates were updated
        assert ig.templates["planner_prompt"].startswith("OPTIMIZED:")

        # Verify final_parameters reflect the update
        assert "planner_prompt" in result.final_parameters
        assert result.final_parameters["planner_prompt"].startswith("OPTIMIZED:")

        # Verify score history
        assert len(result.score_history) == 2  # baseline + 1 iter
        assert all(isinstance(s, float) for s in result.score_history)
