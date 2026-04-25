from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph

from opto.features.graph.adapter import LangGraphAdapter
from opto.trace import node
from opto.trace.io import instrument_graph
from opto.trace.io.bindings import apply_updates
from opto.trace.io.otel_adapter import otlp_traces_to_trace_json
from opto.trace.io.tgj_ingest import ingest_tgj
from opto.trace.nodes import MessageNode, ParameterNode


def _spans(otlp):
    out = []
    for rs in otlp.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            out.extend(ss.get("spans", []))
    return out


def _attrs(span):
    attrs = {}
    for item in span.get("attributes", []):
        key = item.get("key")
        value = item.get("value", {})
        if "stringValue" in value:
            attrs[key] = value["stringValue"]
        elif "boolValue" in value:
            attrs[key] = value["boolValue"]
        elif "intValue" in value:
            attrs[key] = value["intValue"]
        elif "doubleValue" in value:
            attrs[key] = value["doubleValue"]
        else:
            attrs[key] = value
    return attrs


def _truthy(value):
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def _span_named(otlp, name):
    for span in _spans(otlp):
        if span.get("name") == name:
            return span
    raise AssertionError(f"span not found: {name}; saw {[s.get('name') for s in _spans(otlp)]}")


def _make_adapter():
    planner_prompt = node("Plan: {query}", trainable=True, name="planner_prompt")
    synth_prompt = node("Answer: {plan}", trainable=True, name="synth_prompt")

    def planner_node(state):
        query = state.get("query", "")
        return {"query": query, "plan": f"plan({query})"}

    def synth_node(state):
        route = state.get("route_policy", "direct")
        answer = f"answer({state.get('plan', '')})"
        if route == "review":
            answer = "Reviewed " + answer
        return {"query": state.get("query", ""), "plan": state.get("plan", ""), "final_answer": answer}

    def build_graph(planner_node=planner_node, synth_node=synth_node, route_policy="direct"):
        graph = StateGraph(dict)
        graph.add_node("planner_node", planner_node)
        graph.add_node("synth_node", synth_node)
        graph.add_edge(START, "planner_node")
        graph.add_edge("planner_node", "synth_node")
        graph.add_edge("synth_node", END)
        return graph.compile()

    return LangGraphAdapter(
        graph_factory=build_graph,
        function_targets={
            "planner_node": planner_node,
            "synth_node": synth_node,
        },
        prompt_targets={
            "planner_prompt": planner_prompt,
            "synth_prompt": synth_prompt,
        },
        graph_knobs={"route_policy": "direct"},
        input_key="query",
        output_key="final_answer",
    )


def test_otel_adapter_emits_span_for_each_function_target_with_params():
    adapter = _make_adapter()
    graph = instrument_graph(
        adapter=adapter,
        backend="otel",
        service_name="test-langgraph-otel",
        output_key="final_answer",
    )

    result = graph.invoke({"query": "CRISPR"})
    assert result["final_answer"] == "answer(plan(CRISPR))"

    otlp = graph.session.flush_otlp(clear=True)
    names = [span.get("name") for span in _spans(otlp)]

    assert any(name.endswith(".invoke") for name in names)
    assert "planner_node" in names
    assert "synth_node" in names

    planner_attrs = _attrs(_span_named(otlp, "planner_node"))
    assert planner_attrs["message.id"] == "planner_node"
    assert planner_attrs["graph.node.name"] == "planner_node"
    assert planner_attrs["graph.backend"] == "otel"
    assert planner_attrs["inputs.query"] == "CRISPR"

    assert planner_attrs["param.planner_prompt"] == "Plan: {query}"
    assert _truthy(planner_attrs["param.planner_prompt.trainable"])
    assert planner_attrs["param.synth_prompt"] == "Answer: {plan}"
    assert _truthy(planner_attrs["param.synth_prompt.trainable"])
    assert planner_attrs["param.route_policy"] == "direct"
    assert _truthy(planner_attrs["param.route_policy.trainable"])
    assert "outputs.preview" in planner_attrs
    assert "param.__code_planner_node" in planner_attrs


def test_otel_adapter_params_convert_to_tgj_and_trace_nodes():
    adapter = _make_adapter()
    graph = instrument_graph(adapter=adapter, backend="otel", output_key="final_answer")

    graph.invoke({"query": "CRISPR"})
    otlp = graph.session.flush_otlp(clear=True)

    tgj_docs = list(
        otlp_traces_to_trace_json(
            otlp,
            agent_id_hint="test-langgraph-otel",
            use_temporal_hierarchy=True,
        )
    )
    nodes = ingest_tgj(tgj_docs[0])

    param_names = {
        getattr(n, "name", "").split(":")[0].split("/")[-1]
        for n in nodes.values()
        if isinstance(n, ParameterNode) and getattr(n, "trainable", False)
    }
    assert {"planner_prompt", "synth_prompt", "route_policy"}.issubset(param_names)

    message_names = {
        getattr(n, "name", "")
        for n in nodes.values()
        if isinstance(n, MessageNode)
    }
    assert any(name.split("/")[-1].split(":")[0] == "planner_node" for name in message_names)
    assert any(name.split("/")[-1].split(":")[0] == "synth_node" for name in message_names)


def test_otel_adapter_apply_updates_updates_graph_knob_and_next_run_behavior():
    adapter = _make_adapter()
    graph = instrument_graph(adapter=adapter, backend="otel", output_key="final_answer")

    result = graph.invoke({"query": "CRISPR"})
    assert result["final_answer"] == "answer(plan(CRISPR))"
    graph.session.flush_otlp(clear=True)

    apply_updates({"param.route_policy:0": "review"}, graph.bindings, strict=True)

    result = graph.invoke({"query": "CRISPR"})
    assert result["final_answer"] == "Reviewed answer(plan(CRISPR))"

    otlp = graph.session.flush_otlp(clear=True)
    synth_attrs = _attrs(_span_named(otlp, "synth_node"))
    assert synth_attrs["param.route_policy"] == "review"
    assert synth_attrs["graph.knob.route_policy"] == "review"


def test_otel_adapter_apply_updates_updates_code_parameter_without_recompile():
    adapter = _make_adapter()
    graph = instrument_graph(adapter=adapter, backend="otel", output_key="final_answer")

    first = graph.invoke({"query": "CRISPR"})
    assert first["plan"] == "plan(CRISPR)"
    assert first["final_answer"] == "answer(plan(CRISPR))"
    assert len(adapter._compiled_cache) == 1
    graph.session.flush_otlp(clear=True)

    code_key = "__code_planner_node"
    original_code = graph.bindings[code_key].get()
    updated_code = original_code.replace('plan({query})', 'ALT({query})')

    applied = apply_updates({f"param.{code_key}:0": updated_code}, graph.bindings, strict=True)
    assert applied == {code_key: updated_code}
    assert graph.bindings[code_key].get() == updated_code

    second = graph.invoke({"query": "CRISPR"})
    assert second["plan"] == "ALT(CRISPR)"
    assert second["final_answer"] == "answer(ALT(CRISPR))"
    assert len(adapter._compiled_cache) == 1

    otlp = graph.session.flush_otlp(clear=True)
    planner_attrs = _attrs(_span_named(otlp, "planner_node"))
    assert planner_attrs[f"param.{code_key}"] == updated_code
    assert "ALT(CRISPR)" in planner_attrs["outputs.preview"]


def test_otel_adapter_invalid_code_update_raises_and_marks_node_span_error():
    adapter = _make_adapter()
    graph = instrument_graph(adapter=adapter, backend="otel", output_key="final_answer")

    code_key = "__code_planner_node"
    bad_code = """def planner_node(state):
    return {"query": state.get("query", ""), "plan":
"""
    apply_updates({f"param.{code_key}:0": bad_code}, graph.bindings, strict=True)

    with pytest.raises(Exception):
        graph.invoke({"query": "CRISPR"})

    otlp = graph.session.flush_otlp(clear=True)
    planner_attrs = _attrs(_span_named(otlp, "planner_node"))
    assert _truthy(planner_attrs["error"])
    assert planner_attrs["error.type"] == "ExecutionError"
    assert "SyntaxError" in planner_attrs["error.message"]
