import copy
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

langgraph = pytest.importorskip("langgraph.graph")
StateGraph = langgraph.StateGraph
START = langgraph.START
END = langgraph.END

from opto.trace import node
from opto.features.graph import GraphModule, GraphRunSidecar, LangGraphAdapter
from opto.trace.io import TraceGraph, InstrumentedGraph, instrument_graph, optimize_graph


def _raw(x):
    return getattr(x, "data", x)


def _collect_ancestors(n):
    seen = set()
    stack = [n]
    out = []
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        out.append(cur)
        for parent in getattr(cur, "parents", []):
            stack.append(parent)
    return out


def make_adapter():
    planner_prompt = node("Plan: {query}", trainable=True, name="planner_prompt")
    synth_prompt = node("Answer: {query} :: {plan}", trainable=True, name="synth_prompt")

    def planner_node(state):
        query = _raw(state["query"])
        return {
            "query": query,
            "plan": planner_prompt.data.replace("{query}", str(query)),
        }

    def synth_node(state):
        query = _raw(state["query"])
        plan = _raw(state["plan"])
        answer = synth_prompt.data.replace("{query}", str(query)).replace("{plan}", str(plan))
        return {"final_answer": answer}

    def build_graph(planner_node=planner_node, synth_node=synth_node, route_policy="direct"):
        graph = StateGraph(dict)
        graph.add_node("planner", planner_node)
        graph.add_node("synth", synth_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "synth")
        graph.add_edge("synth", END)
        return graph

    return LangGraphAdapter(
        backend="trace",
        graph_factory=build_graph,
        function_targets={"planner_node": planner_node, "synth_node": synth_node},
        prompt_targets={"planner_prompt": planner_prompt, "synth_prompt": synth_prompt},
        graph_knobs={"route_policy": "direct"},
        input_key="query",
        output_key="final_answer",
    )


def test_invoke_runtime_trace_returns_plain_dict_and_sidecar_node():
    adapter = make_adapter()
    result, sidecar = adapter.invoke_runtime({"query": "What is CRISPR?"}, backend="trace")
    assert isinstance(result, dict)
    assert isinstance(sidecar, GraphRunSidecar)
    assert "final_answer" in result
    assert sidecar.output_node is not None
    assert sidecar.output_node.data == result["final_answer"]


def test_shadow_state_preserves_cross_node_dependencies():
    adapter = make_adapter()
    model = adapter.as_module()
    out = model("What is CRISPR?")
    sidecar = model._last_sidecar
    assert out is sidecar.output_node
    assert "planner_node" in sidecar.node_outputs
    assert "synth_node" in sidecar.node_outputs


def test_graph_module_parameters_include_prompts_and_graph_knobs():
    adapter = make_adapter()
    model = adapter.as_module()
    assert isinstance(model, GraphModule)
    names = {getattr(p, "name", "") for p in model.parameters()}
    assert any("planner_prompt" in n for n in names)
    assert any("synth_prompt" in n for n in names)
    assert any("route_policy" in n for n in names)


def test_bindings_are_auto_generated_and_transparent():
    adapter = make_adapter()
    assert adapter.bindings["planner_prompt"].kind == "prompt"
    assert adapter.bindings["route_policy"].kind == "graph"
    adapter.bindings["route_policy"].set("alternate")
    assert adapter.graph_knobs["route_policy"].data == "alternate"


def test_deepcopy_adapter_bindings_target_clone_state():
    adapter = make_adapter()
    clone = copy.deepcopy(adapter)

    clone.bindings["planner_prompt"].set("Clone plan: {query}")
    clone.bindings["route_policy"].set("review")

    assert clone.prompt_targets["planner_prompt"].data == "Clone plan: {query}"
    assert clone.graph_knobs["route_policy"].data == "review"
    assert adapter.prompt_targets["planner_prompt"].data == "Plan: {query}"
    assert adapter.graph_knobs["route_policy"].data == "direct"


def test_deepcopy_adapter_runtime_uses_clone_prompt_targets():
    adapter = make_adapter()
    clone = copy.deepcopy(adapter)

    clone.prompt_targets["planner_prompt"]._set("Clone plan: {query}")
    clone.prompt_targets["synth_prompt"]._set("Clone answer: {query} :: {plan}")

    result, sidecar = clone.invoke_trace({"query": "CRISPR"})

    assert result["final_answer"] == "Clone answer: CRISPR :: Clone plan: CRISPR"
    assert sidecar.output_node.data == result["final_answer"]
    assert adapter.prompt_targets["planner_prompt"].data == "Plan: {query}"
    assert adapter.prompt_targets["synth_prompt"].data == "Answer: {query} :: {plan}"


def test_parallel_invoke_trace_keeps_sidecars_isolated():
    planner_prompt = node("Plan: {query}", trainable=True, name="planner_prompt")
    synth_prompt = node("Answer: {query} :: {plan}", trainable=True, name="synth_prompt")

    def planner_node(state):
        time.sleep(0.05)
        query = _raw(state["query"])
        return {
            "query": query,
            "plan": planner_prompt.data.replace("{query}", str(query)),
        }

    def synth_node(state):
        time.sleep(0.05)
        query = _raw(state["query"])
        plan = _raw(state["plan"])
        answer = synth_prompt.data.replace("{query}", str(query)).replace("{plan}", str(plan))
        return {"final_answer": answer}

    def build_graph(planner_node=planner_node, synth_node=synth_node, route_policy="direct"):
        graph = StateGraph(dict)
        graph.add_node("planner", planner_node)
        graph.add_node("synth", synth_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "synth")
        graph.add_edge("synth", END)
        return graph

    adapter = LangGraphAdapter(
        backend="trace",
        graph_factory=build_graph,
        function_targets={"planner_node": planner_node, "synth_node": synth_node},
        prompt_targets={"planner_prompt": planner_prompt, "synth_prompt": synth_prompt},
        graph_knobs={"route_policy": "direct"},
        input_key="query",
        output_key="final_answer",
    )

    def run(query):
        result, sidecar = adapter.invoke_trace({"query": query})
        return {
            "query": query,
            "answer": result["final_answer"],
            "shadow_query": _raw(sidecar.shadow_state["query"]),
            "shadow_plan": _raw(sidecar.shadow_state["plan"]),
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        runs = list(executor.map(run, ["A", "B"]))

    answers = {item["query"]: item["answer"] for item in runs}
    assert answers["A"] == "Answer: A :: Plan: A"
    assert answers["B"] == "Answer: B :: Plan: B"
    assert {item["query"]: item["shadow_query"] for item in runs} == {"A": "A", "B": "B"}
    assert {item["query"]: item["shadow_plan"] for item in runs} == {"A": "Plan: A", "B": "Plan: B"}


def test_instrument_graph_accepts_adapter_in_trace_mode_and_optimize_graph_uses_sidecar():
    adapter = make_adapter()
    graph = instrument_graph(adapter=adapter, backend="trace", output_key="final_answer")
    assert isinstance(graph, TraceGraph)
    result = optimize_graph(
        graph,
        queries=["What is CRISPR?"],
        iterations=0,
        eval_fn=lambda payload: {
            "score": 1.0 if "CRISPR" in str(payload["answer"]) else 0.0,
            "feedback": "Keep CRISPR in the final answer.",
        },
    )
    assert result.best_iteration == 0
    assert result.best_score == 1.0
    assert result.all_runs[0][0].artifacts["trace_record"]["output_node"] is not None


def test_instrument_graph_accepts_graph_argument_when_it_is_a_graph_adapter():
    adapter = make_adapter()
    graph = instrument_graph(graph=adapter, backend="trace", output_key="final_answer")
    assert isinstance(graph, TraceGraph)
    out = graph.invoke({"query": "What is CRISPR?"})
    assert isinstance(out, dict)
    assert "final_answer" in out


def test_adapter_dispatch_respects_service_override_in_trace_and_otel_modes():
    adapter = make_adapter()

    trace_graph = instrument_graph(
        graph=adapter,
        backend="trace",
        service_name="trace-override",
    )
    assert isinstance(trace_graph, TraceGraph)
    assert trace_graph.service_name == "trace-override"

    otel_graph = instrument_graph(graph=adapter, backend="otel", service_name="otel-override")
    assert isinstance(otel_graph, InstrumentedGraph)
    assert otel_graph.service_name == "otel-override"
    out = otel_graph.invoke({"query": "What is CRISPR?"})
    assert "final_answer" in out
