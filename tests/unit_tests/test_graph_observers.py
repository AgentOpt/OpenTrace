import sys
import pytest

langgraph = pytest.importorskip("langgraph.graph")
StateGraph = langgraph.StateGraph
START = langgraph.START
END = langgraph.END

from opto.trace import node
from opto.trace.io import instrument_graph, TraceGraph, InstrumentedGraph


def _raw(x):
    return getattr(x, "data", x)


def _make_trace_graph():
    planner_prompt = node("Plan: {query}", trainable=True, name="planner_prompt")
    synth_prompt = node("Answer: {query} :: {plan}", trainable=True, name="synth_prompt")

    scope = {}

    def planner_node(state):
        query = _raw(state["query"])
        return {"plan": planner_prompt.data.replace("{query}", str(query))}

    def synth_node(state):
        query = _raw(state["query"])
        plan = _raw(state["plan"])
        answer = synth_prompt.data.replace("{query}", str(query)).replace("{plan}", str(plan))
        return {"final_answer": node(answer, name="final_answer_node")}

    scope.update(
        {
            "planner_prompt": planner_prompt,
            "synth_prompt": synth_prompt,
            "planner_node": planner_node,
            "synth_node": synth_node,
        }
    )

    def build_graph():
        graph = StateGraph(dict)
        graph.add_node("planner", scope["planner_node"])
        graph.add_node("synth", scope["synth_node"])
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "synth")
        graph.add_edge("synth", END)
        return graph

    return build_graph, scope


class _StubLLM:
    model = "stub"

    def __call__(self, messages=None, **kwargs):
        class Msg:
            content = "stub-response"

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


def test_trace_backend_accepts_sysmon_observer():
    if not hasattr(sys, "monitoring"):
        pytest.skip("sys.monitoring unavailable")
    build_graph, scope = _make_trace_graph()
    graph = instrument_graph(
        backend="trace",
        observe_with=("sysmon",),
        graph_factory=build_graph,
        scope=scope,
        graph_agents_functions=["planner_node", "synth_node"],
        graph_prompts_list=[scope["planner_prompt"], scope["synth_prompt"]],
        output_key="final_answer",
    )
    assert isinstance(graph, TraceGraph)
    out = graph.invoke({"query": "What is CRISPR?"})
    assert "final_answer" in out
    assert len(graph._last_observer_artifacts) == 1
    art = graph._last_observer_artifacts[0]
    assert art.carrier == "sysmon"
    assert art.profile_doc["version"] == "trace-json/1.0+sysmon"


def test_trace_backend_accepts_otel_and_sysmon_observers():
    if not hasattr(sys, "monitoring"):
        pytest.skip("sys.monitoring unavailable")
    build_graph, scope = _make_trace_graph()
    graph = instrument_graph(
        backend="trace",
        observe_with=("otel", "sysmon"),
        graph_factory=build_graph,
        scope=scope,
        graph_agents_functions=["planner_node", "synth_node"],
        graph_prompts_list=[scope["planner_prompt"], scope["synth_prompt"]],
        output_key="final_answer",
    )
    out = graph.invoke({"query": "What is CRISPR?"})
    assert "final_answer" in out
    carriers = [a.carrier for a in graph._last_observer_artifacts]
    assert carriers == ["sysmon", "otel"]


def test_otel_backend_rejects_otel_observer():
    with pytest.raises(ValueError, match="invalid"):
        instrument_graph(
            graph=None,
            backend="otel",
            observe_with=("otel",),
            llm=_StubLLM(),
        )


def test_otel_backend_accepts_sysmon_observer():
    if not hasattr(sys, "monitoring"):
        pytest.skip("sys.monitoring unavailable")
    class Graph:
        def invoke(self, state, **kwargs):
            return {"answer": "ok"}
    ig = instrument_graph(
        graph=Graph(),
        backend="otel",
        observe_with=("sysmon",),
        llm=_StubLLM(),
        initial_templates={"prompt_a": "A"},
        output_key="answer",
    )
    assert isinstance(ig, InstrumentedGraph)
    out = ig.invoke({"query": "hi"})
    assert out["answer"] == "ok"
    assert len(ig._last_observer_artifacts) == 1
    assert ig._last_observer_artifacts[0].carrier == "sysmon"
