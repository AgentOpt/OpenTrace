import pytest

langgraph = pytest.importorskip("langgraph.graph")
StateGraph = langgraph.StateGraph
START = langgraph.START
END = langgraph.END

from opto.trace import node
from opto.trace.io import TraceGraph, instrument_graph

_TRACE_SCOPE = {}


def _raw(value):
    return getattr(value, "data", value)


class RawPrompt:
    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


def planner_node(state):
    query = _raw(state["query"])
    template = _raw(_TRACE_SCOPE["planner_prompt"])
    return {
        "query": str(query),
        "plan": str(template).replace("{query}", str(query)),
    }


def synth_node(state):
    query = _raw(state["query"])
    plan = _raw(state["plan"])
    template = _raw(_TRACE_SCOPE["synth_prompt"])
    answer = str(template).replace("{query}", str(query)).replace("{plan}", str(plan))
    return {"final_answer": node(answer, name="final_answer_node")}


def _make_trace_graph(planner_prompt=None, synth_prompt=None):
    planner_prompt = planner_prompt or node(
        "Create a plan for: {query}",
        trainable=True,
        name="planner_prompt",
    )
    synth_prompt = synth_prompt or node(
        "Answer: {query}\nPlan: {plan}",
        trainable=True,
        name="synth_prompt",
    )

    scope = {
        "planner_prompt": planner_prompt,
        "synth_prompt": synth_prompt,
        "planner_node": planner_node,
        "synth_node": synth_node,
    }
    _TRACE_SCOPE.clear()
    _TRACE_SCOPE.update(scope)

    def build_graph():
        graph = StateGraph(dict)
        graph.add_node("planner", scope["planner_node"])
        graph.add_node("synth", scope["synth_node"])
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "synth")
        graph.add_edge("synth", END)
        return graph

    return build_graph, scope


def test_trace_backend_requires_callable_graph_factory():
    build_graph, _scope = _make_trace_graph()
    with pytest.raises(ValueError):
        instrument_graph(backend="trace", graph=build_graph())


def test_trace_backend_requires_scope_when_factory_is_provided():
    build_graph, _scope = _make_trace_graph()
    with pytest.raises(ValueError, match="scope"):
        instrument_graph(
            backend="trace",
            graph_factory=build_graph,
            scope=None,
            graph_agents_functions=["planner_node", "synth_node"],
            output_key="final_answer",
        )


def test_trace_backend_returns_trace_graph():
    build_graph, scope = _make_trace_graph()
    instrumented = instrument_graph(
        backend="trace",
        graph_factory=build_graph,
        scope=scope,
        graph_agents_functions=["planner_node", "synth_node"],
        graph_prompts_list=[scope["planner_prompt"], scope["synth_prompt"]],
        output_key="final_answer",
    )
    assert isinstance(instrumented, TraceGraph)
    assert instrumented.backend == "trace"
    assert len(instrumented.parameters) >= 2


def test_trace_backend_rejects_unknown_function_name():
    build_graph, scope = _make_trace_graph()
    with pytest.raises(KeyError):
        instrument_graph(
            backend="trace",
            graph_factory=build_graph,
            scope=scope,
            graph_agents_functions=["missing_node"],
            output_key="final_answer",
        )


def test_trace_backend_replaces_raw_prompt_in_scope_by_identity():
    raw_prompt = RawPrompt("Create a plan for: {query}")
    build_graph, scope = _make_trace_graph(planner_prompt=raw_prompt)

    instrumented = instrument_graph(
        backend="trace",
        graph_factory=build_graph,
        scope=scope,
        graph_agents_functions=["planner_node", "synth_node"],
        graph_prompts_list=[raw_prompt],
        output_key="final_answer",
    )

    _TRACE_SCOPE.update(scope)
    assert scope["planner_prompt"] is not raw_prompt
    result = instrumented.invoke({"query": "What is CRISPR?"})
    assert "final_answer" in result
    assert "CRISPR" in result["final_answer"].data
