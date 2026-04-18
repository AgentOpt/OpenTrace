import pytest

langgraph = pytest.importorskip("langgraph.graph")
StateGraph = langgraph.StateGraph
START = langgraph.START
END = langgraph.END

from opto.features.priority_search.priority_search import ModuleCandidate
from opto.optimizers.optimizer import Optimizer
from opto.trace import node
from opto.trace.graph import LangGraphAdapter


def _raw(x):
    return getattr(x, "data", x)


class DummyOptimizer(Optimizer):
    def _step(self, *args, **kwargs):
        return {p: p.data for p in self.parameters}


def make_searchable_model():
    answer_prompt = node("Base: {query}", trainable=True, name="answer_prompt")

    def planner_node(state):
        return {"plan": "draft"}

    def synth_node(state):
        query = _raw(state["query"])
        route = _raw(state.get("route_policy", "direct"))
        if route == "review":
            return {"final_answer": f"Reviewed :: {query}"}
        return {"final_answer": answer_prompt.data.replace("{query}", str(query))}

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
        prompt_targets={"answer_prompt": answer_prompt},
        graph_knobs={"route_policy": "direct"},
        input_key="query",
        output_key="final_answer",
    )
    return adapter.as_module()


def test_modulecandidate_get_module_works_with_graphmodule():
    model = make_searchable_model()
    optimizer = DummyOptimizer(model.parameters())
    route_param = next(p for p in model.parameters() if "route_policy" in p.name)
    candidate = ModuleCandidate(
        model,
        update_dict={route_param: "review"},
        optimizer=optimizer,
    )
    new_model = candidate.get_module()
    assert getattr(new_model.adapter, "_active_sidecar", None) is None
    assert getattr(new_model.adapter, "_compiled_cache", {}) == {}
    out = new_model("What is CRISPR?")
    assert isinstance(out.data, str)
