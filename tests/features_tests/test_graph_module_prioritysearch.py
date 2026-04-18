import pytest

langgraph = pytest.importorskip("langgraph.graph")
StateGraph = langgraph.StateGraph
START = langgraph.START
END = langgraph.END

from opto.features.priority_search import PrioritySearch
from opto.optimizers.optimizer import Optimizer
from opto.trace import node
from opto.trace.graph import LangGraphAdapter
from opto.trainer.guide import Guide


def _raw(x):
    return getattr(x, "data", x)


class KeywordGuide(Guide):
    def get_feedback(self, query, response, reference=None, **kwargs):
        score = 1.0 if str(reference) in str(response) else 0.0
        return score, f"Expected keyword: {reference}"


class RouteOptimizer(Optimizer):
    def _step(self, *args, **kwargs):
        updates = {p: p.data for p in self.parameters}
        for p in self.parameters:
            if "route_policy" in getattr(p, "name", ""):
                updates[p] = "review"
        return updates


def build_adapter():
    planner_prompt = node("Plan: {query}", trainable=True, name="planner_prompt")
    synth_prompt = node("Answer: {query} :: {plan}", trainable=True, name="synth_prompt")

    def planner_node(state):
        query = _raw(state["query"])
        return {"plan": planner_prompt.data.replace("{query}", str(query))}

    def synth_node(state):
        query = _raw(state["query"])
        plan = _raw(state["plan"])
        route = _raw(state.get("route_policy", "direct"))
        if route == "review":
            return {"final_answer": f"Reviewed CRISPR :: {query}"}
        return {
            "final_answer": synth_prompt.data.replace("{query}", str(query)).replace("{plan}", str(plan))
        }

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


def test_graphmodule_prioritysearch_smoke_for_graph_knob():
    adapter = build_adapter()
    model = adapter.as_module()
    guide = KeywordGuide()
    optimizer = RouteOptimizer(model.parameters())
    algo = PrioritySearch(model, optimizer, num_threads=1)
    algo.train(
        guide=guide,
        train_dataset={"inputs": ["gene editing"], "infos": ["Reviewed"]},
        num_epochs=1,
        batch_size=1,
        num_batches=1,
        num_candidates=1,
        num_proposals=1,
        validate_exploration_candidates=True,
    )
    out = model("gene editing")
    assert "Reviewed" in out.data
