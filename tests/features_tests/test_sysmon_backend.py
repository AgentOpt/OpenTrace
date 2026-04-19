import sys
import pytest

langgraph = pytest.importorskip("langgraph.graph")
StateGraph = langgraph.StateGraph
START = langgraph.START
END = langgraph.END

from opto.trace.io import instrument_graph, optimize_graph, SysMonInstrumentedGraph


pytestmark = pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="sys.monitoring unavailable")


def build_graph():
    def planner(state):
        return {"plan": f"plan::{state['query']}"}

    def synth(state):
        return {"final_answer": f"answer::{state['query']}::{state['plan']}"}

    graph = StateGraph(dict)
    graph.add_node("planner", planner)
    graph.add_node("synth", synth)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "synth")
    graph.add_edge("synth", END)
    return graph


def test_sysmon_backend_invoke_exports_profile_doc():
    ig = instrument_graph(
        graph=build_graph(),
        backend="sysmon",
        initial_templates={"planner_prompt": "Plan {query}"},
        output_key="final_answer",
    )
    assert isinstance(ig, SysMonInstrumentedGraph)
    out = ig.invoke({"query": "What is CRISPR?"})
    assert "final_answer" in out
    assert ig._last_profile_doc["version"] == "trace-json/1.0+sysmon"
    assert len(ig._last_profile_doc["events"]) > 0


def test_sysmon_backend_optimize_baseline_only():
    ig = instrument_graph(
        graph=build_graph(),
        backend="sysmon",
        initial_templates={"planner_prompt": "Plan {query}"},
        output_key="final_answer",
    )
    result = optimize_graph(
        ig,
        queries=["What is CRISPR?"],
        iterations=0,
        eval_fn=lambda payload: {
            "score": 1.0 if "CRISPR" in str(payload["answer"]) else 0.0,
            "feedback": "Keep CRISPR in the answer.",
        },
    )
    assert result.best_iteration == 0
    assert result.best_score == 1.0
