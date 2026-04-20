import sys
import pytest

langgraph = pytest.importorskip("langgraph.graph")
StateGraph = langgraph.StateGraph
START = langgraph.START
END = langgraph.END

from opto.trace.io import (
    instrument_graph,
    optimize_graph,
    SysMonInstrumentedGraph,
    make_dict_binding,
)
from opto.trace.io.sysmonitoring import sysmon_profile_to_tgj
from opto.trace.io.tgj_ingest import ingest_tgj


pytestmark = pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="sys.monitoring unavailable")


def build_graph(templates=None):
    templates = templates or {
        "planner_prompt": "Plan {query}",
        "synth_prompt": "answer::{query}::{plan}",
    }

    def planner(state):
        return {
            "query": state["query"],
            "plan": templates["planner_prompt"].replace("{query}", str(state["query"])),
        }

    def synth(state):
        return {
            "final_answer": templates["synth_prompt"]
            .replace("{query}", str(state["query"]))
            .replace("{plan}", str(state["plan"])),
        }

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
        initial_templates={"planner_prompt": "Plan {query}", "synth_prompt": "answer::{query}::{plan}"},
        output_key="final_answer",
    )
    assert isinstance(ig, SysMonInstrumentedGraph)
    out = ig.invoke({"query": "What is CRISPR?"})
    assert "final_answer" in out
    assert ig._last_profile_doc["version"] == "trace-json/1.0+sysmon"
    assert len(ig._last_profile_doc["events"]) > 0


class _DictUpdateOptimizer:
    def __init__(self):
        self.calls = 0

    def zero_feedback(self):
        return None

    def backward(self, *_args, **_kwargs):
        return None

    def step(self):
        self.calls += 1
        if self.calls == 1:
            return {"synth_prompt": "CRISPR optimized :: {query} :: {plan}"}
        return {}


def test_sysmon_profile_to_tgj_preserves_parent_chain():
    profile_doc = {
        "version": "trace-json/1.0+sysmon",
        "agent": {"id": "demo"},
        "bindings": {},
        "events": [
            {"id": "p", "parent_id": None, "name": "planner", "file": "demo.py", "lineno": 1},
            {"id": "c", "parent_id": "p", "name": "synth", "file": "demo.py", "lineno": 2},
        ],
    }
    tgj = sysmon_profile_to_tgj(profile_doc, run_id="r", graph_id="g", scope="demo/0")
    mp = ingest_tgj(tgj)
    assert mp["synth"].parents[0] is mp["planner"]


def test_sysmon_backend_optimize_applies_binding_updates():
    templates = {
        "planner_prompt": "Plan {query}",
        "synth_prompt": "answer::{query}::{plan}",
    }
    bindings = {k: make_dict_binding(templates, k, kind="prompt") for k in templates}
    ig = instrument_graph(
        graph=build_graph(templates),
        backend="sysmon",
        bindings=bindings,
        output_key="final_answer",
    )
    result = optimize_graph(
        ig,
        queries=["What is CRISPR?"],
        iterations=2,
        optimizer=_DictUpdateOptimizer(),
        eval_fn=lambda payload: {
            "score": 1.0 if "CRISPR optimized" in str(payload["answer"]) else 0.0,
            "feedback": "Use the optimized synth prompt.",
        },
    )
    assert result.best_iteration == 2
    assert result.best_score == 1.0
    assert templates["synth_prompt"].startswith("CRISPR optimized")
