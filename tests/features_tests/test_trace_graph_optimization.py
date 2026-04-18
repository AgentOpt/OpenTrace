import pytest

langgraph = pytest.importorskip("langgraph.graph")
StateGraph = langgraph.StateGraph
START = langgraph.START
END = langgraph.END

from opto.trace import node
from opto.trace.io import instrument_graph, optimize_graph

_TRACE_SCOPE = {}


def _raw(value):
    return getattr(value, "data", value)


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


def bad_planner_node(state):
    query = _raw(state["query"])
    return {"query": str(query), "plan": "bad-plan"}


def bad_synth_node(state):
    query = _raw(state["query"])
    return {"final_answer": f"plain-text answer for {query}"}


def _make_trace_graph():
    planner_prompt = node(
        "Create a plan for: {query}",
        trainable=True,
        name="planner_prompt",
    )
    synth_prompt = node(
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


def _make_bad_trace_graph():
    scope = {
        "planner_node": bad_planner_node,
        "synth_node": bad_synth_node,
    }

    def build_graph():
        graph = StateGraph(dict)
        graph.add_node("planner", scope["planner_node"])
        graph.add_node("synth", scope["synth_node"])
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "synth")
        graph.add_edge("synth", END)
        return graph

    return build_graph, scope


class MutatingOptimizer:
    def __init__(self, prompt_node):
        self.prompt_node = prompt_node
        self.zero_calls = 0
        self.backward_calls = 0
        self.step_calls = 0

    def zero_feedback(self):
        self.zero_calls += 1

    def backward(self, *_args, **_kwargs):
        self.backward_calls += 1

    def step(self):
        self.step_calls += 1
        if self.step_calls == 1:
            self.prompt_node._data = "CRISPR optimized :: {query} :: {plan}"
            return {"synth_prompt": self.prompt_node._data}
        return {}


class BatchSpyOptimizer:
    def __init__(self):
        self.saw_batched_output = False
        self.feedback_len = None

    def zero_feedback(self):
        return None

    def backward(self, output, feedback):
        output_data = getattr(output, "data", None)
        if (
            isinstance(output_data, str)
            and "ID [0]:" in output_data
            and "ID [1]:" in output_data
        ):
            self.saw_batched_output = True
        if isinstance(feedback, str):
            self.feedback_len = feedback.count("ID [")

    def step(self):
        return {}


def test_optimize_graph_trace_backend_reports_progress_and_best_updates():
    build_graph, scope = _make_trace_graph()
    graph = instrument_graph(
        backend="trace",
        graph_factory=build_graph,
        scope=scope,
        graph_agents_functions=["planner_node", "synth_node"],
        graph_prompts_list=[scope["planner_prompt"], scope["synth_prompt"]],
        output_key="final_answer",
    )

    callbacks = []
    optimizer = MutatingOptimizer(scope["synth_prompt"])

    result = optimize_graph(
        graph,
        queries=["What is gene editing?"],
        iterations=2,
        optimizer=optimizer,
        eval_fn=lambda payload: {
            "score": 1.0 if "CRISPR optimized" in str(payload["answer"]) else 0.0,
            "feedback": "Prefer mentioning CRISPR optimized explicitly.",
        },
        on_iteration=lambda i, runs, updates: callbacks.append((i, len(runs), dict(updates))),
    )

    assert result.baseline_score == 0.0
    assert result.best_score == 1.0
    assert result.best_iteration == 2
    assert result.best_updates == {"synth_prompt": "CRISPR optimized :: {query} :: {plan}"}
    assert optimizer.zero_calls == 2
    assert optimizer.backward_calls == 2
    assert optimizer.step_calls == 2
    assert callbacks == [
        (0, 1, {}),
        (1, 1, {"synth_prompt": "CRISPR optimized :: {query} :: {plan}"}),
        (2, 1, {}),
    ]


def test_optimize_graph_trace_backend_batches_multiple_queries():
    build_graph, scope = _make_trace_graph()
    graph = instrument_graph(
        backend="trace",
        graph_factory=build_graph,
        scope=scope,
        graph_agents_functions=["planner_node", "synth_node"],
        graph_prompts_list=[scope["planner_prompt"], scope["synth_prompt"]],
        output_key="final_answer",
    )

    optimizer = BatchSpyOptimizer()
    result = optimize_graph(
        graph,
        queries=["Q1", "Q2"],
        iterations=1,
        optimizer=optimizer,
        eval_fn=lambda payload: {
            "score": 0.5,
            "feedback": "Keep answers short.",
        },
    )

    assert len(result.all_runs[0]) == 2
    assert optimizer.saw_batched_output is True
    assert optimizer.feedback_len == 2


def test_optimize_graph_trace_requires_eval_fn():
    build_graph, scope = _make_trace_graph()
    graph = instrument_graph(
        backend="trace",
        graph_factory=build_graph,
        scope=scope,
        graph_agents_functions=["planner_node", "synth_node"],
        graph_prompts_list=[scope["planner_prompt"], scope["synth_prompt"]],
        output_key="final_answer",
    )

    with pytest.raises(ValueError, match="eval_fn"):
        optimize_graph(graph, queries=["hi"], iterations=0)


def test_optimize_graph_trace_requires_node_output():
    build_graph, scope = _make_bad_trace_graph()
    graph = instrument_graph(
        backend="trace",
        graph_factory=build_graph,
        scope=scope,
        graph_agents_functions=["planner_node", "synth_node"],
        output_key="final_answer",
    )

    with pytest.raises(TypeError, match="Trace Node"):
        optimize_graph(
            graph,
            queries=["What is CRISPR?"],
            iterations=1,
            optimizer=BatchSpyOptimizer(),
            eval_fn=lambda payload: {
                "score": 0.0,
                "feedback": "This should not be reached.",
            },
        )
