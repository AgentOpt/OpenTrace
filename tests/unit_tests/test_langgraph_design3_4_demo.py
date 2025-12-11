import examples.JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE as base
import examples.JSON_OTEL_trace_optim_demo_LANGGRAPH_DESIGN3_4 as demo


def test_tracer_rebound():
    # The new demo should rebind the TRACER and TRACING_LLM in the base module.
    assert hasattr(base, "TRACING_LLM")
    assert hasattr(demo, "TRACING_LLM")
    assert base.TRACING_LLM is demo.TRACING_LLM
    assert base.TRACER is demo.TRACER


def test_run_graph_with_otel_signature():
    # Only check that the function exists and is callable with a fake graph.
    class DummyGraph:
        def invoke(self, state):
            # Echo the state into the final_state shape expected by the demo.
            return {
                "final_answer": "ok",
                "plan": {"steps": []},
            }

    # Reset exporter state and call the wrapper.
    demo.EXPORTER.clear()
    result = demo.run_graph_with_otel(DummyGraph(), "question?")

    assert result.answer == "ok"
    assert isinstance(result.score, float)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.plan, dict)
