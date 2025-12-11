"""
JSON_OTEL_trace_optim_demo_LANGGRAPH_DESIGN3_4.py

Thin wrapper demo that reuses the SPANOUTNODE LangGraph example but routes
all tracing through ``trace/io/langgraph_otel_runtime.py`` (Design-3) and
uses a generic evaluator-span metrics extractor (Design-4).
"""

from __future__ import annotations

from typing import Any, Dict, List
import json

try:
    from . import JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE as base
except ImportError:
    import JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE as base

from opto.trace.io.langgraph_otel_runtime import (
    init_otel_runtime,
    TracingLLM,
    flush_otlp as runtime_flush_otlp,
    extract_eval_metrics_from_otlp,
)

# Re-export types so this demo is self-contained in IDEs / notebooks.
State = base.State
RunResult = base.RunResult
build_graph = base.build_graph
optimize_iteration = base.optimize_iteration


# ---------------------------------------------------------------------------
# OTEL runtime wiring (Design-3)
# ---------------------------------------------------------------------------

TRACER, EXPORTER = init_otel_runtime("langgraph-design3-4-demo")

# Rebind tracer + TracingLLM inside the base module so that:
# * all LLM nodes use the shared runtime TracerProvider
# * all evaluator spans use the same tracer
base.TRACER = TRACER
TRACING_LLM = TracingLLM(
    llm=base.LLM_CLIENT,
    tracer=TRACER,
    trainable_keys=set(base.OPTIMIZABLE),
    emit_code_param=base._emit_code_param,
)
base.TRACING_LLM = TRACING_LLM


# ---------------------------------------------------------------------------
# High-level LangGraph integration (Design-4)
# ---------------------------------------------------------------------------

def run_graph_with_otel(
    graph: Any,
    query: str,
    planner_template: str | None = None,
    executor_template: str | None = None,
    synthesizer_template: str | None = None,
) -> RunResult:
    """
    Run the LangGraph and capture OTEL traces via the shared runtime.
    """

    # Initial state is the same as in the SPANOUTNODE demo.
    initial_state = State(
        user_query=query,
        planner_template=planner_template or base.PLANNER_TEMPLATE_DEFAULT,
        executor_template=executor_template or base.EXECUTOR_TEMPLATE_DEFAULT,
        synthesizer_template=synthesizer_template or base.SYNTH_TEMPLATE_DEFAULT,
    )

    final_state: Dict[str, Any] = graph.invoke(initial_state)

    # Collect OTLP payload from the shared exporter.
    otlp = runtime_flush_otlp(EXPORTER, scope_name="langgraph-design3-4-demo")

    # Use the generic helper instead of ad-hoc span parsing.
    score, metrics, reasons = extract_eval_metrics_from_otlp(otlp)

    feedback = json.dumps(
        {
            "metrics": metrics,
            "score": score,
            "reasons": reasons,
        }
    )

    return RunResult(
        answer=final_state["final_answer"],
        otlp=otlp,
        feedback=feedback,
        score=score,
        metrics=metrics,
        plan=final_state["plan"],
    )


def main() -> None:
    """
    Minimal executable entrypoint for the design-3/4 demo.

    The heavy lifting (LangGraph structure + optimization loop) is reused from
    the SPANOUTNODE file; this module only owns the tracing / evaluation glue.
    """
    graph = build_graph()

    questions = [
        "What are the key events in the Apollo 11 mission?",
        "Explain the main causes of World War I.",
    ]

    optimizer = None
    for step in range(2):
        runs: List[RunResult] = []
        for q in questions:
            result = run_graph_with_otel(graph, q)
            runs.append(result)

        updates, optimizer = optimize_iteration(runs, optimizer=optimizer)
        print(f"[iter {step}] score={runs[0].score:.3f} updated={list(updates.keys())}")


if __name__ == "__main__":
    main()
