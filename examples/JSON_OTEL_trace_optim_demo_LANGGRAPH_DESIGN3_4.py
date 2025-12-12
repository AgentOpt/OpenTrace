"""
JSON_OTEL_trace_optim_demo_LANGGRAPH_DESIGN3_4.py

Thin wrapper demo that reuses the SPANOUTNODE LangGraph example but routes
all tracing through ``trace/io/langgraph_otel_runtime.py`` (Design-3) and
uses a generic evaluator-span metrics extractor (Design-4).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
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
from opto.trace.io.eval_hooks import (
    EvalFn,
    default_feedback,
    make_document_embedding_analysis_eval,
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
    *,
    eval_fn: Optional[EvalFn] = None,
    eval_data: Optional[Dict[str, Any]] = None,
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
    llm_score, llm_metrics, reasons = extract_eval_metrics_from_otlp(otlp)
    answer_text = final_state["final_answer"]

    if eval_fn is None:
        score = llm_score
        metrics = llm_metrics
        feedback = default_feedback(score, metrics, reasons)
    else:
        score, metrics, feedback = eval_fn(answer_text, llm_score, llm_metrics, reasons, otlp, eval_data or {})

    return RunResult(
        answer=answer_text,
        otlp=otlp,
        feedback=feedback,
        score=score,
        metrics=metrics,
        plan=final_state["plan"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", default="llm", choices=["llm", "dea", "hybrid"], help="Scoring mode")
    parser.add_argument("--dea_solution_json", default=None, help="Path to a DEA solution JSON (optional)")
    parser.add_argument("--dea_root", default=None, help="Path to DEA root containing output/latex/*.json (optional)")
    parser.add_argument("--max_examples", type=int, default=2, help="Max DEA examples to run when using --dea_root")
    parser.add_argument("--candidate_content_type", default="markdown", help="Candidate content type for doc_eval: markdown|latex")
    parser.add_argument("--skip_dea", action="store_true", help="Pass skip_dea=True to doc_eval (debug/fast)")
    args = parser.parse_args()

    graph = build_graph()

    eval_fn: Optional[EvalFn] = None
    if args.eval_mode in ("dea", "hybrid"):
        eval_fn = make_document_embedding_analysis_eval(
            mode=args.eval_mode,
            llm=base.LLM_CLIENT,
            doc_eval_kwargs={"skip_dea": bool(args.skip_dea)},
        )

    # Default demo path (no DEA dataset specified)
    if not args.dea_solution_json and not args.dea_root:
        questions = [
            "What are the key events in the Apollo 11 mission?",
            "Explain the main causes of World War I.",
        ]

        optimizer = None
        for step in range(2):
            runs: List[RunResult] = []
            for q in questions:
                result = run_graph_with_otel(graph, q, eval_fn=eval_fn)
                runs.append(result)

            updates, optimizer = optimize_iteration(runs, optimizer=optimizer)
            print(f"[iter {step}] score={runs[0].score:.3f} updated={list(updates.keys())}")
        return

    # DEA dataset path: one solution json or a root dataset (output/latex/*.json)
    def load_solution_json(p: str) -> dict:
        return json.loads(Path(p).read_text(encoding="utf-8"))

    solutions: List[tuple[str, dict]] = []
    if args.dea_solution_json:
        sol = load_solution_json(args.dea_solution_json)
        solutions.append((sol.get("title") or "topic", sol))

    if args.dea_root:
        # Import load_dea from document_embedding_analysis if available
        # (If not installed, this will raise and tell user what to fix.)
        try:
            m = __import__("document_embedding_analysis.common.doc_eval", fromlist=["load_dea"])
        except Exception:
            m = __import__("document_analysis_embedding.common.doc_eval", fromlist=["load_dea"])
        load_dea = getattr(m, "load_dea")
        for i, (title, _ctx, sol) in enumerate(load_dea(args.dea_root)):
            if i >= args.max_examples:
                break
            solutions.append((title, sol))

    optimizer = None
    runs: List[RunResult] = []
    for title, sol in solutions:
        q = f'Write a wikipedia like article about "{title}"'
        res = run_graph_with_otel(
            graph,
            q,
            eval_fn=eval_fn,
            eval_data={
                "solution": sol,
                "turns": [],
                "content_type": args.candidate_content_type,
            },
        )
        runs.append(res)
        print(f"\n--- Feedback for {title} ({args.eval_mode}) ---")
        print(res.feedback)
        print(f"Score: {res.score}")
        print("------------------------------------------------\n")

    updates, optimizer = optimize_iteration(runs, optimizer=optimizer)
    print(f"[dea] avg_score={sum(r.score for r in runs)/len(runs):.3f} updated={list(updates.keys())}")


if __name__ == "__main__":
    main()
