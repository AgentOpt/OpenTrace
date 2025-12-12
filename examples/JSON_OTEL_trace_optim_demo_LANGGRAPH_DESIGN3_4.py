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


def run_benchmark(
    eval_mode: str,
    steps: int,
    solutions: List[tuple[str, dict]],
    graph: Any,
    eval_fn: Optional[EvalFn],
    candidate_content_type: str = "markdown",
) -> List[Dict[str, Any]]:
    """
    Run the optimization loop for a specified number of steps.
    Returns a list of stats per iteration.
    """
    print(f"\n🚀 Starting Benchmark: mode={eval_mode}, steps={steps}, examples={len(solutions)}")
    
    current_planner_tmpl = base.PLANNER_TEMPLATE_DEFAULT
    current_executor_tmpl = base.EXECUTOR_TEMPLATE_DEFAULT
    current_synthesizer_tmpl = base.SYNTH_TEMPLATE_DEFAULT
    
    optimizer = None
    stats = []

    for step in range(steps):
        print(f"\n=== Iteration {step+1}/{steps} ===")
        runs: List[RunResult] = []
        
        for title, sol in solutions:
            q = f'Write a wikipedia like article about "{title}"'
            res = run_graph_with_otel(
                graph,
                q,
                planner_template=current_planner_tmpl,
                executor_template=current_executor_tmpl,
                synthesizer_template=current_synthesizer_tmpl,
                eval_fn=eval_fn,
                eval_data={
                    "solution": sol,
                    "turns": [],
                    "content_type": candidate_content_type,
                },
            )
            runs.append(res)
            # Print brief feedback for the first example to avoid spam
            if len(runs) == 1:
                print(f"\n--- Feedback for {title} ({eval_mode}) ---")
                print(res.feedback)
                print(f"Score: {res.score}")
                print("------------------------------------------------\n")

        # Calculate average score for reporting
        # For fair comparison, we try to extract 'benchmark_dea_score' from feedback if available.
        report_scores = []
        for r in runs:
            try:
                fb = json.loads(r.feedback)
                if isinstance(fb, dict) and "benchmark_dea_score" in fb:
                    report_scores.append(fb["benchmark_dea_score"])
                else:
                    report_scores.append(r.score)
            except Exception:
                report_scores.append(r.score)

        avg_score = sum(report_scores) / len(report_scores)
        print(f"[iter {step+1}] avg_score={avg_score:.3f} (using benchmark_dea_score if available)")
        
        stats.append({
            "step": step + 1,
            "avg_score": avg_score,
            "scores": report_scores,
            "metrics": [r.metrics for r in runs]
        })

        if step < steps - 1:
            updates, optimizer = optimize_iteration(runs, optimizer=optimizer)
            
            if updates:
                print(f"   Updated params: {list(updates.keys())}")
                
                # Apply prompt updates
                if "planner_prompt" in updates:
                    current_planner_tmpl = updates["planner_prompt"]
                if "executor_prompt" in updates:
                    current_executor_tmpl = updates["executor_prompt"]
                if "synthesizer_prompt" in updates:
                    current_synthesizer_tmpl = updates["synthesizer_prompt"]
                
                # Apply code updates
                for param_name, new_value in updates.items():
                    if param_name.startswith("__code_"):
                        key = param_name[len("__code_"):]
                        # Use base._apply_code_update
                        if hasattr(base, "_apply_code_update"):
                            ok, msg = base._apply_code_update(key, new_value)
                            print(f"   Code update {key}: {msg}")
                        else:
                            print(f"   ⚠️ Cannot apply code update for {key}: _apply_code_update not found in base")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_mode", default="llm", choices=["llm", "dea", "hybrid"], help="Scoring mode")
    parser.add_argument("--dea_solution_json", default=None, help="Path to a DEA solution JSON (optional)")
    parser.add_argument("--dea_root", default=None, help="Path to DEA root containing output/latex/*.json (optional)")
    parser.add_argument("--max_examples", type=int, default=2, help="Max DEA examples to run when using --dea_root")
    parser.add_argument("--candidate_content_type", default="markdown", help="Candidate content type for doc_eval: markdown|latex")
    parser.add_argument("--skip_dea", action="store_true", help="Pass skip_dea=True to doc_eval (debug/fast)")
    parser.add_argument("--steps", type=int, default=1, help="Number of optimization steps")
    args = parser.parse_args()

    graph = build_graph()

    eval_fn: Optional[EvalFn] = None
    # Always create eval_fn if we have DEA args, even for "llm" mode, 
    # so we can compute DEA metrics for the benchmark report.
    if args.eval_mode in ("dea", "hybrid", "llm") and (args.dea_solution_json or args.dea_root):
        eval_fn = make_document_embedding_analysis_eval(
            mode=args.eval_mode,
            llm=base.LLM_CLIENT,
            doc_eval_kwargs={"skip_dea": bool(args.skip_dea)},
        )

    # Default demo path (no DEA dataset specified)
    if not args.dea_solution_json and not args.dea_root:
        # ... (keep existing default logic or adapt it? I'll adapt it to use run_benchmark for consistency)
        questions = [
            "What are the key events in the Apollo 11 mission?",
            "Explain the main causes of World War I.",
        ]
        # Mock solutions for default path
        solutions = [(q, {}) for q in questions]
        
        # For default path, we need to handle run_graph_with_otel slightly differently as it expects 'title' in solutions loop
        # But run_benchmark expects solutions list.
        # Let's just keep the default path simple or adapt run_benchmark to handle it.
        # Actually, run_benchmark constructs query from title: q = f'Write a wikipedia like article about "{title}"'
        # This is specific to DEA.
        # So I will leave the default path as is, or just warn that --steps is only for DEA mode.
        
        print("Running default demo (non-DEA). Use --dea_solution_json for benchmark.")
        optimizer = None
        for step in range(args.steps):
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

    # Run Benchmark
    stats = run_benchmark(
        eval_mode=args.eval_mode,
        steps=args.steps,
        solutions=solutions,
        graph=graph,
        eval_fn=eval_fn,
        candidate_content_type=args.candidate_content_type
    )
    
    # Print Summary
    print("\n" + "="*40)
    print("BENCHMARK SUMMARY")
    print("="*40)
    for s in stats:
        print(f"Step {s['step']}: Avg Score = {s['avg_score']:.3f}")


if __name__ == "__main__":
    main()
