"""
opto.trace.io.optimization
===========================

One-liner ``optimize_graph()`` for running end-to-end optimization on an
instrumented LangGraph:

    instrument → invoke → flush OTLP → TGJ → ingest → optimizer → apply_updates

This module also defines ``EvalResult``, ``EvalFn``, ``RunResult``, and
``OptimizationResult`` as the public data contracts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from opto.trace.io.bindings import Binding, apply_updates
from opto.trace.io.instrumentation import InstrumentedGraph
from opto.trace.io.otel_semconv import emit_reward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation contract
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Normalised output of an evaluation function.

    Attributes
    ----------
    score : float or None
        Numeric reward (some evaluators return only text feedback).
    feedback : str
        Textual feedback (Trace / TextGrad-compatible).
    metrics : dict
        Free-form metrics for logging / diagnostics.
    """

    score: Optional[float] = None
    feedback: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


# eval_fn may return float | str | dict | EvalResult
EvalFn = Callable[[Dict[str, Any]], Union[float, str, Dict[str, Any], EvalResult]]


def _normalise_eval(raw: Any) -> EvalResult:
    """Normalise any ``eval_fn`` return value into ``EvalResult``."""
    if isinstance(raw, EvalResult):
        return raw
    if isinstance(raw, (int, float)):
        return EvalResult(score=float(raw))
    if isinstance(raw, str):
        # Attempt JSON parse
        try:
            d = json.loads(raw)
            if isinstance(d, dict):
                return EvalResult(
                    score=d.get("score"),
                    feedback=str(d.get("feedback", d.get("reasons", ""))),
                    metrics=d,
                )
        except (json.JSONDecodeError, TypeError):
            pass
        return EvalResult(feedback=raw)
    if isinstance(raw, dict):
        return EvalResult(
            score=raw.get("score"),
            feedback=str(raw.get("feedback", raw.get("reasons", ""))),
            metrics=raw,
        )
    return EvalResult(feedback=str(raw))


# ---------------------------------------------------------------------------
# Run / Optimization results
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Result of a single graph execution."""

    answer: Any
    score: Optional[float]
    feedback: str
    metrics: Dict[str, Any]
    otlp: Dict[str, Any]


@dataclass
class OptimizationResult:
    """Result of ``optimize_graph()``."""

    baseline_score: float
    best_score: float
    best_iteration: int
    best_updates: Dict[str, Any]
    final_parameters: Dict[str, Any]
    score_history: List[float]
    all_runs: List[List[RunResult]]


# ---------------------------------------------------------------------------
# Default eval_fn (LLM-as-judge via evaluator span)
# ---------------------------------------------------------------------------


def _default_eval_fn(payload: Dict[str, Any]) -> EvalResult:
    """Extract evaluation from the OTLP trace's evaluator span, if present."""
    from opto.trace.io.langgraph_otel_runtime import extract_eval_metrics_from_otlp

    otlp = payload.get("otlp", {})
    score, metrics, reasons = extract_eval_metrics_from_otlp(otlp)
    return EvalResult(score=score, feedback=reasons, metrics=metrics)


# ---------------------------------------------------------------------------
# optimize_graph
# ---------------------------------------------------------------------------


def optimize_graph(
    graph: InstrumentedGraph,
    queries: Union[List[str], List[Dict[str, Any]]],
    *,
    iterations: int = 5,
    optimizer: Optional[Any] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    eval_fn: Optional[EvalFn] = None,
    initial_templates: Optional[Dict[str, str]] = None,
    bindings: Optional[Dict[str, Binding]] = None,
    apply_updates_flag: bool = True,
    include_log_doc: bool = False,
    on_iteration: Optional[
        Callable[[int, List[RunResult], Dict[str, Any]], None]
    ] = None,
) -> OptimizationResult:
    """Run a complete optimization loop on an instrumented LangGraph.

    Flow per iteration
    ------------------
    1. Invoke graph for each query and capture OTLP traces.
    2. Evaluate each run via ``eval_fn`` (→ ``EvalResult``).
    3. Convert OTLP → TGJ → Trace nodes via ``ingest_tgj``.
    4. Propagate feedback through the Trace graph.
    5. Ask the optimizer for parameter updates.
    6. Apply updates via ``apply_updates(updates, bindings)``.

    Parameters
    ----------
    graph : InstrumentedGraph
        The instrumented graph (from ``instrument_graph``).
    queries : list
        Test queries (strings) or full state dicts.
    iterations : int
        Number of optimisation iterations (after baseline).
    optimizer : OptoPrimeV2, optional
        Pre-configured optimizer.  Created automatically if absent.
    optimizer_kwargs : dict, optional
        Arguments passed to optimizer creation.
    eval_fn : EvalFn, optional
        Custom evaluation function.  Falls back to evaluator-span extraction.
    initial_templates : dict, optional
        Overrides for initial prompt templates.
    bindings : dict, optional
        Overrides for graph.bindings.
    apply_updates_flag : bool
        If *True* (default), apply parameter updates each iteration.
    include_log_doc : bool
        If *True*, emit additional ``log_doc`` TGJ artefacts.
    on_iteration : callable, optional
        ``(iter_num, runs, updates_dict) -> None`` progress callback.

    Returns
    -------
    OptimizationResult
    """
    # Resolve bindings / templates
    effective_bindings = bindings or graph.bindings
    if initial_templates:
        graph.templates.update(initial_templates)

    eval_fn = eval_fn or _default_eval_fn

    score_history: List[float] = []
    all_runs: List[List[RunResult]] = []
    best_score = float("-inf")
    best_iteration = 0
    best_updates: Dict[str, Any] = {}

    # -- lazy imports for Trace framework --
    _ingest_tgj = None
    _GraphPropagator = None
    _optimizer = optimizer

    def _ensure_trace_imports():
        nonlocal _ingest_tgj, _GraphPropagator
        if _ingest_tgj is None:
            from opto.trace.io.tgj_ingest import ingest_tgj as _fn
            _ingest_tgj = _fn
        if _GraphPropagator is None:
            try:
                from opto.trace.propagators.graph_propagator import GraphPropagator
                _GraphPropagator = GraphPropagator
            except ImportError:
                _GraphPropagator = None

    def _ensure_optimizer(param_nodes):
        nonlocal _optimizer
        if _optimizer is not None:
            return
        try:
            from opto.optimizers import OptoPrime
            kw = dict(optimizer_kwargs or {})
            _optimizer = OptoPrime(param_nodes, **kw)
        except ImportError:
            logger.warning(
                "Could not import OptoPrime; running in eval-only mode "
                "(no parameter updates)."
            )

    def _make_state(query: Any) -> Dict[str, Any]:
        if isinstance(query, dict):
            return query
        return {"query": query}

    # ---- iteration loop ---------------------------------------------------

    total_iters = iterations + 1  # baseline + N iterations

    for iteration in range(total_iters):
        is_baseline = iteration == 0
        label = "baseline" if is_baseline else f"iteration {iteration}"
        logger.info("optimize_graph: running %s ...", label)
        print(f"  {'Running baseline' if is_baseline else f'Iteration {iteration}/{iterations}'}...")

        runs: List[RunResult] = []
        for qi, query in enumerate(queries):
            state = _make_state(query)
            result = graph.invoke(state)

            # Flush OTLP *before* evaluation so eval_fn can inspect spans
            otlp = graph.session.flush_otlp(clear=True)

            # Evaluate
            answer = result if isinstance(result, str) else result
            eval_payload = {
                "query": query,
                "answer": answer,
                "result": result,
                "otlp": otlp,
                "iteration": iteration,
            }
            er = _normalise_eval(eval_fn(eval_payload))

            # Record eval reward span
            if er.score is not None:
                emit_reward(
                    graph.session,
                    value=er.score,
                    name="eval_score",
                )
                # Flush the reward span
                graph.session.flush_otlp(clear=True)

            runs.append(
                RunResult(
                    answer=answer,
                    score=er.score,
                    feedback=er.feedback,
                    metrics=er.metrics,
                    otlp=otlp,
                )
            )

            q_display = str(query)[:40] if not isinstance(query, dict) else str(query)[:40]
            print(
                f"    Query {qi + 1}/{len(queries)}: {q_display}... "
                f"score={er.score if er.score is not None else 'N/A'}"
            )

        # Compute average score
        scored_runs = [r for r in runs if r.score is not None]
        if scored_runs:
            avg_score = sum(r.score for r in scored_runs) / len(scored_runs)
        else:
            avg_score = 0.0

        score_history.append(avg_score)
        all_runs.append(runs)

        if avg_score > best_score:
            best_score = avg_score
            best_iteration = iteration
            marker = " * NEW BEST" if not is_baseline else ""
        else:
            marker = ""
        print(f"  {'Baseline' if is_baseline else f'Iteration {iteration}'} average: {avg_score:.4f}{marker}")

        # -- optimization step (skip for baseline) --
        if not is_baseline and effective_bindings:
            _ensure_trace_imports()

            # Convert OTLP → TGJ → Trace nodes
            updates: Dict[str, Any] = {}
            try:
                for run in runs:
                    tgj_docs = graph.session._flush_tgj_from_otlp(run.otlp)
                    if not tgj_docs:
                        # Fall back to direct conversion
                        from opto.trace.io.otel_adapter import otlp_traces_to_trace_json
                        tgj_docs = otlp_traces_to_trace_json(
                            run.otlp,
                            agent_id_hint=graph.session.service_name,
                            use_temporal_hierarchy=True,
                        )

                    for doc in tgj_docs:
                        nodes = _ingest_tgj(doc)

                        # Find trainable ParameterNodes
                        from opto.trace.nodes import ParameterNode as _PN
                        param_nodes = [
                            n for n in nodes.values()
                            if isinstance(n, _PN) and n.trainable
                        ]

                        if not param_nodes:
                            continue

                        _ensure_optimizer(param_nodes)

                        if _optimizer is None:
                            continue

                        # Find the last MessageNode as the output
                        from opto.trace.nodes import MessageNode as _MN
                        msg_nodes = [
                            n for n in nodes.values() if isinstance(n, _MN)
                        ]
                        if not msg_nodes:
                            continue
                        output_node = msg_nodes[-1]

                        # Propagate feedback
                        feedback_text = run.feedback or (
                            f"Score: {run.score}" if run.score is not None else "No feedback"
                        )
                        try:
                            _optimizer.zero_feedback()
                            _optimizer.backward(output_node, feedback_text)
                            raw_updates = _optimizer.step()

                            if isinstance(raw_updates, dict):
                                updates.update(raw_updates)
                        except Exception as exc:
                            logger.warning(
                                "Optimizer step failed: %s", exc, exc_info=True
                            )

            except Exception as exc:
                logger.warning(
                    "TGJ conversion / optimization failed: %s", exc, exc_info=True
                )

            # Apply updates
            if updates and apply_updates_flag:
                try:
                    apply_updates(updates, effective_bindings, strict=False)
                    best_updates = dict(updates)
                    logger.info("Applied updates: %s", sorted(updates.keys()))
                except Exception as exc:
                    logger.warning("apply_updates failed: %s", exc, exc_info=True)

            if on_iteration:
                on_iteration(iteration, runs, updates)

    # -- build final parameters snapshot --
    final_params: Dict[str, Any] = {}
    for key, binding in effective_bindings.items():
        try:
            final_params[key] = binding.get()
        except Exception:
            final_params[key] = "<error reading binding>"

    return OptimizationResult(
        baseline_score=score_history[0] if score_history else 0.0,
        best_score=best_score,
        best_iteration=best_iteration,
        best_updates=best_updates,
        final_parameters=final_params,
        score_history=score_history,
        all_runs=all_runs,
    )
