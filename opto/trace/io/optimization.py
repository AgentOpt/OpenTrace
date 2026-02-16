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
    """Result of ``optimize_graph()``.

    Attributes
    ----------
    baseline_score : float
        Average score of the baseline (iteration 0) run.
    best_score : float
        Highest average score across all iterations.
    best_iteration : int
        Iteration index that produced ``best_score``.
    best_parameters : dict
        Snapshot of all parameter values at ``best_iteration`` (E11).
    best_updates : dict
        The updates dict that was applied to reach ``best_parameters``.
    final_parameters : dict
        Parameter values after the last iteration.
    score_history : list[float]
        Average scores per iteration.
    all_runs : list[list[RunResult]]
        All run results grouped by iteration.
    """

    baseline_score: float
    best_score: float
    best_iteration: int
    best_parameters: Dict[str, Any]
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
# Helpers
# ---------------------------------------------------------------------------


def _snapshot_parameters(bindings: Dict[str, Binding]) -> Dict[str, Any]:
    """Take a snapshot of all current parameter values."""
    snap: Dict[str, Any] = {}
    for key, binding in bindings.items():
        try:
            snap[key] = binding.get()
        except Exception:
            snap[key] = "<error reading binding>"
    return snap


def _deduplicate_param_nodes(param_nodes: list) -> list:
    """Deduplicate trainable ParameterNodes by base name (C7).

    When the same prompt key appears in multiple TGJ docs (e.g. from
    multiple queries in the same iteration), the optimizer should see
    each unique trainable parameter only once.

    Uses the ``name`` attribute (before scope-suffix) as the dedup key,
    falling back to ``py_name`` stripped of trailing digits.
    """
    import re

    seen: Dict[str, Any] = {}
    for n in param_nodes:
        # Prefer the raw name attribute (e.g. "planner_prompt") which
        # doesn't have the scope suffix.  Fall back to py_name with
        # trailing digits stripped (e.g. "planner_prompt0" → "planner_prompt").
        raw_name = getattr(n, "_name", None) or getattr(n, "name", None)
        if raw_name is None:
            raw_name = getattr(n, "py_name", None) or str(id(n))
        # Strip trailing digits added by scope management
        key = re.sub(r"\d+$", "", str(raw_name))
        if key not in seen:
            seen[key] = n
    return list(seen.values())


def _select_output_node(nodes: dict) -> Any:
    """Select the sink (final top-level) MessageNode (C8).

    Excludes child spans — identified by the ``trace.temporal_ignore``
    attribute set during instrumentation — and picks the *last*
    top-level MessageNode.

    This is provider-agnostic: it does not assume any specific LLM
    provider naming convention.
    """
    from opto.trace.nodes import MessageNode as _MN

    # Collect all MessageNodes
    msg_nodes = [n for n in nodes.values() if isinstance(n, _MN)]
    if not msg_nodes:
        return None

    # Filter out child spans using the trace.temporal_ignore marker
    # that was set during instrumentation (see TracingLLM.node_call).
    # Fall back to name-based heuristic only as a safety net.
    top_level = []
    for n in msg_nodes:
        info = getattr(n, "info", None) or {}
        otel_info = info.get("otel", {}) if isinstance(info, dict) else {}

        # Primary gate: trace.temporal_ignore attribute
        if str(otel_info.get("temporal_ignore", "false")).lower() in ("true", "1", "yes"):
            continue

        # Secondary check: the node's description/data may carry the flag
        desc = getattr(n, "description", None) or ""
        if isinstance(desc, dict):
            if str(desc.get("trace.temporal_ignore", "false")).lower() in ("true", "1", "yes"):
                continue

        top_level.append(n)

    if not top_level:
        # Fall back to all msg nodes if filtering was too aggressive
        top_level = msg_nodes

    # Return the last top-level node (the sink / final node)
    return top_level[-1]


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
    output_key: Optional[str] = None,
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
    output_key : str, optional
        Key in the result dict that holds the graph's final answer.
        Used for error fallback and eval payload.  If *None*,
        ``optimize_graph`` passes the full result dict to eval.
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

    # If not provided, fall back to the graph's configured output_key.
    # If both are provided and disagree, prefer the explicit argument.
    graph_output_key = getattr(graph, "output_key", None)
    if output_key is None:
        output_key = graph_output_key
    elif graph_output_key and output_key != graph_output_key:
        logger.debug(
            "optimize_graph: output_key=%r overrides graph.output_key=%r",
            output_key,
            graph_output_key,
        )

    score_history: List[float] = []
    all_runs: List[List[RunResult]] = []
    best_score = float("-inf")
    best_iteration = 0
    best_updates: Dict[str, Any] = {}
    best_parameters: Dict[str, Any] = _snapshot_parameters(effective_bindings)
    # Track the updates applied *before* the current iteration so we know
    # which updates produced the params used in each iteration.
    last_applied_updates: Dict[str, Any] = {}

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

    _input_key = getattr(graph, "input_key", "query") or "query"

    def _make_state(query: Any) -> Dict[str, Any]:
        if isinstance(query, dict):
            return query
        return {_input_key: query}

    # ---- iteration loop ---------------------------------------------------

    total_iters = iterations + 1  # baseline + N iterations

    for iteration in range(total_iters):
        is_baseline = iteration == 0
        # Snapshot which updates were applied to produce this iteration's params
        applied_updates_for_this_iter = dict(last_applied_updates)
        label = "baseline" if is_baseline else f"iteration {iteration}"
        logger.info("optimize_graph: running %s ...", label)
        print(f"  {'Running baseline' if is_baseline else f'Iteration {iteration}/{iterations}'}...")

        runs: List[RunResult] = []
        for qi, query in enumerate(queries):
            state = _make_state(query)

            # E12: Manually control root span lifecycle so we can attach
            # eval attributes *before* the span closes and gets exported.
            query_hint = str(query)[:200] if not isinstance(query, dict) else str(query)[:200]
            invocation_failed = False
            result = None
            er = None

            with graph._root_invocation_span(query_hint) as root_sp:
                try:
                    # Invoke the underlying compiled graph (not graph.invoke
                    # which would create a redundant root span).
                    result = graph.graph.invoke(state)
                except Exception as exc:
                    logger.warning("Graph invocation failed: %s", exc)
                    result = {"_error": str(exc)}
                    invocation_failed = True
                    root_sp.set_attribute("error", "true")
                    root_sp.set_attribute("error.message", str(exc)[:500])

                # E12: Peek at OTLP (child spans are finished and collected,
                # but root span is still open → not yet in exporter).
                otlp_peek = graph.session.flush_otlp(clear=False)

                # Extract the output value (generic — no hardcoded key)
                if output_key and isinstance(result, dict):
                    answer = result.get(output_key, result)
                else:
                    answer = result

                # A4: If invocation failed, force score=0
                if invocation_failed:
                    er = EvalResult(
                        score=0.0,
                        feedback=f"Invocation failed: {result.get('_error', 'unknown')}",
                    )
                else:
                    eval_payload = {
                        "query": query,
                        "answer": answer,
                        "result": result,
                        "otlp": otlp_peek,
                        "iteration": iteration,
                    }
                    er = _normalise_eval(eval_fn(eval_payload))

                # E12: Attach eval score on the root span (still open)
                if er.score is not None:
                    root_sp.set_attribute("eval.score", str(er.score))
                if er.feedback:
                    root_sp.set_attribute(
                        "eval.feedback", str(er.feedback)[:500]
                    )
            # Root span closes here → exported to the in-memory exporter

            # Now flush OTLP with clear=True — includes root span + eval attrs
            otlp = graph.session.flush_otlp(clear=True)

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

        # E11: Track best parameters snapshot
        if avg_score > best_score:
            best_score = avg_score
            best_iteration = iteration
            best_parameters = _snapshot_parameters(effective_bindings)
            best_updates = dict(applied_updates_for_this_iter)
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
                # C7: Collect and deduplicate param nodes across all runs
                all_param_nodes: list = []
                all_output_nodes: list = []

                for run in runs:
                    tgj_docs = graph.session._flush_tgj_from_otlp(run.otlp)
                    if not tgj_docs:
                        from opto.trace.io.otel_adapter import otlp_traces_to_trace_json
                        tgj_docs = otlp_traces_to_trace_json(
                            run.otlp,
                            agent_id_hint=graph.session.service_name,
                            use_temporal_hierarchy=True,
                        )

                    for doc in tgj_docs:
                        nodes = _ingest_tgj(doc)

                        from opto.trace.nodes import ParameterNode as _PN
                        param_nodes = [
                            n for n in nodes.values()
                            if isinstance(n, _PN) and n.trainable
                        ]
                        all_param_nodes.extend(param_nodes)

                        # C8: Select output node properly
                        output_node = _select_output_node(nodes)
                        if output_node is not None:
                            all_output_nodes.append((output_node, run))

                # C7: Deduplicate before passing to optimizer
                unique_params = _deduplicate_param_nodes(all_param_nodes)

                if not unique_params:
                    logger.info("No trainable ParameterNodes found; skipping optimizer step.")
                else:
                    _ensure_optimizer(unique_params)

                    if _optimizer is not None and all_output_nodes:
                        # Use the last output node for backward pass
                        output_node, run_for_output = all_output_nodes[-1]
                        feedback_text = run_for_output.feedback or (
                            f"Score: {run_for_output.score}"
                            if run_for_output.score is not None
                            else "No feedback"
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
                    last_applied_updates = dict(updates)
                    logger.info("Applied updates: %s", sorted(updates.keys()))
                except Exception as exc:
                    logger.warning("apply_updates failed: %s", exc, exc_info=True)

            if on_iteration:
                on_iteration(iteration, runs, updates)

    # -- build final parameters snapshot --
    final_params = _snapshot_parameters(effective_bindings)

    return OptimizationResult(
        baseline_score=score_history[0] if score_history else 0.0,
        best_score=best_score,
        best_iteration=best_iteration,
        best_parameters=best_parameters,
        best_updates=best_updates,
        final_parameters=final_params,
        score_history=score_history,
        all_runs=all_runs,
    )
