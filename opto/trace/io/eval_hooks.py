from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

EvalFn = Callable[
    [str, float, Dict[str, float], str, Dict[str, Any], Dict[str, Any]],
    Tuple[float, Dict[str, float], str],
]


def default_feedback(score: float, metrics: Dict[str, float], reasons: str) -> str:
    return json.dumps({"score": score, "metrics": metrics, "reasons": reasons})


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _ratio_closeness(r: float) -> float:
    """
    Convert ratio-to-target (ideal=1.0) into a [0,1] closeness score.
    """
    try:
        r = float(r)
    except Exception:
        return 0.0
    return _clip01(1.0 - abs(1.0 - r))


def _dea_overall_from_scores(dea_scores: Mapping[str, Any]) -> Optional[float]:
    """
    Robust aggregate over DEA signals:
    - ratios -> closeness
    - similarities/coverage assumed in [0,1]
    - ignore out-of-range values
    """
    if not dea_scores:
        return None

    ratio_keys = {
        "sections_count_ratio_to_target",
        "content_length_ratio_to_target",
        "resources_count_ratio_to_target",
    }

    vals: List[float] = []
    for k, v in dea_scores.items():
        try:
            fv = float(v)
        except Exception:
            continue

        if k in ratio_keys:
            vals.append(_ratio_closeness(fv))
        else:
            if 0.0 <= fv <= 1.0:
                vals.append(_clip01(fv))

    if not vals:
        return None
    return sum(vals) / len(vals)


def _try_import_evaluate_document():
    """
    Best-effort import of doc_eval.evaluate_document.
    We keep this robust because users might have different top-level package names.
    """
    candidates = [
        "document_embedding_analysis.common.doc_eval",
        "document_analysis_embedding.common.doc_eval",
        "common.doc_eval",  # allows running inside the external repo directly
    ]
    for mod in candidates:
        try:
            m = __import__(mod, fromlist=["evaluate_document"])
            fn = getattr(m, "evaluate_document", None)
            if fn is not None:
                return fn, m
        except Exception:
            continue
    return None, None


def _synthesize_hybrid_feedback(
    llm: Any,
    answer: str,
    original_reasons: str,
    dea_scores: Dict[str, Any],
) -> str:
    """
    Use the LLM to synthesize a new feedback string combining the original reasons
    and the objective DEA scores.
    """
    if not llm:
        return original_reasons

    # Format DEA scores for the prompt
    dea_summary = []
    for k, v in dea_scores.items():
        if isinstance(v, (int, float)):
            dea_summary.append(f"{k}: {v:.3f}")
        else:
            dea_summary.append(f"{k}: {v}")
    dea_text = ", ".join(dea_summary)

    prompt = f"""
You are an expert evaluator.
You have evaluated a generated document and provided the following initial feedback:
"{original_reasons}"

Additionally, an automated Document Embedding Analysis (DEA) system has provided the following objective metrics:
{dea_text}

Please synthesize a new, comprehensive feedback explanation that incorporates both your initial qualitative assessment and these quantitative DEA metrics.
Focus on explaining *why* the score is what it is, citing specific metrics where relevant (e.g., "The content is semantically close on plan (0.85) but lacks specific entities...").
Keep the feedback concise and constructive.
""".strip()

    try:
        # Assume LangChain-like interface
        from langchain_core.messages import HumanMessage
        if hasattr(llm, "invoke"):
            response = llm.invoke([HumanMessage(content=prompt)])
            return str(response.content)
    except Exception:
        pass

    try:
        # Assume Opto/AutoGen interface
        # llm(messages=...) returns a response object with choices
        response = llm(messages=[{"role": "user", "content": prompt}])
        
        # Handle object access
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return str(choice.message.content)
        
        # Handle dict access
        if isinstance(response, dict) and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return str(choice["message"]["content"])
                
    except Exception:
        pass

    return original_reasons


def make_document_embedding_analysis_eval(
    mode: str = "dea",
    *,
    llm: Optional[Any] = None,
    weight_llm: float = 0.5,
    weight_dea: float = 0.5,
    doc_eval_kwargs: Optional[Dict[str, Any]] = None,
    dea_score_key: Optional[str] = None,
) -> EvalFn:
    """
    Build an EvalFn backed by document_embedding_analysis.common.doc_eval.evaluate_document.

    eval_data expected keys:
      - solution: dict (required for DEA)
      - turns: list (optional)
      - content_type: "markdown"|"latex" (optional, default "markdown")
      - doc_eval_kwargs: dict (optional overrides per-example)
    """
    mode = (mode or "").lower().strip()
    
    # Default: disable enhanced metrics (Prometheus, WriteHere) unless explicitly enabled
    base_kwargs = {"use_enhanced_metrics": False}
    if doc_eval_kwargs:
        base_kwargs.update(doc_eval_kwargs)

    def _eval(
        answer: str,
        llm_score: float,
        llm_metrics: Dict[str, float],
        reasons: str,
        otlp: Dict[str, Any],
        eval_data: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float], str]:
        evaluate_document, _mod = _try_import_evaluate_document()
        if evaluate_document is None:
            return llm_score, dict(llm_metrics), default_feedback(llm_score, dict(llm_metrics), reasons)

        solution = eval_data.get("solution")
        if solution is None:
            return llm_score, dict(llm_metrics), default_feedback(llm_score, dict(llm_metrics), reasons)

        turns = eval_data.get("turns") or []
        content_type = eval_data.get("content_type") or "markdown"

        kwargs = dict(base_kwargs)
        if isinstance(eval_data.get("doc_eval_kwargs"), dict):
            kwargs.update(eval_data["doc_eval_kwargs"])

        try:
            result = evaluate_document(
                answer,
                turns=turns,
                solution=solution,
                content_type=content_type,
                **kwargs,
            )
        except Exception as e:
            metrics = dict(llm_metrics)
            metrics["dea.error"] = 1.0
            feedback = json.dumps(
                {
                    "score": llm_score,
                    "reasons": reasons,
                    "metrics": metrics,
                    "dea_exception": repr(e),
                }
            )
            return llm_score, metrics, feedback

        if not isinstance(result, dict):
            return llm_score, dict(llm_metrics), default_feedback(llm_score, dict(llm_metrics), reasons)

        dea_scores = result.get("dea_evaluation_scores") or {}
        article_metrics = result.get("article_metrics") or {}
        prometheus_scores = result.get("prometheus_scores") or {}
        writehere_scores = result.get("writehere_scores") or {}

        # Keep backward compatibility: base metrics are the LLM-as-judge ones.
        metrics: Dict[str, float] = dict(llm_metrics)

        # DEA metrics
        if isinstance(dea_scores, Mapping):
            for k, v in dea_scores.items():
                try:
                    metrics[f"dea.{k}"] = float(v)
                except Exception:
                    continue

        # Article metrics (ROUGE f scores + entity recall)
        if isinstance(article_metrics, Mapping):
            rouge_scores = article_metrics.get("rouge_scores") or {}
            if isinstance(rouge_scores, Mapping):
                for name, vals in rouge_scores.items():
                    if not isinstance(vals, Mapping):
                        continue
                    if "f" in vals:
                        try:
                            metrics[f"{name}_f"] = float(vals["f"])
                        except Exception:
                            pass
            if "entity_recall" in article_metrics:
                try:
                    metrics["entity_recall"] = float(article_metrics["entity_recall"])
                except Exception:
                    pass

        # Enhanced metrics if enabled
        if isinstance(prometheus_scores, Mapping):
            for k, v in prometheus_scores.items():
                if isinstance(v, (int, float)):
                    metrics[f"prometheus.{k}"] = float(v)
        if isinstance(writehere_scores, Mapping):
            for k, v in writehere_scores.items():
                if isinstance(v, (int, float)):
                    metrics[f"writehere.{k}"] = float(v)

        dea_scalar: Optional[float] = None
        if dea_score_key and isinstance(dea_scores, Mapping) and dea_score_key in dea_scores:
            try:
                dea_scalar = float(dea_scores[dea_score_key])
            except Exception:
                dea_scalar = None
        if dea_scalar is None and isinstance(dea_scores, Mapping):
            dea_scalar = _dea_overall_from_scores(dea_scores)
        if dea_scalar is None:
            dea_scalar = llm_score

        final_reasons = reasons
        if mode == "dea":
            score = float(dea_scalar)
        elif mode == "hybrid":
            # Hybrid mode: Use DEA score for optimization, but enrich feedback with LLM synthesis
            # The user requested "measure should be all a DEA measure" for the benchmark.
            # So we return DEA score as the primary score.
            score = float(dea_scalar)
            if llm:
                final_reasons = _synthesize_hybrid_feedback(llm, answer, reasons, dea_scores)
        elif mode == "llm":
            # LLM mode: Use LLM score for optimization, but include DEA metrics in the payload
            # for benchmarking purposes.
            score = llm_score
        else:  # unknown
            score = llm_score

        feedback_payload: Dict[str, Any] = {
            "score": score,
            "reasons": final_reasons,
            "metrics": metrics,
            "dea_evaluation_scores": dea_scores,
            "article_metrics": article_metrics,
            "prometheus_scores": prometheus_scores,
            "writehere_scores": writehere_scores,
            # Explicitly store DEA score for benchmark extraction regardless of optimization score
            "benchmark_dea_score": float(dea_scalar)
        }
        return score, metrics, json.dumps(feedback_payload)

    return _eval
