from __future__ import annotations

import contextlib
import contextvars
import copy
import inspect
import io
import json
import statistics
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_LOG_CONTEXT = contextvars.ContextVar("opto_optimize_anything_log_context", default=None)


def set_log_context(logs: Optional[List[str]]):
    """Set the context-local optimize_anything log sink and return its token."""
    return _LOG_CONTEXT.set(logs)


def reset_log_context(token) -> None:
    _LOG_CONTEXT.reset(token)


def get_log_context() -> Optional[List[str]]:
    return _LOG_CONTEXT.get()


def log(*values: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Append to the current evaluation log sink, or print outside one."""
    text = sep.join(str(v) for v in values)
    sink = get_log_context()
    if sink is None:
        print(text, end=end, flush=flush)
    else:
        sink.append(text)


@dataclass
class EngineConfig:
    max_metric_calls: int = 20
    max_steps: Optional[int] = None
    higher_is_better: bool = True
    cache_evaluation: bool = True
    capture_stdio: bool = False
    candidate_selection_strategy: str = "best"
    frontier_type: str = "score"
    random_seed: int = 0


@dataclass
class ReflectionConfig:
    custom_candidate_proposer: Optional[Callable[..., Any]] = None
    reflection_lm: Optional[Any] = None
    reflection_minibatch_size: int = 1


@dataclass
class RefinerConfig:
    enabled: bool = False
    max_refinements: int = 0


@dataclass
class MergeConfig:
    enabled: bool = False
    max_merge_candidates: int = 4


@dataclass
class TrackingConfig:
    enabled: bool = True
    run_name: Optional[str] = None


@dataclass
class GEPAConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


@dataclass
class OptimizationState:
    step: int = 0
    metric_calls: int = 0
    candidate: Any = None
    best_candidate: Any = None
    best_score: Optional[float] = None
    objective: Optional[str] = None
    config: GEPAConfig = field(default_factory=GEPAConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRecord:
    candidate: Any
    example: Any
    score: float
    side_info: Any = None
    stdout: str = ""
    stderr: str = ""
    logs: List[str] = field(default_factory=list)
    cached: bool = False
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": _json_like(self.candidate),
            "example": _json_like(self.example),
            "score": self.score,
            "side_info": _json_like(self.side_info),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "logs": list(self.logs),
            "cached": self.cached,
            "step": self.step,
        }


@dataclass
class GEPAResult:
    best_candidate: Any
    best_score: Optional[float]
    candidate_scores: List[Tuple[Any, Optional[float]]]
    history: List[EvaluationRecord]
    config: GEPAConfig
    total_metric_calls: int
    validation_score: Optional[float] = None
    validation_records: List[EvaluationRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def candidates(self) -> List[Any]:
        return [candidate for candidate, _ in self.candidate_scores]

    @property
    def scores(self) -> List[float]:
        return [score for _, score in self.candidate_scores if score is not None]

    @property
    def metric_calls(self) -> int:
        return self.total_metric_calls

    @property
    def validation_history(self) -> List[EvaluationRecord]:
        return self.validation_records

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_candidate": _json_like(self.best_candidate),
            "best_score": self.best_score,
            "candidate_scores": [
                {"candidate": _json_like(candidate), "score": score}
                for candidate, score in self.candidate_scores
            ],
            "candidates": _json_like(self.candidates),
            "scores": self.scores,
            "history": [r.to_dict() for r in self.history],
            "validation_score": self.validation_score,
            "validation_records": [r.to_dict() for r in self.validation_records],
            "validation_history": [r.to_dict() for r in self.validation_records],
            "total_metric_calls": self.total_metric_calls,
            "metric_calls": self.total_metric_calls,
            "config": _json_like(asdict(self.config)),
            "metadata": _json_like(self.metadata),
        }


def make_litellm_lm(*args: Any, **kwargs: Any) -> Any:
    """Return Trace's LiteLLM backend lazily, matching GEPA-style helpers."""
    from opto.utils.llm import LiteLLM

    return LiteLLM(*args, **kwargs)



def _json_like(value: Any) -> Any:
    if is_dataclass(value):
        return _json_like(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_like(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_like(v) for v in value]
    if isinstance(value, set):
        return sorted((_json_like(v) for v in value), key=repr)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _stable(value: Any) -> Any:
    if is_dataclass(value):
        return _stable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in sorted(value.items(), key=lambda item: repr(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    if isinstance(value, set):
        return sorted((_stable(v) for v in value), key=repr)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _copy_config(config: Optional[GEPAConfig]) -> GEPAConfig:
    return copy.deepcopy(config) if config is not None else GEPAConfig()


def _patch_config_from_kwargs(config: GEPAConfig, kwargs: Dict[str, Any]) -> GEPAConfig:
    groups = {
        "engine": EngineConfig,
        "reflection": ReflectionConfig,
        "refiner": RefinerConfig,
        "merge": MergeConfig,
        "tracking": TrackingConfig,
    }
    for key in list(kwargs):
        for attr, cls in groups.items():
            if key in cls.__dataclass_fields__:
                setattr(getattr(config, attr), key, kwargs.pop(key))
                break
    return config


def _stable_json(value: Any) -> str:
    return json.dumps(_stable(value), sort_keys=True, separators=(",", ":"), default=repr)


def _cache_key(candidate: Any, example: Any) -> Tuple[str, str]:
    return _stable_json(candidate), _stable_json(example)


def _examples(dataset: Optional[Iterable[Any]]) -> List[Any]:
    if dataset is None:
        return [None]
    values = dataset if isinstance(dataset, list) else list(dataset)
    return values or [None]


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _score_from_side_info(side_info: Any) -> Optional[float]:
    if not isinstance(side_info, dict) or "scores" not in side_info:
        return None
    scores = side_info["scores"]
    values = scores.values() if isinstance(scores, dict) else scores if isinstance(scores, (list, tuple)) else []
    numeric = [float(v) for v in values if isinstance(v, (int, float, bool))]
    return _mean(numeric) if numeric else None


def _coerce_evaluator_return(value: Any) -> Tuple[float, Any]:
    score, side_info = (value if isinstance(value, tuple) and len(value) == 2 else (value, None))
    if isinstance(score, (int, float, bool)):
        return float(score), side_info
    inferred = _score_from_side_info(side_info)
    if inferred is not None:
        return inferred, side_info
    raise TypeError("Evaluator must return a numeric score, bool, or (score, side_info) with numeric side_info['scores'].")


def _positional_capacity(sig: inspect.Signature) -> Tuple[int, bool]:
    count = 0
    varargs = False
    for p in sig.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            varargs = True
        elif p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty:
            count += 1
    return count, varargs


def _call_flex(fn: Callable[..., Any], ordered: Sequence[Any], **available: Any) -> Any:
    """Call a GEPA-style evaluator/proposer with flexible signatures.

    Prefer keyword dispatch only when all required positional-or-keyword
    parameters can be satisfied by known names. Otherwise fall back to
    positional dispatch using the supplied ordered arguments.

    This avoids a subtle bug for callables such as:

        def evaluator(candidate, e): ...

    where the first parameter name is known but the second one is arbitrary.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*ordered)

    params = list(sig.parameters.values())

    required_positional = [
        p
        for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    has_varkw = any(p.kind == p.VAR_KEYWORD for p in params)
    has_positional_only = any(p.kind == p.POSITIONAL_ONLY for p in params)

    required_names_are_known = all(
        p.kind != p.POSITIONAL_ONLY and p.name in available
        for p in required_positional
    )

    if required_names_are_known and not has_positional_only:
        kwargs = dict(available) if has_varkw else {
            k: v
            for k, v in available.items()
            if k in sig.parameters
            and sig.parameters[k].kind
            in (sig.parameters[k].POSITIONAL_OR_KEYWORD, sig.parameters[k].KEYWORD_ONLY)
        }
        return fn(**kwargs)

    required, varargs = _positional_capacity(sig)
    n = len(ordered) if (varargs or has_varargs) else min(len(ordered), max(required, 1))
    positional = list(ordered[:n])
    kwargs = {
        p.name: available[p.name]
        for p in params
        if p.kind == p.KEYWORD_ONLY and p.name in available
    }
    if has_varkw:
        kwargs.update({k: v for k, v in available.items() if k not in kwargs})
    return fn(*positional, **kwargs)


class EvaluatorWrapper:
    def __init__(self, evaluator: Callable[..., Any], config: EngineConfig):
        self.evaluator = evaluator
        self.config = config
        self.cache: Dict[Tuple[str, str], EvaluationRecord] = {}

    def __call__(self, *, candidate: Any, example: Any, opt_state: OptimizationState, count_budget: bool = True) -> EvaluationRecord:
        key = _cache_key(candidate, example)
        if self.config.cache_evaluation and key in self.cache:
            cached = copy.deepcopy(self.cache[key])
            cached.cached = True
            cached.step = opt_state.step
            return cached

        logs: List[str] = []
        stdout, stderr = io.StringIO(), io.StringIO()
        token = set_log_context(logs)
        try:
            out_cm = contextlib.redirect_stdout(stdout) if self.config.capture_stdio else contextlib.nullcontext()
            err_cm = contextlib.redirect_stderr(stderr) if self.config.capture_stdio else contextlib.nullcontext()
            with out_cm, err_cm:
                raw = _call_flex(
                    self.evaluator,
                    (candidate, example, opt_state),
                    candidate=candidate,
                    example=example,
                    opt_state=opt_state,
                )
        finally:
            reset_log_context(token)

        score, side_info = _coerce_evaluator_return(raw)
        if count_budget:
            opt_state.metric_calls += 1
        record = EvaluationRecord(
            candidate=copy.deepcopy(candidate),
            example=copy.deepcopy(example),
            score=score,
            side_info=copy.deepcopy(side_info),
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
            logs=logs,
            step=opt_state.step,
        )
        if self.config.cache_evaluation:
            self.cache[key] = copy.deepcopy(record)
        return record


def _is_better(score: Optional[float], incumbent: Optional[float], higher_is_better: bool) -> bool:
    if score is None:
        return False
    if incumbent is None:
        return True
    return score > incumbent if higher_is_better else score < incumbent


def _aggregate(records: Sequence[EvaluationRecord]) -> Optional[float]:
    return _mean([r.score for r in records]) if records else None


def _feedback(candidate: Any, objective: Optional[str], score: Optional[float], records: Sequence[EvaluationRecord]) -> str:
    lines = []
    if objective:
        lines.append(f"Objective: {objective}")
    lines.append(f"Candidate: {candidate!r}")
    lines.append(f"Aggregate score: {score}")
    for i, r in enumerate(records):
        lines.append(f"Example {i}: score={r.score}, side_info={r.side_info!r}")
        if r.logs:
            lines.append("Logs: " + " | ".join(r.logs))
        if r.stdout:
            lines.append("Stdout: " + r.stdout.strip())
        if r.stderr:
            lines.append("Stderr: " + r.stderr.strip())
    return "\n".join(lines)


def _normalize_proposals(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return [raw]


def _default_proposer() -> Callable[..., Any]:
    from opto.optimize_anything.trace_backend import TraceOptimizerBackend

    return TraceOptimizerBackend()


def _call_proposer(proposer: Callable[..., Any], *, candidate: Any, feedback: str, objective: Optional[str], side_info: Any, opt_state: OptimizationState) -> List[Any]:
    raw = _call_flex(
        proposer,
        (candidate, feedback),
        candidate=candidate,
        feedback=feedback,
        objective=objective,
        side_info=side_info,
        opt_state=opt_state,
    )
    return _normalize_proposals(raw)


def _evaluate_candidate(wrapper: EvaluatorWrapper, candidate: Any, examples: Sequence[Any], opt_state: OptimizationState, budget: int) -> List[EvaluationRecord]:
    records: List[EvaluationRecord] = []
    for example in examples:
        key = _cache_key(candidate, example)
        if opt_state.metric_calls >= budget and not (wrapper.config.cache_evaluation and key in wrapper.cache):
            break
        opt_state.candidate = candidate
        records.append(wrapper(candidate=candidate, example=example, opt_state=opt_state))
    return records


def optimize_anything(
    *,
    seed_candidate: Any = None,
    evaluator: Callable[..., Any],
    dataset: Optional[Iterable[Any]] = None,
    valset: Optional[Iterable[Any]] = None,
    objective: Optional[str] = None,
    config: Optional[GEPAConfig] = None,
    **direct_config_kwargs: Any,
) -> GEPAResult:
    if evaluator is None:
        raise ValueError("evaluator is required")

    config = _patch_config_from_kwargs(_copy_config(config), direct_config_kwargs)
    if direct_config_kwargs:
        raise TypeError("Unknown optimize_anything keyword argument(s): " + ", ".join(sorted(direct_config_kwargs)))

    train_examples = _examples(dataset)
    validation_examples = _examples(valset) if valset is not None else []
    wrapper = EvaluatorWrapper(evaluator, config.engine)
    proposer = config.reflection.custom_candidate_proposer or _default_proposer()
    opt_state = OptimizationState(objective=objective, config=config)

    best_candidate = None
    best_score: Optional[float] = None
    current_candidate = seed_candidate
    current_score: Optional[float] = None
    current_records: List[EvaluationRecord] = []
    history: List[EvaluationRecord] = []
    candidate_scores: List[Tuple[Any, Optional[float]]] = []
    candidate_records: Dict[str, List[EvaluationRecord]] = {}

    def evaluate(candidate: Any) -> Tuple[Optional[float], List[EvaluationRecord]]:
        nonlocal best_candidate, best_score
        records = _evaluate_candidate(wrapper, candidate, train_examples, opt_state, config.engine.max_metric_calls)
        if not records:
            return None, []
        score = _aggregate(records)
        history.extend(records)
        candidate_scores.append((copy.deepcopy(candidate), score))
        candidate_records[_stable_json(candidate)] = records
        if _is_better(score, best_score, config.engine.higher_is_better):
            best_candidate = copy.deepcopy(candidate)
            best_score = score
            opt_state.best_candidate = best_candidate
            opt_state.best_score = best_score
        return score, records

    current_score, current_records = evaluate(current_candidate)
    max_steps = 0 if config.engine.max_steps == 0 else (config.engine.max_steps or config.engine.max_metric_calls)

    for step in range(max_steps):
        if opt_state.metric_calls >= config.engine.max_metric_calls:
            break
        opt_state.step = step + 1
        source = best_candidate if config.engine.candidate_selection_strategy == "best" else current_candidate
        source_score = best_score if source == best_candidate else current_score
        source_records = candidate_records.get(_stable_json(source), current_records)
        proposals = _call_proposer(
            proposer,
            candidate=source,
            feedback=_feedback(source, objective, source_score, source_records),
            objective=objective,
            side_info=[r.side_info for r in source_records],
            opt_state=opt_state,
        )
        if not proposals:
            break
        for proposal in proposals:
            if opt_state.metric_calls >= config.engine.max_metric_calls:
                break
            current_candidate = proposal
            current_score, current_records = evaluate(proposal)

    validation_records: List[EvaluationRecord] = []
    validation_score = None
    if validation_examples and best_candidate is not None:
        opt_state.step += 1
        validation_records = _evaluate_candidate(wrapper, best_candidate, validation_examples, opt_state, config.engine.max_metric_calls)
        validation_score = _aggregate(validation_records)

    return GEPAResult(
        best_candidate=best_candidate,
        best_score=best_score,
        candidate_scores=candidate_scores,
        history=history,
        config=config,
        total_metric_calls=opt_state.metric_calls,
        validation_score=validation_score,
        validation_records=validation_records,
        metadata={"objective": objective},
    )
