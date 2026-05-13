from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type, Union

from opto.trace import bundle, node


def resolve_optimizer_cls(optimizer_cls: Optional[Union[str, Type[Any]]] = None) -> Type[Any]:
    if optimizer_cls is None:
        try:
            from opto.optimizers import OptoPrimeV2

            return OptoPrimeV2
        except Exception:
            from opto.optimizers import OptoPrime

            return OptoPrime
    if isinstance(optimizer_cls, str):
        import opto.optimizers as optimizers

        try:
            return getattr(optimizers, optimizer_cls)
        except AttributeError as exc:
            available = sorted(name for name in getattr(optimizers, "__all__", dir(optimizers)) if not name.startswith("_"))
            raise ValueError(f"Unknown Trace optimizer '{optimizer_cls}'. Available optimizers include: {available}") from exc
    if not isinstance(optimizer_cls, type):
        raise ValueError(f"optimizer_cls must be a class or string name, got {type(optimizer_cls).__name__}")
    return optimizer_cls


def _jsonable(candidate: Any) -> bool:
    try:
        json.dumps(candidate)
        return True
    except TypeError:
        return False


def default_candidate_serializer(candidate: Any) -> Any:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict) and len(candidate) == 1:
        value = next(iter(candidate.values()))
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
    if _jsonable(candidate):
        return json.dumps(candidate, sort_keys=True)
    return repr(candidate)


def default_candidate_deserializer(original: Any, proposed: Any) -> Any:
    if isinstance(original, str):
        return str(proposed)
    if isinstance(proposed, str):
        try:
            decoded = json.loads(proposed)
        except Exception:
            decoded = None
        if isinstance(original, dict) and isinstance(decoded, dict):
            return decoded
        if isinstance(original, list) and isinstance(decoded, list):
            return decoded
    if isinstance(original, dict) and len(original) == 1 and not isinstance(proposed, dict):
        key = next(iter(original))
        return {key: proposed}
    return copy.deepcopy(proposed)


@bundle(
    description="[optimize_anything_candidate] Identity wrapper used to expose an optimize_anything candidate to Trace.",
    trainable=False,
)
def _identity_candidate(candidate):
    return candidate


@dataclass
class TraceOptimizerBackend:
    """Adapt Trace optimizers to the optimize_anything candidate-proposer protocol."""

    optimizer_cls: Optional[Union[str, Type[Any]]] = None
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    parameter_name: str = "candidate"
    candidate_serializer: Callable[[Any], Any] = default_candidate_serializer
    candidate_deserializer: Optional[Callable[[Any, Any], Any]] = None

    def __post_init__(self) -> None:
        self.optimizer_cls = resolve_optimizer_cls(self.optimizer_cls)

    def __call__(
        self,
        *,
        candidate: Any,
        feedback: str,
        objective: Optional[str] = None,
        side_info: Optional[Any] = None,
        opt_state: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        del side_info, opt_state, kwargs
        original = copy.deepcopy(candidate)
        parameter = node(
            self.candidate_serializer(original),
            name=self.parameter_name,
            trainable=True,
            description="Candidate optimized through the optimize_anything compatibility layer.",
        )
        output = _identity_candidate(parameter)
        optimizer_kwargs = dict(self.optimizer_kwargs)
        optimizer = self._make_optimizer(parameter, optimizer_kwargs, objective)
        optimizer.zero_feedback()
        optimizer.backward(output, feedback)
        updates = self._propose_without_mutating(optimizer)
        if not updates or parameter not in updates:
            return original
        proposed = updates[parameter]
        if self.candidate_deserializer is not None:
            return self.candidate_deserializer(original, proposed)
        return default_candidate_deserializer(original, proposed)

    def _make_optimizer(self, parameter: Any, optimizer_kwargs: Dict[str, Any], objective: Optional[str]) -> Any:
        if objective is not None and "objective" not in optimizer_kwargs:
            try:
                return self.optimizer_cls([parameter], objective=objective, **optimizer_kwargs)
            except TypeError as exc:
                if "objective" not in str(exc):
                    raise
        return self.optimizer_cls([parameter], **optimizer_kwargs)

    @staticmethod
    def _propose_without_mutating(optimizer: Any) -> Dict[Any, Any]:
        if hasattr(optimizer, "step"):
            try:
                return optimizer.step(bypassing=True)
            except TypeError:
                return optimizer.step()
        if hasattr(optimizer, "propose"):
            return optimizer.propose()
        raise TypeError(f"{optimizer.__class__.__name__} does not implement step() or propose()")
