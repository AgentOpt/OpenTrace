"""Multi-objective configuration and selection utilities.

Provides ObjectiveConfig and pure functions for multi-objective candidate
selection: weighted scalarization, Pareto ranking, and backward-compatible
scalar max.

All functions are pure (no side effects) and depend only on numpy, typing,
and dataclasses. No imports from opto.trainer to avoid circular dependencies.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# --- Type aliases ---
ScalarScore = float
VectorScore = Dict[str, float]
ScoreLike = Union[int, float, bool, Dict[str, float]]


@dataclass(frozen=True)
class ObjectiveConfig:
    """Immutable configuration for multi-objective candidate selection.

    Attributes:
        mode: Selection strategy.
            - "scalar": existing scalar comparison (default, backward-compatible).
            - "weighted": scalarize via weighted sum, then select max.
            - "pareto": Pareto dominance ranking with configurable tie-break.
        weights: Per-metric weights for weighted scalarization.
            Missing metrics use missing_value. Metrics not in weights are ignored.
            Empty dict in weighted mode -> equal weight 1.0 for all metrics.
        minimize: Frozenset of metric names where lower is better.
            These are negated internally ("higher-is-better" normalization).
            Users can pass a set; it is auto-converted to frozenset.
        missing_value: Score assigned to missing metrics (default: -inf).
        pareto_metrics: Subset of metrics for Pareto dominance.
            None -> use all metrics present across candidates.
        tie_break: Strategy for Pareto-equivalent candidates.
            - "weighted": fall back to weighted scalarization.
            - "lexicographic": sort by metric names alphabetically.
            - "random_seeded": seeded random shuffle.
        seed: Random seed for deterministic tie-breaking.
    """
    mode: str = "scalar"
    weights: Dict[str, float] = field(default_factory=dict)
    minimize: frozenset = field(default_factory=frozenset)
    missing_value: float = float("-inf")
    pareto_metrics: Optional[Tuple[str, ...]] = None
    tie_break: str = "weighted"
    seed: int = 0

    def __post_init__(self):
        if isinstance(self.minimize, set):
            object.__setattr__(self, 'minimize', frozenset(self.minimize))
        if self.mode not in ("scalar", "weighted", "pareto"):
            raise ValueError(
                f"mode must be 'scalar', 'weighted', or 'pareto', got '{self.mode}'"
            )
        if self.tie_break not in ("weighted", "lexicographic", "random_seeded"):
            raise ValueError(
                f"tie_break must be 'weighted', 'lexicographic', or "
                f"'random_seeded', got '{self.tie_break}'"
            )
        for k, v in self.weights.items():
            if v < 0:
                raise ValueError(f"Weight for '{k}' must be non-negative, got {v}")
        if self.pareto_metrics is not None and len(self.pareto_metrics) == 0:
            raise ValueError(
                "pareto_metrics must be None (auto) or non-empty tuple"
            )


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

def normalize_score(score: ScoreLike) -> Dict[str, float]:
    """Convert any score to dict form.

    - bool/int/float -> {"score": float(value)}
    - Dict[str, float] -> returned as-is (validated: all values finite)

    Raises:
        TypeError: if score is not int, float, bool, or dict.
        ValueError: if dict is empty or contains non-finite values.
    """
    if isinstance(score, bool):
        return {"score": float(score)}
    if isinstance(score, (int, float)):
        val = float(score)
        if not np.isfinite(val):
            raise ValueError(f"Score must be finite, got {score}")
        return {"score": val}
    if isinstance(score, dict):
        if len(score) == 0:
            raise ValueError("Score dict must not be empty")
        for k, v in score.items():
            if not isinstance(v, (int, float)) or not np.isfinite(float(v)):
                raise ValueError(
                    f"Score dict value for '{k}' must be finite float, got {v}"
                )
        return {k: float(v) for k, v in score.items()}
    raise TypeError(
        f"Score must be int, float, bool, or Dict[str, float], "
        f"got {type(score).__name__}"
    )


def apply_minimize(score_dict: Dict[str, float],
                   minimize: frozenset) -> Dict[str, float]:
    """Negate values for minimize metrics (higher-is-better normalization).

    Returns a new dict; metrics not in *minimize* are unchanged.
    """
    return {k: -v if k in minimize else v for k, v in score_dict.items()}


def weighted_scalarize(score_dict: Dict[str, float],
                       weights: Dict[str, float],
                       missing_value: float = float("-inf")) -> float:
    """Compute weighted sum of score dict.

    If *weights* is empty, all present metrics get equal weight 1.0.
    Metrics in *score_dict* but NOT in *weights* are ignored.
    """
    if not weights:
        weights = {k: 1.0 for k in score_dict}
    total = 0.0
    for metric, weight in weights.items():
        value = score_dict.get(metric, missing_value)
        total += weight * value
    return total


def dominates(a: Dict[str, float], b: Dict[str, float],
              metrics: Optional[Tuple[str, ...]] = None) -> bool:
    """Check if candidate *a* Pareto-dominates candidate *b*.

    a dominates b iff:
      - a[m] >= b[m] for ALL metrics m, AND
      - a[m] >  b[m] for AT LEAST ONE metric m

    Both dicts must be in "higher-is-better" form (post apply_minimize).
    """
    if metrics is None:
        metrics = tuple(sorted(set(a.keys()) | set(b.keys())))
    at_least_one_better = False
    for m in metrics:
        va = a.get(m, float("-inf"))
        vb = b.get(m, float("-inf"))
        if va < vb:
            return False
        if va > vb:
            at_least_one_better = True
    return at_least_one_better


def pareto_rank(candidates: List[Dict[str, float]],
                metrics: Optional[Tuple[str, ...]] = None) -> List[int]:
    """Assign Pareto rank to each candidate (0 = non-dominated front).

    Uses standard non-dominated sorting.
    """
    n = len(candidates)
    ranks = [0] * n
    remaining = set(range(n))
    current_rank = 0

    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and dominates(candidates[j], candidates[i], metrics):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            ranks[i] = current_rank
            remaining.remove(i)
        current_rank += 1

    return ranks


def select_best(candidates: List[Tuple[ScoreLike, Any]],
                config: Optional[ObjectiveConfig] = None) -> int:
    """Select index of the single best candidate.

    Args:
        candidates: List of (score, payload) tuples.
        config: Selection config. None -> scalar max (backward-compatible).

    Returns:
        Index of the best candidate.
    """
    if config is None or config.mode == "scalar":
        scores = []
        for score, _ in candidates:
            if isinstance(score, dict):
                scores.append(np.mean(list(score.values())))
            else:
                scores.append(float(score))
        return int(np.argmax(scores))

    score_dicts = [normalize_score(s) for s, _ in candidates]
    score_dicts = [apply_minimize(sd, config.minimize) for sd in score_dicts]

    if config.mode == "weighted":
        weighted = [
            weighted_scalarize(sd, config.weights, config.missing_value)
            for sd in score_dicts
        ]
        return int(np.argmax(weighted))

    if config.mode == "pareto":
        ranks = pareto_rank(score_dicts, config.pareto_metrics)
        front_indices = [i for i, r in enumerate(ranks) if r == 0]

        if len(front_indices) == 1:
            return front_indices[0]

        # Tie-break among front
        if config.tie_break == "weighted":
            front_scores = [
                weighted_scalarize(
                    score_dicts[i], config.weights, config.missing_value
                )
                for i in front_indices
            ]
            return front_indices[int(np.argmax(front_scores))]

        if config.tie_break == "lexicographic":
            metrics = sorted(score_dicts[front_indices[0]].keys())

            def lex_key(idx):
                return tuple(
                    score_dicts[idx].get(m, config.missing_value) for m in metrics
                )

            return max(front_indices, key=lex_key)

        if config.tie_break == "random_seeded":
            rng = np.random.RandomState(config.seed)
            return front_indices[rng.randint(len(front_indices))]

    raise ValueError(f"Unknown mode: {config.mode}")


def select_top_k(candidates: List[Tuple[ScoreLike, Any]],
                 config: Optional[ObjectiveConfig] = None,
                 k: int = 1) -> List[int]:
    """Select the top-k candidate indices.

    Same logic as select_best but returns *k* indices.
    For Pareto mode: rank-0 front first (up to k), then rank-1, etc.
    """
    if config is None or config.mode == "scalar":
        scores = []
        for score, _ in candidates:
            if isinstance(score, dict):
                scores.append(np.mean(list(score.values())))
            else:
                scores.append(float(score))
        return list(np.argsort(scores)[::-1][:k])

    score_dicts = [normalize_score(s) for s, _ in candidates]
    score_dicts = [apply_minimize(sd, config.minimize) for sd in score_dicts]

    if config.mode == "weighted":
        weighted = [
            weighted_scalarize(sd, config.weights, config.missing_value)
            for sd in score_dicts
        ]
        return list(np.argsort(weighted)[::-1][:k])

    if config.mode == "pareto":
        ranks = pareto_rank(score_dicts, config.pareto_metrics)
        result: List[int] = []
        max_rank = max(ranks)
        for rank in range(max_rank + 1):
            rank_indices = [i for i, r in enumerate(ranks) if r == rank]
            if config.tie_break == "weighted":
                rank_indices.sort(
                    key=lambda i: weighted_scalarize(
                        score_dicts[i], config.weights, config.missing_value
                    ),
                    reverse=True,
                )
            elif config.tie_break == "lexicographic":
                metrics = (
                    sorted(score_dicts[rank_indices[0]].keys())
                    if rank_indices else []
                )
                rank_indices.sort(
                    key=lambda i: tuple(
                        score_dicts[i].get(m, config.missing_value)
                        for m in metrics
                    ),
                    reverse=True,
                )
            elif config.tie_break == "random_seeded":
                rng = np.random.RandomState(config.seed + rank)
                rng.shuffle(rank_indices)
            result.extend(rank_indices)
            if len(result) >= k:
                break
        return result[:k]

    raise ValueError(f"Unknown mode: {config.mode}")
