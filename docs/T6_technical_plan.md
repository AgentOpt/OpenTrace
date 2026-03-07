# T6 Technical Plan — Multi-Objective Vector Scores for Trainer Selection

**Version:** 1.0 (Refined)
**Author:** Carlos Rodriguez
**Date:** February 9, 2026
**Status:** M0 Deliverable — Analysis + Architecture + Interface Spec

**Target repos / branches:**
- **Primary (implementation + PR):** [`AgentOpt/OpenTrace@experimental`](https://github.com/AgentOpt/OpenTrace/tree/experimental)
- **Benchmark integration (M3):** [`AgentOpt/Trace-Bench`](https://github.com/AgentOpt/Trace-Bench)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Goals, Non-Goals, Success Criteria](#2-goals-non-goals-success-criteria)
3. [Current Code Reality (Baseline)](#3-current-code-reality-baseline)
4. [Proposed Architecture (Minimal Delta)](#4-proposed-architecture-minimal-delta)
5. [Public API & Data Contracts](#5-public-api--data-contracts)
6. [Module Modifications (Files to Create / Modify)](#6-module-modifications)
7. [Edge Cases & Defensive Design](#7-edge-cases--defensive-design)
8. [Milestones & Validation Gates](#8-milestones--validation-gates)
9. [Test Plan](#9-test-plan)
10. [Risks & Mitigation](#10-risks--mitigation)
11. [Design Decisions (Resolved)](#11-design-decisions-resolved)
12. [Appendix: Code Touchpoints](#12-appendix-code-touchpoints)

---

## 1. Executive Summary

Today, trainer selection in Trace is driven by a **single scalar score**. Guides return `Tuple[float, str]` via `get_feedback()`, evaluators produce `np.array` of floats, and trainers (`BasicSearchAlgorithm`, `BeamsearchAlgorithm`) select candidates via scalar comparison (`max(candidates, key=lambda x: x[0])` and `sorted(..., key=lambda x: x[0])` respectively). This blocks trainer-side search from exploiting multiple metrics like `{accuracy, latency_ms, cost}`.

**Motivation note (from team discussion):**
Putting multiple metrics into the *feedback dict/text* is useful for optimizers (OptoPrime/OPRO), but trainers (BasicSearch/UCB/PrioritySearch/GEPA) typically only inspect the **scalar score** for ranking/UCB and ignore additional feedback structure. Therefore, enabling **vector score / score-as-dict** (with backward-compatible scalar reduction) is required for multi-objective trainer selection.

### What this plan adds

| Component | Change |
|-----------|--------|
| **Score contract** | `Dict[str, float]` returned by guides (optional), with backward-compatible scalar fallback |
| **ObjectiveConfig** | Frozen dataclass defining selection mode: `scalar` (default), `weighted`, or `pareto` |
| **objectives.py** (new) | All multi-objective logic isolated in pure, testable functions |
| **Evaluators** | Vector-score aggregation helpers (`evaluate_vector`, `aggregate_vector_scores`) |
| **BasicSearchAlgorithm** | Selection via `select_best(candidates, objective_config)` |
| **BeamsearchAlgorithm** | Selection via `select_top_k(candidates, objective_config, k)` |
| **PrioritySearch** (optional) | Scalarize heap priority via ObjectiveConfig; store dict for logging |
| **Benchmarks** (M3) | 3 simple benchmarks integrated into Trace-Bench |

### Guiding principles

- **Backward compatibility is non-negotiable.** `mode="scalar"` (the default) preserves identical behavior.
- **Isolate complexity.** All multi-objective logic lives in `objectives.py` — pure functions, easy to test.
- **Minimal churn.** Trainers gain an optional `objective_config` parameter; existing call sites are untouched.
- **Determinism.** Fixed `seed` → deterministic selection, especially Pareto tie-breaks.

---

## 2. Goals, Non-Goals, Success Criteria

### 2.1 Goals

| ID | Goal | Acceptance Signal |
|----|------|-------------------|
| G1 | **Backward compatibility** | Existing scalar-score guides/trainers produce identical results when `objective_config` is `None` or `mode="scalar"` |
| G2 | **Vector score support** | Guide returns `{"accuracy": 1.0, "latency_ms": 120.0}` and trainers select candidates using weighted or Pareto mode |
| G3 | **Determinism** | Fixed `seed` → identical selection across runs (tested in CI) |
| G4 | **Actionability** | Every milestone: Colab notebook + pytest coverage (M1+) |
| G5 | **Benchmarks** | 3 benchmarks defined, integrated into Trace-Bench, runnable from notebooks |

### 2.2 Non-goals (explicit)

- No multi-objective UCB (MO-UCB) — too risky for v1 scope.
- No Pareto archive / non-dominated set management inside PrioritySearch.
- No changes to optimizer internals or new telemetry infrastructure.
- No modification to `get_feedback()` return signature (we use a helper instead).

### 2.3 Crisp success criteria

All of the following must be true:

1. Scalar-only trainers still work and produce same results by default.
2. Multi-objective guide dict works end-to-end for BasicSearch + Beamsearch.
3. Deterministic behavior with fixed seed (tests + notebook).
4. Each milestone delivers a runnable Colab notebook.
5. From M1 onward, new functions have pytest tests and CI is green.
6. M3: three benchmarks exist, run, and Trace-Bench integration works.

---

## 3. Current Code Reality (Baseline)

### 3.1 Guide — scalar score contract

```python
# opto/trainer/guide.py

class Guide:
    def get_feedback(self, query, response, reference=None, **kwargs) -> Tuple[float, str]:
        raise NotImplementedError

    def metric(self, query, response, reference=None, **kwargs) -> float:
        return self.get_feedback(query, response, reference)[0]  # extracts scalar
```

**Implication:** `metric()` always returns `float`. Multi-metric feedback is not usable for selection.

### 3.2 Evaluators — scalar arrays

```python
# opto/trainer/evaluators.py

def evaluate(agent, guide, inputs, infos, ...) -> np.ndarray:
    # Calls guide.metric() per example → float
    # Returns np.array of shape (N,) or (N, num_samples)
```

**Implication:** All scores are numeric scalars aggregated via `np.mean()`.

### 3.3 BasicSearchAlgorithm — scalar max selection

```python
# opto/trainer/algorithms/basic_algorithms.py :: BasicSearchAlgorithm.optimizer_step()

def validate():
    scores = evaluate(self.agent, self.validate_guide, ...)
    return np.mean(scores) if all([s is not None for s in scores]) else -np.inf

# Selection:
candidates.append((score, update_dict))        # score is float
best_score, best_update = max(candidates, key=lambda x: x[0])  # scalar max
```

**Insertion point:** Replace `max(candidates, ...)` with `select_best(candidates, objective_config)`.

### 3.4 BeamsearchAlgorithm — scalar sort selection

```python
# opto/trainer/algorithms/beamsearch_algorithm.py :: BeamsearchAlgorithm.select()

scored_candidates.append((validation_score, candidate_params))  # float
sorted_candidates = sorted(scored_candidates, key=lambda x: x[0], reverse=True)
selected_candidates = sorted_candidates[:beam_width]  # take top-k by scalar
```

**Insertion point:** Replace scalar sort with `select_top_k(scored_candidates, objective_config, k=beam_width)`.

### 3.5 Shared patterns across both trainers

| Pattern | BasicSearch | Beamsearch |
|---------|-------------|------------|
| Validate | `np.mean(scores)` → float | `np.mean(validation_scores)` → float |
| Store | `(score, update_dict)` | `(validation_score, candidate_params)` |
| Select | `max(candidates, key=λ x: x[0])` | `sorted(candidates, key=λ x: x[0])[:k]` |
| Fallback | `-np.inf` | `-np.inf` |

Both converge to the same abstraction: **given a list of `(score, params)` pairs, select the best or top-k.** This is exactly what `objectives.py` will provide.

### 3.6 Existing infrastructure we leverage

- **Logger abstraction:** `BaseLogger` with `log(name, value, step)` — can log each metric in a vector score.
- **StubLLM / DummyLLM:** Wraps deterministic callables — usable for CI and no-keys notebooks.
- **`batch_run` / `async_run`:** Parallelism utilities already in place.

---

## 4. Proposed Architecture (Minimal Delta)

### 4.1 Core idea

Isolate all multi-objective logic into one new module (`opto/trainer/objectives.py`) containing **pure functions**:

```
to_score_dict()     →  scalar/dict to dict conversion (neutral name)
apply_minimize()    →  flip signs for minimize metrics
weighted_scalarize()→  dict → float via weighted sum
pareto_rank()       →  dominance ranking + tie-break
select_best()       →  given candidates + config, return best index
select_top_k()      →  given candidates + config, return top-k indices
```

Trainers call these functions instead of inline `max()` / `sorted()`. When `objective_config` is `None`, the functions fall through to scalar comparison — **identical to current behavior**.

### 4.2 Data flow (target)

```
Guide.get_feedback()
    │
    ├── returns (float, str)          ← existing path, unchanged
    └── returns (Dict[str,float], str) ← new path (via get_score_dict helper)
            │
            ▼
Evaluator.evaluate_vector()
    │
    ├── per-example: List[Dict[str, float]]
    └── aggregated:  Dict[str, float]  (mean per metric)
            │
            ▼
Trainer selection (objectives.py)
    │
    ├── mode="scalar"   → max(mean_scores)           ← unchanged
    ├── mode="weighted"  → max(weighted_scalarize())  ← new
    └── mode="pareto"    → pareto_rank() + tie-break  ← new
```

### 4.3 Backward compatibility guarantee

The entire vector-score path is **opt-in**:

1. If `objective_config` is `None` → existing scalar path, no new code executed.
2. If guide returns `float` and `objective_config` is provided → `to_score_dict()` wraps it as `{"score": float}`, weights default to `{"score": 1.0}`.
3. If guide returns `Dict[str, float]` and `objective_config` is `None` → `ValueError` is raised (no hidden hard-coded dict→scalar reduction). Pass an explicit `ObjectiveConfig(mode="scalar", scalarize_dict="mean")` to reduce via mean, or `scalarize_dict="score"` to use a single key.

---

## 5. Public API & Data Contracts

### 5.1 Score types

```python
from typing import Union, Dict

ScalarScore = float
VectorScore = Dict[str, float]          # JSON-serializable, all values finite
ScoreLike   = Union[int, float, bool, Dict[str, float]]
```

**Contract:**
- "Higher is better" by default for all metrics.
- Metrics to minimize are declared in `ObjectiveConfig.minimize` (semantics: negate internally).
- All dict values must be finite floats. `NaN` / `±inf` in a dict raises `ValueError`.
- `int` and `bool` scalar scores are accepted and converted to `float` (e.g., `LLMJudge` returns `int` 0/1, test guides return `bool`).

### 5.2 ObjectiveConfig

```python
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Tuple

@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for multi-objective candidate selection.

    Attributes:
        mode: Selection strategy.
            - "scalar": Use existing scalar comparison (default, backward-compatible).
            - "weighted": Scalarize via weighted sum, then select max.
            - "pareto": Pareto dominance ranking with configurable tie-break.
        weights: Per-metric weights for weighted scalarization.
            Missing metrics use missing_value. Metrics not present in the weights dict
            are ignored (not included in the weighted sum).
            If empty dict in weighted mode, all present metrics get equal weight 1.0.
        minimize: Frozenset of metric names where lower is better (users can pass set; auto-converted).
            These are negated internally before comparison ("higher-is-better" normalization).
        missing_value: Score assigned to missing metrics in a candidate's score dict.
            Default: float('-inf') (effectively disqualifies candidates missing required metrics).
        pareto_metrics: Subset of metrics to use for Pareto dominance.
            If None, all metrics present across candidates are used.
        tie_break: Strategy for breaking ties among Pareto-equivalent candidates.
            - "weighted": Fall back to weighted scalarization among tied candidates.
            - "lexicographic": Sort by metrics in alphabetical order.
            - "random_seeded": Seeded random shuffle.
        seed: Random seed for deterministic tie-breaking.
    """
    mode: Literal["scalar", "weighted", "pareto"] = "scalar"
    weights: Dict[str, float] = field(default_factory=dict)
    minimize: frozenset = field(default_factory=frozenset)
    missing_value: float = float("-inf")
    pareto_metrics: Optional[Tuple[str, ...]] = None
    tie_break: Literal["weighted", "lexicographic", "random_seeded"] = "weighted"
    seed: int = 0

    def __post_init__(self):
        # Convert set → frozenset for true immutability + hashability
        if isinstance(self.minimize, set):
            object.__setattr__(self, 'minimize', frozenset(self.minimize))
        # Validate weights are non-negative
        for k, v in self.weights.items():
            if v < 0:
                raise ValueError(f"Weight for '{k}' must be non-negative, got {v}")
        # Validate pareto_metrics
        if self.pareto_metrics is not None and len(self.pareto_metrics) == 0:
            raise ValueError("pareto_metrics must be None (auto) or non-empty tuple")
```

**Validation rules (enforced in `__post_init__`):**
- `minimize` is stored as `frozenset` for true immutability (users can pass `set` for convenience; it's auto-converted).
- `mode="weighted"` with empty `weights` → auto-assign equal weight 1.0 to all encountered metrics.
- `mode="pareto"` with `pareto_metrics=None` → use union of all metric keys across candidates.
- `mode="pareto"` with `pareto_metrics=()` → `ValueError`.
- All weight values must be non-negative.
- `minimize` metric names must be valid strings (warning if not found in any candidate).

### 5.3 Guide helper method

```python
# Added to Guide base class (non-breaking)

class Guide:
    # ... existing methods unchanged ...

    def get_score_dict(self, query: str, response: str, reference=None, **kwargs) -> Dict[str, float]:
        """Return evaluation score as a dict (multi-objective selection path).

        Default implementation wraps the scalar training score from get_feedback() as:
            {"score": float_value}

        Guides that need multiple metrics should override *get_score_dict()* and return
        e.g. {"accuracy": 0.9, "brevity": 0.8, "latency_s": 0.05}.

        Note: get_feedback() should remain scalar (float) for training-loop backward
        compatibility. If a subclass returns a dict from get_feedback(), metric() and
        scalar evaluators may break; prefer overriding get_score_dict().
        """
        score, _ = self.get_feedback(query, response, reference, **kwargs)
        if isinstance(score, dict):
            return {k: float(v) for k, v in score.items()}
        return {"score": float(score)}
```

**Why this approach:**
- `get_score_dict()` is a new method — zero risk of breaking existing subclasses.
- `metric()` always returns `float` — the existing `evaluate()` function (which calls `guide.metric()` and passes results to `np.array()`) and the training loop (which calls `np.mean(scores)`) are completely unaffected.
- Dict scores are only accessible via `get_score_dict()` → `evaluate_vector()`, keeping the two data paths cleanly separated.

### 5.4 Evaluator additions

```python
# Added to opto/trainer/evaluators.py

def evaluate_vector(agent, guide, inputs, infos, min_score=None,
                    num_samples=1, num_threads=None, description=None
                    ) -> list:
    """Like evaluate(), but returns List[ScoreLike] (float or dict per example).

    Uses guide.get_score_dict() to obtain dict scores per example.
    When guide returns scalar, get_score_dict() wraps it as {"score": float}.

    When num_samples > 1: for each example, collects num_samples score dicts,
    computes per-key mean across the samples, and returns one aggregated dict
    per example. Final output is always List[Dict[str, float]] of length N.
    """
    ...

def aggregate_vector_scores(scores: list) -> Union[float, Dict[str, float]]:
    """Aggregate per-example scores into a single summary score.

    - If all scores are float: returns np.mean (existing behavior).
    - If all scores are dict: returns per-metric mean dict.
    - Mixed float/dict: normalizes all to dict via to_score_dict(), then averages.

    Args:
        scores: List of float or Dict[str, float] values.

    Returns:
        float (if all scalar) or Dict[str, float] (if any dicts present).
    """
    ...
```

### 5.5 objectives.py — complete function signatures

```python
# opto/trainer/objectives.py (NEW FILE)

from typing import Union, Dict, List, Set, Optional, Tuple, Literal
from dataclasses import dataclass, field

# --- ObjectiveConfig defined here (see §5.2) ---

# --- Score type aliases ---
ScalarScore = float
VectorScore = Dict[str, float]
ScoreLike = Union[float, Dict[str, float]]

# --- Pure utility functions ---

def to_score_dict(score: ScoreLike) -> Dict[str, float]:
    """Convert any score to dict form (neutral name).

    - int/float/bool → {"score": float(value)}
    - Dict[str, float] → returned as-is (validated: all values finite)

    Handles int (LLMJudge returns 0/1) and bool (test guides) via isinstance(score, (int, float, bool)).
    Backward-compatible alias: `normalize_score = to_score_dict`

    Raises:
        TypeError: if score is not int, float, bool, or dict
        ValueError: if dict contains non-finite values or is empty
    """
    ...

def apply_minimize(score_dict: Dict[str, float],
                   minimize: Set[str]) -> Dict[str, float]:
    """Negate values for minimize metrics (higher-is-better normalization).

    Returns a new dict with minimize metrics negated.
    Metrics not in minimize set are unchanged.
    """
    ...

def weighted_scalarize(score_dict: Dict[str, float],
                       weights: Dict[str, float],
                       missing_value: float = float("-inf")) -> float:
    """Compute weighted sum of score dict.

    For each metric in weights:
      - If present in score_dict: weight * value
      - If missing: weight * missing_value

    Metrics in score_dict but NOT in weights are ignored.
    If weights is empty, all metrics get equal weight 1.0.

    Returns:
        Weighted scalar score.
    """
    ...

def dominates(a: Dict[str, float], b: Dict[str, float],
              metrics: Optional[Tuple[str, ...]] = None) -> bool:
    """Check if candidate 'a' Pareto-dominates candidate 'b'.

    a dominates b iff:
      - a[m] >= b[m] for all metrics m, AND
      - a[m] > b[m] for at least one metric m

    Both dicts must already be in "higher-is-better" form (post apply_minimize).
    Missing metrics are treated as missing_value (caller should handle before call).

    Args:
        a, b: Score dicts (higher-is-better normalized).
        metrics: Subset of metrics to compare. If None, use union of keys.
    """
    ...

def pareto_rank(candidates: List[Dict[str, float]],
                metrics: Optional[Tuple[str, ...]] = None) -> List[int]:
    """Assign Pareto rank to each candidate (0 = non-dominated front).

    Uses standard non-dominated sorting.

    Args:
        candidates: List of score dicts (higher-is-better normalized).
        metrics: Subset of metrics for dominance. If None, use all present.

    Returns:
        List of integer ranks (same length as candidates). Rank 0 = Pareto front.
    """
    ...

def select_best(candidates: List[Tuple[ScoreLike, any]],
                objective_config: Optional['ObjectiveConfig'] = None) -> int:
    """Select the single best candidate index.

    Args:
        candidates: List of (score, payload) tuples.
        objective_config: Selection config. If None, uses scalar max (backward-compatible).

    Returns:
        Index of best candidate.

    Behavior by mode:
        - scalar/None: max(score) where score is float (or mean of dict values).
        - weighted: max(weighted_scalarize(normalize(score), config.weights)).
        - pareto: rank candidates, tie-break among rank-0 front, return winner.

    Call-site transformation (BasicSearch):
        # Current:
        best_score, best_update = max(candidates, key=lambda x: x[0])
        # Target:
        best_idx = select_best(candidates, objective_config)
        best_score, best_update = candidates[best_idx]
    """
    ...

def select_top_k(candidates: List[Tuple[ScoreLike, any]],
                 objective_config: Optional['ObjectiveConfig'] = None,
                 k: int = 1) -> List[int]:
    """Select the top-k candidate indices.

    Same logic as select_best, but returns k indices.

    For pareto mode: returns rank-0 front (up to k). If front < k,
    includes rank-1 candidates by tie-break order, etc.

    Deterministic ordering guaranteed with fixed seed.
    """
    ...
```

---

## 6. Module Modifications

### 6.1 Files to CREATE

| File | Contents | Milestone |
|------|----------|-----------|
| `opto/trainer/objectives.py` | `ObjectiveConfig`, `to_score_dict`, `apply_minimize`, `weighted_scalarize`, `dominates`, `pareto_rank`, `select_best`, `select_top_k`, `score_dict_to_scalar`, `to_scalar_score`, `aggregate_score_dicts` | M1 |
| `tests/test_objectives.py` | Unit tests for all functions in objectives.py | M1 |
| `tests/test_evaluators_vector.py` | Tests for evaluate_vector + aggregate_vector_scores | M1 |
| `tests/test_trainers_multiobjective.py` | Integration tests for BasicSearch + Beamsearch with ObjectiveConfig | M2 |
| `examples/notebooks/t6_m0_analysis.ipynb` | M0 analysis notebook | M0 |
| `examples/notebooks/t6_m1_vector_scores.ipynb` | M1 demo notebook | M1 |
| `examples/notebooks/t6_m2_trainers.ipynb` | M2 demo notebook | M2 |
| `examples/notebooks/t6_m3_benchmarks.ipynb` | M3 benchmark notebook | M3 |
| `docs/T6_technical_plan.md` | This document | M0 |
| `docs/multi_objective_scores.md` | User-facing documentation | M4 |

### 6.2 Files to MODIFY

| File | Change | Milestone |
|------|--------|-----------|
| `opto/trainer/guide.py` | Add `get_score_dict()` method to `Guide` base class. Keep training loop scalar-safe (`metric()` returns `float`). Dict/vector scores are accessed via `get_score_dict()` for trainer-side selection. | M1 |
| `opto/trainer/evaluators.py` | Add `evaluate_vector()` and `aggregate_vector_scores()`. Existing `evaluate()` unchanged. | M1 |
| `opto/trainer/algorithms/basic_algorithms.py` | Add `objective_config` param to `BasicSearchAlgorithm.train()`. Replace `max(candidates, ...)` with `select_best()` in `optimizer_step()`. | M1 (minimal) / M2 (robust) |
| `opto/trainer/algorithms/beamsearch_algorithm.py` | Add `objective_config` param to `BeamsearchAlgorithm.train()`. Replace scalar sort in `select()` with `select_top_k()`. | M2 |
| `opto/features/priority_search/priority_search.py` | (Optional) Add `objective_config` param. Scalarize heap key via weighted mode. Store dict for logging. Pareto falls back to weighted. | M2 |

### 6.3 Files NOT modified

- `opto/trace/` — no changes to trace primitives.
- `opto/optimizers/` — optimizers are upstream of selection; they produce candidates, not rank them.
- Existing tests — no modifications; they validate backward compatibility by continuing to pass.

---

## 7. Edge Cases & Defensive Design

### 7.1 Score validation

| Case | Behavior |
|------|----------|
| `score = 0.85` (float) | `to_score_dict()` → `{"score": 0.85}` |
| `score = 1` (int) | `to_score_dict()` → `{"score": 1.0}` (LLMJudge returns int 0/1) |
| `score = True` (bool) | `to_score_dict()` → `{"score": 1.0}` (test guides return bool) |
| `score = {"accuracy": 0.9, "latency_ms": 120.0}` | Returned as-is after validation |
| `score = {}` (empty dict) | `ValueError("Score dict must not be empty")` |
| `score = {"accuracy": float('nan')}` | `ValueError("Score dict contains non-finite value")` |
| `score = {"accuracy": float('inf')}` | `ValueError("Score dict contains non-finite value")` |
| `score = "text"` (wrong type) | `TypeError("Score must be int, float, bool, or Dict[str, float]")` |

### 7.2 Missing metrics across candidates

| Case | Behavior |
|------|----------|
| Candidate A has `{accuracy, latency}`, B has `{accuracy}` | B gets `latency = missing_value` (default `-inf`) |
| `weights = {"accuracy": 0.7, "latency": 0.3}`, candidate missing `latency` | Weighted sum uses `0.3 * missing_value` |
| All candidates missing a weighted metric | Warning logged; metric still contributes `weight * missing_value` |

### 7.3 Mixed scalar/dict batches

| Case | Behavior |
|------|----------|
| All scores are `float` (or `int`/`bool`) | `aggregate_vector_scores()` returns `float` via `np.mean()` (existing behavior) |
| All scores are `dict` with same keys | `aggregate_vector_scores()` returns per-metric mean `Dict[str, float]` |
| Mixed `float` and `dict` in same batch | `ValueError("All scores in a batch must be the same type (all float or all dict)")` |

A mixed batch most likely indicates a bug in the guide implementation (e.g., returning `float` on some inputs and `dict` on others). Failing loudly prevents silent incorrect aggregation.

### 7.4 Single-metric dict

| Case | Behavior |
|------|----------|
| Guide returns `{"accuracy": 0.9}` with `mode="weighted"` | Weighted sum = `weight * 0.9` (trivially correct) |
| Guide returns `{"accuracy": 0.9}` with `mode="pareto"` | Pareto degenerates to scalar max (single dimension — no tradeoffs). Warning logged. |

### 7.5 Tie-breaking

| Case | Behavior |
|------|----------|
| Two candidates with identical weighted score | Deterministic: lower original index wins (stable sort) |
| Pareto front with 3 equivalent candidates, `tie_break="weighted"` | Fall back to weighted scalarization among the 3; select max |
| Pareto front with 3 equivalent candidates, `tie_break="lexicographic"` | Sort by metric names alphabetically, compare values in order |
| Pareto front with 3 equivalent candidates, `tie_break="random_seeded"` | Seeded shuffle with `config.seed`; same seed → same order always |

### 7.7 Training loop safety

The training loop has a **separate data path** from evaluation/selection. In `standard_optimization_step()` (basic_algorithms.py:46) and `standard_forward()` (sampler.py:130):

```python
score, feedback = guide(x, target.data, info)
```

This `score` flows into `MinibatchAlgorithm.update()` where `np.mean(scores)` is computed (basic_algorithms.py:511). **This path must always receive `float`.**

| Constraint | Enforcement |
|-----------|-------------|
| `guide.__call__()` / `get_feedback()` return type is **NOT widened** | No changes to `get_feedback()` signature; it still returns `Tuple[float, str]` |
| Training loop always receives scalar `score` | `metric()` always returns `float`. Vector/dict scores are not used by the training loop and are accessed via `get_score_dict()` for trainer-side selection. |
| Dict scores flow through a separate path | `get_score_dict()` → `evaluate_vector()` → `select_best()` / `select_top_k()` |
| A multi-objective guide must return `(float, str)` from `get_feedback()` for the training loop | The float is a collapsed scalar summary; the full dict is extracted via `get_score_dict()` during selection |

**Two data paths (by design):**
```
Training loop:    guide() → score (float) → np.mean(scores)       ← UNCHANGED
Selection path:   get_score_dict() → evaluate_vector() → objectives.py  ← NEW
```

### 7.6 ObjectiveConfig validation

| Case | Behavior |
|------|----------|
| `mode="weighted"`, `weights={}` | Auto-assign equal weight 1.0 to all metrics encountered at selection time |
| `mode="pareto"`, `pareto_metrics=()` (empty tuple) | `ValueError("pareto_metrics must be None (auto) or non-empty tuple")` |
| `weights={"accuracy": -0.5}` (negative weight) | `ValueError("All weights must be non-negative")` |
| `minimize={"unknown_metric"}` | Warning logged at selection time if metric never appears; no error (tolerant) |

---

## 8. Milestones & Validation Gates

### Milestone 0 — Analysis + technical plan + interface spec

**Deliverables:**
- `docs/T6_technical_plan.md` — this document, finalized
- `examples/notebooks/t6_m0_analysis.ipynb` — Colab-ready notebook

**Notebook demonstrates:**
- Current Guide score contract (`get_feedback` → `Tuple[float, str]`, `metric` → `float`)
- Where scalar selection happens in BasicSearch (`max(candidates, ...)`) and Beamsearch (`sorted(...)[:k]`)
- Planned behavior prototype: deterministic toy guide returning dict metrics, showing weighted vs Pareto selection on dummy candidates

**SMART validation:**
- Plan includes final API signatures and precise file list (create/modify) ✓
- Notebook runs without API keys ✓
- Notebook prints: current score contract, selection touchpoints, planned selection outputs ✓

---

### Milestone 1 — ObjectiveConfig + utilities + evaluator support + BasicSearch minimal

**Deliverables:**
- `opto/trainer/objectives.py` (new)
- `opto/trainer/guide.py` (add `get_score_dict`)
- `opto/trainer/evaluators.py` (add `evaluate_vector`, `aggregate_vector_scores`)
- `opto/trainer/algorithms/basic_algorithms.py` (BasicSearch: accept/use ObjectiveConfig)
- `tests/test_objectives.py`, `tests/test_evaluators_vector.py`
- `examples/notebooks/t6_m1_vector_scores.ipynb`

**Notebook demonstrates:**
- StubLLM mode: BasicSearchAlgorithm on small candidate set (5-10) with deterministic dummy guide returning dict metrics
- Shows: (a) scalar baseline, (b) weighted mode, (c) Pareto mode, (d) deterministic tie-break under fixed seed
- Real LLM mode (required): tiny dataset (≤5 items) producing ≥2 metrics

**SMART validation:**
- `pytest -q` passes (all new functions covered)
- Notebook runs in Colab: weighted selection result changes when weights change
- Pareto returns tradeoffs and is deterministic under fixed seed
- Scalar path produces identical results to pre-change behavior

---

### Milestone 2 — Trainer upgrades (Beamsearch + robust BasicSearch)

**Deliverables:**
- `opto/trainer/algorithms/beamsearch_algorithm.py` (accept ObjectiveConfig, vector selection)
- Expanded BasicSearch tests (edge cases, missing metrics, tie-break policies)
- Optional: minimal PrioritySearch support (weighted scalarization for heap, dict stored for logging)
- `tests/test_trainers_multiobjective.py`
- `examples/notebooks/t6_m2_trainers.ipynb`

**Notebook demonstrates:**
- BasicSearch + Beamsearch in: scalar mode (baseline), weighted mode, Pareto mode
- StubLLM + real LLM sections

**SMART validation:**
- `pytest -q` green
- Integration test confirms: weighted vs Pareto select different candidates where expected
- Scalar-only example produces same final best score when `objective_config=None`
- Deterministic tie-break is stable across runs

---

### Milestone 3 — Benchmarks (Trace-Bench integration)

**Deliverables:**
- PR to Trace-Bench: benchmark configs/tasks + notebook
  - **Trace-Bench touchpoints (update `main` if default branch differs):**
    - https://github.com/AgentOpt/Trace-Bench/blob/main/LLM4AD/trainers_benchmark.py
    - https://github.com/AgentOpt/Trace-Bench/blob/main/LLM4AD/trainers_benchmark_tasks_validation.py
    - https://github.com/AgentOpt/Trace-Bench/blob/main/LLM4AD/benchmark_tasks/index.json
    - https://github.com/AgentOpt/Trace-Bench/tree/main/LLM4AD/benchmark_tasks
    - https://github.com/AgentOpt/Trace-Bench/blob/main/LLM4AD/llm4ad_loader.py
    - https://github.com/AgentOpt/Trace-Bench/blob/main/tests/test_lite_optimize_llm4ad.py
- 3 benchmarks:
  1. **Accuracy vs latency** (toy QA dataset)
  2. **Accuracy vs response length** (penalize verbosity)
  3. **Accuracy vs tool calls** (penalize excessive tool usage)
- Trace-Bench notebook: `notebooks/t6_multiobjective_benchmarks.ipynb` (in Trace-Bench repo)

**SMART validation:**
- Notebook outputs per-benchmark table: weighted-mode best candidate metrics + Pareto-mode set of tradeoffs
- Benchmarks run in StubLLM mode (fast/deterministic) and real LLM mode (small sample)
- Trace-Bench run completes without private datasets
- `pytest -q` green (smoke tests for benchmark integration)

---

### Milestone 4 — Documentation + polished notebooks

**Deliverables:**
- `docs/multi_objective_scores.md` — user-facing documentation
- README update with pointers to docs and notebooks
- Polished "How-to" notebook: installs from GitHub, runs BasicSearch weighted + Pareto, prints metric tradeoffs

**SMART validation:**
- Fresh Colab runtime runs how-to notebook without manual patching
- CI green, no behavioral changes beyond documentation/polish

---

## 9. Test Plan

### 9.1 Unit tests — `tests/test_objectives.py` (M1)

| Test | Validates |
|------|-----------|
| `test_to_score_dict_from_float` | `0.85` → `{"score": 0.85}` |
| `test_to_score_dict_from_dict` | `{"a": 1.0, "b": 2.0}` → same dict |
| `test_to_score_dict_empty_dict_raises` | `{}` → `ValueError` |
| `test_to_score_dict_nan_raises` | `{"a": float('nan')}` → `ValueError` |
| `test_to_score_dict_wrong_type_raises` | `"text"` → `TypeError` |
| `test_apply_minimize` | `{"acc": 0.9, "lat": 100}` with `minimize={"lat"}` → `{"acc": 0.9, "lat": -100}` |
| `test_apply_minimize_empty_set` | No metrics negated |
| `test_weighted_scalarize_basic` | `{"a": 0.8, "b": 0.2}` with `weights={"a": 0.7, "b": 0.3}` → `0.7*0.8 + 0.3*0.2` |
| `test_weighted_scalarize_missing_metric` | Missing metric uses `missing_value` |
| `test_weighted_scalarize_empty_weights` | Equal weight 1.0 for all metrics |
| `test_dominates_true` | A dominates B (all ≥, at least one >) |
| `test_dominates_false_equal` | A == B → does not dominate |
| `test_dominates_false_tradeoff` | A better on one, B better on another |
| `test_pareto_rank_simple` | 3 candidates with clear rank 0, 1, 2 |
| `test_pareto_rank_all_nondominated` | All candidates rank 0 |
| `test_select_best_scalar_mode` | Falls back to scalar max |
| `test_select_best_weighted_mode` | Returns highest weighted score |
| `test_select_best_pareto_mode` | Returns Pareto-optimal by tie-break |
| `test_select_best_none_config` | `objective_config=None` → scalar max (backward compat) |
| `test_select_top_k_weighted` | Returns k highest weighted scores |
| `test_select_top_k_pareto` | Returns k from Pareto front + spillover |
| `test_deterministic_tie_break_seeded` | Same seed → same result across 100 runs |
| `test_deterministic_tie_break_different_seeds` | Different seeds → potentially different result |

### 9.2 Unit tests — `tests/test_evaluators_vector.py` (M1)

| Test | Validates |
|------|-----------|
| `test_aggregate_vector_scores_all_scalar` | `[0.8, 0.9, 0.7]` → `np.mean` (backward compat) |
| `test_aggregate_vector_scores_all_dict` | Per-metric mean computed correctly |
| `test_aggregate_vector_scores_mixed` | Scalars normalized to dict, then averaged |
| `test_evaluate_vector_returns_correct_types` | Returns list of ScoreLike matching guide output |

### 9.3 Integration tests — `tests/test_trainers_multiobjective.py` (M2)

| Test | Validates |
|------|-----------|
| `test_basicsearch_scalar_unchanged` | Default behavior identical to pre-change |
| `test_basicsearch_weighted_selects_expected` | Weighted mode picks correct candidate |
| `test_basicsearch_pareto_selects_expected` | Pareto mode picks different candidate than weighted |
| `test_beamsearch_scalar_unchanged` | Default behavior identical |
| `test_beamsearch_weighted_selects_top_k` | Weighted mode picks correct top-k |
| `test_beamsearch_pareto_selects_front` | Pareto mode returns non-dominated front |
| `test_deterministic_across_runs` | Fixed seed → same selections in 5 repeated runs |

### 9.4 Notebook validation (human / Trace team)

Each notebook contains:
- **StubLLM (no keys) section:** deterministic dummy guide, runs quickly
- **Real LLM section (required):** small N (5-20 examples), prints cost/latency caveats, requires API key

---

## 10. Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| **R1: Missing metrics across candidates** | Medium | `missing_value` in ObjectiveConfig (default `-inf`). Enforce metric presence for configured weights (or warn). |
| **R2: Pareto nondeterminism** | High | Deterministic ordering via stable sort + explicit tie-break rules. Seeded randomness only when requested. |
| **R3: Multi-thread eval ordering** | Medium | Tests run with `num_threads=1` to guarantee stability. Document thread-safety considerations. |
| **R4: Breaking Guide subclasses** | High | Use `get_score_dict()` helper — never change `get_feedback()` signature. Union type on `metric()` is safe because existing callers only receive floats. |
| **R5: Performance regression** | Low | `objectives.py` functions are O(n²) for Pareto ranking on n candidates, but n is typically ≤20 (num_proposals). No concern at this scale. |
| **R6: Mixed scalar/dict in same batch** | Medium | `aggregate_vector_scores()` rejects mixed batches with `ValueError`. A mixed batch indicates a bug in the guide. |
| **R7: Training loop receives dict score** | High | `guide.__call__()` / `get_feedback()` return type is NOT widened. `metric()` always returns `float`. Dict scores only flow through `get_score_dict()` → `evaluate_vector()`. See §7.7. |

---

## 11. Design Decisions (Resolved)

> **Post-review update (Ching-An, Feb 2026):** All dict→scalar reduction is now controlled by `ObjectiveConfig.scalarize_dict` (values: `"score"`, `"mean"`, `"weighted"`). Guide produces raw metrics only. `normalize_score` renamed to `to_score_dict` (neutral name; backward-compat alias kept). `aggregate_score_dicts()` moved from evaluators to objectives.py (Objective-side policy). Dict scores with `config=None` now raise `ValueError` (no hidden hard-coded reduction).

### D1: Where to implement scalar→dict normalization?

**Decision: Option A — `Guide.get_score_dict()` helper + `objectives.to_score_dict()`**

- `get_score_dict()` on Guide provides a clean entry point for subclasses.
- `to_score_dict()` in objectives.py is the canonical utility (pure function, testable). Renamed from `normalize_score` per Ching-An's review (neutral name; backward-compat alias kept).
- All dict→scalar reduction is controlled by `ObjectiveConfig` (via `scalarize_dict` field). No hidden hard-coded defaults.
- Avoids widening `get_feedback()` return type (higher churn, breaks typing).

### D2: Pareto selection definition

**Decision: Option A — Standard dominance on aggregated metrics, return single best by tie-break.**

- `select_best()` returns one winner. `select_top_k()` returns k winners.
- Trainers don't need to manage a "front" — they just get indices.
- Beamsearch naturally uses `select_top_k(k=beam_width)`.

### D3: PrioritySearch scope

**Decision: Minimal (in-scope).**

- Scalarize heap priority via `weighted_scalarize()`.
- Store full `score_dict` on each candidate for logging.
- `mode="pareto"` falls back to weighted with documented warning.
- Pareto archive is out-of-scope for v1.

---

## 12. Appendix: Code Touchpoints

### OpenTrace / experimental

| File | URL |
|------|-----|
| Guide base | [guide.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/trainer/guide.py) |
| Evaluators | [evaluators.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/trainer/evaluators.py) |
| BasicSearch | [basic_algorithms.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/trainer/algorithms/basic_algorithms.py) |
| Beamsearch | [beamsearch_algorithm.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/trainer/algorithms/beamsearch_algorithm.py) |
| PrioritySearch | [priority_search.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/features/priority_search/priority_search.py) |

### Trace-Bench

| File | URL |
|------|-----|
| Repo | [Trace-Bench](https://github.com/AgentOpt/Trace-Bench) |

### Selection logic summary (current → target)

| Trainer | Current Code | Target Code |
|---------|-------------|-------------|
| BasicSearch | `max(candidates, key=lambda x: x[0])` | `select_best(candidates, objective_config)` |
| Beamsearch | `sorted(candidates, key=lambda x: x[0], reverse=True)[:k]` | `select_top_k(candidates, objective_config, k)` |
| PrioritySearch | scalar heap key | `weighted_scalarize(score_dict, config)` for heap key |
