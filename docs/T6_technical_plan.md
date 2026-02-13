# T6 Technical Plan: Multi‑Objective Vector Scores for Trainer Selection

**Target PR:** [`AgentOpt/OpenTrace@experimental`](https://github.com/AgentOpt/OpenTrace/tree/experimental)  
**Benchmark integration:** [`AgentOpt/Trace-Bench`](https://github.com/AgentOpt/Trace-Bench)  
**Status:** Final – M0 deliverable (revised per client feedback)  
**Last updated:** 2026-02-13

------

## Table of Contents

1. Executive summary
2. Goals, non-goals, crisp success criteria
3. Current code reality (baseline)
4. Proposed architecture (minimal delta)
5. Public API & data contracts (ObjectiveConfig, Score types)
6. Module modifications (files to create/modify)
7. Milestones & validation gates (each milestone ships Colab notebook + pytest from M1+)
8. Tests & validation plan (StubLLM + real LLM)
9. Risks, edge cases, and mitigation
10. Options / decisions (if Trace team wants to choose)
11. Appendix: direct repo touchpoints

---

## 1. Executive Summary

Today, `opto` trainers (BasicSearch, Beamsearch, PrioritySearch) select candidates based on a **single scalar score**, even though guides/evaluators can already produce rich feedback. This prevents the trainer from exploiting **multiple objectives** (e.g., accuracy, latency, cost, complexity) during candidate search.

This plan introduces a **minimal, backward‑compatible extension** that allows guides/evaluators to return a `Dict[str, float]` vector score. Trainers are upgraded to support two multi‑objective selection modes:

- **Weighted scalarization** – linear combination of metrics with user‑defined weights and direction.
    
- **Pareto dominance** – non‑dominated sorting for true trade‑off selection.
    

All existing scalar‑only pipelines continue to work **without modification**. New functionality is isolated in a single module (`objectives.py`) and tested with both deterministic stubs and real LLMs. Every milestone ships a **Google Colab notebook**; from M1 onward **pytest coverage** is mandatory.

---

## 2. Goals, Non‑Goals & Success Criteria

### 2.1 Goals (In Scope)

| ID     | Goal                                                                                                                      |
| ------ | ------------------------------------------------------------------------------------------------------------------------- |
| **G1** | **100% backward compatibility** – existing scalar‑only guides/trainers produce identical results.                         |
| **G2** | **Vector score support** – guides may return `Dict[str, float]`; trainers can select using `weighted` or `pareto` modes.  |
| **G3** | **Determinism** – with a fixed `seed`, selection is reproducible (especially Pareto tie‑breaks).                          |
| **G4** | **Actionable validation** – each milestone includes a Colab notebook (StubLLM + real LLM) and, from M1+, pytest coverage. |
| **G5** | **Benchmarks** – 3 simple multi‑objective benchmarks defined and integrated into Trace‑Bench (M3).                        |

### 2.2 Non‑Goals (Explicitly Out of Scope)

- Full multi‑objective Bayesian optimisation (e.g., MO‑UCB) – too complex for v1.
    
- Pareto archive / non‑dominated set management inside PrioritySearch.
    
- Changing the `get_feedback` signature in `BaseGuide` – we add a helper instead.
    
- New telemetry infrastructure – logging leverages existing `BaseLogger`.
    

### 2.3 Success Criteria (Definition of Done)

The project is accepted when:

1. Scalar‑only trainers still work and produce the same best candidate.
    
2. A guide returning `Dict[str, float]` works end‑to‑end with BasicSearch and Beamsearch.
    
3. Weighted and Pareto selections are **deterministic** under fixed seed.
    
4. All M1 onwards, new functions have pytest tests and CI remains green.
    
5. M3: three benchmarks runnable from Trace‑Bench.
    
6. M4: documentation and polished how‑to notebooks are published.
    

---

## 3. Current Baseline (Without Changes)

- **Guide:** `Guide.get_feedback(...) -> Tuple[float, str]` – only the scalar score is used for trainer‑side selection.
    
- **Evaluator:** `evaluate(...)` returns a 1D array of scalar scores (per example). Aggregation is a simple mean.
    
- **Trainers:** `BasicSearchAlgorithm` and `BeamsearchAlgorithm` select the candidate with the **highest mean score**. PrioritySearch uses a scalar heap key.
    
- **Logging:** `BaseLogger` can log arbitrary metrics; currently only the primary scalar is logged.
    
- **StubLLM:** A `DummyLLM` exists for deterministic testing – we reuse this for CI and notebook “no‑keys” sections.
    

---

## 4. Proposed Architecture – Minimal Delta

The core idea: **isolate all new complexity into a single, easily testable module** (`objectives.py`). Trainers call a small set of pure functions to convert vector scores into selection decisions.

**Data flow (new, optional path):**

Guide                            Evaluator
   │                                   │
   └─► returns Dict[str,float]         └─► per-example dicts → mean dict
                                         │
                                         ▼
Trainer (with ObjectiveConfig)
   │
   ├─► Weighted mode: scalarize → sort
   └─► Pareto mode: non‑dominated sort → tie‑break

All changes are **backward compatible**:

- If `objective_config=None`, trainers fall back to scalar behaviour.
    
- If a guide returns a scalar, it is transparently wrapped as `{"score": value}`.
    
- Existing `Guide` subclasses that only implement `get_feedback` need **no changes** – we provide a helper `get_score_dict()`.
    

---

## 5. Detailed API Design

### 5.1 Score types

```python
ScalarScore = float
VectorScore = dict[str, float]          # JSON-serializable
ScoreLike = float | dict[str, float]
```

Contract:
* “Higher is better” by default.
* Metrics to minimize must be specified via `ObjectiveConfig.minimize`. 

### 5.2 `ObjectiveConfig` (new, in `objectives.py`)

```python
@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for multi‑objective candidate selection."""
    mode: Literal["scalar", "weighted", "pareto"] = "scalar"
    # Weighted mode
    weights: Optional[Dict[str, float]] = None      # required if mode="weighted"
    minimize: Union[List[str], Set[str], None] = None
    # Pareto mode
    pareto_metrics: Optional[Tuple[str, ...]] = None  # None = use all metrics
    tie_break: Literal["weighted", "lexicographic", "first", "last", "random"] = "weighted"
    # Determinism
    seed: Optional[int] = None
    # Fallback for missing metrics
    missing_value: float = float("-inf")
```
**Validation rules** (enforced in `__post_init__`):

- If `mode="weighted"`, `weights` must be provided and non‑empty.
- If `mode="pareto"`, `weights` are ignored for dominance calculations but may be used for `tie-break`- a warning is logged if weights are missing in that case.
- `apply_minimize` can be a list/set of metric names that should be **minimised** (others are maximised).
- `seed` is used only when `tie_break="random"`.

### 5.3 Sign Conventions

To maintain a **uniform “higher is better”** rule across all internal comparisons:

1. **Minimisation handling** – metrics listed in `minimize` are multiplied by `-1` via `apply_minimize()`. After this transformation, **higher scores are always better** for every metric.
    
2. **Weights** – because all metrics are already oriented “higher is better”, **weights should normally be non‑negative**. Negative weights are **not prohibited**, but they invert the intended direction and may cause counter‑intuitive results; users are advised against them.
    
This convention is applied **before** any weighted scalarization or Pareto dominance check.

### 5.4 Score Normalization & Utilities (in `objectives.py`)

All functions are **pure** and fully tested.

```python

def normalize_score(score: Union[float, Dict[str, float]]) -> Dict[str, float]:
    """Convert scalar → {"score": value}, pass through dict."""
def apply_minimize(score_dict: Dict[str, float], minimize: Set[str]) -> Dict[str, float]:
    """Multiply minimised metrics by -1 so that higher is always better."""
def weighted_scalarize(
    score_dict: Dict[str, float],
    weights: Dict[str, float],
    missing_value: float = float("-inf")
) -> float:
    """Compute weighted sum. Missing metrics get `missing_value`."""
def pareto_dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """True if a is strictly better on at least one metric and not worse on all."""
def pareto_front(
    scores: List[Dict[str, float]],
    metrics: Optional[List[str]] = None,
    tie_break: str = "weighted",
    weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None
) -> List[int]:
    """Return indices of non‑dominated candidates, with deterministic tie‑break."""
```
### 5.5 Guide Extensions (minimal, backward‑compatible)

In `opto/trainer/guide.py`:

```python

class BaseGuide(ABC):
    # ... existing abstract methods ...
    def get_score_dict(self, params: Parameterized) -> Dict[str, float]:
        """Unified interface to obtain a vector score.
        - If the guide returns a scalar, wrap as {"score": value}.
        - If it already returns a dict, pass through.
        Subclasses may override for efficiency.
        """
        feedback = self.get_feedback(params)   # (score, message)
        if isinstance(feedback[0], dict):
            return feedback[0]
        return {"score": float(feedback[0])}
```
No change to `get_feedback` signature – **no breakage**.

### 5.6 Evaluator Extensions

In `opto/trainer/evaluators.py`:

```python

def evaluate_vector(
    guide: BaseGuide,
    params_list: List[Parameterized],
    objective_config: Optional[ObjectiveConfig] = None,
    **kwargs
) -> List[Dict[str, float]]:
    """Evaluate each candidate and return per‑example dict scores."""
def aggregate_vector_scores(
    per_example_scores: List[Dict[str, float]]
) -> Dict[str, float]:
    """Element‑wise mean of all dicts."""
```
The existing `evaluate()` method remains unchanged for scalar‑only use.

### 5.7 Trainer Upgrades – Selection Logic

Both `BasicSearchAlgorithm` and `BeamsearchAlgorithm` gain an optional `objective_config: Optional[ObjectiveConfig] = None` parameter.

**Selection step** (pseudocode):

```python

if objective_config is None or objective_config.mode == "scalar":
    # Legacy path: use mean scalar score
    best_idx = argmax(mean_scalar_scores)
else:
    # Obtain per‑candidate dict scores (already aggregated by evaluator)
    dict_scores = [candidate.score_dict for candidate in candidates]
    if objective_config.mode == "weighted":
        # Transform direction, scalarize, sort descending
        transformed = [apply_minimize(d, minimize_set) for d in dict_scores]
        values = [weighted_scalarize(d, weights, missing_value) for d in transformed]
        best_idx = argmax(values)
    elif objective_config.mode == "pareto":
        # Pareto front indices, then tie‑break
        front_idxs = pareto_front(dict_scores, ...)
        # If multiple candidates remain, use tie_break rule
        best_idx = select_from_front(front_idxs, ...)
```

**Beamsearch** uses the same logic to select the top‑k candidates.

**PrioritySearch** (minimal upgrade):

- Add `objective_config` to config.
    
- Compute heap priority via `weighted_scalarize` (or fallback to primary metric).
    
- Store the full `score_dict` on each rollout for logging.
    
- If `mode="pareto"`, fallback to weighted with a logged warning – Pareto archive is out of scope.
    

---

## 6. Module Modification Plan (Exact Files)

| File                                                         | Change Type  | Description                                                                                                      |
| ------------------------------------------------------------ | ------------ | ---------------------------------------------------------------------------------------------------------------- |
| `opto/trainer/objectives.py`                                 | **New**      | Core utilities: `ObjectiveConfig`, normalisation, weighted scalarization, Pareto dominance, Pareto front.        |
| `opto/trainer/guide.py`                                      | **Modify**   | Add `get_score_dict()` helper.                                                                                   |
| `opto/trainer/evaluators.py`                                 | **Modify**   | Add `evaluate_vector` and `aggregate_vector_scores`.                                                             |
| `opto/trainer/algorithms/basic_algorithms.py`                | **Modify**   | Accept `objective_config`, replace selection logic with dispatch to `objectives.py`. Keep scalar path identical. |
| `opto/trainer/algorithms/beamsearch_algorithm.py`            | **Modify**   | Same as above.                                                                                                   |
| `opto/features/priority_search/priority_search.py`           | **Modify**   | Add `objective_config`; use weighted scalarization for heap key; store vector score; fallback if pareto.         |
| `tests/opto/trainer/test_objectives.py`                      | **New**      | Unit tests for all pure functions.                                                                               |
| `tests/opto/trainer/test_evaluators.py`                      | **Modify**   | Tests for vector evaluation and aggregation.                                                                     |
| `tests/opto/trainer/algorithms/test_basic_algorithms.py`     | **Modify**   | Integration‑style tests for multi‑objective selection.                                                           |
| `tests/opto/trainer/algorithms/test_beamsearch_algorithm.py` | **Modify**   | Same.                                                                                                            |
| `tests/features/priority_search/test_priority_search.py`     | **Modify**   | Smoke test for vector score support.                                                                             |
| `examples/notebooks/`                                        | **Add**      | Milestone notebooks (M0–M4).                                                                                     |
| `docs/multi_objective_scores.md`                             | **New (M4)** | End‑user documentation.                                                                                          |

---

## 7. Milestones & Validation Gates

Each milestone ships a **Colab notebook** with:

- **StubLLM (deterministic, no keys)** – demonstrates correctness.
    
- **Real LLM (optional, needs env var)** – shows realistic usage.
    
- **Clear “How to validate” section**.
    

**From M1 onward**: every new function/behaviour must be covered by `pytest` and CI must pass `pytest -q`.

### Milestone 0 (M0) – Analysis & Plan

-     Refined technical plan (this document).
    
-      **Notebook `t6_m0_analysis.ipynb`**:
    
    - Demos baseline scalar selection.
        
    - Shows intended API signatures via stubs.
        
    - Illustrates Pareto front vs weighted selection with toy candidates.
        
    - No code changes – pure design demonstration.
        

### Milestone 1 (M1) – Core Utilities + BasicSearch

- **Code:**
    
    - `objectives.py` complete with tests.
        
    - `guide.py` helper.
        
    - `evaluators.py` vector methods.
        
    - **BasicSearchAlgorithm** upgraded (minimal integration).
        
- **Tests:** Unit tests for objectives, evaluators, and BasicSearch multi‑objective selection.
    
- **Notebook `t6_m1_vector_scores.ipynb`**:
    
    - BasicSearch with deterministic dummy guide.
        
    - Show weighted vs Pareto selections.
        
    - Demonstrate deterministic tie‑break.
        

### Milestone 2 (M2) – Full Trainer Upgrades

- **Code:**
    
    - **BeamsearchAlgorithm** upgraded.
        
    - **PrioritySearch** minimal support.
        
    - Expanded BasicSearch tests.
        
- **Tests:** Integration tests confirming weighted vs Pareto differ; deterministic behaviour.
    
- **Notebook `t6_m2_trainers.ipynb`**:
    
    - Both trainers in scalar, weighted, Pareto modes.
        
    - Logging of per‑metric curves.
        

### Milestone 3 (M3) – Trace‑Bench Benchmarks

- **Code:**
    
    - 3 simple multi‑objective benchmarks defined.
        
    - PR to `AgentOpt/Trace-Bench` with benchmark configs and notebook.
        
- **Notebook `t6_m3_benchmarks.ipynb`** (in Trace‑Bench repo):
    
    - Runs benchmarks with tiny budget.
        
    - Outputs comparison table (scalar vs weighted vs Pareto).
        
- **Smoke tests** for benchmark integration.
    

### Milestone 4 (M4) – Documentation & Polishing

- **Code:**
    
    - `docs/multi_objective_scores.md` – explains how to enable multi‑objective mode, declare minimise/weights, interpret Pareto results.
        
    - README update.
        
- **Notebook `how_to_multi_objective.ipynb`** – polished, self‑contained, installs from GitHub.
    

---

## 8. Test & Validation Strategy

### 8.1 Unit Tests (pytest, CI)

- **Pure functions** in `objectives.py`: 100% coverage.
    
- **Evaluator vector helpers**: correct aggregation, edge cases (empty list, mismatched keys).
    
- **Determinism**: same seed → same selection, especially Pareto tie‑break.
    

### 8.2 Integration Tests (pytest, CI)

- **BasicSearch/Beamsearch** with dummy guide:
    
    - Scalar mode yields same result as before.
        
    - Weighted mode respects weights and minimisation.
        
    - Pareto mode returns a non‑dominated candidate.
        
    - Tie‑break stability.
        

### 8.3 Notebook Validation (manual, Colab)

- **StubLLM section** – must run without any API keys, fast, deterministic.
    
- **Real LLM section** – small dataset, clearly marked, requires user to supply key.
    

### 8.4 Benchmark Smoke Tests (pytest, CI)

- Minimal run of each benchmark with `budget=1` to ensure no import/configuration errors.
    

---

## 9. Edge Cases & Mitigations

| Edge Case                                             | Handling Strategy                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Guide returns scalar**                              | Automatically wrapped as `{"score": value}`. Trainer scalar path unchanged.                                   |
| **Dict contains only one metric**                     | Weighted and Pareto modes still work; Pareto reduces to simple sort.                                          |
| **Metric missing from dict but present in weights**   | Use `missing_value` (default `-inf`). User warned if configured.                                              |
| **Minimisation mixed with maximisation**              | `minimize` set; `apply_minimize` flips sign internally.                                                       |
| **All candidates have identical scores**              | Tie‑break rule (`first`/`last`/`random`) guarantees deterministic selection.                                  |
| **User provides weights that sum to 0 or negative**   | No normalisation – user responsibility. Weighted sum works as defined.                                        |
| **Pareto with >3 objectives**                         | Non‑dominated sort is O(n²). For typical beam sizes (<20) this is fine. Document limitation.                  |
| **Parallel evaluation (multithreading)**              | Determinism can break if order nondeterministic. **Recommendation:** for tests/notebooks use `num_threads=1`. |
| **Existing Guide subclasses override `get_feedback`** | `get_score_dict()` calls `get_feedback()` – no need to override. Subclasses may override for efficiency.      |

---

## 10. Open Decisions (to be finalised in M0 review)

1. **Scalar→dict key name:** Use `"score"` (default) or allow customisation?  
    _Proposal:_ Hardcode `"score"` – simplest, fully backward‑compatible.
    
2. **Pareto tie‑break default:** `"weighted"` (use weights as secondary sort) vs `"lexicographic"` (use first metric)?  
    _Proposal:_ `"weighted"` – most intuitive when weights are provided; fallback to `"lexicographic"` if no weights.
    
3. **Logging of vector components:** Should we automatically log `val/<metric_name>` for each aggregated metric?  
    _Proposal:_ Yes, but optional behind a flag (to avoid log spam). We implement it in M2.
    
4. **PrioritySearch Pareto fallback:** Log warning or silently fall back?  
    _Proposal:_ Log a clear warning and fall back to weighted.
    
---

## 11. Appendix: Direct Code Touchpoints (for implementer)

**OpenTrace / experimental branch:**

- [opto/trainer/guide.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/trainer/guide.py)
    
- [opto/trainer/evaluators.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/trainer/evaluators.py)
    
- [opto/trainer/algorithms/basic_algorithms.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/trainer/algorithms/basic_algorithms.py)
    
- [opto/trainer/algorithms/beamsearch_algorithm.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/trainer/algorithms/beamsearch_algorithm.py)
    
- [opto/features/priority_search/priority_search.py](https://github.com/AgentOpt/OpenTrace/blob/experimental/opto/features/priority_search/priority_search.py)
    

**Trace‑Bench:**

- [AgentOpt/Trace-Bench](https://github.com/AgentOpt/Trace-Bench)
