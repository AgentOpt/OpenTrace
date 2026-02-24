# Multi-Query Backward Pass Issue in `optimize_graph()`

## Summary

After applying the recommended diff (`T1_PR_M1_latest_fix.diff`), the live optimization fails with:

```
AssertionError: user feedback should be the same for all children
```

at `graph_propagator.py:81` during `_optimizer.step()` → `summarize()` → `aggregate()`.

The `param_cache` fix and the multi-output backward change from the diff are fundamentally incompatible with Trace's `GraphPropagator`.

---

## Root Cause

### The Shared Graph Problem

`param_cache` (introduced by the diff) ensures stable `ParameterNode` instances across queries — this is necessary so the optimizer can track canonical parameters. However, when all 3 queries' TGJ docs are ingested with the same `param_cache`, the Trace graph becomes:

```
planner_prompt (ONE shared ParameterNode)
 ├── Q1_planner_msg → Q1_executor_msg → ... → Q1_output_node
 ├── Q2_planner_msg → Q2_executor_msg → ... → Q2_output_node
 └── Q3_planner_msg → Q3_executor_msg → ... → Q3_output_node
```

All queries' message nodes are **children** of the same `ParameterNode`.

### Why Multi-Output Backward Fails

The diff changed the backward pass to iterate over all output nodes:

```python
_optimizer.zero_feedback()
for output_node, run_for_output in all_output_nodes:
    feedback_text = f"Score: {run_for_output.score}"  # different per query
    _optimizer.backward(output_node, feedback_text)
_optimizer.step()  # crashes
```

When `step()` calls `summarize()` → `propagator.aggregate()`, it sums the `Propagator` objects from all children of each shared `ParameterNode`. The `Propagator.__add__` method asserts:

```python
assert self.user_feedback == other.user_feedback
```

This fails because Q1 has `"Score: 0.6"` and Q2 has `"Score: 0.85"`.

### Why Per-Query Backward Also Fails

We attempted processing each query independently:

```python
for output_node, run_for_output in all_output_nodes:
    _optimizer.zero_feedback()
    _optimizer.backward(output_node, feedback_text)
    _optimizer.step()
```

This still fails because:

1. `zero_feedback()` clears feedback on **parameter nodes**, not on intermediate **message nodes**
2. `Q1_planner_msg` (an intermediate node) retains its graph connection to the shared `planner_prompt`
3. When Query 2's backward reaches `planner_prompt`, the propagator still sees Q1's children with stale feedback
4. The assertion fires again

### Why the BBEH Notebook Doesn't Have This Problem

In the BBEH notebook, `ParameterNode` objects are created **once** in Python memory (via `@bundle` / `FunModule`) and the graph runs through the **same live nodes** for each example. Each example's execution replaces the previous message nodes, so there's only ever one set of children per `ParameterNode`. The flow is:

```
invoke → output → zero_feedback → backward → step → apply_updates → next example
```

There is no TGJ reconstruction, no `param_cache`, and no multi-query trace accumulation.

### Why Dropping `param_cache` Doesn't Work Either

Without `param_cache`, each query creates **independent** `ParameterNode` instances. The optimizer is initialized with Query 1's nodes. When Query 2's backward traces to Query 2's different `ParameterNode` objects, the optimizer has no reference to them — backward has no effect.

---

## Current Workaround

We use a **single backward pass on the last output node** with an aggregated feedback string:

```python
output_node, _ = all_output_nodes[-1]
all_scores = [r.score for _, r in all_output_nodes if r.score is not None]
feedback_text = f"Average score: {avg:.4f} across {len(all_scores)} queries (individual: ...)"

_optimizer.zero_feedback()
_optimizer.backward(output_node, feedback_text)
raw_updates = _optimizer.step()
```

**Pros:**
- No assertion error — single feedback string, single backward pass
- Optimizer sees the combined signal from all queries
- `param_cache`, `_normalize_key()`, and stale span flush all work correctly

**Cons:**
- Only the last query's graph path is used for gradient computation
- Earlier queries' graph structures don't contribute to the backward trace

---

## Possible Approaches

### Option A: Single Backward with Aggregated Feedback (current workaround)

Keep the current approach. Simple, working, and gives the optimizer a summary of all queries. The optimizer's LLM-based reasoning can interpret the aggregated feedback to decide on prompt changes.

### Option B: Sequential Per-Query Full Cycles

Process queries one at a time with **separate** `param_cache` per query and a fresh ingest each time. After each query: ingest → backward → step → apply_updates → next query.

This mirrors the BBEH pattern but requires restructuring the optimization loop significantly. Each query would need its own OTLP → TGJ → ingest cycle, and `apply_updates` would need to happen between queries so the next query runs with updated prompts.

### Option C: Unified Feedback String Across All Backward Passes

Call `backward()` on all output nodes but with the **same** aggregated feedback string:

```python
_optimizer.zero_feedback()
for output_node, _ in all_output_nodes:
    _optimizer.backward(output_node, aggregated_feedback)  # same string
_optimizer.step()
```

This satisfies the assertion (`user_feedback` is identical for all children) and lets all graph paths contribute to the gradient. However, this hasn't been tested and may have other propagator assumptions that break.

---

## Questions for Review

1. Is **Option A** (single backward, aggregated feedback) acceptable for M1 validation?
2. Should we pursue **Option B** (sequential per-query cycles, matching BBEH) for a future milestone?
3. Is **Option C** worth testing, or does the propagator have other constraints that would break?
4. Any other approach you'd recommend given Trace's propagator internals?
