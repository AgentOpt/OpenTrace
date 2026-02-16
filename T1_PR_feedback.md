# 1) `optimize_graph()` best_updates tracking is incorrect (can be overwritten by later iterations)

### Problem

Current code sets `best_updates = dict(updates)` whenever updates are applied, even if that iteration is not best. That makes `best_updates` inconsistent with `best_iteration` / `best_parameters`.

---

## Approach A (minimal): track “updates applied to reach this iteration”

```diff
diff --git a/opto/trace/io/optimization.py b/opto/trace/io/optimization.py
--- a/opto/trace/io/optimization.py
+++ b/opto/trace/io/optimization.py
@@ -318,6 +318,8 @@
     best_score = float("-inf")
     best_iteration = 0
     best_updates: Dict[str, Any] = {}
+    # Updates applied *before* the current iteration (used to reach current params)
+    last_applied_updates: Dict[str, Any] = {}
@@ -363,6 +365,8 @@
     total_iters = iterations + 1  # baseline + N iterations
 
     for iteration in range(total_iters):
+        # Snapshot the updates that produced the parameters used in this iteration
+        applied_updates_for_this_iter = dict(last_applied_updates)
@@ -461,6 +465,7 @@
             best_iteration = iteration
             best_parameters = _snapshot_parameters(effective_bindings)
             marker = " * NEW BEST" if not is_baseline else ""
+            best_updates = dict(applied_updates_for_this_iter)
@@ -542,6 +547,7 @@
             if updates and apply_updates_flag:
                 try:
                     apply_updates(updates, effective_bindings, strict=False)
+                    last_applied_updates = dict(updates)
                     logger.info("Applied updates: %s", sorted(updates.keys()))
                 except Exception as exc:
                     logger.warning("apply_updates failed: %s", exc, exc_info=True)
+            else:
+                last_applied_updates = {}
```

---

## Approach B (robust): store `applied_updates_to_reach_iter[]` history

```diff
diff --git a/opto/trace/io/optimization.py b/opto/trace/io/optimization.py
--- a/opto/trace/io/optimization.py
+++ b/opto/trace/io/optimization.py
@@ -362,6 +362,12 @@
 
     total_iters = iterations + 1  # baseline + N iterations
 
+    # Track which updates were applied to reach the parameter values
+    # used in each iteration i (i=0 baseline has no prior updates).
+    applied_updates_to_reach_iter: List[Dict[str, Any]] = [
+        {} for _ in range(total_iters)
+    ]
+
     for iteration in range(total_iters):
@@ -461,6 +467,7 @@
             best_iteration = iteration
             best_parameters = _snapshot_parameters(effective_bindings)
             marker = " * NEW BEST" if not is_baseline else ""
+            best_updates = dict(applied_updates_to_reach_iter[iteration])
@@ -538,7 +545,8 @@
             if updates and apply_updates_flag:
                 try:
                     apply_updates(updates, effective_bindings, strict=False)
-                    best_updates = dict(updates)
+                    if iteration + 1 < total_iters:
+                        applied_updates_to_reach_iter[iteration + 1] = dict(updates)
                     logger.info("Applied updates: %s", sorted(updates.keys()))
```

---

# 2) `optimize_graph()` ignores `graph.output_key` unless caller passes `output_key=...`

### Problem

The instrumented graph now supports `output_key`, but optimize_graph does not default to it. This is a usability issue and can cause incorrect eval payload shape.

---

## Approach A (minimal): fallback to `graph.output_key`

```diff
diff --git a/opto/trace/io/optimization.py b/opto/trace/io/optimization.py
--- a/opto/trace/io/optimization.py
+++ b/opto/trace/io/optimization.py
@@ -313,6 +313,10 @@
 
     eval_fn = eval_fn or _default_eval_fn
 
+    # If not provided, fall back to the graph's configured output_key
+    if output_key is None:
+        output_key = getattr(graph, "output_key", None)
+
     score_history: List[float] = []
```

---

## Approach B (robust): fallback + log when caller overrides graph config

```diff
diff --git a/opto/trace/io/optimization.py b/opto/trace/io/optimization.py
--- a/opto/trace/io/optimization.py
+++ b/opto/trace/io/optimization.py
@@ -313,6 +313,18 @@
 
     eval_fn = eval_fn or _default_eval_fn
 
+    # If not provided, fall back to the graph's configured output_key.
+    # If both are provided and disagree, prefer the explicit argument.
+    graph_output_key = getattr(graph, "output_key", None)
+    if output_key is None:
+        output_key = graph_output_key
+    elif graph_output_key and output_key != graph_output_key:
+        logger.debug(
+            "optimize_graph: output_key=%r overrides graph.output_key=%r",
+            output_key,
+            graph_output_key,
+        )
+
     score_history: List[float] = []
```

---

# 3) `enable_code_optimization` in `instrument_graph()` is currently a no-op

### Problem

The parameter is exposed/documented but not wired into `TracingLLM.emit_code_param`.

---

## Approach A (minimal): emit compact code preview + hash into span attrs when enabled

```diff
diff --git a/opto/trace/io/instrumentation.py b/opto/trace/io/instrumentation.py
--- a/opto/trace/io/instrumentation.py
+++ b/opto/trace/io/instrumentation.py
@@ -10,6 +10,8 @@
 import logging
+import hashlib
+import inspect
@@ -180,11 +182,34 @@
         for key in templates:
             bindings[key] = make_dict_binding(templates, key, kind="prompt")
 
+    # -- optional code parameter emission -------------------------------
+    emit_code_param = None
+    if enable_code_optimization:
+        def _emit_code_param(span, code_key: str, code_fn: Any) -> None:
+            try:
+                src = inspect.getsource(code_fn)
+            except Exception:
+                src = repr(code_fn)
+            digest = hashlib.sha256(src.encode("utf-8", errors="ignore")).hexdigest()
+            preview = (src[:500] + "...") if len(src) > 500 else src
+            span.set_attribute(f"param.__code_{code_key}", preview)
+            span.set_attribute(f"param.__code_{code_key}.sha256", digest)
+            span.set_attribute(f"param.__code_{code_key}.trainable", True)
+        emit_code_param = _emit_code_param
+
     tracing_llm = TracingLLM(
         llm=llm,
         tracer=session.tracer,
         trainable_keys=trainable_keys,
+        emit_code_param=emit_code_param,
         provider_name=provider_name,
         llm_span_name=llm_span_name,
         emit_llm_child_span=emit_genai_child_spans,
     )
```

---

## Approach B (robust): emit full (capped) source + truncation metadata

```diff
diff --git a/opto/trace/io/instrumentation.py b/opto/trace/io/instrumentation.py
--- a/opto/trace/io/instrumentation.py
+++ b/opto/trace/io/instrumentation.py
@@ -10,6 +10,8 @@
 import logging
+import hashlib
+import inspect
@@ -180,11 +182,40 @@
         for key in templates:
             bindings[key] = make_dict_binding(templates, key, kind="prompt")
 
+    emit_code_param = None
+    if enable_code_optimization:
+        CODE_ATTR_MAX_CHARS = 10_000
+        def _emit_code_param(span, code_key: str, code_fn: Any) -> None:
+            try:
+                src = inspect.getsource(code_fn)
+            except Exception:
+                src = repr(code_fn)
+            digest = hashlib.sha256(src.encode("utf-8", errors="ignore")).hexdigest()
+            was_truncated = False
+            if len(src) > CODE_ATTR_MAX_CHARS:
+                src = src[:CODE_ATTR_MAX_CHARS] + "\n# ... (truncated)"
+                was_truncated = True
+            span.set_attribute(f"param.__code_{code_key}", src)
+            span.set_attribute(f"param.__code_{code_key}.sha256", digest)
+            span.set_attribute(f"param.__code_{code_key}.truncated", str(was_truncated))
+            span.set_attribute(f"param.__code_{code_key}.trainable", True)
+        emit_code_param = _emit_code_param
+
     tracing_llm = TracingLLM(
         llm=llm,
         tracer=session.tracer,
         trainable_keys=trainable_keys,
+        emit_code_param=emit_code_param,
         provider_name=provider_name,
         llm_span_name=llm_span_name,
         emit_llm_child_span=emit_genai_child_spans,
     )
```

---

# 4) `2 necessary TGJ/OTEL adjustments`

## (A) Avoid dangling TGJ parent refs to skipped root spans — Approach A

```diff
diff --git a/opto/trace/io/otel_adapter.py b/opto/trace/io/otel_adapter.py
--- a/opto/trace/io/otel_adapter.py
+++ b/opto/trace/io/otel_adapter.py
@@ -94,6 +94,12 @@ def otlp_traces_to_trace_json(...):
                 if use_temporal_hierarchy and prev_span_id and not temporal_ignore:
                     if not psid or psid in root_span_ids:
                         effective_psid = prev_span_id
+
+                # If our effective parent is a skipped root invocation span,
+                # do not emit a parent edge that would dangle in TGJ.
+                if effective_psid and effective_psid in root_span_ids:
+                    effective_psid = None
 
                 if effective_psid and "parent" not in inputs:
                     inputs["parent"] = f"{svc}:{effective_psid}"
```

## (B) Ensure child span also records errors when LLMCallError is raised — Approach A

```diff
diff --git a/opto/trace/io/langgraph_otel_runtime.py b/opto/trace/io/langgraph_otel_runtime.py
--- a/opto/trace/io/langgraph_otel_runtime.py
+++ b/opto/trace/io/langgraph_otel_runtime.py
@@ -120,6 +120,7 @@ class TracingLLM:
             # -- invoke LLM, optionally under a child span --
+            llm_sp_ref = None
             try:
                 if self.emit_llm_child_span:
                     with self.tracer.start_as_current_span(self.llm_span_name) as llm_sp:
+                        llm_sp_ref = llm_sp
                         llm_sp.set_attribute("trace.temporal_ignore", "true")
                         ...
                         resp = self.llm(messages=messages, **llm_kwargs)
                         content = self._validate_content(resp.choices[0].message.content)
                 else:
                     resp = self.llm(messages=messages, **llm_kwargs)
                     content = self._validate_content(resp.choices[0].message.content)
             except LLMCallError:
                 sp.set_attribute("error", "true")
                 sp.set_attribute("error.type", "LLMCallError")
+                if llm_sp_ref is not None:
+                    llm_sp_ref.set_attribute("error", "true")
+                    llm_sp_ref.set_attribute("error.type", "LLMCallError")
                 raise
```

# 5) Notebook stub scoring saturates at 1.0 while baseline is 1.0 → optimization “performance” cannot be demonstrated

### Problem
Notebook’s `stub_eval_fn` uses `min(len(answer)/100, 1.0)` and the stub outputs are long enough to saturate → **baseline = 1.0**, best = 1.0.

---

## Approach A (minimal): fix eval_fn only (non-saturating), keep existing stub LLM

```diff
diff --git a/examples/notebooks/01_m1_instrument_and_optimize.ipynb b/examples/notebooks/01_m1_instrument_and_optimize.ipynb
--- a/examples/notebooks/01_m1_instrument_and_optimize.ipynb
+++ b/examples/notebooks/01_m1_instrument_and_optimize.ipynb
@@ -1,14 +1,22 @@
 def stub_eval_fn(payload):
     answer = str(payload.get("answer", ""))
     if isinstance(answer, dict):
         answer = str(answer.get("answer", ""))
-    return EvalResult(
-        score=min(len(answer) / 100.0, 1.0),
-        feedback=f"Answer length: {len(answer)} chars",
-    )
+    # Non-saturating: logistic-like curve capped below 1.0
+    n = max(0, len(answer))
+    score = 1.0 - (1.0 / (1.0 + (n / 200.0)))
+    score = min(score, 0.95)
+    return EvalResult(score=score, feedback=f"Len={n}, score={score:.3f}")
```

**Tradeoff:** fixes saturation, but if the stub LLM doesn’t change output quality with prompt updates, score may still not improve meaningfully.

---

# 6) Notebook trace validation is brittle (name-heuristics) and does not verify root span invariants

### Problem

Notebook checks child spans by `"openai" in name` and will silently pass when the set is empty. It also doesn’t assert the **root invocation span** exists, which is a core D9 requirement.

---

## Approach A (minimal): detect child spans by `trace.temporal_ignore` and assert root span exists

```diff
diff --git a/examples/notebooks/01_m1_instrument_and_optimize.ipynb b/examples/notebooks/01_m1_instrument_and_optimize.ipynb
--- a/examples/notebooks/01_m1_instrument_and_optimize.ipynb
+++ b/examples/notebooks/01_m1_instrument_and_optimize.ipynb
@@ -1,30 +1,43 @@
 spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
+root_spans = [s for s in spans if s["name"].endswith(".invoke")]
+assert root_spans, "Missing root invocation span (*.invoke). D9 invariant failed."
@@
-# Collect child LLM span IDs
-llm_span_ids = set()
-for nid, n in tgj_nodes.items():
-    if n.get("kind") == "msg" and "openai" in n.get("name", ""):
-        otel_info = (n.get("info") or {}).get("otel", {})
-        llm_span_ids.add(otel_info.get("span_id"))
+# Collect child spans using temporal_ignore marker (D10)
+llm_span_ids = set()
+for nid, n in tgj_nodes.items():
+    otel_info = (n.get("info") or {}).get("otel", {})
+    if str(otel_info.get("temporal_ignore", "false")).lower() in ("true","1","yes"):
+        llm_span_ids.add(otel_info.get("span_id"))
```

---

## Approach B (robust): use the runtime’s configured span name + validate topology systematically

```diff
diff --git a/examples/notebooks/01_m1_instrument_and_optimize.ipynb b/examples/notebooks/01_m1_instrument_and_optimize.ipynb
--- a/examples/notebooks/01_m1_instrument_and_optimize.ipynb
+++ b/examples/notebooks/01_m1_instrument_and_optimize.ipynb
@@ -1,10 +1,44 @@
+def validate_trace_invariants(otlp, tgj_doc, service_name):
+    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
+    names = [s["name"] for s in spans]
+    assert any(n.endswith(".invoke") for n in names), "Missing root span (*.invoke)"
+
+    # Root must be parent of node spans (not necessarily of child spans)
+    root = next(s for s in spans if s["name"].endswith(".invoke"))
+    root_id = root["spanId"]
+    node_spans = [s for s in spans if s["name"] in ("planner","synthesizer")]
+    assert all(s.get("parentSpanId") == root_id for s in node_spans), "Node spans not parented by root"
+
+    # TGJ chaining must not use child spans
+    nodes = tgj_doc["nodes"]
+    child_ids = set()
+    for _, n in nodes.items():
+        otel = (n.get("info") or {}).get("otel", {})
+        if str(otel.get("temporal_ignore","false")).lower() in ("true","1","yes"):
+            child_ids.add(otel.get("span_id"))
+    synth = next(n for n in nodes.values() if n.get("kind")=="msg" and n.get("name")=="synthesizer")
+    parent_ref = (synth.get("inputs") or {}).get("parent","")
+    parent_id = parent_ref.split(":")[1] if ":" in parent_ref else ""
+    assert parent_id and parent_id not in child_ids, "Temporal parent incorrectly points to child span"
+
+validate_trace_invariants(otlp, docs[0], "m1-notebook")
```

---

# Quick validation checklist (what to re-run / verify after applying these)

* OTLP contains **root `*.invoke` span** and node spans are children (D9).
* TGJ conversion uses `trace.temporal_ignore` and chaining does not use child spans (D10).
* Baseline stub score is **< 1.0** and optimization **improves score** (F13).
* `OptimizationResult.best_updates` matches the iteration that achieved `best_score`.
* `optimize_graph` uses `graph.output_key` automatically unless overridden.
