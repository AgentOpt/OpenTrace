Got it — we’ll implement **Strategy 2 (Trace-first, dual semconv)** so that your LangGraph demo emits OTEL spans that are:

* **Agent Lightning–compatible** (via `gen_ai.*` + `agentlightning.reward.*` spans) ([microsoft.github.io][1])
* **Still convertible to TGJ** using your existing `otlp_traces_to_trace_json` adapter (and still optimizable by Trace).

Below I provide:

1. A **side-by-side API comparison table** (New API vs legacy demo vs design3_4 vs Agent Lightning)
2. **Colored code diffs** (Agent Lightning “ExampleAgentLightning.svg” vs New API, and Deep Research agent demo vs New API)
3. A **single unified git diff** against your current codebase (`Trace_main_code.txt` aggregated)
4. A **short explanation you can send to a colleague**

---

## 1) “Per API element” table (New Strategy 2 vs legacy vs design3_4 vs Agent Lightning)

> Goal: show *where* the code lives and *what it looks like*, for each “API element”.

Legend:

* ✅ built-in / intended
* 🟡 possible but manual
* ❌ not present

| API element                                    | **New API (Strategy 2)**                                                                                        | **Legacy demo** `JSON_OTEL_trace_optim_demo_LANGGRAPH.py`                                               | **design3_4 demo** `...DESIGN3_4.py`                                                | **Agent Lightning**                                                                                                                                           |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tracer + exporter init                         | `init_otel_runtime()` (Trace IO runtime)                                                                        | Inline OTEL exporter + provider in demo                                                                 | `init_otel_runtime()` from runtime and rebinding base tracer                        | Uses OTEL tracer/processoinfra; you write spans normally ([microsoft.github.io][2])                                                                           |
| Node span creation                             | Node functions use `TRACER.start_as_current_span("node")` *or* `TracingLLM.node_call(span_name="planner", ...)` | Manual `TRACER.start_as_current_span(...)` all over nodes                                               | Base nodes call `TRACING_LLM.node_call(...)` (Design 3)                             | `@rollo create “agent ops”, plus normal OTEL spans ([microsoft.github.io][3])                                                                                 |
| Prompt parameter capture (Trace optimization)  | **Still**: `param.<name>` + `param.<name>.trainable` on node span (same as today)                               | Manual `sp.set_attribute("param.*", ...)` per node                                                      | Centralized in `TracingLLM._record_llm_call()` in runtime (Design 3)                | Uses **resources** / configs for prompt templates; tources ([GitHub][4])                                                                                      |
| LLM tracing (fine-grained, AL-compatible)      | `TracingLLM.node_call()` automatically emits **child span** named `openai.chat.completion` carrying `gen_ai.*`  | LLM call happens inside node span; only `gen_ai.model` + `inputs.gen_ai.prompt` manually (non-standard) | Uses runtime `TracingLLM` but previously did not guarantee `gen_ai.*`; we’ll add it | Auto instrumentation/proxy creates spans like `openai.chat.completion` and training extracts from `gen_ai.*` ([microsoft.github.io][5])search7turn0search16 |
| **Problem**: temporal hierarchy TGJ conversion | With child spans, you must avoid “child span becomes prev span” (we’ll fix with `trace.temporal_ignore`)        | No child spans → not an issue                                                                           | Not previously emitting child gen-ai spans → not an issue                           | Not TGJ-based; they store spans with their own sequencing logic ([microsoft.github.io][2])                                                                    |
| Evaluation extraction for optimization         | `extract_eval_metrics_from_otlp()` stays (Design 4) and becomes type-robust                                     | Ad-hoc parser loop over OTLP spans                                                                      | Uses `extract_eval_metrics_from_otlp()` already                                     | Uses reward/annotation emitters like `emit_reward()` ([microsoft.github.io][6])                                                                               |
| Reward emission (AL-compatible)                | Evaluator emits **child span** `agentlightning.annotation` with `agentlightning.reward.0.value`                 | Only `eval.score                                                                                        | Previously only Trace eval attributes (we’ll add AL reward emission in SPANOUTNODE) | `emit_reward(value: float)` creates reward spans (wrapper around annotation) ([microsoft.github.io][6])                                                       |
| “One-liner” set attributes                     | `set_span_attributes(span, {...})` helper (new)                                                                 | manual `sp.set_attribute()` repeated                                                                    | runtime already centralized + we add helper                                         | `emit_annotation({..([microsoft.github.io][6])                                                                                                                |
| Optimization loop                              | unchanged: `optimize_iteration(runs, ...)` and TGJ conversion via `otlp_traces_to_trace_json`                   | same                                                                                                    | same (design34 calls base’s `optimize_iteration`)                                   | Training loop is RL/APO/SFT (Trainer) rather than “patch prompts/code” ([microsoft.github.io][3])                                                             |

---

## 2) Colored code comparisons (Agent Lightning vs New API, and Deep Research demo vs New API)

### 2.A Agent Lightning “reference example” (from docs + your SVG) vs New API

Agent Lightning’s docs show: write an agent (often `@rollout`) and emit rewards via emitters; training is done via a `Trainer` and algorithm (e.g., APO). ([microsoft.github.io][7])

Here’s the conceptual diff:

```diff
# --------------------------
# Agent Lightning (concept)
# --------------------------
+ import agentlightning as agl
+ from agentlightning import emit_reward
+ from agentlightning import rollout
+
+ @rollout
+ def agent(task: dict, prompt_template: str):
+     # ... call LLM / tools ...
+     # compute intermediate/final reward
+     emit_reward(0.82)
+     return result
+
+ trainer = agl.Trainer(algorithm=agl.APO(), initial_resources={"prompt_template": prompt_template})
+ trainer.fit(agent=agent, train_dataset=tasks)


# --------------------------
# Trace New API (Strategy 2)
# --------------------------
+ from opto.trace.io.langgraph_otel_runtime import init_otel_runtime, TracingLLM
+ from opto.trace.io.otel_semconv import emit_agentlightning_reward  # reward span format
+
+ TRACER, EXPORTER = init_otel_runtime("my-graph")
+ TRACING_LLM = TracingLLM(llm=LLM_CLIENT, tracer=TRACER, trainable_keys={"planner","executor"})
+
+ def planner_node(state):
+     # no manual OTEL + gen_ai work; wrapper does it
+     plan = TRACING_LLM.node_call(
+         span_name="planner",
+         template_name="planner_prompt",
+         template=state.planner_template,
+         optimizable_key="planner",
+         messages=[...],
+     )
+     return {...}
+
+ def evaluator_node(state):
+     with TRACER.start_as_current_span("evaluator") as sp:
+         # produce Trace eval attrs (as before)
+         sp.set_attribute("eval.score", score)
+         ...
+         # AND ALSO produce Agent Lightning compatible reward span:
+         emit_agentlightning_reward(value=float(score), name="final_score")
```

Key point: **Strategy 2 does not try to reproduce RL training**. It only emits spans **compatible** with Lightning’s expectations while keeping your **TGJ/OPTO patch optimization** intact.

---

### 2.B Deep Research agent: Legacy demo vs design3_4 vs New API (Strategy 2)

In the legacy demo you manually set the prompt parameters + prompt input + `gen_ai.model` inside each node span.
In design3_4, those responsibilities move into the shared runtime `TracingLLM`.

This is the “core simplification” you already did:

```diff
# Legacy demo (manual OTEL inside each node)
  with TRACER.start_as_current_span("synthesizer") as sp:
      sp.set_attribute("param.synthesizer_prompt", template)
      sp.set_attribute("param.synthesizer_prompt.trainable", "synthesizer" in OPTIMIZABLE)
-     sp.set_attribute("gen_ai.model", "llm")
      sp.set_attribute("inputs.gen_ai.prompt", prompt)
      _emit_code_param(sp, "synthesizer", synthesizer_node)
      answer = LLM_CLIENT(messages=[...]).:contentReference[oaicite:29]{index=29}tent

# design3_4 + New API (wrapper)
+ answer = TRACING_LLM.node_call(
+     span_name="synthesizer",
+     template_name="synthesizer_prompt",
+     template=template,
+     optimizable_key="synthesizer",
+     code_key="synthesizer",
+     code_fn=synthesizer_node,
+     user_query=state.user_query,
+     messages=[{"role":"system","content":"..."}, {"role":"user","content":prompt}],
+ )
```

What Strategy 2 adds **on top** of design3_4:

* the wrapper emits a **child LLM span** named `openai.chat.completion` with `gen_ai.*` attributes (Lightning-friendly) ([OpenTelemetry][8])
* evaluator emits a **child reward span** `agentlightning.annotation` with `agentlightning.reward.*` attributes ([microsoft.github.io][1])
* we prevent these child spans from breaking TGJ “temporal hierarchy” conversion by marking them `trace.temporal_ignore=true` and teaching `otel_adapter` not to advance `prev_span_id` on them.

---

## 3) Unified git diff to apply (against current codebase from `Trace_main_code.txt`)

This patch adds **one helper module**, updates the runtime `TracingLLM`, updates `otel_adapter` for temporal-ignore safety, and updates the SPANOUTNODE evaluator to emit Agent Lightning rewards.

> ✅ This is minimal and should not break legacy demos.
> ✅ It keeps TGJ conversion stable even with child spans.

```diff
diff --git a/opto/trace/io/__init__.py b/opto/trace/io/__init__.py
index e69de29..7b9c3a1 100644
--- a/opto/trace/io/__init__.py
+++ b/opto/trace/io/__init__.py
@@ -0,0 +1,9 @@
+from .otel_semconv import (
+    set_span_attributes,
+    record_genai_chat,
+    emit_agentlightning_reward,
+)
+
+__all__ = [
+    "set_span_attributes", "record_genai_chat", "emit_agentlightning_reward",
+]

diff --git a/opto/trace/io/otel_semconv.py b/opto/trace/io/otel_semconv.py
new file mode 100644
index 0000000..b1a2c3d
--- /dev/null
+++ b/opto/trace/io/otel_semconv.py
@@ -0,0 +1,176 @@
+from __future__ import annotations
+
+import json
+from typing import Any, Dict, List, Optional
+
+from opentelemetry import trace as oteltrace
+
+
+def _json(v: Any) -> str:
+    return json.dumps(v, ensure_ascii=False)
+
+
+def set_span_attributes(span, attrs: Dict[str, Any]) -> None:
+    """
+    Convenience helper: set many span attributes at once.
+    - dict/list -> JSON string
+    - None values -> skipped
+    """
+    for k, v in (attrs or {}).items():
+        if v is None:
+            continue
+        if isinstance(v, (dict, list)):
+            span.set_attribute(k, _json(v))
+        else:
+            span.set_attribute(k, v)
+
+
+def record_genai_chat(
+    span,
+    *,
+    provider: str,
+    model: str,
+    input_messages: List[Dict[str, Any]],
+    output_text: Optional[str] = None,
+    request_type_compat: str = "chat.completion",
+) -> None:
+    """
+    Record OTEL GenAI semantic convention attributes in a span.
+
+    We store messages as JSON strings (span attrs must be primitive/sequence types).
+    """
+    out_messages = None
+    if output_text is not None:
+        out_messages = [{"role": "assistant", "content": output_text}]
+
+    set_span_attributes(
+        span,
+        {
+            # Spec-ish keys that many adapters expect
+            "gen_ai.operation.name": "chat",
+            "gen_ai.provider.name": provider,
+            "gen_ai.request.model": model,
+            # Back-compat / convenience for other tools (and Trace's existing heuristics)
+            "gen_ai.operation": "chat",
+            "gen_ai.model": model,
+            "gen_ai.request.type": request_type_compat,
+            # We keep these as JSON strings
+            "gen_ai.input.messages": input_messages,
+            "gen_ai.output.messages": out_messages,
+        },
+    )
+
+
+def emit_agentlightning_reward(
+    *,
+    value: float,
+    name: str = "final_score",
+    tracer_name: str = "opto.trace",
+    index: int = 0,
+    span_name: str = "agentlightning.annotation",
+    temporal_ignore: bool = True,
+    extra_attributes: Optional[Dict[str, Any]] = None,
+) -> None:
+    """
+    Emit a reward span compatible with Agent Lightning semconv.
+
+    Docs: emit_reward is a wrapper of emit_annotation; reward attrs use
+    agentlightning.reward.<i>.name / agentlightning.reward.<i>.value. :contentReference[oaicite:32]{index=32}
+    """
+    tracer = oteltrace.get_tracer(tracer_name)
+    with tracer.start_as_current_span(span_name) as sp:
+        attrs: Dict[str, Any] = {
+            f"agentlightning.reward.{index}.name": name,
+            f"agentlightning.reward.{index}.value": float(value),
+        }
+        if temporal_ignore:
+            attrs["trace.temporal_ignore"] = True
+        if extra_attributes:
+            attrs.update(extra_attributes)
+        set_span_attributes(sp, attrs)

diff --git a/opto/trace/io/langgraph_otel_runtime.py b/opto/trace/io/langgraph_otel_runtime.py
index 4f3aa11..c0f77df 100644
--- a/opto/trace/io/langgraph_otel_runtime.py
+++ b/opto/trace/io/langgraph_otel_runtime.py
@@ -1,9 +1,11 @@
 from __future__ import annotations
 
+import json
 import time
 from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
 
 from opentelemetry import trace as oteltrace
 from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
 from opentelemetry.sdk.trace.export import (
     SimpleSpanProcessor,
     SpanExporter,
     SpanExportResult,
 )
+
+from .otel_semconv import record_genai_chat, set_span_attributes
 
 
 class InMemorySpanExporter(SpanExporter):
@@ -56,6 +58,22 @@ def init_otel_runtime(
     tracer = provider.get_tracer(service_name)
     return tracer, exporter
 
 
+def _to_otlp_anyvalue(v: Any) -> Dict[str, Any]:
+    """
+    Encode a Python attr into an OTLP JSON AnyValue.
+    Keep it simple/robust: primitives keep type; everything else stringified.
+    """
+    if isinstance(v, bool):
+        return {"boolValue": v}
+    if isinstance(v, int) and not isinstance(v, bool):
+        # OTLP JSON commonly uses strings for intValue
+        return {"intValue": str(v)}
+    if isinstance(v, float):
+        return {"doubleValue": float(v)}
+    if isinstance(v, str):
+        return {"stringValue": v}
+    return {"stringValue": str(v)}
+
+
 def flush_otlp(
     exporter: InMemorySpanExporter,
     scope_name: str = "demo",
@@ -78,10 +96,10 @@ def flush_otlp(
     otlp_spans: List[Dict[str, Any]] = []
     for s in spans:
         attributes = getattr(s, "attributes", {}) or {}
         attrs = [
-            {"key": k, "value": {"stringValue": str(v)}}
+            {"key": k, "value": _to_otlp_anyvalue(v)}
             for k, v in attributes.items()
         ]
         kind = getattr(s, "kind", 1)
         if hasattr(kind, "value"):
@@ -121,6 +139,26 @@ def flush_otlp(
     }
 
 
 class TracingLLM:
@@ -137,6 +175,10 @@ class TracingLLM:
     def __init__(
         self,
         llm: Any,
         tracer: oteltrace.Tracer,
         *,
         trainable_keys: Optional[Iterable[str]] = None,
         emit_code_param: Optional[Any] = None,
+        provider_name: str = "openai",
+        llm_span_name: str = "openai.chat.completion",
+        emit_llm_child_span: bool = True,
     ) -> None:
         self.llm = llm
         self.tracer = tracer
         self.trainable_keys = set(trainable_keys or [])
         self.emit_code_param = emit_code_param
+        self.provider_name = provider_name
+        self.llm_span_name = llm_span_name
+        self.emit_llm_child_span = emit_llm_child_span
 
     # ---- helpers ---------------------------------------------------------
@@ -166,8 +208,8 @@ class TracingLLM:
         if code_key and code_fn is not None and self.emit_code_param:
             self.emit_code_param(sp, code_key, code_fn)
 
-        sp.set_attribute("gen_ai.model", "llm")
+        # Keep Trace-style prompt capture on the node span (TGJ-friendly).
         sp.set_attribute("inputs.gen_ai.prompt", prompt)
         if user_query is not None:
             sp.set_attribute("inputs.user_query", user_query)
@@ -186,6 +228,17 @@ class TracingLLM:
         """
         Invoke the wrapped LLM under an OTEL span.
         """
         with self.tracer.start_as_current_span(span_name) as sp:
             prompt = ""
             if messages:
                 user_msgs = [m for m in messages if m.get("role") == "user"]
                 if user_msgs:
                     prompt = user_msgs[-1].get("content", "") or ""
                 else:
                     prompt = messages[-1].get("content", "") or ""
 
             self._record_llm_call(
                 sp,
                 template_name=template_name,
                 template=template,
                 optimizable_key=optimizable_key,
                 code_key=code_key,
                 code_fn=code_fn,
                 user_query=user_query,
                 prompt=prompt,
                 extra_inputs=extra_inputs or {},
             )
-
-            resp = self.llm(messages=messages, **llm_kwargs)
-            # Compatible with OpenAI-style chat responses.
-            return resp.choices[0].message.content
+            # Infer model name best-effort.
+            model = (
+                str(llm_kwargs.get("model"))
+                if llm_kwargs.get("model") is not None
+                else str(getattr(self.llm, "model", "") or "unknown")
+            )
+
+            # Emit a child span that looks like common GenAI client spans.
+            # Important: mark it temporal-ignore so TGJ temporal parenting stays stable.
+            if self.emit_llm_child_span:
+                with self.tracer.start_as_current_span(self.llm_span_name) as llm_sp:
+                    set_span_attributes(llm_sp, {"trace.temporal_ignore": True})
+                    # record request-side gen_ai.* first
+                    record_genai_chat(
+                        llm_sp,
+                        provider=self.provider_name,
+                        model=model,
+                        input_messages=messages or [],
+                        output_text=None,
+                    )
+                    resp = self.llm(messages=messages, **llm_kwargs)
+                    text = resp.choices[0].message.content
+                    # now attach response-side gen_ai.*
+                    record_genai_chat(
+                        llm_sp,
+                        provider=self.provider_name,
+                        model=model,
+                        input_messages=messages or [],
+                        output_text=text,
+                    )
+                    return text
+
+            # Fallback: no child span; just call LLM.
+            resp = self.llm(messages=messages, **llm_kwargs)
+            return resp.choices[0].message.content
 
 
 DEFAULT_EVAL_METRIC_KEYS: Mapping[str, str] = {
@@ -198,15 +251,31 @@ DEFAULT_EVAL_METRIC_KEYS: Mapping[str, str] = {
 }
 
 
-def _attrs_to_dict(attrs: List[Dict[str, Any]]) -> Dict[str, str]:
+def _anyvalue_to_py(v: Any) -> Any:
+    if not isinstance(v, dict) or not v:
+        return v
+    if "stringValue" in v:
+        return v["stringValue"]
+    if "doubleValue" in v:
+        return v["doubleValue"]
+    if "intValue" in v:
+        try:
+            return int(v["intValue"])
+        except Exception:
+            return v["intValue"]
+    if "boolValue" in v:
+        return bool(v["boolValue"])
+    # arrays/kvlist unsupported here; stringify
+    return str(v)
+
+
+def _attrs_to_dict(attrs: List[Dict[str, Any]]) -> Dict[str, Any]:
     out: Dict[str, str] = {}
     for a in attrs or []:
         key = a.get("key")
-        val = a.get("value", {})
+        val = a.get("value", {})
         if key is None:
             continue
-        if isinstance(val, dict) and "stringValue" in val:
-            out[key] = val["stringValue"]
-        else:
-            out[key] = str(val)
+        out[key] = _anyvalue_to_py(val)
     return out
 
 
 def extract_eval_metrics_from_otlp(
@@ -241,7 +310,7 @@ def extract_eval_metrics_from_otlp(
                 if sp.get("name") != evaluator_span_name:
                     continue
                 attrs = _attrs_to_dict(sp.get("attributes", []))
                 raw_score = attrs.get(score_key)
                 if raw_score is not None:
                     try:
                         score = float(raw_score)
                     except ValueError:
                         score = default_score
                 reasons = attrs.get("eval.reasons", "") or ""
@@ -252,7 +321,7 @@ def extract_eval_metrics_from_otlp(
                     raw = attrs.get(attr_key)
                     if raw is None:
                         continue
                     try:
                         metrics[friendly] = float(raw)
                     except ValueError:
                         metrics[friendly] = default_metric
diff --git a/opto/trace/io/otel_adapter.py b/opto/trace/io/otel_adapter.py
index 1c0d111..2b7e222 100644
--- a/opto/trace/io/otel_adapter.py
+++ b/opto/trace/io/otel_adapter.py
@@ -1,6 +1,7 @@
 from __future__ import annotations
 from typing import Dict, Any, List
 
 
 PROFILE_VERSION = "trace-json/1.0+otel"
@@ -10,6 +11,14 @@ def _sanitize(name: str) -> str:
     return (name or "node").replace(":", "_")
 
+def _truthy(v: Any) -> bool:
+    if isinstance(v, bool):
+        return v
+    if isinstance(v, (int, float)):
+        return v != 0
+    if isinstance(v, str):
+        return v.strip().lower() in ("1", "true", "yes", "y", "on")
+    return bool(v)
 
 def _op(attrs, span):
     if "gen_ai.operation" in attrs or "gen_ai.model" in attrs:
         return "llm_call"
@@ -109,8 +118,12 @@ def otlp_traces_to_trace_json(otlp: Dict[str, Any], agent_id_hint: str = "", use_temporal_hierarchy: bool = False) -> List[Dict[str, Any]]:
                 node_id = f"{svc}:{sid}"
                 nodes[node_id] = rec
                 
-                # Update prev_span_id for next iteration (temporal parenting)
-                prev_span_id = sid
+                # Update prev_span_id for next iteration (temporal parenting).
+                # If a span is marked "temporal_ignore", don't let it become the sequential parent.
+                if not _truthy(attrs.get("trace.temporal_ignore")):
+                    prev_span_id = sid
 
             docs.append(
                 {
diff --git a/JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE.py b/JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE.py
index 9abc111..9abc222 100644
--- a/JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE.py
+++ b/JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE.py
@@ -1,6 +1,7 @@
 ...
+from opto.trace.io.otel_semconv import emit_agentlightning_reward
 ...
 def evaluator_node(state: State) -> Command[Literal[END]]:
     """
     Evaluator node with multi-metric assessment.
     """
@@ -40,6 +41,12 @@ def evaluator_node(state: State) -> Command[Literal[END]]:
         for k, v in metrics.items():
             sp.set_attribute(f"eval.{k}", str(v))
         sp.set_attribute("eval.score", str(score))
         sp.set_attribute("eval.reasons", reasons)
         _emit_code_param(sp, "evaluator", evaluator_node)
+
+        # Also emit an Agent Lightning compatible reward span as a child.
+        # (This is just OTEL; safe even if Agent Lightning isn't installed.) :contentReference[oaicite:33]{index=33}
+        emit_agentlightning_reward(value=float(score), name="final_score")
 
     feedback = f"[Metrics] {list(metrics.values())} ; Reasons: {reasons}"
 
diff --git a/tests/test_dual_semconv.py b/tests/test_dual_semconv.py
new file mode 100644
index 0000000..ddee111
--- /dev/null
+++ b/tests/test_dual_semconv.py
@@ -0,0 +1,148 @@
+from __future__ import annotations
+
+from typing import Any
+
+from opto.trace.io.langgraph_otel_runtime import init_otel_runtime, TracingLLM, flush_otlp
+from opto.trace.io.otel_adapter import otlp_traces_to_trace_json
+
+
+class _DummyResp:
+    def __init__(self, txt: str):
+        self.choices = [type("C", (), {"message": type("M", (), {"content": txt})()})()]
+
+
+class DummyLLM:
+    def __call__(self, messages=None, **kwargs):
+        return _DummyResp("ok")
+
+
+def _find_span(otlp: dict, name: str) -> dict | None:
+    for rs in otlp.get("resourceSpans", []):
+        for ss in rs.get("scopeSpans", []):
+            for sp in ss.get("spans", []):
+                if sp.get("name") == name:
+                    return sp
+    return None
+
+
+def _span_attrs(sp: dict) -> dict:
+    out = {}
+    for a in sp.get("attributes", []) or []:
+        k = a.get("key")
+        v = a.get("value", {}) or {}
+        # pick first value variant
+        if isinstance(v, dict) and v:
+            out[k] = next(iter(v.values()))
+        else:
+            out[k] = v
+    return out
+
+
+def test_tracingllm_emits_child_genai_span_and_temporal_ignore():
+    tracer, exporter = init_otel_runtime("test-dual-semconv")
+    llm = DummyLLM()
+    tl = TracingLLM(
+        llm=llm,
+        tracer=tracer,
+        trainable_keys={"planner"},
+        provider_name="openai",
+        llm_span_name="openai.chat.completion",
+        emit_llm_child_span=True,
+    )
+
+    out = tl.node_call(
+        span_name="planner",
+        template_name="planner_prompt",
+        template="Hello {x}",
+        optimizable_key="planner",
+        messages=[{"role": "user", "content": "hi"}],
+    )
+    assert out == "ok"
+
+    otlp = flush_otlp(exporter, scope_name="test")
+
+    node_sp = _find_span(otlp, "planner")
+    llm_sp = _find_span(otlp, "openai.chat.completion")
+    assert node_sp is not None
+    assert llm_sp is not None
+
+    llm_attrs = _span_attrs(llm_sp)
+    assert llm_attrs.get("trace.temporal_ignore") in (True, "true", "True", 1, "1")
+    assert llm_attrs.get("gen_ai.operation") == "chat"
+    assert llm_attrs.get("gen_ai.provider.name") == "openai"
+
+
+def test_otel_adapter_temporal_hierarchy_ignores_child_spans():
+    # Build a minimal OTLP payload with:
+    # - A (t=1)
+    # - child C (t=2, parentSpanId=A, trace.temporal_ignore=true)
+    # - B (t=3, no parentSpanId)  -> should attach to A (not C) under temporal hierarchy
+    otlp = {
+        "resourceSpans": [
+            {
+                "resource": {"attributes": []},
+                "scopeSpans": [
+                    {
+                        "scope": {"name": "x"},
+                        "spans": [
+                            {
+                                "traceId": "t",
+                                "spanId": "A",
+                                "parentSpanId": "",
+                                "name": "A",
+                                "kind": "INTERNAL",
+                                "startTimeUnixNano": 1,
+                                "endTimeUnixNano": 1,
+                                "attributes": [],
+                            },
+                            {
+                                "traceId": "t",
+                                "spanId": "C",
+                                "parentSpanId": "A",
+                                "name": "openai.chat.completion",
+                                "kind": "CLIENT",
+                                "startTimeUnixNano": 2,
+                                "endTimeUnixNano": 2,
+                                "attributes": [
+                                    {"key": "trace.temporal_ignore", "value": {"boolValue": True}},
+                                    {"key": "gen_ai.operation", "value": {"stringValue": "chat"}},
+                                ],
+                            },
+                            {
+                                "traceId": "t",
+                                "spanId": "B",
+                                "parentSpanId": "",
+                                "name": "B",
+                                "kind": "INTERNAL",
+                                "startTimeUnixNano": 3,
+                                "endTimeUnixNano": 3,
+                                "attributes": [],
+                            },
+                        ],
+                    }
+                ],
+            }
+        ]
+    }
+
+    docs = otlp_traces_to_trace_json(otlp, agent_id_hint="svc", use_temporal_hierarchy=True)
+    assert len(docs) == 1
+    nodes = docs[0]["nodes"]
+
+    # Node IDs are svc:<spanId>
+    b = nodes["svc:B"]
+    assert b["inputs"]["parent"] == "svc:A"
```

---

## 4) What changed / how to explain it to a colleague (short + clear)

**Problem:** We want a “minimal-intrusion” LangGraph optimization demo where Trace can optimize prompts/code/graph using TGJ, but the emitted OTEL spans are also compatible with Agent Lightning tooling (so future RL/APO pipelines can reuse the same traces).

**What we did (Strategy 2):**

1. **Keep Trace as the primary instrumentation and optimization system** (TGJ conversion and OPTO optimizers unchanged).

2. **Emit OTEL spans using two “schemas” at once**:

   * Trace-specific attrs remain: `param.*`, `inputs.gen_ai.prompt`, `eval.*`
   * AgentLightning-compatible spans are added:

     * each LLM call produces a child span named `openai.chat.completion` with `gen_ai.*` fields ([OpenTelemetry][8])
     * evaluator produces a child span named `agentlightning.annotation` with `agentlightning.reward.0.value` ([microsoft.github.io][6])

3. **Prevent a subtle TGJ bug**: TGJ conversion with `use_temporal_hierarchy=True` links spans in time order. If we introduce child spans, they could accidentally become the “previous span” and break the main chain.
   → Fix: child spans are marked `trace.temporal_ignore=true`, and `otel_adapter` ignores them when advancing `prev_span_id`.

**Result:**

* The demo stays clean (no repeated OTEL boilerplate).
* Trace can still optimize prompts/code.
* The same run also yields “Agent Lightning-shaped” traces for future reuse.

---

If you want, next step is: I can add a small “How to run in Agent Lightning later” note: i.e., export your OTLP to their store or proxy, and confirm which subset of `gen_ai.*` attributes their adapters require (but with this patch you’re already aligned with the standard span names and reward encoding).

[1]: https://microsoft.github.io/agent-lightning/stable/reference/semconv/ "https://microsoft.github.io/agent-lightning/stable/reference/semconv/"
[2]: https://microsoft.github.io/agent-lightning/latest/tutorials/traces/ "https://microsoft.github.io/agent-lightning/latest/tutorials/traces/"
[3]: https://microsoft.github.io/agent-lightning/latest/reference/agent/ "https://microsoft.github.io/agent-lightning/latest/reference/agent/"
[4]: https://github.com/microsoft/agent-lightning "https://github.com/microsoft/agent-lightning"
[5]: https://microsoft.github.io/agent-lightning/stable/reference/algorithm/ "https://microsoft.github.io/agent-lightning/stable/reference/algorithm/"
[6]: https://microsoft.github.io/agent-lightning/stable/tutorials/emitter/ "https://microsoft.github.io/agent-lightning/stable/tutorials/emitter/"
[7]: https://microsoft.github.io/agent-lightning/latest/tutorials/write-agents/ "https://microsoft.github.io/agent-lightning/latest/tutorials/write-agents/"
[8]: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/ "https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/"
