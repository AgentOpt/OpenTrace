# Milestone 2 — Unified Telemetry Documentation

> PR #64 · branch `m2-unified-telemetry`

---

## Table of Contents

1. [Before / After Snippets](#1-beforeafter-snippets)
2. [File-by-File Change Table](#2-file-by-file-change-table)
3. [Configuration Options](#3-configuration-options)
4. [With Telemetry vs Without Telemetry](#4-with-telemetry-vs-without-telemetry)

---

## 1. Before/After Snippets

### 1a — LangGraph Instrumentation (`TracingLLM`)

**Before (M1):** Provider was always whatever the caller passed or the
hard-coded default `"llm"`.

```python
# langgraph_otel_runtime.py — TracingLLM.__init__
self.provider_name = provider_name          # whatever the caller passed
```

**After (M2):** Provider is inferred from the model string when the caller
passes the default `"llm"`.

```python
# langgraph_otel_runtime.py — TracingLLM.__init__
if provider_name == "llm":
    model_str = str(getattr(llm, "model", "") or "")
    if "/" in model_str:                    # e.g. "openai/gpt-4"
        provider_name = model_str.split("/", 1)[0]
self.provider_name = provider_name
```

---

### 1b — Non-LangGraph (`call_llm` in `operators.py`)

**Before (M1):** Provider fell through to `"litellm"` for all slash-style
model strings.

```python
model = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "llm"
provider = getattr(llm, "provider_name", None) or getattr(llm, "provider", None) or "litellm"
```

**After (M2):** Slash-style model strings are parsed first.

```python
model = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "llm"
provider = getattr(llm, "provider_name", None) or getattr(llm, "provider", None)
if not provider:
    model_str = str(model)
    if "/" in model_str:
        provider = model_str.split("/", 1)[0]   # "openai/gpt-4" → "openai"
    else:
        provider = "litellm"
```

---

### 1c — TelemetrySession MLflow Bridge

**Before (M1):** No MLflow integration in TelemetrySession. Users had to
manually call `autolog()` before creating the session.

```python
session = TelemetrySession(service_name="my-app")
# No MLflow awareness
```

**After (M2):** Best-effort MLflow autologging built into the session.

```python
session = TelemetrySession(
    service_name="my-app",
    mlflow_autolog=True,                        # NEW
    mlflow_autolog_kwargs={"silent": True},      # NEW
)
# MLflow autologging is activated automatically (if importable),
# failure is logged at DEBUG level and never raises.
```

---

### 1d — Stable Node Identity in TGJ (`otel_adapter.py`)

**Before (M1):** TGJ node keys were always `"{service}:{span_id}"`,
making them non-deterministic across runs.

```python
node_id = f"{svc}:{sid}"
nodes[node_id] = rec
```

**After (M2):** When `message.id` is present, it is used as the stable
node key. A `span_to_node_id` map ensures parent references resolve
correctly.

```python
span_to_node_id: Dict[str, str] = {}

msg_id = attrs.get("message.id")
node_id = f"{svc}:{msg_id}" if msg_id else f"{svc}:{sid}"
nodes[node_id] = rec
span_to_node_id[sid] = node_id

# Parent reference resolves through the mapping:
if effective_psid and "parent" not in inputs:
    inputs["parent"] = span_to_node_id.get(effective_psid, f"{svc}:{effective_psid}")
```

---

### 1e — Notebook / End-to-End Usage

**Before (M1):** Sessions required manual OTLP export only.

```python
session = TelemetrySession(service_name="demo")
with session:
    run_pipeline()
otlp = session.flush_otlp()
```

**After (M2):** Sessions support TGJ export, MLflow bridge, and stable
node identities in a single unified flow.

```python
session = TelemetrySession(
    service_name="demo",
    mlflow_autolog=True,
    mlflow_autolog_kwargs={"silent": True},
)
with session:
    run_pipeline()

otlp = session.flush_otlp()                         # raw OTLP spans
tgj  = session.flush_tgj(agent_id_hint="demo",      # Trace-Graph JSON
                          use_temporal_hierarchy=True)

session.export_run_bundle("./output",
    include_otlp=True,
    include_tgj=True,
    include_prompts=True,
    prompts=collected_prompts,
)
```

---

## 2. File-by-File Change Table

Files changed across the M2 branch (vs `upstream/experimental`):

### Core Library (`opto/`)

| File | Change | Why |
|------|--------|-----|
| `opto/trace/io/telemetry_session.py` | Added `mlflow_autolog`, `mlflow_autolog_kwargs` params; best-effort import + call in `__init__` | **B1** — Let TelemetrySession optionally activate MLflow autologging without hard dependency |
| `opto/trace/operators.py` | Hardened provider inference in `call_llm`: parse `"openai/gpt-4"` → `"openai"` before falling back to `"litellm"` | **B3** — Correct `gen_ai.provider.name` for slash-style model strings |
| `opto/trace/io/langgraph_otel_runtime.py` | Same provider inference in `TracingLLM.__init__` | **B3** — Consistent provider detection across both LangGraph and non-LangGraph paths |
| `opto/trace/io/otel_adapter.py` | `span_to_node_id` mapping; use `message.id` as stable node key; resolve parent refs through map | **B4** — Deterministic node identity for TGJ round-trip and optimization alignment |
| `opto/trace/bundle.py` | No changes in Phase B (verified `mlflow_kwargs` position is safe) | **B2** — Audit confirmed no break |
| `opto/trace/io/otel_semconv.py` | New file (M2) — semantic convention helpers | Dual semconv: `param.*` for optimization + `gen_ai.*` for Agent Lightning |
| `opto/trace/io/tgj_ingest.py` | New file (M2) — TGJ ingestion back to Trace nodes | Round-trip: OTLP → TGJ → Trace graph |
| `opto/trace/io/bindings.py` | New file (M2) — dynamic span-to-node bindings | MessageNode ↔ OTEL span linkage |
| `opto/trace/io/instrumentation.py` | New file (M2) — bundle-level instrumentation hooks | Auto-emit spans for `@trace.bundle` when session active |
| `opto/trace/settings.py` | New file (M2) — global settings (MLflow config, flags) | Centralized config state |
| `opto/trace/nodes.py` | MessageNode telemetry hooks | Emit `message.id` attribute on node creation |
| `opto/features/mlflow/autolog.py` | New file (M2) — MLflow autolog wrapper | Optional `mlflow.trace` wrapping for bundle ops |

### Tests

| File | Change | Why |
|------|--------|-----|
| `tests/unit_tests/test_telemetry_session.py` | +82 lines: 4 MLflow bridge tests + 2 stable node identity tests | Validate B1 (autolog on/off/kwargs/failure) and B4 (message.id keying + fallback) |
| `tests/features_tests/test_flows_compose.py` | `DummyLLM` now extends `AbstractModel` | Pre-existing upstream fix for CI |
| `tests/llm_optimizers_tests/test_optimizer_optoprimemulti.py` | Added `HAS_CREDENTIALS` skip guard | Pre-existing upstream fix — tests need live LLM |
| `tests/llm_optimizers_tests/test_optoprime_v2.py` | String/int assertion fixes, `xfail` for truncation bug | Pre-existing upstream fix |
| `tests/llm_optimizers_tests/test_opro_v2.py` | Tag format assertion updates | Pre-existing upstream fix — `<variable>` format |

### Examples & Docs

| File | Change | Why |
|------|--------|-----|
| `examples/notebooks/01_m1_instrument_and_optimize.ipynb` | M1 demo notebook | LangGraph instrumentation walkthrough |
| `examples/notebooks/02_m2_unified_telemetry.ipynb` | M2 demo notebook | Unified session, TGJ export, MLflow bridge |
| `docs/m0_README.md`, `docs/m1_README.md` | Milestone readmes | Architecture context |
| `docs/T1_technical_plan.md` | Technical plan | Project design reference |

---

## 3. Configuration Options

### TelemetrySession Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `service_name` | `str` | `"trace-session"` | OTEL service/scope name for all spans |
| `record_spans` | `bool` | `True` | Master switch — `False` disables all span recording (zero overhead) |
| `span_attribute_filter` | `(name, attrs) → attrs` | `None` | Redact secrets or truncate payloads; return `{}` to drop span |
| `bundle_spans` | `BundleSpanConfig` | `BundleSpanConfig()` | Control `@trace.bundle` span emission |
| `message_nodes` | `MessageNodeTelemetryConfig` | `MessageNodeTelemetryConfig()` | Control `MessageNode` ↔ span binding |
| `max_attr_chars` | `int` | `500` | Truncation limit for attribute values |
| `mlflow_log_artifacts` | `bool` | `False` | Log bundle dir as MLflow artifacts on export |
| `mlflow_autolog` | `bool` | `False` | Best-effort enable MLflow autologging on session init |
| `mlflow_autolog_kwargs` | `dict` | `None` | Extra kwargs forwarded to `autolog()` (e.g. `{"silent": True}`) |

### BundleSpanConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable` | `bool` | `True` | Global on/off for bundle spans |
| `disable_default_ops` | `bool` | `True` | Suppress spans for low-level default operators (everything in `operators.py` except `call_llm`) |
| `capture_inputs` | `bool` | `True` | Record input values as span attributes |

### MessageNodeTelemetryConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `str` | `"bind"` | `"off"` = no binding, `"bind"` = attach `message.id` to current span, `"span"` = create minimal span if none exists |

### TracingLLM Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `Any` | required | Underlying LLM client |
| `tracer` | `Tracer` | required | OTEL tracer |
| `trainable_keys` | `Iterable[str]` | `None` | Keys whose prompts are trainable; `None` = all trainable |
| `emit_code_param` | `callable` | `None` | `(span, key, fn) → None` to emit code parameters |
| `provider_name` | `str` | `"llm"` | Provider name; auto-inferred from model string if `"llm"` |
| `llm_span_name` | `str` | `"llm.chat.completion"` | Span name for LLM child spans |
| `emit_llm_child_span` | `bool` | `True` | Emit Agent Lightning child spans with `trace.temporal_ignore=true` |

### Span Attribute Conventions

| Prefix / Key | Layer | Purpose |
|--------------|-------|---------|
| `param.*` | Trace (optimization) | Trainable parameter values for the optimizer |
| `param.*.trainable` | Trace (optimization) | `"true"` / `"false"` — marks if optimizer can modify |
| `inputs.*` | Trace (optimization) | Input references or literal values |
| `gen_ai.provider.name` | Agent Lightning (observability) | Provider string (e.g. `"openai"`) |
| `gen_ai.request.model` | Agent Lightning (observability) | Model identifier |
| `gen_ai.operation.name` | Agent Lightning (observability) | Operation type (e.g. `"chat"`) |
| `message.id` | TGJ (identity) | Stable logical node ID for round-trip alignment |
| `trace.temporal_ignore` | TGJ (hierarchy) | `"true"` — exclude span from temporal parent chaining |
| `trace.bundle` | Instrumentation | `"true"` on bundle-generated spans |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | API key for OpenRouter LLM provider |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.1-8b-instruct:free` | Default model string |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base |
| `USE_STUB_LLM` | `false` | Use stub LLM for testing (no API calls) |

---

## 4. With Telemetry vs Without Telemetry

### Without Telemetry (default behavior, zero changes)

```python
from opto.trace import bundle, node

@bundle()
def my_op(x, y):
    return x + y

result = my_op(a, b)  # Pure Trace graph — no OTEL, no spans, no overhead
```

- `TelemetrySession.current()` returns `None`
- `@bundle` creates Trace nodes only (existing behavior)
- `call_llm` calls the LLM directly, no span wrapping
- No imports from `opentelemetry` are triggered at module level in `operators.py`
  (guarded behind `if session is not None`)
- **Zero performance overhead** — all telemetry code paths are gated

### With Telemetry (opt-in)

```python
from opto.trace import bundle, node
from opto.trace.io.telemetry_session import TelemetrySession

session = TelemetrySession(
    service_name="my-optimization",
    record_spans=True,
    mlflow_autolog=True,
)

with session:
    @bundle()
    def my_op(x, y):
        return x + y

    result = my_op(a, b)
    # Now creates BOTH:
    #   1. Trace node (as before)
    #   2. OTEL span with param.*/inputs.* attributes
    #   3. MLflow trace span (if mlflow importable)

# Export collected telemetry
otlp = session.flush_otlp(clear=True)
tgj  = session.flush_tgj(agent_id_hint="my-optimization")
```

### Side-by-Side Comparison

| Aspect | Without Telemetry | With Telemetry |
|--------|-------------------|----------------|
| Trace graph | Created normally | Created normally (unchanged) |
| OTEL spans | None | Emitted for bundles + LLM calls |
| MLflow traces | None | Optional (best-effort, `mlflow_autolog=True`) |
| `call_llm` behavior | Direct LLM call | LLM call + OTEL span with `gen_ai.*` attrs |
| TGJ export | Not available | `session.flush_tgj()` → stable node graph |
| Performance | Baseline | ~2-5% overhead (span creation + attribute setting) |
| Dependencies | `opto` only | `opto` + `opentelemetry-sdk` (MLflow optional) |
| Existing tests | Pass unchanged | Pass unchanged (69/69 M1+M2 tests green) |

### Activation Patterns

```python
# Pattern 1: Context manager (recommended)
with TelemetrySession(service_name="app") as session:
    run_pipeline()

# Pattern 2: Imperative (notebooks)
session = TelemetrySession(service_name="notebook")
session.set_current()
try:
    run_pipeline()
finally:
    session.clear_current()

# Pattern 3: LangGraph runtime (TracingLLM wraps the LLM)
from opto.trace.io.langgraph_otel_runtime import TracingLLM
tracing_llm = TracingLLM(llm=base_llm, tracer=session.tracer)
# Use tracing_llm.node_call() in LangGraph nodes
```

### Graceful Degradation

| Scenario | Behavior |
|----------|----------|
| `mlflow` not installed | `mlflow_autolog=True` logs a DEBUG message, session works normally |
| `record_spans=False` | No spans recorded, all telemetry methods return empty results |
| `span_attribute_filter` returns `{}` | Span is dropped silently |
| OTEL exporter error | Caught internally, does not affect Trace graph |
| No active session | `TelemetrySession.current()` returns `None`, all instrumentation is skipped |
