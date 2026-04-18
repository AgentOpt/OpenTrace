# Graph Optimization

OpenTrace provides a unified API for instrumenting LangGraph agents with OpenTelemetry (OTEL) tracing and running prompt optimization loops. It reduces ~645 lines of manual instrumentation boilerplate to two function calls: `instrument_graph()` and `optimize_graph()`. Traces are emitted with dual semantic conventions compatible with both Trace (TGJ) and Agent Lightning, enabling optimization via the Trace framework while supporting standard observability tooling.

---

## Table of Contents

1. [Before / After](#1-before--after)
2. [Architecture](#2-architecture)
3. [Public API Reference](#3-public-api-reference)
4. [Data Flow Pipeline](#4-data-flow-pipeline)
5. [Semantic Conventions](#5-semantic-conventions)
6. [Temporal Chaining](#6-temporal-chaining)
7. [Core Modules](#7-core-modules)

---

## 1. Before / After

### Boilerplate Comparison

| Step | Before (manual) | After (this API) |
|------|-----------------|------------------|
| **Create session** | ~50 lines: TracerProvider, InMemorySpanExporter, SimpleSpanProcessor, tracer init | Created inside `instrument_graph()`; no explicit session code |
| **Instrument graph** | ~25 lines per node: manual span creation, attribute setting, TracingLLM wiring | `instrument_graph(graph, ...)` |
| **Run optimize loop** | ~150 lines: loop, trace capture, TGJ conversion, score tracking, template update | `optimize_graph(ig, queries, iterations=5)` |
| **Persist artifacts** | ~50 lines: OTLP export, file write, optional MLflow log | `ig.session.flush_otlp()` |

**Total: ~645 lines reduced to ~10 lines.**

### Backend modes

`instrument_graph()` and `optimize_graph()` support two backends:

| Backend | Carrier | Best for |
|---|---|---|
| `backend="otel"` (default) | OTLP spans → TGJ → ingest | observability-first optimization |
| `backend="trace"` | native Trace nodes (`bundle()` / `node()`) | direct graph-native optimization |

The OTEL path remains the default and most interoperable mode.

### Code Diff

```diff
- # --- BEFORE: Manual setup (~255+ lines for 4 steps) ---
- from opentelemetry.sdk.trace import TracerProvider
- from opentelemetry.sdk.trace.export import SimpleSpanProcessor, InMemorySpanExporter
- exporter = InMemorySpanExporter()
- provider = TracerProvider()
- provider.add_span_processor(SimpleSpanProcessor(exporter))
- tracer = provider.get_tracer("my-agent")
- # ... per-node: with tracer.start_as_current_span(name): ...
- # ... manual optimization loop with flush, TGJ, optimizer.step() ...
- # ... manual export to JSON / MLflow ...

+ # --- AFTER: Minimal API ---
+ from opto.trace.io import instrument_graph, optimize_graph
+
+ ig = instrument_graph(
+     graph=my_graph,
+     llm=my_llm,
+     initial_templates={"planner_prompt": "Plan for: {query}"},
+     trainable_keys={"planner", "synthesizer"},
+ )
+ result = optimize_graph(ig, queries=["Q1", "Q2"], iterations=5)
+ otlp = ig.session.flush_otlp()
```

---

## 2. Architecture

```
+---------------------------------------------------------------------+
|                        User Code                                     |
|                                                                      |
|   graph = StateGraph(...)          # define LangGraph                |
|   graph.add_node("planner", ...)   # add nodes                      |
|                                                                      |
|   ig = instrument_graph(           # ONE-LINER instrumentation       |
|       graph=graph, llm=my_llm,                                       |
|       initial_templates={...},                                       |
|   )                                                                  |
|   result = optimize_graph(ig, queries=[...])  # ONE-LINER optimize   |
+-------------------------------------+-------------------------------+
                                      |
          +---------------------------v---------------------------+
          |              instrument_graph()                        |
          |                                                       |
          |  +--------------+  +-------------+  +-------------+   |
          |  | Telemetry    |  |  TracingLLM |  |  Bindings   |   |
          |  | Session      |  |  (dual      |  |  (param ->  |   |
          |  |              |  |   semconv)  |  |   setter)   |   |
          |  | TracerProv.  |  |             |  |             |   |
          |  | InMemoryExp. |  | param.*     |  | get() /     |   |
          |  | flush_otlp() |  | gen_ai.*    |  | set()       |   |
          |  +------+-------+  +------+------+  +------+------+   |
          |         |                 |                 |          |
          |         +--------+--------+                 |          |
          |                  |                          |          |
          |    +-------------v-----------------+        |          |
          |    |   InstrumentedGraph           |        |          |
          |    |   .graph   (CompiledGraph)    |--------+          |
          |    |   .session (TelemetrySession) |                   |
          |    |   .tracing_llm (TracingLLM)   |                   |
          |    |   .templates (dict)           |                   |
          |    |   .bindings  (dict)           |                   |
          |    |   .invoke()  .stream()        |                   |
          |    +-------------------------------+                   |
          +--------------------------------------------------------+
```

### Component Responsibilities

| Component | Module | Purpose |
|-----------|--------|---------|
| `InstrumentedGraph` | `instrumentation.py` | Wrapper returned by `instrument_graph()`; holds graph, session, tracing_llm, templates, bindings |
| `TelemetrySession` | `telemetry_session.py` | Manages `TracerProvider` + `InMemorySpanExporter`; provides `flush_otlp()`, `flush_tgj()`, `export_run_bundle()` |
| `TracingLLM` | `langgraph_otel_runtime.py` | Wraps any OpenAI-compatible LLM; emits parent spans (`param.*`) and child spans (`gen_ai.*`) |
| `Binding` | `bindings.py` | Dataclass with `get()`/`set()` callables mapping optimizer keys to live variables |
| `optimize_graph()` | `optimization.py` | Orchestrates the optimization loop: invoke, flush OTLP, convert to TGJ, run optimizer, apply updates |
| `otel_adapter` | `otel_adapter.py` | Converts OTLP JSON to Trace-Graph JSON (TGJ) with temporal hierarchy |
| `tgj_ingest` | `tgj_ingest.py` | Ingests TGJ documents into `ParameterNode` / `MessageNode` objects |
| `otel_semconv` | `otel_semconv.py` | Helpers: `emit_reward()`, `emit_trace()`, `record_genai_chat()` |
| `graph_instrumentation` | `graph_instrumentation.py` | Trace-native graph instrumentation (`TraceGraph`) |

### Supported Graph Kinds

| Kind | Support | Notes |
|------|---------|--------|
| Sync graphs | Yes | `invoke()` on compiled `StateGraph`; node wrappers run synchronously |
| Async graphs | Planned | `ainvoke()` / `astream()`; same wrapper model, async span handling |
| Streaming | Planned | `stream()` / `astream()`; spans emitted per node completion |
| Tools | Yes | Tool calls inside nodes traced via the same LLM wrapper |
| Loops | Yes | Cyclic graphs and conditional edges; each node execution gets a span |

Instrumentation uses **node-level wrappers** (not LangChain/LangGraph callbacks). This provides full control over span boundaries and parent-child relationships, guarantees `param.*` and `gen_ai.*` attributes for TGJ and Agent Lightning, and works identically for custom and default graphs.

---

## 3. Public API Reference

### `instrument_graph()`

Wraps a LangGraph with automatic OTEL instrumentation.

```python
from opto.trace.io import instrument_graph

ig = instrument_graph(
    graph=my_state_graph,           # StateGraph or CompiledGraph (auto-compiled)
    service_name="my-agent",        # OTEL service name
    trainable_keys={"planner"},     # None = all trainable
    llm=my_llm_client,              # Any OpenAI-compatible client
    initial_templates={             # Starting prompt templates
        "planner_prompt": "Plan for: {query}",
    },
    emit_genai_child_spans=True,    # Agent Lightning gen_ai.* child spans
    bindings=None,                  # Auto-derived from templates if None
    in_place=False,                 # Don't permanently mutate original graph
    provider_name="openai",         # For gen_ai.provider.name attribute
) -> InstrumentedGraph
```

**Returns** an `InstrumentedGraph` with `.invoke()`, `.session`, `.tracing_llm`, `.templates`, and `.bindings`.

### `optimize_graph()`

Runs the optimization loop on an instrumented graph.

```python
from opto.trace.io import optimize_graph, EvalResult

result = optimize_graph(
    graph=ig,                       # InstrumentedGraph from instrument_graph()
    queries=["q1", "q2"],           # List of queries or state dicts
    iterations=5,                   # Optimization iterations (after baseline)
    optimizer=None,                 # Auto-creates OptoPrime if None
    eval_fn=my_eval_fn,             # float | str | dict | EvalResult -> normalized
    apply_updates_flag=True,        # Apply optimizer suggestions via bindings
    on_iteration=my_callback,       # (iter, runs, updates) progress callback
) -> OptimizationResult
```

### `EvalResult`

```python
@dataclass
class EvalResult:
    score: float | None = None    # Numeric reward
    feedback: str = ""             # Textual feedback (Trace/TextGrad-compatible)
    metrics: dict = {}             # Free-form metrics
```

The `EvalFn` type accepts any of these return types and auto-normalizes:

| Return type | Conversion |
|-------------|------------|
| `float` / `int` | `EvalResult(score=value)` |
| `str` | Tries JSON parse, falls back to `EvalResult(feedback=value)` |
| `dict` | `EvalResult(score=d["score"], feedback=d["feedback"])` |
| `EvalResult` | Passed through |

### `OptimizationResult`

```python
@dataclass
class OptimizationResult:
    baseline_score: float          # Average score of the baseline run
    best_score: float              # Best average score across iterations
    best_iteration: int            # Which iteration achieved best_score
    best_updates: dict             # The parameter updates that achieved best
    final_parameters: dict         # Current values of all bound parameters
    score_history: list[float]     # Average score per iteration [baseline, iter1, ...]
    all_runs: list[list[RunResult]]  # Nested: all_runs[iteration][query_idx]
```

### `Binding` and `apply_updates()`

Bindings decouple the optimizer's string-keyed updates from the runtime location of the actual variable. This makes optimization generic -- no hard-coded node names.

```python
from opto.trace.io import Binding, apply_updates, make_dict_binding

# Binding wraps any get/set pair
binding = Binding(
    get=lambda: my_config["prompt"],
    set=lambda v: my_config.__setitem__("prompt", v),
    kind="prompt",   # "prompt" | "code" | "graph"
)

# Convenience: bind to a dict entry
binding = make_dict_binding(my_dict, "key_name", kind="prompt")

# Apply optimizer output
apply_updates(
    {"prompt_key": "new value"},
    {"prompt_key": binding},
    strict=True,     # raise KeyError on unknown keys
)
```

**Binding kinds:**

| Kind | Description | Example |
|------|-------------|---------|
| `"prompt"` | Text template / system prompt | `"Plan for: {query}"` |
| `"code"` | Function source code (via `param.__code_*`) | `"def route(state): ..."` |
| `"graph"` | Graph routing knob | `"param.route_threshold"` |

**How bindings are created:**

1. **Auto-derived** (default): When `bindings=None` and `initial_templates` is provided, `instrument_graph()` creates one `Binding` per template key, backed by the `templates` dict.
2. **Explicit**: Pass `bindings={"key": Binding(get=..., set=...)}` for custom targets (e.g., class attributes, database rows, config files).

### Span Helpers

```python
from opto.trace.io import emit_reward, emit_trace

# Emit a reward span (Agent Lightning compatible)
emit_reward(session, value=0.85, name="eval_score")

# Emit a custom debug span
emit_trace(session, name="my_debug_span", attrs={"key": "value"})
```

---

## 4. Data Flow Pipeline

The end-to-end pipeline executed by `optimize_graph()` per iteration:

```
  +---------+     +----------+     +-----------+     +-----------+
  | invoke()|---->| flush    |---->| OTLP->TGJ |---->| ingest    |
  | LangGraph|    | _otlp()  |    | adapter    |    | _tgj()    |
  +---------+     +----------+     +-----------+     +-----+-----+
                                                           |
                                                           v
  +---------+     +----------+     +-----------+     +-----------+
  | apply   |<----| optimizer|<----| backward() |<----| Parameter |
  |_updates()|    | .step()  |    | feedback   |    | Node +    |
  +----+----+     +----------+     +-----------+     | Message   |
       |                                              | Node      |
       v                                              +-----------+
  +---------+
  |templates| <- updated via Binding.set()
  |  dict   | -> next invoke() uses new prompts
  +---------+
```

### Step-by-step

1. **`invoke()`** -- Execute the LangGraph. Each node calls `TracingLLM.node_call()` which creates OTEL spans with `param.*` attributes.
2. **`flush_otlp()`** -- Extract all collected spans from the `InMemorySpanExporter` as an OTLP JSON payload and clear the exporter.
3. **`eval_fn()`** -- Evaluate the graph output. The `EvalFn` signature accepts `float | str | dict | EvalResult` and auto-normalizes.
4. **OTLP to TGJ** -- `otlp_traces_to_trace_json()` converts OTLP spans into Trace-Graph JSON format with temporal hierarchy.
5. **`ingest_tgj()`** -- Parse TGJ into `ParameterNode` (trainable prompts) and `MessageNode` (span outputs) objects.
6. **`backward()`** -- Propagate evaluation feedback through the trace graph to trainable parameters.
7. **`optimizer.step()`** -- The optimizer (e.g., `OptoPrime`) suggests parameter updates based on the feedback.
8. **`apply_updates()`** -- Push the optimizer's output through `Binding.set()` to update live template values.
9. **Next iteration** -- The updated templates are automatically used by `TracingLLM.node_call()` on the next `invoke()`.

---

## 5. Semantic Conventions

`TracingLLM` implements **dual semantic conventions** -- a single LLM call emits two spans:

```
+--------------------------------------------------+
|  Parent span: "planner"                          |
|                                                  |
|  param.planner_prompt = "Plan for: {query}"      |  <- Trace/TGJ optimization
|  param.planner_prompt.trainable = true           |
|  inputs.gen_ai.prompt = "Plan for: cats"         |
|  gen_ai.model = "llama-3.1-8b"                   |
|                                                  |
|  +--------------------------------------------+  |
|  |  Child span: "openai.chat.completion"      |  |
|  |                                            |  |
|  |  gen_ai.operation.name = "chat"            |  |  <- Agent Lightning observability
|  |  gen_ai.provider.name = "openai"           |  |
|  |  gen_ai.request.model = "llama-3.1-8b"    |  |
|  |  gen_ai.output.preview = "Step 1: ..."     |  |
|  |  trace.temporal_ignore = "true"            |  |  <- prevents TGJ chain break
|  +--------------------------------------------+  |
+--------------------------------------------------+
```

### Attribute Reference

| Attribute | Purpose | Span Level | Consumed By |
|-----------|---------|------------|-------------|
| `param.*` | Trainable parameter values | Parent | Optimizer (via TGJ `ParameterNode`) |
| `param.*.trainable` | Whether the parameter is optimizable | Parent | TGJ adapter |
| `inputs.*` | Input signals to the node | Parent | TGJ `MessageNode` edges |
| `gen_ai.operation.name` | LLM operation type (e.g., `"chat"`) | Child | Agent Lightning dashboards |
| `gen_ai.provider.name` | LLM provider (e.g., `"openai"`, `"openrouter"`) | Child | Agent Lightning dashboards |
| `gen_ai.request.model` | Model identifier | Child | Agent Lightning dashboards |
| `gen_ai.input.messages` | JSON array of input messages | Child | Agent Lightning dashboards |
| `gen_ai.output.messages` | JSON array of response messages | Child | Agent Lightning dashboards |
| `trace.temporal_ignore` | Exclude from TGJ temporal chain (`"true"`) | Child | `otel_adapter.py` |
| `agentlightning.reward.0.name` | Evaluation reward name | Reward span | Agent Lightning |
| `agentlightning.reward.0.value` | Stringified numeric reward (e.g., `"0.933"`) | Reward span | Agent Lightning |

### OTEL Span Types

**Node spans** (one per node execution):
- `param.{template_name}` -- prompt template text (if node has a trainable template)
- `param.{template_name}.trainable` -- `"True"` or `"False"`
- `inputs.gen_ai.prompt` -- user-facing input snippet
- `gen_ai.model` -- model identifier

**LLM spans** (child of node span):
- `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`
- `gen_ai.input.messages`, `gen_ai.output.messages`
- `trace.temporal_ignore` = `"true"`

**Evaluation / reward spans** (Agent Lightning compatibility):
- Span name: `agentlightning.annotation`
- `trace.temporal_ignore` = `"true"`
- `agentlightning.reward.0.name`, `agentlightning.reward.0.value`

### `message.id`

Each span is assigned a unique `message.id` (span ID) used by the TGJ adapter to reconstruct parent-child and temporal edges in the trace graph. The `traceId` groups all spans from a single `invoke()` call.

---

## 6. Temporal Chaining

When `use_temporal_hierarchy=True`, the OTLP-to-TGJ adapter creates parent-child edges between sequential top-level spans. This enables the optimizer to propagate feedback **backward** through the full execution chain.

### The Critical Invariant

Child spans (those with a `parentSpanId` in OTEL) must **not** advance the temporal chain. Without this rule, a child LLM span from node A could become the temporal parent of node B, breaking sequential optimization.

```
  OTEL spans (time order)           TGJ temporal chain
  -----------------------           ------------------
  planner (root)          --------> planner
    +- openai.chat (child)          (skipped -- has parentSpanId)
  synthesizer (root)      --------> synthesizer (parent = planner)
    +- openai.chat (child)          (skipped)
```

The adapter achieves this with a simple check:

```python
# Only advance the temporal chain on spans that were NOT children in OTEL
if not orig_has_parent:
    prev_span_id = sid
```

Child spans carry `trace.temporal_ignore = "true"` as an additional signal for downstream consumers.

### Without vs. With temporal_ignore

```
Without temporal_ignore:
  planner -> openrouter.chat.completion -> researcher   (WRONG)

With temporal_ignore:
  planner -> researcher   (CORRECT -- child span excluded from chain)
```

---

## 7. Core Modules

### `opto/trace/io/`

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 82 | Public API surface -- exports all symbols |
| `instrumentation.py` | 138 | `instrument_graph()` + `InstrumentedGraph` dataclass |
| `optimization.py` | 412 | `optimize_graph()` loop + `EvalResult`, `EvalFn`, `RunResult`, `OptimizationResult` |
| `telemetry_session.py` | 188 | `TelemetrySession` -- unified OTEL session manager |
| `bindings.py` | 105 | `Binding` dataclass + `apply_updates()` + `make_dict_binding()` |
| `otel_semconv.py` | 126 | `emit_reward()`, `emit_trace()`, `record_genai_chat()`, `set_span_attributes()` |
| `langgraph_otel_runtime.py` | 367 | `TracingLLM` (dual semconv), `InMemorySpanExporter`, `flush_otlp()` |
| `otel_adapter.py` | 168 | `otlp_traces_to_trace_json()` -- OTLP to TGJ with temporal hierarchy |
| `tgj_ingest.py` | 234 | `ingest_tgj()`, `merge_tgj()` -- TGJ to `ParameterNode`/`MessageNode` |
| `tgj_export.py` | -- | Export Trace subgraphs back to TGJ (pre-existing) |
| `eval_hooks.py` | -- | Evaluation hook utilities (pre-existing) |

### Tests

| File | Tests | Scope |
|------|-------|-------|
| `tests/unit_tests/test_bindings.py` | 10 | `Binding`, `apply_updates()`, `make_dict_binding()` |
| `tests/unit_tests/test_otel_semconv.py` | 5 | `emit_reward()`, `emit_trace()`, `record_genai_chat()` |
| `tests/unit_tests/test_telemetry_session.py` | 6 | `TelemetrySession` flush, clear, filter, export |
| `tests/unit_tests/test_instrumentation.py` | 10 | `instrument_graph()`, `TracingLLM` child spans, temporal chaining |
| `tests/unit_tests/test_optimization.py` | 11 | `EvalResult`, `_normalise_eval()`, data classes |
| `tests/features_tests/test_e2e_m1_pipeline.py` | 21 | Full E2E: instrument, invoke, OTLP, TGJ, optimizer, apply_updates |
