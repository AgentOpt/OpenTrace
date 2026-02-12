# M1: Drop-in Instrumentation & End-to-End Optimization

> **Milestone 1** of the LangGraph OTEL Instrumentation API.
> Branch: `feature/M1-instrument-and-optimize`

[![Open Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mjehanzaib999/NewTrace/blob/feature/M1-instrument-and-optimize/examples/notebooks/01_m1_instrument_and_optimize.ipynb)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Flow Pipeline](#3-data-flow-pipeline)
4. [Public API Reference](#4-public-api-reference)
5. [Semantic Convention Design](#5-semantic-convention-design)
6. [Bindings & Parameter Update Mechanism](#6-bindings--parameter-update-mechanism)
7. [Temporal Chaining Contract](#7-temporal-chaining-contract)
8. [File Map](#8-file-map)
9. [Quick Start](#9-quick-start)
10. [Testing](#10-testing)
11. [Acceptance Criteria Status](#11-acceptance-criteria-status)
12. [What Changed from M0](#12-what-changed-from-m0)

---

## 1. Overview

M1 delivers the **core value proposition**: two function calls to instrument and optimize any LangGraph agent.

**Before M1** (M0 prototype — ~300 lines of boilerplate per agent):

```python
exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
tracer = provider.get_tracer("my-agent")
# ... manually create spans in every node ...
# ... manually flush, convert OTLP to TGJ, run optimizer ...
```

**After M1** (2 function calls):

```python
from opto.trace.io import instrument_graph, optimize_graph

ig = instrument_graph(graph=my_graph, llm=my_llm, initial_templates={...})
result = optimize_graph(ig, queries=["What is AI?"], iterations=3)
```

### Key capabilities

| Capability | How it works |
|---|---|
| **Instrument any LangGraph** | `instrument_graph()` wraps a `StateGraph`/`CompiledGraph` with OTEL tracing |
| **Optimize prompts** | `param.*` attributes + `Binding` objects map optimizer output to live templates |
| **Optimize code** | `param.__code_*` attributes (opt-in via `enable_code_optimization=True`) |
| **Optimize routing** | Expose routing knobs as `param.*` (e.g. `param.route_threshold`) |
| **Dual semantic conventions** | `param.*` for Trace/TGJ optimization + `gen_ai.*` for Agent Lightning observability |
| **Flexible evaluation** | `EvalFn` accepts `float`, `str`, `dict`, or `EvalResult` — auto-normalized |
| **Non-intrusive mode** | `in_place=False` (default) avoids permanent mutation of the original graph |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Code                                    │
│                                                                     │
│   graph = StateGraph(...)          # define LangGraph               │
│   graph.add_node("planner", ...)   # add nodes                     │
│   graph.add_node("synth", ...)                                      │
│                                                                     │
│   ig = instrument_graph(           # ONE-LINER instrumentation      │
│       graph=graph,                                                  │
│       llm=my_llm,                                                   │
│       initial_templates={...},                                      │
│   )                                                                 │
│                                                                     │
│   result = optimize_graph(ig, queries=[...])  # ONE-LINER optimize  │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
          ┌───────────────────────────▼───────────────────────────┐
          │              instrument_graph()                        │
          │                                                       │
          │  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  │
          │  │ Telemetry    │  │  TracingLLM  │  │  Bindings   │  │
          │  │ Session      │  │  (dual       │  │  (param →   │  │
          │  │              │  │   semconv)   │  │   setter)   │  │
          │  │ TracerProv.  │  │              │  │             │  │
          │  │ InMemoryExp. │  │ param.*      │  │ get() /     │  │
          │  │ flush_otlp() │  │ gen_ai.*     │  │ set()       │  │
          │  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘  │
          │         │                 │                  │         │
          │         └────────┬────────┘                  │         │
          │                  │                           │         │
          │    ┌─────────────▼───────────────┐           │         │
          │    │   InstrumentedGraph         │           │         │
          │    │   .graph   (CompiledGraph)  │           │         │
          │    │   .session (TelemetrySession)│          │         │
          │    │   .tracing_llm (TracingLLM) │           │         │
          │    │   .templates (dict)         ├───────────┘         │
          │    │   .bindings  (dict)         │                     │
          │    │   .invoke()  .stream()      │                     │
          │    └─────────────────────────────┘                     │
          └───────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Module | Purpose |
|-----------|--------|---------|
| **`InstrumentedGraph`** | `instrumentation.py` | Wrapper returned by `instrument_graph()`; holds graph, session, tracing_llm, templates, bindings |
| **`TelemetrySession`** | `telemetry_session.py` | Manages `TracerProvider` + `InMemorySpanExporter`; provides `flush_otlp()`, `flush_tgj()`, `export_run_bundle()` |
| **`TracingLLM`** | `langgraph_otel_runtime.py` | Wraps any OpenAI-compatible LLM; emits parent spans (`param.*`) and child spans (`gen_ai.*`) |
| **`Binding`** | `bindings.py` | Dataclass with `get()`/`set()` callables mapping optimizer keys to live variables |
| **`optimize_graph()`** | `optimization.py` | Orchestrates the full optimization loop: invoke → OTLP → TGJ → optimizer → `apply_updates()` |
| **`otel_adapter`** | `otel_adapter.py` | Converts OTLP JSON → Trace-Graph JSON (TGJ) with temporal hierarchy |
| **`tgj_ingest`** | `tgj_ingest.py` | Ingests TGJ documents into `ParameterNode` / `MessageNode` objects |
| **`otel_semconv`** | `otel_semconv.py` | Helpers: `emit_reward()`, `emit_trace()`, `record_genai_chat()` |

---

## 3. Data Flow Pipeline

The end-to-end pipeline executed by `optimize_graph()` per iteration:

```
  ┌─────────┐     ┌──────────┐     ┌───────────┐     ┌───────────┐
  │ invoke() │────►│ flush    │────►│ OTLP→TGJ  │────►│ ingest    │
  │ LangGraph│     │ _otlp()  │     │ adapter    │     │ _tgj()    │
  └─────────┘     └──────────┘     └───────────┘     └─────┬─────┘
                                                            │
                                                            ▼
  ┌─────────┐     ┌──────────┐     ┌───────────┐     ┌───────────┐
  │ apply   │◄────│ optimizer│◄────│ backward() │◄────│ Parameter │
  │_updates()│    │ .step()  │     │ feedback   │     │ Node +    │
  └────┬────┘     └──────────┘     └───────────┘     │ Message   │
       │                                              │ Node      │
       ▼                                              └───────────┘
  ┌─────────┐
  │templates│ ← updated via Binding.set()
  │  dict   │ → next invoke() uses new prompts
  └─────────┘
```

### Step-by-step

1. **`invoke()`** — Execute the LangGraph. Each node calls `TracingLLM.node_call()` which creates OTEL spans with `param.*` attributes.
2. **`flush_otlp()`** — Extract all collected spans from the `InMemorySpanExporter` as an OTLP JSON payload and clear the exporter.
3. **`eval_fn()`** — Evaluate the graph output. The `EvalFn` signature accepts `float | str | dict | EvalResult` and auto-normalizes.
4. **OTLP → TGJ** — `otlp_traces_to_trace_json()` converts OTLP spans into Trace-Graph JSON format with temporal hierarchy.
5. **`ingest_tgj()`** — Parse TGJ into `ParameterNode` (trainable prompts) and `MessageNode` (span outputs) objects.
6. **`backward()`** — Propagate evaluation feedback through the trace graph to trainable parameters.
7. **`optimizer.step()`** — The optimizer (e.g., `OptoPrime`) suggests parameter updates based on the feedback.
8. **`apply_updates()`** — Push the optimizer's output through `Binding.set()` to update live template values.
9. **Next iteration** — The updated templates are automatically used by `TracingLLM.node_call()` on the next `invoke()`.

---

## 4. Public API Reference

### High-level (2 function calls)

#### `instrument_graph()`

```python
from opto.trace.io import instrument_graph

ig = instrument_graph(
    graph=my_state_graph,           # StateGraph or CompiledGraph (auto-compiled)
    service_name="my-agent",        # OTEL service name
    trainable_keys={"planner"},     # None = all trainable (no hard-coded names)
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

#### `optimize_graph()`

```python
from opto.trace.io import optimize_graph, EvalResult

result = optimize_graph(
    graph=ig,                       # InstrumentedGraph from instrument_graph()
    queries=["q1", "q2"],           # List of queries or state dicts
    iterations=5,                   # Optimization iterations (after baseline)
    optimizer=None,                 # Auto-creates OptoPrime if None
    eval_fn=my_eval_fn,             # float | str | dict | EvalResult → normalized
    apply_updates_flag=True,        # Apply optimizer suggestions via bindings
    on_iteration=my_callback,       # (iter, runs, updates) progress callback
) -> OptimizationResult
```

### Data Contracts

#### `EvalResult`

```python
@dataclass
class EvalResult:
    score: float | None = None    # Numeric reward
    feedback: str = ""             # Textual feedback (Trace/TextGrad-compatible)
    metrics: dict = {}             # Free-form metrics
```

The `EvalFn` type accepts any of these return types and auto-normalizes:
- `float` / `int` → `EvalResult(score=value)`
- `str` → tries JSON parse, falls back to `EvalResult(feedback=value)`
- `dict` → `EvalResult(score=d["score"], feedback=d["feedback"])`
- `EvalResult` → passed through

#### `OptimizationResult`

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

### Binding System

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

### Span Helpers

```python
from opto.trace.io import emit_reward, emit_trace

# Emit a reward span (Agent Lightning compatible)
emit_reward(session, value=0.85, name="eval_score")

# Emit a custom debug span
emit_trace(session, name="my_debug_span", attrs={"key": "value"})
```

---

## 5. Semantic Convention Design

`TracingLLM` implements **dual semantic conventions** — a single LLM call emits two spans:

```
┌─────────────────────────────────────────────────┐
│  Parent span: "planner"                         │
│                                                 │
│  param.planner_prompt = "Plan for: {query}"     │  ← Trace/TGJ optimization
│  param.planner_prompt.trainable = true          │
│  inputs.gen_ai.prompt = "Plan for: cats"        │
│  gen_ai.model = "llama-3.1-8b"                  │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │  Child span: "openai.chat.completion"     │  │
│  │                                           │  │
│  │  gen_ai.operation.name = "chat"           │  │  ← Agent Lightning observability
│  │  gen_ai.provider.name = "openai"          │  │
│  │  gen_ai.request.model = "llama-3.1-8b"   │  │
│  │  gen_ai.output.preview = "Step 1: ..."    │  │
│  │  trace.temporal_ignore = "true"           │  │  ← prevents TGJ chain break
│  │                                           │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

| Convention | Purpose | Span Level | Used By |
|---|---|---|---|
| `param.*` | Trainable parameter values | Parent | Optimizer (via TGJ `ParameterNode`) |
| `param.*.trainable` | Whether the parameter is optimizable | Parent | TGJ adapter |
| `inputs.*` | Input signals to the node | Parent | TGJ `MessageNode` edges |
| `gen_ai.*` | LLM call metadata | Child | Agent Lightning dashboards |
| `trace.temporal_ignore` | Exclude from TGJ temporal chain | Child | `otel_adapter.py` |
| `agentlightning.reward.*` | Evaluation reward signals | Reward span | Agent Lightning |

---

## 6. Bindings & Parameter Update Mechanism

Bindings decouple the optimizer's string-keyed updates from the runtime location of the actual variable. This is the key mechanism that makes optimization **generic** — no hard-coded node names.

```
  Optimizer output                    Binding layer                  Runtime
  ─────────────────                   ─────────────                  ───────
  {"planner_prompt":  ──────►  bindings["planner_prompt"]  ──────►  templates["planner_prompt"]
   "new template"}              .set("new template")                = "new template"
                                                                         │
                                                                         ▼
                                                               next invoke() reads
                                                               updated template
```

### How bindings are created

1. **Auto-derived** (default): When `bindings=None` and `initial_templates` is provided, `instrument_graph()` creates one `Binding` per template key, backed by the `templates` dict.

2. **Explicit**: Pass `bindings={"key": Binding(get=..., set=...)}` for custom targets (e.g., class attributes, database rows, config files).

### Binding kinds

| Kind | Description | Example |
|------|-------------|---------|
| `"prompt"` | Text template / system prompt | `"Plan for: {query}"` |
| `"code"` | Function source code (via `param.__code_*`) | `"def route(state): ..."` |
| `"graph"` | Graph routing knob | `"param.route_threshold"` |

---

## 7. Temporal Chaining Contract

When `use_temporal_hierarchy=True`, the OTLP → TGJ adapter creates parent-child edges between sequential top-level spans. This enables the optimizer to propagate feedback **backward** through the full execution chain.

**The critical invariant**: Child spans (those with a `parentSpanId` in OTEL) must **NOT** advance the temporal chain. Without this, a child LLM span from node A could become the temporal parent of node B, breaking sequential optimization.

```
  OTEL spans (time order)           TGJ temporal chain
  ───────────────────────           ──────────────────
  planner (root)          ────────► planner
    └─ openai.chat (child)          (skipped — has parentSpanId)
  synthesizer (root)      ────────► synthesizer (parent = planner)
    └─ openai.chat (child)          (skipped)
```

The adapter achieves this with a simple check:

```python
# Only advance the temporal chain on spans that were NOT children in OTEL
if not orig_has_parent:
    prev_span_id = sid
```

Child spans carry `trace.temporal_ignore = "true"` as an additional signal for downstream consumers.

**Verified by**: `TestE2ETemporalIntegrity` (2 tests) + `TestTemporalChaining` (1 test).

---

## 8. File Map

### Core Modules (`opto/trace/io/`)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 82 | Public API surface — exports all M1 symbols |
| `instrumentation.py` | 138 | `instrument_graph()` + `InstrumentedGraph` dataclass |
| `optimization.py` | 412 | `optimize_graph()` loop + `EvalResult`, `EvalFn`, `RunResult`, `OptimizationResult` |
| `telemetry_session.py` | 188 | `TelemetrySession` — unified OTEL session manager |
| `bindings.py` | 105 | `Binding` dataclass + `apply_updates()` + `make_dict_binding()` |
| `otel_semconv.py` | 126 | `emit_reward()`, `emit_trace()`, `record_genai_chat()`, `set_span_attributes()` |
| `langgraph_otel_runtime.py` | 367 | `TracingLLM` (dual semconv), `InMemorySpanExporter`, `flush_otlp()` |
| `otel_adapter.py` | 168 | `otlp_traces_to_trace_json()` — OTLP → TGJ conversion with temporal hierarchy |
| `tgj_ingest.py` | 234 | `ingest_tgj()`, `merge_tgj()` — TGJ → `ParameterNode`/`MessageNode` |
| `tgj_export.py` | — | Export Trace subgraphs back to TGJ (pre-existing) |
| `eval_hooks.py` | — | Evaluation hook utilities (pre-existing) |

### Tests

| File | Tests | Scope |
|------|-------|-------|
| `tests/unit_tests/test_bindings.py` | 10 | `Binding`, `apply_updates()`, `make_dict_binding()` |
| `tests/unit_tests/test_otel_semconv.py` | 5 | `emit_reward()`, `emit_trace()`, `record_genai_chat()` |
| `tests/unit_tests/test_telemetry_session.py` | 6 | `TelemetrySession` flush, clear, filter, export |
| `tests/unit_tests/test_instrumentation.py` | 10 | `instrument_graph()`, `TracingLLM` child spans, temporal chaining |
| `tests/unit_tests/test_optimization.py` | 11 | `EvalResult`, `_normalise_eval()`, data classes |
| `tests/features_tests/test_e2e_m1_pipeline.py` | 21 | **Full E2E**: instrument → invoke → OTLP → TGJ → optimizer → apply_updates |
| **Total** | **63** | All pass (StubLLM only, CI-safe) |

### Notebook

| File | Sections | Modes |
|------|----------|-------|
| `examples/notebooks/01_m1_instrument_and_optimize.ipynb` | 10 | StubLLM (deterministic) + Live LLM (OpenRouter, guarded) |

### Artifacts (generated by notebook execution)

```
notebook_outputs/m1/
├── stub_sample_otlp.json       # Single-run OTLP trace
├── stub_sample_tgj.json        # Converted TGJ document
├── stub_all_traces.json        # All optimization traces (9 runs)
├── stub_summary.json           # Optimization summary
├── live_all_traces.json        # Live LLM traces (if API key set)
└── live_summary.json           # Live optimization summary
```

---

## 9. Quick Start

### Installation

```bash
# Create virtual environment
uv venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate     # Linux/macOS

# Install dependencies + project
uv pip install -r requirements.txt
uv pip install -e .
```

### Minimal Example (StubLLM)

```python
from typing import Any, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from opto.trace.io import instrument_graph, optimize_graph, apply_updates, EvalResult


# 1. Define state and graph
class State(TypedDict, total=False):
    query: str
    answer: str


# 2. StubLLM (no API calls)
class StubLLM:
    model = "stub"
    def __call__(self, messages=None, **kw):
        class R:
            class C:
                class M:
                    content = "Stub answer"
                message = M()
            choices = [C()]
        return R()


# 3. Instrument
templates = {"qa_prompt": "Answer: {query}"}
ig = instrument_graph(
    graph=None,
    llm=StubLLM(),
    initial_templates=templates,
    trainable_keys={"qa"},
)

# 4. Build graph (node closes over ig.tracing_llm and ig.templates)
def qa_node(state):
    tmpl = ig.templates.get("qa_prompt", "{query}")
    response = ig.tracing_llm.node_call(
        span_name="qa",
        template_name="qa_prompt",
        template=tmpl,
        optimizable_key="qa",
        messages=[{"role": "user", "content": tmpl.replace("{query}", state["query"])}],
    )
    return {"answer": response}

graph = StateGraph(State)
graph.add_node("qa", qa_node)
graph.add_edge(START, "qa")
graph.add_edge("qa", END)
ig.graph = graph.compile()

# 5. Invoke
result = ig.invoke({"query": "What is Python?"})
print(result["answer"])

# 6. Inspect OTLP
otlp = ig.session.flush_otlp()
print(f"Spans: {len(otlp['resourceSpans'][0]['scopeSpans'][0]['spans'])}")

# 7. Optimize (with custom eval)
opt = optimize_graph(
    ig,
    queries=["What is AI?"],
    iterations=2,
    eval_fn=lambda p: EvalResult(score=0.8, feedback="good"),
)
print(f"Score history: {opt.score_history}")
```

---

## 10. Testing

### Run all M1 tests

```bash
python -m pytest tests/unit_tests/test_bindings.py \
    tests/unit_tests/test_otel_semconv.py \
    tests/unit_tests/test_telemetry_session.py \
    tests/unit_tests/test_instrumentation.py \
    tests/unit_tests/test_optimization.py \
    tests/features_tests/test_e2e_m1_pipeline.py \
    -v
```

### Run only the E2E integration test

```bash
python -m pytest tests/features_tests/test_e2e_m1_pipeline.py -v
```

### Test structure

The E2E test (`test_e2e_m1_pipeline.py`) builds a **real 2-node LangGraph** (planner → synthesizer) with `StubLLM` and validates every stage of the pipeline:

| Test Class | Tests | What it verifies |
|---|---|---|
| `TestE2EInstrumentAndInvoke` | 4 | Graph invocation produces result + OTLP spans |
| `TestE2EParamAttributes` | 2 | `param.*` and `param.*.trainable` on spans |
| `TestE2EOtlpToTgj` | 3 | OTLP → TGJ → `ParameterNode` + `MessageNode` with parent edges |
| `TestE2ETemporalIntegrity` | 2 | Child spans don't break temporal chain |
| `TestE2EBindingRoundTrip` | 3 | `apply_updates()` → template change → visible in next invoke |
| `TestE2EOptimizeEvalOnly` | 2 | `optimize_graph()` eval-only mode (no optimizer) |
| `TestE2EOptimizeWithMockOptimizer` | 3 | Full loop with mock optimizer verifying `apply_updates()` |
| `TestE2EFullRoundTrip` | 2 | Ultimate E2E: instrument → invoke → OTLP → TGJ → update → re-invoke |

---

## 11. Acceptance Criteria Status

All 7 M1 acceptance gates from the technical plan (`T1_technical_plan.md` §10):

| # | Gate | Status | Evidence |
|---|------|--------|----------|
| 1 | **OTLP export works** — `flush_otlp(clear=True)` returns >=1 span; second flush returns 0 | **PASS** | `test_flush_otlp_returns_spans`, `test_flush_otlp_clears_by_default` |
| 2 | **TGJ conversion works** — `flush_tgj()` produces docs consumable by `ingest_tgj()` | **PASS** | `test_tgj_has_parameter_nodes`, `test_tgj_has_message_nodes`, `test_message_node_has_parameter_parent` |
| 3 | **Temporal chaining contract** — child spans do NOT advance TGJ temporal chain | **PASS** | `test_synthesizer_temporal_parent_is_planner_not_child_span`, `test_child_spans_do_not_advance_temporal_chain` |
| 4 | **Bindings apply deterministically** — `strict=True` raises on missing; `strict=False` skips | **PASS** | `test_strict_missing_key_raises`, `test_non_strict_missing_key_skips`, `test_apply_updates_changes_template` |
| 5 | **E2E update path (CI/StubLLM)** — `optimize_graph(iterations>=2)` changes at least one prompt | **PASS** | `test_mock_optimizer_updates_are_applied`, `test_full_pipeline_end_to_end`, `test_optimize_graph_full_integration` |
| 6 | **Notebook live validation** — OTLP+TGJ artifacts with `param.*` from real provider | **PASS** | Notebook §9 (live section with OpenRouter), `live_summary.json` artifact |
| 7 | **Tests + notebook gate** — all new APIs have >=1 pytest; notebook has Colab badge | **PASS** | 63 pytest, Colab badge in notebook §1 |

### Notebook compliance

| Constraint | Status |
|---|---|
| Dual mode (StubLLM + Live) | Sections 4-8 (stub) + Section 9 (live) |
| Tiny dataset (<=3 items) | 3 queries (stub), 2 queries (live) |
| Deterministic settings | `temperature=0`, fixed model name |
| Budget guard | `max_tokens=256` per call |
| No secrets committed | Keys from Colab Secrets / env / `.env` only |
| Committed with executed outputs | `nbconvert --execute` with outputs captured |
| Open in Colab badge | First markdown cell |

---

## 12. What Changed from M0

M1 was built on top of M0's foundation, addressing all client review feedback:

| M0 (prototype) | M1 (production) |
|---|---|
| Hard-coded node names ("planner", "synthesizer") in optimization API | **Generic** — `trainable_keys=None` means all, or pass explicit set |
| `optimize_langgraph()` — LangGraph-specific name | **`optimize_graph()`** — framework-agnostic |
| No formal parameter binding mechanism | **`Binding` + `apply_updates()`** — explicit get/set contract |
| Eval function returned raw dicts | **`EvalResult` + `EvalFn`** — flexible contract (float/str/dict/EvalResult) |
| No non-intrusive mode | **`in_place=False`** (default) — no permanent graph mutation |
| No safety features on TelemetrySession | **`record_spans`** flag + **`span_attribute_filter`** for redaction |
| No `emit_trace()` helper | **`emit_trace()`** for manual span emission |
| Single semconv (param.* only) | **Dual semconv** — `param.*` (optimization) + `gen_ai.*` (observability) |
| Child LLM spans could break TGJ chain | **`trace.temporal_ignore`** + adapter skip logic — verified by tests |
| No milestone-based acceptance criteria | **7 SMART acceptance gates** — all verified with 63 tests + notebook |

---

## Dependencies

Core runtime:
- `opentelemetry-api >= 1.38.0`
- `opentelemetry-sdk >= 1.38.0`
- `langgraph >= 1.0.7`
- `typing-extensions >= 4.15.0`
- `graphviz >= 0.20.1`

Testing:
- `pytest >= 7.4.4`

Optional (live mode):
- `python-dotenv >= 1.1.0`
- `requests >= 2.28.0` (for OpenRouter client)
