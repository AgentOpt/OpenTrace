# LangGraph OTEL Instrumentation API

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mjehanzaib999/NewTrace/blob/feature/M0-technical-plan/examples/notebooks/prototype_api_validation.ipynb)

A simplified API for instrumenting LangGraph agents with OpenTelemetry (OTEL) tracing, enabling optimization via the Trace framework while maintaining compatibility with Agent Lightning semantic conventions.

---

## Before vs After: Boilerplate Reduction (Top Success Metric)

The design goal is **minimal code change** for a developer to create a session, instrument a graph, run the optimize loop, and persist artifacts. Below: comparison table and a minimal unified diff.

### Before vs After Table

| Step | Before (manual) | After (this API) |
|------|-----------------|------------------|
| **Create session** | ~50 lines: TracerProvider, InMemorySpanExporter, SimpleSpanProcessor, tracer init | `session` created inside `instrument_graph()`; no explicit session creation in user code |
| **Instrument graph** | ~25 lines per node: manual span creation, attribute setting, TracingLLM wiring | One call: `instrumented = instrument_graph(graph, ...)` |
| **Run optimize loop** | ~150 lines: loop, trace capture, TGJ conversion, score tracking, template update | One call: `result = optimize_langgraph(instrumented, queries, iterations=5)` |
| **Persist artifacts** | ~50 lines: OTLP export, file write, optional MLflow log | `otlp = instrumented.session.flush_otlp()`; optional `session.export_run_bundle()` or MLflow |

### Minimal Code Diff (Before → After)

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
+ from prototype_api_validation import instrument_graph, optimize_langgraph
+
+ instrumented = instrument_graph(
+     graph=my_graph,
+     service_name="my-agent",
+     trainable_keys={"planner", "synthesizer"},
+ )
+ result = optimize_langgraph(instrumented, queries=["Q1", "Q2"], iterations=5)
+ otlp = instrumented.session.flush_otlp()
+ # Optional: save to file, or session.export_run_bundle(output_dir)
```

### Before vs After Optimization (Design Overview)

| Aspect | Before (manual / SPANOUTNODE-style) | After (this API) |
|--------|-------------------------------------|------------------|
| **Instrumentation** | Manual per-node spans + custom TracingLLM wiring | Single `instrument_graph()`; nodes wrapped automatically |
| **Optimization loop** | Copy-paste loop: invoke → flush OTLP → TGJ → optimizer | Single `optimize_langgraph()`; internal capture and (future) TGJ/optimizer |
| **Telemetry surface** | Ad hoc logging, file-based logs | Unified OTEL spans + (planned) MLflow; one session per run |
| **Boilerplate** | ~645 lines typical | ~10 lines for session + instrument + optimize + persist |

*(For a visual “before vs after optimization” diagram similar to [agent-lightning readme-diff](https://github.com/microsoft/agent-lightning/blob/main/docs/assets/readme-diff.svg), see the table above and the Architecture section.)*

---

## Overview

This project addresses the challenge of **excessive boilerplate code** when integrating OTEL tracing with LangGraph for optimization purposes. The goal is to reduce ~645 lines of manual instrumentation code to just **2 function calls**.

### Key Features

- **One-liner instrumentation**: `instrument_graph()` wraps any LangGraph with full OTEL tracing
- **One-liner optimization**: `optimize_langgraph()` runs optimization loops with telemetry capture
- **Dual semantic conventions**: Emits spans compatible with both Trace TGJ and Agent Lightning
- **Flexible LLM backend**: Supports OpenRouter API or StubLLM for testing
- **OTLP export**: Full trace export to JSON files for analysis

### Generalization: Any LangGraph (Not Demo-Specific)

The optimization and instrumentation plan applies to **any LangGraph**, not only a fixed "planner / researcher / synthesizer / evaluator" topology.

**Supported graph kinds:**

| Kind | Support | Notes |
|------|---------|--------|
| **Sync graphs** | Yes | `invoke()` on compiled `StateGraph`; node wrappers run synchronously. |
| **Async graphs** | Planned | `ainvoke()` / `astream()`; same wrapper model, async span handling. |
| **Streaming** | Planned | `stream()` / `astream()`; spans emitted per node completion. |
| **Tools** | Yes | Tool calls inside nodes are traced via the same LLM wrapper; tool name/args can be added as span attributes. |
| **Loops** | Yes | Cyclic graphs and conditional edges are supported; each node execution gets a span. |

**Instrumentation approach: node wrappers (not callbacks).**

- **Chosen method:** Wrapping node logic with a **node-level wrapper** that creates a session span and injects a `TracingLLM` (or tool tracer) into the node’s execution context. The graph is not modified by LangChain/LangGraph **callbacks** for core tracing.
- **Why wrappers:** (1) Full control over span boundaries and parent-child relationship (e.g. node → LLM child span). (2) Guaranteed `param.*` and `gen_ai.*` attributes for TGJ and Agent Lightning without depending on callback event stability. (3) Works the same for custom graphs and the default research graph.
- **Callbacks (optional):** If we add optional LangChain/LangGraph callback-based observability, we will document exactly which events we depend on (e.g. `on_chain_start` / `on_llm_end`). See [LangChain observability](https://docs.langchain.com/oss/python/langgraph/observability) and [reference.langchain.com](https://reference.langchain.com/python/langgraph/graphs/). Currently, **we do not rely on callbacks** for the core optimization path.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER CODE (Minimal)                         │
│  instrumented = instrument_graph(...)   # ONE call              │
│  result = optimize_langgraph(...)       # ONE call              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    InstrumentedGraph                            │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────┐        │
│  │ StateGraph  │  │ TelemetrySession │  │  TracingLLM  │        │
│  │ (LangGraph) │  │   (OTEL spans)   │  │ (dual semconv)│        │
│  └─────────────┘  └─────────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Backend                                │
│  ┌─────────────────────┐    ┌─────────────────────────┐        │
│  │   OpenRouterLLM     │ OR │       StubLLM           │        │
│  │  (Real API calls)   │    │ (Deterministic testing) │        │
│  └─────────────────────┘    └─────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## API Matrix by Optimization Mode

Public APIs by optimization/observability mode. Exact names and signatures are below.

| Mode | Primary API(s) | Purpose |
|------|----------------|--------|
| **Trace-only instrumentation** | `instrument_graph(...)`, `TelemetrySession`, `InstrumentedGraph.invoke` | Add OTEL spans to a graph; no optimization. |
| **Prompt optimization** | `instrument_graph(..., trainable_keys=...)`, `optimize_langgraph(...)` | Mark trainable nodes; run optimization loop over prompts. |
| **Code optimization** | `instrument_graph(..., enable_code_optimization=True)` (planned), `TracingLLM` with `emit_code_param` | Emit `param.__code_*` for function-source optimization. |
| **Hyperparameter optimization** | `optimize_langgraph(..., optimizer=..., optimizer_kwargs=...)` (planned) | Pass optimizer config (e.g. learning rate, steps). |
| **Partial graph selection** | `instrument_graph(..., node_selector=...)` (planned) | Select nodes by name set, tags, or regex; only those nodes get full tracing. |
| **Observability tuning** | `TelemetrySession(..., capture_state=..., truncation=..., redaction=...)` (planned) | Control state capture, truncation, and PII redaction in spans. |

### Proposed API Signatures

**`instrument_graph`**

```text
instrument_graph(
    graph: StateGraph | CompiledGraph | None = None,
    *,
    service_name: str = "langgraph-agent",
    trainable_keys: Set[str] | None = None,
    llm: Any | None = None,
    initial_templates: Dict[str, str] | None = None,
    emit_genai_child_spans: bool = True,
    use_stub_llm: bool = False,
    # Planned: enable_code_optimization, node_selector (nodes | tags | regex)
) -> InstrumentedGraph
```

**`optimize_langgraph`**

```text
optimize_langgraph(
    graph: InstrumentedGraph,
    queries: List[str],
    *,
    iterations: int = 3,
    on_iteration: Callable[[int, List[RunResult], Dict], None] | None = None,
    # Planned: optimizer, optimizer_kwargs, eval_fn, initial_templates, log_to_mlflow
) -> OptimizationResult
```

**LLM / tool wrappers**

- **`TracingLLM`**: wraps an LLM; `node_call(span_name, template_name, template, messages, ...)` — used internally by instrumented nodes.
- **Tool wrapper** (planned): `trace_tool_call(tool_name, args, result)` or similar for tool spans.

**Selection config (planned)**

- **Selector**: `node_selector: Literal["all"] | Set[str] | Sequence[str]` (node names) or `tags: Set[str]` or `node_pattern: str` (regex).
- **Nodes**: set of node names to treat as trainable or to include in partial trace.
- **Tags**: node metadata tags used for selection (when LangGraph node metadata is used).

## Unified OTEL + MLflow Telemetry Plan

How telemetry is initiated and how it covers trainers, optimizer internals, node spans, and LLM/tool calls.

| Component | Telemetry hook | OTEL output | MLflow output |
|-----------|----------------|-------------|---------------|
| **TelemetrySession** | `session.flush_otlp()`, `session.start_span()` | OTLP JSON (resourceSpans / scopeSpans / spans) | — |
| **Trainers (BaseLogger)** | Logger `log(name, data, step)` | — | Metrics/params via `MLflowTelemetryLogger` (planned) |
| **Optimizer internal logs** | `summary_log`, iteration callback | Optional span or event with `optimizer.iteration`, `optimizer.score` | Metrics at each step (e.g. `score`, `iteration`) |
| **Node execution** | Node wrapper `start_span(node_name)` | One span per node with `param.*`, `inputs.*` | — (traces as artifacts if logged) |
| **LLM calls** | `TracingLLM.node_call()` | Parent node span + child span `gen_ai.*` | — (or token/latency metrics if added) |
| **Tool calls** | Tool wrapper (planned) | Child span under node with tool name/args | — |
| **Evaluation / reward** | `emit_agentlightning_reward()` or eval span | Span `agentlightning.annotation` with `agentlightning.reward.0.*` | Metric `reward` or `score` |

**Initiation:** Telemetry is started when the user creates an `InstrumentedGraph` via `instrument_graph()`, which creates a `TelemetrySession`. The session is bound to that graph’s execution. No global OTEL provider is required for the prototype; the session holds an in-memory exporter and flushes to OTLP JSON on demand.

**MLflow concurrency:** The MLflow fluent API (`mlflow.log_metric`, `mlflow.log_param`, etc.) is **not thread-safe**. Concurrent callers (e.g. multiple optimization runs or parallel eval) must use either: (1) **mutual exclusion** (e.g. a lock around MLflow log calls), or (2) the **MLflow Client API** with explicit run IDs and thread-local or process-local clients. The plan is to use a single active run per `optimize_langgraph()` call and serialize logging, or to document that concurrent MLflow logging requires the client API and explicit run management. See [MLflow documentation](https://mlflow.org/docs/latest/python_api/index.html) for client usage.

## OTEL Span / Attribute Contract

Guaranteed attributes by span type.

**Node spans** (one per node execution):

- `param.{template_name}` — prompt template text (if node has a trainable template).
- `param.{template_name}.trainable` — `"True"` or `"False"`.
- `inputs.gen_ai.prompt` — user-facing input snippet (e.g. last user message).
- `gen_ai.model` — model identifier (e.g. `meta-llama/llama-3.1-8b-instruct:free`).

**LLM spans** (child of node span; prefer OpenTelemetry GenAI conventions as child):

- `gen_ai.operation.name` — e.g. `"chat"`.
- `gen_ai.provider.name` — e.g. `"openrouter"`, `"stub"`.
- `gen_ai.request.model` — model ID.
- `gen_ai.input.messages` — JSON array of messages.
- `gen_ai.output.messages` — JSON array of response messages.
- `trace.temporal_ignore` — `"true"` so the child is excluded from TGJ temporal chain.

**Evaluation / reward spans** (optional Agent Lightning compatibility):

- Span name: `agentlightning.annotation`.
- `trace.temporal_ignore` — `"true"`.
- `agentlightning.reward.0.name` — e.g. `"final_score"`.
- `agentlightning.reward.0.value` — stringified numeric reward (e.g. `"0.933"`).

References: [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), [Agent Lightning reward convention](https://microsoft.github.io/agent-lightning/latest/).

## Tests and Notebook Plan

For each new public function, at least one pytest test (using StubLLM for determinism) and the milestone notebook that demonstrates it.

| Public function / type | Pytest test (StubLLM) | Milestone notebook |
|------------------------|----------------------|--------------------|
| `TelemetrySession` | `test_telemetry_session_span_capture`, `test_telemetry_session_flush_otlp`, `test_span_attributes` | M0: `prototype_api_validation.ipynb` (session creation, flush) |
| `TracingLLM` | `test_tracing_llm_parent_span_attributes`, `test_tracing_llm_child_span_gen_ai`, `test_tracing_llm_temporal_ignore` | M0: same notebook (LLM node calls) |
| `instrument_graph()` | `test_instrument_graph_returns_instrumented`, `test_instrument_graph_session_configured`, `test_instrument_graph_trainable_keys` | M0: same notebook (instrument + invoke) |
| `InstrumentedGraph.invoke` | `test_instrumented_graph_invoke_with_stubllm`, `test_instrumented_graph_generates_spans` | M0: same notebook (single run) |
| `optimize_langgraph()` | `test_optimize_langgraph_returns_result`, `test_optimize_langgraph_score_history`, `test_optimize_langgraph_best_iteration` | M0: same notebook (optimization loop) |
| `emit_agentlightning_reward` (planned) | `test_emit_reward_span_attributes` | M0 or M1: notebook (evaluation step) |

All tests use **StubLLM** (deterministic) so they do not require API keys and are CI-friendly.

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- (Optional) OpenRouter API key for real LLM calls

## Installation

### Option 1: Using pip

```bash
# Clone or navigate to the project
cd H:\Freelance_Projects\Upwork\OTEL_Trace_Langraph\NewTrace

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using uv (faster)

```bash
cd H:\Freelance_Projects\Upwork\OTEL_Trace_Langraph\NewTrace

# Initialize uv project
uv init

# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Required for real LLM calls (get from https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Model selection (default: free Llama 3.1 8B)
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct:free

# API base URL (usually no need to change)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Set to "true" to force stub mode (no API calls)
USE_STUB_LLM=false
```

### Available Models

| Model | Cost | Environment Variable Value |
|-------|------|---------------------------|
| Llama 3.1 8B | Free | `meta-llama/llama-3.1-8b-instruct:free` |
| Mistral 7B | Free | `mistralai/mistral-7b-instruct:free` |
| Gemma 2 9B | Free | `google/gemma-2-9b-it:free` |
| Claude 3.5 Sonnet | Paid | `anthropic/claude-3.5-sonnet` |
| GPT-4o | Paid | `openai/gpt-4o` |

## Running the Prototype

### With StubLLM (No API calls - for testing)

```bash
# Windows PowerShell
$env:USE_STUB_LLM="true"; python examples\prototype_api_validation.py

# Linux/macOS Bash
USE_STUB_LLM=true python examples/prototype_api_validation.py

# Using uv
uv run python examples/prototype_api_validation.py  # Uses .env settings
```

### With Real LLM (OpenRouter)

```bash
# Ensure OPENROUTER_API_KEY is set in .env, then:
python examples/prototype_api_validation.py

# Or set inline (PowerShell)
$env:OPENROUTER_API_KEY="sk-or-v1-your-key"; python examples\prototype_api_validation.py

# Or set inline (Bash)
OPENROUTER_API_KEY="sk-or-v1-your-key" python examples/prototype_api_validation.py
```

## Expected Output

### 1. Configuration Display

```
============================================================
PROTOTYPE API VALIDATION
LangGraph OTEL Instrumentation API
============================================================

Configuration:
  OPENROUTER_API_KEY: [SET]
  OPENROUTER_MODEL: meta-llama/llama-3.1-8b-instruct:free
  USE_STUB_LLM: False
  Mode: REAL LLM (OpenRouter)
```

### 2. Unit Tests

```
============================================================
UNIT TESTS (using StubLLM)
============================================================

[TEST] TelemetrySession
----------------------------------------
  [OK] Span capture works
  [OK] OTLP export works
  [OK] Attributes correctly formatted

[TEST] TracingLLM
----------------------------------------
  [OK] Parent span has Trace-compatible attributes
  [OK] Child span has Agent Lightning-compatible attributes
  [OK] trace.temporal_ignore is set on child span

[TEST] instrument_graph()
----------------------------------------
  [OK] instrument_graph() creates InstrumentedGraph
  [OK] Session configured correctly
  [OK] TracingLLM configured with trainable_keys
  [OK] Templates initialized

[TEST] Real LangGraph with StubLLM
----------------------------------------
  [OK] LangGraph executed successfully
  [OK] Generated 10 spans
  [OK] Score: 0.500

[TEST] Optimization Loop with StubLLM
----------------------------------------
  [OK] optimize_langgraph() returns OptimizationResult
  [OK] Score history tracked correctly
  [OK] Best iteration identified

============================================================
ALL UNIT TESTS PASSED [OK]
============================================================
```

### 3. Demo Execution

```
============================================================
DEMO: Real LLM Execution
============================================================

1. Instrument a LangGraph (ONE function call):
----------------------------------------
  -> Created InstrumentedGraph with session: demo-api
  -> LLM type: OpenRouterLLM

2. Single graph execution:
----------------------------------------
  Query: What are the main causes of climate change?
  Score: 0.933
  Metrics: {'answer_relevance': 0.95, 'groundedness': 0.9, 'plan_quality': 0.95}
  Answer preview: Based on the provided research...
  Spans generated: 10
  Trace saved to: H:\...\examples\trace_output.json

3. OTLP Trace Output (Single Execution):
----------------------------------------

  Total spans: 10
  Showing first 10 spans:

  1. [NODE] planner (id: span_0001)
       - param.planner_prompt.trainable: True
       - gen_ai.model: meta-llama/llama-3.1-8b-instruct:free
       - inputs.gen_ai.prompt: You are a planning agent...

  2. [CHILD/GenAI] openrouter.chat.completion (id: span_0002)
       - gen_ai.operation.name: chat
       - gen_ai.provider.name: openrouter
       - trace.temporal_ignore: true
  ...

4. Run optimization loop:
----------------------------------------
  Running baseline...
    Query 1/2: What is artificial intelligence?...
      Score: 0.933
  ...
  Results:
    Baseline: 0.933
    Best: 0.933 (iteration 0)
    History: ['0.933', '0.917', '0.917']

5. Optimization Traces:
----------------------------------------
  All optimization traces saved to: H:\...\examples\optimization_traces.json
  Total trace files: 6 (baseline + 2 iterations x 2 queries)

============================================================
DEMO COMPLETE [OK]
============================================================
```

## Output Files

After running the prototype, you'll find:

| File | Description |
|------|-------------|
| `examples/trace_output.json` | OTLP trace from single graph execution |
| `examples/optimization_traces.json` | All traces from optimization loop |

### Sample OTLP Trace Structure

```json
{
  "resourceSpans": [{
    "resource": {"attributes": []},
    "scopeSpans": [{
      "scope": {"name": "demo-api"},
      "spans": [
        {
          "traceId": "trace_1738851234567",
          "spanId": "span_0001",
          "name": "planner",
          "attributes": [
            {"key": "param.planner_prompt", "value": {"stringValue": "..."}},
            {"key": "param.planner_prompt.trainable", "value": {"stringValue": "True"}},
            {"key": "gen_ai.model", "value": {"stringValue": "llama-3.1-8b"}}
          ]
        }
      ]
    }]
  }]
}
```

## API Reference

### `instrument_graph()`

Wraps a LangGraph with automatic OTEL instrumentation.

```python
from prototype_api_validation import instrument_graph

instrumented = instrument_graph(
    graph=None,                    # StateGraph (or None for default research graph)
    service_name="my-agent",       # OTEL service name
    trainable_keys={"planner"},    # Nodes with optimizable prompts
    llm=None,                      # Custom LLM client (or auto-detect)
    initial_templates={},          # Initial prompt templates
    emit_genai_child_spans=True,   # Emit Agent Lightning spans
    use_stub_llm=False,            # Force stub mode
)

# Execute
result = instrumented.invoke({"query": "What is AI?"})
print(result["answer"])
print(result["score"])
```

### `optimize_langgraph()`

Runs optimization loop on instrumented graph.

```python
from prototype_api_validation import optimize_langgraph

result = optimize_langgraph(
    graph=instrumented,            # InstrumentedGraph
    queries=["Query 1", "Query 2"], # Test queries
    iterations=3,                  # Number of optimization iterations
    on_iteration=None,             # Callback after each iteration
)

print(f"Baseline: {result.baseline_score}")
print(f"Best: {result.best_score} (iteration {result.best_iteration})")
print(f"History: {result.score_history}")
```

### `TelemetrySession`

Manages OTEL span collection and export.

```python
from prototype_api_validation import TelemetrySession

session = TelemetrySession("my-service")

with session.start_span("my_operation") as span:
    span.set_attribute("key", "value")
    # ... do work ...

# Export to OTLP JSON
otlp = session.flush_otlp()
```

### `TracingLLM`

LLM wrapper with dual semantic conventions.

```python
from prototype_api_validation import TracingLLM, TelemetrySession, StubLLM

session = TelemetrySession("test")
llm = StubLLM()

tracing_llm = TracingLLM(
    llm=llm,
    session=session,
    trainable_keys={"planner"},
    emit_genai_child_span=True,
)

response = tracing_llm.node_call(
    span_name="planner",
    template_name="planner_prompt",
    template="Plan: {query}",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Project Structure

```
NewTrace/
├── .env                          # Environment configuration (create from .env.example)
├── .env.example                  # Template for .env
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── docs/
│   └── T1_technical_plan.md      # Detailed technical specification
└── examples/
    ├── prototype_api_validation.py    # Main prototype script
    ├── trace_output.json              # Generated: Single execution trace
    ├── optimization_traces.json       # Generated: All optimization traces
    ├── JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE.py  # Original demo (reference)
    └── JSON_OTEL_trace_optim_demo_LANGGRAPH_DESIGN3_4.py    # Design 3/4 demo (reference)
```

## Key Components Explained

### Dual Semantic Conventions

The `TracingLLM` emits two types of spans for each LLM call:

| Parent Span (Trace-compatible) | Child Span (Agent Lightning-compatible) |
|-------------------------------|----------------------------------------|
| `param.{name}` - Template text | `gen_ai.operation.name` - "chat" |
| `param.{name}.trainable` - Optimizable flag | `gen_ai.provider.name` - Provider |
| `inputs.gen_ai.prompt` - User input | `gen_ai.input.messages` - Full messages |
| | `trace.temporal_ignore=true` - TGJ stability |

### The `trace.temporal_ignore` Attribute

This attribute prevents child spans from disrupting the temporal hierarchy in Trace-Graph JSON (TGJ) conversion:

```
Without temporal_ignore:
  planner -> openrouter.chat.completion -> researcher (WRONG!)

With temporal_ignore:
  planner -> researcher (CORRECT - child span excluded from chain)
```

### LangGraph Flow

```
START -> planner -> researcher -> synthesizer -> evaluator -> END
           │            │              │              │
           ▼            ▼              ▼              ▼
      Creates plan  Gathers info  Final answer  Quality scores
```

## Troubleshooting

### "No module named 'langgraph'"

```bash
pip install langgraph
# or
uv pip install langgraph
```

### "OpenRouter API key not provided"

1. Get a key from https://openrouter.ai/keys
2. Add to `.env`: `OPENROUTER_API_KEY=sk-or-v1-your-key`
3. Or use stub mode: `USE_STUB_LLM=true`

### "Connection error" with OpenRouter

- Check your internet connection
- Verify the API key is valid
- Try a different model (some may be temporarily unavailable)

### Unicode errors on Windows

The prototype uses ASCII-only characters to avoid encoding issues on Windows terminals.

## Future Enhancements

- [ ] Real OpenTelemetry SDK integration
- [ ] MLflow integration for monitoring
- [ ] Support for conditional graph edges
- [ ] Human-in-the-loop optimization
- [ ] Trace visualization dashboard
- [ ] Integration with Jaeger/Zipkin

## Notebook Requirements (When Pushed to GitHub)

For the prototype notebook (`examples/prototype_api_validation.ipynb`) when the repo is on GitHub:

1. **Open in Colab**  
   Add an "Open in Colab" badge at the top of the README or in the notebook description, linking to:
   `https://colab.research.google.com/github/<org>/<repo>/blob/<branch>/examples/prototype_api_validation.ipynb`

2. **API key retrieval**  
   Do **not** pass API keys as parameters. Use:
   - **Google Colab**: [Colab Secrets](https://colab.research.google.com/notebooks/secrets.ipynb) (e.g. `userdata.get("OPENROUTER_API_KEY")`) or `os.environ.get("OPENROUTER_API_KEY")` after setting the secret in the notebook’s secret manager.
   - **Local / env**: `python-dotenv` and `.env` (or `os.environ`); keys in `.env` or environment, never in notebook parameters.

3. **Auto-save results to Google Drive**  
   In Colab, mount Drive and write outputs (e.g. `trace_output.json`, `optimization_traces.json`) to a persistent folder (e.g. `MyDrive/NewTrace_runs/run_<timestamp>`), then **print the run folder path** so the user can find results after closing the notebook.

4. **GitHub fork/branch or PR**  
   Prefer sharing a **GitHub fork/branch link or PR** for review so reviewers can run and re-run the notebook (e.g. on Colab) directly from the repo. Example:
   - Branch: `https://github.com/<org>/<repo>/tree/<branch>`
   - PR: `https://github.com/<org>/<repo>/pull/<num>`  
   The notebook should be runnable with results; reviewers should also be able to re-execute it quickly on Google Colab.

## Related Documentation

- [Technical Plan](docs/T1_technical_plan.md) - Detailed API specification
- [Architecture and Strategy](docs/architecture_and_strategy.md) - Design and data flow
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [OpenRouter API](https://openrouter.ai/docs)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
