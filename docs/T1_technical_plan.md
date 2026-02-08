# T1 Technical Plan: LangGraph OTEL Instrumentation API

**Version:** 1.1
**Date:** February 6, 2026  
**Author:** Jahanzeb Javed, Xavier Daull
**Status:** Review v1

This technical plan is **reusable for any LangGraph**, not tied to a specific demo graph (e.g. planner/researcher/synthesizer/evaluator). This doc explicitly addresses: (a) configurable evaluation via `eval_fn` that may return a numeric score *or* string feedback, (b) generic node selection (no hard-coded node names), (c) explicit `bindings={...}` + `apply_updates(...)` for robust mapping from `param.*` keys to real prompts/functions/graph knobs, and (d) `emit_reward()` + `emit_trace()` helpers; see the [README](../README.md) for the longer before/after diff + API matrix + telemetry tables. # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)

This plan also distinguishes the **optimization TGJ** (minimal, used by Trace backprop) from optional **observability TGJ/log artifacts** (full OTEL detail); merging via `merge_tgj([base_graph_doc, log_doc])` is **opt-in** and must not be required for a minimal optimization API. # 🔴 (keep optimization graph minimal while still allowing rich trace artifacts when needed)
---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Generalization: Supported Graphs and Instrumentation](#2-generalization-supported-graphs-and-instrumentation)
3. [Problem Analysis](#3-problem-analysis)
4. [Architecture Overview](#4-architecture-overview)
5. [Target API Specification](#5-target-api-specification)
6. [Module Modifications](#6-module-modifications)
7. [Implementation Plan](#7-implementation-plan)
8. [Agent Lightning Comparison](#8-agent-lightning-comparison)
9. [Notebooks (Deliverables from M1 onward)](#9-notebooks-deliverables-from-m1-onward)
10. [Acceptance Criteria (SMART, verifiable)](#10-acceptance-criteria-smart-verifiable)
11. [Test & Validation Plan](#11-test--validation-plan)
12. [Appendix: Prototype Snippet](#12-appendix-prototype-snippet)

---

## 1. Executive Summary

### Goal

Create a **minimal, reusable library/API** that allows developers to:

1. **Add OTEL instrumentation** to any LangGraph in a few lines (no copy-paste boilerplate)
2. **Run optimization loops** (flush OTLP → convert to TGJ → optimizer step → apply updates)
3. **Standardize telemetry** across trainers/optimizers/nodes, exportable to:
   - OTEL (for optimization + debugging)
   - MLflow (for monitoring: metrics + artifacts)

### Key Deliverables

| Deliverable | Description |
|-------------|-------------|
| `instrument_graph()` | Auto-instrument a LangGraph with OTEL tracing |
| `TracingLLM` (enhanced) | Wrapper with dual semantic conventions (Trace + Agent Lightning) |
| `TelemetrySession` | Unified session manager for OTEL + MLflow |
| `optimize_graph()` | One-liner optimization loop (# 🔴 just renamed `optimize_langgraph()` into `optimize_graph()` to align naming and future support of other graphs) |
| `emit()` helpers | Manual telemetry emission (`emit_reward()`, `emit_trace()`, custom spans/events) # 🔴 (provide a simple manual additional trace emission helper) |

---

## 2. Generalization: Supported Graphs and Instrumentation

The plan applies to **any LangGraph**, not only a fixed topology.

**Supported graph kinds:**

| Kind | Support | Notes |
|------|---------|--------|
| Sync graphs | Yes | `invoke()` on compiled StateGraph. |
| Async graphs | Planned | `ainvoke()` / `astream()`; same wrapper model. |
| Streaming | Planned | `stream()` / `astream()`; spans per node completion. |
| Tools | Yes | Tool calls inside nodes traced via LLM/tool wrapper. |
| Loops | Yes | Cyclic and conditional edges; one span per node execution. |

**Instrumentation: node wrappers (not callbacks).**

- We use **node-level wrappers** that create a session span and inject `TracingLLM` (or tool tracer) into the node execution context. We do **not** rely on LangChain/LangGraph **callbacks** for core tracing.
- **Why:** (1) Full control over span boundaries and parent-child (e.g. node → LLM child). (2) Guaranteed `param.*` and `gen_ai.*` for TGJ and Agent Lightning without depending on callback event stability. (3) Same behavior for any custom graph.
- If we add optional callback-based observability later, we will document exactly which events we depend on (e.g. [LangChain observability](https://docs.langchain.com/oss/python/langgraph/observability), [reference.langchain.com](https://reference.langchain.com/python/langgraph/graphs/)).

- **Instrumentation modes (to prove non-intrusive + generic):** # 🔴 (support non-intrusive optimization without modifying original code file)
- **Inline/minimal-change mode:** user passes `TracingLLM`/templates into the graph builder; `instrument_graph(..., in_place=True)` wraps nodes directly. # 🔴 (support non-intrusive optimization without modifying original code)
- **Non-intrusive mode (required demo):** `instrument_graph(..., in_place=False, bindings=...)` wraps/patches callables at runtime and restores them after the run, so the original **source files are unchanged**; updates still occur **in memory** via bindings/setters (trade-off: you cannot add new manual `emit_*` calls inside node bodies; you can still patch the LLM, prompts, and node callables). # 🔴 (clarify that “non-intrusive” means no source-file/permanent mutation, not “no in-memory updates”)
- **Capability checklist (must be demonstrated in examples):** # 🔴 (make acceptance criteria explicit for what the API must support)
- Optimize prompts/variables (via `param.<key>` + bindings). # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
- Optimize functions/code (Trace `bundle(traceable_code=True, trainable=True)` on callables => `param.__code_<node>`). # 🔴 (declare individual code/function optimization support)
- Optimize graph routing *indirectly* by exposing routing knobs as `param.*` (e.g. `param.router_policy`, `param.route_threshold`) in node/router spans emitted by `instrument_graph()`, and applying updates via `optimize_graph(..., bindings=...)` (topology/edge mutation). # 🔴 (routing is a trainable knob contract, not a graph rewrite)
- Trace LangGraph node execution via `instrument_graph()` (exactly one OTEL parent span per node invocation; LLM/tool spans are children). # 🔴 (span boundary contract is implemented by node wrappers, not by `trace.node(variable, trainable=True)` ? validate better option)
- Trace LLM calls via `TracingLLM`: the OTEL span that participates in optimization MUST carry `param.*` (+ `.trainable`), and also emits `gen_ai.*` keys for Agent-Lightning compatibility; child spans are deferred beyond M1. # 🔴 (optimizer links params via param.*; gen_ai.* is compatibility/observability)

---

## 3. Problem Analysis

### 3.1 Current Boilerplate in Demo Code

The current `JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE.py` (~1350 lines) contains extensive boilerplate that must be copied for each new LangGraph:

| Category | Lines | Code Example |
|----------|-------|--------------|
| **OTEL Setup** | ~50 | `InMemorySpanExporter`, `TracerProvider`, `SimpleSpanProcessor` |
| **TracingLLM Class** | ~60 | Duplicate of `langgraph_otel_runtime.py` |
| **flush_otlp()** | ~25 | Span serialization to OTLP JSON |
| **Logging Helpers** | ~180 | `_init_log_dir`, `_save_run_logs`, `_rebuild_aggregate_markdown` |
| **Parameter Mapping** | ~100 | `_remap_params_in_graph`, `_ensure_code_desc_on_optimizer` |
| **Optimization Loop** | ~150 | `optimize_iteration`, TGJ conversion, backward/step |
| **Code Patching** | ~80 | `_apply_code_update`, `_emit_code_param` | # (for information: it assumes that we provided before the necessary bindings/mapping info between the otel trace namings and the real code/variables to patch so that the optimizer made it possible)
| **Total Boilerplate** | **~645** | **~48% of demo is reusable infrastructure** |

### 3.2 Fragmented Logging Infrastructure

| Component | Current Logger | Issue |
|-----------|---------------|-------|
| Trainers | `BaseLogger` subclasses | Console/TensorBoard/WandB only |
| Optimizers | In-memory `log` list | Not exportable |
| Node execution | Custom `LOG_DIR` files | Not integrated with OTEL |
| MLflow | Not implemented | Manual artifact logging |

### 3.3 Manual LLM Wrapping

Every node requires explicit `TracingLLM.node_call()` with all parameters:

```python
# Current: 8 parameters per call
answer = TRACING_LLM.node_call(
    span_name="synthesizer",
    template_name="synthesizer_prompt", 
    template=template,
    optimizable_key="synthesizer",
    code_key="synthesizer",
    code_fn=synthesizer_node,
    user_query=state.user_query,
    messages=[...],
)
````

---

## 4. Architecture Overview

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Code (LangGraph)                        │
├─────────────────────────────────────────────────────────────────────┤
│  @traced_node("planner")                                            │
│  def planner_node(state): ...                                       │
│                                                                      │
│  graph = build_graph()                                               │
│  instrumented = instrument_graph(graph, trainable=["planner"])      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Trace OTEL Instrumentation Layer                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ TracingLLM   │  │ TelemetryS.. │  │ otel_semconv helpers     │  │
│  │ (enhanced)   │  │ (new)        │  │ - emit_reward()          │  │
│  │              │  │              │  │ - emit_trace()           │  │  # 🔴 (provide a simple manual trace emission helper)
│  │ - node_call  │  │ - start()    │  │ - record_genai_chat()    │  │
│  │ - child LLM  │  │ - flush()    │  │ - set_span_attributes()  │  │
│  │   spans      │  │ - to_mlflow  │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────────┐
            │ OTEL JSON │   │ TGJ Format│   │ MLflow        │
            │ (debug)   │   │ (optim)   │   │ (monitoring)  │
            └───────────┘   └───────────┘   └───────────────┘
                                    │
                                    ▼
            ┌─────────────────────────────────────────────────┐
            │              OPTO Optimizer                      │
            │  (OptoPrimeV2 / TextGrad / etc.)                │
            └─────────────────────────────────────────────────┘
```

### 4.2 Data Flow

```
LangGraph Execution
        │
        ▼
┌───────────────────┐
│ OTEL Spans        │ ← Dual semantic conventions:
│ - param.*         │   • Trace-specific (TGJ-compatible)
│ - gen_ai.*        │   • Agent Lightning-compatible
│ - eval.*          │
└───────────────────┘
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
┌───────────────────┐               ┌───────────────────┐
│ flush_otlp()      │               │ MLflow Export     │
│ → OTLP JSON       │               │ → metrics/artifacts│
└───────────────────┘               └───────────────────┘
        │
        ▼
┌───────────────────┐
│ otlp_to_tgj()     │
│ → Trace-Graph JSON│
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ ingest_tgj()      │
│ → ParameterNode   │
│ → MessageNode     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ optimizer.backward│
│ optimizer.step    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Updated prompts/  │
│ code parameters   │
└───────────────────┘
```

---

## 5. Target API Specification

### 5.1 `instrument_graph()`

**Purpose:** Auto-instrument a LangGraph StateGraph with OTEL tracing.

```python
def instrument_graph(
    graph: StateGraph | CompiledGraph,
    *,
    session: Optional["TelemetrySession"] = None,
    service_name: str = "langgraph-agent",
    trainable_keys: Optional[Set[str]] = None,
    enable_code_optimization: bool = False,
    llm: Optional[Any] = None,
    emit_genai_child_spans: bool = True,
    bindings: Optional[Dict[str, "Binding"]] = None,  # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    in_place: bool = False,  # 🔴 (support non-intrusive optimization without modifying original code)
) -> InstrumentedGraph:
    """
    Wrap a LangGraph with automatic OTEL instrumentation.
    
    Parameters
    ----------
    graph : StateGraph | CompiledGraph
        The LangGraph to instrument.
    session : TelemetrySession, optional
        If provided, reuse this TelemetrySession for OTEL capture and (optionally) MLflow logging; otherwise a new session is created using service_name. # 🔴 (required for clean notebook MLflow + OTEL usage)
    service_name : str
        OTEL service name for trace identification.
    trainable_keys : Set[str], optional
        Node names whose prompts are trainable.
        If None, all nodes are trainable; otherwise provide explicit node names (glob/regex support is optional future work). # 🔴 (default: None => all nodes trainable; defer glob/regex matching beyond M1)
    enable_code_optimization : bool
        If True, emit `param.__code_*` attributes for function source optimization.
    llm : Any, optional
        LLM client to use for nodes. If provided, will be wrapped with TracingLLM.
    emit_genai_child_spans : bool
        If True, emit gen_ai.* child spans for Agent Lightning compatibility.
    bindings : Dict[str, Binding], optional # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
        Explicit mapping from OTEL/TGJ parameter keys (e.g., "planner_prompt", "__code_planner") to getter/setter bindings used by apply_updates(); if None, bindings are auto-derived for common cases (templates dict + wrapped node fns). # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    in_place : bool # 🔴 (support non-intrusive optimization without modifying original code)
        If False (default), avoid **permanent** mutation of the original graph objects: apply updates via bindings/setters and restore wrappers after the run; set True only if you accept in-place monkey-patching for lower overhead (both modes still update parameters **in memory** during optimization). # 🔴 (avoid confusion: “non-intrusive” ≠ “no in-memory updates”)
    
    Returns
    -------
    InstrumentedGraph
        Wrapper with `invoke()`, `stream()`, and access to telemetry session.
    
    Example
    -------
    >>> graph = build_my_langgraph()
    >>> instrumented = instrument_graph(
    ...     graph,
    ...     trainable_keys={"<node_name_1>", "<node_name_2>"},  # 🔴 (example: replace placeholders with real node names to avoid accidental training)
    ...     llm=my_llm_client,
    ...     bindings={"<param_key>": binding},  # e.g., {"planner_prompt": binding}  # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    ... )
    >>> result = instrumented.invoke(initial_state)
    >>> otlp = instrumented.session.flush_otlp()
    """
```

**Output Type:**

```python
@dataclass
class InstrumentedGraph:
    """Instrumented LangGraph wrapper."""
    
    graph: CompiledGraph
    session: TelemetrySession
    tracing_llm: TracingLLM
    
    def invoke(self, state: Any, **kwargs) -> Dict[str, Any]:
        """Execute graph and capture telemetry."""
        ...
    
    def stream(self, state: Any, **kwargs) -> Iterator[Dict[str, Any]]:
        """Stream graph execution with telemetry."""
        ...
```

---

### 5.2 `TelemetrySession`

**Purpose:** Unified session manager for OTEL traces and MLflow integration.

```python
class TelemetrySession:
    """
    Manages OTEL tracing session with export capabilities.
    
    Responsibilities:
    - Initialize and manage TracerProvider + InMemorySpanExporter
    - Provide flush_otlp() for trace extraction
    - Export to MLflow (metrics, artifacts, parameters) # IMPORTANT: see https://github.com/AgentOpt/OpenTrace/blob/feature/mlflow/opto/features/mlflow/autolog.py # 🔴 (see previous work on this support)
    - Support multiple export formats (OTLP JSON, TGJ)
    """
    
    def __init__(
        self,
        service_name: str = "trace-session",
        *,
        mlflow_experiment: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        auto_log_to_mlflow: bool = False,
        record_spans: bool = True,  # 🔴 (allow disabling span recording for minimal/robust runs)
        span_attribute_filter: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,  # 🔴 (allow redaction/truncation and dropping spans to avoid secrets/large payloads)
    ) -> None:
        """
        Initialize telemetry session.
        
        Parameters
        ----------
        service_name : str
            OTEL service/scope name.
        mlflow_experiment : str, optional
            MLflow experiment name. If provided, enables MLflow logging.
        mlflow_run_name : str, optional
            MLflow run name. Auto-generated if not provided.
        auto_log_to_mlflow : bool
            If True, automatically log to MLflow on flush.
        record_spans : bool  # 🔴 (allow disabling span recording for minimal/robust runs)
            If False, disable span recording/export entirely (safe no-op); useful for minimal runs or when only MLflow metrics are desired. # 🔴 (define 'record_spans=False' as safe no-op (no exporter, no OTLP/TGJ output))
        span_attribute_filter : Callable[[str, Dict[str, Any]], Dict[str, Any]], optional # 🔴 (allow redaction/truncation and dropping spans to avoid secrets/large payloads)
            Optional hook to filter/redact/truncate span attributes before they are attached/exported (and to disable recording of some spans by returning {}). # 🔴 (allow redaction/truncation and dropping spans to avoid secrets/large payloads)
        """
    
    @property
    def tracer(self) -> oteltrace.Tracer:
        """Get the OTEL tracer for manual span creation."""
    
    @property
    def exporter(self) -> InMemorySpanExporter:
        """Get the span exporter for direct access."""
    
    def flush_otlp(self, clear: bool = True) -> Dict[str, Any]:
        """
        Flush collected spans to OTLP JSON format.
        
        Parameters
        ----------
        clear : bool
            If True, clear the exporter after flush.
        
        Returns
        -------
        Dict[str, Any]
            OTLP JSON payload compatible with otel_adapter.
        """
    
    def flush_tgj(
        self,
        agent_id_hint: str = "",
        use_temporal_hierarchy: bool = True,
        clear: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Flush collected spans to Trace-Graph JSON format.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of TGJ documents ready for ingest_tgj().
        """
    
    def log_to_mlflow(
        self,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics, parameters, and artifacts to MLflow.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics to log (e.g., {"score": 0.85, "latency_ms": 120}).
        params : Dict[str, Any], optional
            Parameters to log (logged once per run).
        artifacts : Dict[str, str], optional
            Artifacts to log as {name: file_path}.
        step : int, optional
            Step number for metric logging.
        """
    
    def export_run_bundle(
        self,
        output_dir: str,
        *,
        include_otlp: bool = True,
        include_tgj: bool = True,
        include_prompts: bool = True,
    ) -> str:
        """
        Export all session data to a directory bundle.
        
        Returns path to the bundle directory.
        """
```

---

### 5.3 Enhanced `TracingLLM`

**Purpose:** LLM wrapper with dual semantic conventions for Trace and Agent Lightning compatibility.

```python
class TracingLLM:
    """
    Design-3+ wrapper around an LLM client.
    
    Enhancements over current implementation:
+    - (Optional) emits child `openai.chat.completion` spans with gen_ai.* attributes
    - Supports Agent Lightning reward emission
    """
    
    def __init__(
        self,
        llm: Any,
        tracer: oteltrace.Tracer,
        *,
        trainable_keys: Optional[Iterable[str]] = None,
        emit_code_param: Optional[Callable] = None,
        # New parameters for dual semantic conventions
        provider_name: str = "openai",
        llm_span_name: str = "openai.chat.completion",
        emit_llm_child_span: bool = True,
    ) -> None:
        """
        Initialize TracingLLM.
        
        Parameters
        ----------
        llm : Any
            Underlying LLM client (OpenAI-compatible interface).
        tracer : oteltrace.Tracer
            OTEL tracer for span creation.
        trainable_keys : Iterable[str], optional
            Keys that are trainable. Empty string "" matches all.
        emit_code_param : Callable, optional
            Function to emit code parameters: (span, key, fn) -> None.
        provider_name : str
            Provider name for gen_ai.provider.name attribute.
        llm_span_name : str
            Name for child LLM spans (e.g., "openai.chat.completion").
        emit_llm_child_span : bool
            If True, emit Agent Lightning-compatible child spans.
        """
    
    def node_call(
        self,
        *,
        span_name: str,
        template_name: Optional[str] = None,
        template: Optional[str] = None,
        optimizable_key: Optional[str] = None,
        code_key: Optional[str] = None,
        code_fn: Any = None,
        user_query: Optional[str] = None,
        extra_inputs: Optional[Dict[str, str]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **llm_kwargs: Any,
    ) -> str:
        """
        Invoke LLM under an OTEL span with full tracing.
        
        Emits:
        - Parent span with `param.*` and `inputs.*` (Trace-compatible)
        - Child span with `gen_ai.*` (Agent Lightning-compatible)
        
        Returns
        -------
        str
            LLM response content.
        """
```

---

### 5.4 `optimize_graph()`

**Purpose:** One-liner optimization loop.
**TGJ policy (minimal by default):** the optimizer must run on a **minimal TGJ** (`base_graph_doc`) produced from node spans + `param.*` + `eval.*`; rich OTEL details (LLM-call spans, tool spans, etc.) should be stored as OTLP/JSON artifacts and optionally as a separate `log_doc`. # 🔴 (prevent observability spans from polluting the optimization subgraph)
**Optional traces merge logs for inspection only:** if `include_log_doc=True`, create `log_doc` and optionally export `merge_tgj([base_graph_doc, log_doc])` as an artifact for UI/debugging, but do not require merge for optimization correctness. # 🔴 (support rich trace inspection without adding boilerplate to the optimization path)
**Evaluation contract:** `eval_fn` may return a numeric score, a Trace-style string feedback, or a structured dict; the runner normalizes it into a single `EvalResult` and records `eval.score` when numeric is available (required by some optimizers) while always preserving raw feedback as `eval.feedback`/`eval.reasons` artifacts (if only string feedback is available and the optimizer requires a numeric reward, fall back to a secondary `score_fn` or skip the update with a clear warning). # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))

```python
@dataclass  # 🔴 (public contract: EvalResult is the normalized output of eval_fn)
class EvalResult:  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    score: Optional[float] = None  # 🔴 (optional numeric reward (some evals return only text feedback))
    feedback: str = ""  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    metrics: Dict[str, Any] = field(default_factory=dict)  # 🔴 (free-form metrics dict for logging/diagnostics (not required by optimizers))

EvalFn = Callable[[Dict[str, Any]], Union[float, str, Dict[str, Any], EvalResult]]  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
``` 

```python
def optimize_graph(
    graph: InstrumentedGraph | CompiledGraph,
    queries: List[str] | List[Dict[str, Any]],
    *,
    iterations: int = 5,
    optimizer: Optional[OptoPrimeV2] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    eval_fn: Optional[EvalFn] = None,
    initial_templates: Optional[Dict[str, str]] = None,
    bindings: Optional[Dict[str, "Binding"]] = None,  # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    apply_updates: bool = True,  # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    include_log_doc: bool = False,  # 🔴 (opt-in: export/merge rich trace info without impacting minimal optimization TGJ)
    on_iteration: Optional[Callable[[int, List[RunResult], Dict[str, Any]], None]] = None,  # 🔴 (optional progress hook for UI/logging integrations; keep signature stable)
    log_to_mlflow: bool = False,
    mlflow_session: Optional[TelemetrySession] = None,
) -> OptimizationResult:
    """
    Run a complete optimization loop on a LangGraph.
    
    Parameters
    ----------
    graph : InstrumentedGraph | CompiledGraph
        The instrumented graph to optimize.
    queries : List[str] | List[Dict[str, Any]]
        Test queries or full state dicts for each run.
    iterations : int
        Number of optimization iterations.
    optimizer : OptoPrimeV2, optional
        Pre-configured optimizer. Created if not provided.
    optimizer_kwargs : Dict[str, Any], optional
        Arguments for optimizer creation if not provided.
    eval_fn : EvalFn, optional
        Custom evaluation function. Can return float score, string feedback, or structured dict; normalized into EvalResult (Trace-style feedback + TextGrad-friendly). # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    initial_templates : Dict[str, str], optional
        Initial prompt templates. Uses graph defaults if not provided.
    bindings : Dict[str, Binding], optional # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
        Mapping from OTEL/TGJ parameter keys to concrete setter/getter bindings (used by apply_updates to update prompts/functions/graph knobs deterministically). # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    apply_updates : bool # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
        If True (default), apply updates each iteration via apply_updates(updates, bindings); if False, return updates only (caller applies manually). # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    include_log_doc : bool # 🔴 (opt-in: export/merge rich trace info without impacting minimal optimization TGJ)
        If True, emit an additional `log_doc` (full spans) and optionally export `merge_tgj([base_graph_doc, log_doc])` as an artifact for inspection/UI; optimization itself still uses `base_graph_doc`. # 🔴 (keep optimizer path minimal while still enabling rich trace inspection)
    on_iteration : Callable, optional
        Callback after each iteration: (iter_num, runs, updates_dict) -> None (updates_dict keys match `param.<key>` / bindings keys). # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    log_to_mlflow : bool
        If True, log metrics to MLflow after each iteration.
    mlflow_session : TelemetrySession, optional
        If provided, overrides graph.session for MLflow logging only; otherwise optimize_graph logs via InstrumentedGraph.session when available. # 🔴 (clarifies single-session intent)
    
    Returns
    -------
    OptimizationResult
        Contains final parameters (templates/code/graph knobs via bindings), score history, best iteration, etc. # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    """

@dataclass
class OptimizationResult:
    """Result of optimize_graph()."""
    
    baseline_score: float
    best_score: float
    best_iteration: int
    best_updates: Dict[str, Any]  # raw best update dict (param-keyed)  # 🔴 (persist raw param-keyed updates for reproducibility/debugging)
    final_parameters: Dict[str, Any]  # resolved via bindings (prompts/code/graph knobs)  # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    score_history: List[float]
    all_runs: List[List[RunResult]]
    optimizer: OptoPrimeV2
```

#### 5.4.1 Bindings + `apply_updates()` (robust update mapping) # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
Optimizer updates are keyed by OTEL/TGJ parameter names (e.g., `param.planner_prompt` → key `planner_prompt`, `param.__code_planner` → key `__code_planner`). To apply them deterministically (and to support non-intrusive optimization), we require explicit bindings from key → (get,set) and a single `apply_updates(...)` entrypoint. # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
- `bindings` is mandatory for *non-intrusive* optimization (imported graphs / module-level variables); for inline demos we can auto-derive it from the templates dict + wrapped node callables. # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
(Implementation note: keys must match the exact `template_name` / `code_key` used in `param.*` so we never rely on fragile string parsing.) # 🔴 (deterministic mapping: param keys must exactly match bindings to avoid heuristics)

```python
# opto/trace/io/bindings.py  # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
from dataclasses import dataclass  # 🔴 (spec snippet imports (exact import list can be adjusted in implementation))
from typing import Any, Callable, Dict, Literal  # 🔴 (spec snippet typing imports (kept explicit for copy/paste clarity))

@dataclass  # 🔴 (Binding is a small public primitive (needed by apply_updates and instrument_graph))
class Binding:  # 🔴 (Binding keys must match TGJ/OTEL param keys (prompt/code/graph knobs))
    """Minimal get/set binding for a trainable target."""  # 🔴 (binding contract: minimal get/set indirection for non-intrusive updates)
    get: Callable[[], Any]  # 🔴 (getter returns current value for logging + optimizer initialization)
    set: Callable[[Any], None]  # 🔴 (setter applies updated value in-memory (prompts/code/graph knobs))
    kind: Literal["prompt", "code", "graph"] = "prompt"  # 🔴 (binding kind supports prompt/code/graph validation + reporting)

def apply_updates(  # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    updates: Dict[str, Any],  # 🔴 (updates dict is keyed by param names (without 'param.' prefix))
    bindings: Dict[str, Binding],  # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    *,  # 🔴 (keyword-only args: avoid accidental positional mis-order in API)
    strict: bool = True,  # 🔴 (strict=True by default to fail fast on missing/unknown bindings)
) -> None:  # 🔴 (apply_updates is side-effecting (mutates bound targets in memory))
    """Apply optimizer updates using the binding map (raise if strict and a key is missing)."""  # 🔴 (single entrypoint for deterministic update application across prompts/code/graph)
    ...  # 🔴 (implementation: loop keys, set via bindings, raise on missing if strict)
``` 

---

### 5.5 OTEL Semantic Convention Helpers

**Purpose:** Emit spans compatible with both Trace and Agent Lightning.

```python
# opto/trace/io/otel_semconv.py

def set_span_attributes(span, attrs: Dict[str, Any]) -> None:
    """
    Set multiple span attributes at once.
    
    Handles:
    - dict/list → JSON string
    - None values → skipped
    """

def record_genai_chat(
    span,
    *,
    provider: str,
    model: str,
    input_messages: List[Dict[str, Any]],
    output_text: Optional[str] = None,
    request_type_compat: str = "chat.completion",
) -> None:
    """
    Record OTEL GenAI semantic convention attributes.
    
    Emits:
    - gen_ai.operation.name
    - gen_ai.provider.name
    - gen_ai.request.model
    - gen_ai.input.messages (JSON)
    - gen_ai.output.messages (JSON)
    """

def emit_reward(  # 🔴 (Agent Lightning-compatible reward span helper (naming + attrs contract))
    *,
    value: float,
    name: str = "final_score",
    tracer_name: str = "opto.trace",
    index: int = 0,
    span_name: str = "agentlightning.annotation",
    extra_attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit a reward span compatible with Agent Lightning semconv.
    
    Creates child span with:
    - agentlightning.reward.<i>.name
    - agentlightning.reward.<i>.value
    """
emit_agentlightning_reward = emit_reward  # backwards-compat alias  # 🔴 (align naming with standard emit_reward while keeping backward-compatible alias)

def emit_trace(  # 🔴 (provide a simple manual trace emission helper)
    *,  # 🔴 (keyword-only to keep callsites explicit and stable)
    name: str,  # 🔴 (required span/event name (used as OTEL span name))
    attrs: Optional[Dict[str, Any]] = None,  # 🔴 (optional attributes payload (kept small; can be filtered/redacted))
    tracer_name: str = "opto.trace",  # 🔴 (tracer namespace for manual spans (matches TelemetrySession default))
) -> None:  # 🔴 (emit_trace is intentionally side-effecting (records OTEL span/event))
    """Emit a lightweight OTEL span (or span event) for arbitrary debug/optimization signals."""  # 🔴 (manual lightweight span for custom signals (debug/optimization annotations))
    ...  # 🔴 (implementation: start span, set attrs, end span (or add event); emit as child span under current node span when possible)
```

---

### 5.6 MLflow Integration

**Purpose:** Standardized logging to MLflow for monitoring.

```python
# opto/trace/io/mlflow_logger.py

class MLflowTelemetryLogger(BaseLogger):
    """
    Logger that exports telemetry to MLflow.
    
    Integrates with TelemetrySession to provide:
    - Metric logging (scores, latencies, token counts)
    - Parameter logging (prompt templates, model configs)
    - Artifact logging (OTLP JSON, TGJ, optimization logs)
    """
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        log_dir: str = "./logs",
        **kwargs,
    ) -> None:
        """Initialize MLflow logger."""
    
    def log(
        self,
        name: str,
        data: Any,
        step: int,
        **kwargs,
    ) -> None:
        """Log metric/param to MLflow."""
    
    def log_otlp_artifact(
        self,
        otlp: Dict[str, Any],
        artifact_name: str = "otlp_trace.json",
    ) -> None:
        """Log OTLP trace as artifact."""
    
    def log_tgj_artifact(
        self,
        tgj_docs: List[Dict[str, Any]],
        artifact_name: str = "trace_graph.json",
    ) -> None:
        """Log TGJ documents as artifact."""
    
    def log_templates(
        self,
        templates: Dict[str, str],
        step: Optional[int] = None,
    ) -> None:
        """Log current prompt templates as parameters or artifacts."""
```

---

## 6. Module Modifications

### 6.1 Files to Create

| File                               | Purpose                                                                |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `opto/trace/io/otel_semconv.py`    | Semantic convention helpers                                            |
| `opto/trace/io/mlflow_logger.py`   | MLflow integration                                                     |
| `opto/trace/io/instrumentation.py` | `instrument_graph()` and `InstrumentedGraph`                           |
| `opto/trace/io/optimization.py`    | `optimize_graph()` and related                                     |
| `opto/trace/io/bindings.py`        | `Binding` + `apply_updates()` mapping layer (param key → get/set) # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates) |

### 6.2 Files to Modify

| File                                      | Changes                                          |
| ----------------------------------------- | ------------------------------------------------ |
| `opto/trace/io/langgraph_otel_runtime.py` | Optional child span emission (gen_ai.* compatibility) |
| `opto/trace/io/otel_adapter.py`           | Do not advance temporal chain on OTEL child spans (`parentSpanId` present) |
| `opto/trace/io/__init__.py`               | Export new public APIs                           |
| `opto/trainer/loggers.py`                 | Add `MLflowTelemetryLogger`                      |

### 6.3 Detailed Changes to `otel_adapter.py`  # 🔴 (modification is already available in commit https://github.com/doxav/NewTrace/commit/237abb320b201abbd45a36f68b03ad951cd6011c)

```python
# In otlp_traces_to_trace_json(), do not advance temporal chaining on OTEL child spans:
psid = sp.get("parentSpanId")
orig_has_parent = bool(psid)
...
# Before:
#     prev_span_id = sid
# After:
if not orig_has_parent:
    prev_span_id = sid
```

---

## 7. Implementation Plan

### Phase 1: Core Infrastructure (Priority: High)

| Task                                         | Effort | Dependencies                                  |
| -------------------------------------------- | ------ | --------------------------------------------- |
| Create `otel_semconv.py` with helpers        | Xh     | None                                          |
| Enhance `TracingLLM` with child spans        | Xh     | otel_semconv.py                               |
| Update `otel_adapter.py` for temporal_ignore | 0h     | None 🔴 (available in commit https://github.com/doxav/NewTrace/commit/237abb320b201abbd45a36f68b03ad951cd6011c)                                          |
| Create `TelemetrySession` class              | Xh     | langgraph_otel_runtime.py                     |
| Add `bindings.py` (Binding + apply_updates)  | Xh     | optimize_graph(), instrument_graph() # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates) |

### Phase 2: High-Level API (Priority: High)

| Task                               | Effort | Dependencies                 |
| ---------------------------------- | ------ | ---------------------------- |
| Implement `instrument_graph()`     | Xh     | TelemetrySession, TracingLLM |
| Implement `optimize_graph()`   | Xh     | instrument_graph             |
| Create `InstrumentedGraph` wrapper | Xh     | instrument_graph             |

### Phase 3: MLflow Integration (Priority: Medium)

| Task                            | Effort | Dependencies          |
| ------------------------------- | ------ | --------------------- |
| Create `MLflowTelemetryLogger` (OTEL/MLFlow)  | Xh     | BaseLogger            | # 🔴 (to be cleared: identical or differences?)
| Integrate with TelemetrySession | Xh     | MLflowTelemetryLogger |
| Add artifact export helpers     | Xh     | MLflowTelemetryLogger |

### Phase 4: Testing & Documentation (Priority: High)

| Task                          | Effort | Dependencies |
| ----------------------------- | ------ | ------------ |
| Unit tests for new modules    | Xh     | All modules  |
| Integration test with StubLLM | Xh     | All modules  |
| Update README and examples    | Xh     | All modules  |
| Prototype notebook            | Xh     | All modules  |

---

## 8. Agent Lightning Comparison

### 8.1 API Comparison Table

| Aspect                         | Agent Lightning                             | Trace (New API)                                                                                                                                                           |
| ------------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Initialization**             | `import agentlightning as agl`              | `from opto.trace.io import instrument_graph`                                                                                                                              |
| **Agent / Graph Definition**   | `@rollout` decorator                        | `instrument_graph(graph, ...)` (generic; supports `in_place=False` for non-intrusive wrapping) # 🔴 (support non-intrusive optimization without modifying original code)                                                                       |
| **Trainable Fn/Var**           | `initial_resources={...}` / agent args      | Trace trainables: `trace.node(var, trainable=True)` and/or `trace.bundle(trainable=..., traceable_code=..., allow_external_dependencies=...)(fn)` + `bindings={...}` # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates) |
| **LLM Calls**                  | Auto-instrumented via proxy                 | `TracingLLM.node_call()` wrapper                                                                                                                                          |
| **Custom trace emission**      | `emit_annotation(...)` / `emit_reward(...)` | `emit_trace(name, attrs)` + `TelemetrySession.tracer.start_as_current_span(...)` (manual spans/events) # 🔴 (provide a simple manual trace emission helper)                                                               |
| **Reward / feedback emission** | `emit_reward(value)`                        | `emit_reward(value, name)` (Agent Lightning semconv; `emit_agentlightning_reward` remains as an alias) # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))                                                               |
| **Bindings & update apply**    | Trainer updates resources internally        | `apply_updates(updates, bindings)` (keys align with `param.<key>`) # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)                                                                                                   |
| **Training Loop**              | `Trainer.fit(agent, dataset)`               | `optimize_graph(graph, queries)`                                                                                                                                      |
| **Optimization**               | RL/APO/SFT algorithms                       | TGJ → OPTO (OptoPrimeV2, TextGrad)                                                                                                                                        |
| **Span Format**                | `gen_ai.*` conventions                      | Dual: `param.*` + `gen_ai.*` (+ optional `agentlightning.reward.*`) # 🔴 (confirm we emit both param.* (optimizer) and gen_ai.* (observability) semconv)                                                                                                  |

### 8.2 Code Comparison

**Agent Lightning (conceptual):**

```python
import agentlightning as agl
from agentlightning import emit_reward, rollout

@rollout
def agent(task: dict, prompt_template: str):
    # LLM calls auto-instrumented
    result = llm.chat(messages=[...])
    emit_reward(0.82)
    return result

trainer = agl.Trainer(
    algorithm=agl.APO(),
    initial_resources={"prompt_template": template}
)
trainer.fit(agent=agent, train_dataset=tasks)
```

**Trace (New API):**

```python
from opto.trace.io import instrument_graph, optimize_graph

# One-time instrumentation
graph = build_my_langgraph()
instrumented = instrument_graph(
    graph,
    trainable_keys={"<node_name_1>", "<node_name_2>"},  # 🔴 (example: replace placeholders with real node names to avoid accidental training)
    llm=my_llm,
)

# One-liner optimization
result = optimize_graph(
    instrumented,
    queries=test_queries,
    iterations=5,
)
```

### 8.3 Key Differences

| Feature                 | Agent Lightning         | Trace                                 |
| ----------------------- | ----------------------- | ------------------------------------- |
| **Optimization Target** | Prompt templates via RL | Prompts + code via gradient descent   |
| **Trace Format**        | Custom span storage     | OTLP → TGJ → Trace nodes              |
| **Feedback Signal**     | Reward values           | Structured feedback (score + reasons) |
| **Code Optimization**   | Not supported           | Supported via `__code_*` params       |
| **Graph Support**       | Generic agents          | LangGraph-native                      |

---

## 9) Notebooks (Deliverables from M1 onward)
Lock notebook deliverables per milestone to keep validation reviewable. # 🔴 (deliverables mirror Trace‑Bench M0 notebook policy)

Rule: each milestone delivers a notebook that is: # 🔴 (keep validation reviewable without running local code)
- committed with **executed outputs** (reviewers can inspect results without re-running) # 🔴 (avoid out-of-band validation)
- includes an **“Open in Colab”** badge in the first markdown cell (if repo policy permits) # 🔴 (one-click reproduction)
- writes outputs to a deterministic folder (e.g., `./logs/notebooks/<milestone>/`) and keeps artifacts small # 🔴 (keeps PRs reviewable)

**Notebooks**
- **M1**: `notebooks/01_m1_instrument_and_optimize.ipynb` — runs in two modes: (a) StubLLM mode (no keys; deterministic) and (b) Live LLM mode (requires `OPENROUTER_API_KEY`, check colab secrets) to validate real-provider tracing + optimization; show at least one `param.*` prompt value changes across iterations. # 🔴 (CI uses stub; notebook validates live)
- **M2**: `notebooks/02_m2_unified_telemetry.ipynb` — demonstrate unified telemetry surface across node spans + trainer metrics + optimizer logs (export at least one optimizer summary artifact + one metric series). # 🔴 (standard OTEL logger across Trace)
- **M3**: `notebooks/03_m3_mlflow_monitoring.ipynb` — demonstrate MLflow run containing metrics in general (any trace code) + OTLP/TGJ artifacts by constructing a `TelemetrySession(mlflow_experiment=..., auto_log_to_mlflow=True)` and passing it to `instrument_graph(session=...)` (so the same session captures OTEL and logs to MLflow). # 🔴 (monitoring integration)
---

## 10) Acceptance Criteria (SMART, verifiable)
Milestone-based checks (SMART) replacing the removed "Validation Criteria" table. # 🔴 (keeps validation minimal and verifiable)

**Milestone definitions used in this plan:** # 🔴 (align acceptance wording with delivery phases)
- **M0**: Technical plan accepted (this document) # 🔴 (locks contracts before implementation)
- **M1**: Drop-in instrumentation + optimization driver (end-to-end): `instrument_graph` + `optimize_*` + demo refactor + Notebook M1. # 🔴 (prove core value early)
- **M2**: Standard telemetry across Trace components (trainer/optimizer/node): unified telemetry surface + Notebook M2. # 🔴 (standard OTEL logger)
- **M3**: MLflow monitoring + hardening + Notebook M3. # 🔴 (monitoring + artifacts)
- **M4 (optional)**: extra docs/notebooks polish if time. # 🔴 (do not block contract completion)

### M0 (this document)
- **No unresolved review markers:** `grep -n "review required" T1_technical_plan_v3.md` returns **0** matches. # 🔴 (ensures the plan is unambiguous)
- **Navigation updated:** Table of contents includes sections 9–12 and anchors resolve in GitHub markdown preview. # 🔴 (prevents review friction)

### M1 (instrumentation + optimization driver, end-to-end)
- **OTLP export works:** after emitting ≥1 manual span, `TelemetrySession.flush_otlp(clear=True)` returns OTLP JSON with ≥1 span and a second flush returns 0 spans (cleared). # 🔴 (verifies exporter + clear semantics)
- **TGJ conversion works:** `flush_tgj()` (or `otlp_to_tgj()`) produces TGJ docs that can be ingested by `ingest_tgj()` (or pass a schema validation) without exceptions. # 🔴 (verifies optimizer-compatible trace output)
- **Temporal chaining contract:** a unit test proves OTEL child spans (spans with `parentSpanId`) do **not** advance TGJ temporal chaining (i.e., they cannot become temporal parents of subsequent top-level spans). # 🔴 (prevents child spans from breaking sequential node chaining)
- **Bindings apply deterministically:** `apply_updates({...}, bindings, strict=True)` updates bound values in memory; missing keys raise a clear error; `strict=False` ignores unknown keys. # 🔴 (robust update application)
- **End-to-end update path (CI/StubLLM):** using a minimal LangGraph and StubLLM, `optimize_* (iterations>=2, apply_updates=True)` produces `best_updates` where keys ⊆ `bindings.keys()` and at least one bound prompt value changes between iteration 0 and final. # 🔴 (deterministic CI proof)
- **Notebook live validation:** with `OPENROUTER_API_KEY` set (check colab secrets), Notebook M1 runs the same loop against a real provider (small dataset; deterministic settings) and produces OTLP+TGJ artifacts containing at least one LLM call span plus `param.*` attributes. # 🔴 (real-world proof)
- **Tests + notebook gate:** new public APIs introduced for M1 have ≥1 pytest each; CI runs stub-only; Notebook M1 includes an “Open in Colab” badge and a live-run section. # 🔴 (hard requirement)
- **Notebook - Live run constraints:** live mode must use a tiny dataset (≤3 items), deterministic settings (`temperature=0`, fixed model name), and a hard budget guard (e.g., max tokens per call) to keep cost predictable and reduce output variance. **No secrets committed:** Notebook must read keys from environment / Colab secrets; no API keys or sensitive prompts are committed in outputs. # 🔴 (simple acceptance criteria + security)

 
### M2 (standard telemetry across Trace components)
- **Unified telemetry surface:** trainer metrics (BaseLogger), optimizer summary logs, and node spans can be exported through one telemetry surface (`TelemetrySession` / `UnifiedTelemetry`). # 🔴 (deliverable B)
- **Optimizer logs exported:** at least one optimizer summary artifact is exported (file or MLflow artifact later) and at least one metric series is emitted (e.g., `score`, `loss`, `latency_ms`). # 🔴 (monitoring completeness)
- **Non-intrusive instrumentation (if claimed):** `instrument_graph(..., in_place=False)` restores wrapped callables after run (no persistent graph mutation). # 🔴 (prevents accidental graph mutation)
- **Tests + notebook gate:** new public behaviors in M2 have pytest coverage, and Notebook M2 demonstrates unified telemetry with executed outputs + Colab badge. # 🔴 (hard requirement)

### M3 (MLflow + export bundle)
- **MLflow is optional but robust:** when MLflow is unavailable/misconfigured, the run continues and logs a warning (no hard crash). # 🔴 (optional dependency hardening)
- **Bundle export is portable:** `export_run_bundle(output_dir, include_otlp=True, include_tgj=True, include_prompts=True)` creates a directory containing OTLP JSON, TGJ JSON, and a prompt snapshot file. # 🔴 (portable artifacts for review/debugging)

### M4 (tests + docs + notebooks)
- **CI green:** unit + integration tests referenced in this plan pass in CI (stub mode; no paid LLM calls). # 🔴 (keeps PR review cheap and deterministic)
- **Docs complete:** README includes a minimal quickstart for `instrument_graph()` + `optimize_graph()`, plus a short “Bindings & apply_updates” guide. # 🔴 (developer adoption)
- **Notebooks delivered:** notebooks listed in Section 9 run end-to-end in StubLLM mode (no keys) AND include a live-provider section that runs when `OPENROUTER_API_KEY` is set. # 🔴 (reviewable + real validation)
---

## 11. Test & Validation Plan

### 11.1 Unit Tests

| Test File                         | Coverage                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------- |
| `tests/test_otel_semconv.py`      | Semantic convention helpers                                                      |
| `tests/test_tracing_llm.py`       | TracingLLM with child spans                                                      |
| `tests/test_telemetry_session.py` | Session management and export (incl span_attribute_filter) # 🔴 (allow redaction/truncation and dropping spans to avoid secrets/large payloads)                  |
| `tests/test_instrumentation.py`   | instrument_graph() (incl bindings/in_place) # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)                                 |
| `tests/test_optimization.py`      | optimize_graph() (incl EvalFn returning str/dict/float + apply_updates) # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates) |

### 11.2 Integration Tests

```python
# tests/test_integration_stubllm.py

def test_full_optimization_flow_with_stubllm():
    """
    End-to-end test using StubLLM (no API calls).
    
    1. Build a simple LangGraph
    2. Instrument with instrument_graph()
    3. Run optimize_graph() for 2 iterations
    4. Verify:
       - OTLP spans contain expected attributes
       - TGJ conversion produces valid nodes
       - Optimizer produces parameter updates
       - Updates are applied via bindings (or returned if apply_updates=False)  # 🔴 (necessary binding between trace OTEL names and real variables/functions to allow optimizer updates)
    """
```

### 11.3 StubLLM for Testing

```python
class StubLLM:
    """Deterministic LLM stub for testing."""
    
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.call_count = 0
    
    def __call__(self, messages, **kwargs):
        self.call_count += 1
        # Return deterministic response based on input
        user_msg = messages[-1]["content"] if messages else ""
        
        # Match against known patterns
        for pattern, response in self.responses.items():
            if pattern in user_msg:
                return self._make_response(response)
        
        # Default response
        return self._make_response('{"result": "stub response"}')
    
    def _make_response(self, content):
        return type("R", (), {
            "choices": [type("C", (), {
                "message": type("M", (), {"content": content})()
            })()]
        })()
```

---

## 12. Appendix: Prototype Snippet

This prototype demonstrates the target API working with a StubLLM.

```python
"""
Prototype: instrument_graph + optimize_graph with StubLLM
============================================================

Run this to validate the API design before full implementation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Literal, Union
import json

# ============================================================
# STUB IMPLEMENTATIONS (to be replaced by real modules)
# ============================================================

class StubLLM:
    """Deterministic LLM for testing."""
    
    def __init__(self):
        self.call_count = 0
    
    def __call__(self, messages, **kwargs):
        self.call_count += 1
        user_msg = messages[-1].get("content", "") if messages else ""
        
        # Generic heuristic responses (demo-only)
        if "evaluate" in user_msg.lower():
            return self._resp('{"answer_relevance": 0.8, "groundedness": 0.7, "plan_quality": 0.9, "reasons": "Good structure"}')
        return self._resp("stub response")
    
    def _resp(self, content):
        return type("R", (), {
            "choices": [type("C", (), {
                "message": type("M", (), {"content": content})()
            })()]
        })()


@dataclass
class EvalResult:  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    score: Optional[float] = None  # 🔴 (optional numeric reward (prototype supports text-only eval too))
    feedback: str = ""  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    metrics: Dict[str, Any] = field(default_factory=dict)  # 🔴 (prototype: metrics capture parsed JSON fields for logging)


EvalFn = Callable[[Dict[str, Any]], Union[float, str, Dict[str, Any], EvalResult]]  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))


def default_eval_fn(payload: Dict[str, Any]) -> EvalResult:  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    """Default eval: accept numeric score or JSON dict; always preserve textual feedback."""  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    raw = payload.get("raw_eval", "")  # 🔴 (prototype: accept evaluator output as number, JSON string, or dict)
    if isinstance(raw, (int, float)):  # 🔴 (if numeric, treat as score directly (no JSON parsing))
        return EvalResult(score=float(raw), feedback="", metrics={})  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    if isinstance(raw, str):  # 🔴 (if string, attempt JSON parse; else treat as feedback text)
        try:  # 🔴 (prototype: JSON parse is best-effort (never crash optimization loop))
            d = json.loads(raw)  # 🔴 (parse JSON-formatted evaluator output when present)
            score = sum([d.get("answer_relevance", 0.5), d.get("groundedness", 0.5), d.get("plan_quality", 0.5)]) / 3  # 🔴 (demo-only scoring heuristic (simple average; weights TBD))
            return EvalResult(score=float(score), feedback=str(d.get("reasons", "")), metrics=d)  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
        except Exception:  # 🔴 (fallback: preserve raw string as feedback when parse fails)
            return EvalResult(score=None, feedback=raw, metrics={})  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    if isinstance(raw, dict):  # 🔴 (if dict, treat as metrics payload and stringify feedback)
        return EvalResult(score=None, feedback=str(raw), metrics=raw)  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    return EvalResult(score=None, feedback=str(raw), metrics={})  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))


# Minimal TelemetrySession stub
class TelemetrySession:
    def __init__(self, service_name: str = "test", *, record_spans: bool = True, span_attribute_filter: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None):  # 🔴 (allow disabling span recording for minimal/robust runs)
        self.spans = []
        self.service_name = service_name
        self.record_spans = record_spans  # 🔴 (allow disabling span recording for minimal/robust runs)
        self.span_attribute_filter = span_attribute_filter  # 🔴 (allow redaction/truncation and dropping spans to avoid secrets/large payloads)
    
    def record_span(self, name: str, attrs: Dict[str, Any]):  # 🔴 (stub-only: collect spans in memory to emulate exporter behaviour)
        if not self.record_spans:  # 🔴 (allow disabling span recording for minimal/robust runs)
            return  # 🔴 (early-exit when span recording is disabled (safe no-op mode))
        if self.span_attribute_filter is not None:  # 🔴 (allow redaction/truncation and dropping spans to avoid secrets/large payloads)
            attrs = self.span_attribute_filter(name, dict(attrs))  # 🔴 (allow redaction/truncation and dropping spans to avoid secrets/large payloads)
        if attrs == {}:  # allow filter to drop span # 🔴 (allow filter hook to drop spans by returning an empty dict)
            return  # 🔴 (early-exit when span is dropped by filter (do not record))
        self.spans.append({"name": name, "attributes": attrs})
    
    def flush_otlp(self) -> Dict[str, Any]:
        otlp_spans = [
            {
                "spanId": f"span_{i}",
                "name": s["name"],
                "attributes": [
                    {"key": k, "value": {"stringValue": str(v)}}
                    for k, v in s["attributes"].items()
                ]
            }
            for i, s in enumerate(self.spans)
        ]
        self.spans.clear()
        return {
            "resourceSpans": [{
                "resource": {"attributes": []},
                "scopeSpans": [{
                    "scope": {"name": self.service_name},
                    "spans": otlp_spans
                }]
            }]
        }


# Minimal TracingLLM stub
class TracingLLM:
    def __init__(self, llm, session: TelemetrySession, trainable_keys=None):
        self.llm = llm
        self.session = session
        self.trainable_keys = trainable_keys  # keep None meaning "all trainable" # 🔴 (prototype: None => all nodes trainable; matches instrument_graph default)
    
    def node_call(self, *, span_name, template_name=None, template=None,
                  optimizable_key=None, messages=None, **kwargs) -> str:
        # Record span
        attrs = {}
        if template_name and template:
            attrs[f"param.{template_name}"] = template
            # If trainable_keys is None => all trainable; else explicit membership # 🔴 (emit explicit trainable marker for TGJ/optimizer consumption)
            trainable = True if self.trainable_keys is None else (optimizable_key in self.trainable_keys)  # 🔴 (trainable flag depends on trainable_keys (None means all))
            attrs[f"param.{template_name}.trainable"] = trainable  # 🔴 (record trainable flag alongside param value for debuggability)
        attrs["gen_ai.model"] = "stub"
        attrs["inputs.gen_ai.prompt"] = messages[-1]["content"] if messages else ""
        
        self.session.record_span(span_name, attrs)
        
        # Call LLM
        return self.llm(messages=messages, **kwargs).choices[0].message.content


# ============================================================
# PROTOTYPE: instrument_graph()
# ============================================================

@dataclass
class InstrumentedGraph:
    """Instrumented LangGraph wrapper."""
    
    graph: Any  # The actual LangGraph
    session: TelemetrySession
    tracing_llm: TracingLLM
    templates: Dict[str, str] = field(default_factory=dict)
    eval_fn: EvalFn = default_eval_fn  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute graph with telemetry capture."""
        # For prototype, simulate a minimal flow without hard-coding node names as "the API" (only the demo does). # 🔴 (prototype-only flow; real implementation wraps arbitrary node callables)
        query = state.get("query", "")  # 🔴 (prototype state shape; real graphs use user-defined state schema)
        
        # Simulate a generic "answer" node (demo-only)
        answer = self.tracing_llm.node_call(
            span_name="answer_node",
            template_name="answer_prompt",
            template=self.templates.get("answer_prompt", "Default answer template"),
            optimizable_key="answer_node",
            messages=[{"role": "user", "content": f"Answer: {query}"}],
        )
        
        # Simulate evaluator
        raw_eval = self.tracing_llm.node_call(
            span_name="evaluator",
            messages=[{"role": "user", "content": f"Evaluate: {answer}"}],
        )
        
        er = self.eval_fn({"query": query, "answer": answer, "raw_eval": raw_eval})  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
        if isinstance(er, (int, float)):  # 🔴 (normalize eval_fn return types into EvalResult (float/str/dict))
            er = EvalResult(score=float(er), feedback="", metrics={})  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
        elif isinstance(er, str):  # 🔴 (normalize eval_fn return types into EvalResult (float/str/dict))
            er = EvalResult(score=None, feedback=er, metrics={})  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
        elif isinstance(er, dict):  # 🔴 (normalize eval_fn return types into EvalResult (float/str/dict))
            er = EvalResult(score=er.get("score"), feedback=str(er.get("feedback", "")), metrics=er)  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
        
        # Record eval span (score optional; feedback always preserved) # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
        eval_attrs = {"eval.feedback": er.feedback, "eval.reasons": er.feedback}  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
        if er.score is not None:  # 🔴 (only set eval.score when numeric is available (optimizer requirement))
            eval_attrs["eval.score"] = str(er.score)  # 🔴 (record numeric eval.score for optimizers that require rewards)
        self.session.record_span("evaluator", eval_attrs)  # 🔴 (record eval attributes as a separate span/event for traceability)
        
        return {"answer": answer, "score": er.score, "feedback": er.feedback, "metrics": er.metrics}  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))


def instrument_graph(
    graph: Any,
    *,
    service_name: str = "langgraph-agent",
    trainable_keys: Optional[set] = None,
    llm: Optional[Any] = None,
    initial_templates: Optional[Dict[str, str]] = None,
    eval_fn: Optional[EvalFn] = None,  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
) -> InstrumentedGraph:
    """
    Wrap a LangGraph with automatic OTEL instrumentation.
    
    This is the main entry point for the new API.
    """
    session = TelemetrySession(service_name)
    
    tracing_llm = TracingLLM(
        llm=llm or StubLLM(),
        session=session,
        trainable_keys=trainable_keys,  # None means "all trainable"; no hard-coded planner/synthesizer # 🔴 (prototype: trainable_keys=None means train all prompts by default)
    )
    
    return InstrumentedGraph(
        graph=graph,
        session=session,
        tracing_llm=tracing_llm,
        templates=initial_templates or {},
        eval_fn=eval_fn or default_eval_fn,  # 🔴 (support evaluation as score or string feedback (Trace/TextGrad compatible))
    )
