# T1 Technical Plan: LangGraph OTEL Instrumentation API

**Version:** 1.0  
**Date:** February 6, 2026  
**Author:** Jahanzeb Javed  
**Status:** Draft for Review

This technical plan is **reusable for any LangGraph**, not tied to a specific demo graph (e.g. planner/researcher/synthesizer/evaluator). For before/after boilerplate diff, API matrix by optimization mode, OTEL+MLflow telemetry plan, OTEL span contract, tests/notebook plan, and notebook requirements (Colab, Secrets, Drive, GitHub), see the [README](../README.md).

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
9. [Test & Validation Plan](#9-test--validation-plan)
10. [Appendix: Prototype Snippet](#10-appendix-prototype-snippet)

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
| `optimize_langgraph()` | One-liner optimization loop |
| `emit()` helpers | Manual telemetry emission (rewards, custom spans) |

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
| **Code Patching** | ~80 | `_apply_code_update`, `_emit_code_param` |
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
```

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
│  │              │  │              │  │ - record_genai_chat()    │  │
│  │ - node_call  │  │ - start()    │  │ - set_span_attributes()  │  │
│  │ - child LLM  │  │ - flush()    │  │                          │  │
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
    service_name: str = "langgraph-agent",
    trainable_keys: Optional[Set[str]] = None,
    enable_code_optimization: bool = False,
    llm: Optional[Any] = None,
    emit_genai_child_spans: bool = True,
) -> InstrumentedGraph:
    """
    Wrap a LangGraph with automatic OTEL instrumentation.
    
    Parameters
    ----------
    graph : StateGraph | CompiledGraph
        The LangGraph to instrument.
    service_name : str
        OTEL service name for trace identification.
    trainable_keys : Set[str], optional
        Node names whose prompts are trainable. If None, all nodes are trainable.
        Use empty string "" to match all nodes.
    enable_code_optimization : bool
        If True, emit `param.__code_*` attributes for function source optimization.
    llm : Any, optional
        LLM client to use for nodes. If provided, will be wrapped with TracingLLM.
    emit_genai_child_spans : bool
        If True, emit gen_ai.* child spans for Agent Lightning compatibility.
    
    Returns
    -------
    InstrumentedGraph
        Wrapper with `invoke()`, `stream()`, and access to telemetry session.
    
    Example
    -------
    >>> graph = build_my_langgraph()
    >>> instrumented = instrument_graph(
    ...     graph,
    ...     trainable_keys={"planner", "executor", "synthesizer"},
    ...     llm=my_llm_client,
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
    - Export to MLflow (metrics, artifacts, parameters)
    - Support multiple export formats (OTLP JSON, TGJ)
    """
    
    def __init__(
        self,
        service_name: str = "trace-session",
        *,
        mlflow_experiment: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        auto_log_to_mlflow: bool = False,
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
    - Emits child `openai.chat.completion` spans with gen_ai.* attributes
    - Marks child spans with `trace.temporal_ignore=True` for TGJ stability
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

### 5.4 `optimize_langgraph()`

**Purpose:** One-liner optimization loop.

```python
def optimize_langgraph(
    graph: InstrumentedGraph | CompiledGraph,
    queries: List[str] | List[Dict[str, Any]],
    *,
    iterations: int = 5,
    optimizer: Optional[OptoPrimeV2] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    eval_fn: Optional[EvalFn] = None,
    initial_templates: Optional[Dict[str, str]] = None,
    on_iteration: Optional[Callable[[int, List[RunResult], Dict[str, str]], None]] = None,
    log_to_mlflow: bool = False,
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
        Custom evaluation function. Uses default LLM-as-judge if not provided.
    initial_templates : Dict[str, str], optional
        Initial prompt templates. Uses graph defaults if not provided.
    on_iteration : Callable, optional
        Callback after each iteration: (iter_num, runs, updates) -> None.
    log_to_mlflow : bool
        If True, log metrics to MLflow after each iteration.
    
    Returns
    -------
    OptimizationResult
        Contains final templates, score history, best iteration, etc.
    
    Example
    -------
    >>> result = optimize_langgraph(
    ...     instrumented_graph,
    ...     queries=["Query 1", "Query 2", "Query 3"],
    ...     iterations=5,
    ...     log_to_mlflow=True,
    ... )
    >>> print(f"Improved: {result.baseline_score:.3f} → {result.best_score:.3f}")
    """

@dataclass
class OptimizationResult:
    """Result of optimize_langgraph()."""
    
    baseline_score: float
    best_score: float
    best_iteration: int
    final_templates: Dict[str, str]
    score_history: List[float]
    all_runs: List[List[RunResult]]
    optimizer: OptoPrimeV2
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

def emit_agentlightning_reward(
    *,
    value: float,
    name: str = "final_score",
    tracer_name: str = "opto.trace",
    index: int = 0,
    span_name: str = "agentlightning.annotation",
    temporal_ignore: bool = True,
    extra_attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit a reward span compatible with Agent Lightning semconv.
    
    Creates child span with:
    - agentlightning.reward.<i>.name
    - agentlightning.reward.<i>.value
    - trace.temporal_ignore (for TGJ stability)
    """
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

| File | Purpose |
|------|---------|
| `opto/trace/io/otel_semconv.py` | Semantic convention helpers |
| `opto/trace/io/mlflow_logger.py` | MLflow integration |
| `opto/trace/io/instrumentation.py` | `instrument_graph()` and `InstrumentedGraph` |
| `opto/trace/io/optimization.py` | `optimize_langgraph()` and related |

### 6.2 Files to Modify

| File | Changes |
|------|---------|
| `opto/trace/io/langgraph_otel_runtime.py` | Add child span emission, temporal_ignore support |
| `opto/trace/io/otel_adapter.py` | Handle `trace.temporal_ignore` in TGJ conversion |
| `opto/trace/io/__init__.py` | Export new public APIs |
| `opto/trainer/loggers.py` | Add `MLflowTelemetryLogger` |

### 6.3 Detailed Changes to `otel_adapter.py`

```python
# Add helper for temporal_ignore handling
def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)

# In otlp_traces_to_trace_json(), modify the prev_span_id update:
# Before:
#     prev_span_id = sid
# After:
if not _truthy(attrs.get("trace.temporal_ignore")):
    prev_span_id = sid
```

---

## 7. Implementation Plan

### Phase 1: Core Infrastructure (Priority: High)

| Task | Effort | Dependencies |
|------|--------|--------------|
| Create `otel_semconv.py` with helpers | 2h | None |
| Enhance `TracingLLM` with child spans | 3h | otel_semconv.py |
| Update `otel_adapter.py` for temporal_ignore | 1h | None |
| Create `TelemetrySession` class | 4h | langgraph_otel_runtime.py |

### Phase 2: High-Level API (Priority: High)

| Task | Effort | Dependencies |
|------|--------|--------------|
| Implement `instrument_graph()` | 4h | TelemetrySession, TracingLLM |
| Implement `optimize_langgraph()` | 4h | instrument_graph |
| Create `InstrumentedGraph` wrapper | 2h | instrument_graph |

### Phase 3: MLflow Integration (Priority: Medium)

| Task | Effort | Dependencies |
|------|--------|--------------|
| Create `MLflowTelemetryLogger` | 3h | BaseLogger |
| Integrate with TelemetrySession | 2h | MLflowTelemetryLogger |
| Add artifact export helpers | 2h | MLflowTelemetryLogger |

### Phase 4: Testing & Documentation (Priority: High)

| Task | Effort | Dependencies |
|------|--------|--------------|
| Unit tests for new modules | 4h | All modules |
| Integration test with StubLLM | 2h | All modules |
| Update README and examples | 2h | All modules |
| Prototype notebook | 2h | All modules |

---

## 8. Agent Lightning Comparison

### 8.1 API Comparison Table

| Aspect | Agent Lightning | Trace (New API) |
|--------|----------------|-----------------|
| **Initialization** | `import agentlightning as agl` | `from opto.trace.io import instrument_graph` |
| **Agent Definition** | `@rollout` decorator | `instrument_graph(graph, ...)` |
| **LLM Calls** | Auto-instrumented via proxy | `TracingLLM.node_call()` wrapper |
| **Reward Emission** | `emit_reward(value)` | `emit_agentlightning_reward(value, name)` |
| **Training Loop** | `Trainer.fit(agent, dataset)` | `optimize_langgraph(graph, queries)` |
| **Optimization** | RL/APO/SFT algorithms | TGJ → OPTO (OptoPrimeV2, TextGrad) |
| **Span Format** | `gen_ai.*` conventions | Dual: `param.*` + `gen_ai.*` |

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
from opto.trace.io import instrument_graph, optimize_langgraph

# One-time instrumentation
graph = build_my_langgraph()
instrumented = instrument_graph(
    graph,
    trainable_keys={"planner", "executor"},
    llm=my_llm,
)

# One-liner optimization
result = optimize_langgraph(
    instrumented,
    queries=test_queries,
    iterations=5,
)
```

### 8.3 Key Differences

| Feature | Agent Lightning | Trace |
|---------|----------------|-------|
| **Optimization Target** | Prompt templates via RL | Prompts + code via gradient descent |
| **Trace Format** | Custom span storage | OTLP → TGJ → Trace nodes |
| **Feedback Signal** | Reward values | Structured feedback (score + reasons) |
| **Code Optimization** | Not supported | Supported via `__code_*` params |
| **Graph Support** | Generic agents | LangGraph-native |

---

## 9. Test & Validation Plan

### 9.1 Unit Tests

| Test File | Coverage |
|-----------|----------|
| `tests/test_otel_semconv.py` | Semantic convention helpers |
| `tests/test_tracing_llm.py` | TracingLLM with child spans |
| `tests/test_telemetry_session.py` | Session management and export |
| `tests/test_instrumentation.py` | instrument_graph() |
| `tests/test_optimization.py` | optimize_langgraph() |

### 9.2 Integration Tests

```python
# tests/test_integration_stubllm.py

def test_full_optimization_flow_with_stubllm():
    """
    End-to-end test using StubLLM (no API calls).
    
    1. Build a simple LangGraph
    2. Instrument with instrument_graph()
    3. Run optimize_langgraph() for 2 iterations
    4. Verify:
       - OTLP spans contain expected attributes
       - TGJ conversion produces valid nodes
       - Optimizer produces parameter updates
       - Score improves or stays stable
    """
```

### 9.3 Validation Criteria

| Criterion | Validation Method |
|-----------|------------------|
| **OTLP Correctness** | Check span attributes match spec |
| **TGJ Compatibility** | `ingest_tgj()` produces valid nodes |
| **Temporal Ignore** | Child spans don't break TGJ hierarchy |
| **Agent Lightning Compat** | Spans have `gen_ai.*` and reward attrs |
| **MLflow Export** | Metrics/artifacts appear in MLflow UI |
| **Boilerplate Reduction** | Demo code < 100 lines (vs ~645) |

### 9.4 StubLLM for Testing

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

## 10. Appendix: Prototype Snippet

This prototype demonstrates the target API working with a StubLLM.

```python
"""
Prototype: instrument_graph + optimize_langgraph with StubLLM
============================================================

Run this to validate the API design before full implementation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
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
        
        # Planner response
        if "planner" in user_msg.lower() or "break" in user_msg.lower():
            return self._resp('{"1": {"agent": "researcher", "goal": "find info"}, "2": {"agent": "synthesizer", "goal": "answer"}}')
        
        # Executor response
        if "executor" in user_msg.lower() or "route" in user_msg.lower():
            return self._resp('{"goto": "synthesizer", "query": "test query"}')
        
        # Evaluator response
        if "evaluate" in user_msg.lower():
            return self._resp('{"answer_relevance": 0.8, "groundedness": 0.7, "plan_quality": 0.9, "reasons": "Good structure"}')
        
        # Default synthesizer response
        return self._resp("This is a synthesized answer based on the context provided.")
    
    def _resp(self, content):
        return type("R", (), {
            "choices": [type("C", (), {
                "message": type("M", (), {"content": content})()
            })()]
        })()


# Minimal TelemetrySession stub
class TelemetrySession:
    def __init__(self, service_name: str = "test"):
        self.spans = []
        self.service_name = service_name
    
    def record_span(self, name: str, attrs: Dict[str, Any]):
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
        self.trainable_keys = trainable_keys or set()
    
    def node_call(self, *, span_name, template_name=None, template=None,
                  optimizable_key=None, messages=None, **kwargs) -> str:
        # Record span
        attrs = {}
        if template_name and template:
            attrs[f"param.{template_name}"] = template
            attrs[f"param.{template_name}.trainable"] = optimizable_key in self.trainable_keys
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
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute graph with telemetry capture."""
        # In real impl, this wraps graph.invoke() with automatic tracing
        # For prototype, simulate execution
        
        # Simulate planner
        plan_resp = self.tracing_llm.node_call(
            span_name="planner",
            template_name="planner_prompt",
            template=self.templates.get("planner_prompt", "Default planner template"),
            optimizable_key="planner",
            messages=[{"role": "user", "content": f"Plan for: {state.get('query', '')}"}]
        )
        
        # Simulate synthesizer
        answer = self.tracing_llm.node_call(
            span_name="synthesizer",
            template_name="synthesizer_prompt",
            template=self.templates.get("synthesizer_prompt", "Default synth template"),
            optimizable_key="synthesizer",
            messages=[{"role": "user", "content": f"Synthesize answer for: {state.get('query', '')}"}]
        )
        
        # Simulate evaluator
        eval_resp = self.tracing_llm.node_call(
            span_name="evaluator",
            messages=[{"role": "user", "content": f"Evaluate: {answer}"}]
        )
        
        # Parse eval
        try:
            eval_data = json.loads(eval_resp)
            score = sum([
                eval_data.get("answer_relevance", 0.5),
                eval_data.get("groundedness", 0.5),
                eval_data.get("plan_quality", 0.5)
            ]) / 3
        except:
            score = 0.5
            eval_data = {}
        
        # Record eval span
        self.session.record_span("evaluator", {
            "eval.score": str(score),
            "eval.answer_relevance": str(eval_data.get("answer_relevance", 0.5)),
            "eval.groundedness": str(eval_data.get("groundedness", 0.5)),
            "eval.plan_quality": str(eval_data.get("plan_quality", 0.5)),
            "eval.reasons": eval_data.get("reasons", ""),
        })
        
        return {
            "answer": answer,
            "plan": plan_resp,
            "score": score,
            "metrics": eval_data,
        }


def instrument_graph(
    graph: Any,
    *,
    service_name: str = "langgraph-agent",
    trainable_keys: Optional[set] = None,
    llm: Optional[Any] = None,
    initial_templates: Optional[Dict[str, str]] = None,
) -> InstrumentedGraph:
    """
    Wrap a LangGraph with automatic OTEL instrumentation.
    
    This is the main entry point for the new API.
    """
    session = TelemetrySession(service_name)
    
    tracing_llm = TracingLLM(
        llm=llm or StubLLM(),
        session=session,
        trainable_keys=trainable_keys or {"planner", "synthesizer"},
    )
    
    return InstrumentedGraph(
        graph=graph,
        session=session,
        tracing_llm=tracing_llm,
        templates=initial_templates or {},
    )


# ============================================================
# PROTOTYPE: optimize_langgraph()
# ============================================================

@dataclass
class RunResult:
    answer: str
    score: float
    metrics: Dict[str, float]
    otlp: Dict[str, Any]


@dataclass
class OptimizationResult:
    baseline_score: float
    best_score: float
    best_iteration: int
    final_templates: Dict[str, str]
    score_history: List[float]


def optimize_langgraph(
    graph: InstrumentedGraph,
    queries: List[str],
    *,
    iterations: int = 3,
) -> OptimizationResult:
    """
    Run optimization loop on instrumented graph.
    
    This is a simplified prototype - real impl uses OptoPrimeV2.
    """
    score_history = []
    best_score = 0.0
    best_iteration = 0
    
    # Baseline run
    baseline_runs = []
    for q in queries:
        result = graph.invoke({"query": q})
        baseline_runs.append(RunResult(
            answer=result["answer"],
            score=result["score"],
            metrics=result.get("metrics", {}),
            otlp=graph.session.flush_otlp(),
        ))
    
    baseline_score = sum(r.score for r in baseline_runs) / len(baseline_runs)
    score_history.append(baseline_score)
    best_score = baseline_score
    
    print(f"Baseline score: {baseline_score:.3f}")
    
    # Optimization iterations
    for iteration in range(1, iterations + 1):
        runs = []
        for q in queries:
            result = graph.invoke({"query": q})
            runs.append(RunResult(
                answer=result["answer"],
                score=result["score"],
                metrics=result.get("metrics", {}),
                otlp=graph.session.flush_otlp(),
            ))
        
        iter_score = sum(r.score for r in runs) / len(runs)
        score_history.append(iter_score)
        
        if iter_score > best_score:
            best_score = iter_score
            best_iteration = iteration
        
        print(f"Iteration {iteration}: score={iter_score:.3f}")
        
        # In real impl: TGJ conversion → optimizer.backward() → optimizer.step()
        # For prototype, we just simulate
    
    return OptimizationResult(
        baseline_score=baseline_score,
        best_score=best_score,
        best_iteration=best_iteration,
        final_templates=dict(graph.templates),
        score_history=score_history,
    )


# ============================================================
# MAIN: Run prototype
# ============================================================

def main():
    print("=" * 60)
    print("PROTOTYPE: LangGraph OTEL Instrumentation API")
    print("=" * 60)
    
    # 1. Create a "graph" (placeholder for real LangGraph)
    graph = {"name": "research_agent"}
    
    # 2. Instrument with ONE function call
    instrumented = instrument_graph(
        graph,
        service_name="prototype-demo",
        trainable_keys={"planner", "synthesizer"},
        llm=StubLLM(),
        initial_templates={
            "planner_prompt": "You are a planner. Break down the task.",
            "synthesizer_prompt": "You are a synthesizer. Combine the results.",
        },
    )
    
    print("\n✓ Graph instrumented")
    print(f"  Service: {instrumented.session.service_name}")
    print(f"  Trainable keys: {instrumented.tracing_llm.trainable_keys}")
    
    # 3. Run optimization with ONE function call
    result = optimize_langgraph(
        instrumented,
        queries=[
            "What are the causes of WWI?",
            "Explain quantum entanglement.",
            "Summarize the French Revolution.",
        ],
        iterations=3,
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline: {result.baseline_score:.3f}")
    print(f"Best: {result.best_score:.3f} (iteration {result.best_iteration})")
    print(f"History: {[f'{s:.3f}' for s in result.score_history]}")
    
    # 4. Show OTLP output (demonstrating export capability)
    print("\n" + "=" * 60)
    print("SAMPLE OTLP OUTPUT")
    print("=" * 60)
    
    # Run one more time to capture OTLP
    instrumented.invoke({"query": "Test query"})
    otlp = instrumented.session.flush_otlp()
    
    print(json.dumps(otlp, indent=2)[:1000] + "...")
    
    print("\n✓ Prototype complete!")
    print("  - instrument_graph(): Creates instrumented wrapper")
    print("  - optimize_langgraph(): Runs optimization loop")
    print("  - TelemetrySession: Manages OTEL + exports")


if __name__ == "__main__":
    main()
```

---

## Summary

This technical plan outlines a minimal, reusable API for instrumenting LangGraph agents with OTEL tracing and running optimization loops. The key components are:

1. **`instrument_graph()`** - One-liner to add OTEL instrumentation
2. **`TelemetrySession`** - Unified session management with MLflow export
3. **Enhanced `TracingLLM`** - Dual semantic conventions for Trace + Agent Lightning
4. **`optimize_langgraph()`** - One-liner optimization loop
5. **OTEL semantic convention helpers** - Standardized span emission

The implementation follows a phased approach, prioritizing core infrastructure first, followed by high-level APIs and MLflow integration. All components will be validated with StubLLM tests before production use.

**Next Steps:**
1. Review and approve this technical plan
2. Begin Phase 1 implementation (core infrastructure)
3. Create prototype notebook for validation
4. Iterate based on feedback
