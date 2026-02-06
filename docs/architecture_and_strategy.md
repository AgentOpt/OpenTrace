# LangGraph OTEL Instrumentation: Architecture & Strategy

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Strategy Overview](#strategy-overview)
4. [System Architecture](#system-architecture)
5. [Component Deep Dive](#component-deep-dive)
6. [Data Flow](#data-flow)
7. [Semantic Conventions](#semantic-conventions)
8. [Optimization Pipeline](#optimization-pipeline)
9. [Integration Points](#integration-points)
10. [Implementation Roadmap](#implementation-roadmap)

---

## Executive Summary

This document outlines the architecture and strategy for creating a **unified OTEL instrumentation API** for LangGraph agents. The solution enables:

- **Simplified tracing**: One function call instruments entire graphs
- **Dual compatibility**: Traces work with both Trace (TGJ) and Agent Lightning
- **Unified optimization**: Single API for running optimization loops
- **Flexible backends**: Support for multiple LLM providers

---

## Problem Statement

### Current State (Before)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CURRENT: Manual OTEL Instrumentation                     │
│                         (~645 lines of boilerplate)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │ OTEL Setup       │  ~80 lines: TracerProvider, SpanProcessor,           │
│  │ (Boilerplate)    │           InMemoryExporter, Tracer init              │
│  └──────────────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ TracingLLM Class │  ~100 lines: Wrapper class definition,               │
│  │ (Boilerplate)    │            span creation, attribute setting          │
│  └──────────────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ Node Functions   │  ~25 lines PER NODE: Manual span creation,           │
│  │ (Per-node code)  │                      attribute recording             │
│  └──────────────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ Optimization     │  ~150 lines: Loop setup, trace capture,              │
│  │ Loop (Manual)    │             score tracking, template update          │
│  └──────────────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │ Export & Convert │  ~50 lines: OTLP export, TGJ conversion,             │
│  │ (Manual)         │            file saving                               │
│  └──────────────────┘                                                       │
│                                                                             │
│  TOTAL: ~645 lines of repeated boilerplate across demos                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Issues Identified

| Issue | Impact | Lines Affected |
|-------|--------|----------------|
| OTEL setup repeated in every demo | Code duplication | ~80 lines |
| TracingLLM redefined per file | Inconsistent behavior | ~100 lines |
| Manual span creation per node | Error-prone, verbose | ~25 lines/node |
| Optimization loop copy-pasted | Hard to maintain | ~150 lines |
| No Agent Lightning compatibility | Limited observability | N/A |
| Fragmented logging | Inconsistent metrics | ~50 lines |

---

## Strategy Overview

### Chosen Approach: "Trace-first, Dual Semconv"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STRATEGY: Trace-First, Dual Semconv                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DESIGN PRINCIPLES                            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  1. TRACE-FIRST: Optimize for Trace framework compatibility        │   │
│  │     - param.* attributes for trainable parameters                  │   │
│  │     - inputs.* / outputs.* for data flow                           │   │
│  │     - Temporal hierarchy preserved for TGJ                         │   │
│  │                                                                     │   │
│  │  2. DUAL SEMCONV: Also emit Agent Lightning conventions            │   │
│  │     - gen_ai.* attributes on child spans                           │   │
│  │     - agentlightning.reward.* for evaluation metrics               │   │
│  │     - Compatible with standard OTEL dashboards                     │   │
│  │                                                                     │   │
│  │  3. MINIMAL USER CODE: Hide complexity behind simple API           │   │
│  │     - instrument_graph() - one call to add tracing                 │   │
│  │     - optimize_langgraph() - one call for optimization             │   │
│  │     - No manual span creation required                             │   │
│  │                                                                     │   │
│  │  4. TEMPORAL ISOLATION: Child spans don't break TGJ                │   │
│  │     - trace.temporal_ignore attribute on GenAI spans               │   │
│  │     - Preserves node-to-node execution flow                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Target State (After)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TARGET: Simplified API (~10 lines)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  from trace_api import instrument_graph, optimize_langgraph                 │
│                                                                             │
│  # ONE CALL to instrument                                                   │
│  instrumented = instrument_graph(                                           │
│      graph=my_langgraph,                                                    │
│      trainable_keys={"planner", "synthesizer"},                             │
│  )                                                                          │
│                                                                             │
│  # ONE CALL to optimize                                                     │
│  result = optimize_langgraph(                                               │
│      instrumented,                                                          │
│      queries=["Q1", "Q2"],                                                  │
│      iterations=5,                                                          │
│  )                                                                          │
│                                                                             │
│  print(f"Best score: {result.best_score}")                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌─────────────┐                                │
│                              │  User Code  │                                │
│                              └──────┬──────┘                                │
│                                     │                                       │
│                     ┌───────────────┼───────────────┐                       │
│                     │               │               │                       │
│                     ▼               ▼               ▼                       │
│            ┌────────────────┐ ┌──────────┐ ┌────────────────┐              │
│            │instrument_graph│ │  invoke  │ │optimize_langgraph│            │
│            └───────┬────────┘ └────┬─────┘ └───────┬────────┘              │
│                    │               │               │                       │
│                    └───────────────┼───────────────┘                       │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      InstrumentedGraph                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │                                                             │   │   │
│  │  │  ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐  │   │   │
│  │  │  │  StateGraph  │  │ TelemetrySession │  │  TracingLLM  │  │   │   │
│  │  │  │  (LangGraph) │  │   (OTEL Spans)   │  │  (Wrapper)   │  │   │   │
│  │  │  └──────┬───────┘  └────────┬─────────┘  └──────┬───────┘  │   │   │
│  │  │         │                   │                   │          │   │   │
│  │  │         └───────────────────┼───────────────────┘          │   │   │
│  │  │                             │                              │   │   │
│  │  └─────────────────────────────┼──────────────────────────────┘   │   │
│  │                                │                                  │   │
│  └────────────────────────────────┼──────────────────────────────────┘   │
│                                   │                                       │
│                                   ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         LLM Backend                                 │   │
│  │                                                                     │   │
│  │    ┌─────────────────┐              ┌─────────────────┐            │   │
│  │    │  OpenRouterLLM  │      OR      │     StubLLM     │            │   │
│  │    │ (Real API calls)│              │ (Testing mode)  │            │   │
│  │    └─────────────────┘              └─────────────────┘            │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                       │
│                                   ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Output Layer                                │   │
│  │                                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │   │
│  │  │  OTLP JSON  │  │  TGJ Format │  │   MLflow    │  │  Console  │  │   │
│  │  │   Export    │  │  (Future)   │  │  (Future)   │  │   Logs    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COMPONENT INTERACTIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    instrument_graph()                              │    │
│  │                                                                    │    │
│  │  Input:                          Output:                           │    │
│  │  - graph (StateGraph)            - InstrumentedGraph               │    │
│  │  - service_name                    ├── .graph (compiled)           │    │
│  │  - trainable_keys                  ├── .session (TelemetrySession) │    │
│  │  - initial_templates               ├── .tracing_llm (TracingLLM)   │    │
│  │  - llm (optional)                  └── .templates (Dict)           │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                           │                                                 │
│                           │ creates                                         │
│                           ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    InstrumentedGraph                               │    │
│  │                                                                    │    │
│  │  .invoke(state)                                                    │    │
│  │      │                                                             │    │
│  │      ├──► Initializes AgentState                                   │    │
│  │      ├──► Runs compiled graph                                      │    │
│  │      │       │                                                     │    │
│  │      │       ├──► planner_node() ──► TracingLLM.node_call()       │    │
│  │      │       ├──► researcher_node() ──► TracingLLM.node_call()    │    │
│  │      │       ├──► synthesizer_node() ──► TracingLLM.node_call()   │    │
│  │      │       └──► evaluator_node() ──► TracingLLM.node_call()     │    │
│  │      │                                                             │    │
│  │      ├──► Records evaluation metrics span                          │    │
│  │      └──► Returns {answer, score, metrics, ...}                    │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                           │                                                 │
│                           │ uses                                            │
│                           ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                      TracingLLM                                    │    │
│  │                                                                    │    │
│  │  .node_call(span_name, template_name, template, messages)          │    │
│  │      │                                                             │    │
│  │      ├──► Creates PARENT span (Trace-compatible)                   │    │
│  │      │       - param.{template_name} = template                    │    │
│  │      │       - param.{template_name}.trainable = true/false        │    │
│  │      │       - inputs.gen_ai.prompt = user_message                 │    │
│  │      │                                                             │    │
│  │      ├──► Creates CHILD span (Agent Lightning-compatible)          │    │
│  │      │       - trace.temporal_ignore = "true"                      │    │
│  │      │       - gen_ai.operation.name = "chat"                      │    │
│  │      │       - gen_ai.provider.name = "openrouter"                 │    │
│  │      │       - gen_ai.input.messages = [...]                       │    │
│  │      │       - gen_ai.output.messages = [...]                      │    │
│  │      │                                                             │    │
│  │      ├──► Calls underlying LLM (OpenRouter/Stub)                   │    │
│  │      └──► Returns response content                                 │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                           │                                                 │
│                           │ records to                                      │
│                           ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                   TelemetrySession                                 │    │
│  │                                                                    │    │
│  │  .start_span(name) -> SpanContext                                  │    │
│  │      - Creates span with traceId, spanId, timestamps               │    │
│  │      - Returns context manager for attribute setting               │    │
│  │                                                                    │    │
│  │  .flush_otlp() -> Dict                                             │    │
│  │      - Exports all spans to OTLP JSON format                       │    │
│  │      - Clears internal span buffer                                 │    │
│  │      - Returns format compatible with otel_adapter                 │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. TelemetrySession

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TelemetrySession                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PURPOSE: Centralized OTEL span management and export                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Internal State:                                                    │   │
│  │                                                                     │   │
│  │  service_name: str          # Identifies the service in traces     │   │
│  │  _spans: List[Dict]         # In-memory span storage               │   │
│  │  _span_counter: int         # Auto-incrementing span IDs           │   │
│  │  _trace_id: str             # Current trace identifier             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Methods:                                                           │   │
│  │                                                                     │   │
│  │  start_span(name) -> SpanContext                                    │   │
│  │      │                                                              │   │
│  │      └──► Creates span dict with:                                   │   │
│  │              - traceId: current trace ID                            │   │
│  │              - spanId: auto-generated                               │   │
│  │              - name: provided name                                  │   │
│  │              - startTimeUnixNano: current timestamp                 │   │
│  │              - attributes: {} (empty, filled by SpanContext)        │   │
│  │                                                                     │   │
│  │  flush_otlp(clear=True) -> Dict                                     │   │
│  │      │                                                              │   │
│  │      └──► Exports to OTLP JSON:                                     │   │
│  │              {                                                      │   │
│  │                "resourceSpans": [{                                  │   │
│  │                  "scopeSpans": [{                                   │   │
│  │                    "scope": {"name": service_name},                 │   │
│  │                    "spans": [... all spans ...]                     │   │
│  │                  }]                                                 │   │
│  │                }]                                                   │   │
│  │              }                                                      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. TracingLLM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TracingLLM                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PURPOSE: Wrap LLM calls with dual semantic convention spans                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Configuration:                                                     │   │
│  │                                                                     │   │
│  │  llm: Any                   # Underlying LLM client                 │   │
│  │  session: TelemetrySession  # For span recording                   │   │
│  │  trainable_keys: Set[str]   # Which nodes have trainable prompts   │   │
│  │  provider_name: str         # "openrouter", "openai", etc.         │   │
│  │  emit_genai_child_span: bool # Whether to emit Agent Lightning spans│   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  node_call() Flow:                                                  │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ STEP 1: Create Parent Span (Trace-compatible)               │   │   │
│  │  │                                                             │   │   │
│  │  │   span_name: "planner"                                      │   │   │
│  │  │   attributes:                                               │   │   │
│  │  │     param.planner_prompt: "You are a planning agent..."     │   │   │
│  │  │     param.planner_prompt.trainable: "True"                  │   │   │
│  │  │     gen_ai.model: "llama-3.1-8b"                            │   │   │
│  │  │     inputs.gen_ai.prompt: "Plan for: What is AI?"           │   │   │
│  │  │                                                             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                          │                                         │   │
│  │                          ▼                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ STEP 2: Create Child Span (Agent Lightning-compatible)      │   │   │
│  │  │                                                             │   │   │
│  │  │   span_name: "openrouter.chat.completion"                   │   │   │
│  │  │   attributes:                                               │   │   │
│  │  │     trace.temporal_ignore: "true"  ◄── KEY ATTRIBUTE        │   │   │
│  │  │     gen_ai.operation.name: "chat"                           │   │   │
│  │  │     gen_ai.provider.name: "openrouter"                      │   │   │
│  │  │     gen_ai.request.model: "llama-3.1-8b"                    │   │   │
│  │  │     gen_ai.input.messages: "[{role: user, ...}]"            │   │   │
│  │  │                                                             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                          │                                         │   │
│  │                          ▼                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ STEP 3: Call LLM                                            │   │   │
│  │  │                                                             │   │   │
│  │  │   response = llm(messages=messages, **kwargs)               │   │   │
│  │  │   content = response.choices[0].message.content             │   │   │
│  │  │                                                             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                          │                                         │   │
│  │                          ▼                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ STEP 4: Record Output & Return                              │   │   │
│  │  │                                                             │   │   │
│  │  │   Child span attribute:                                     │   │   │
│  │  │     gen_ai.output.messages: "[{role: assistant, ...}]"      │   │   │
│  │  │                                                             │   │   │
│  │  │   Return: content (string)                                  │   │   │
│  │  │                                                             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. InstrumentedGraph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        InstrumentedGraph                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PURPOSE: Wrapper that adds telemetry to LangGraph execution                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Properties:                                                        │   │
│  │                                                                     │   │
│  │  graph: CompiledGraph       # The compiled LangGraph                │   │
│  │  session: TelemetrySession  # For span export                       │   │
│  │  tracing_llm: TracingLLM    # For instrumented LLM calls            │   │
│  │  templates: Dict[str, str]  # Prompt templates                      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  invoke(state) Flow:                                                │   │
│  │                                                                     │   │
│  │  INPUT: {"query": "What is AI?"}                                    │   │
│  │                                                                     │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Build Initial State                                         │   │   │
│  │  │   query: "What is AI?"                                      │   │   │
│  │  │   plan: {}                                                  │   │   │
│  │  │   research_results: []                                      │   │   │
│  │  │   answer: ""                                                │   │   │
│  │  │   evaluation: {}                                            │   │   │
│  │  │   planner_template: <from templates>                        │   │   │
│  │  │   synthesizer_template: <from templates>                    │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Execute Graph (generates spans via TracingLLM)              │   │   │
│  │  │                                                             │   │   │
│  │  │   START ──► planner ──► researcher ──► synthesizer          │   │   │
│  │  │                                              │               │   │   │
│  │  │                                              ▼               │   │   │
│  │  │                                         evaluator ──► END    │   │   │
│  │  │                                                             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ Record Evaluation Metrics                                   │   │   │
│  │  │                                                             │   │   │
│  │  │   Span: "evaluation_metrics"                                │   │   │
│  │  │     eval.score: 0.933                                       │   │   │
│  │  │     eval.answer_relevance: 0.95                             │   │   │
│  │  │     eval.groundedness: 0.90                                 │   │   │
│  │  │     eval.plan_quality: 0.95                                 │   │   │
│  │  │                                                             │   │   │
│  │  │   Child Span: "agentlightning.annotation"                   │   │   │
│  │  │     trace.temporal_ignore: "true"                           │   │   │
│  │  │     agentlightning.reward.0.name: "final_score"             │   │   │
│  │  │     agentlightning.reward.0.value: "0.933"                  │   │   │
│  │  │                                                             │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │       │                                                             │   │
│  │       ▼                                                             │   │
│  │  OUTPUT:                                                            │   │
│  │    {                                                                │   │
│  │      "answer": "AI is...",                                          │   │
│  │      "plan": {...},                                                 │   │
│  │      "research_results": [...],                                     │   │
│  │      "score": 0.933,                                                │   │
│  │      "metrics": {"answer_relevance": 0.95, ...},                    │   │
│  │      "reasons": "Good structure..."                                 │   │
│  │    }                                                                │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Single Execution Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE EXECUTION DATA FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  USER INPUT                                                                 │
│      │                                                                      │
│      │  {"query": "What is AI?"}                                           │
│      │                                                                      │
│      ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       PLANNER NODE                                   │  │
│  │                                                                      │  │
│  │  Input:  query = "What is AI?"                                       │  │
│  │  Template: "You are a planning agent..."                             │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ SPAN: planner                                                  │ │  │
│  │  │   param.planner_prompt = <template>                            │ │  │
│  │  │   param.planner_prompt.trainable = "True"                      │ │  │
│  │  │                                                                │ │  │
│  │  │   ┌────────────────────────────────────────────────────────┐  │ │  │
│  │  │   │ SPAN: openrouter.chat.completion                       │  │ │  │
│  │  │   │   trace.temporal_ignore = "true"                       │  │ │  │
│  │  │   │   gen_ai.input.messages = [...]                        │  │ │  │
│  │  │   │   gen_ai.output.messages = [...]                       │  │ │  │
│  │  │   └────────────────────────────────────────────────────────┘  │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  │  Output: plan = {"1": {"action": "research"}, ...}                   │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      RESEARCHER NODE                                 │  │
│  │                                                                      │  │
│  │  Input:  query, plan                                                 │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ SPAN: researcher                                               │ │  │
│  │  │   (no trainable template)                                      │ │  │
│  │  │                                                                │ │  │
│  │  │   ┌────────────────────────────────────────────────────────┐  │ │  │
│  │  │   │ SPAN: openrouter.chat.completion                       │  │ │  │
│  │  │   │   trace.temporal_ignore = "true"                       │  │ │  │
│  │  │   └────────────────────────────────────────────────────────┘  │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  │  Output: research_results = ["AI is...", ...]                        │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     SYNTHESIZER NODE                                 │  │
│  │                                                                      │  │
│  │  Input:  query, research_results                                     │  │
│  │  Template: "You are a synthesis agent..."                            │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ SPAN: synthesizer                                              │ │  │
│  │  │   param.synthesizer_prompt = <template>                        │ │  │
│  │  │   param.synthesizer_prompt.trainable = "True"                  │ │  │
│  │  │                                                                │ │  │
│  │  │   ┌────────────────────────────────────────────────────────┐  │ │  │
│  │  │   │ SPAN: openrouter.chat.completion                       │  │ │  │
│  │  │   │   trace.temporal_ignore = "true"                       │  │ │  │
│  │  │   └────────────────────────────────────────────────────────┘  │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  │  Output: answer = "AI is a field of computer science..."             │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      EVALUATOR NODE                                  │  │
│  │                                                                      │  │
│  │  Input:  query, answer                                               │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ SPAN: evaluator                                                │ │  │
│  │  │                                                                │ │  │
│  │  │   ┌────────────────────────────────────────────────────────┐  │ │  │
│  │  │   │ SPAN: openrouter.chat.completion                       │  │ │  │
│  │  │   │   trace.temporal_ignore = "true"                       │  │ │  │
│  │  │   └────────────────────────────────────────────────────────┘  │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  │  Output: evaluation = {                                              │  │
│  │            "answer_relevance": 0.95,                                 │  │
│  │            "groundedness": 0.90,                                     │  │
│  │            "plan_quality": 0.95                                      │  │
│  │          }                                                           │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  FINAL OUTPUT                                                               │
│      │                                                                      │
│      │  {                                                                   │
│      │    "answer": "AI is a field...",                                    │
│      │    "score": 0.933,                                                  │
│      │    "metrics": {...}                                                 │
│      │  }                                                                  │
│      │                                                                      │
│      ▼                                                                      │
│  OTLP EXPORT                                                                │
│      │                                                                      │
│      │  trace_output.json                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Semantic Conventions

### Dual Semantic Convention Mapping

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DUAL SEMANTIC CONVENTIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PARENT SPAN (Trace-compatible)                   │   │
│  │                    Used for: TGJ Optimization                       │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Attribute                      │ Purpose                           │   │
│  │  ───────────────────────────────┼────────────────────────────────── │   │
│  │  param.{name}                   │ Stores trainable prompt template  │   │
│  │  param.{name}.trainable         │ Marks if parameter is optimizable │   │
│  │  inputs.gen_ai.prompt           │ User input to the LLM             │   │
│  │  gen_ai.model                   │ Which model was used              │   │
│  │                                                                     │   │
│  │  Example:                                                           │   │
│  │    span_name: "planner"                                             │   │
│  │    attributes:                                                      │   │
│  │      param.planner_prompt: "You are a planning agent..."            │   │
│  │      param.planner_prompt.trainable: "True"                         │   │
│  │      inputs.gen_ai.prompt: "Plan for: What is AI?"                  │   │
│  │      gen_ai.model: "llama-3.1-8b"                                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   CHILD SPAN (Agent Lightning-compatible)           │   │
│  │                   Used for: OTEL Dashboards, Monitoring             │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Attribute                      │ Purpose                           │   │
│  │  ───────────────────────────────┼────────────────────────────────── │   │
│  │  trace.temporal_ignore          │ Exclude from TGJ temporal chain   │   │
│  │  gen_ai.operation.name          │ Type of operation ("chat")        │   │
│  │  gen_ai.provider.name           │ LLM provider ("openrouter")       │   │
│  │  gen_ai.request.model           │ Model identifier                  │   │
│  │  gen_ai.input.messages          │ Full message array (JSON)         │   │
│  │  gen_ai.output.messages         │ Response messages (JSON)          │   │
│  │                                                                     │   │
│  │  Example:                                                           │   │
│  │    span_name: "openrouter.chat.completion"                          │   │
│  │    attributes:                                                      │   │
│  │      trace.temporal_ignore: "true"                                  │   │
│  │      gen_ai.operation.name: "chat"                                  │   │
│  │      gen_ai.provider.name: "openrouter"                             │   │
│  │      gen_ai.request.model: "llama-3.1-8b"                           │   │
│  │      gen_ai.input.messages: "[{\"role\": \"user\", ...}]"           │   │
│  │      gen_ai.output.messages: "[{\"role\": \"assistant\", ...}]"     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   REWARD SPAN (Agent Lightning evaluation)          │   │
│  │                   Used for: Tracking optimization metrics           │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  Attribute                      │ Purpose                           │   │
│  │  ───────────────────────────────┼────────────────────────────────── │   │
│  │  trace.temporal_ignore          │ Exclude from TGJ temporal chain   │   │
│  │  agentlightning.reward.0.name   │ Metric name ("final_score")       │   │
│  │  agentlightning.reward.0.value  │ Metric value ("0.933")            │   │
│  │                                                                     │   │
│  │  Example:                                                           │   │
│  │    span_name: "agentlightning.annotation"                           │   │
│  │    attributes:                                                      │   │
│  │      trace.temporal_ignore: "true"                                  │   │
│  │      agentlightning.reward.0.name: "final_score"                    │   │
│  │      agentlightning.reward.0.value: "0.933"                         │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why `trace.temporal_ignore`?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL HIERARCHY PRESERVATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: Child spans disrupt TGJ temporal ordering                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  WITHOUT trace.temporal_ignore:                                     │   │
│  │                                                                     │   │
│  │  Time: t=0          t=1                    t=2          t=3        │   │
│  │        │            │                      │            │          │   │
│  │        ▼            ▼                      ▼            ▼          │   │
│  │    ┌────────┐  ┌──────────────────┐  ┌────────────┐  ┌────────┐   │   │
│  │    │planner │  │openrouter.chat   │  │ researcher │  │ ... │   │   │
│  │    │        │  │.completion       │  │            │  │        │   │   │
│  │    └────────┘  └──────────────────┘  └────────────┘  └────────┘   │   │
│  │                                                                     │   │
│  │  TGJ builds temporal chain:                                         │   │
│  │    planner -> openrouter.chat.completion -> researcher              │   │
│  │                                                                     │   │
│  │  WRONG! The LLM call span shouldn't be part of node-to-node flow   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  WITH trace.temporal_ignore:                                        │   │
│  │                                                                     │   │
│  │  Time: t=0          t=1                    t=2          t=3        │   │
│  │        │            │                      │            │          │   │
│  │        ▼            ▼                      ▼            ▼          │   │
│  │    ┌────────┐  ┌──────────────────┐  ┌────────────┐  ┌────────┐   │   │
│  │    │planner │  │openrouter.chat   │  │ researcher │  │ ... │   │   │
│  │    │        │  │.completion       │  │            │  │        │   │
│  │    └────────┘  │ [IGNORED]        │  └────────────┘  └────────┘   │   │
│  │                └──────────────────┘                                 │   │
│  │                                                                     │   │
│  │  TGJ builds temporal chain:                                         │   │
│  │    planner -> researcher -> synthesizer -> evaluator                │   │
│  │                                                                     │   │
│  │  CORRECT! Node-to-node flow preserved for optimization              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Optimization Pipeline

### Optimization Loop Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZATION PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  optimize_langgraph(graph, queries, iterations=3)                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        BASELINE (Iteration 0)                       │   │
│  │                                                                     │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐                       │   │
│  │   │ Query 1  │   │ Query 2  │   │ Query N  │                       │   │
│  │   │ Score:   │   │ Score:   │   │ Score:   │                       │   │
│  │   │ 0.85     │   │ 0.90     │   │ 0.80     │                       │   │
│  │   └──────────┘   └──────────┘   └──────────┘                       │   │
│  │                                                                     │   │
│  │   Average: 0.850                                                    │   │
│  │   OTLP: [captured for each query]                                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       ITERATION 1                                   │   │
│  │                                                                     │   │
│  │   [Templates may be updated by optimizer - future]                  │   │
│  │                                                                     │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐                       │   │
│  │   │ Query 1  │   │ Query 2  │   │ Query N  │                       │   │
│  │   │ Score:   │   │ Score:   │   │ Score:   │                       │   │
│  │   │ 0.88     │   │ 0.92     │   │ 0.85     │                       │   │
│  │   └──────────┘   └──────────┘   └──────────┘                       │   │
│  │                                                                     │   │
│  │   Average: 0.883 (+0.033)                                           │   │
│  │   OTLP: [captured for each query]                                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       ITERATION 2                                   │   │
│  │                                                                     │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐                       │   │
│  │   │ Query 1  │   │ Query 2  │   │ Query N  │                       │   │
│  │   │ Score:   │   │ Score:   │   │ Score:   │                       │   │
│  │   │ 0.91     │   │ 0.93     │   │ 0.89     │                       │   │
│  │   └──────────┘   └──────────┘   └──────────┘                       │   │
│  │                                                                     │   │
│  │   Average: 0.910 (+0.027) ★ NEW BEST                                │   │
│  │   OTLP: [captured for each query]                                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       ITERATION 3                                   │   │
│  │                                                                     │   │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐                       │   │
│  │   │ Query 1  │   │ Query 2  │   │ Query N  │                       │   │
│  │   │ Score:   │   │ Score:   │   │ Score:   │                       │   │
│  │   │ 0.90     │   │ 0.91     │   │ 0.88     │                       │   │
│  │   └──────────┘   └──────────┘   └──────────┘                       │   │
│  │                                                                     │   │
│  │   Average: 0.897 (-0.013)                                           │   │
│  │   OTLP: [captured for each query]                                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                           │                                                 │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    OPTIMIZATION RESULT                              │   │
│  │                                                                     │   │
│  │   OptimizationResult:                                               │   │
│  │     baseline_score: 0.850                                           │   │
│  │     best_score: 0.910                                               │   │
│  │     best_iteration: 2                                               │   │
│  │     score_history: [0.850, 0.883, 0.910, 0.897]                     │   │
│  │     final_templates: {planner_prompt: "...", ...}                   │   │
│  │     all_runs: [[Run1, Run2, ...], ...]                              │   │
│  │                                                                     │   │
│  │   Files Generated:                                                  │   │
│  │     - optimization_traces.json (all OTLP traces)                    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### Current Integrations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INTEGRATION POINTS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  IMPLEMENTED                                                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  ┌────────────────┐                                                 │   │
│  │  │   LangGraph    │  Real StateGraph with nodes and edges           │   │
│  │  │                │  Supports custom graphs via instrument_graph()  │   │
│  │  └────────────────┘                                                 │   │
│  │          │                                                          │   │
│  │          ▼                                                          │   │
│  │  ┌────────────────┐                                                 │   │
│  │  │  OpenRouter    │  HTTP API calls to OpenRouter                   │   │
│  │  │                │  Supports any model available on platform       │   │
│  │  └────────────────┘                                                 │   │
│  │          │                                                          │   │
│  │          ▼                                                          │   │
│  │  ┌────────────────┐                                                 │   │
│  │  │  OTLP JSON     │  Full OTLP export compatible with               │   │
│  │  │  Export        │  otel_adapter.otlp_traces_to_trace_json()       │   │
│  │  └────────────────┘                                                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PLANNED (Future)                                                   │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  ┌────────────────┐                                                 │   │
│  │  │  OpenTelemetry │  Real OTEL SDK integration                      │   │
│  │  │  SDK           │  TracerProvider, SpanProcessor, etc.            │   │
│  │  └────────────────┘                                                 │   │
│  │          │                                                          │   │
│  │          ▼                                                          │   │
│  │  ┌────────────────┐                                                 │   │
│  │  │  MLflow        │  Metrics logging, artifact storage              │   │
│  │  │  Integration   │  Run tracking, model registry                   │   │
│  │  └────────────────┘                                                 │   │
│  │          │                                                          │   │
│  │          ▼                                                          │   │
│  │  ┌────────────────┐                                                 │   │
│  │  │  Jaeger/Zipkin │  Trace visualization and analysis               │   │
│  │  │  Export        │  Distributed tracing dashboards                 │   │
│  │  └────────────────┘                                                 │   │
│  │          │                                                          │   │
│  │          ▼                                                          │   │
│  │  ┌────────────────┐                                                 │   │
│  │  │  TGJ Converter │  Direct integration with                        │   │
│  │  │                │  otel_adapter for Trace optimization            │   │
│  │  └────────────────┘                                                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      IMPLEMENTATION ROADMAP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: Core Infrastructure (COMPLETED)                                   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                   │
│  [x] TelemetrySession - span management and OTLP export                     │
│  [x] TracingLLM - dual semantic convention wrapper                          │
│  [x] OpenRouterLLM - real API integration                                   │
│  [x] StubLLM - deterministic testing                                        │
│  [x] instrument_graph() - one-liner instrumentation                         │
│  [x] optimize_langgraph() - optimization loop                               │
│  [x] Real LangGraph nodes - planner, researcher, synthesizer, evaluator     │
│  [x] OTLP JSON export to files                                              │
│  [x] Comprehensive documentation                                            │
│                                                                             │
│  PHASE 2: OTEL SDK Integration (PLANNED)                                    │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                   │
│  [ ] Replace prototype TelemetrySession with real OTEL SDK                  │
│  [ ] TracerProvider configuration                                           │
│  [ ] SpanProcessor pipeline                                                 │
│  [ ] OTLP exporter to backends (Jaeger, Zipkin)                            │
│  [ ] Context propagation                                                    │
│                                                                             │
│  PHASE 3: MLflow Integration (PLANNED)                                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                   │
│  [ ] MLflowTelemetryLogger class                                            │
│  [ ] Metrics logging (scores, latencies)                                    │
│  [ ] Artifact storage (traces, templates)                                   │
│  [ ] Run tracking and comparison                                            │
│  [ ] Model registry integration                                             │
│                                                                             │
│  PHASE 4: TGJ Integration (PLANNED)                                         │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                   │
│  [ ] Direct otel_adapter integration                                        │
│  [ ] Automatic OTLP-to-TGJ conversion                                       │
│  [ ] Trace framework optimizer integration                                  │
│  [ ] Template update from optimization feedback                             │
│                                                                             │
│  PHASE 5: Advanced Features (PLANNED)                                       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                   │
│  [ ] Conditional graph edges                                                │
│  [ ] Human-in-the-loop optimization                                         │
│  [ ] Multi-agent graph support                                              │
│  [ ] Streaming response handling                                            │
│  [ ] Custom evaluation functions                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This architecture provides a **clean separation of concerns**:

1. **User Layer**: Simple `instrument_graph()` and `optimize_langgraph()` API
2. **Instrumentation Layer**: `InstrumentedGraph`, `TracingLLM`, `TelemetrySession`
3. **Execution Layer**: Real LangGraph nodes with automatic tracing
4. **Backend Layer**: Pluggable LLM providers (OpenRouter, Stub, future: OpenAI, Anthropic)
5. **Export Layer**: OTLP JSON, future TGJ, MLflow, Jaeger

The **dual semantic convention** approach ensures compatibility with both:
- **Trace framework** (for optimization via TGJ)
- **Agent Lightning** (for standard OTEL monitoring)

The `trace.temporal_ignore` attribute is the key innovation that allows both paradigms to coexist without breaking the temporal hierarchy required for optimization.
