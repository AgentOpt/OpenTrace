"""
opto.trace.io – OTEL instrumentation & optimization for LangGraph
=================================================================

Public API
----------
* ``instrument_graph()`` – one-liner to add OTEL tracing to any LangGraph
* ``optimize_graph()``   – one-liner optimisation loop
* ``TelemetrySession``   – unified session manager (OTEL + optional MLflow)
* ``Binding`` / ``apply_updates()`` – param-key → getter/setter mapping
* ``EvalResult`` / ``EvalFn`` – flexible evaluation contract
* ``emit_reward()`` / ``emit_trace()`` – manual span helpers

Lower-level
~~~~~~~~~~~~
* ``TracingLLM``           – LLM wrapper with dual semconv
* ``InstrumentedGraph``    – wrapper returned by ``instrument_graph()``
* ``RunResult`` / ``OptimizationResult`` – result data classes
* ``otlp_traces_to_trace_json()`` – OTLP → TGJ adapter
* ``ingest_tgj()`` / ``merge_tgj()`` – TGJ → Trace nodes
"""

# -- high-level API --------------------------------------------------------
from opto.trace.io.instrumentation import instrument_graph, InstrumentedGraph
from opto.trace.io.optimization import (
    optimize_graph,
    EvalResult,
    EvalFn,
    RunResult,
    OptimizationResult,
)
from opto.trace.io.telemetry_session import TelemetrySession
from opto.trace.io.bindings import Binding, apply_updates, make_dict_binding
from opto.trace.io.otel_semconv import (
    emit_reward,
    emit_agentlightning_reward,
    emit_trace,
    set_span_attributes,
    record_genai_chat,
)

# -- lower-level -----------------------------------------------------------
from opto.trace.io.langgraph_otel_runtime import (
    TracingLLM,
    LLMCallError,
    InMemorySpanExporter,
    init_otel_runtime,
    flush_otlp,
    extract_eval_metrics_from_otlp,
)
from opto.trace.io.otel_adapter import otlp_traces_to_trace_json
from opto.trace.io.tgj_ingest import ingest_tgj, merge_tgj

__all__ = [
    # High-level
    "instrument_graph",
    "optimize_graph",
    "TelemetrySession",
    "Binding",
    "apply_updates",
    "make_dict_binding",
    "EvalResult",
    "EvalFn",
    "emit_reward",
    "emit_agentlightning_reward",
    "emit_trace",
    "set_span_attributes",
    "record_genai_chat",
    # Data classes
    "InstrumentedGraph",
    "RunResult",
    "OptimizationResult",
    # Lower-level
    "TracingLLM",
    "LLMCallError",
    "InMemorySpanExporter",
    "init_otel_runtime",
    "flush_otlp",
    "extract_eval_metrics_from_otlp",
    "otlp_traces_to_trace_json",
    "ingest_tgj",
    "merge_tgj",
]
