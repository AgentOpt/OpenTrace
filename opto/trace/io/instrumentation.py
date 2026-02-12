"""
opto.trace.io.instrumentation
==============================

One-liner ``instrument_graph()`` to add OTEL instrumentation to any
LangGraph ``StateGraph`` / ``CompiledGraph``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Set

from opto.trace.io.bindings import Binding, make_dict_binding
from opto.trace.io.langgraph_otel_runtime import TracingLLM
from opto.trace.io.telemetry_session import TelemetrySession

logger = logging.getLogger(__name__)


@dataclass
class InstrumentedGraph:
    """Instrumented LangGraph wrapper with telemetry.

    Attributes
    ----------
    graph : CompiledGraph
        The compiled LangGraph.
    session : TelemetrySession
        Manages OTEL tracing and export.
    tracing_llm : TracingLLM
        LLM wrapper with dual semantic conventions.
    templates : dict
        Current prompt templates (keyed by param name).
    bindings : dict
        Mapping from param key -> ``Binding`` (for ``apply_updates``).
    """

    graph: Any  # CompiledGraph
    session: TelemetrySession
    tracing_llm: TracingLLM
    templates: Dict[str, str] = field(default_factory=dict)
    bindings: Dict[str, Binding] = field(default_factory=dict)

    def invoke(self, state: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute graph and capture telemetry."""
        return self.graph.invoke(state, **kwargs)

    def stream(self, state: Any, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        """Stream graph execution with telemetry."""
        yield from self.graph.stream(state, **kwargs)


def instrument_graph(
    graph: Any = None,
    *,
    session: Optional[TelemetrySession] = None,
    service_name: str = "langgraph-agent",
    trainable_keys: Optional[Set[str]] = None,
    enable_code_optimization: bool = False,
    llm: Optional[Any] = None,
    emit_genai_child_spans: bool = True,
    bindings: Optional[Dict[str, Binding]] = None,
    in_place: bool = False,
    initial_templates: Optional[Dict[str, str]] = None,
    provider_name: str = "openai",
) -> InstrumentedGraph:
    """Wrap a LangGraph with automatic OTEL instrumentation.

    Parameters
    ----------
    graph : StateGraph | CompiledGraph, optional
        The LangGraph to instrument.  If it has a ``compile()`` method it
        will be compiled automatically.
    session : TelemetrySession, optional
        Reuse an existing session; otherwise a new one is created.
    service_name : str
        OTEL service name for trace identification.
    trainable_keys : set[str] or None
        Node names whose prompts are trainable.  ``None`` means **all
        trainable** (no hard-coded node names).
    enable_code_optimization : bool
        If *True*, emit ``param.__code_*`` attributes.
    llm : Any, optional
        LLM client.  Will be wrapped with ``TracingLLM``.
    emit_genai_child_spans : bool
        Emit ``gen_ai.*`` child spans for Agent Lightning compatibility.
    bindings : dict, optional
        Explicit ``{param_key: Binding}`` map.  If *None*, auto-derived
        from *initial_templates*.
    in_place : bool
        If *False* (default), avoid permanent mutation of the original
        graph.
    initial_templates : dict, optional
        Starting prompt templates ``{param_name: template_str}``.
    provider_name : str
        LLM provider name for ``gen_ai.provider.name``.

    Returns
    -------
    InstrumentedGraph
    """
    # -- compile graph if needed --
    compiled = graph
    if graph is not None and hasattr(graph, "compile"):
        compiled = graph.compile()

    # -- session --
    if session is None:
        session = TelemetrySession(service_name=service_name)

    # -- templates --
    templates = dict(initial_templates or {})

    # -- bindings: auto-derive from templates dict when not provided --
    if bindings is None:
        bindings = {}
        for key in templates:
            bindings[key] = make_dict_binding(templates, key, kind="prompt")

    # -- TracingLLM --
    tracing_llm = TracingLLM(
        llm=llm,
        tracer=session.tracer,
        trainable_keys=trainable_keys,
        provider_name=provider_name,
        emit_llm_child_span=emit_genai_child_spans,
    )

    return InstrumentedGraph(
        graph=compiled,
        session=session,
        tracing_llm=tracing_llm,
        templates=templates,
        bindings=bindings,
    )
