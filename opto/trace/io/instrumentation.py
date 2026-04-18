"""
opto.trace.io.instrumentation
==============================

One-liner ``instrument_graph()`` to add OTEL instrumentation to any
LangGraph ``StateGraph`` / ``CompiledGraph``.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from opto.trace.io.bindings import Binding, make_dict_binding
from opto.trace.io.graph_instrumentation import instrument_trace_graph
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
    service_name : str
        OTEL service / scope name.
    """

    graph: Any  # CompiledGraph
    session: TelemetrySession
    tracing_llm: TracingLLM
    templates: Dict[str, str] = field(default_factory=dict)
    bindings: Dict[str, Binding] = field(default_factory=dict)
    service_name: str = "langgraph-agent"
    input_key: str = "query"
    output_key: Optional[str] = None

    # Holds the active root span context for eval_fn to attach reward spans
    _root_span: Any = field(default=None, repr=False, init=False)

    @contextmanager
    def _root_invocation_span(self, query_hint: str = ""):
        """Context manager that creates a root invocation span (D9).

        All node spans created inside this context become children
        of the root span, producing a **single trace ID** per invocation.
        """
        span_name = f"{self.service_name}.invoke"
        with self.session.activate():
            with self.session.tracer.start_as_current_span(span_name) as root_sp:
                root_sp.set_attribute("langgraph.service", self.service_name)
                if query_hint:
                    root_sp.set_attribute("langgraph.query", str(query_hint)[:200])
                self._root_span = root_sp
                try:
                    yield root_sp
                finally:
                    self._root_span = None

    def invoke(self, state: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute graph under a root invocation span and capture telemetry.

        A root span wraps the entire graph invocation so that all node
        spans share a single trace ID (D9).
        """
        query_hint = ""
        if isinstance(state, dict):
            query_hint = str(state.get(self.input_key, ""))

        with self._root_invocation_span(query_hint) as root_sp:
            result = self.graph.invoke(state, **kwargs)
            # Attach a summary attribute to the root span (generic)
            if isinstance(result, dict) and self.output_key and self.output_key in result:
                root_sp.set_attribute(
                    "langgraph.output.preview",
                    str(result[self.output_key])[:500],
                )
            return result

    def stream(self, state: Any, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        """Stream graph execution with telemetry."""
        query_hint = ""
        if isinstance(state, dict):
            query_hint = str(state.get(self.input_key, ""))

        with self._root_invocation_span(query_hint):
            yield from self.graph.stream(state, **kwargs)


def instrument_graph(
    graph: Any = None,
    *,
    adapter: Optional[Any] = None,
    backend: str = "otel",
    graph_factory: Optional[Callable[[], Any]] = None,
    scope: Optional[Dict[str, Any]] = None,
    graph_agents_functions: Optional[List[str]] = None,
    graph_prompts_list: Optional[List[Any]] = None,
    train_graph_agents_functions: bool = True,
    session: Optional[TelemetrySession] = None,
    service_name: str = "langgraph-agent",
    trainable_keys: Optional[Set[str]] = None,
    enable_code_optimization: bool = False,
    llm: Optional[Any] = None,
    emit_genai_child_spans: bool = True,
    bindings: Optional[Dict[str, Binding]] = None,
    in_place: bool = False,
    initial_templates: Optional[Dict[str, str]] = None,
    provider_name: str = "llm",
    llm_span_name: str = "llm.chat.completion",
    input_key: str = "query",
    output_key: Optional[str] = None,
) -> Any:
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
    llm_span_name : str
        Name for child LLM spans.  Defaults to ``"llm.chat.completion"``.
        Override to match your provider (e.g. ``"openai.chat.completion"``).
    input_key : str
        Key in the input state dict used as a query hint for the root span.
        Defaults to ``"query"``.  Override to match your graph's state schema.
    output_key : str, optional
        Key in the result dict that holds the graph's final answer.
        If *None*, no preview is attached to the root span.

    Returns
    -------
    InstrumentedGraph
    """
    try:
        from opto.trace.graph.adapter import GraphAdapter
    except Exception:
        GraphAdapter = None

    if adapter is not None:
        if GraphAdapter is not None and not isinstance(adapter, GraphAdapter):
            raise TypeError("adapter must be an instance of GraphAdapter")
        return adapter.instrument(
            backend=backend,
            service_name=service_name,
            input_key=input_key,
            output_key=output_key,
            session=session,
            trainable_keys=trainable_keys,
            enable_code_optimization=enable_code_optimization,
            llm=llm,
            emit_genai_child_spans=emit_genai_child_spans,
            bindings=bindings,
            in_place=in_place,
            initial_templates=initial_templates,
            provider_name=provider_name,
            llm_span_name=llm_span_name,
        )

    if GraphAdapter is not None and isinstance(graph, GraphAdapter):
        return graph.instrument(
            backend=backend,
            service_name=service_name,
            input_key=input_key,
            output_key=output_key,
            session=session,
            trainable_keys=trainable_keys,
            enable_code_optimization=enable_code_optimization,
            llm=llm,
            emit_genai_child_spans=emit_genai_child_spans,
            bindings=bindings,
            in_place=in_place,
            initial_templates=initial_templates,
            provider_name=provider_name,
            llm_span_name=llm_span_name,
        )

    if backend == "trace":
        if graph_factory is None:
            if callable(graph):
                graph_factory = graph
            else:
                raise ValueError(
                    "backend='trace' requires graph_factory or a callable graph"
                )
        return instrument_trace_graph(
            graph_factory,
            scope=scope,
            graph_agents_functions=list(graph_agents_functions or []),
            graph_prompts_list=graph_prompts_list,
            train_graph_agents_functions=train_graph_agents_functions,
            service_name=service_name,
            input_key=input_key,
            output_key=output_key,
        )

    if backend != "otel":
        raise ValueError("Unsupported backend. Expected 'otel' or 'trace'.")

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

    # -- optional code parameter emission -----------------------------------
    emit_code_param = None
    if enable_code_optimization:
        CODE_ATTR_MAX_CHARS = 10_000

        def _emit_code_param(span, code_key: str, code_fn: Any) -> None:
            try:
                src = inspect.getsource(code_fn)
            except Exception:
                src = repr(code_fn)
            digest = hashlib.sha256(
                src.encode("utf-8", errors="ignore")
            ).hexdigest()
            was_truncated = False
            if len(src) > CODE_ATTR_MAX_CHARS:
                src = src[:CODE_ATTR_MAX_CHARS] + "\n# ... (truncated)"
                was_truncated = True
            span.set_attribute(f"param.__code_{code_key}", src)
            span.set_attribute(f"param.__code_{code_key}.sha256", digest)
            span.set_attribute(
                f"param.__code_{code_key}.truncated", str(was_truncated)
            )
            span.set_attribute(f"param.__code_{code_key}.trainable", True)

        emit_code_param = _emit_code_param

    # -- TracingLLM --
    tracing_llm = TracingLLM(
        llm=llm,
        tracer=session.tracer,
        trainable_keys=trainable_keys,
        emit_code_param=emit_code_param,
        provider_name=provider_name,
        llm_span_name=llm_span_name,
        emit_llm_child_span=emit_genai_child_spans,
    )

    return InstrumentedGraph(
        graph=compiled,
        session=session,
        tracing_llm=tracing_llm,
        templates=templates,
        bindings=bindings,
        service_name=service_name,
        input_key=input_key,
        output_key=output_key,
    )
