"""Adapters that build graph objects for trace-native and OTEL execution."""

from __future__ import annotations

import contextlib
import json
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from opto.trace import bundle, node
from opto.trace.bundle import FunModule, to_data
from opto.features.graph.module import GraphModule
from opto.features.graph.sidecars import GraphRunSidecar, OTELRunSidecar
from opto.trace.io.bindings import Binding
from opto.features.graph.graph_instrumentation import TraceGraph
from opto.trace.nodes import Node, ParameterNode


def _raw(value: Any) -> Any:
    """Return the underlying Python value for Trace nodes and wrappers."""
    return getattr(value, "data", value)


def _otel_attr_value(value: Any, *, max_chars: int = 2_000) -> str:
    """Serialize runtime values into bounded OTEL string attributes."""
    value = _raw(value)
    if isinstance(value, str):
        out = value
    elif isinstance(value, (int, float, bool)) or value is None:
        out = str(value)
    else:
        try:
            out = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            out = repr(value)
    if len(out) > max_chars:
        return out[:max_chars] + "...[truncated]"
    return out


def _trainable_attr(value: Any, *, default: bool = True) -> bool:
    """Best-effort trainability flag for ParameterNode-like values."""
    return bool(getattr(value, "trainable", default))


def _normalize_named_callables(
    targets: Union[None, List[str], List[Callable[..., Any]], Mapping[str, Callable[..., Any]]],
    scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Callable[..., Any]]:
    """Normalize function targets into a ``{name: callable}`` mapping."""
    if targets is None:
        return {}
    if isinstance(targets, Mapping):
        return dict(targets)
    out: Dict[str, Callable[..., Any]] = {}
    for item in targets:
        if isinstance(item, str):
            if scope is None or item not in scope:
                raise KeyError(f"Function {item!r} not found in adapter scope")
            out[item] = scope[item]
        else:
            out[getattr(item, "__name__", f"fn_{len(out)}")] = item
    return out


def _as_parameter(name: str, value: Any) -> ParameterNode:
    """Coerce a raw value or node into a trainable ``ParameterNode``."""
    if isinstance(value, ParameterNode):
        return value
    if isinstance(value, Node):
        return node(value, name=name, trainable=True)
    return node(value, name=name, trainable=True)


@dataclass
class GraphAdapter:
    """Abstract adapter that exposes a graph factory as an instrumentable object."""

    graph_factory: Callable[..., Any]
    backend: str = "trace"
    bindings: Dict[str, Binding] = field(default_factory=dict)
    service_name: str = "graph-adapter"
    input_key: str = "query"
    output_key: Optional[str] = None

    def build_graph(self, backend: Optional[str] = None):
        """Build and compile the graph for the requested backend."""
        raise NotImplementedError

    def invoke_runtime(self, state: Dict[str, Any], **kwargs: Any):
        """Execute the runtime-facing graph and return ``(result, sidecar)``."""
        raise NotImplementedError

    def invoke_trace(self, state: Dict[str, Any], **kwargs: Any):
        """Execute the trace-native graph and return ``(result, sidecar)``."""
        raise NotImplementedError

    def new_run_sidecar(self):
        """Create the per-invocation sidecar used to preserve traced state."""
        return GraphRunSidecar()

    def bindings_dict(self) -> Dict[str, Binding]:
        """Return a shallow copy of the adapter bindings."""
        return dict(self.bindings)

    def parameters(self) -> List[ParameterNode]:
        """Return the trainable parameters surfaced by this adapter."""
        raise NotImplementedError

    def as_module(self) -> GraphModule:
        """Expose the adapter through the ``trace.modules.Module`` interface."""
        return GraphModule(self)

    def instrument(self, backend: Optional[str] = None, **kwargs: Any):
        """Wrap the adapter with the instrumentation backend requested by the caller."""
        effective_backend = backend or self.backend
        service_name = kwargs.pop("service_name", self.service_name)
        input_key = kwargs.pop("input_key", self.input_key)
        output_key = kwargs.pop("output_key", self.output_key)
        if effective_backend == "trace":
            return TraceGraph(
                graph=self,
                parameters=self.parameters(),
                bindings=self.bindings_dict(),
                service_name=service_name,
                input_key=input_key,
                output_key=output_key,
            )
        if effective_backend == "otel":
            from opto.trace.io.instrumentation import instrument_graph

            merged = self.bindings_dict()
            merged.update(kwargs.pop("bindings", {}) or {})
            graph = self.build_graph(backend="otel")
            return instrument_graph(
                graph=graph,
                backend="otel",
                bindings=merged,
                service_name=service_name,
                input_key=input_key,
                output_key=output_key,
                **kwargs,
            )
        raise ValueError(f"Unsupported backend: {effective_backend!r}")


@dataclass
class _AdapterOTELRuntimeGraph:
    """Invoke OTEL graphs through the adapter so knobs stay live per run."""

    adapter: "LangGraphAdapter"

    def _runtime_state(self, state: Any) -> Any:
        """Inject current graph knobs into dict-like runtime state."""
        if not isinstance(state, dict):
            return state
        runtime_state = dict(state)
        runtime_state.update(self.adapter._knob_values())
        return runtime_state

    def invoke(self, state: Any, **kwargs: Any):
        """Build or reuse the current OTEL graph and invoke it."""
        graph = self.adapter.build_graph(backend="otel")
        return graph.invoke(self._runtime_state(state), **kwargs)

    def stream(self, state: Any, **kwargs: Any):
        """Build or reuse the current OTEL graph and stream it."""
        graph = self.adapter.build_graph(backend="otel")
        yield from graph.stream(self._runtime_state(state), **kwargs)


@dataclass
class LangGraphAdapter(GraphAdapter):
    """Concrete adapter for LangGraph-style factories and scoped callables."""

    function_targets: Union[None, List[str], List[Callable[..., Any]], Mapping[str, Callable[..., Any]]] = None
    prompt_targets: Optional[Mapping[str, Any]] = None
    graph_knobs: Optional[Mapping[str, Any]] = None
    scope: Optional[Dict[str, Any]] = None
    train_graph_agents_functions: bool = True

    def __post_init__(self) -> None:
        """Normalize targets, create traced wrappers, and auto-build bindings."""
        self.function_targets = _normalize_named_callables(self.function_targets, self.scope)
        self.prompt_targets = {k: _as_parameter(k, v) for k, v in dict(self.prompt_targets or {}).items()}
        self.graph_knobs = {k: _as_parameter(k, v) for k, v in dict(self.graph_knobs or {}).items()}
        self._active_sidecar: Optional[GraphRunSidecar] = None
        self._compiled_cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Any] = {}
        self._original_functions = dict(self.function_targets)
        self._traced_functions = {
            name: (fn if isinstance(fn, FunModule) or hasattr(fn, "_fun") else bundle(
                trainable=self.train_graph_agents_functions,
                traceable_code=True,
                allow_external_dependencies=True,
            )(fn))
            for name, fn in self.function_targets.items()
        }
        for fn in self._original_functions.values():
            fn_globals = getattr(fn, "__globals__", {})
            for name, prompt in self.prompt_targets.items():
                fn_globals[name] = prompt
            for name, knob in self.graph_knobs.items():
                fn_globals[name] = knob
        self._build_bindings()

    def __getstate__(self):
        """Drop transient runtime state so the adapter remains pickle-friendly."""
        state = self.__dict__.copy()
        state["_active_sidecar"] = None
        state["_compiled_cache"] = {}
        return state

    def instrument(self, backend: Optional[str] = None, **kwargs: Any):
        """Wrap the adapter, keeping OTEL graph knobs live across invocations."""
        effective_backend = backend or self.backend
        if effective_backend != "otel":
            return super().instrument(backend=backend, **kwargs)

        from opto.trace.io.instrumentation import instrument_graph

        service_name = kwargs.pop("service_name", self.service_name)
        input_key = kwargs.pop("input_key", self.input_key)
        output_key = kwargs.pop("output_key", self.output_key)

        merged = self.bindings_dict()
        merged.update(kwargs.pop("bindings", {}) or {})
        runtime_graph = _AdapterOTELRuntimeGraph(self)
        return instrument_graph(
            graph=runtime_graph,
            backend="otel",
            bindings=merged,
            service_name=service_name,
            input_key=input_key,
            output_key=output_key,
            **kwargs,
        )

    def _build_bindings(self) -> None:
        """Derive bindings for prompts, graph knobs, and traced code parameters."""
        auto: Dict[str, Binding] = {}
        for name, prompt in self.prompt_targets.items():
            auto[name] = Binding(
                get=lambda p=prompt: p.data,
                set=lambda v, p=prompt: p._set(v),
                kind="prompt",
            )
        for name, knob in self.graph_knobs.items():
            auto[name] = Binding(
                get=lambda p=knob: p.data,
                set=lambda v, p=knob: p._set(v),
                kind="graph",
            )
        for name, fn in self._traced_functions.items():
            if getattr(fn, "parameter", None) is not None:
                code_param = fn.parameter
                auto[f"__code_{name}"] = Binding(
                    get=lambda p=code_param: p.data,
                    set=lambda v, p=code_param: p._set(v),
                    kind="code",
                )
        user = dict(self.bindings)
        auto.update(user)
        self.bindings = auto

    def parameters(self) -> List[ParameterNode]:
        """Collect the unique trainable parameters owned by the adapter."""
        params: List[ParameterNode] = []
        params.extend(self.prompt_targets.values())
        params.extend(self.graph_knobs.values())
        for fn in self._traced_functions.values():
            if getattr(fn, "parameter", None) is not None:
                params.append(fn.parameter)
            try:
                params.extend(list(fn.parameters()))
            except Exception:
                pass
        out = []
        seen = set()
        for p in params:
            if id(p) in seen:
                continue
            seen.add(id(p))
            out.append(p)
        return out

    def _knob_values(self) -> Dict[str, Any]:
        """Read graph knob values as raw Python objects."""
        return {k: _raw(v) for k, v in self.graph_knobs.items()}

    def _cache_key(self, backend: str):
        """Build the compiled-graph cache key for a backend/knob combination."""
        return backend, tuple(sorted((k, repr(v)) for k, v in self._knob_values().items()))

    @contextlib.contextmanager
    def _scope_override(self, overrides: Dict[str, Any]):
        """Temporarily patch adapter scope entries while constructing the graph."""
        if not self.scope:
            yield
            return
        backup = {k: self.scope[k] for k in overrides if k in self.scope}
        self.scope.update(overrides)
        try:
            yield
        finally:
            for key in overrides:
                if key in backup:
                    self.scope[key] = backup[key]

    def _merge_shadow(self, sidecar: GraphRunSidecar, runtime_out: Any, traced_out: Any) -> None:
        """Merge traced outputs back into the sidecar's shadow state."""
        if not isinstance(runtime_out, dict):
            return
        if isinstance(traced_out, Node) and isinstance(getattr(traced_out, "data", None), dict):
            for key in runtime_out:
                try:
                    sidecar.shadow_state[key] = traced_out[key]
                except Exception:
                    sidecar.shadow_state[key] = node(runtime_out[key], name=key)
        else:
            for key, value in runtime_out.items():
                sidecar.shadow_state[key] = value if isinstance(value, Node) else node(value, name=key)

    def _trace_runtime_wrapper(self, name: str, traced_fn: FunModule):
        """Wrap a traced function so runtime execution still updates Trace state."""
        def _wrapped(state: Dict[str, Any], *args: Any, **kwargs: Any):
            """Replay shadow inputs through the traced callable for one graph node."""
            if self._active_sidecar is None:
                raise RuntimeError("Trace runtime wrapper called without active sidecar")
            sidecar = self._active_sidecar
            trace_state = dict(state)
            for key, traced_value in sidecar.shadow_state.items():
                trace_state[key] = traced_value
            traced_out = traced_fn(trace_state, *args, **kwargs)
            runtime_out = to_data(traced_out)
            sidecar.record_node_output(name, traced_out, runtime_out)
            self._merge_shadow(sidecar, runtime_out, traced_out)
            return runtime_out

        _wrapped.__name__ = name
        return _wrapped

    def _emit_otel_parameters(self, span: Any, *, node_name: str) -> None:
        """Emit adapter parameters on a node-level OTEL span.

        The converter looks for ``param.*`` attributes and their
        ``.trainable`` flags. The binding layer later normalizes optimizer
        keys like ``param.route_policy:0`` back to ``route_policy``.
        """
        for key, param in self.prompt_targets.items():
            span.set_attribute(f"param.{key}", _otel_attr_value(param, max_chars=10_000))
            span.set_attribute(f"param.{key}.trainable", _trainable_attr(param))

        for key, param in self.graph_knobs.items():
            span.set_attribute(f"param.{key}", _otel_attr_value(param, max_chars=10_000))
            span.set_attribute(f"param.{key}.trainable", _trainable_attr(param))
            span.set_attribute(f"graph.knob.{key}", _otel_attr_value(param))

        traced_fn = self._traced_functions.get(node_name)
        code_param = getattr(traced_fn, "parameter", None)
        if code_param is not None:
            span.set_attribute(f"param.__code_{node_name}", _otel_attr_value(code_param, max_chars=50_000))
            span.set_attribute(f"param.__code_{node_name}.trainable", _trainable_attr(code_param))

    def _emit_otel_inputs(self, span: Any, state: Any, *args: Any, **kwargs: Any) -> None:
        """Emit bounded input/state previews for graph reconstruction."""
        if isinstance(state, Mapping):
            for key, value in state.items():
                span.set_attribute(f"inputs.{key}", _otel_attr_value(value))
        else:
            span.set_attribute("inputs.state", _otel_attr_value(state))
        if args:
            span.set_attribute("inputs.args", _otel_attr_value(args))
        for key, value in kwargs.items():
            span.set_attribute(f"inputs.kwargs.{key}", _otel_attr_value(value))

    def _resolve_otel_runtime_fn(
        self,
        name: str,
        fallback_fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Resolve the live callable used by OTEL execution.

        OTEL optimization updates land on adapter bindings, including
        ``__code_*`` entries backed by ``FunModule.parameter``. To make those
        updates affect runtime behavior, OTEL execution must consult the
        traced wrapper's dynamic ``.fun`` property at call time instead of the
        original function object captured when the graph was built.
        """
        traced_fn = self._traced_functions.get(name)
        if isinstance(traced_fn, FunModule):
            return traced_fn.fun
        runtime_fn = getattr(traced_fn, "fun", None)
        if callable(runtime_fn):
            return runtime_fn
        return fallback_fn

    def _otel_runtime_wrapper(self, name: str, fn: Callable[..., Any]):
        """Wrap a graph function with an OTEL node span.

        ``InstrumentedGraph`` creates the root invocation span and activates a
        ``TelemetrySession``. This wrapper emits the per-node spans that the
        old LangGraph+OTEL prototype emitted manually inside each node.
        """
        def _wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                from opto.trace.io.telemetry_session import TelemetrySession
            except Exception:
                return fn(state, *args, **kwargs)

            session = TelemetrySession.current()
            if session is None:
                return fn(state, *args, **kwargs)

            runtime_state = state
            if isinstance(state, Mapping):
                runtime_state = dict(state)
                runtime_state.update(self._knob_values())

            with session.tracer.start_as_current_span(name) as span:
                span.set_attribute("message.id", name)
                span.set_attribute("graph.node.name", name)
                span.set_attribute("graph.backend", "otel")
                self._emit_otel_inputs(span, runtime_state, *args, **kwargs)
                self._emit_otel_parameters(span, node_name=name)

                try:
                    runtime_fn = self._resolve_otel_runtime_fn(name, fn)
                    output = runtime_fn(runtime_state, *args, **kwargs)
                except BaseException as exc:
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_attribute("error.message", str(exc))
                    raise

                span.set_attribute("outputs.preview", _otel_attr_value(output))
                return output

        _wrapped.__name__ = name
        return _wrapped

    def build_graph(self, backend: Optional[str] = None):
        """Build, compile, and cache the graph for ``trace`` or ``otel`` execution."""
        effective_backend = backend or self.backend
        key = self._cache_key(effective_backend)
        if key in self._compiled_cache:
            return self._compiled_cache[key]

        if effective_backend == "trace":
            fn_overrides = {
                name: self._trace_runtime_wrapper(name, fn)
                for name, fn in self._traced_functions.items()
            }
        elif effective_backend == "otel":
            fn_overrides = {
                name: self._otel_runtime_wrapper(name, fn)
                for name, fn in self._original_functions.items()
            }
        else:
            raise ValueError(f"Unsupported backend: {effective_backend!r}")

        call_kwargs = dict(self._knob_values())
        sig = inspect.signature(self.graph_factory)
        for name, fn in fn_overrides.items():
            if name in sig.parameters:
                call_kwargs[name] = fn

        with self._scope_override({**fn_overrides, **call_kwargs}):
            graph = self.graph_factory(**{k: v for k, v in call_kwargs.items() if k in sig.parameters})

        compiled = graph.compile() if hasattr(graph, "compile") else graph
        self._compiled_cache[key] = compiled
        return compiled

    def invoke_runtime(self, state: Dict[str, Any], backend: Optional[str] = None, **kwargs: Any):
        """Run the adapter using the runtime backend selected for this call."""
        effective_backend = backend or self.backend
        if effective_backend == "otel":
            graph = self.build_graph(backend="otel")
            result = graph.invoke(state, **kwargs)
            sidecar = OTELRunSidecar()
            sidecar.otlp = None
            sidecar.tgj_docs = None
            return result, sidecar
        return self.invoke_trace(state, **kwargs)

    def invoke_trace(self, state: Dict[str, Any], **kwargs: Any):
        """Execute the graph with traced wrappers and capture the output node."""
        sidecar = self.new_run_sidecar()
        for key, value in state.items():
            sidecar.shadow_state[key] = value if isinstance(value, Node) else node(value, name=key)
        for key, value in self.graph_knobs.items():
            sidecar.shadow_state[key] = value

        self._active_sidecar = sidecar
        runtime_state = dict(state)
        runtime_state.update(self._knob_values())
        try:
            graph = self.build_graph(backend="trace")
            result = graph.invoke(runtime_state, **kwargs)
        finally:
            self._active_sidecar = None

        output_node = None
        if self.output_key and self.output_key in sidecar.shadow_state:
            output_node = sidecar.shadow_state[self.output_key]
        elif isinstance(result, Node):
            output_node = result

        if output_node is None and self.output_key and isinstance(result, dict):
            output_value = result.get(self.output_key)
            output_node = output_value if isinstance(output_value, Node) else node(output_value, name=self.output_key)

        sidecar.set_output(output_node, result)
        return result, sidecar
