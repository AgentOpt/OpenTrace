from __future__ import annotations

import contextlib
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
    return getattr(value, "data", value)


def _normalize_named_callables(
    targets: Union[None, List[str], List[Callable[..., Any]], Mapping[str, Callable[..., Any]]],
    scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Callable[..., Any]]:
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
    if isinstance(value, ParameterNode):
        return value
    if isinstance(value, Node):
        return node(value, name=name, trainable=True)
    return node(value, name=name, trainable=True)


@dataclass
class GraphAdapter:
    graph_factory: Callable[..., Any]
    backend: str = "trace"
    bindings: Dict[str, Binding] = field(default_factory=dict)
    service_name: str = "graph-adapter"
    input_key: str = "query"
    output_key: Optional[str] = None

    def build_graph(self, backend: Optional[str] = None):
        raise NotImplementedError

    def invoke_runtime(self, state: Dict[str, Any], **kwargs: Any):
        raise NotImplementedError

    def invoke_trace(self, state: Dict[str, Any], **kwargs: Any):
        raise NotImplementedError

    def new_run_sidecar(self):
        return GraphRunSidecar()

    def bindings_dict(self) -> Dict[str, Binding]:
        return dict(self.bindings)

    def parameters(self) -> List[ParameterNode]:
        raise NotImplementedError

    def as_module(self) -> GraphModule:
        return GraphModule(self)

    def instrument(self, backend: Optional[str] = None, **kwargs: Any):
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
class LangGraphAdapter(GraphAdapter):
    function_targets: Union[None, List[str], List[Callable[..., Any]], Mapping[str, Callable[..., Any]]] = None
    prompt_targets: Optional[Mapping[str, Any]] = None
    graph_knobs: Optional[Mapping[str, Any]] = None
    scope: Optional[Dict[str, Any]] = None
    train_graph_agents_functions: bool = True

    def __post_init__(self) -> None:
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
        state = self.__dict__.copy()
        state["_active_sidecar"] = None
        state["_compiled_cache"] = {}
        return state

    def _build_bindings(self) -> None:
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
        return {k: _raw(v) for k, v in self.graph_knobs.items()}

    def _cache_key(self, backend: str):
        return backend, tuple(sorted((k, repr(v)) for k, v in self._knob_values().items()))

    @contextlib.contextmanager
    def _scope_override(self, overrides: Dict[str, Any]):
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
        def _wrapped(state: Dict[str, Any], *args: Any, **kwargs: Any):
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

    def build_graph(self, backend: Optional[str] = None):
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
            fn_overrides = dict(self._original_functions)
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
