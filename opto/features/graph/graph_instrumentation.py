"""Graph instrumentation helpers shared by multiple IO backends."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opto.trace import node
from opto.trace.bundle import FunModule
from opto.trace.io.bindings import Binding
from opto.trace.io.observers import GraphObserver


@dataclass
class TraceGraph:
    """Trace-native instrumented graph wrapper.

    This backend rebuilds a graph after rebinding selected node functions with
    ``trace.bundle()``. Graph execution then emits native Trace nodes directly,
    without relying on OTEL as the primary optimization carrier.
    """

    graph: Any
    parameters: List[Any] = field(default_factory=list)
    bindings: Dict[str, Any] = field(default_factory=dict)
    service_name: str = "langgraph-trace"
    input_key: str = "query"
    output_key: Optional[str] = None
    backend: str = "trace"
    _last_sidecar: Any = field(default=None, repr=False, init=False)
    observers: List[GraphObserver] = field(default_factory=list)
    _last_observer_artifacts: List[Any] = field(default_factory=list, init=False, repr=False)
    observer_meta: Dict[str, Any] = field(default_factory=dict)

    def invoke(self, state: Any, **kwargs: Any) -> Any:
        for obs in self.observers:
            meta = {"service_name": self.service_name}
            meta.update(self.observer_meta)
            obs.start(bindings=self.bindings, meta=meta)

        result = None
        error = None
        try:
            if hasattr(self.graph, "invoke_runtime"):
                result, sidecar = self.graph.invoke_runtime(state, backend="trace", **kwargs)
                self._last_sidecar = sidecar
                return result
            result = self.graph.invoke(state, **kwargs)
            return result
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._last_observer_artifacts = []
            for obs in reversed(self.observers):
                self._last_observer_artifacts.append(obs.stop(result=result, error=error))

    def stream(self, state: Any, **kwargs: Any):
        yield from self.graph.stream(state, **kwargs)


def _dedupe_identity(values: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for value in values:
        obj_id = id(value)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        out.append(value)
    return out


def _to_funmodule(
    func: Any,
    *,
    trainable: bool = True,
    traceable_code: bool = True,
    allow_external_dependencies: bool = True,
    scope: Optional[Dict[str, Any]] = None,
) -> Any:
    if isinstance(func, FunModule) or hasattr(func, "_fun"):
        return func

    wrapped = FunModule(
        fun=func,
        trainable=trainable,
        traceable_code=traceable_code,
        allow_external_dependencies=allow_external_dependencies,
        _ldict=(scope or {}),
    )

    try:
        wrapped.__signature__ = inspect.signature(wrapped._fun)
    except Exception:
        # Signature metadata is nice-to-have only.
        pass

    return wrapped


def _replace_scope_object(scope: Dict[str, Any], old_obj: Any, new_obj: Any) -> bool:
    replaced = False
    for key, value in list(scope.items()):
        if value is old_obj:
            scope[key] = new_obj
            replaced = True
    return replaced


def instrument_trace_graph(
    graph_factory: Callable[[], Any],
    *,
    scope: Dict[str, Any],
    graph_agents_functions: List[str],
    graph_prompts_list: Optional[List[Any]] = None,
    train_graph_agents_functions: bool = True,
    service_name: str = "langgraph-trace",
    input_key: str = "query",
    output_key: Optional[str] = None,
) -> TraceGraph:
    """Instrument a graph in trace-native mode.

    The graph factory is rebuilt *after* rebinding selected functions in scope
    with ``trace.bundle()``. Prompt objects can be passed as already-trace nodes
    or raw objects that are replaced in the given scope by identity.
    """
    if scope is None:
        raise ValueError("backend='trace' requires scope=globals() or equivalent")
    if not callable(graph_factory):
        raise ValueError("backend='trace' requires a callable graph_factory")

    parameters: List[Any] = []
    bindings: Dict[str, Binding] = {}

    for name in graph_agents_functions:
        if name not in scope:
            raise KeyError(f"Function '{name}' not found in scope")
        fn = scope[name]
        if fn is None or not callable(fn):
            raise ValueError(f"Function '{name}' is not callable: {fn!r}")

        wrapped = _to_funmodule(
            fn,
            trainable=train_graph_agents_functions,
            traceable_code=True,
            allow_external_dependencies=True,
            scope=scope,
        )
        scope[name] = wrapped
        if hasattr(wrapped, "parameters"):
            parameters.extend(list(wrapped.parameters()))

    if graph_prompts_list:
        for idx, prompt in enumerate(list(graph_prompts_list)):
            if hasattr(prompt, "data") and hasattr(prompt, "name"):
                parameters.append(prompt)
                prompt_name = str(getattr(prompt, "name", f"prompt_{idx}")).split(":")[0]
                if hasattr(prompt, "_set"):
                    bindings[prompt_name] = Binding(
                        get=lambda p=prompt: p.data,
                        set=lambda v, p=prompt: p._set(v),
                        kind="prompt",
                    )
                continue

            new_prompt = node(str(getattr(prompt, "data", prompt)), trainable=True)
            if not _replace_scope_object(scope, prompt, new_prompt):
                raise ValueError(
                    "Prompt object was not found in scope by identity. Pass a trace "
                    "node prompt or provide scope that contains the prompt object."
                )

            graph_prompts_list[idx] = new_prompt
            parameters.append(new_prompt)
            prompt_name = str(getattr(new_prompt, "name", f"prompt_{idx}")).split(":")[0]
            if hasattr(new_prompt, "_set"):
                bindings[prompt_name] = Binding(
                    get=lambda p=new_prompt: p.data,
                    set=lambda v, p=new_prompt: p._set(v),
                    kind="prompt",
                )

    graph = graph_factory()
    compiled = graph.compile() if hasattr(graph, "compile") else graph

    return TraceGraph(
        graph=compiled,
        parameters=_dedupe_identity(parameters),
        bindings=bindings,
        service_name=service_name,
        input_key=input_key,
        output_key=output_key,
        observer_meta={
            "semantic_names": [str(name).split(".")[-1] for name in (graph_agents_functions or [])]
        },
    )
