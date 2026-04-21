"""
opto.trace.io.telemetry_session
===============================

Unified session manager for OTEL traces and (optionally) MLflow.

A ``TelemetrySession`` owns a ``TracerProvider`` + ``InMemorySpanExporter``
and exposes:

* ``flush_otlp()`` – extract collected spans as OTLP JSON and optionally clear
* ``flush_tgj()`` – convert spans to Trace-Graph JSON via ``otel_adapter``
* ``export_run_bundle()`` – dump all session data to a directory

In addition, when a session is **activated** (``with TelemetrySession()`` or
``TelemetrySession.activate()``), Trace-level operators can optionally emit
spans for non-LangGraph pipelines (e.g. ``@trace.bundle`` operations).
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from opentelemetry import trace as oteltrace
from opentelemetry.sdk.trace import TracerProvider

from opto.trace.io.otel_runtime import (
    InMemorySpanExporter,
    flush_otlp as _flush_otlp_raw,
)
from opto.trace.io.otel_adapter import otlp_traces_to_trace_json

logger = logging.getLogger(__name__)

_CURRENT_SESSION: contextvars.ContextVar[Optional["TelemetrySession"]] = (
    contextvars.ContextVar("opto_trace_current_telemetry_session", default=None)
)


@dataclass(frozen=True)
class BundleSpanConfig:
    """Controls optional OTEL spans around ``@trace.bundle`` ops.

    The defaults are intentionally conservative to avoid span noise.
    """

    enable: bool = True
    disable_default_ops: bool = True
    capture_inputs: bool = True


@dataclass(frozen=True)
class MessageNodeTelemetryConfig:
    """Controls how MessageNodes are associated to OTEL spans.

    Modes:
    - ``"off"``: no binding/spans
    - ``"bind"``: attach ``message.id`` to the current span (if any)
    - ``"span"``: if no current span, create a minimal span for the node
    """

    mode: str = "bind"


class TelemetrySession:
    """Manages an OTEL tracing session with export capabilities.

    Parameters
    ----------
    service_name : str
        OTEL service / scope name.
    record_spans : bool
        If *False*, disable span recording entirely (safe no-op).
    span_attribute_filter : callable, optional
        ``(span_name, attrs_dict) -> attrs_dict``.  Return ``{}`` to drop the
        span entirely.  Useful for redacting secrets or truncating payloads.
    bundle_spans : BundleSpanConfig, optional
        Enable optional OTEL spans around ``@trace.bundle`` operations when this
        session is active (non-LangGraph pipelines).
    message_nodes : MessageNodeTelemetryConfig, optional
        Controls how ``MessageNode`` creation binds to spans (used to keep a
        stable Node-to-Span mapping for TGJ conversion).
    max_attr_chars : int
        Max characters for any attribute value written by the session helpers.
    mlflow_log_artifacts : bool
        If True, ``export_run_bundle()`` will also attempt to log the bundle
        directory as MLflow artifacts (best-effort no-op when unavailable).
    mlflow_autolog : bool
        If True, best-effort enable MLflow autologging so ``@trace.bundle`` ops
        can also be wrapped by ``mlflow.trace`` while this session is used.
        This keeps MLflow optional and preserves backward compatibility.
    """

    def __init__(
        self,
        service_name: str = "trace-session",
        *,
        record_spans: bool = True,
        span_attribute_filter: Optional[
            Callable[[str, Dict[str, Any]], Dict[str, Any]]
        ] = None,
        bundle_spans: Optional[BundleSpanConfig] = None,
        message_nodes: Optional[MessageNodeTelemetryConfig] = None,
        max_attr_chars: int = 500,
        mlflow_log_artifacts: bool = False,
        mlflow_autolog: bool = False,
        mlflow_autolog_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.service_name = service_name
        self.record_spans = record_spans
        self.span_attribute_filter = span_attribute_filter
        self.bundle_spans = bundle_spans or BundleSpanConfig()
        self.message_nodes = message_nodes or MessageNodeTelemetryConfig()
        self.max_attr_chars = int(max_attr_chars)
        self.mlflow_log_artifacts = bool(mlflow_log_artifacts)
        self.mlflow_autolog = bool(mlflow_autolog)
        self.mlflow_autolog_kwargs = dict(mlflow_autolog_kwargs or {})

        # OTEL plumbing
        self._exporter = InMemorySpanExporter()
        self._provider = TracerProvider()

        if self.record_spans:
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            self._provider.add_span_processor(
                SimpleSpanProcessor(self._exporter)
            )

        self._tracer = self._provider.get_tracer(service_name)

        # Node -> OTEL span-id mapping for "inputs.*" reference lifting.
        # WeakKeyDictionary avoids preventing GC for graphs created during optimization loops.
        self._node_span_ids: "weakref.WeakKeyDictionary[object, str]" = (
            weakref.WeakKeyDictionary()
        )

        self._message_node_records: List[Dict[str, Any]] = []

        # Activation token stack (supports nested with-blocks on the same instance)
        self._token_stack: List[contextvars.Token] = []

        # Optional MLflow bridge: keep MLflow optional and do not fail session
        # construction if MLflow is unavailable.
        if self.mlflow_autolog:
            try:
                from opto.features.mlflow.autolog import autolog as _mlflow_autolog
                kwargs = {"silent": True}
                kwargs.update(self.mlflow_autolog_kwargs)
                _mlflow_autolog(**kwargs)
            except Exception as e:
                logger.debug(
                    "TelemetrySession could not enable MLflow autologging: %s",
                    e,
                )

    # -- activation -----------------------------------------------------------

    @classmethod
    def current(cls) -> Optional["TelemetrySession"]:
        """Return the currently-active session (if any)."""
        return _CURRENT_SESSION.get()

    @contextlib.contextmanager
    def activate(self):
        """Activate this session in the current context.

        When active, instrumentation hooks (e.g. bundle spans, MessageNode binding)
        can discover the session via ``TelemetrySession.current()``.
        """
        token = _CURRENT_SESSION.set(self)
        try:
            yield self
        finally:
            _CURRENT_SESSION.reset(token)

    def __enter__(self) -> "TelemetrySession":
        token = _CURRENT_SESSION.set(self)
        self._token_stack.append(token)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token_stack:
            token = self._token_stack.pop()
            _CURRENT_SESSION.reset(token)

    def set_current(self) -> "TelemetrySession":
        """Activate this session without a context manager.

        Useful in notebooks or scripts where indenting all code under a
        ``with`` block is impractical.  Must be paired with a later call
        to :meth:`clear_current`.

        Returns the session instance for chaining.
        """
        token = _CURRENT_SESSION.set(self)
        self._token_stack.append(token)
        return self

    def clear_current(self) -> None:
        """Deactivate the most recent :meth:`set_current` activation."""
        if self._token_stack:
            token = self._token_stack.pop()
            _CURRENT_SESSION.reset(token)

    # -- properties -----------------------------------------------------------

    @property
    def tracer(self) -> oteltrace.Tracer:
        """The OTEL tracer for manual span creation."""
        return self._tracer

    @property
    def exporter(self) -> InMemorySpanExporter:
        """Direct access to the in-memory span exporter."""
        return self._exporter

    # -- span helpers ---------------------------------------------------------

    @staticmethod
    def _span_id_hex(span) -> Optional[str]:
        try:
            ctx = span.get_span_context()
            if not getattr(ctx, "is_valid", False):
                return None
            return f"{ctx.span_id:016x}"
        except Exception:
            return None

    def _truncate(self, v: Any) -> str:
        s = str(v)
        if self.max_attr_chars and len(s) > self.max_attr_chars:
            return s[: self.max_attr_chars] + "…"
        return s

    def _is_trace_node(self, obj: Any) -> bool:
        mod = getattr(obj.__class__, "__module__", "")
        return mod.startswith("opto.trace") and hasattr(obj, "name") and hasattr(obj, "data")

    def _is_parameter_node(self, obj: Any) -> bool:
        return self._is_trace_node(obj) and obj.__class__.__name__ == "ParameterNode"

    def _param_key(self, param_node: Any) -> str:
        raw = getattr(param_node, "name", "param")
        return str(raw).split(":")[0]

    def _remember_node_span(self, node: Any, span) -> None:
        sid = self._span_id_hex(span)
        if sid is None:
            return
        try:
            self._node_span_ids[node] = sid
        except TypeError:
            return

    def _lookup_node_ref(self, node: Any) -> Optional[str]:
        try:
            sid = self._node_span_ids.get(node)
        except Exception:
            sid = None
        if not sid:
            return None
        # Prefer stable message.id (node.name) over raw span ID
        msg_id = getattr(node, "name", None)
        if msg_id:
            return f"{self.service_name}:{msg_id}"
        return f"{self.service_name}:{sid}"

    def _inputs_and_params_from_trace_inputs(
        self, inputs: Dict[str, Any]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Convert a Trace inputs dict into OTEL attribute fragments.

        Returns ``(inputs_attrs, params_attrs)`` where:
        - ``inputs_attrs`` maps ``inputs.<k>`` to a reference-or-literal
        - ``params_attrs`` maps ``param.<name>`` (+ trainable) to a value
        """
        inputs_attrs: Dict[str, str] = {}
        params_attrs: Dict[str, str] = {}

        for k, v in (inputs or {}).items():
            if self._is_parameter_node(v):
                pname = self._param_key(v)
                params_attrs[f"param.{pname}"] = self._truncate(getattr(v, "data", ""))
                params_attrs[f"param.{pname}.trainable"] = str(
                    bool(getattr(v, "trainable", False))
                ).lower()

            if self._is_trace_node(v):
                ref = self._lookup_node_ref(v)
                if ref is not None:
                    inputs_attrs[f"inputs.{k}"] = ref
                else:
                    inputs_attrs[f"inputs.{k}"] = f"lit:{self._truncate(getattr(v, 'data', ''))}"
            else:
                inputs_attrs[f"inputs.{k}"] = f"lit:{self._truncate(v)}"

        return inputs_attrs, params_attrs

    def _is_default_op(self, fun_name: str, file_path: str) -> bool:
        if fun_name == "call_llm":
            return False
        norm = str(file_path).replace("\\", "/")
        return norm.endswith("/trace/operators.py")

    @contextlib.contextmanager
    def bundle_span(self, *, fun_name: str, file_path: str, inputs: Dict[str, Any]):
        """Context manager for an OTEL span around a bundle op."""
        if not (self.record_spans and self.bundle_spans.enable):
            yield None
            return

        if self.bundle_spans.disable_default_ops and self._is_default_op(fun_name, file_path):
            yield None
            return

        attrs: Dict[str, Any] = {
            "trace.bundle": "true",
            "trace.bundle.fun_name": fun_name,
            "trace.bundle.file": str(file_path),
        }

        if self.bundle_spans.capture_inputs:
            in_attrs, p_attrs = self._inputs_and_params_from_trace_inputs(inputs or {})
            attrs.update(in_attrs)
            attrs.update(p_attrs)

        with self.tracer.start_as_current_span(fun_name) as sp:
            for k, v in attrs.items():
                try:
                    sp.set_attribute(k, v)
                except Exception:
                    sp.set_attribute(k, str(v))
            yield sp

    def on_message_node_created(self, node: Any, *, inputs: Optional[Dict[str, Any]] = None) -> None:
        """Hook invoked from ``MessageNode.__init__`` (best-effort).

        - If there's a current span: bind ``message.id`` and remember Node-to-Span mapping.
        - Optionally, if mode == "span" and no current span exists, create a minimal span.
        """
        mode = (self.message_nodes.mode or "off").lower()
        if mode == "off" or not self.record_spans:
            return

        try:
            rec = {
                "name": getattr(node, "name", None),
                "op": getattr(node, "op_name", None) if hasattr(node, "op_name") else None,
            }
            if inputs:
                rec["inputs"] = {
                    k: getattr(v, "name", None) if self._is_trace_node(v) else v
                    for k, v in inputs.items()
                }
            self._message_node_records.append(rec)
        except Exception:
            pass

        cur = oteltrace.get_current_span()
        if cur is not None:
            try:
                ctx = cur.get_span_context()
                if getattr(ctx, "is_valid", False) and cur.is_recording():
                    if inputs:
                        in_attrs, p_attrs = self._inputs_and_params_from_trace_inputs(inputs)
                        for attrs in (in_attrs, p_attrs):
                            for key, value in attrs.items():
                                try:
                                    cur.set_attribute(key, value)
                                except Exception:
                                    cur.set_attribute(key, str(value))
                    cur.set_attribute("message.id", str(getattr(node, "name", "")))
                    self._remember_node_span(node, cur)
                    return
            except Exception:
                pass

        if mode != "span":
            return

        span_name = str(getattr(node, "name", "message_node"))
        attrs: Dict[str, Any] = {"message.id": span_name}
        if inputs:
            in_attrs, p_attrs = self._inputs_and_params_from_trace_inputs(inputs)
            attrs.update(in_attrs)
            attrs.update(p_attrs)

        with self.tracer.start_as_current_span(span_name) as sp:
            for k, v in attrs.items():
                try:
                    sp.set_attribute(k, v)
                except Exception:
                    sp.set_attribute(k, str(v))
            self._remember_node_span(node, sp)

    # -- flush methods --------------------------------------------------------

    def flush_otlp(self, *, clear: bool = True) -> Dict[str, Any]:
        """Flush collected spans to OTLP JSON.

        Parameters
        ----------
        clear : bool
            If *True* (default), clear the exporter after flushing.
            If *False*, peek at current spans without clearing.

        Returns
        -------
        dict
            OTLP JSON payload compatible with ``otel_adapter``.
        """
        if not self.record_spans:
            return {"resourceSpans": []}

        otlp = _flush_otlp_raw(
            self._exporter,
            scope_name=self.service_name,
            clear=clear,
        )

        if self.span_attribute_filter is not None:
            otlp = self._apply_attribute_filter(otlp)

        return otlp

    def _apply_attribute_filter(self, otlp: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ``span_attribute_filter`` to all spans in the OTLP payload."""
        if self.span_attribute_filter is None:
            return otlp

        filtered_rs = []
        for rs in otlp.get("resourceSpans", []):
            filtered_ss = []
            for ss in rs.get("scopeSpans", []):
                filtered_spans = []
                for sp in ss.get("spans", []):
                    span_name = sp.get("name", "")
                    attrs_dict: Dict[str, Any] = {}
                    for a in sp.get("attributes", []):
                        key = a.get("key")
                        val = a.get("value", {})
                        if isinstance(val, dict) and "stringValue" in val:
                            attrs_dict[key] = val["stringValue"]
                        else:
                            attrs_dict[key] = str(val)

                    new_attrs = self.span_attribute_filter(span_name, attrs_dict)

                    if not new_attrs and new_attrs is not None:
                        continue

                    if new_attrs is not None:
                        sp = dict(sp)
                        sp["attributes"] = [
                            {"key": k, "value": {"stringValue": str(v)}}
                            for k, v in new_attrs.items()
                        ]
                    filtered_spans.append(sp)

                ss_copy = dict(ss)
                ss_copy["spans"] = filtered_spans
                filtered_ss.append(ss_copy)

            rs_copy = dict(rs)
            rs_copy["scopeSpans"] = filtered_ss
            filtered_rs.append(rs_copy)

        return {"resourceSpans": filtered_rs}

    def flush_tgj(
        self,
        *,
        agent_id_hint: str = "",
        use_temporal_hierarchy: bool = True,
        clear: bool = True,
    ) -> List[Dict[str, Any]]:
        """Flush collected spans to Trace-Graph JSON format."""
        otlp = self.flush_otlp(clear=clear)
        return otlp_traces_to_trace_json(
            otlp,
            agent_id_hint=agent_id_hint or self.service_name,
            use_temporal_hierarchy=use_temporal_hierarchy,
        )

    # -- internal helpers (used by optimization.py) ---------------------------

    def _flush_tgj_from_otlp(self, otlp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert an already-flushed OTLP payload to TGJ (no exporter access)."""
        return otlp_traces_to_trace_json(
            otlp,
            agent_id_hint=self.service_name,
            use_temporal_hierarchy=True,
        )

    # -- MLflow helpers (best-effort) -----------------------------------------

    def _mlflow_log_artifacts(self, output_dir: str) -> None:
        if not self.mlflow_log_artifacts:
            return
        try:
            import mlflow  # type: ignore
        except Exception:
            return
        try:
            mlflow.log_artifacts(output_dir)
        except Exception as e:
            logger.debug("MLflow artifact logging skipped: %s", e)

    def log_metric(self, key: str, value: float, *, step: Optional[int] = None) -> None:
        """Best-effort metric logging to MLflow (if available)."""
        try:
            import mlflow  # type: ignore
        except Exception:
            return
        try:
            if step is None:
                mlflow.log_metric(key, float(value))
            else:
                mlflow.log_metric(key, float(value), step=int(step))
        except Exception:
            return

    def log_param(self, key: str, value: Any) -> None:
        """Best-effort param logging to MLflow (if available)."""
        try:
            import mlflow  # type: ignore
        except Exception:
            return
        try:
            mlflow.log_param(key, str(value))
        except Exception:
            return

    # -- export helpers -------------------------------------------------------

    def export_run_bundle(
        self,
        output_dir: str,
        *,
        include_otlp: bool = True,
        include_tgj: bool = True,
        include_prompts: bool = True,
        prompts: Optional[Dict[str, str]] = None,
        include_node_records: bool = True,
        include_manifest: bool = True,
    ) -> str:
        """Export all session data to a directory bundle.

        File naming is aligned with the repository demos:

        - ``otlp.json`` (and legacy alias ``otlp_trace.json``)
        - ``tgj.json`` (and legacy alias ``trace_graph.json``)
        - ``prompts.json`` (optional)
        - ``message_nodes.jsonl`` (optional lightweight debug log)
        - ``manifest.json`` (optional)

        Returns the path to the bundle directory.
        """
        os.makedirs(output_dir, exist_ok=True)

        otlp = self.flush_otlp(clear=True)

        manifest: Dict[str, Any] = {
            "created_at": time.time(),
            "service_name": self.service_name,
            "files": {},
        }

        if include_otlp:
            otlp_path = os.path.join(output_dir, "otlp.json")
            with open(otlp_path, "w") as f:
                json.dump(otlp, f, indent=2)
            manifest["files"]["otlp"] = "otlp.json"

            alias = os.path.join(output_dir, "otlp_trace.json")
            try:
                if not os.path.exists(alias):
                    with open(alias, "w") as f:
                        json.dump(otlp, f, indent=2)
            except Exception:
                pass

        if include_tgj:
            tgj_docs = otlp_traces_to_trace_json(
                otlp,
                agent_id_hint=self.service_name,
                use_temporal_hierarchy=True,
            )
            tgj_path = os.path.join(output_dir, "tgj.json")
            with open(tgj_path, "w") as f:
                json.dump(tgj_docs, f, indent=2)
            manifest["files"]["tgj"] = "tgj.json"

            alias = os.path.join(output_dir, "trace_graph.json")
            try:
                if not os.path.exists(alias):
                    with open(alias, "w") as f:
                        json.dump(tgj_docs, f, indent=2)
            except Exception:
                pass

        if include_prompts and prompts:
            prompts_path = os.path.join(output_dir, "prompts.json")
            with open(prompts_path, "w") as f:
                json.dump(prompts, f, indent=2)
            manifest["files"]["prompts"] = "prompts.json"

        if include_node_records and self._message_node_records:
            p = os.path.join(output_dir, "message_nodes.jsonl")
            with open(p, "w") as f:
                for rec in self._message_node_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest["files"]["message_nodes"] = "message_nodes.jsonl"

        if include_manifest:
            p = os.path.join(output_dir, "manifest.json")
            with open(p, "w") as f:
                json.dump(manifest, f, indent=2)

        self._mlflow_log_artifacts(output_dir)

        logger.info("Exported run bundle to %s", output_dir)
        return output_dir
