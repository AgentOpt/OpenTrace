"""
opto.trace.io.telemetry_session
===============================

Unified session manager for OTEL traces and (optionally) MLflow.

A ``TelemetrySession`` owns a ``TracerProvider`` + ``InMemorySpanExporter``
and exposes:

* ``flush_otlp()`` – extract collected spans as OTLP JSON and optionally clear
* ``flush_tgj()`` – convert spans to Trace-Graph JSON via ``otel_adapter``
* ``export_run_bundle()`` – dump all session data to a directory
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace as oteltrace
from opentelemetry.sdk.trace import TracerProvider

from opto.trace.io.langgraph_otel_runtime import (
    InMemorySpanExporter,
    flush_otlp as _flush_otlp_raw,
)
from opto.trace.io.otel_adapter import otlp_traces_to_trace_json

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        service_name: str = "trace-session",
        *,
        record_spans: bool = True,
        span_attribute_filter: Optional[
            Callable[[str, Dict[str, Any]], Dict[str, Any]]
        ] = None,
    ) -> None:
        self.service_name = service_name
        self.record_spans = record_spans
        self.span_attribute_filter = span_attribute_filter

        # OTEL plumbing
        self._exporter = InMemorySpanExporter()
        self._provider = TracerProvider()

        if self.record_spans:
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            self._provider.add_span_processor(
                SimpleSpanProcessor(self._exporter)
            )

        self._tracer = self._provider.get_tracer(service_name)

    # -- properties ----------------------------------------------------------

    @property
    def tracer(self) -> oteltrace.Tracer:
        """The OTEL tracer for manual span creation."""
        return self._tracer

    @property
    def exporter(self) -> InMemorySpanExporter:
        """Direct access to the in-memory span exporter."""
        return self._exporter

    # -- flush methods -------------------------------------------------------

    def flush_otlp(self, *, clear: bool = True) -> Dict[str, Any]:
        """Flush collected spans to OTLP JSON.

        Parameters
        ----------
        clear : bool
            If *True* (default), clear the exporter after flushing.
            If *False*, peek at current spans without clearing (B5).

        Returns
        -------
        dict
            OTLP JSON payload compatible with ``otel_adapter``.
        """
        if not self.record_spans:
            return {"resourceSpans": []}

        # Delegate clear semantics to the low-level flush helper
        otlp = _flush_otlp_raw(
            self._exporter,
            scope_name=self.service_name,
            clear=clear,
        )

        # Apply span_attribute_filter if configured (B6)
        if self.span_attribute_filter is not None:
            otlp = self._apply_attribute_filter(otlp)

        return otlp

    def _apply_attribute_filter(self, otlp: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ``span_attribute_filter`` to all spans in the OTLP payload.

        * If the filter returns ``{}``, the span is **dropped** entirely.
        * Otherwise the returned dict replaces the span's attributes.
        """
        if self.span_attribute_filter is None:
            return otlp

        filtered_rs = []
        for rs in otlp.get("resourceSpans", []):
            filtered_ss = []
            for ss in rs.get("scopeSpans", []):
                filtered_spans = []
                for sp in ss.get("spans", []):
                    span_name = sp.get("name", "")
                    # Build a plain dict from OTLP attributes
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
                        # Filter returned {} → drop this span
                        continue

                    if new_attrs is not None:
                        # Rebuild OTLP attributes from the filtered dict
                        sp = dict(sp)  # shallow copy
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
        """Flush collected spans to Trace-Graph JSON format.

        Returns
        -------
        list[dict]
            TGJ documents ready for ``ingest_tgj()``.
        """
        otlp = self.flush_otlp(clear=clear)
        return otlp_traces_to_trace_json(
            otlp,
            agent_id_hint=agent_id_hint or self.service_name,
            use_temporal_hierarchy=use_temporal_hierarchy,
        )

    # -- internal helpers (used by optimization.py) --------------------------

    def _flush_tgj_from_otlp(self, otlp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert an already-flushed OTLP payload to TGJ (no exporter access)."""
        return otlp_traces_to_trace_json(
            otlp,
            agent_id_hint=self.service_name,
            use_temporal_hierarchy=True,
        )

    # -- export helpers ------------------------------------------------------

    def export_run_bundle(
        self,
        output_dir: str,
        *,
        include_otlp: bool = True,
        include_tgj: bool = True,
        include_prompts: bool = True,
        prompts: Optional[Dict[str, str]] = None,
    ) -> str:
        """Export all session data to a directory bundle.

        Returns the path to the bundle directory.
        """
        os.makedirs(output_dir, exist_ok=True)

        otlp = self.flush_otlp(clear=True)

        if include_otlp:
            otlp_path = os.path.join(output_dir, "otlp_trace.json")
            with open(otlp_path, "w") as f:
                json.dump(otlp, f, indent=2)

        if include_tgj:
            tgj_docs = otlp_traces_to_trace_json(
                otlp,
                agent_id_hint=self.service_name,
                use_temporal_hierarchy=True,
            )
            tgj_path = os.path.join(output_dir, "trace_graph.json")
            with open(tgj_path, "w") as f:
                json.dump(tgj_docs, f, indent=2)

        if include_prompts and prompts:
            prompts_path = os.path.join(output_dir, "prompts.json")
            with open(prompts_path, "w") as f:
                json.dump(prompts, f, indent=2)

        logger.info("Exported run bundle to %s", output_dir)
        return output_dir
