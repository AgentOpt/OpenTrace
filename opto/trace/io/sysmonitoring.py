"""Lightweight graph observers built on Python's ``sys.monitoring`` hooks."""

from __future__ import annotations

import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from opto.trace.io.observers import ObserverArtifact


@dataclass
class SysMonEvent:
    """Recorded function-level event captured during a sys.monitoring session."""

    id: str
    parent_id: str | None
    name: str
    filename: str
    lineno: int
    start_ns: int
    end_ns: int | None = None
    duration_ns: int | None = None
    return_preview: str | None = None
    thread_id: int | None = None


class SysMonitoringSession:
    """Small execution observer built on Python's sys.monitoring API."""

    def __init__(self, tool_id: int = 7, service_name: str = "langgraph-sysmon") -> None:
        """Prepare a reusable session for collecting Python call events."""
        if not hasattr(sys, "monitoring"):
            raise RuntimeError("sys.monitoring is unavailable on this Python runtime")
        self.tool_id = tool_id
        self.service_name = service_name
        self._events: List[SysMonEvent] = []
        self._tls = threading.local()
        self._bindings_snapshot: Dict[str, Dict[str, Any]] = {}

    def _claim_tool_id(self) -> int:
        """Claim a valid sys.monitoring tool id.

        Python accepts tool ids in a small runtime-defined range (commonly 0..5).
        If the configured id is invalid or already taken, fall back to the first
        available id in that valid range.
        """
        candidate_ids = [self.tool_id] + [i for i in range(5, -1, -1) if i != self.tool_id]
        for candidate in candidate_ids:
            try:
                sys.monitoring.use_tool_id(candidate, self.service_name)
                self.tool_id = candidate
                return candidate
            except Exception:
                continue
        raise RuntimeError("Unable to claim a valid sys.monitoring tool id")

    def _stack(self) -> List[SysMonEvent]:
        """Return the thread-local event stack for nested Python calls."""
        if not hasattr(self._tls, "stack"):
            self._tls.stack = []
        return self._tls.stack

    def start(
        self,
        *,
        bindings: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start collecting Python start/return/unwind events for a graph run."""
        self._events.clear()
        self._bindings_snapshot = {
            k: {"value": b.get(), "kind": b.kind, "trainable": True}
            for k, b in (bindings or {}).items()
        }
        semantic_names = set((meta or {}).get("semantic_names") or ())

        def _safe_preview(value: Any) -> str:
            """Render a short preview without letting ``repr`` failures escape."""
            try:
                return repr(value)[:200]
            except Exception:
                return f"<{type(value).__name__}>"

        def on_start(code, instruction_offset):
            """Record the beginning of a monitored Python call."""
            if semantic_names and code.co_name not in semantic_names:
                return
            stack = self._stack()
            eid = uuid.uuid4().hex[:16]
            ev = SysMonEvent(
                id=eid,
                parent_id=stack[-1].id if stack else None,
                name=code.co_name,
                filename=code.co_filename,
                lineno=code.co_firstlineno,
                start_ns=time.perf_counter_ns(),
                thread_id=threading.get_ident(),
            )
            stack.append(ev)
            self._events.append(ev)

        def on_return(code, instruction_offset, retval):
            """Close the most recent matching event on normal return."""
            stack = self._stack()
            if not stack or stack[-1].name != code.co_name:
                return
            ev = stack.pop()
            ev.end_ns = time.perf_counter_ns()
            ev.duration_ns = ev.end_ns - ev.start_ns
            ev.return_preview = _safe_preview(retval)

        def on_unwind(code, instruction_offset, exc):
            """Close the most recent matching event when the frame unwinds."""
            stack = self._stack()
            if not stack or stack[-1].name != code.co_name:
                return
            ev = stack.pop()
            ev.end_ns = time.perf_counter_ns()
            ev.duration_ns = ev.end_ns - ev.start_ns
            ev.return_preview = f"[UNWIND] {type(exc).__name__}: {_safe_preview(exc)}"

        self._on_start = on_start
        self._on_return = on_return
        self._on_unwind = on_unwind

        self._claim_tool_id()
        sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.PY_START, on_start)
        sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.PY_RETURN, on_return)
        sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.PY_UNWIND, on_unwind)
        sys.monitoring.set_events(
            self.tool_id,
            sys.monitoring.events.PY_START
            | sys.monitoring.events.PY_RETURN
            | sys.monitoring.events.PY_UNWIND,
        )

    def stop(self, *, result: Any = None, error: BaseException | None = None) -> Dict[str, Any]:
        """Stop monitoring and export the captured profile document."""
        try:
            sys.monitoring.set_events(self.tool_id, 0)
            sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.PY_START, None)
            sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.PY_RETURN, None)
            sys.monitoring.register_callback(self.tool_id, sys.monitoring.events.PY_UNWIND, None)
        finally:
            free_tool = getattr(sys.monitoring, "free_tool_id", None)
            if callable(free_tool):
                try:
                    free_tool(self.tool_id)
                except Exception:
                    pass
            else:
                clear_tool = getattr(sys.monitoring, "clear_tool_id", None)
                if callable(clear_tool):
                    try:
                        clear_tool(self.tool_id)
                    except Exception:
                        pass

        return {
            "version": "trace-json/1.0+sysmon",
            "agent": {"id": self.service_name},
            "bindings": self._bindings_snapshot,
            "events": [
                {
                    "id": ev.id,
                    "parent_id": ev.parent_id,
                    "name": ev.name,
                    "file": ev.filename,
                    "lineno": ev.lineno,
                    "start_ns": ev.start_ns,
                    "end_ns": ev.end_ns,
                    "duration_ns": ev.duration_ns,
                    "return_preview": ev.return_preview,
                    "thread_id": ev.thread_id,
                }
                for ev in self._events
            ],
            "result_preview": repr(result)[:200] if result is not None else None,
            "error": repr(error)[:200] if error else None,
        }


class SysMonObserver:
    """Observer adapter that exposes ``SysMonitoringSession`` through the protocol."""

    name = "sysmon"

    def __init__(self, session: Optional[SysMonitoringSession] = None) -> None:
        """Reuse or create a monitoring session for passive observation."""
        self.session = session or SysMonitoringSession()

    def start(
        self,
        *,
        bindings: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delegate observer startup to the underlying monitoring session."""
        self.session.start(bindings=bindings, meta=meta)

    def stop(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> ObserverArtifact:
        """Stop monitoring and package the resulting profile as an artifact."""
        doc = self.session.stop(result=result, error=error)
        return ObserverArtifact(carrier="sysmon", raw=doc, profile_doc=doc)


def sysmon_profile_to_tgj(
    doc: Dict[str, Any],
    *,
    run_id: str,
    graph_id: str,
    scope: str,
) -> Dict[str, Any]:
    """Convert a simple sys.monitoring profile document into TGJ 1.0."""
    nodes = {}

    for pname, spec in (doc.get("bindings") or {}).items():
        nodes[f"param:{pname}"] = {
            "id": f"param:{pname}",
            "kind": "parameter",
            "name": pname,
            "value": spec["value"],
            "trainable": spec.get("trainable", True),
            "description": f"[{spec.get('kind', 'prompt')}]",
        }

    for ev in doc.get("events", []):
        inputs = {}
        if ev.get("parent_id"):
            inputs["parent"] = {"ref": f"msg:{ev['parent_id']}"}
        nodes[f"msg:{ev['id']}"] = {
            "id": f"msg:{ev['id']}",
            "kind": "message",
            "name": ev["name"],
            "description": f"[sysmon] {ev['file']}:{ev['lineno']}",
            "inputs": inputs,
            "output": {
                "name": f"{ev['name']}:out",
                "value": ev.get("return_preview"),
            },
            "info": {
                "sysmon": {
                    "duration_ns": ev.get("duration_ns"),
                    "thread_id": ev.get("thread_id"),
                }
            },
        }

    return {
        "tgj": "1.0",
        "run_id": run_id,
        "agent_id": (doc.get("agent") or {}).get("id", "agent"),
        "graph_id": graph_id,
        "scope": scope,
        "nodes": nodes,
    }
