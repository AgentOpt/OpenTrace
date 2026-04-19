from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from opto.trace.io.telemetry_session import TelemetrySession


@dataclass
class ObserverArtifact:
    carrier: str
    raw: Any
    profile_doc: Optional[Dict[str, Any]] = None


class GraphObserver(Protocol):
    name: str

    def start(
        self,
        *,
        bindings: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    def stop(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> ObserverArtifact:
        ...


class OTelObserver:
    """Passive OTEL observer for a non-OTEL primary run."""

    name = "otel"

    def __init__(
        self,
        session: Optional[TelemetrySession] = None,
        *,
        service_name: str = "langgraph-otel-observer",
    ) -> None:
        self.session = session or TelemetrySession(service_name=service_name)
        self._ctx = None

    def start(
        self,
        *,
        bindings: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._ctx = self.session.activate()
        self._ctx.__enter__()

    def stop(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> ObserverArtifact:
        try:
            otlp = self.session.flush_otlp(clear=True)
        finally:
            if self._ctx is not None:
                self._ctx.__exit__(None, None, None)
                self._ctx = None
        return ObserverArtifact(carrier="otel", raw=otlp, profile_doc=None)
