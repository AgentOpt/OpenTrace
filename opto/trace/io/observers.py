"""Observer protocols used to collect passive artifacts alongside graph runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from opto.trace.io.telemetry_session import TelemetrySession


@dataclass
class ObserverArtifact:
    """Container for the raw payload emitted by an observer backend."""

    carrier: str
    raw: Any
    profile_doc: Optional[Dict[str, Any]] = None


class GraphObserver(Protocol):
    """Protocol implemented by passive observers attached to graph executions."""

    name: str

    def start(
        self,
        *,
        bindings: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Begin collecting artifacts for a new graph invocation."""
        ...

    def stop(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> ObserverArtifact:
        """Finish collection and return the observer-specific artifact bundle."""
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
        """Create an observer backed by its own or a shared telemetry session."""
        self.session = session or TelemetrySession(service_name=service_name)
        self._ctx = None

    def start(
        self,
        *,
        bindings: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Activate the telemetry session before the primary graph run starts."""
        self._ctx = self.session.activate()
        self._ctx.__enter__()

    def stop(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> ObserverArtifact:
        """Flush OTLP artifacts and close the activation context."""
        try:
            otlp = self.session.flush_otlp(clear=True)
        finally:
            if self._ctx is not None:
                self._ctx.__exit__(None, None, None)
                self._ctx = None
        return ObserverArtifact(carrier="otel", raw=otlp, profile_doc=None)
