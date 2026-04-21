"""Lightweight per-run sidecars shared by graph instrumentation backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GraphRunSidecar:
    """Per-run sidecar preserving optimization state alongside runtime outputs."""

    node_outputs: Dict[str, Any] = field(default_factory=dict)
    shadow_state: Dict[str, Any] = field(default_factory=dict)
    output_node: Any | None = None
    runtime_result: Any | None = None

    def record_node_output(
        self,
        node_name: str,
        traced_output: Any,
        runtime_value: Any = None,
    ) -> None:
        """Store the traced node output and any dict-shaped runtime shadow state."""
        self.node_outputs[node_name] = traced_output
        if runtime_value is not None and isinstance(runtime_value, dict):
            self.shadow_state.update(runtime_value)

    def set_output(self, output_node: Any, runtime_result: Any) -> None:
        """Record the final traced output node alongside the raw runtime result."""
        self.output_node = output_node
        self.runtime_result = runtime_result

    def clear(self) -> None:
        """Reset the sidecar for reuse in tests or debugging flows."""
        self.node_outputs.clear()
        self.shadow_state.clear()
        self.output_node = None
        self.runtime_result = None


@dataclass
class OTELRunSidecar:
    """OTEL artefacts sidecar for a single graph run."""

    otlp: Dict[str, Any] | None = None
    tgj_docs: List[Dict[str, Any]] | None = None


@dataclass
class GraphCandidateSnapshot:
    """Debug/introspection snapshot for graph candidate state."""

    graph_knobs: Dict[str, Any] = field(default_factory=dict)
    parameter_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
