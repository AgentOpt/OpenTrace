"""``Module`` wrapper that lets graph adapters participate in Trace models."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from opto.trace.modules import Module

if TYPE_CHECKING:
    from opto.features.graph.adapter import GraphAdapter


class GraphModule(Module):
    """Module view over a graph adapter."""

    def __init__(self, adapter: "GraphAdapter"):
        """Store the adapter and initialize per-run sidecar tracking."""
        self.adapter = adapter
        self._last_sidecar = None

    def forward(self, x: Any):
        """Run the adapter in trace mode and return the traced output node."""
        state = x if isinstance(x, dict) else {self.adapter.input_key: x}
        _runtime, sidecar = self.adapter.invoke_trace(state)
        self._last_sidecar = sidecar
        if sidecar.output_node is None:
            raise TypeError("GraphModule.forward expected sidecar.output_node to be set")
        return sidecar.output_node

    def invoke(self, state: Any, **kwargs: Any) -> Any:
        """Invoke the underlying runtime graph and preserve the latest sidecar."""
        result, sidecar = self.adapter.invoke_runtime(state, **kwargs)
        self._last_sidecar = sidecar
        return result

    def parameters(self):
        """Expose the adapter's trainable parameters."""
        return self.adapter.parameters()

    def __getstate__(self):
        """Clear transient sidecar state before serialization."""
        state = self.__dict__.copy()
        state["_last_sidecar"] = None
        return state
