"""
opto.trace.io.otel_semconv
==========================

Semantic convention helpers for emitting OTEL spans compatible with both
the Trace TGJ format and Agent Lightning ``gen_ai.*`` conventions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from opentelemetry import trace as oteltrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Span attribute helpers
# ---------------------------------------------------------------------------

def set_span_attributes(span: oteltrace.Span, attrs: Dict[str, Any]) -> None:
    """Set multiple span attributes at once.

    * ``dict`` / ``list`` values are serialized to JSON strings.
    * ``None`` values are silently skipped.
    """
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        span.set_attribute(key, value)


def record_genai_chat(
    span: oteltrace.Span,
    *,
    provider: str,
    model: str,
    input_messages: Optional[List[Dict[str, Any]]] = None,
    output_text: Optional[str] = None,
    request_type_compat: str = "chat.completion",
) -> None:
    """Record OTEL GenAI semantic convention attributes on *span*.

    Emits
    -----
    * ``gen_ai.operation.name``
    * ``gen_ai.provider.name``
    * ``gen_ai.request.model``
    * ``gen_ai.input.messages`` (JSON)
    * ``gen_ai.output.messages`` (JSON)
    """
    span.set_attribute("gen_ai.operation.name", request_type_compat)
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", model)
    if input_messages is not None:
        span.set_attribute(
            "gen_ai.input.messages",
            json.dumps(input_messages, default=str),
        )
    if output_text is not None:
        span.set_attribute(
            "gen_ai.output.messages",
            json.dumps([{"role": "assistant", "content": output_text}], default=str),
        )


# ---------------------------------------------------------------------------
# Reward / annotation helpers
# ---------------------------------------------------------------------------

def emit_reward(
    session: Any,  # TelemetrySession or anything with a .tracer property
    *,
    value: float,
    name: str = "final_score",
    index: int = 0,
    span_name: str = "agentlightning.annotation",
    extra_attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a reward span compatible with Agent Lightning semconv.

    Creates a child span with:
    * ``agentlightning.reward.<i>.name``
    * ``agentlightning.reward.<i>.value``
    * ``trace.temporal_ignore = true``
    """
    tracer = session.tracer if hasattr(session, "tracer") else session
    with tracer.start_as_current_span(span_name) as sp:
        sp.set_attribute("trace.temporal_ignore", "true")
        sp.set_attribute(f"agentlightning.reward.{index}.name", name)
        sp.set_attribute(f"agentlightning.reward.{index}.value", str(value))
        if extra_attributes:
            set_span_attributes(sp, extra_attributes)


# Backward-compat alias
emit_agentlightning_reward = emit_reward


def emit_trace(
    session: Any,
    *,
    name: str,
    attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a lightweight OTEL span for arbitrary debug / optimization signals.

    Parameters
    ----------
    session
        A ``TelemetrySession`` (or anything with a ``.tracer`` attribute).
    name : str
        Span name.
    attrs : dict, optional
        Attributes to attach.
    """
    tracer = session.tracer if hasattr(session, "tracer") else session
    with tracer.start_as_current_span(name) as sp:
        if attrs:
            set_span_attributes(sp, attrs)
