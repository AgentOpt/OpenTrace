from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from opentelemetry import trace as oteltrace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

logger = logging.getLogger(__name__)


class LLMCallError(Exception):
    """Raised when the underlying LLM provider returns a non-success response."""

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class InMemorySpanExporter(SpanExporter):
    """In-memory span exporter used by LangGraph + OTEL demos."""

    def __init__(self) -> None:
        self._finished_spans: List[ReadableSpan] = []

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        self._finished_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._finished_spans.clear()

    def get_finished_spans(self) -> List[ReadableSpan]:
        return list(self._finished_spans)

    def clear(self) -> None:
        self._finished_spans.clear()


def init_otel_runtime(
    service_name: str = "trace-langgraph-demo",
) -> Tuple[oteltrace.Tracer, InMemorySpanExporter]:
    """
    Initialize a TracerProvider + in-memory exporter for demos.

    Returns
    -------
    (tracer, exporter)
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Best effort: set as global provider if not already set; even if another
    # provider is active, we still return a tracer bound to this provider so
    # spans flow to the passed exporter.
    try:
        oteltrace.set_tracer_provider(provider)
    except Exception:
        pass

    tracer = provider.get_tracer(service_name)
    return tracer, exporter


def flush_otlp(
    exporter: InMemorySpanExporter,
    scope_name: str = "demo",
    *,
    clear: bool = True,
) -> Dict[str, Any]:
    """
    Convert exported spans into a minimal OTLP JSON payload.

    Parameters
    ----------
    exporter : InMemorySpanExporter
        The in-memory exporter holding collected spans.
    scope_name : str
        Scope name for the OTLP payload.
    clear : bool
        If *True* (default), clear the exporter after flushing.
        If *False*, spans remain in the exporter (peek mode).

    This is compatible with trace/io/otel_adapter.py::otlp_traces_to_trace_json.
    """

    spans = exporter.get_finished_spans()

    def hex_id(x: int, n: int) -> str:
        return f"{x:0{2*n}x}"

    otlp_spans: List[Dict[str, Any]] = []
    for s in spans:
        attributes = getattr(s, "attributes", {}) or {}
        attrs = [
            {"key": k, "value": {"stringValue": str(v)}}
            for k, v in attributes.items()
        ]
        kind = getattr(s, "kind", 1)
        if hasattr(kind, "value"):
            kind = kind.value

        otlp_spans.append(
            {
                "traceId": hex_id(s.context.trace_id, 16),
                "spanId": hex_id(s.context.span_id, 8),
                "parentSpanId": hex_id(s.parent.span_id, 8)
                if getattr(s, "parent", None)
                else "",
                "name": getattr(s, "name", ""),
                "kind": {
                    0: "UNSPECIFIED",
                    1: "INTERNAL",
                    2: "SERVER",
                    3: "CLIENT",
                    4: "PRODUCER",
                    5: "CONSUMER",
                }.get(kind, "INTERNAL"),
                "startTimeUnixNano": int(
                    getattr(s, "start_time", None) or time.time_ns()
                ),
                "endTimeUnixNano": int(
                    getattr(s, "end_time", None) or time.time_ns()
                ),
                "attributes": attrs,
            }
        )

    if clear:
        exporter.clear()

    return {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "scope": {"name": scope_name},
                        "spans": otlp_spans,
                    }
                ],
            }
        ]
    }


class TracingLLM:
    """
    Design-3+ wrapper around an LLM client with dual semantic conventions.

    Responsibilities
    ----------------
    * Create an OTEL **parent** span per LLM node (``span_name``) carrying
      ``param.*`` and ``inputs.*`` attributes (Trace-compatible).
    * Optionally create a **child** span with ``gen_ai.*`` attributes
      (Agent Lightning-compatible) marked with ``trace.temporal_ignore``
      so it does not break TGJ temporal chaining.
    * Emit trainable code parameters via ``emit_code_param`` when provided.
    * **Raise ``LLMCallError``** if the provider returns an error instead of
      silently converting it to assistant content (A1).

    Parameters
    ----------
    llm : Any
        Underlying LLM client (OpenAI-compatible interface).
    tracer : oteltrace.Tracer
        OTEL tracer for span creation.
    trainable_keys : Iterable[str] or None
        Keys whose prompts are trainable.  ``None`` means **all trainable**.
        Empty string ``""`` in the set also matches all.
    emit_code_param : callable, optional
        ``(span, key, fn) -> None``.
    provider_name : str
        Provider name for ``gen_ai.provider.name`` attribute.
        Should match the actual provider (e.g. ``"openrouter"``).
    llm_span_name : str
        Name for child LLM spans (e.g. ``"openai.chat.completion"``).
    emit_llm_child_span : bool
        If *True*, emit Agent Lightning-compatible child spans.
    """

    def __init__(
        self,
        llm: Any,
        tracer: oteltrace.Tracer,
        *,
        trainable_keys: Optional[Iterable[str]] = None,
        emit_code_param: Optional[Any] = None,
        # -- dual semconv additions --
        provider_name: str = "openai",
        llm_span_name: str = "openai.chat.completion",
        emit_llm_child_span: bool = True,
    ) -> None:
        self.llm = llm
        self.tracer = tracer
        # None -> all trainable; explicit set otherwise
        self._trainable_keys_all = trainable_keys is None
        self.trainable_keys = set(trainable_keys) if trainable_keys is not None else set()
        self.emit_code_param = emit_code_param
        self.provider_name = provider_name
        self.llm_span_name = llm_span_name
        self.emit_llm_child_span = emit_llm_child_span

    # ---- helpers ---------------------------------------------------------

    def _is_trainable(self, optimizable_key: Optional[str]) -> bool:
        if optimizable_key is None:
            return False
        if self._trainable_keys_all:
            return True
        if "" in self.trainable_keys:
            return True
        return optimizable_key in self.trainable_keys

    def _record_llm_call(
        self,
        sp,
        *,
        template_name: Optional[str],
        template: Optional[str],
        optimizable_key: Optional[str],
        code_key: Optional[str],
        code_fn: Any,
        user_query: Optional[str],
        prompt: str,
        extra_inputs: Optional[Dict[str, str]] = None,
    ) -> None:
        if template_name and template is not None:
            sp.set_attribute(f"param.{template_name}", template)
            sp.set_attribute(
                f"param.{template_name}.trainable",
                self._is_trainable(optimizable_key),
            )
        if code_key and code_fn is not None and self.emit_code_param:
            self.emit_code_param(sp, code_key, code_fn)

        sp.set_attribute("gen_ai.model", getattr(self.llm, "model", "llm"))
        sp.set_attribute("inputs.gen_ai.prompt", prompt)
        if user_query is not None:
            sp.set_attribute("inputs.user_query", user_query)
        for k, v in (extra_inputs or {}).items():
            sp.set_attribute(f"inputs.{k}", v)

    @staticmethod
    def _validate_content(content: Optional[str]) -> str:
        """Validate LLM response content.  Raise on empty or error markers."""
        if content is None:
            raise LLMCallError("LLM returned None content")
        if not content.strip():
            raise LLMCallError("LLM returned empty content")
        # Detect error strings that were smuggled as content (A1)
        if content.strip().startswith("[ERROR]"):
            raise LLMCallError(
                f"LLM provider returned an error: {content.strip()}"
            )
        return content

    # ---- public API ------------------------------------------------------

    def node_call(
        self,
        *,
        span_name: str,
        template_name: Optional[str] = None,
        template: Optional[str] = None,
        optimizable_key: Optional[str] = None,
        code_key: Optional[str] = None,
        code_fn: Any = None,
        user_query: Optional[str] = None,
        extra_inputs: Optional[Dict[str, str]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **llm_kwargs: Any,
    ) -> str:
        """
        Invoke the wrapped LLM under an OTEL span.

        Creates a **parent** span with ``param.*`` / ``inputs.*`` (Trace-
        compatible) and optionally a **child** span with ``gen_ai.*``
        attributes (Agent Lightning-compatible).  The child span is tagged
        ``trace.temporal_ignore=true`` so it does not break TGJ chaining.

        Raises
        ------
        LLMCallError
            If the provider call fails or returns empty/error content.
        """
        with self.tracer.start_as_current_span(span_name) as sp:
            prompt = ""
            if messages:
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    prompt = user_msgs[-1].get("content", "") or ""
                else:
                    prompt = messages[-1].get("content", "") or ""

            self._record_llm_call(
                sp,
                template_name=template_name,
                template=template,
                optimizable_key=optimizable_key,
                code_key=code_key,
                code_fn=code_fn,
                user_query=user_query,
                prompt=prompt,
                extra_inputs=extra_inputs or {},
            )

            # -- invoke LLM, optionally under a child span --
            try:
                if self.emit_llm_child_span:
                    with self.tracer.start_as_current_span(self.llm_span_name) as llm_sp:
                        # Tag child span so TGJ adapter skips temporal chaining
                        llm_sp.set_attribute("trace.temporal_ignore", "true")
                        llm_sp.set_attribute("gen_ai.operation.name", "chat")
                        llm_sp.set_attribute("gen_ai.provider.name", self.provider_name)
                        llm_sp.set_attribute(
                            "gen_ai.request.model",
                            getattr(self.llm, "model", "llm"),
                        )

                        resp = self.llm(messages=messages, **llm_kwargs)
                        content = resp.choices[0].message.content
                        content = self._validate_content(content)

                        llm_sp.set_attribute(
                            "gen_ai.output.preview", (content or "")[:500]
                        )
                else:
                    resp = self.llm(messages=messages, **llm_kwargs)
                    content = resp.choices[0].message.content
                    content = self._validate_content(content)
            except LLMCallError:
                # Record the error on the span and re-raise
                sp.set_attribute("error", "true")
                sp.set_attribute("error.type", "LLMCallError")
                raise
            except Exception as exc:
                # Unexpected provider error — record and raise as LLMCallError
                sp.set_attribute("error", "true")
                sp.set_attribute("error.type", type(exc).__name__)
                raise LLMCallError(
                    f"LLM provider call failed: {exc}"
                ) from exc

            return content


DEFAULT_EVAL_METRIC_KEYS: Mapping[str, str] = {
    "answer_relevance": "eval.answer_relevance",
    "groundedness": "eval.groundedness",
    "plan_quality": "eval.plan_quality",
}


def _attrs_to_dict(attrs: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for a in attrs or []:
        key = a.get("key")
        val = a.get("value", {})
        if key is None:
            continue
        if isinstance(val, dict) and "stringValue" in val:
            out[key] = val["stringValue"]
        else:
            out[key] = str(val)
    return out


def extract_eval_metrics_from_otlp(
    otlp: Dict[str, Any],
    *,
    evaluator_span_name: str = "evaluator",
    score_key: str = "eval.score",
    metric_keys: Optional[Mapping[str, str]] = None,
    default_score: float = 0.5,
    default_metric: float = 0.5,
) -> Tuple[float, Dict[str, float], str]:
    """
    Extract evaluation score + metrics + reasons from an OTLP payload.
    """
    metric_keys = metric_keys or DEFAULT_EVAL_METRIC_KEYS
    metrics: Dict[str, float] = {}
    reasons = ""
    score = default_score

    found = False
    for rs in otlp.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                if sp.get("name") != evaluator_span_name:
                    continue
                attrs = _attrs_to_dict(sp.get("attributes", []))
                raw_score = attrs.get(score_key)
                if raw_score is not None:
                    try:
                        score = float(raw_score)
                    except ValueError:
                        score = default_score
                reasons = attrs.get("eval.reasons", "") or ""

                for friendly, attr_key in metric_keys.items():
                    raw = attrs.get(attr_key)
                    if raw is None:
                        continue
                    try:
                        metrics[friendly] = float(raw)
                    except ValueError:
                        metrics[friendly] = default_metric

                found = True
                break
            if found:
                break
        if found:
            break

    if not metrics and metric_keys:
        metrics = {k: default_metric for k in metric_keys.keys()}

    return score, metrics, reasons
