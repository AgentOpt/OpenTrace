#!/usr/bin/env python3
"""
Live LangGraph optimization comparison across Trace / OTEL / sys.monitoring.

This script benchmarks optimization over 5 iterations using a real
OpenRouter-backed LLM when OPENROUTER_API_KEY is available, then converts
every backend's captured artifacts to a shared TGJ view so the notebook can
show the same semantic graph logic across configurations.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, START, END
from opto.trace import node
from opto.trace.io import (
    EvalResult,
    instrument_graph,
    make_dict_binding,
    optimize_graph,
    otlp_traces_to_trace_json,
    LLMCallError,
)
from opto.trace.io.sysmonitoring import sysmon_profile_to_tgj
from opto.trace.io.tgj_export import export_subgraph_to_tgj
from opto.trace.io.tgj_ingest import ingest_tgj
from opto.trace.nodes import MessageNode, ParameterNode
from opto.utils.llm import LLM

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


HAS_SYSMON = hasattr(sys, "monitoring")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
ITERATIONS = int(os.environ.get("COMPARE_OBSERVERS_ITERATIONS", "5"))
QUERIES = [
    "What is CRISPR?",
    "How does CRISPR enable gene editing?",
]
_QUERY_LIMIT = os.environ.get("COMPARE_OBSERVERS_QUERY_LIMIT")
if _QUERY_LIMIT:
    QUERIES = QUERIES[: max(1, int(_QUERY_LIMIT))]
_CASE_FILTER = tuple(
    part.strip()
    for part in os.environ.get("COMPARE_OBSERVERS_CASES", "").split(",")
    if part.strip()
)
SYNTH_UPDATE_SCHEDULE = [
    {
        "synth_prompt": (
            "Answer directly in the first sentence. "
            "Then add two short titled sections with concrete details: {query}\nPlan: {plan}"
        )
    },
    {
        "synth_prompt": (
            "Answer directly in the first sentence. "
            "Then add three short titled sections with concrete mechanisms, examples, "
            "and caveats when useful. Keep it factual and concise: {query}\nPlan: {plan}"
        )
    },
]
PLANNER_SYSTEM_PROMPT = "You are a careful planner."
SYNTH_SYSTEM_PROMPT = "You are a careful scientific assistant."
DEFAULT_TEMPLATES = {
    "planner_prompt": "Create a short plan for: {query}",
    "synth_prompt": "Answer briefly and factually: {query}\nPlan: {plan}",
}
PROMPT_CONSUMERS = {
    "planner_prompt": ["planner_node"],
    "synth_prompt": ["synth_node"],
}
SEMANTIC_NAMES = ("planner_node", "synth_node")
STOPWORDS = {
    "about",
    "add",
    "also",
    "answer",
    "briefly",
    "carefully",
    "concise",
    "does",
    "directly",
    "exactly",
    "factually",
    "from",
    "give",
    "have",
    "into",
    "keep",
    "plan",
    "short",
    "start",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "this",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}


def _raw(value: Any) -> Any:
    return getattr(value, "data", value)


def _truncate(value: Any, limit: int = 140) -> str:
    text = str(_raw(value))
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _base_name(value: Any) -> str:
    name = str(getattr(value, "name", getattr(value, "py_name", "")))
    return name.split("/")[-1].split(":")[0]


def _semantic_alias(name: str) -> str | None:
    suffix = name.split(".")[-1]
    if suffix in SEMANTIC_NAMES:
        return suffix
    return None


def _str_map(values: Mapping[str, Any]) -> Dict[str, str]:
    return {key: str(_raw(value)) for key, value in values.items()}


def _node_records(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [record for _node_id, record in _node_items(doc)]


def _node_items(doc: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return TGJ nodes as ``(node_id, record)`` pairs, preserving dict keys.

    OTEL-derived TGJ docs store node identities in the ``nodes`` mapping keys,
    while some other carriers inline ``id`` directly in each record. The
    notebook renderer needs the stable node id in both cases.
    """
    raw_nodes = doc.get("nodes") or {}
    if isinstance(raw_nodes, dict):
        items = []
        for node_id, record in raw_nodes.items():
            enriched = dict(record)
            enriched.setdefault("id", str(node_id))
            items.append((str(node_id), enriched))
        return items

    items = []
    for idx, record in enumerate(raw_nodes):
        enriched = dict(record)
        node_id = str(enriched.get("id") or f"node_{idx}")
        enriched.setdefault("id", node_id)
        items.append((node_id, enriched))
    return items


def _spans_from_otlp(otlp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten all spans across every OTLP resource/scope block."""
    spans: List[Dict[str, Any]] = []
    for resource in otlp.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            spans.extend(scope.get("spans", []))
    return spans


def _merge_tgj_docs(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple TGJ docs into one node-preserving document."""
    if not docs:
        return {"version": "trace-json/1.0+otel", "nodes": {}, "context": {}}

    merged = dict(docs[0])
    merged_nodes: Dict[str, Dict[str, Any]] = {}
    for doc in docs:
        for node_id, record in _node_items(doc):
            merged_nodes[node_id] = record
    merged["nodes"] = merged_nodes
    return merged


def _edge_count(doc: Dict[str, Any]) -> int:
    records = _node_items(doc)
    known_ids = {node_id for node_id, _record in records}
    count = 0
    for _child_id, record in records:
        for ref in (record.get("inputs") or {}).values():
            parent_id = ref.get("ref") if isinstance(ref, dict) else ref
            if parent_id is not None and str(parent_id) in known_ids:
                count += 1
    return count


def _unique_nodes(nodes: Dict[str, Any], cls: type) -> List[Any]:
    return list(
        {
            id(obj): obj
            for obj in nodes.values()
            if isinstance(obj, cls)
        }.values()
    )


def _terms(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 4 and token not in STOPWORDS
    ]


def _coverage(needles: List[str], haystack: str) -> float:
    keys = list(dict.fromkeys(needles))
    if not keys:
        return 1.0
    body = haystack.lower()
    return sum(token in body for token in keys) / len(keys)


def _lead_text(answer: str) -> str:
    first_line = answer.splitlines()[0] if answer.splitlines() else answer
    first_sentence = re.split(r"(?<=[.!?])\s+", answer, maxsplit=1)[0]
    return (first_sentence if len(first_sentence) >= len(first_line) else first_line)[:220]


def _structure_score(answer: str) -> float:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    headings = sum(line.startswith("#") for line in lines)
    bullets = sum(
        line.startswith(("-", "*")) or re.match(r"^\d+[.)]\s", line) is not None
        for line in lines
    )
    return min(
        1.0,
        0.35 * min(headings, 3) / 3
        + 0.45 * min(bullets, 4) / 4
        + 0.20 * min(len(lines), 12) / 12,
    )


def _length_score(answer: str) -> float:
    return min(max(len(answer) - 120, 0) / 720.0, 1.0)


def _directness_score(query: str, answer: str) -> float:
    lead = _lead_text(answer).strip()
    if not lead or lead.startswith(("#", "-", "*")):
        return 0.0
    return 0.5 * _coverage(_terms(query), lead) + 0.5 * float(len(lead) >= 60)


def render_template(template: str, **variables: Any) -> str:
    return template.format(**_str_map(variables))


def _extract_response_text(response: Any) -> str:
    """Return assistant text from OpenAI-compatible demo responses."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMCallError("LLM response missing choices/content")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
                elif isinstance(text, Mapping):
                    value = text.get("value")
                    if isinstance(value, str) and value.strip():
                        parts.append(value)
        joined = "\n".join(parts).strip()
        if joined:
            return joined
    raise LLMCallError("LLM returned None content")


def call_chat_text(
    llm,
    *,
    system_prompt: str,
    user_prompt: str,
    **kwargs: Any,
) -> str:
    response = llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=kwargs.pop("temperature", 0),
        **kwargs,
    )
    return _extract_response_text(response)


def _has_response_content(response: Any) -> bool:
    """Best-effort guard for empty provider payloads in demo live mode."""
    try:
        return bool(_extract_response_text(response).strip())
    except Exception:
        return False


def _is_retryable_provider_error(exc: Exception) -> bool:
    """Detect transient OpenRouter/OpenAI client failures worth retrying."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "500",
            "502",
            "503",
            "504",
            "rate limit",
            "temporarily",
            "timeout",
            "connection",
            "none content",
            "missing choices",
        )
    )


def summarize_tgj(doc: Dict[str, Any]) -> Dict[str, Any]:
    nodes = ingest_tgj(doc)
    message_nodes = _unique_nodes(nodes, MessageNode)
    param_nodes = _unique_nodes(nodes, ParameterNode)
    message_names = sorted(_base_name(obj) for obj in message_nodes if _base_name(obj))
    param_names = sorted(_base_name(obj) for obj in param_nodes if _base_name(obj))
    semantic_messages = sorted(
        {
            alias
            for name in message_names
            if (alias := _semantic_alias(name)) is not None
        }
    )
    param_values = {
        name: _truncate(obj.data, 220)
        for obj in param_nodes
        if (name := _base_name(obj)).endswith("_prompt")
    }
    return {
        "node_count": len(_node_records(doc)),
        "message_names": message_names,
        "semantic_messages": semantic_messages,
        "param_names": param_names,
        "param_values": param_values,
    }


def _make_trace_view(
    trace_nodes: List[Any],
    *,
    config: str,
    origin: str,
) -> Dict[str, Any]:
    doc = export_subgraph_to_tgj(
        trace_nodes,
        run_id="compare",
        agent_id=config,
        graph_id="trace",
        scope=f"{config}/{origin}",
    )
    return {
        "carrier": "trace",
        "origin": origin,
        "doc": doc,
        "summary": summarize_tgj(doc),
    }


def _make_otel_view(
    otlp: Dict[str, Any],
    *,
    config: str,
    origin: str,
) -> Dict[str, Any]:
    spans = _spans_from_otlp(otlp)
    param_keys = sorted(
        {
            attr["key"]
            for span in spans
            for attr in span.get("attributes", [])
            if str(attr.get("key", "")).startswith("param.")
        }
    )
    docs = otlp_traces_to_trace_json(
        otlp,
        agent_id_hint=config,
        use_temporal_hierarchy=True,
    )
    doc = _merge_tgj_docs(list(docs))
    summary = summarize_tgj(doc)
    summary["span_count"] = len(spans)
    summary["span_names"] = [span.get("name") for span in spans]
    summary["param_keys"] = param_keys
    return {
        "carrier": "otel",
        "origin": origin,
        "doc": doc,
        "summary": summary,
    }


def _make_sysmon_view(
    profile_doc: Dict[str, Any],
    *,
    config: str,
    origin: str,
) -> Dict[str, Any]:
    doc = sysmon_profile_to_tgj(
        profile_doc,
        run_id="compare",
        graph_id=config,
        scope=f"{config}/{origin}",
    )
    summary = summarize_tgj(doc)
    summary["event_count"] = len(profile_doc.get("events", []))
    return {
        "carrier": "sysmon",
        "origin": origin,
        "doc": doc,
        "summary": summary,
    }


def tgj_to_digraph(doc: Dict[str, Any], *, title: str):
    try:
        from graphviz import Digraph
    except Exception:
        return None

    records = _node_items(doc)
    known_ids = {node_id for node_id, _record in records}
    dot_ids = {node_id: f"node_{idx}" for idx, (node_id, _record) in enumerate(records)}
    graph = Digraph(comment=title)
    graph.attr(rankdir="LR")

    for node_id, record in records:
        kind = str(record.get("kind", "value")).lower()
        name = str(record.get("name", node_id))
        if kind == "parameter":
            preview = record.get("value", record.get("data", ""))
            fill = "khaki1"
            kind_label = "parameter"
        elif kind in {"message", "msg"}:
            preview = (record.get("output") or {}).get("value", "")
            if preview in (None, ""):
                preview = record.get("value", record.get("data", ""))
            fill = "lightblue"
            kind_label = "message"
        elif kind == "exception":
            preview = (record.get("error") or {}).get("message", "")
            fill = "mistyrose"
            kind_label = "exception"
        else:
            preview = record.get("value", record.get("data", ""))
            fill = "white"
            kind_label = kind
        label = f"{name}\\n[{kind_label}]"
        if preview not in (None, ""):
            label += f"\\n{_truncate(preview, 80)}"
        graph.node(
            dot_ids[node_id],
            label=label,
            shape="box",
            style="rounded,filled",
            fillcolor=fill,
            tooltip=node_id,
        )

    for child_id, record in records:
        for ref in (record.get("inputs") or {}).values():
            parent_id = ref.get("ref") if isinstance(ref, dict) else ref
            if parent_id is not None and str(parent_id) in known_ids:
                graph.edge(dot_ids[str(parent_id)], dot_ids[child_id])

    return graph


class DictUpdateOptimizer:
    def __init__(self, update_spec: Dict[str, Any] | List[Dict[str, Any]]):
        if isinstance(update_spec, list):
            self.update_schedule = [dict(update) for update in update_spec]
        else:
            self.update_schedule = [dict(update_spec)]
        self.calls = 0

    def zero_feedback(self):
        return None

    def backward(self, *_args, **_kwargs):
        return None

    def step(self):
        if self.calls < len(self.update_schedule):
            update = dict(self.update_schedule[self.calls])
            self.calls += 1
            return update
        self.calls += 1
        return {}


def _optimizer_model_name() -> str:
    model = os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL or "google/gemini-3-flash-preview")
    if model == "gemini-3-flash-preview":
        model = "google/gemini-3-flash-preview"
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"
    return model


def _comparison_optimizer_kwargs() -> Dict[str, Any] | None:
    if not os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY):
        return None
    return {"llm": LLM(backend="LiteLLM", model=_optimizer_model_name())}


def make_live_llm():
    if not OPENROUTER_API_KEY or OpenAI is None:
        return None

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    def _llm(messages=None, **kwargs):
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=messages or [],
                    max_tokens=kwargs.get("max_tokens", 220),
                    temperature=kwargs.get("temperature", 0),
                )
                if _has_response_content(response):
                    return response
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return response
            except Exception as exc:
                if _is_retryable_provider_error(exc) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

    _llm.model = OPENROUTER_MODEL
    return _llm


def eval_fn(payload: Dict[str, Any]) -> EvalResult:
    query = str(payload.get("query", ""))
    answer = str(_raw(payload.get("answer", ""))).strip()
    if answer.startswith("[ERROR]") or not answer:
        return EvalResult(score=0.0, feedback="LLM failure/empty answer")

    coverage = _coverage(_terms(query), answer)
    directness = _directness_score(query, answer)
    structure = _structure_score(answer)
    length = _length_score(answer)
    score = 0.08 + 0.30 * coverage + 0.26 * directness + 0.20 * structure + 0.16 * length
    return EvalResult(
        score=round(min(score, 0.95), 4),
        feedback=(
            f"coverage={coverage:.2f}, directness={directness:.2f}, "
            f"structure={structure:.2f}, length={length:.2f}"
        ),
    )


def build_semantic_graph(planner_fn, synth_fn):
    graph = StateGraph(dict)
    graph.add_node("planner", planner_fn)
    graph.add_node("synth", synth_fn)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "synth")
    graph.add_edge("synth", END)
    return graph.compile()


def make_semantic_nodes(
    *,
    planner_call: Callable[[str], str],
    synth_call: Callable[[str, str], Any],
    wrap_final_answer: Callable[[Any], Any] | None = None,
):
    def planner_node(state):
        query = str(_raw(state["query"]))
        return {"query": query, "plan": planner_call(query)}

    def synth_node(state):
        query = str(_raw(state["query"]))
        plan = str(_raw(state["plan"]))
        answer = synth_call(query, plan)
        if wrap_final_answer is not None:
            answer = wrap_final_answer(answer)
        return {"final_answer": answer}

    return planner_node, synth_node


def make_trace_case(llm, observe_with: Tuple[str, ...] = ()):
    planner_prompt = node(
        DEFAULT_TEMPLATES["planner_prompt"],
        trainable=True,
        name="planner_prompt",
    )
    synth_prompt = node(
        DEFAULT_TEMPLATES["synth_prompt"],
        trainable=True,
        name="synth_prompt",
    )
    scope: Dict[str, Any] = {}

    def planner_node(state):
        query = str(_raw(state["query"]))
        prompt = render_template(planner_prompt.data, query=query)
        plan = call_chat_text(
            llm,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0,
        )
        return {"query": query, "plan": MessageNode(plan, inputs={"prompt": planner_prompt}, description="[llm] planner", name="planner_answer")}

    def synth_node(state):
        query = str(_raw(state["query"]))
        plan_node = state["plan"]; plan = str(_raw(plan_node))
        prompt = render_template(synth_prompt.data, query=query, plan=plan)
        answer = call_chat_text(
            llm,
            system_prompt=SYNTH_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0,
        )
        return {"final_answer": MessageNode(answer, inputs={"prompt": synth_prompt, "plan": node(plan_node)}, description="[llm] synth", name="final_answer_node")}

    scope.update(
        {
            "llm": llm,
            "planner_prompt": planner_prompt,
            "synth_prompt": synth_prompt,
            "render_template": render_template,
            "call_chat_text": call_chat_text,
            "PLANNER_SYSTEM_PROMPT": PLANNER_SYSTEM_PROMPT,
            "SYNTH_SYSTEM_PROMPT": SYNTH_SYSTEM_PROMPT,
            "node": node,
            "_raw": _raw,
            "planner_node": planner_node,
            "synth_node": synth_node,
        }
    )

    def build_graph():
        return build_semantic_graph(scope["planner_node"], scope["synth_node"])

    instrumented = instrument_graph(
        backend="trace",
        observe_with=observe_with,
        graph_factory=build_graph,
        scope=scope,
        graph_agents_functions=list(SEMANTIC_NAMES),
        graph_prompts_list=[planner_prompt, synth_prompt],
        binding_consumers=PROMPT_CONSUMERS,
        train_graph_agents_functions=False,
        output_key="final_answer",
    )
    optimizer = None if os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY) else DictUpdateOptimizer(SYNTH_UPDATE_SCHEDULE)
    return instrumented, optimizer, lambda: synth_prompt.data


def make_otel_case(llm, observe_with: Tuple[str, ...] = ()):
    instrumented = instrument_graph(
        graph=None,
        backend="otel",
        observe_with=observe_with,
        graph_agents_functions=list(SEMANTIC_NAMES),
        binding_consumers=PROMPT_CONSUMERS,
        llm=llm,
        initial_templates=dict(DEFAULT_TEMPLATES),
        output_key="final_answer",
    )
    instrumented.backend = "otel"
    templates = instrumented.templates
    tracing_llm = instrumented.tracing_llm

    def planner_call(query: str) -> str:
        return tracing_llm.template_prompt_call(
            span_name="planner_node",
            template_name="planner_prompt",
            template=templates["planner_prompt"],
            variables={"query": query},
            system_prompt=PLANNER_SYSTEM_PROMPT,
            optimizable_key="planner",
            temperature=0,
        )

    def synth_call(query: str, plan: str) -> str:
        return tracing_llm.template_prompt_call(
            span_name="synth_node",
            template_name="synth_prompt",
            template=templates["synth_prompt"],
            variables={"query": query, "plan": plan},
            system_prompt=SYNTH_SYSTEM_PROMPT,
            optimizable_key="synth",
            temperature=0,
        )

    instrumented.graph = build_semantic_graph(
        *make_semantic_nodes(
            planner_call=planner_call,
            synth_call=synth_call,
        )
    )
    optimizer = None if os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY) else DictUpdateOptimizer(SYNTH_UPDATE_SCHEDULE)
    return instrumented, optimizer, lambda: instrumented.templates["synth_prompt"]


def make_sysmon_case(llm):
    templates = dict(DEFAULT_TEMPLATES)
    bindings = {k: make_dict_binding(templates, k, kind="prompt") for k in templates}

    def planner_call(query: str) -> str:
        prompt = render_template(templates["planner_prompt"], query=query)
        return call_chat_text(
            llm,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0,
        )

    def synth_call(query: str, plan: str) -> str:
        prompt = render_template(templates["synth_prompt"], query=query, plan=plan)
        return call_chat_text(
            llm,
            system_prompt=SYNTH_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0,
        )

    instrumented = instrument_graph(
        graph=build_semantic_graph(
            *make_semantic_nodes(
                planner_call=planner_call,
                synth_call=synth_call,
            )
        ),
        backend="sysmon",
        bindings=bindings,
        graph_agents_functions=list(SEMANTIC_NAMES),
        binding_consumers=PROMPT_CONSUMERS,
        output_key="final_answer",
    )
    optimizer = None if os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY) else DictUpdateOptimizer(SYNTH_UPDATE_SCHEDULE)
    return instrumented, optimizer, lambda: templates["synth_prompt"]


def build_cases(llm):
    cases = [
        ("trace", lambda: make_trace_case(llm, ())),
        ("trace+otel", lambda: make_trace_case(llm, ("otel",))),
        ("otel", lambda: make_otel_case(llm, ())),
    ]
    if HAS_SYSMON:
        cases.extend(
            [
                ("trace+sysmon", lambda: make_trace_case(llm, ("sysmon",))),
                (
                    "trace+otel+sysmon",
                    lambda: make_trace_case(llm, ("otel", "sysmon")),
                ),
                ("otel+sysmon", lambda: make_otel_case(llm, ("sysmon",))),
                ("sysmon", lambda: make_sysmon_case(llm)),
            ]
        )
    if _CASE_FILTER:
        cases = [case for case in cases if case[0] in _CASE_FILTER]
    return cases


def run_case(name: str, builder):
    instrumented, optimizer, prompt_getter = builder()
    started_at = time.perf_counter()
    result = optimize_graph(
        instrumented,
        queries=QUERIES,
        iterations=ITERATIONS,
        optimizer=optimizer,
        optimizer_kwargs=_comparison_optimizer_kwargs(),
        eval_fn=eval_fn,
        output_key="final_answer",
    )
    runtime_s = time.perf_counter() - started_at

    probe = instrumented.invoke({"query": QUERIES[0]})
    if hasattr(probe, "data") and isinstance(probe.data, dict):
        answer_value = probe.data.get("final_answer", probe.data)
    elif isinstance(probe, dict):
        answer_value = probe.get("final_answer", probe)
    else:
        answer_value = probe
    answer_text = str(_raw(answer_value))
    views = []

    backend = getattr(instrumented, "backend", None)
    if backend == "trace":
        views.append(
            _make_trace_view(
                [probe, answer_value, *list(getattr(instrumented, "parameters", []))],
                config=name,
                origin="backend",
            )
        )
    elif backend == "otel":
        views.append(
            _make_otel_view(
                instrumented.session.flush_otlp(clear=True),
                config=name,
                origin="backend",
            )
        )
    elif backend == "sysmon":
        views.append(
            _make_sysmon_view(
                instrumented._last_profile_doc or {},
                config=name,
                origin="backend",
            )
        )

    for artifact in getattr(instrumented, "_last_observer_artifacts", []):
        if artifact.carrier == "otel":
            views.append(
                _make_otel_view(
                    artifact.raw,
                    config=name,
                    origin="observer",
                )
            )
        elif artifact.carrier == "sysmon":
            views.append(
                _make_sysmon_view(
                    artifact.profile_doc,
                    config=name,
                    origin="observer",
                )
            )

    final_prompt = prompt_getter()
    assert bool(final_prompt) if optimizer is None else final_prompt == SYNTH_UPDATE_SCHEDULE[-1]["synth_prompt"]
    tail_scores = result.score_history[max(2, result.best_iteration):]
    primary_summary = views[0]["summary"] if views else {}

    return {
        "config": name,
        "runtime_s": round(runtime_s, 3),
        "baseline_score": round(result.baseline_score, 3),
        "best_score": round(result.best_score, 3),
        "score_gain": round(result.best_score - result.baseline_score, 3),
        "best_iteration": result.best_iteration,
        "score_history": [round(x, 3) for x in result.score_history],
        "stability_std": round(
            statistics.pstdev(tail_scores) if len(tail_scores) > 1 else 0.0,
            3,
        ),
        "node_count": int(primary_summary.get("node_count", 0)),
        "edge_count": _edge_count(views[0]["doc"]) if views else 0,
        "best_updates": {(_base_name(key) or str(key).split("/")[-1].split(":")[0]): value for key, value in result.best_updates.items()},
        "final_synth_prompt": final_prompt,
        "final_answer": answer_text,
        "answer_preview": _truncate(answer_text, 180),
        "observers": [
            artifact.carrier
            for artifact in getattr(instrumented, "_last_observer_artifacts", [])
        ],
        "views": views,
    }


def live_skip_reason() -> str | None:
    if not OPENROUTER_API_KEY:
        return (
            "[SKIP] OPENROUTER_API_KEY is not set. "
            "This comparison stays live-only so notebook CI can skip cleanly."
        )
    if OpenAI is None:
        return "[SKIP] openai package is unavailable."
    return None


def run_live_comparison(*, echo_progress: bool = True) -> List[Dict[str, Any]]:
    reason = live_skip_reason()
    if reason is not None:
        if echo_progress:
            print(reason)
        return []

    llm = make_live_llm()
    context = nullcontext()
    sink = None
    if not echo_progress:
        sink = StringIO()
        context = redirect_stdout(sink)
    with context:
        rows = [run_case(name, builder) for name, builder in build_cases(llm)]
    return rows


def print_cli_report(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    print(f"\nOptimization comparison ({ITERATIONS} iterations)\n")
    print(
        "| config | runtime_s | baseline | best | gain | best_iteration | stability_std | "
        "node_count | edge_count | score_history | semantic_messages | params |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for row in rows:
        primary = row["views"][0]["summary"] if row["views"] else {}
        print(
            f"| {row['config']} | {row['runtime_s']:.3f} | {row['baseline_score']:.3f} "
            f"| {row['best_score']:.3f} | {row['score_gain']:.3f} | {row['best_iteration']} "
            f"| {row['stability_std']:.3f} | {row['node_count']} | {row['edge_count']} | {row['score_history']} "
            f"| {primary.get('semantic_messages', [])} | {primary.get('param_names', [])} |"
        )

    print("\nPer-configuration artifacts\n")
    for row in rows:
        print(f"## {row['config']}")
        print(f"runtime_s: {row['runtime_s']:.3f}")
        print(f"baseline_score: {row['baseline_score']:.3f}")
        print(f"best_score: {row['best_score']:.3f}")
        print(f"score_gain: {row['score_gain']:.3f}")
        print(f"stability_std: {row['stability_std']:.3f}")
        print(f"score_history: {row['score_history']}")
        print(f"best_updates: {row['best_updates']}")
        print(f"final_synth_prompt: {row['final_synth_prompt']}")
        print(f"final_answer: {row['answer_preview']}")
        for view in row["views"]:
            summary = view["summary"]
            extras = []
            if "span_count" in summary:
                extras.append(f"span_count={summary['span_count']}")
            if "event_count" in summary:
                extras.append(f"event_count={summary['event_count']}")
            extra_text = f" ({', '.join(extras)})" if extras else ""
            print(
                f"  - {view['origin']} {view['carrier']}{extra_text}: "
                f"messages={summary['message_names']} "
                f"params={summary['param_names']}"
            )
        print()


def display_notebook_report(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        from IPython.display import Markdown, display
    except Exception:
        print_cli_report(rows)
        return rows

    if not rows:
        display(Markdown(live_skip_reason() or "_No rows captured._"))
        return rows

    lines = [
        "| config | runtime_s | baseline | best | gain | best_iteration | stability_std | node_count | edge_count | score_history |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['runtime_s']:.3f} | {row['baseline_score']:.3f} "
            f"| {row['best_score']:.3f} | {row['score_gain']:.3f} | {row['best_iteration']} "
            f"| {row['stability_std']:.3f} | {row['node_count']} | {row['edge_count']} | {row['score_history']} |"
        )
    note = (
        "_Topology metrics remain useful even when score trajectories match, "
        "for example under the fixed offline prompt schedule._"
    )
    display(Markdown("## Optimization comparison\n\n" + note + "\n\n" + "\n".join(lines)))

    for row in rows:
        display(
            Markdown(
                "\n".join(
                    [
                        f"## {row['config']}",
                        f"- Runtime: `{row['runtime_s']:.3f}s`",
                        f"- Baseline score: `{row['baseline_score']:.3f}`",
                        f"- Best score: `{row['best_score']:.3f}`",
                        f"- Score gain: `{row['score_gain']:.3f}`",
                        f"- Best iteration: `{row['best_iteration']}`",
                        f"- Post-update stability std: `{row['stability_std']:.3f}`",
                        f"- Node count: `{row['node_count']}`",
                        f"- Edge count: `{row['edge_count']}`",
                        f"- Score history: `{row['score_history']}`",
                        f"- Best updates: `{list(row['best_updates'].keys())}`",
                        "",
                        "### Final synth prompt",
                        "```text",
                        str(row["final_synth_prompt"]),
                        "```",
                        "### Final answer",
                        "```text",
                        _truncate(row["final_answer"], 500),
                        "```",
                    ]
                )
            )
        )
        for view in row["views"]:
            summary = view["summary"]
            extra_lines = []
            if "span_count" in summary:
                extra_lines.append(f"- Span count: `{summary['span_count']}`")
                extra_lines.append(f"- Span names: `{summary['span_names']}`")
            if "event_count" in summary:
                extra_lines.append(f"- Event count: `{summary['event_count']}`")
            display(
                Markdown(
                    "\n".join(
                        [
                            f"### {view['origin']} {view['carrier']}",
                            f"- Semantic message names: `{summary['semantic_messages']}`",
                            f"- All message names: `{summary['message_names']}`",
                            f"- Parameter names: `{summary['param_names']}`",
                            *extra_lines,
                            "",
                            "```json",
                            json.dumps(summary["param_values"], indent=2),
                            "```",
                        ]
                    )
                )
            )
            graph = tgj_to_digraph(
                view["doc"],
                title=f"{row['config']} {view['origin']} {view['carrier']}",
            )
            if graph is not None:
                display(graph)

    return rows


def run_notebook_demo() -> List[Dict[str, Any]]:
    rows = run_live_comparison(echo_progress=False)
    return display_notebook_report(rows)


def main():
    print("\n" + "=" * 80)
    print("LangGraph live optimization comparison")
    print("=" * 80)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"sys.monitoring available: {HAS_SYSMON}")
    print(f"OPENROUTER_MODEL={OPENROUTER_MODEL}")

    rows = run_live_comparison(echo_progress=True)
    print_cli_report(rows)


if __name__ == "__main__":
    main()
