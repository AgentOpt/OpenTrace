#!/usr/bin/env python3
"""
Live LangGraph optimization comparison across Trace / OTEL / sys.monitoring.

This script intentionally benchmarks optimization over 5 iterations using
a real OpenRouter-backed LLM when OPENROUTER_API_KEY is available.

Compared configurations:
  - trace
  - trace + otel
  - trace + sysmon
  - trace + otel + sysmon
  - otel
  - otel + sysmon
  - sysmon

When OPENROUTER_API_KEY is not set, the script exits successfully after
printing a skip message. This keeps notebook CI deterministic while still
making the demo a true live benchmark for local/manual use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, START, END
from opto.trace import node
from opto.trace.nodes import MessageNode, ParameterNode
from opto.trace.io import (
    instrument_graph,
    optimize_graph,
    make_dict_binding,
    otlp_traces_to_trace_json,
)
from opto.trace.io.sysmonitoring import sysmon_profile_to_tgj
from opto.trace.io.tgj_ingest import ingest_tgj

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


HAS_SYSMON = hasattr(sys, "monitoring")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "gpt-4o-mini")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
ITERATIONS = 5
QUERIES = [
    "What is CRISPR?",
    "How does CRISPR enable gene editing?",
]
OPTIMIZED_SYNTH_PROMPT = (
    "Start the answer exactly with [BENCH_OK]. "
    "Then answer carefully: {query}\nPlan: {plan}"
)
PLANNER_SYSTEM_PROMPT = "You are a careful planner."
SYNTH_SYSTEM_PROMPT = "You are a careful scientific assistant."
DEFAULT_TEMPLATES = {
    "planner_prompt": "Create a short plan for: {query}",
    "synth_prompt": "Answer briefly and factually: {query}\nPlan: {plan}",
}


def _raw(value: Any) -> Any:
    return getattr(value, "data", value)


def _str_map(values: Mapping[str, Any]) -> Dict[str, str]:
    return {key: str(_raw(value)) for key, value in values.items()}


def render_template(template: str, **variables: Any) -> str:
    return template.format(**_str_map(variables))


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
    return response.choices[0].message.content


def _message_names(nodes: Dict[str, Any]):
    names = []
    seen = set()
    for obj in nodes.values():
        if isinstance(obj, MessageNode):
            nm = str(getattr(obj, "name", getattr(obj, "py_name", "")))
            base = nm.split("/")[-1].split(":")[0]
            if base not in seen:
                seen.add(base)
                names.append(base)
    return sorted(names)


class DictUpdateOptimizer:
    def __init__(self, update_dict: Dict[str, Any]):
        self.update_dict = dict(update_dict)
        self.calls = 0

    def zero_feedback(self):
        return None

    def backward(self, *_args, **_kwargs):
        return None

    def step(self):
        self.calls += 1
        if self.calls == 1:
            return dict(self.update_dict)
        return {}


class TraceMutatingOptimizer:
    def __init__(self, prompt_node, update_value: str, key: str):
        self.prompt_node = prompt_node
        self.update_value = update_value
        self.key = key
        self.calls = 0

    def zero_feedback(self):
        return None

    def backward(self, *_args, **_kwargs):
        return None

    def step(self):
        self.calls += 1
        if self.calls == 1:
            self.prompt_node._set(self.update_value)
            return {self.key: self.update_value}
        return {}


def make_live_llm():
    if not OPENROUTER_API_KEY or OpenAI is None:
        return None

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    def _llm(messages=None, **kwargs):
        return client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages or [],
            max_tokens=kwargs.get("max_tokens", 220),
            temperature=kwargs.get("temperature", 0),
        )

    _llm.model = OPENROUTER_MODEL
    return _llm


def eval_fn(payload: Dict[str, Any]) -> Dict[str, Any]:
    answer = str(_raw(payload.get("answer", ""))).strip()
    ok = answer.startswith("[BENCH_OK]")
    return {
        "score": 1.0 if ok else 0.0,
        "feedback": "Start the answer exactly with [BENCH_OK].",
    }


def summarize_otlp(otlp: Dict[str, Any]) -> Dict[str, Any]:
    spans = otlp.get("resourceSpans", [{}])[0].get("scopeSpans", [{}])[0].get("spans", [])
    param_keys = sorted(
        {
            a["key"]
            for s in spans
            for a in s.get("attributes", [])
            if str(a.get("key", "")).startswith("param.")
        }
    )
    docs = otlp_traces_to_trace_json(
        otlp,
        agent_id_hint="compare",
        use_temporal_hierarchy=True,
    )
    nodes = ingest_tgj(docs[0]) if docs else {}
    return {
        "span_count": len(spans),
        "span_names": [s.get("name") for s in spans],
        "param_keys": param_keys,
        "message_names": _message_names(nodes),
    }


def summarize_sysmon(profile_doc: Dict[str, Any]) -> Dict[str, Any]:
    tgj = sysmon_profile_to_tgj(profile_doc, run_id="compare", graph_id="demo", scope="compare/0")
    nodes = ingest_tgj(tgj)
    param_names = sorted(
        {
            str(getattr(obj, "name", getattr(obj, "py_name", ""))).split("/")[-1].split(":")[0]
            for obj in nodes.values()
            if isinstance(obj, ParameterNode)
        }
    )
    return {
        "event_count": len(profile_doc.get("events", [])),
        "tgj_node_count": len(tgj.get("nodes", {})),
        "message_names": _message_names(nodes),
        "param_names": param_names,
    }


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
        return {"query": query, "plan": plan}

    def synth_node(state):
        query = str(_raw(state["query"]))
        plan = str(_raw(state["plan"]))
        prompt = render_template(synth_prompt.data, query=query, plan=plan)
        answer = call_chat_text(
            llm,
            system_prompt=SYNTH_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0,
        )
        return {"final_answer": node(answer, name="final_answer_node")}

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
        graph_agents_functions=["planner_node", "synth_node"],
        graph_prompts_list=[planner_prompt, synth_prompt],
        train_graph_agents_functions=False,
        output_key="final_answer",
    )
    optimizer = TraceMutatingOptimizer(synth_prompt, OPTIMIZED_SYNTH_PROMPT, "synth_prompt")
    return instrumented, optimizer, lambda: synth_prompt.data


def make_otel_case(llm, observe_with: Tuple[str, ...] = ()):
    instrumented = instrument_graph(
        graph=None,
        backend="otel",
        observe_with=observe_with,
        llm=llm,
        initial_templates=dict(DEFAULT_TEMPLATES),
        output_key="final_answer",
    )
    instrumented.backend = "otel"
    templates = instrumented.templates
    tracing_llm = instrumented.tracing_llm

    def planner_call(query: str) -> str:
        return tracing_llm.template_prompt_call(
            span_name="planner",
            template_name="planner_prompt",
            template=templates["planner_prompt"],
            variables={"query": query},
            system_prompt=PLANNER_SYSTEM_PROMPT,
            optimizable_key="planner",
            temperature=0,
        )

    def synth_call(query: str, plan: str) -> str:
        return tracing_llm.template_prompt_call(
            span_name="synth",
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
    optimizer = DictUpdateOptimizer({"synth_prompt": OPTIMIZED_SYNTH_PROMPT})
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
        output_key="final_answer",
    )
    optimizer = DictUpdateOptimizer({"synth_prompt": OPTIMIZED_SYNTH_PROMPT})
    return instrumented, optimizer, lambda: templates["synth_prompt"]


def run_case(name: str, builder):
    instrumented, optimizer, prompt_getter = builder()
    result = optimize_graph(
        instrumented,
        queries=QUERIES,
        iterations=ITERATIONS,
        optimizer=optimizer,
        eval_fn=eval_fn,
        output_key="final_answer",
    )

    probe = instrumented.invoke({"query": "What is CRISPR?"})
    answer_preview = str(_raw(probe.get("final_answer", probe)))[:120]

    summary = {
        "config": name,
        "score_history": [round(x, 3) for x in result.score_history],
        "best_iteration": result.best_iteration,
        "best_updates": dict(result.best_updates),
        "final_synth_prompt": prompt_getter(),
        "answer_preview": answer_preview,
        "observers": [a.carrier for a in getattr(instrumented, "_last_observer_artifacts", [])],
        "trace_summary": None,
        "otel_summary": None,
        "sysmon_summary": None,
    }

    if getattr(instrumented, "backend", None) == "trace":
        answer_node = probe.get("final_answer")
        summary["trace_summary"] = {
            "is_node": hasattr(answer_node, "parents"),
            "parent_count": len(getattr(answer_node, "parents", [])),
            "parameter_count": len(getattr(instrumented, "parameters", [])),
        }
    elif getattr(instrumented, "backend", None) == "otel":
        otlp = instrumented.session.flush_otlp(clear=True)
        summary["otel_summary"] = summarize_otlp(otlp)
    elif getattr(instrumented, "backend", None) == "sysmon":
        summary["sysmon_summary"] = summarize_sysmon(instrumented._last_profile_doc)

    for artifact in getattr(instrumented, "_last_observer_artifacts", []):
        if artifact.carrier == "otel":
            summary["otel_summary"] = summarize_otlp(artifact.raw)
        elif artifact.carrier == "sysmon":
            summary["sysmon_summary"] = summarize_sysmon(artifact.profile_doc)

    assert summary["best_iteration"] >= 2
    assert "Start the answer exactly with [BENCH_OK]." in summary["final_synth_prompt"]
    return summary


def main():
    print("\n" + "=" * 80)
    print("LangGraph live optimization comparison")
    print("=" * 80)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"sys.monitoring available: {HAS_SYSMON}")
    print(f"OPENROUTER_MODEL={OPENROUTER_MODEL}")

    if not OPENROUTER_API_KEY:
        print("\n[SKIP] OPENROUTER_API_KEY is not set.")
        print("This demo is intentionally live-only. Set OPENROUTER_API_KEY to run the benchmark.")
        return
    if OpenAI is None:
        print("\n[SKIP] openai package is unavailable.")
        return

    llm = make_live_llm()
    cases = [
        ("trace", lambda: make_trace_case(llm, ())),
        ("trace+otel", lambda: make_trace_case(llm, ("otel",))),
        ("otel", lambda: make_otel_case(llm, ())),
    ]
    if HAS_SYSMON:
        cases.extend(
            [
                ("trace+sysmon", lambda: make_trace_case(llm, ("sysmon",))),
                ("trace+otel+sysmon", lambda: make_trace_case(llm, ("otel", "sysmon"))),
                ("otel+sysmon", lambda: make_otel_case(llm, ("sysmon",))),
                ("sysmon", lambda: make_sysmon_case(llm)),
            ]
        )

    rows = [run_case(name, builder) for name, builder in cases]

    print("\nOptimization comparison (5 iterations)\n")
    print("| config | score_history | best_iteration | observers |")
    print("|---|---|---:|---|")
    for row in rows:
        print(
            f"| {row['config']} | {row['score_history']} | {row['best_iteration']} "
            f"| {','.join(row['observers']) or '-'} |"
        )

    print("\nBinding / update inspection\n")
    for row in rows:
        print(f"## {row['config']}")
        print(f"best_updates: {row['best_updates']}")
        print(f"final_synth_prompt: {row['final_synth_prompt']}")
        print(f"answer_preview: {row['answer_preview']}")
        if row['trace_summary'] is not None:
            print(f"trace_summary: {row['trace_summary']}")
        if row['otel_summary'] is not None:
            print(f"otel_summary: {row['otel_summary']}")
        if row['sysmon_summary'] is not None:
            print(f"sysmon_summary: {row['sysmon_summary']}")
        print()


if __name__ == "__main__":
    main()
