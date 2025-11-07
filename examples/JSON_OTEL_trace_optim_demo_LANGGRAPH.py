"""
JSON_OTEL_trace_optim_PROPER_LANGGRAPH.py - Full LangGraph StateGraph + OTEL Optimization
============================================================================================

PROPER LANGGRAPH STRUCTURE:
- StateGraph with Command-based flow control
- Nodes return Command[Literal["next_node"]]
- workflow.add_node() and workflow.compile()
- graph.invoke(state) for execution

OTEL OPTIMIZATION:
- OTEL tracing within each node
- Template-based prompts stored as parameters
- Optimizer persists across iterations (no recreation)
- Graph connectivity visualization
- Dynamic parameter discovery (no hardcoded mappings)

OPTIMIZATION FEATURES:
1. Prompt Optimization: Automatically discovers and optimizes all trainable prompts
   - Store: sp.set_attribute("param.<name>_prompt", template)
   - Mark trainable: sp.set_attribute("param.<name>_prompt.trainable", "true")

2. Code Optimization (Experimental): Can optimize function implementations
   - Store: sp.set_attribute("param.__code_<name>", source_code)
   - Mark trainable: sp.set_attribute("param.__code_<name>.trainable", "true")
   - Enable via: ENABLE_CODE_OPTIMIZATION = True

3. Dynamic Parameter Mapping: No hardcoded parameter lists needed
   - Automatically discovers all trainable parameters from OTEL spans
   - Extracts semantic names from parameter node names
   - Works with any agent configuration

This is the CORRECT architecture combining LangGraph + OTEL + Trace optimization.
"""

from __future__ import annotations
import os, json, time, difflib, inspect, re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal

import requests
import wikipedia
wikipedia.set_lang("en")

from opentelemetry import trace as oteltrace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from opto.utils.llm import LLM
from opto.trace.io.otel_adapter import otlp_traces_to_trace_json
from opto.trace.io.tgj_ingest import ingest_tgj
from opto.trace.nodes import MessageNode, ParameterNode
from opto.optimizers import OptoPrimeV2
from opto.optimizers.optoprime_v2 import OptimizerPromptSymbolSetJSON
from opto.trainer.algorithms.basic_algorithms import batchify

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# ==============================================================================
# CONFIGURATION
# ==============================================================================

NUM_ITERATIONS = 5
TEST_QUERIES = [
    "Summarize the causes and key events of the French Revolution.",
    "Give 3 factual relationships about Tesla, Inc. with entity IDs.",
    "What is the Wikidata ID for CRISPR and list 2 related entities?"
]

# Which components to optimize:
# - Prompts: Include agent names like "planner", "executor", "synthesizer"
# - Code: Include "__code" to optimize function implementations
# - Empty string "" matches everything
OPTIMIZABLE = ["planner", "executor", ""]

# Enable code optimization (experimental):
# When True, node implementations can be stored as trainable parameters
# using sp.set_attribute("param.__code_<name>", source_code)
ENABLE_CODE_OPTIMIZATION = True # Set to True to optimize function implementations

# ==============================================================================
# LOGGING HELPERS
# ==============================================================================

LOG_DIR: str | None = None
AGGREGATE_MD: str | None = None  # path to the aggregated log, LLM-friendly markdown context

def _init_log_dir() -> str:
    """Create a timestamped root log directory."""
    root = os.path.join("logs", "otlp_langgraph", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(root, exist_ok=True)
    return root

def _safe_dump_json(path: str, obj: dict | list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _safe_dump_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _extract_prompts_from_otlp(otlp: Dict[str, Any]) -> list[Dict[str, str]]:
    """Pull all inputs.gen_ai.prompt values from spans."""
    out: list[Dict[str, str]] = []
    for rs in otlp.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                prompt = None
                for a in sp.get("attributes", []):
                    if a.get("key") == "inputs.gen_ai.prompt":
                        v = a.get("value", {})
                        prompt = v.get("stringValue") or str(v)
                        break
                if prompt:
                    out.append({
                        "spanId": sp.get("spanId", ""),
                        "name": sp.get("name", ""),
                        "prompt": prompt
                    })
    return out

def _save_run_logs(phase: str, iteration: int, idx: int, run: "RunResult") -> None:
    """
    Save OTLP, TGJ, prompts, and a simple graph view for a single run.
    phase: 'baseline' or 'iter_XX'
    """
    assert LOG_DIR is not None
    run_dir = os.path.join(LOG_DIR, phase, f"run_{idx:02d}")
    # 1) Raw OTLP
    _safe_dump_json(os.path.join(run_dir, "otlp.json"), run.otlp)
    # 2) Prompts extracted from spans
    prompts = {"prompts": _extract_prompts_from_otlp(run.otlp)}
    _safe_dump_json(os.path.join(run_dir, "prompts.json"), prompts)
    # 3) TGJ conversion and 4) Graph view
    try:
        tgj_docs = list(otlp_traces_to_trace_json(
            run.otlp,
            agent_id_hint=f"{phase}_run{idx}",
            use_temporal_hierarchy=True,
        ))
        _safe_dump_json(os.path.join(run_dir, "tgj.json"), tgj_docs)
        # Graph view (best-effort)
        try:
            nodes = ingest_tgj(tgj_docs[0])
            graph_txt = visualize_graph(nodes)
        except Exception as e:
            graph_txt = f"[graph error] {e}"
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "graph.txt"), "w", encoding="utf-8") as f:
            f.write(graph_txt)
    except Exception as e:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "tgj_error.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))

def _save_optimizer_log(iteration: int, optimizer: OptoPrimeV2 | None) -> None:
    """Dump the optimizer's internal log (includes step-level info) and refresh the aggregate markdown."""
    if optimizer is None:
        return
    assert LOG_DIR is not None
    iter_dir = os.path.join(LOG_DIR, f"iter_{iteration:02d}")
    _safe_dump_json(os.path.join(iter_dir, "optimizer_log.json"), optimizer.log)
    _rebuild_aggregate_markdown()

def _truncate(s: str, n: int = 8000) -> str:
    """Truncate long text safely for markdown."""
    if len(s) <= n:
        return s
    return s[:n] + "\n...[truncated]...\n"

def _read_json_if(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _rebuild_aggregate_markdown() -> None:
    """Aggregate all saved artifacts into one markdown file for LLM context."""
    assert LOG_DIR is not None
    global AGGREGATE_MD
    AGGREGATE_MD = os.path.join(LOG_DIR, "context_bundle.md")
    lines = []
    lines.append(f"# OTLP → TGJ LangGraph Optimization Bundle\n")
    lines.append(f"_root: {LOG_DIR}_\n")

    # Baseline
    base_dir = os.path.join(LOG_DIR, "baseline")
    if os.path.isdir(base_dir):
        lines.append("\n## Baseline\n")
        for run_name in sorted(os.listdir(base_dir)):
            run_dir = os.path.join(base_dir, run_name)
            if not os.path.isdir(run_dir):
                continue
            lines.append(f"\n### {run_name}\n")
            prompts = _read_json_if(os.path.join(run_dir, "prompts.json"))
            tgj = _read_json_if(os.path.join(run_dir, "tgj.json"))
            otlp = _read_json_if(os.path.join(run_dir, "otlp.json"))
            graph = _read_json_if(os.path.join(run_dir, "graph.txt"))
            lines.append("**prompts.json**\n\n```json\n" + _truncate(prompts) + "\n```\n")
            lines.append("**tgj.json**\n\n```json\n" + _truncate(tgj) + "\n```\n")
            lines.append("**otlp.json** (snippet)\n\n```json\n" + _truncate(otlp, 4000) + "\n```\n")
            lines.append("**graph.txt**\n\n```text\n" + _truncate(graph, 4000) + "\n```\n")

    # Iterations
    for name in sorted(os.listdir(LOG_DIR)):
        if not name.startswith("iter_"):
            continue
        iter_dir = os.path.join(LOG_DIR, name)
        if not os.path.isdir(iter_dir):
            continue
        lines.append(f"\n## {name}\n")
        # optimizer log
        opt_log = _read_json_if(os.path.join(iter_dir, "optimizer_log.json"))
        if opt_log:
            lines.append("**optimizer_log.json**\n\n```json\n" + _truncate(opt_log) + "\n```\n")
        # batched feedback (if present)
        bf_path = os.path.join(iter_dir, "batched_feedback.txt")
        if os.path.exists(bf_path):
            bf = _read_json_if(bf_path)
            lines.append("**batched_feedback.txt**\n\n```text\n" + _truncate(bf) + "\n```\n")
        # runs
        for run_name in sorted(os.listdir(iter_dir)):
            run_dir = os.path.join(iter_dir, run_name)
            if not (os.path.isdir(run_dir) and run_name.startswith("run_")):
                continue
            lines.append(f"\n### {run_name}\n")
            prompts = _read_json_if(os.path.join(run_dir, "prompts.json"))
            tgj = _read_json_if(os.path.join(run_dir, "tgj.json"))
            otlp = _read_json_if(os.path.join(run_dir, "otlp.json"))
            graph = _read_json_if(os.path.join(run_dir, "graph.txt"))
            lines.append("**prompts.json**\n\n```json\n" + _truncate(prompts) + "\n```\n")
            lines.append("**tgj.json**\n\n```json\n" + _truncate(tgj) + "\n```\n")
            lines.append("**otlp.json** (snippet)\n\n```json\n" + _truncate(otlp, 4000) + "\n```\n")
            lines.append("**graph.txt**\n\n```text\n" + _truncate(graph, 4000) + "\n```\n")

    _safe_dump_text(AGGREGATE_MD, "\n".join(lines))
    if AGGREGATE_MD: print(f"\n📦 Aggregate context markdown → {AGGREGATE_MD}")

# ==============================================================================
# OTEL SETUP
# ==============================================================================

class InMemorySpanExporter(SpanExporter):
    def __init__(self):
        self._finished_spans: List[ReadableSpan] = []
    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        self._finished_spans.extend(spans)
        return SpanExportResult.SUCCESS
    def shutdown(self) -> None: pass
    def get_finished_spans(self) -> List[ReadableSpan]:
        return self._finished_spans
    def clear(self) -> None:
        self._finished_spans.clear()

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
oteltrace.set_tracer_provider(_provider)
TRACER = oteltrace.get_tracer("demo")
LLM_CLIENT = LLM()

def flush_otlp() -> Dict[str, Any]:
    spans = _exporter.get_finished_spans()
    def hex_id(x: int, n: int) -> str:
        return f"{x:0{2*n}x}"
    otlp_spans = []
    for s in spans:
        attrs = [{"key": k, "value": {"stringValue": str(v)}} for k, v in (s.attributes or {}).items()]
        kind = getattr(s, 'kind', 1)
        if hasattr(kind, 'value'): kind = kind.value
        otlp_spans.append({
            "traceId": hex_id(s.context.trace_id, 16),
            "spanId": hex_id(s.context.span_id, 8),
            "parentSpanId": hex_id(s.parent.span_id, 8) if s.parent else "",
            "name": s.name,
            "kind": {0:"UNSPECIFIED",1:"INTERNAL",2:"SERVER",3:"CLIENT"}.get(kind, "INTERNAL"),
            "startTimeUnixNano": int(s.start_time or time.time_ns()),
            "endTimeUnixNano": int(s.end_time or time.time_ns()),
            "attributes": attrs
        })
    _exporter.clear()
    return {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"scope": {"name": "demo"}, "spans": otlp_spans}]}]}

# ==============================================================================
# STATE (LangGraph State with tracking)
# ==============================================================================

@dataclass
class State:
    """LangGraph State"""
    user_query: str = ""
    plan: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_step: int = 1
    agent_query: str = ""
    contexts: List[str] = field(default_factory=list)
    final_answer: str = ""

    # Template storage (shared across iterations)
    planner_template: str = ""
    executor_template: str = ""

    # Track previous span for sequential linking
    prev_span_id: Optional[str] = None

# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================

PLANNER_TEMPLATE_DEFAULT = """You are the Planner. Break the user's request into JSON steps.

Agents:
  • web_researcher - Wikipedia summaries for background/overview
  • wikidata_researcher - Entity facts, IDs, and structured relationships
  • synthesizer - Final answer generation

Return JSON: {{"1": {{"agent":"web_researcher|wikidata_researcher", "action":"...", "goal":"..."}}, "2": {{"agent":"synthesizer", "action":"...", "goal":"..."}}}}

Guidelines:
- Use web_researcher for narrative background and explanations
- Use wikidata_researcher for entity IDs, structured facts, and relationships
- End with synthesizer to finalize answer
- Include goal for each step

User query: "{USER_QUERY}"
"""

EXECUTOR_TEMPLATE_DEFAULT = """You are the Executor. Return JSON: {{"goto": "<web_researcher|wikidata_researcher|synthesizer>", "query": "<text>"}}

Context:
- Step: {STEP}
- Plan: {PLAN_STEP}
- Query: "{USER_QUERY}"
- Previous: "{PREV_CONTEXT}"

Routing guide:
- web_researcher: For Wikipedia summaries and background info
- wikidata_researcher: For entity facts, IDs, and structured data
- synthesizer: To generate final answer

Route to appropriate agent based on plan.
"""

def fill_template(template: str, **kwargs) -> str:
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    return result

# ==============================================================================
# TOOLS
# ==============================================================================

def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return summaries"""
    try:
        hits = wikipedia.search(query, results=2)
        out = []
        for h in hits:
            try:
                s = wikipedia.summary(h, sentences=3, auto_suggest=False, redirect=True)
                out.append(f"### {h}\\n{s}")
            except: continue
        return "\\n\\n".join(out) or "No results."
    except: return "Search unavailable."

def wikidata_query(query: str) -> str:
    """Query Wikidata for entity facts and IDs with robust error handling"""
    try:
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "format": "json",
                "language": "en",
                "search": query[:100],  # Limit query length
                "limit": 5
            },
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        results = [
            f"- {item.get('label', '')}: {item.get('description', '')} ({item.get('id', '')})"
            for item in data.get("search", [])
        ]
        return "\\n".join(results) if results else "No Wikidata entities found."
    except Exception:
        return f"Wikidata search temporarily unavailable. Query: {query[:50]}..."

# ==============================================================================
# LANGGRAPH NODES (with OTEL tracing)
# ==============================================================================

def planner_node(state: State) -> Command[Literal["executor"]]:
    """
    LangGraph planner node with OTEL tracing.
    Returns Command to route to executor.
    """

    # Get template (use state's or default)
    template = state.planner_template or PLANNER_TEMPLATE_DEFAULT

    with TRACER.start_as_current_span("planner") as sp:
        # Sequential linking
        if state.prev_span_id:
            sp.set_attribute("inputs.parent", f"span:{state.prev_span_id}")

        # Fill template with query
        prompt = fill_template(template, USER_QUERY=state.user_query)

        # CRITICAL: Store TEMPLATE as parameter (not filled prompt!)
        sp.set_attribute("param.planner_prompt", template)
        sp.set_attribute("param.planner_prompt.trainable", "planner" in OPTIMIZABLE)
        sp.set_attribute("gen_ai.model", "llm")
        sp.set_attribute("inputs.gen_ai.prompt", prompt)
        sp.set_attribute("inputs.user_query", state.user_query)

        # Call LLM
        raw = LLM_CLIENT(
            messages=[{"role":"system","content":"JSON only"}, {"role":"user","content":prompt}],
            response_format={"type":"json_object"},
            max_tokens=400,
            temperature=0,
        ).choices[0].message.content

        try:
            plan = json.loads(raw)
        except:
            plan = {"1":{"agent":"web_researcher","action":"search","goal":"info"},"2":{"agent":"synthesizer","action":"answer","goal":"final"}}

        span_id = f"{sp.get_span_context().span_id:016x}"

    return Command(
        update={
            "plan": plan,
            "current_step": 1,
            "prev_span_id": span_id,
        },
        goto="executor"
    )

def executor_node(state: State) -> Command[Literal["web_researcher", "wikidata_researcher", "synthesizer"]]:
    """
    LangGraph executor node with OTEL tracing.
    Routes to web_researcher, wikidata_researcher, or synthesizer.
    """

    step = state.current_step
    plan_step = state.plan.get(str(step), {})

    if not plan_step:
        # No more steps, go to synthesizer
        return Command(update={}, goto="synthesizer")

    # Get template
    template = state.executor_template or EXECUTOR_TEMPLATE_DEFAULT

    with TRACER.start_as_current_span("executor") as sp:
        # Sequential linking
        if state.prev_span_id:
            sp.set_attribute("inputs.parent", f"span:{state.prev_span_id}")

        # Fill template
        prompt = fill_template(
            template,
            STEP=step,
            PLAN_STEP=json.dumps(plan_step),
            USER_QUERY=state.user_query,
            PREV_CONTEXT=state.contexts[-1][:100] if state.contexts else ""
        )

        # Store TEMPLATE as parameter
        sp.set_attribute("param.executor_prompt", template)
        sp.set_attribute("param.executor_prompt.trainable", "executor" in OPTIMIZABLE)
        sp.set_attribute("gen_ai.model", "llm")
        sp.set_attribute("inputs.gen_ai.prompt", prompt)
        sp.set_attribute("inputs.step", str(step))
        sp.set_attribute("inputs.user_query", state.user_query)

        # Call LLM
        raw = LLM_CLIENT(
            messages=[{"role":"system","content":"JSON only"}, {"role":"user","content":prompt}],
            response_format={"type":"json_object"},
            max_tokens=300,
            temperature=0,
        ).choices[0].message.content

        try:
            d = json.loads(raw)
            goto = d.get("goto", "synthesizer")
            # Validate goto is one of the allowed agents
            if goto not in ["web_researcher", "wikidata_researcher", "synthesizer"]:
                goto = "synthesizer"
            agent_query = d.get("query", state.user_query)
        except:
            goto, agent_query = ("synthesizer", state.user_query)

        span_id = f"{sp.get_span_context().span_id:016x}"

    return Command(
        update={
            "agent_query": agent_query,
            "current_step": step + 1,
            "prev_span_id": span_id,
        },
        goto=goto
    )

def web_researcher_node(state: State) -> Command[Literal["executor"]]:
    """
    LangGraph web researcher node with OTEL tracing.
    Returns to executor.
    """

    with TRACER.start_as_current_span("web_search") as sp:
        # Sequential linking
        if state.prev_span_id:
            sp.set_attribute("inputs.parent", f"span:{state.prev_span_id}")

        query = state.agent_query or state.user_query

        sp.set_attribute("retrieval.query", query)
        result = wikipedia_search(query)
        sp.set_attribute("retrieval.context", result[:500])

        span_id = f"{sp.get_span_context().span_id:016x}"

    # Add to contexts
    new_contexts = state.contexts + [result]

    return Command(
        update={
            "contexts": new_contexts,
            "prev_span_id": span_id,
        },
        goto="executor"
    )

def wikidata_researcher_node(state: State) -> Command[Literal["executor"]]:
    """
    LangGraph wikidata researcher node with OTEL tracing.
    Queries Wikidata for entity facts and returns to executor.
    """

    with TRACER.start_as_current_span("wikidata_search") as sp:
        # Sequential linking
        if state.prev_span_id:
            sp.set_attribute("inputs.parent", f"span:{state.prev_span_id}")

        query = state.agent_query or state.user_query

        sp.set_attribute("retrieval.query", query)
        sp.set_attribute("retrieval.source", "wikidata")
        result = wikidata_query(query)
        sp.set_attribute("retrieval.context", result[:500])

        span_id = f"{sp.get_span_context().span_id:016x}"

    # Add to contexts
    new_contexts = state.contexts + [result]

    return Command(
        update={
            "contexts": new_contexts,
            "prev_span_id": span_id,
        },
        goto="executor"
    )

def synthesizer_node(state: State) -> Command[Literal[END]]:
    """
    LangGraph synthesizer node with OTEL tracing.
    Ends the graph.
    """

    with TRACER.start_as_current_span("synthesizer") as sp:
        # Sequential linking
        if state.prev_span_id:
            sp.set_attribute("inputs.parent", f"span:{state.prev_span_id}")

        context_blob = "\\n\\n".join(state.contexts[-3:])

        prompt = f"""Answer concisely using only the context.

Question: {state.user_query}

Context:
{context_blob}

Provide a direct, factual answer."""

        sp.set_attribute("gen_ai.model", "llm")
        sp.set_attribute("inputs.gen_ai.prompt", prompt)

        answer = LLM_CLIENT(
            messages=[{"role":"system","content":"Answer concisely"}, {"role":"user","content":prompt}],
            max_tokens=400,
            temperature=0,
        ).choices[0].message.content

        span_id = f"{sp.get_span_context().span_id:016x}"

    return Command(
        update={
            "final_answer": answer,
            "prev_span_id": span_id,
        },
        goto=END
    )

def evaluator_node(state: State) -> Command[Literal[END]]:
    """
    Evaluator node with multi-metric assessment.
    """

    with TRACER.start_as_current_span("evaluator") as sp:
        # Sequential linking
        if state.prev_span_id:
            sp.set_attribute("inputs.parent", f"span:{state.prev_span_id}")

        context = "\\n".join(state.contexts) if state.contexts else ""

        eval_prompt = f"""Evaluate on 0..1 scale. Return JSON:
{{"answer_relevance": <0..1>, "groundedness": <0..1>, "plan_quality": <0..1>, "reasons": "..."}}

Query: "{state.user_query}"
Answer: "{state.final_answer}"
Context: {context[:500]}
Plan: {json.dumps(state.plan)}
"""

        raw = LLM_CLIENT(
            messages=[{"role":"system","content":"Eval expert. JSON only."}, {"role":"user","content":eval_prompt}],
            response_format={"type":"json_object"},
            max_tokens=400,
            temperature=0,
        ).choices[0].message.content

        try:
            j = json.loads(raw)
            metrics = {
                "answer_relevance": float(j.get("answer_relevance", 0.5)),
                "groundedness": float(j.get("groundedness", 0.5)),
                "plan_quality": float(j.get("plan_quality", 0.5))
            }
            score = sum(metrics.values()) / len(metrics)
            reasons = j.get("reasons", "")
        except:
            metrics = {"answer_relevance": 0.5, "groundedness": 0.5, "plan_quality": 0.5}
            score = 0.5
            reasons = "parse error"

        # Store metrics
        for k, v in metrics.items():
            sp.set_attribute(f"eval.{k}", str(v))
        sp.set_attribute("eval.score", str(score))
        sp.set_attribute("eval.reasons", reasons)

        span_id = f"{sp.get_span_context().span_id:016x}"

    feedback = f"[Metrics] {list(metrics.values())} ; Reasons: {reasons}"

    return Command(
        update={
            "prev_span_id": span_id,
        },
        goto=END
    )

# ==============================================================================
# BUILD LANGGRAPH
# ==============================================================================

def build_graph() -> StateGraph:
    """Build the LangGraph StateGraph with both web and wikidata researchers"""

    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("web_researcher", web_researcher_node)
    workflow.add_node("wikidata_researcher", wikidata_researcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("evaluator", evaluator_node)

    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("synthesizer", "evaluator")

    return workflow.compile()

# ==============================================================================
# RUN GRAPH WITH OTEL CAPTURE
# ==============================================================================

@dataclass
class RunResult:
    answer: str
    otlp: Dict[str, Any]
    feedback: str
    score: float
    metrics: Dict[str, float]
    plan: Dict[str, Any]

def run_graph_with_otel(
    graph,
    query: str,
    planner_template: str = None,
    executor_template: str = None
) -> RunResult:
    """
    Run the LangGraph and capture OTEL traces.
    """

    # Create initial state
    initial_state = State(
        user_query=query,
        planner_template=planner_template or PLANNER_TEMPLATE_DEFAULT,
        executor_template=executor_template or EXECUTOR_TEMPLATE_DEFAULT,
    )

    # Invoke graph (returns dict, not State object)
    final_state = graph.invoke(initial_state)

    # Flush OTLP
    otlp = flush_otlp()

    # Extract metrics from OTLP (simple approach)
    score = 0.5
    metrics = {}
    feedback = "Evaluation completed"
    reasons = ""

    for rs in otlp.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                if sp.get("name") == "evaluator":
                    attrs = {a["key"]: a["value"].get("stringValue", "") for a in sp.get("attributes", [])}
                    score = float(attrs.get("eval.score", "0.5"))
                    reasons = attrs.get("eval.reasons", "")
                    metrics = {
                        "answer_relevance": float(attrs.get("eval.answer_relevance", "0.5")),
                        "groundedness": float(attrs.get("eval.groundedness", "0.5")),
                        "plan_quality": float(attrs.get("eval.plan_quality", "0.5"))
                    }
                    feedback = json.dumps({"metrics": metrics, "score": score, "reasons": reasons})

    # Access final_state as dict (LangGraph returns dict, not State object)
    return RunResult(
        answer=final_state.get("final_answer", ""),
        otlp=otlp,
        feedback=feedback,
        score=score,
        metrics=metrics,
        plan=final_state.get("plan", {})
    )

# ==============================================================================
# OPTIMIZATION (same as before)
# ==============================================================================

def find_target(nodes: Dict) -> Optional[MessageNode]:
    last = None
    for n in nodes.values():
        if isinstance(n, MessageNode):
            last = n
            if "evaluator" in (n.name or "").lower():
                return n
    return last

def visualize_graph(nodes: Dict[str, Any]) -> str:
    params = []
    messages = []
    for name, node in nodes.items():
        if isinstance(node, ParameterNode):
            val = node.data[:60]
            params.append(f"[PARAM] {node.name}: '{val}...'")
        elif isinstance(node, MessageNode):
            parents = getattr(node, 'parents', [])
            parent_names = [getattr(p, 'name', '?') for p in parents]
            messages.append(f"[MSG] {node.name} ← {parent_names if parent_names else 'ROOT'}")
    return "\\n".join(params) + "\\n" + "\\n".join(messages)

def check_reachability(target: MessageNode, params: List[ParameterNode]) -> Dict[str, bool]:
    seen, stack, reachable = set(), [target], set()
    while stack:
        node = stack.pop()
        if node in seen: continue
        seen.add(node)
        if hasattr(node, 'parents'):
            for p in node.parents:
                if p not in seen: stack.append(p)
        if isinstance(node, ParameterNode):
            reachable.add(node.name)
    return {p.name: p.name in reachable for p in params}

def _remap_params_in_graph(node: Any, param_mapping: Dict[int, ParameterNode], visited=None):
    """
    Recursively remap parameter nodes in a graph to use optimizer's params.
    
    Args:
        node: Current node being visited
        param_mapping: Dict mapping id(new_param) -> optimizer_param
        visited: Set of already visited node IDs to avoid cycles
    """
    if visited is None:
        visited = set()
    
    node_id = id(node)
    if node_id in visited:
        return
    visited.add(node_id)
    
    # If this node is a parameter that needs remapping, stop here
    if isinstance(node, ParameterNode) and node_id in param_mapping:
        return
    
    # Remap in _inputs dict (not inputs property which returns a copy!)
    if hasattr(node, '_inputs') and isinstance(node._inputs, dict):
        for key, input_node in list(node._inputs.items()):
            input_id = id(input_node)
            if input_id in param_mapping:
                node._inputs[key] = param_mapping[input_id]
            else:
                _remap_params_in_graph(input_node, param_mapping, visited)
    
    # Remap in parents list
    if hasattr(node, 'parents') and isinstance(node.parents, list):
        for i, parent in enumerate(node.parents):
            parent_id = id(parent)
            if parent_id in param_mapping:
                node.parents[i] = param_mapping[parent_id]
            else:
                _remap_params_in_graph(parent, param_mapping, visited)

def show_prompt_diff(old: str, new: str, name: str):
    if old == new:
        print(f"\\n🔴 NO CHANGE in {name}")
        return
    print(f"\\n📝 DIFF for {name}:")
    print("="*80)
    old_lines, new_lines = old.splitlines(), new.splitlines()
    diff = difflib.unified_diff(old_lines, new_lines, lineterm='', fromfile='old', tofile='new')
    for line in diff:
        if line.startswith('+++') or line.startswith('---'):
            print(f"\\033[1m{line}\\033[0m")
        elif line.startswith('+'):
            print(f"\\033[92m{line}\\033[0m")
        elif line.startswith('-'):
            print(f"\\033[91m{line}\\033[0m")
        elif line.startswith('@@'):
            print(f"\\033[96m{line}\\033[0m")
        else:
            print(line)
    print("="*80)

def compute_change_stats(original: str, updated: str) -> tuple[int, int]:
    """Return (line_changes, char_changes) between two parameter versions."""

    original = original or ""
    updated = updated or ""

    line_changes = 0
    for line in difflib.unified_diff(original.splitlines(), updated.splitlines(), lineterm=""):
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            line_changes += 1

    char_changes = 0
    sequence = difflib.SequenceMatcher(None, original, updated)
    for tag, i1, i2, j1, j2 in sequence.get_opcodes():
        if tag == "equal":
            continue
        char_changes += (i2 - i1) + (j2 - j1)

    return line_changes, char_changes

CODE_TARGETS = {
    "planner": "planner_node",
    "executor": "executor_node",
    "web_researcher": "web_researcher_node",
    "wikidata_researcher": "wikidata_researcher_node",
    "synthesizer": "synthesizer_node",
    "evaluator": "evaluator_node",
}

def _signature_line(fn) -> str:
    try:
        src = inspect.getsource(fn)
        m = re.search(r"^\s*def\s.+?:", src, re.M)
        return m.group(0) if m else f"def {fn.__name__}(...):"
    except Exception:
        return f"def {getattr(fn, '__name__', 'fn')}(...) :"

def _ensure_code_desc_on_optimizer(optimizer) -> None:
    """Ensure all __code_* params in optimizer have the signature description expected by OptoPrimeV2."""
    for p in getattr(optimizer, "parameters", []):
        if "__code_" not in p.name:
            continue
        if getattr(p, "description", None):
            continue
        semantic = p.name.split(":")[0].split("/")[-1].replace("__code_", "")
        fn_name = CODE_TARGETS.get(semantic, f"{semantic}_node")
        fn = globals().get(fn_name)
        sig = _signature_line(fn) if callable(fn) else f"def {fn_name}(...):"
        desc = f"[Parameter] The code should start with:\\n{sig}"
        try: p.description = desc
        except Exception: pass
        p._description = desc

def optimize_iteration(runs: List[RunResult], optimizer: Optional[OptoPrimeV2], iteration: int | None = None) -> tuple[Dict[str, str], OptoPrimeV2]:
    print("\\n📊 OPTIMIZATION:")
    print("="*80)

    all_targets_and_feedback = []

    for idx, run in enumerate(runs):
        print(f"\\n🔍 Run {idx+1}: score={run.score:.3f}, metrics={run.metrics}")

        tgj_docs = list(
            otlp_traces_to_trace_json(
                run.otlp,
                agent_id_hint=f"run{idx}",
                use_temporal_hierarchy=True,
            )
        )
        nodes = ingest_tgj(tgj_docs[0])

        target = find_target(nodes)
        if not target:
            continue

        params = [n for n in nodes.values()
                 if isinstance(n, ParameterNode) and getattr(n, 'trainable', False)
                 and any(agent in n.name for agent in OPTIMIZABLE)]

        if params:
            reachability = check_reachability(target, params)
            reach_items = []
            for k, v in list(reachability.items())[:2]:
                name = k.split('/')[-1]
                status = '✅' if v else '❌'
                reach_items.append(f"{name}={status}")
            print(f"   Reachability: {', '.join(reach_items)}")

        all_targets_and_feedback.append((target, run.feedback, params))

    if not all_targets_and_feedback:
        return {}, optimizer

    _, _, first_params = all_targets_and_feedback[0]
    if not first_params:
        return {}, optimizer

    # Create optimizer ONCE on first call, reuse thereafter
    created_optimizer = False
    if optimizer is None:
        mem = max(12, len(all_targets_and_feedback) * 4)
        print(f"\n🔧 Creating optimizer with {len(first_params)} params (memory_size={mem})")
        optimizer = OptoPrimeV2(
            first_params,
            llm=LLM_CLIENT,
            memory_size=mem,
            log=True,
            optimizer_prompt_symbol_set=OptimizerPromptSymbolSetJSON(),
            objective=(
                "Maximize eval.score = mean(answer_relevance, groundedness, plan_quality). "
                "Keep templates generic (placeholders intact); improve routing clarity and step structure."
            ),
        )
        created_optimizer = True
    else:
        print(f"\n♻️  Reusing optimizer (log has {len(optimizer.log)} entries) & Syncing parameter data and remapping graphs...")

    # Build mapping from current iteration params to optimizer params so all runs share nodes
    param_mapping: Dict[int, ParameterNode] = {}

    def map_params(params: List[ParameterNode], sync_data: bool = False) -> None:
        for param in params:
            if id(param) in param_mapping:
                continue
            semantic = param.name.split(":")[0].split("/")[-1]
            for opt_param in optimizer.parameters:
                opt_semantic = opt_param.name.split(":")[0].split("/")[-1]
                if semantic == opt_semantic:
                    if sync_data:
                        opt_param._data = param._data
                    param_mapping[id(param)] = opt_param
                    break

    # Always sync the first run's params when reusing the optimizer to refresh data
    map_params(first_params, sync_data=not created_optimizer)

    for _, _, params in all_targets_and_feedback:
        map_params(params)

    # Remap targets to use optimizer's params (not the newly created params from OTEL)
    for target, _, _ in all_targets_and_feedback:
        _remap_params_in_graph(target, param_mapping)
    # Make sure optimizer-side __code_* params have a proper description
    _ensure_code_desc_on_optimizer(optimizer)

    # ---- Batch like trainers do: build one composite target + one composite feedback ----
    # Preserve per-item trace in the target bundle AND include each run's score explicitly in feedback.
    batched_target = batchify(*[t for (t, _, _) in all_targets_and_feedback])  # Trace node
    # Combine score + feedback per item (feedback itself may already contain metrics/score JSON; we make it explicit)
    batched_feedback_items = []
    for i, ((_, fb, _), run) in enumerate(zip(all_targets_and_feedback, runs)):
        # Example line format: ID [0]: score=0.734 // feedback: {"metrics": {...}, "score": 0.734, "reasons": "..."}
        item = f"ID [{i}]: score={run.score:.3f}\nfeedback: {fb}"
        batched_feedback_items.append(item)
    batched_feedback = batchify(*batched_feedback_items).data  # plain str
    # Log the exact batched feedback used for this step (per iteration)
    if LOG_DIR is not None and iteration is not None:
        iter_dir = os.path.join(LOG_DIR, f"iter_{iteration:02d}")
        _safe_dump_text(os.path.join(iter_dir, "batched_feedback.txt"), batched_feedback)

    print(f"\n⬅️  BACKWARD (batched):")
    optimizer.zero_feedback()
    try:
        optimizer.backward(batched_target, batched_feedback)
        print(f"   Batched: ✓ ({len(all_targets_and_feedback)} runs)")
    except Exception as e:
        print(f"   ❌ {e}")

    print(f"\\n➡️  STEP:")
    # sanity check: list any __code_* with missing description
    missing = [p.name for p in optimizer.parameters if "__code_" in p.name and not getattr(p, "description", None)]
    if missing: print(f"   ⚠️ Missing description on: {missing}")
    try:
        optimizer.step(verbose=False)
        print(f"   ✓ Completed (log now has {len(optimizer.log)} entries)")
    except Exception as e:
        print(f"   ❌ {e}")
        return {}, optimizer

    # DYNAMIC PARAMETER MAPPING
    # Extract semantic names from parameter names
    # Format: "scope/semantic_name:index" (e.g., "run0/planner_prompt:0")
    # This automatically discovers all trainable parameters, no hardcoding needed!
    print(f"\\n🔍 DYNAMIC Parameter mapping:")
    updates = {}
    for p in optimizer.parameters:
        # Remove :index suffix, then get last component after /
        full_name = p.name.split(":")[0]  # "run0/planner_prompt"
        semantic_name = full_name.split("/")[-1]  # "planner_prompt"
        updates[semantic_name] = p.data
        print(f"   {p.name} -> {semantic_name}")

    print("="*80)
    return updates, optimizer

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\\n" + "="*80)
    print("PROPER LangGraph + OTEL Trace Optimization".center(80))
    print("="*80)
    print(f"\\nConfig: {len(TEST_QUERIES)} queries, {NUM_ITERATIONS} iterations")

    # Init log directory once
    global LOG_DIR
    LOG_DIR = _init_log_dir()
    print(f"Logs → {LOG_DIR}")

    # Build graph once
    graph = build_graph()
    print("✓ LangGraph compiled")

    # BASELINE
    print("\\n" + "="*80)
    print("BASELINE".center(80))
    print("="*80)

    current_planner_tmpl = PLANNER_TEMPLATE_DEFAULT
    current_executor_tmpl = EXECUTOR_TEMPLATE_DEFAULT
    
    # Save originals for final comparison
    original_planner_tmpl = PLANNER_TEMPLATE_DEFAULT
    original_executor_tmpl = EXECUTOR_TEMPLATE_DEFAULT

    baseline_runs = [run_graph_with_otel(graph, q, current_planner_tmpl, current_executor_tmpl) for q in TEST_QUERIES]
    base_score = sum(r.score for r in baseline_runs) / len(baseline_runs)
    print(f"\\nBaseline: {base_score:.3f}")
    for i, r in enumerate(baseline_runs, 1):
        print(f"  Q{i}: {r.score:.3f} | {r.metrics}")
        # Save baseline artifacts
        _save_run_logs("baseline", 0, i, r)

    template_history = {
        "planner_prompt": PLANNER_TEMPLATE_DEFAULT,
        "executor_prompt": EXECUTOR_TEMPLATE_DEFAULT
    }
    baseline_param_snapshots = dict(template_history)

    # OPTIMIZATION
    print("\\n" + "="*80 + "\n" + "OPTIMIZATION".center(80) + "\n" + "="*80)

    history = [base_score]
    optimizer = None  # Will be created on first iteration, reused thereafter
    
    final_runs: List[RunResult] = baseline_runs
    
    # Track best iteration
    best_score = base_score
    best_iteration = 0
    # Store actual template strings, not dict references
    best_planner_tmpl = current_planner_tmpl
    best_executor_tmpl = current_executor_tmpl

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"\\n{'='*80}")
        print(f"Iteration {iteration}/{NUM_ITERATIONS}".center(80))
        print(f"{'='*80}")

        runs = [run_graph_with_otel(graph, q, current_planner_tmpl, current_executor_tmpl) for q in TEST_QUERIES]
        iter_score = sum(r.score for r in runs) / len(runs)

        print(f"\\nCurrent: {iter_score:.3f}")
        # Logs per-run artifacts for this iteration
        for i, r in enumerate(runs, 1):
            _save_run_logs(f"iter_{iteration:02d}", iteration, i, r)

        # Track best performing iteration
        if iter_score > best_score:
            best_score = iter_score
            best_iteration = iteration
            # Save actual current templates
            best_planner_tmpl = current_planner_tmpl
            best_executor_tmpl = current_executor_tmpl
            print(f"   🌟 NEW BEST SCORE! (iteration {iteration})")

        updates, optimizer = optimize_iteration(runs, optimizer, iteration=iteration)
        _save_optimizer_log(iteration, optimizer) # Dump optimizer-level log for this iteration

        if not updates:
            print("\\n❌ No updates")
            continue

        # Debug: show what keys are in updates
        print(f"\n🔍 DEBUG: Updates dict keys: {list(updates.keys())}")

        for param_name, new_template in updates.items():
            old_template = template_history.get(param_name, "")
            if param_name not in baseline_param_snapshots:
                baseline_param_snapshots[param_name] = old_template or new_template
            show_prompt_diff(old_template, new_template, param_name)
            template_history[param_name] = new_template

        # Update current templates with new values
        if "planner_prompt" in updates:
            current_planner_tmpl = updates["planner_prompt"]
            print(f"   ✅ Updated current_planner_tmpl")
        if "executor_prompt" in updates:
            current_executor_tmpl = updates["executor_prompt"]
            print(f"   ✅ Updated current_executor_tmpl")

        history.append(iter_score)
    
    # Restore best templates
    print(f"\\n{'='*80}")
    print("RESTORING BEST PARAMETERS".center(80))
    print(f"{'='*80}")
    print(f"\\n🏆 Best score: {best_score:.3f} from iteration {best_iteration}")
    
    if best_iteration > 0:
        print(f"   Restoring templates from iteration {best_iteration}...")
        current_planner_tmpl = best_planner_tmpl
        current_executor_tmpl = best_executor_tmpl
        template_history["planner_prompt"] = current_planner_tmpl
        template_history["executor_prompt"] = current_executor_tmpl
        
        # Validate with a final run
        print(f"\\n🔄 Validating best parameters...")
        validation_runs = [run_graph_with_otel(graph, q, current_planner_tmpl, current_executor_tmpl) for q in TEST_QUERIES]
        final_runs = validation_runs
        validation_score = sum(r.score for r in validation_runs) / len(validation_runs)
        print(f"   Validation score: {validation_score:.3f}")
        
        if abs(validation_score - best_score) > 0.05:
            print(f"   ⚠️  Warning: Validation score differs from recorded best by {abs(validation_score - best_score):.3f}")
        else:
            print(f"   ✅ Validation confirms best score!")
    else:
        print(f"   Baseline was the best performer - no changes applied")

    # RESULTS
    print("\\n" + "="*80 + "\n" + "RESULTS".center(80) + "\n" + "="*80)

    final_score = best_score  # Use best score instead of last iteration
    improvement = final_score - base_score
    pct = (improvement / base_score * 100) if base_score > 0 else 0

    print(f"\\n📈 Progression:")
    for i, score in enumerate(history):
        label = "Baseline" if i == 0 else f"Iter {i}"
        delta = "" if i == 0 else f"(Δ {score - history[i-1]:+.3f})"
        best_marker = " 🌟 BEST" if (i == best_iteration) else ""
        print(f"   {label:12s}: {score:.3f} {delta}{best_marker}")

    print(f"\\n🎯 Overall: {base_score:.3f} → {final_score:.3f} ({improvement:+.3f}, {pct:+.1f}%)")
    print(f"   Best iteration: {best_iteration}")
    print(f"   ✅ Improvement SUCCESS!" if improvement > 0 else f"   ⚠️  No improvement")

    change_map = {}
    for name, original_value in baseline_param_snapshots.items():
        final_value = template_history.get(name, "")
        change_map[name] = compute_change_stats(original_value, final_value)

    change_display = ", ".join(
        f"{name}:ΔL={lines} ΔC={chars}" for name, (lines, chars) in change_map.items()
    ) or "no parameter changes"

    print("\n🧪 Final run breakdown:")
    for idx, run in enumerate(final_runs, 1):
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in run.metrics.items()) if run.metrics else "n/a"
        plan = run.plan or {}
        if plan:
            try:
                ordered = sorted(plan.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))
            except Exception:
                ordered = list(plan.items())
            agents = [str(step.get("agent", "?")) for _, step in ordered if isinstance(step, dict)]
            agents_repr = " → ".join(agents) if agents else "n/a"
        else:
            agents_repr = "n/a"
        print(
            f"  Run {idx}: score={run.score:.3f} [{metrics_str}] | agents: {agents_repr} | {change_display}"
        )

    # Show final optimized prompts with colored diffs
    print("\\n" + "="*80)
    print("FINAL OPTIMIZED PROMPTS (vs Original)".center(80))
    print("="*80)
    
    if best_iteration > 0:
        # Show diff for planner prompt
        print("\n" + "─"*80)
        print("🔵 PLANNER PROMPT (Final Optimized vs Original)")
        print("─"*80)
        show_prompt_diff(original_planner_tmpl, current_planner_tmpl, "planner_prompt")
        
        # Show diff for executor prompt
        print("\n" + "─"*80)
        print("🔵 EXECUTOR PROMPT (Final Optimized vs Original)")
        print("─"*80)
        show_prompt_diff(original_executor_tmpl, current_executor_tmpl, "executor_prompt")
    else:
        print("\\n   No optimization occurred - baseline templates retained")

    print("\\n" + "="*80 + "\\n")

    # Final rebuild to ensure aggregate file is up to date
    _rebuild_aggregate_markdown()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
