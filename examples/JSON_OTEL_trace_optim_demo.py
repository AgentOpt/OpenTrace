"""
JSON_OTEL_trace_optim_demo.py - Compact OTEL→Trace→OptoPrimeV2 Demonstration
===============================================================================

This demo shows end-to-end optimization of research agent prompts using:
- OpenTelemetry (OTEL) for span capture → OTLP JSON
- Trace-Graph JSON (TGJ) ingestion → Trace nodes
- GraphPropagator for backward propagation of rich feedback
- OptoPrimeV2 with history-aware prompt generation

FILE STRUCTURE:
==============
1. CONFIGURATION & CONSTANTS (lines 40-120)
   - NUM_OPTIMIZATION_ITERATIONS, TEST_QUERIES
   - OPTIMIZABLE_AGENTS (configurable: ["planner", "executor"] or ["all"])
   - ENABLED_AGENTS, AGENT_PROMPTS
   - JUDGE_METRICS, log_file

2. IMPORTS & INFRASTRUCTURE (lines 122-220)
   - OpenTelemetry setup, InMemory

SpanExporter
   - Trace imports, LLM client initialization

3. AGENT PROMPTS (lines 222-400)
   - plan_prompt(), executor_prompt(), synthesizer_prompt(), judge_prompt()
   - All prompts in one location for easy editing

4. EXTERNAL TOOLS (lines 402-480)
   - wikipedia_search(), wikidata_query()
   - Free APIs (no auth required)

5. OTEL HELPERS (lines 482-560)
   - _set_attr(), flush_otlp_json()
   - Span→OTLP JSON conversion

6. LLM WRAPPERS (lines 562-600)
   - call_llm(), call_llm_json()
   - Unified LLM interface

7. DATA CLASSES (lines 602-680)
   - AgentMetrics, RunOutput

8. GRAPH EXECUTION (lines 682-900)
   - run_graph_once() - main research graph
   - Planner → Executor → Tools → Synthesizer → Judge pipeline

9. OPTIMIZATION PIPELINE (lines 902-1100)
   - ingest_runs_as_trace(), find_last_llm_node(), mode_b_optimize()
   - OTLP→TGJ→Trace→Backward→OptoPrimeV2

10. DISPLAY FUNCTIONS (lines 1102-1300)
    - print_section_header(), print_metrics_table(), print_per_query_scores(),
      print_per_prompt_contribution(), log_json_traces()

11. MAIN FUNCTION (lines 1302-1600)
    - Baseline → Iterative Optimization → Final Results
    - Configurable optimizable agents

USAGE:
=====
python -m examples.JSON_OTEL_trace_optim_demo

Set OPTIMIZABLE_AGENTS = ["all"] to optimize all agents (planner, executor, synthesizer, judge).
Default: ["planner", "executor"] only.

REQUIREMENTS:
============
pip install wikipedia requests opentelemetry-sdk opentelemetry-api
"""

from __future__ import annotations
import os, json, time, random, requests, traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import wikipedia
wikipedia.set_lang("en")
from opentelemetry import trace as oteltrace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opto.utils.llm import LLM
from opto.trace.io.otel_adapter import otlp_traces_to_trace_json
from opto.trace.io.tgj_ingest import ingest_tgj
from opto.trace.propagators import GraphPropagator
from opto.trace.nodes import MessageNode, ParameterNode
from opto.optimizers.optoprime_v2 import OptoPrimeV2

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

# Optimization settings
NUM_OPTIMIZATION_ITERATIONS = 10

# Test queries for evaluation
TEST_QUERIES = [
    "Summarize the causes and key events of the French Revolution.",
    "Give 3 factual relationships about the company Tesla, Inc. (entities & IDs).",
    "Explain what CRISPR is and name 2 notable applications."
]

# Which agents' prompts to optimize
# Options: ["planner", "executor"] (default) or ["all"] (planner, executor, synthesizer, judge)
OPTIMIZABLE_AGENTS = ["planner", "executor"]  # Change to ["all"] for full optimization

# Available agents in the research graph
ENABLED_AGENTS = ["web_researcher", "wikidata_researcher", "synthesizer"]

# Agent prompt templates (filled in section 3)
AGENT_PROMPTS = {}

# Judge metrics (fixed evaluation criteria)
JUDGE_METRICS = ["answer_relevance", "groundedness", "plan_adherence", "execution_efficiency", "logical_consistency"]

log_file = "examples/JSON_OTEL_trace_optim_sample_output.txt"

# ==============================================================================
# 2. IMPORTS & INFRASTRUCTURE
# ==============================================================================

class InMemorySpanExporter(SpanExporter):
    """Simple in-memory span exporter for demo/testing"""
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

# OTEL setup
_mem_exporter = InMemorySpanExporter()
_otel_provider = TracerProvider()
_otel_provider.add_span_processor(SimpleSpanProcessor(_mem_exporter))
oteltrace.set_tracer_provider(_otel_provider)
TRACER = oteltrace.get_tracer("trace-demo")

# LLM client (unified wrapper)
LLM_CLIENT = LLM()

# ==============================================================================
# 3. AGENT PROMPTS
# ==============================================================================

def plan_prompt(user_query: str, enabled_agents: List[str]) -> str:
    """Planner prompt: Break query into steps"""
    agent_list = [f"  • `{a}` – {{'wikidata_researcher':'entity facts/relations','web_researcher':'Wikipedia summaries','synthesizer':'finalize answer'}}" for a in enabled_agents if a in ('wikidata_researcher','web_researcher','synthesizer')]
    agent_enum = " | ".join([a for a in enabled_agents if a in ("web_researcher","wikidata_researcher","synthesizer")])
    return f"""You are the Planner. Break the user's request into JSON steps, one agent per step.
Agents available:
{os.linesep.join(agent_list)}

Return ONLY JSON like: {{"1": {{"agent":"{agent_enum}", "action":"string"}}, "2": {{"agent":"{agent_enum}", "action":"string"}}}}

Guidelines:
- Use `wikidata_researcher` for entity facts/IDs/relations.
- Use `web_researcher` for background/overview.
- End with `synthesizer` to produce final answer.

User query: "{user_query}" """.strip()

def executor_prompt(step_idx: int, plan_step: Dict[str, Any], user_query: str, tail_context: str, enabled_agents: List[str]) -> str:
    """Executor prompt: Route to next agent"""
    goto_enum = " | ".join([a for a in enabled_agents if a in ("web_researcher","wikidata_researcher","synthesizer","planner")])
    return f"""You are the Executor. Respond ONLY with JSON: {{"replan": <true|false>, "goto": "<{goto_enum}>", "reason": "<1 sentence>", "query": "<text for chosen agent>"}}

Context: step={step_idx}, plan={json.dumps(plan_step)}, query="{user_query}", previous="{tail_context}"
Rules: Replan only if blocked; build "query" as standalone instruction for chosen agent.""".strip()

def synthesizer_prompt() -> str:
    """Synthesizer system prompt"""
    return "You are the Synthesizer. Answer concisely using only the given context. If context lacks details, say what's missing."

def judge_prompt() -> str:
    """Judge system prompt"""
    return "You are a strict evaluator. Return JSON with five 0..1 scores and a reasons paragraph."

# Register prompts for easy access
AGENT_PROMPTS = {
    "planner": plan_prompt,
    "executor": executor_prompt,
    "synthesizer": synthesizer_prompt,
    "judge": judge_prompt
}

# ==============================================================================
# 4. EXTERNAL TOOLS
# ==============================================================================

def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return top 3 summaries"""
    hits = wikipedia.search(query, results=3)
    out = []
    for h in hits:
        try:
            s = wikipedia.summary(h, sentences=4, auto_suggest=False, redirect=True)
            out.append(f"### {h}\n{s}")
        except Exception:
            continue
    return "\n\n".join(out) or "No results."

def wikidata_query(query: str) -> str:
    """Query Wikidata with error handling"""
    try:
        r = requests.get("https://www.wikidata.org/w/api.php", params={"action": "wbsearchentities", "format": "json", "language": "en", "search": query[:100], "limit": 5}, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = [f"- {item.get('label', '')}: {item.get('description', '')} ({item.get('id', '')})" for item in data.get("search", [])]
        return "\n".join(results) if results else "No Wikidata entities found."
    except Exception as e:
        return f"Wikidata search temporarily unavailable. Query: {query[:50]}..."

# ==============================================================================
# 5. OTEL HELPERS
# ==============================================================================

def _set_attr(span, key: str, val: Any):
    """Set span attribute as string"""
    try:
        span.set_attribute(key, str(val))
    except Exception:
        pass

def flush_otlp_json() -> Dict[str, Any]:
    """Convert in-memory spans to OTLP JSON payload"""
    spans = _mem_exporter.get_finished_spans()
    def hex_id(x: int, nbytes: int) -> str:
        return f"{x:0{2*nbytes}x}"
    KIND_NAMES = {0: "UNSPECIFIED", 1: "INTERNAL", 2: "SERVER", 3: "CLIENT", 4: "PRODUCER", 5: "CONSUMER"}

    otlp_spans = []
    for s in spans:
        attrs = [{"key": k, "value": {"stringValue": str(v)}} for k, v in (s.attributes or {}).items()]
        kind_val = getattr(s, 'kind', 1)
        if hasattr(kind_val, 'value'): kind_val = kind_val.value
        kind_str = KIND_NAMES.get(kind_val, "INTERNAL")
        otlp_spans.append({"traceId": hex_id(s.context.trace_id, 16), "spanId": hex_id(s.context.span_id, 8), "parentSpanId": (hex_id(s.parent.span_id, 8) if s.parent else ""), "name": s.name, "kind": kind_str, "startTimeUnixNano": int(s.start_time or time.time_ns()), "endTimeUnixNano": int(s.end_time or time.time_ns()), "attributes": attrs})
    payload = {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"scope": {"name": "trace-demo"}, "spans": otlp_spans}]}]}
    _mem_exporter.clear()
    return payload

# ==============================================================================
# 6. LLM WRAPPERS
# ==============================================================================

def call_llm_json(system: str, user: str, response_format_json=True) -> str:
    """Call LLM expecting JSON response"""
    rf = {"type": "json_object"} if response_format_json else None
    resp = LLM_CLIENT(messages=[{"role":"system","content":system}, {"role":"user","content":user}], response_format=rf, max_tokens=800)
    return resp.choices[0].message.content

def call_llm(system: str, user: str) -> str:
    """Call LLM for text response"""
    resp = LLM_CLIENT(messages=[{"role":"system","content":system}, {"role":"user","content":user}], max_tokens=900)
    return resp.choices[0].message.content

# ==============================================================================
# 7. DATA CLASSES
# ==============================================================================

@dataclass
class AgentMetrics:
    """Track per-agent call counts"""
    planner_calls: int = 0
    executor_calls: int = 0
    retrieval_calls: int = 0
    synthesizer_calls: int = 0
    judge_calls: int = 0
    def total_calls(self) -> int:
        return self.planner_calls + self.executor_calls + self.retrieval_calls + self.synthesizer_calls + self.judge_calls

@dataclass
class RunOutput:
    """Single run output with metrics"""
    final_answer: str
    contexts: List[str]
    otlp_payload: Dict[str, Any]
    feedback_text: str
    score: float
    llm_calls: int = 0
    execution_time: float = 0.0
    agent_metrics: Optional[AgentMetrics] = None

    def get_metrics_dict(self) -> Dict[str, float]:
        """Extract individual metrics from feedback_text"""
        try:
            if "[Scores]" in self.feedback_text:
                scores_line = self.feedback_text.split("[Scores]")[1].split(";")[0].strip().strip("[]")
                metrics = [float(x.strip()) for x in scores_line.split(",")]
                return {"answer_relevance": metrics[0] if len(metrics) > 0 else 0.0, "groundedness": metrics[1] if len(metrics) > 1 else 0.0, "plan_adherence": metrics[2] if len(metrics) > 2 else 0.0, "execution_efficiency": metrics[3] if len(metrics) > 3 else 0.0, "logical_consistency": metrics[4] if len(metrics) > 4 else 0.0}
        except:
            pass
        return {"overall": self.score}

# ==============================================================================
# 8. GRAPH EXECUTION
# ==============================================================================

def run_graph_once(user_query: str, overrides: Dict[str,str]) -> RunOutput:
    """Execute research graph once: planner → executor → tools → synthesizer → judge"""
    enabled = ENABLED_AGENTS
    start_time = time.time()
    llm_call_count = 0
    agent_metrics = AgentMetrics()

    # Planner LLM
    with TRACER.start_as_current_span("planner_llm") as sp:
        llm_call_count += 1
        agent_metrics.planner_calls += 1
        planner_txt = overrides.get("planner_prompt") or plan_prompt(user_query, enabled)
        _set_attr(sp, "param.planner_prompt", planner_txt)
        _set_attr(sp, "param.planner_prompt.trainable", "planner" in OPTIMIZABLE_AGENTS or "all" in OPTIMIZABLE_AGENTS)
        _set_attr(sp, "gen_ai.model", "trace-llm")
        _set_attr(sp, "gen_ai.operation", "chat.completions")
        _set_attr(sp, "inputs.gen_ai.prompt", planner_txt)
        raw_plan = call_llm_json(system="You output JSON only.", user=planner_txt)
        try:
            plan = json.loads(raw_plan)
        except json.JSONDecodeError:
            plan = {"1":{"agent":"web_researcher","action":"get background"},"2":{"agent":"wikidata_researcher","action":"get entity facts"},"3":{"agent":"synthesizer","action":"finalize"}}

    messages: List[str] = []
    tail_context = ""
    step_idx = 1
    FINAL = None

    # Execution loop (max 6 steps)
    for _ in range(6):
        plan_step = plan.get(str(step_idx), {}) or {}

        # Executor LLM
        with TRACER.start_as_current_span("executor_llm") as sp:
            llm_call_count += 1
            agent_metrics.executor_calls += 1
            exec_txt = overrides.get("executor_prompt") or executor_prompt(step_idx, plan_step, user_query, tail_context, enabled)
            _set_attr(sp, "param.executor_prompt", exec_txt)
            _set_attr(sp, "param.executor_prompt.trainable", "executor" in OPTIMIZABLE_AGENTS or "all" in OPTIMIZABLE_AGENTS)
            _set_attr(sp, "gen_ai.model", "trace-llm")
            _set_attr(sp, "gen_ai.operation", "chat.completions")
            _set_attr(sp, "inputs.gen_ai.prompt", exec_txt)
            raw = call_llm_json(system="Return ONLY JSON.", user=exec_txt)

        try:
            d = json.loads(raw)
            replan = bool(d.get("replan", False))
            goto = d.get("goto", plan_step.get("agent","synthesizer"))
            agent_query = d.get("query", user_query)
        except Exception:
            replan = False
            goto, agent_query = (plan_step.get("agent","synthesizer"), user_query)

        if replan:
            plan = {"1":{"agent":"web_researcher","action":"collect info"},"2":{"agent":"synthesizer","action":"finalize"}}
            step_idx = 1
            continue

        # Route to tools/synthesizer
        if goto == "web_researcher":
            with TRACER.start_as_current_span("web_research") as sp:
                agent_metrics.retrieval_calls += 1
                _set_attr(sp, "retrieval.query", agent_query)
                out = wikipedia_search(agent_query)
                _set_attr(sp, "retrieval.context", out[:500])
                messages.append(out)
                tail_context = out[-400:]
            step_idx += 1
        elif goto == "wikidata_researcher":
            with TRACER.start_as_current_span("wikidata_research") as sp:
                agent_metrics.retrieval_calls += 1
                _set_attr(sp, "retrieval.query", agent_query)
                out = wikidata_query(agent_query)
                _set_attr(sp, "retrieval.context", out[:500])
                messages.append(out)
                tail_context = out[-400:]
            step_idx += 1
        elif goto == "synthesizer":
            context_blob = "\n\n---\n\n".join(messages[-4:])
            with TRACER.start_as_current_span("synthesizer_llm") as sp:
                llm_call_count += 1
                agent_metrics.synthesizer_calls += 1
                sys = overrides.get("synthesizer_prompt") or synthesizer_prompt()
                user = f"User question: {user_query}\n\nContext:\n{context_blob}"
                _set_attr(sp, "param.synthesizer_prompt", sys)
                _set_attr(sp, "param.synthesizer_prompt.trainable", "synthesizer" in OPTIMIZABLE_AGENTS or "all" in OPTIMIZABLE_AGENTS)
                _set_attr(sp, "gen_ai.model", "trace-llm")
                _set_attr(sp, "gen_ai.operation", "chat.completions")
                _set_attr(sp, "inputs.gen_ai.prompt", user)
                ans = call_llm(sys, user)
                FINAL = ans.strip()
                messages.append(ans)
            break
        else:
            step_idx += 1

    # Judge (rich feedback + scalar score)
    with TRACER.start_as_current_span("judge_llm") as sp:
        llm_call_count += 1
        agent_metrics.judge_calls += 1
        judge_sys = overrides.get("judge_prompt") or judge_prompt()
        context_blob = "\n\n---\n\n".join(messages[-4:])
        judge_user = f"""Evaluate the answer quality for the user query below.
Return ONLY JSON: {{"answer_relevance": <0..1>, "groundedness": <0..1>, "plan_adherence": <0..1>, "execution_efficiency": <0..1>, "logical_consistency": <0..1>, "reasons": "<short detailed explanation>"}}
User query: "{user_query}"
Answer: "{FINAL}"
Context used: {context_blob}""".strip()
        _set_attr(sp, "param.judge_prompt", judge_sys)
        _set_attr(sp, "param.judge_prompt.trainable", "judge" in OPTIMIZABLE_AGENTS or "all" in OPTIMIZABLE_AGENTS)
        _set_attr(sp, "inputs.gen_ai.prompt", judge_user)
        raw = call_llm_json(judge_sys, judge_user)

    try:
        j = json.loads(raw)
    except Exception:
        j = {"answer_relevance":0.5,"groundedness":0.5,"plan_adherence":0.5,"execution_efficiency":0.5,"logical_consistency":0.5,"reasons":"fallback"}

    metrics = [float(j.get(k,0.0)) for k in JUDGE_METRICS]
    score = sum(metrics)/len(metrics)
    feedback_text = f"[Scores] {metrics} ;\nReasons:\n{j.get('reasons','')}".strip()
    otlp = flush_otlp_json()
    execution_time = time.time() - start_time

    return RunOutput(final_answer=FINAL or "", contexts=messages, otlp_payload=otlp, feedback_text=feedback_text, score=score, llm_calls=llm_call_count, execution_time=execution_time, agent_metrics=agent_metrics)

# ==============================================================================
# 9. OPTIMIZATION PIPELINE
# ==============================================================================

def ingest_runs_as_trace(all_runs: List[RunOutput]) -> Tuple[Dict[str,Any], Dict[str,Any], List[Dict[str,Any]]]:
    """OTLP→TGJ→Trace: Return (nodes_map, params_map, per_run_nodes)"""
    per_run_nodes = []
    params: Dict[str, ParameterNode] = {}
    all_nodes: Dict[str, Any] = {}
    for ridx, run in enumerate(all_runs):
        docs = list(otlp_traces_to_trace_json(run.otlp_payload, agent_id_hint=f"demo-{ridx}"))
        for d in docs:
            nodes = ingest_tgj(d)
            per_run_nodes.append(nodes)
            all_nodes.update(nodes)
            for name, n in nodes.items():
                if isinstance(n, ParameterNode) and getattr(n, "trainable", True):
                    params[name] = n
    return all_nodes, params, per_run_nodes

def find_last_llm_node(nodes: Dict[str, Any]) -> Optional[MessageNode]:
    """Find last LLM message node (prefer synthesizer)"""
    last = None
    for n in nodes.values():
        if isinstance(n, MessageNode):
            last = n
            if "synthesizer" in (n.name or ""):
                return n
    return last

def mode_b_optimize(params: Dict[str, ParameterNode], per_run_nodes: List[Dict[str,Any]], all_runs: List[RunOutput]) -> Dict[ParameterNode, Any]:
    """OptoPrimeV2 Mode-B: Generate candidates with history, rank, return best"""
    prop = GraphPropagator()
    targets: List[MessageNode] = []
    for nodes, run in zip(per_run_nodes, all_runs):
        tgt = find_last_llm_node(nodes)
        if tgt is None: continue
        prop.init_feedback(tgt, run.feedback_text)
        tgt.backward(run.feedback_text, propagator=prop, retain_graph=True)
        targets.append(tgt)

    trainables = list(params.values())
    if not trainables:
        print("⚠️  No trainable parameters found in trace.")
        return {}

    opt = OptoPrimeV2(parameters=trainables, llm=LLM_CLIENT, memory_size=3, max_tokens=700)
    opt.zero_feedback()
    for t in targets:
        opt.backward(t, "see attached")

    cand1 = opt.step(bypassing=True)
    cand2 = opt.step(bypassing=True)

    def score_candidate(update_dict: Dict[ParameterNode,Any]) -> Tuple[float,str]:
        var_txt = "\n".join([f"{p.py_name} := {val}" for p,val in update_dict.items()])
        reasons = "\n\n".join([r.feedback_text for r in all_runs])
        judge_user = f"""We tuned prompts below. Score expected quality on 0(min)..1(max) across 5 metrics and give short reasons.
Return ONLY JSON: {{"answer_relevance": <0..1>, "groundedness": <0..1>, "plan_adherence": <0..1>, "execution_efficiency": <0..1>, "logical_consistency": <0..1>, "reasons": "<why this will help>"}}
[Candidate Variables]
{var_txt}
[Observed Failures/Rationale]
{reasons}""".strip()
        raw = call_llm_json("Evaluator", judge_user)
        try:
            j = json.loads(raw)
            metrics = [float(j.get(k,0.0)) for k in JUDGE_METRICS]
            return (sum(metrics)/len(metrics), j.get("reasons",""))
        except Exception:
            return (0.0, "parse_error")

    scores = []
    if cand1: scores.append(("cand1", cand1, *score_candidate(cand1)))
    if cand2: scores.append(("cand2", cand2, *score_candidate(cand2)))
    if not scores: return {}

    scores.sort(key=lambda x: x[2], reverse=True)
    name, update, s, why = scores[0]
    print(f"Selected {name} with judge score={s:.3f}.")
    return update

# ==============================================================================
# 10. DISPLAY FUNCTIONS
# ==============================================================================

def print_section_header(title: str, width: int = 80):
    """Print formatted section header"""
    print(f"\n{'='*width}\n{title:^{width}}\n{'='*width}")

def print_metrics_table(history_scores: List[float], history_llm_calls: List[float], all_runs_history: List[List[RunOutput]], base_score: float):
    """Print comprehensive metrics table (averages across queries)"""
    print(f"\n📊 COMPREHENSIVE METRICS TABLE (Averages Across Queries)\n{'='*100}")
    print(f"{'Iter':<6} {'Score':>7} {'Δ Score':>8} {'LLM':>5} {'Time(s)':>8} {'Plan':>5} {'Exec':>5} {'Retr':>5} {'Synth':>6} {'Judge':>6}\n{'-'*100}")
    if len(all_runs_history) > 0:
        baseline_runs = all_runs_history[0]
        avg_time = sum(r.execution_time for r in baseline_runs) / len(baseline_runs)
        avg_plan = sum(r.agent_metrics.planner_calls for r in baseline_runs if r.agent_metrics) / len(baseline_runs)
        avg_exec = sum(r.agent_metrics.executor_calls for r in baseline_runs if r.agent_metrics) / len(baseline_runs)
        avg_retr = sum(r.agent_metrics.retrieval_calls for r in baseline_runs if r.agent_metrics) / len(baseline_runs)
        avg_synth = sum(r.agent_metrics.synthesizer_calls for r in baseline_runs if r.agent_metrics) / len(baseline_runs)
        avg_judge = sum(r.agent_metrics.judge_calls for r in baseline_runs if r.agent_metrics) / len(baseline_runs)
        print(f"{'Base':<6} {base_score:>7.3f} {'':>8} {history_llm_calls[0]:>5.1f} {avg_time:>8.2f} {avg_plan:>5.1f} {avg_exec:>5.1f} {avg_retr:>5.1f} {avg_synth:>6.1f} {avg_judge:>6.1f}")
    for i in range(1, len(history_scores)):
        delta = history_scores[i] - history_scores[i-1]
        if i < len(all_runs_history):
            iter_runs = all_runs_history[i]
            avg_time = sum(r.execution_time for r in iter_runs) / len(iter_runs)
            avg_plan = sum(r.agent_metrics.planner_calls for r in iter_runs if r.agent_metrics) / len(iter_runs)
            avg_exec = sum(r.agent_metrics.executor_calls for r in iter_runs if r.agent_metrics) / len(iter_runs)
            avg_retr = sum(r.agent_metrics.retrieval_calls for r in iter_runs if r.agent_metrics) / len(iter_runs)
            avg_synth = sum(r.agent_metrics.synthesizer_calls for r in iter_runs if r.agent_metrics) / len(iter_runs)
            avg_judge = sum(r.agent_metrics.judge_calls for r in iter_runs if r.agent_metrics) / len(iter_runs)
        else:
            avg_time = avg_plan = avg_exec = avg_retr = avg_synth = avg_judge = 0
        print(f"{f'{i}'::<6} {history_scores[i]:>7.3f} {delta:>+8.3f} {history_llm_calls[i]:>5.1f} {avg_time:>8.2f} {avg_plan:>5.1f} {avg_exec:>5.1f} {avg_retr:>5.1f} {avg_synth:>6.1f} {avg_judge:>6.1f}")
    print(f"{'='*100}")

def print_per_query_scores(all_runs_history: List[List[RunOutput]], subjects: List[str]):
    """Print per-query score breakdown"""
    print(f"\n📊 PER-QUERY SCORE BREAKDOWN\n{'='*100}")
    for q_idx, query in enumerate(subjects):
        print(f"\n🔍 Query {q_idx + 1}: {query[:60]}...\n{'Iter':<10} {'Score':>8} {'Δ':>8} {'Relevance':>10} {'Grounded':>10} {'Adherence':>10}\n{'-'*80}")
        prev_score = None
        for iter_idx, runs in enumerate(all_runs_history):
            if q_idx < len(runs):
                run = runs[q_idx]
                metrics = run.get_metrics_dict()
                delta_str = '' if prev_score is None else f"{run.score - prev_score:+.3f}"
                iter_name = 'Baseline' if iter_idx == 0 else f'Iter {iter_idx}'
                print(f"{iter_name:<10} {run.score:>8.3f} {delta_str:>8} {metrics.get('answer_relevance', 0):>10.2f} {metrics.get('groundedness', 0):>10.2f} {metrics.get('plan_adherence', 0):>10.2f}")
                prev_score = run.score
    print(f"{'='*100}")

def print_per_prompt_contribution(all_runs_history: List[List[RunOutput]]):
    """Print per-prompt quality metrics (planner vs executor)"""
    print(f"\n📊 PER-PROMPT QUALITY METRICS\n{'='*100}\nThis shows how each trainable prompt contributes to overall quality:\n  • Planner quality → measured by 'plan_adherence' metric\n  • Executor quality → measured by 'execution_efficiency' metric\n  • Overall quality → average of all 5 metrics\n")
    print(f"{'Iter':<10} {'Overall':>8} {'Planner':>10} {'Executor':>10} {'Planner Δ':>12} {'Executor Δ':>12}\n{'-'*100}")
    prev_planner = None
    prev_executor = None
    for iter_idx, runs in enumerate(all_runs_history):
        avg_overall = sum(r.score for r in runs) / len(runs)
        planner_scores = [r.get_metrics_dict().get('plan_adherence', 0) for r in runs]
        executor_scores = [r.get_metrics_dict().get('execution_efficiency', 0) for r in runs]
        avg_planner = sum(planner_scores) / len(planner_scores) if planner_scores else 0
        avg_executor = sum(executor_scores) / len(executor_scores) if executor_scores else 0
        planner_delta = '' if prev_planner is None else f"{avg_planner - prev_planner:+.3f}"
        executor_delta = '' if prev_executor is None else f"{avg_executor - prev_executor:+.3f}"
        iter_name = 'Baseline' if iter_idx == 0 else f'Iter {iter_idx}'
        print(f"{iter_name:<10} {avg_overall:>8.3f} {avg_planner:>10.3f} {avg_executor:>10.3f} {planner_delta:>12} {executor_delta:>12}")
        prev_planner = avg_planner
        prev_executor = avg_executor
    print(f"{'='*100}\n💡 Interpretation:\n   • Planner score improving → better task decomposition and agent selection\n   • Executor score improving → better routing decisions and query formulation\n   • Both contribute to the overall end-to-end quality score")

def log_json_traces(iteration: int, tgj_docs: List[Dict], params: Dict[str, ParameterNode], log_file: str):
    """Log JSON traces and parameter values to file"""
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\nIteration {iteration} - JSON Traces\n{'='*80}\n")
        for idx, doc in enumerate(tgj_docs):
            f.write(f"\n--- TGJ Document {idx+1} ---\n{json.dumps(doc, indent=2)}\n")
        f.write(f"\n--- Trainable Parameters ---\n")
        for name, param in params.items():
            f.write(f"{name}: {getattr(param, 'data', 'N/A')}\n")
        f.write(f"\n")

# ==============================================================================
# 11. MAIN FUNCTION
# ==============================================================================

def main():
    """Main demo: Baseline → Iterative Optimization → Final Results"""
    os.environ.setdefault("TRULENS_OTEL_TRACING", "1")
    global OPTIMIZABLE_AGENTS

    subjects = TEST_QUERIES
    enabled_agents = ENABLED_AGENTS
    if "all" in OPTIMIZABLE_AGENTS:
        OPTIMIZABLE_AGENTS = ["planner", "executor", "synthesizer", "judge"]

    # Clear log file
    with open(log_file, 'w') as f:
        f.write(f"JSON OTEL Trace Optimization Demo - Run Log\n{'='*80}\nOPTIMIZABLE AGENTS:\n{OPTIMIZABLE_AGENTS}\n\nTEST QUERIES:\n{len(subjects)}\n\nITERATIONS:\n{NUM_OPTIMIZATION_ITERATIONS}\n{'='*80}\n")

    print_section_header("JSON OTEL + Trace + OptoPrimeV2 Demo")
    print(f"\n📋 Configuration:\n   • Test queries: {len(subjects)}\n   • Optimization iterations: {NUM_OPTIMIZATION_ITERATIONS}\n   • Enabled agents: {', '.join(enabled_agents)}\n   • Optimizable agents: {', '.join(OPTIMIZABLE_AGENTS)}")

    # BASELINE RUN
    print_section_header("BASELINE (Initial Prompts)")
    overrides: Dict[str,str] = {}
    sample_query = subjects[0]
    initial_planner = plan_prompt(sample_query, enabled_agents)
    initial_executor = executor_prompt(1, {"agent": "web_researcher", "action": "search"}, sample_query, "", enabled_agents)
    print(f"\n📝 COMPLETE Initial Planner Prompt:\n{'-'*80}\n{initial_planner}\n{'-'*80}")
    print(f"\n📝 COMPLETE Initial Executor Prompt:\n{'-'*80}\n{initial_executor}\n{'-'*80}")

    print(f"\n⏳ Running baseline on {len(subjects)} queries...")
    baseline_runs: List[RunOutput] = []
    for idx, q in enumerate(subjects, 1):
        out = run_graph_once(q, overrides)
        baseline_runs.append(out)
        metrics = out.get_metrics_dict()
        am = out.agent_metrics
        print(f"   Query {idx}: score={out.score:.3f} | LLM calls={out.llm_calls} | time={out.execution_time:.2f}s | Relevance={metrics.get('answer_relevance', 0):.2f} | Grounded={metrics.get('groundedness', 0):.2f} | Adherence={metrics.get('plan_adherence', 0):.2f}")
        if am: print(f"            Agent calls: Plan={am.planner_calls} Exec={am.executor_calls} Retr={am.retrieval_calls} Synth={am.synthesizer_calls} Judge={am.judge_calls}")

    base_score, base_llm_calls, base_time = sum(r.score for r in baseline_runs)/len(baseline_runs), sum(r.llm_calls for r in baseline_runs)/len(baseline_runs), sum(r.execution_time for r in baseline_runs)/len(baseline_runs)

    print(f"\n📊 Baseline Summary:\n   • Mean Score: {base_score:.3f}\n   • Avg LLM Calls: {base_llm_calls:.1f}\n   • Avg")
    print(f"\n💡 Score Explanation:\n   The score represents END-TO-END quality of the final answer produced by the entire research pipeline (planner → executor → tools → synthesizer). It's computed by the judge evaluating 5 metrics: answer relevance, groundedness, plan adherence, execution efficiency, and logical consistency.")

    # ITERATIVE OPTIMIZATION
    print_section_header("ITERATIVE OPTIMIZATION")
    history_scores, history_llm_calls, all_runs_history, current_runs = [base_score], [base_llm_calls], [baseline_runs], baseline_runs

    for iteration in range(1, NUM_OPTIMIZATION_ITERATIONS + 1):
        print(f"\n🔄 Optimization Iteration {iteration}/{NUM_OPTIMIZATION_ITERATIONS}\n   {'-'*60}")
        all_nodes, params, per_run_nodes = ingest_runs_as_trace(current_runs)

        # Filter trainable params based on OPTIMIZABLE_AGENTS
        trainables = {name: p for name, p in params.items() if any(name == f"{a}_prompt" for a in OPTIMIZABLE_AGENTS)}

        if not trainables: raise ValueError("   ⚠️  No trainable parameters found; stopping optimization.")

        # Log JSON traces and params
        tgj_docs = [otlp_traces_to_trace_json(run.otlp_payload, agent_id_hint=f"demo-{i}") for i, run in enumerate(current_runs)]
        log_json_traces(iteration, [doc for docs in tgj_docs for doc in docs], trainables, log_file)

        print(f"   📈 Optimizing {OPTIMIZABLE_AGENTS} / {len(trainables)} trainable parameters: {list(trainables.keys())}")

        update = mode_b_optimize(trainables, per_run_nodes, current_runs)

        if not update:
            print("   ⚠️  No updates generated; stopping optimization.")
        else:
            print(f"   ✏️  Applying updates to prompts: {', '.join([p.py_name for p in update.keys()])}")
            # Apply updates
            for p, v in update.items():
                for agent in ["planner", "executor", "synthesizer", "judge"]:
                    if f"{agent}_prompt" in p.py_name:
                        overrides[f"{agent}_prompt"] = v
                        with open(log_file, 'a') as f:
                            f.write(f"Iteration {iteration} - Updated {agent}_prompt:\n{v[:500]}...\n\n")

            # Re-run with updated prompts
            print(f"   ⏳ Validating with {len(subjects)} queries...")
            iteration_runs: List[RunOutput] = []
            for idx, q in enumerate(subjects, 1):
                out = run_graph_once(q, overrides)
                iteration_runs.append(out)
                print(f"      Query {idx}: score={out.score:.3f} | LLM calls={out.llm_calls}")

            iter_score = sum(r.score for r in iteration_runs)/len(iteration_runs)
            iter_llm_calls = sum(r.llm_calls for r in iteration_runs)/len(iteration_runs)
            iter_time = sum(r.execution_time for r in iteration_runs)/len(iteration_runs)
            delta_score = iter_score - history_scores[-1]
            delta_llm = iter_llm_calls - history_llm_calls[-1]

            print(f"\n   📊 Iteration {iteration} Results:\n      • Score: {iter_score:.3f} (Δ {delta_score:+.3f})\n      • Avg LLM Calls: {iter_llm_calls:.1f} (Δ {delta_llm:+.1f})\n      • Avg Time: {iter_time:.2f}s")
            print(f"      {'✅ Improvement detected!' if delta_score > 0 else '⚠️  No improvement in this iteration'}")

            history_scores.append(iter_score)
            history_llm_calls.append(iter_llm_calls)
            all_runs_history.append(iteration_runs)
            current_runs = iteration_runs

    # FINAL RESULTS
    print_section_header("FINAL RESULTS")
    final_score = history_scores[-1]
    total_improvement = final_score - base_score
    pct_improvement = (total_improvement / base_score * 100) if base_score > 0 else 0

    print(f"\n📈 Score Progression:")
    for i, score in enumerate(history_scores):
        if i == 0: print(f"   Baseline:     {score:.3f}")
        else:
            delta = score - history_scores[i-1]
            print(f"   Iteration {i}:  {score:.3f}  (Δ {delta:+.3f})")

    print(f"\n🎯 Overall Improvement:\n   • Initial Score:  {base_score:.3f}\n   • Final Score:    {final_score:.3f}\n   • Improvement:    {total_improvement:+.3f}  ({pct_improvement:+.1f}%)\n   • Efficiency:     {history_llm_calls[0]:.1f} → {history_llm_calls[-1]:.1f} avg LLM calls")
    print(f"\n   {'✅ SUCCESS: OptoPrimeV2 improved prompt quality by ' + f'{pct_improvement:.1f}%!' if total_improvement > 0 else '⚠️  No net improvement achieved'}")

    # Display tables
    print_metrics_table(history_scores, history_llm_calls, all_runs_history, base_score)
    print(f"\n💡 Note: Plan/Exec/Retr/Synth/Judge columns show similar values across iterations because the graph structure (which agents are called) remains constant. Only the prompt quality improves through optimization, leading to better scores without changing the call pattern.")
    print_per_query_scores(all_runs_history, subjects)
    print_per_prompt_contribution(all_runs_history)

    # Show FULL optimized prompts
    print(f"\n📝 COMPLETE Optimized Planner Prompt:\n{'-'*80}\n{overrides.get('planner_prompt', initial_planner)}\n{'-'*80}")
    print(f"\n📝 COMPLETE Optimized Executor Prompt:\n{'-'*80}\n{overrides.get('executor_prompt', initial_executor)}\n{'-'*80}")

    if "synthesizer" in OPTIMIZABLE_AGENTS or "all" in OPTIMIZABLE_AGENTS:
        print(f"\n📝 COMPLETE Optimized Synthesizer Prompt:\n{'-'*80}\n{overrides.get('synthesizer_prompt', synthesizer_prompt())}\n{'-'*80}")
    if "judge" in OPTIMIZABLE_AGENTS or "all" in OPTIMIZABLE_AGENTS:
        print(f"\n📝 COMPLETE Optimized Judge Prompt:\n{'-'*80}\n{overrides.get('judge_prompt', judge_prompt())}\n{'-'*80}")

    print(f"\n{'='*80}\n✅ Demo complete! Logs saved to: {log_file}\n{'='*80}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
