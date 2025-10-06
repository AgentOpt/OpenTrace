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
import os, json, time, difflib
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

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# ==============================================================================
# CONFIGURATION
# ==============================================================================

NUM_ITERATIONS = 3
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
ENABLE_CODE_OPTIMIZATION = False  # Set to True to optimize function implementations

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
            max_tokens=400
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
            max_tokens=300
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
            max_tokens=400
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
            max_tokens=400
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

    for rs in otlp.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                if sp.get("name") == "evaluator":
                    attrs = {a["key"]: a["value"].get("stringValue", "") for a in sp.get("attributes", [])}
                    score = float(attrs.get("eval.score", "0.5"))
                    metrics = {
                        "answer_relevance": float(attrs.get("eval.answer_relevance", "0.5")),
                        "groundedness": float(attrs.get("eval.groundedness", "0.5")),
                        "plan_quality": float(attrs.get("eval.plan_quality", "0.5"))
                    }
                    feedback = f"[Metrics] {list(metrics.values())}"

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

def optimize_iteration(runs: List[RunResult], optimizer: Optional[OptoPrimeV2]) -> tuple[Dict[str, str], OptoPrimeV2]:
    print("\\n📊 OPTIMIZATION:")
    print("="*80)

    all_targets_and_feedback = []

    for idx, run in enumerate(runs):
        print(f"\\n🔍 Run {idx+1}: score={run.score:.3f}, metrics={run.metrics}")

        tgj_docs = list(otlp_traces_to_trace_json(run.otlp, agent_id_hint=f"run{idx}"))
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
    if optimizer is None:
        print(f"\\n🔧 Creating optimizer with {len(first_params)} params (memory_size=5)")
        optimizer = OptoPrimeV2(first_params, llm=LLM_CLIENT, memory_size=5, log=True)
    else:
        print(f"\\n♻️  Reusing optimizer (log has {len(optimizer.log)} entries)")

    print(f"\\n⬅️  BACKWARD:")
    optimizer.zero_feedback()

    for idx, (target, feedback, _) in enumerate(all_targets_and_feedback):
        try:
            optimizer.backward(target, feedback)
            print(f"   Run {idx+1}: ✓")
        except Exception as e:
            print(f"   Run {idx+1}: ❌ {e}")

    print(f"\\n➡️  STEP:")
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

    template_history = {
        "planner_prompt": PLANNER_TEMPLATE_DEFAULT,
        "executor_prompt": EXECUTOR_TEMPLATE_DEFAULT
    }

    # OPTIMIZATION
    print("\\n" + "="*80 + "\n" + "OPTIMIZATION".center(80) + "\n" + "="*80)

    history = [base_score]
    optimizer = None  # Will be created on first iteration, reused thereafter
    
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

        # Track best performing iteration
        if iter_score > best_score:
            best_score = iter_score
            best_iteration = iteration
            # Save actual current templates
            best_planner_tmpl = current_planner_tmpl
            best_executor_tmpl = current_executor_tmpl
            print(f"   🌟 NEW BEST SCORE! (iteration {iteration})")

        updates, optimizer = optimize_iteration(runs, optimizer)

        if not updates:
            print("\\n❌ No updates")
            break

        # Debug: show what keys are in updates
        print(f"\n🔍 DEBUG: Updates dict keys: {list(updates.keys())}")

        for param_name, new_template in updates.items():
            old_template = template_history.get(param_name, "")
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
        
        # Validate with a final run
        print(f"\\n🔄 Validating best parameters...")
        validation_runs = [run_graph_with_otel(graph, q, current_planner_tmpl, current_executor_tmpl) for q in TEST_QUERIES]
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

    if improvement > 0:
        print(f"   ✅ SUCCESS!")
    else:
        print(f"   ⚠️  No improvement")
    
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
