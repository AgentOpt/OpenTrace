"""
prototype_api_validation.py
===========================

Prototype validation script for the LangGraph OTEL Instrumentation API.
This demonstrates the target API design with:
- Real LangGraph StateGraph
- Real LLM calls via OpenRouter (or StubLLM for testing)

Environment Variables (can be set in .env file):
    OPENROUTER_API_KEY - Your OpenRouter API key
    OPENROUTER_MODEL - Model to use (default: meta-llama/llama-3.1-8b-instruct:free)
    USE_STUB_LLM - Set to "true" to use StubLLM instead of real API calls

Usage:
    # Setup: Copy .env.example to .env and add your API key
    cp .env.example .env
    # Edit .env and set OPENROUTER_API_KEY=sk-or-v1-your-key
    
    # Run with real LLM calls:
    python examples/prototype_api_validation.py

    # Run with stub LLM (no API calls):
    USE_STUB_LLM=true python examples/prototype_api_validation.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Literal, Annotated
import json
import time
import os
import logging
import requests
from pathlib import Path

# Configure logger with line numbers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env in parent directory (NewTrace/) when running from examples/
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Also try current directory
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
USE_STUB_LLM = os.environ.get("USE_STUB_LLM", "").lower() in ("true", "1", "yes")


# ============================================================================
# OPENROUTER LLM CLIENT
# ============================================================================

class OpenRouterLLM:
    """
    LLM client for OpenRouter API.
    
    Compatible with OpenAI-style interface: response.choices[0].message.content
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = base_url or OPENROUTER_BASE_URL
        self.call_count = 0
        self.call_log: List[Dict[str, Any]] = []
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Any:
        """Make an LLM call via OpenRouter."""
        self.call_count += 1
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/microsoft/Trace",
            "X-Title": "Trace OTEL Prototype",
        }
        
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
        }
        
        # Handle response_format for JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            payload["response_format"] = {"type": "json_object"}
        
        # Log the call
        self.call_log.append({
            "call_num": self.call_count,
            "model": payload["model"],
            "messages_count": len(messages),
            "user_message_preview": messages[-1].get("content", "")[:100] if messages else "",
        })
        
        # Make request
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            
            # Return OpenAI-compatible response object
            return self._make_response(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API call failed: {e}")
            # Return fallback response
            return self._make_response({
                "choices": [{
                    "message": {
                        "content": json.dumps({"error": str(e), "fallback": True})
                    }
                }]
            })
    
    def _make_response(self, data: Dict[str, Any]) -> Any:
        """Convert API response to OpenAI-compatible object."""
        class Message:
            def __init__(self, content: str):
                self.content = content
        
        class Choice:
            def __init__(self, message_content: str):
                self.message = Message(message_content)
        
        class Response:
            def __init__(self, choices_data: List[Dict]):
                self.choices = [
                    Choice(c.get("message", {}).get("content", ""))
                    for c in choices_data
                ]
        
        return Response(data.get("choices", [{"message": {"content": ""}}]))


# ============================================================================
# STUB LLM (Deterministic responses for testing without API calls)
# ============================================================================

class StubLLM:
    """
    Deterministic LLM stub for testing without API calls.
    
    Returns predefined responses based on message patterns.
    """
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.call_log: List[Dict[str, Any]] = []
    
    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        self.call_count += 1
        user_msg = messages[-1].get("content", "") if messages else ""
        
        # Log the call
        self.call_log.append({
            "call_num": self.call_count,
            "user_message": user_msg[:200],
            "kwargs": {k: str(v)[:50] for k, v in kwargs.items()},
        })
        
        # Check custom responses first
        for pattern, response in self.responses.items():
            if pattern.lower() in user_msg.lower():
                return self._make_response(response)
        
        # Default responses based on context
        if "plan" in user_msg.lower() or "break" in user_msg.lower():
            return self._make_response(json.dumps({
                "1": {"agent": "researcher", "action": "search", "goal": "gather background"},
                "2": {"agent": "synthesizer", "action": "combine", "goal": "final answer"}
            }))
        
        if "route" in user_msg.lower() or "executor" in user_msg.lower():
            return self._make_response(json.dumps({
                "goto": "synthesizer",
                "query": "synthesize the information"
            }))
        
        if "evaluat" in user_msg.lower():
            # Simulate slight variation in eval scores
            base_score = 0.7 + (self.call_count % 3) * 0.05
            return self._make_response(json.dumps({
                "answer_relevance": round(base_score, 2),
                "groundedness": round(base_score - 0.05, 2),
                "plan_quality": round(base_score + 0.05, 2),
                "reasons": f"Evaluation {self.call_count}: Good structure and content."
            }))
        
        # Default synthesizer response
        return self._make_response(
            f"Synthesized response #{self.call_count}: Based on the available context, "
            "the answer incorporates key facts and maintains logical structure."
        )
    
    def _make_response(self, content: str) -> Any:
        """Create OpenAI-compatible response object."""
        class Message:
            def __init__(self, c):
                self.content = c
        
        class Choice:
            def __init__(self, c):
                self.message = Message(c)
        
        class Response:
            def __init__(self, c):
                self.choices = [Choice(c)]
        
        return Response(content)


def get_llm(use_stub: bool = False) -> Any:
    """Get LLM client based on configuration."""
    if use_stub or USE_STUB_LLM or not OPENROUTER_API_KEY:
        if not use_stub and not USE_STUB_LLM and not OPENROUTER_API_KEY:
            logger.info("No OPENROUTER_API_KEY found. Using StubLLM.")
        return StubLLM()
    return OpenRouterLLM()


# ============================================================================
# LANGGRAPH STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State for the research agent LangGraph."""
    query: str
    plan: Dict[str, Any]
    research_results: List[str]
    answer: str
    evaluation: Dict[str, Any]
    # Template storage (for optimization)
    planner_template: str
    synthesizer_template: str


# ============================================================================
# TELEMETRY SESSION (OTEL span management)
# ============================================================================

class TelemetrySession:
    """
    Manages OTEL tracing session with export capabilities.
    
    This is a prototype implementation demonstrating the target API.
    Real implementation will use opentelemetry SDK.
    """
    
    def __init__(self, service_name: str = "trace-session"):
        self.service_name = service_name
        self._spans: List[Dict[str, Any]] = []
        self._span_counter = 0
        self._trace_id = f"trace_{int(time.time() * 1000)}"
    
    def start_span(self, name: str) -> "SpanContext":
        """Start a new span and return context for attributes."""
        self._span_counter += 1
        span = {
            "traceId": self._trace_id,
            "spanId": f"span_{self._span_counter:04d}",
            "parentSpanId": "",
            "name": name,
            "kind": "INTERNAL",
            "startTimeUnixNano": time.time_ns(),
            "endTimeUnixNano": 0,
            "attributes": {},
        }
        self._spans.append(span)
        return SpanContext(span)
    
    def flush_otlp(self, clear: bool = True) -> Dict[str, Any]:
        """
        Export collected spans to OTLP JSON format.
        
        Compatible with otel_adapter.otlp_traces_to_trace_json().
        """
        # Finalize any open spans
        for span in self._spans:
            if span["endTimeUnixNano"] == 0:
                span["endTimeUnixNano"] = time.time_ns()
        
        # Convert to OTLP format
        otlp_spans = []
        for span in self._spans:
            attrs = [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in span["attributes"].items()
            ]
            otlp_spans.append({
                "traceId": span["traceId"],
                "spanId": span["spanId"],
                "parentSpanId": span["parentSpanId"],
                "name": span["name"],
                "kind": span["kind"],
                "startTimeUnixNano": span["startTimeUnixNano"],
                "endTimeUnixNano": span["endTimeUnixNano"],
                "attributes": attrs,
            })
        
        result = {
            "resourceSpans": [{
                "resource": {"attributes": []},
                "scopeSpans": [{
                    "scope": {"name": self.service_name},
                    "spans": otlp_spans,
                }]
            }]
        }
        
        if clear:
            self._spans.clear()
            self._span_counter = 0
            self._trace_id = f"trace_{int(time.time() * 1000)}"
        
        return result
    
    def get_span_count(self) -> int:
        """Get number of recorded spans."""
        return len(self._spans)


class SpanContext:
    """Context manager for span attribute setting."""
    
    def __init__(self, span: Dict[str, Any]):
        self._span = span
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self._span["attributes"][key] = value
    
    def end(self) -> None:
        """End the span."""
        self._span["endTimeUnixNano"] = time.time_ns()
    
    def __enter__(self) -> "SpanContext":
        return self
    
    def __exit__(self, *args) -> None:
        self.end()


# ============================================================================
# TRACING LLM (Wrapper with dual semantic conventions)
# ============================================================================

class TracingLLM:
    """
    LLM wrapper with OTEL tracing and dual semantic conventions.
    
    Emits spans compatible with both Trace TGJ and Agent Lightning.
    """
    
    def __init__(
        self,
        llm: Any,
        session: TelemetrySession,
        *,
        trainable_keys: Optional[Set[str]] = None,
        provider_name: str = "openrouter",
        emit_genai_child_span: bool = True,
    ):
        self.llm = llm
        self.session = session
        self.trainable_keys = trainable_keys or set()
        self.provider_name = provider_name
        self.emit_genai_child_span = emit_genai_child_span
    
    def _is_trainable(self, key: Optional[str]) -> bool:
        if key is None:
            return False
        if "" in self.trainable_keys:
            return True
        return key in self.trainable_keys
    
    def node_call(
        self,
        *,
        span_name: str,
        template_name: Optional[str] = None,
        template: Optional[str] = None,
        optimizable_key: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **llm_kwargs,
    ) -> str:
        """
        Invoke LLM under an OTEL span with full tracing.
        
        Emits:
        - Parent span: param.*, inputs.* (Trace-compatible)
        - Child span: gen_ai.* (Agent Lightning-compatible)
        """
        messages = messages or []
        
        # Get user prompt for input recording
        user_prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break
        
        # Start parent (node) span
        with self.session.start_span(span_name) as sp:
            # Record Trace-compatible attributes
            if template_name and template is not None:
                sp.set_attribute(f"param.{template_name}", template)
                sp.set_attribute(
                    f"param.{template_name}.trainable",
                    str(self._is_trainable(optimizable_key))
                )
            
            sp.set_attribute("gen_ai.model", getattr(self.llm, "model", "llm"))
            sp.set_attribute("inputs.gen_ai.prompt", user_prompt[:500])  # Truncate for storage
            
            # Emit Agent Lightning-compatible child span
            if self.emit_genai_child_span:
                with self.session.start_span(f"{self.provider_name}.chat.completion") as llm_sp:
                    # Mark as temporal ignore for TGJ stability
                    llm_sp.set_attribute("trace.temporal_ignore", "true")
                    
                    # GenAI semantic conventions
                    llm_sp.set_attribute("gen_ai.operation.name", "chat")
                    llm_sp.set_attribute("gen_ai.provider.name", self.provider_name)
                    llm_sp.set_attribute("gen_ai.request.model", getattr(self.llm, "model", "unknown"))
                    llm_sp.set_attribute("gen_ai.input.messages", json.dumps(messages)[:1000])
                    
                    # Call LLM
                    response = self.llm(messages=messages, **llm_kwargs)
                    content = response.choices[0].message.content
                    
                    # Record output
                    llm_sp.set_attribute("gen_ai.output.messages", json.dumps([
                        {"role": "assistant", "content": content[:500]}
                    ]))
            else:
                # No child span, just call LLM
                response = self.llm(messages=messages, **llm_kwargs)
                content = response.choices[0].message.content
        
        return content


# ============================================================================
# REAL LANGGRAPH NODES
# ============================================================================

# Global references (will be set by instrument_graph)
_TRACING_LLM: Optional[TracingLLM] = None
_TEMPLATES: Dict[str, str] = {}

# Default templates
DEFAULT_PLANNER_TEMPLATE = """You are a planning agent. Given a user query, create a simple plan.

Output a JSON object with numbered steps:
{
    "1": {"action": "research", "goal": "gather information"},
    "2": {"action": "synthesize", "goal": "create final answer"}
}

User query: {query}

Respond with ONLY the JSON object, no other text."""

DEFAULT_SYNTHESIZER_TEMPLATE = """You are a synthesis agent. Given a query and research results, provide a comprehensive answer.

Query: {query}

Research/Context: {context}

Provide a clear, factual answer based on the information provided. Be concise but thorough."""

DEFAULT_EVALUATOR_TEMPLATE = """You are an evaluation agent. Evaluate the quality of an answer on a 0-1 scale.

Query: {query}
Answer: {answer}

Evaluate on these metrics (0-1 scale):
- answer_relevance: How relevant is the answer to the query?
- groundedness: Is the answer factual and well-supported?
- plan_quality: Was the approach/plan effective?

Output a JSON object:
{
    "answer_relevance": 0.8,
    "groundedness": 0.7,
    "plan_quality": 0.9,
    "reasons": "Brief explanation"
}

Respond with ONLY the JSON object."""


def planner_node(state: AgentState) -> Dict[str, Any]:
    """Planner node - creates execution plan."""
    global _TRACING_LLM, _TEMPLATES
    
    template = state.get("planner_template") or _TEMPLATES.get("planner_prompt", DEFAULT_PLANNER_TEMPLATE)
    prompt = template.replace("{query}", state["query"])
    
    response = _TRACING_LLM.node_call(
        span_name="planner",
        template_name="planner_prompt",
        template=template,
        optimizable_key="planner",
        messages=[
            {"role": "system", "content": "You are a planning agent. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500,
    )
    
    try:
        plan = json.loads(response)
    except json.JSONDecodeError:
        plan = {"1": {"action": "synthesize", "goal": "answer directly"}}
    
    return {"plan": plan}


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Researcher node - gathers information (simulated)."""
    global _TRACING_LLM
    
    # In a real implementation, this would call search APIs
    # For now, we simulate with an LLM call
    response = _TRACING_LLM.node_call(
        span_name="researcher",
        messages=[
            {"role": "system", "content": "You are a research assistant. Provide relevant facts about the topic."},
            {"role": "user", "content": f"Provide 3-5 key facts about: {state['query']}"}
        ],
        temperature=0.5,
        max_tokens=500,
    )
    
    return {"research_results": [response]}


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizer node - creates final answer."""
    global _TRACING_LLM, _TEMPLATES
    
    template = state.get("synthesizer_template") or _TEMPLATES.get("synthesizer_prompt", DEFAULT_SYNTHESIZER_TEMPLATE)
    context = "\n".join(state.get("research_results", ["No research results available."]))
    
    prompt = template.replace("{query}", state["query"]).replace("{context}", context)
    
    response = _TRACING_LLM.node_call(
        span_name="synthesizer",
        template_name="synthesizer_prompt",
        template=template,
        optimizable_key="synthesizer",
        messages=[
            {"role": "system", "content": "You are a synthesis agent. Provide comprehensive answers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=800,
    )
    
    return {"answer": response}


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """Evaluator node - assesses answer quality."""
    global _TRACING_LLM
    
    prompt = DEFAULT_EVALUATOR_TEMPLATE.replace("{query}", state["query"]).replace("{answer}", state.get("answer", ""))
    
    response = _TRACING_LLM.node_call(
        span_name="evaluator",
        messages=[
            {"role": "system", "content": "You are an evaluation agent. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300,
    )
    
    try:
        evaluation = json.loads(response)
    except json.JSONDecodeError:
        evaluation = {
            "answer_relevance": 0.5,
            "groundedness": 0.5,
            "plan_quality": 0.5,
            "reasons": "Failed to parse evaluation"
        }
    
    return {"evaluation": evaluation}


def build_research_graph() -> StateGraph:
    """Build a real LangGraph for research tasks."""
    
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("evaluator", evaluator_node)
    
    # Add edges
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "synthesizer")
    graph.add_edge("synthesizer", "evaluator")
    graph.add_edge("evaluator", END)
    
    return graph


# ============================================================================
# INSTRUMENTED GRAPH (Wrapper for LangGraph)
# ============================================================================

@dataclass
class InstrumentedGraph:
    """
    Instrumented LangGraph wrapper.
    
    Provides invoke() method that captures telemetry.
    """
    
    graph: Any  # Compiled LangGraph
    session: TelemetrySession
    tracing_llm: TracingLLM
    templates: Dict[str, str] = field(default_factory=dict)
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute graph and capture telemetry.
        """
        # Ensure query is present
        query = state.get("query", state.get("user_query", ""))
        
        # Build initial state
        initial_state: AgentState = {
            "query": query,
            "plan": {},
            "research_results": [],
            "answer": "",
            "evaluation": {},
            "planner_template": self.templates.get("planner_prompt", ""),
            "synthesizer_template": self.templates.get("synthesizer_prompt", ""),
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Extract evaluation metrics
        evaluation = final_state.get("evaluation", {})
        metrics = {
            "answer_relevance": float(evaluation.get("answer_relevance", 0.5)),
            "groundedness": float(evaluation.get("groundedness", 0.5)),
            "plan_quality": float(evaluation.get("plan_quality", 0.5)),
        }
        score = sum(metrics.values()) / len(metrics)
        reasons = evaluation.get("reasons", "")
        
        # Record evaluation metrics span
        with self.session.start_span("evaluation_metrics") as sp:
            sp.set_attribute("eval.score", str(score))
            for k, v in metrics.items():
                sp.set_attribute(f"eval.{k}", str(v))
            sp.set_attribute("eval.reasons", reasons)
            
            # Emit Agent Lightning-compatible reward span
            with self.session.start_span("agentlightning.annotation") as reward_sp:
                reward_sp.set_attribute("trace.temporal_ignore", "true")
                reward_sp.set_attribute("agentlightning.reward.0.name", "final_score")
                reward_sp.set_attribute("agentlightning.reward.0.value", str(score))
        
        return {
            "answer": final_state.get("answer", ""),
            "plan": final_state.get("plan", {}),
            "research_results": final_state.get("research_results", []),
            "score": score,
            "metrics": metrics,
            "reasons": reasons,
        }


# ============================================================================
# INSTRUMENT_GRAPH() - Main entry point
# ============================================================================

def instrument_graph(
    graph: Optional[StateGraph] = None,
    *,
    service_name: str = "langgraph-agent",
    trainable_keys: Optional[Set[str]] = None,
    llm: Optional[Any] = None,
    initial_templates: Optional[Dict[str, str]] = None,
    emit_genai_child_spans: bool = True,
    use_stub_llm: bool = False,
) -> InstrumentedGraph:
    """
    Wrap a LangGraph with automatic OTEL instrumentation.
    
    Parameters
    ----------
    graph : StateGraph, optional
        The LangGraph to instrument. If None, builds default research graph.
    service_name : str
        OTEL service name for trace identification.
    trainable_keys : Set[str], optional
        Node names whose prompts are trainable.
    llm : Any, optional
        LLM client. Uses OpenRouterLLM or StubLLM based on config.
    initial_templates : Dict[str, str], optional
        Initial prompt templates.
    emit_genai_child_spans : bool
        If True, emit Agent Lightning-compatible child spans.
    use_stub_llm : bool
        If True, force use of StubLLM regardless of config.
    
    Returns
    -------
    InstrumentedGraph
        Wrapper with invoke() and telemetry session.
    """
    global _TRACING_LLM, _TEMPLATES
    
    # Build default graph if none provided
    if graph is None:
        graph = build_research_graph()
    
    # Compile if needed
    if hasattr(graph, 'compile'):
        compiled_graph = graph.compile()
    else:
        compiled_graph = graph
    
    # Create session
    session = TelemetrySession(service_name)
    
    # Get LLM
    if llm is None:
        llm = get_llm(use_stub=use_stub_llm)
    
    # Create TracingLLM
    tracing_llm = TracingLLM(
        llm=llm,
        session=session,
        trainable_keys=trainable_keys or {"planner", "synthesizer"},
        provider_name="openrouter" if isinstance(llm, OpenRouterLLM) else "stub",
        emit_genai_child_span=emit_genai_child_spans,
    )
    
    # Set global references for node functions
    _TRACING_LLM = tracing_llm
    _TEMPLATES = initial_templates or {}
    
    return InstrumentedGraph(
        graph=compiled_graph,
        session=session,
        tracing_llm=tracing_llm,
        templates=initial_templates or {},
    )


# ============================================================================
# OPTIMIZE_LANGGRAPH() - One-liner optimization loop
# ============================================================================

@dataclass
class RunResult:
    """Result of a single graph execution."""
    answer: str
    score: float
    metrics: Dict[str, float]
    otlp: Dict[str, Any]


@dataclass
class OptimizationResult:
    """Result of optimization loop."""
    baseline_score: float
    best_score: float
    best_iteration: int
    final_templates: Dict[str, str]
    score_history: List[float]
    all_runs: List[List[RunResult]]


def optimize_langgraph(
    graph: InstrumentedGraph,
    queries: List[str],
    *,
    iterations: int = 3,
    on_iteration: Optional[callable] = None,
) -> OptimizationResult:
    """
    Run optimization loop on instrumented graph.
    
    Parameters
    ----------
    graph : InstrumentedGraph
        The instrumented graph to optimize.
    queries : List[str]
        Test queries for each iteration.
    iterations : int
        Number of optimization iterations.
    on_iteration : callable, optional
        Callback after each iteration.
    
    Returns
    -------
    OptimizationResult
        Contains scores, history, and final templates.
    """
    score_history = []
    all_runs = []
    best_score = 0.0
    best_iteration = 0
    
    # Baseline
    logger.info("Running baseline...")
    baseline_runs = []
    for i, q in enumerate(queries):
        logger.info(f"Query {i+1}/{len(queries)}: {q[:50]}...")
        result = graph.invoke({"query": q})
        baseline_runs.append(RunResult(
            answer=result["answer"],
            score=result["score"],
            metrics=result["metrics"],
            otlp=graph.session.flush_otlp(),
        ))
        logger.info(f"Score: {result['score']:.3f}")
    
    baseline_score = sum(r.score for r in baseline_runs) / len(baseline_runs)
    score_history.append(baseline_score)
    all_runs.append(baseline_runs)
    best_score = baseline_score
    
    logger.info(f"Baseline average: {baseline_score:.3f}")
    
    # Optimization iterations
    for iteration in range(1, iterations + 1):
        logger.info(f"Iteration {iteration}/{iterations}...")
        runs = []
        for i, q in enumerate(queries):
            logger.info(f"Query {i+1}/{len(queries)}: {q[:50]}...")
            result = graph.invoke({"query": q})
            runs.append(RunResult(
                answer=result["answer"],
                score=result["score"],
                metrics=result["metrics"],
                otlp=graph.session.flush_otlp(),
            ))
            logger.info(f"Score: {result['score']:.3f}")
        
        iter_score = sum(r.score for r in runs) / len(runs)
        score_history.append(iter_score)
        all_runs.append(runs)
        
        if iter_score > best_score:
            best_score = iter_score
            best_iteration = iteration
            logger.info(f"Iteration {iteration} average: {iter_score:.3f} * NEW BEST")
        else:
            logger.info(f"Iteration {iteration} average: {iter_score:.3f}")
        
        if on_iteration:
            on_iteration(iteration, runs, {})
    
    return OptimizationResult(
        baseline_score=baseline_score,
        best_score=best_score,
        best_iteration=best_iteration,
        final_templates=dict(graph.templates),
        score_history=score_history,
        all_runs=all_runs,
    )


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def test_telemetry_session():
    """Test TelemetrySession span capture and OTLP export."""
    logger.info("[TEST] TelemetrySession")
    logger.info("-" * 40)
    
    session = TelemetrySession("test-session")
    
    # Create some spans
    with session.start_span("test_span_1") as sp:
        sp.set_attribute("key1", "value1")
        sp.set_attribute("param.test_prompt", "Hello {x}")
        sp.set_attribute("param.test_prompt.trainable", "true")
    
    with session.start_span("test_span_2") as sp:
        sp.set_attribute("gen_ai.model", "test-model")
        sp.set_attribute("inputs.gen_ai.prompt", "Test prompt")
    
    # Export OTLP
    otlp = session.flush_otlp()
    
    # Validate
    assert "resourceSpans" in otlp
    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 2
    
    # Check attributes
    span1_attrs = {a["key"]: a["value"]["stringValue"] for a in spans[0]["attributes"]}
    assert span1_attrs["key1"] == "value1"
    assert span1_attrs["param.test_prompt"] == "Hello {x}"
    assert span1_attrs["param.test_prompt.trainable"] == "true"
    
    logger.info("[OK] Span capture works")
    logger.info("[OK] OTLP export works")
    logger.info("[OK] Attributes correctly formatted")


def test_tracing_llm():
    """Test TracingLLM with dual semantic conventions."""
    logger.info("[TEST] TracingLLM")
    logger.info("-" * 40)
    
    session = TelemetrySession("test-tracing-llm")
    llm = StubLLM()
    
    tracing_llm = TracingLLM(
        llm=llm,
        session=session,
        trainable_keys={"planner"},
        emit_genai_child_span=True,
    )
    
    # Make a call
    result = tracing_llm.node_call(
        span_name="planner",
        template_name="planner_prompt",
        template="Plan for: {query}",
        optimizable_key="planner",
        messages=[{"role": "user", "content": "Test query"}],
    )
    
    # Get OTLP
    otlp = session.flush_otlp()
    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    
    # Should have 2 spans: parent (planner) + child (openrouter.chat.completion)
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"
    
    # Find spans by name
    parent_span = next(s for s in spans if s["name"] == "planner")
    child_span = next(s for s in spans if "chat.completion" in s["name"])
    
    parent_attrs = {a["key"]: a["value"]["stringValue"] for a in parent_span["attributes"]}
    child_attrs = {a["key"]: a["value"]["stringValue"] for a in child_span["attributes"]}
    
    # Validate parent span (Trace-compatible)
    assert "param.planner_prompt" in parent_attrs
    assert parent_attrs["param.planner_prompt.trainable"] == "True"
    assert "inputs.gen_ai.prompt" in parent_attrs
    
    # Validate child span (Agent Lightning-compatible)
    assert child_attrs["trace.temporal_ignore"] == "true"
    assert child_attrs["gen_ai.operation.name"] == "chat"
    
    logger.info("[OK] Parent span has Trace-compatible attributes")
    logger.info("[OK] Child span has Agent Lightning-compatible attributes")
    logger.info("[OK] trace.temporal_ignore is set on child span")


def test_instrument_graph():
    """Test instrument_graph() function."""
    logger.info("[TEST] instrument_graph()")
    logger.info("-" * 40)
    
    # Instrument with stub LLM
    instrumented = instrument_graph(
        service_name="test-instrument",
        trainable_keys={"planner", "synthesizer"},
        initial_templates={
            "planner_prompt": "Test planner template",
            "synthesizer_prompt": "Test synthesizer template",
        },
        use_stub_llm=True,
    )
    
    assert isinstance(instrumented, InstrumentedGraph)
    assert instrumented.session.service_name == "test-instrument"
    assert "planner" in instrumented.tracing_llm.trainable_keys
    assert "planner_prompt" in instrumented.templates
    
    logger.info("[OK] instrument_graph() creates InstrumentedGraph")
    logger.info("[OK] Session configured correctly")
    logger.info("[OK] TracingLLM configured with trainable_keys")
    logger.info("[OK] Templates initialized")


def test_real_langgraph_with_stub():
    """Test real LangGraph execution with StubLLM."""
    logger.info("[TEST] Real LangGraph with StubLLM")
    logger.info("-" * 40)
    
    instrumented = instrument_graph(
        service_name="test-langgraph",
        trainable_keys={"planner", "synthesizer"},
        use_stub_llm=True,
    )
    
    # Run a query
    result = instrumented.invoke({"query": "What is machine learning?"})
    
    assert "answer" in result
    assert "score" in result
    assert result["score"] > 0
    assert "plan" in result
    
    # Check OTLP
    otlp = instrumented.session.flush_otlp()
    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    
    # Should have spans for planner, researcher, synthesizer, evaluator + child spans + eval metrics
    assert len(spans) >= 4, f"Expected at least 4 spans, got {len(spans)}"
    
    span_names = [s["name"] for s in spans]
    assert "planner" in span_names
    assert "synthesizer" in span_names
    
    logger.info(f"[OK] LangGraph executed successfully")
    logger.info(f"[OK] Generated {len(spans)} spans")
    logger.info(f"[OK] Score: {result['score']:.3f}")
    logger.info(f"[OK] Answer preview: {result['answer'][:100]}...")


def test_optimize_with_stub():
    """Test optimization loop with StubLLM."""
    logger.info("[TEST] Optimization Loop with StubLLM")
    logger.info("-" * 40)
    
    instrumented = instrument_graph(
        trainable_keys={"planner", "synthesizer"},
        use_stub_llm=True,
    )
    
    result = optimize_langgraph(
        instrumented,
        queries=["Query 1", "Query 2"],
        iterations=2,
    )
    
    assert isinstance(result, OptimizationResult)
    assert len(result.score_history) == 3  # baseline + 2 iterations
    assert result.baseline_score > 0
    assert result.best_score >= 0
    assert len(result.all_runs) == 3
    
    logger.info("[OK] optimize_langgraph() returns OptimizationResult")
    logger.info("[OK] Score history tracked correctly")
    logger.info("[OK] Best iteration identified")


# ============================================================================
# TRACE OUTPUT HELPERS
# ============================================================================

def print_trace_summary(spans: List[Dict[str, Any]], max_spans: int = 10) -> None:
    """Print a human-readable summary of OTLP spans."""
    logger.info(f"Total spans: {len(spans)}")
    logger.info(f"Showing first {min(len(spans), max_spans)} spans:")
    
    for i, span in enumerate(spans[:max_spans]):
        name = span.get("name", "unknown")
        span_id = span.get("spanId", "?")
        attrs = {a["key"]: a["value"].get("stringValue", "") for a in span.get("attributes", [])}
        
        # Determine span type
        if "trace.temporal_ignore" in attrs:
            span_type = "[CHILD/GenAI]"
        elif name in ("planner", "researcher", "synthesizer", "evaluator"):
            span_type = "[NODE]"
        elif "eval." in str(attrs):
            span_type = "[EVAL]"
        else:
            span_type = "[SPAN]"
        
        logger.info(f"{i+1}. {span_type} {name} (id: {span_id})")
        
        # Show key attributes
        important_attrs = [
            "param.planner_prompt.trainable",
            "param.synthesizer_prompt.trainable",
            "gen_ai.model",
            "gen_ai.operation.name",
            "gen_ai.provider.name",
            "trace.temporal_ignore",
            "eval.score",
            "eval.answer_relevance",
            "agentlightning.reward.0.value",
        ]
        
        for key in important_attrs:
            if key in attrs:
                value = attrs[key]
                if len(value) > 60:
                    value = value[:60] + "..."
                logger.info(f"   - {key}: {value}")
        
        # Show inputs/outputs preview
        if "inputs.gen_ai.prompt" in attrs:
            prompt = attrs["inputs.gen_ai.prompt"]
            if len(prompt) > 80:
                prompt = prompt[:80] + "..."
            logger.info(f"   - inputs.gen_ai.prompt: {prompt}")


def save_trace_to_file(otlp: Dict[str, Any], filename: str = "trace_output.json") -> Path:
    """Save OTLP trace to JSON file."""
    trace_file = Path(__file__).parent / filename
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(otlp, f, indent=2)
    return trace_file


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("PROTOTYPE API VALIDATION")
    logger.info("LangGraph OTEL Instrumentation API")
    logger.info("=" * 60)
    
    # Show configuration
    logger.info("Configuration:")
    logger.info(f"OPENROUTER_API_KEY: {'[SET]' if OPENROUTER_API_KEY else '[NOT SET]'}")
    logger.info(f"OPENROUTER_MODEL: {OPENROUTER_MODEL}")
    logger.info(f"USE_STUB_LLM: {USE_STUB_LLM}")
    
    use_real_llm = bool(OPENROUTER_API_KEY) and not USE_STUB_LLM
    logger.info(f"Mode: {'REAL LLM (OpenRouter)' if use_real_llm else 'STUB LLM (no API calls)'}")
    
    # Run tests with StubLLM first
    logger.info("=" * 60)
    logger.info("UNIT TESTS (using StubLLM)")
    logger.info("=" * 60)
    
    test_telemetry_session()
    test_tracing_llm()
    test_instrument_graph()
    test_real_langgraph_with_stub()
    test_optimize_with_stub()
    
    logger.info("=" * 60)
    logger.info("ALL UNIT TESTS PASSED [OK]")
    logger.info("=" * 60)
    
    # Demo with real or stub LLM based on config
    logger.info("=" * 60)
    logger.info(f"DEMO: {'Real LLM' if use_real_llm else 'Stub LLM'} Execution")
    logger.info("=" * 60)
    
    logger.info("1. Instrument a LangGraph (ONE function call):")
    logger.info("-" * 40)
    
    instrumented = instrument_graph(
        service_name="demo-api",
        trainable_keys={"planner", "synthesizer"},
        initial_templates={
            "planner_prompt": DEFAULT_PLANNER_TEMPLATE,
            "synthesizer_prompt": DEFAULT_SYNTHESIZER_TEMPLATE,
        },
        use_stub_llm=not use_real_llm,
    )
    logger.info(f"-> Created InstrumentedGraph with session: {instrumented.session.service_name}")
    logger.info(f"-> LLM type: {type(instrumented.tracing_llm.llm).__name__}")
    
    logger.info("2. Single graph execution:")
    logger.info("-" * 40)
    
    test_query = "What are the main causes of climate change?"
    logger.info(f"Query: {test_query}")
    
    result = instrumented.invoke({"query": test_query})
    
    logger.info(f"Score: {result['score']:.3f}")
    logger.info(f"Metrics: {result['metrics']}")
    logger.info(f"Answer preview: {result['answer'][:200]}...")
    
    # Export OTLP
    otlp = instrumented.session.flush_otlp()
    spans = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    logger.info(f"Spans generated: {len(spans)}")
    
    # Save trace to file
    trace_file = Path(__file__).parent / "trace_output.json"
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(otlp, f, indent=2)
    logger.info(f"Trace saved to: {trace_file}")
    
    # Display trace summary
    logger.info("3. OTLP Trace Output (Single Execution):")
    logger.info("-" * 40)
    print_trace_summary(spans)
    
    logger.info("4. Run optimization loop:")
    logger.info("-" * 40)
    
    queries = [
        "What is artificial intelligence?",
        "Explain quantum computing basics.",
    ]
    
    opt_result = optimize_langgraph(
        instrumented,
        queries=queries,
        iterations=2,
    )
    
    logger.info("Results:")
    logger.info(f"Baseline: {opt_result.baseline_score:.3f}")
    logger.info(f"Best: {opt_result.best_score:.3f} (iteration {opt_result.best_iteration})")
    logger.info(f"History: {[f'{s:.3f}' for s in opt_result.score_history]}")
    
    # Save all optimization traces
    logger.info("5. Optimization Traces:")
    logger.info("-" * 40)
    all_traces = []
    for iter_idx, runs in enumerate(opt_result.all_runs):
        iter_name = "baseline" if iter_idx == 0 else f"iteration_{iter_idx}"
        for run_idx, run in enumerate(runs):
            all_traces.append({
                "iteration": iter_name,
                "query_index": run_idx,
                "score": run.score,
                "otlp": run.otlp,
            })
    
    # Save all traces to file
    all_traces_file = Path(__file__).parent / "optimization_traces.json"
    with open(all_traces_file, "w", encoding="utf-8") as f:
        json.dump(all_traces, f, indent=2)
    logger.info(f"All optimization traces saved to: {all_traces_file}")
    logger.info(f"Total trace files: {len(all_traces)} (baseline + {len(opt_result.all_runs)-1} iterations x {len(queries)} queries)")
    
    logger.info("=" * 60)
    logger.info("DEMO COMPLETE [OK]")
    logger.info("=" * 60)
    
    logger.info("""
SUMMARY: The prototype demonstrates:

1. instrument_graph() - ONE function call to add OTEL instrumentation
2. Real LangGraph - StateGraph with planner/researcher/synthesizer/evaluator
3. OpenRouter LLM - Real API calls (or StubLLM for testing)
4. TelemetrySession - Unified span management with OTLP export
5. TracingLLM - Dual semantic conventions (Trace + Agent Lightning)
6. optimize_langgraph() - ONE function call for optimization loop

Environment Variables:
OPENROUTER_API_KEY - Set this to enable real LLM calls
OPENROUTER_MODEL - Model to use (default: meta-llama/llama-3.1-8b-instruct:free)
USE_STUB_LLM - Set to "true" to force stub mode
    """)


if __name__ == "__main__":
    main()
