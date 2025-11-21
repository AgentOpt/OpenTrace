# LangGraph + OTEL Trace Optimization Demo

**End-to-end optimization of LangGraph research agent prompts using OpenTelemetry tracing and OptoPrime**

## Quick Start

```bash
# Install dependencies
pip install wikipedia requests opentelemetry-sdk opentelemetry-api langgraph

# Set LLM API key
export OPENAI_API_KEY=your_key_here  # or the LLM calls

# Run demo (3 optimization iterations by default)
python examples/JSON_OTEL_trace_optim_demo_LANGGRAPH.py
```

## Overview

This demo implements a **LangGraph-based research agent** using proper StateGraph architecture with Command-based flow control. It demonstrates:
- **LangGraph StateGraph** with proper node registration and compilation
- **Dual retrieval agents**: Wikipedia (web_researcher) + Wikidata (wikidata_researcher)
- **OTEL tracing** with trainable prompt parameters
- **Iterative optimization** using OptoPrime with best-iteration restoration
- **Colored diff visualization** showing prompt evolution
- **Sequential span linking** for proper trace graph connectivity

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Baseline  │────>│ Optimization │────>│   Results   │
│   Run       │     │   Loop (5x)  │     │   & Table   │
└─────────────┘     └──────────────┘     └─────────────┘
      │                     │                     │
      v                     v                     v
 Capture OTEL          OTLP → TGJ           Display all
 Trainable Params      Backprop             metrics in
 Evaluate (3 metrics)  OptoPrimeV2          compact table
```

**Flow:**
1. **Baseline**: Run test queries with default prompts, capture OTEL traces
2. **Optimization Loop** (×N): 
   - Run queries with current prompts
   - Track score and save if best
   - Convert OTLP → TraceJSON → Trace nodes
   - Backpropagate feedback to parameters
   - Generate improved prompts via OptoPrime
3. **Restoration**: Restore prompts from best-scoring iteration
4. **Results**: Show progression, validate best score, display colored diffs

## Features

| Feature | Description |
|---------|-------------|
| **LangGraph StateGraph** | Proper Command-based flow control with node registration |
| **Dual Retrieval** | Wikipedia (general knowledge) + Wikidata (structured entity data) |
| **OTEL Tracing** | OpenTelemetry spans with trainable parameter attributes |
| **Prompt Optimization** | Optimizes planner, executor, and synthesizer prompts |
| **Code Optimization** | Experimental hot-patching of function implementations |
| **OptoPrime** | Gradient-free optimization with memory |
| **Best Iteration Tracking** | Automatically saves and restores best-performing prompts |
| **Colored Diffs** | Visual comparison of original vs optimized prompts |
| **Sequential Linking** | Proper span parent-child relationships for graph connectivity |
| **Parameter Mapping** | Handles numeric indices → semantic names (0→planner_prompt, 1→executor_prompt) |
| **Configurable** | Adjustable iterations, test queries, and optimizable components |

## Key Components

### Agents (LangGraph Nodes)
1. **planner_node**: Analyzes query, creates multi-step execution plan
2. **executor_node**: Routes to appropriate researcher or synthesizer
3. **web_researcher_node**: Searches Wikipedia for general knowledge
4. **wikidata_researcher_node**: Queries Wikidata for entity facts/IDs
5. **synthesizer_node**: Combines contexts into final answer
6. **evaluator_node**: Scores answer quality (0-1 scale)

### Optimizable Parameters
- **planner_prompt**: Instructions for the planning agent
- **executor_prompt**: Instructions for the executor/routing agent  
- **synthesizer_prompt**: Instructions for the answer synthesis agent
- **__code_<node>**: Function implementations for all nodes (experimental)
- Configured via `OPTIMIZABLE = ["planner", "executor", "synthesizer", ""]`
- Code optimization enabled via `ENABLE_CODE_OPTIMIZATION = True`

### Test Queries (Default)
1. "Summarize the causes and key events of the French Revolution."
2. "Give 3 factual relationships about Tesla, Inc. with entity IDs."
3. "What is the Wikidata ID for CRISPR and list 2 related entities?"

## Sample Output

### Baseline Run
```
================================================================================
                                   BASELINE                                    
================================================================================

Baseline: 0.500
  Q1: 0.367 | {'answer_relevance': 0.4, 'groundedness': 0.2, 'plan_quality': 0.5}
  Q2: 0.533 | {'answer_relevance': 0.6, 'groundedness': 0.5, 'plan_quality': 0.5}
  Q3: 0.900 | {'answer_relevance': 1.0, 'groundedness': 0.8, 'plan_quality': 0.9}
```

### Optimization Iterations
```
================================================================================
                          Iteration 1/5                           
================================================================================

Current: 0.511
   🌟 NEW BEST SCORE! (iteration 1)

📊 OPTIMIZATION:
================================================================================

🔍 Run 1: score=0.367, metrics={'answer_relevance': 0.2, 'groundedness': 0.1, 'plan_quality': 0.8}
   Reachability: planner_prompt:0=✅, __code_planner:0=✅

🔍 Run 2: score=0.267, metrics={'answer_relevance': 0.2, 'groundedness': 0.1, 'plan_quality': 0.5}
   Reachability: planner_prompt:0=✅, __code_planner:0=✅

🔍 DYNAMIC Parameter mapping:
   run0/0/planner_prompt:0 -> planner_prompt
   run0/0/__code_planner:0 -> __code_planner
   run0/0/executor_prompt:0 -> executor_prompt
   run0/0/__code_executor:0 -> __code_executor

🔍 DEBUG: Updates dict keys: ['planner_prompt', '__code_planner', 'executor_prompt', '__code_executor', '__code_web_researcher', '__code_wikidata_researcher', '__code_synthesizer', '__code_evaluator']

📝 DIFF for planner_prompt:
================================================================================
--- old
+++ new
@@ -1,4 +1,4 @@
-You are the Planner. Break the user's request into JSON steps.
+You are the Planner. Break the user's request into JSON steps while considering context availability constraints.
   Ensure analysis comprehensively uncovers backgrounds, facts, relationships, and conclusions.
================================================================================
   ⤷ apply __code_planner: patched
   ✅ Updated current_executor_tmpl
```

### Best Iteration Restoration
```
================================================================================
                           RESTORING BEST PARAMETERS                            
================================================================================

🏆 Best score: 0.778 from iteration 1
   Restoring templates from iteration 1...

🔄 Validating best parameters...
   Validation score: 0.578
   ⚠️  Warning: Validation score differs from recorded best by 0.200
```

### Final Results
```
================================================================================
                                    RESULTS                                     
================================================================================

📈 Progression:
   Baseline    : 0.500 
   Iter 1      : 0.511 (Δ +0.011) 🌟 BEST
   Iter 2      : 0.767 (Δ +0.256) 🌟 BEST
   Iter 3      : 0.567 (Δ -0.200)
   Iter 4      : 0.644 (Δ +0.077)
   Iter 5      : 0.500 (Δ -0.144)

🎯 Overall: 0.500 → 0.767 (+0.267, +53.4%)
   Best iteration: 2
   ✅ SUCCESS!
```

### Colored Diffs (Final Optimized vs Original)
```
================================================================================
                     FINAL OPTIMIZED PROMPTS (vs Original)                      
================================================================================

────────────────────────────────────────────────────────────────────────────────
🔵 PLANNER PROMPT (Final Optimized vs Original)
────────────────────────────────────────────────────────────────────────────────

📝 DIFF for planner_prompt:
================================================================================
--- old
+++ new
@@ -1,10 +1,12 @@
-You are the Planner. Analyze the user query and create a step-by-step plan.
+You are the Strategic Planner. Thoroughly analyze the user query and create
+a comprehensive, step-by-step execution plan with clear goals.
 
 Available agents:
   • web_researcher - General knowledge from Wikipedia
   • wikidata_researcher - Entity facts, IDs, and structured relationships
 
-Return JSON: {{"1": {{"agent":"...", "action":"...", "goal":"..."}}...}}
+Return JSON with numbered steps:
+{{"1": {{"agent":"web_researcher|wikidata_researcher", "action":"...", "goal":"..."}}, "2": {{"agent":"synthesizer", "action":"...", "goal":"..."}}}}
================================================================================
```

## Configuration Options

### Iterations
Edit `NUM_ITERATIONS` at the top of the file:
```python
NUM_ITERATIONS = 3  # Default
# NUM_ITERATIONS = 5  # More refinement
# NUM_ITERATIONS = 1  # Quick test
```

### Test Queries
Edit `TEST_QUERIES` list:
```python
TEST_QUERIES = [
    "Your custom query 1",
    "Your custom query 2",
    # Add more queries...
]
```

### Optimizable Components
Edit `OPTIMIZABLE` list to control which prompts are optimized:
```python
OPTIMIZABLE = ["planner", "executor", "synthesizer", ""]  # All prompts + code
# OPTIMIZABLE = ["planner", "executor"]    # Only planner and executor prompts
# OPTIMIZABLE = ["__code"]                 # Only code optimization
# OPTIMIZABLE = []                         # No optimization (baseline only)
```

### Code Optimization
Enable experimental code optimization (hot-patches function implementations):
```python
ENABLE_CODE_OPTIMIZATION = True   # Optimize function code
# ENABLE_CODE_OPTIMIZATION = False  # Prompts only (safer)
```

### Debug Output
The demo includes debug output showing:
- Parameter name mapping (numeric indices → semantic names)
- Updates dict keys (which prompts are being updated)
- Template update confirmations

To disable, remove or comment out the debug print statements in `optimize_iteration()` and the main loop.

## Key Metrics Tracked

### Quality Metrics
- **answer_relevance**: How well the answer addresses the query (0-1)
- **groundedness**: Answer accuracy based on retrieved context (0-1)
- **plan_quality**: Effectiveness of the execution plan (0-1)
- **Score**: Average of all metrics (0-1 scale) from evaluator_node
- Stored per query, averaged across queries per iteration

### Output Data
- **Final Answer**: Generated response from synthesizer
- **Contexts**: Retrieved information from web/wikidata researchers
- **Feedback**: Evaluation feedback text
- **Plan**: Multi-step execution plan from planner
- **Metrics**: Dictionary of evaluation metrics

## Files

```
examples/
├── JSON_OTEL_trace_optim_demo_LANGGRAPH.py           # Main demo (LangGraph + OTEL)
├── JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE.py  # Simplified OTEL variant
├── JSON_OTEL_trace_optim_demo_LANGGRAPH_TIMESPAN.py     # Alternative OTEL approach
├── JSON_OTEL_trace_optim_README.md                   # This file
└── __init__.py                                        # Module marker
```

### Demo Variants

The repository includes **three versions** of the demo exploring different OTEL tracing approaches:

1. **JSON_OTEL_trace_optim_demo_LANGGRAPH.py** (Main)
   - OTEL tracing code embedded directly in node functions
   - Each node manages its own span creation and parameter emission
   - Most explicit and educational approach
   
2. **JSON_OTEL_trace_optim_demo_LANGGRAPH_SPANOUTNODE.py**
   - Simplified OTEL approach with `TracingLLM` wrapper
   - Moves span management outside node code into helper class
   - Cleaner node implementations, centralized tracing logic
   - **Recommended for production use**
   
3. **JSON_OTEL_trace_optim_demo_LANGGRAPH_TIMESPAN.py**
   - Alternative time-based span approach
   - Different span lifecycle management strategy
   - Experimental variation for comparison

**All variants** support the same optimization features (prompt + code) and produce equivalent results. The differences are purely in how OTEL spans are created and managed.

## Running the Demo

### Standard Run
```bash
python examples/JSON_OTEL_trace_optim_demo_LANGGRAPH.py
```

### As Python Module
```bash
python -m examples.JSON_OTEL_trace_optim_demo_LANGGRAPH
```

### Expected Runtime
- **3 queries × 6 iterations** (baseline + 5 optimization rounds)
- **~2-5 seconds per query** (depends on LLM latency)
- **Total: ~3-6 minutes**
- Code optimization adds minimal overhead (<5%)

## Technical Details

### Data Classes

**State** (LangGraph State)
```python
@dataclass
class State:
    user_query: str
    plan: Dict[str, Dict[str, Any]]
    current_step: int
    agent_query: str
    contexts: List[str]
    final_answer: str
    planner_template: str        # Current planner prompt
    executor_template: str       # Current executor prompt
    synthesizer_template: str    # Current synthesizer prompt
    prev_span_id: Optional[str]  # For sequential span linking
```

**RunResult**
```python
@dataclass
class RunResult:
    answer: str
    otlp: Dict[str, Any]       # OTLP trace payload
    feedback: str               # Evaluation feedback
    score: float                # Evaluation score (0-1)
    metrics: Dict[str, float]   # Additional metrics
    plan: Dict[str, Any]        # Execution plan
```

### Key Functions

- `build_graph()`: Constructs LangGraph StateGraph with all nodes
- `run_graph_with_otel()`: Executes graph and captures OTEL traces
- `optimize_iteration()`: Converts OTLP → TraceJSON → Trace nodes, runs OptoPrime
- `show_prompt_diff()`: Displays colored unified diff between prompts
- `flush_otlp()`: Extracts OTLP payload from InMemorySpanExporter

### OTEL Span Attributes

Trainable parameters are captured as:

**Prompts:**
```python
span.set_attribute("param.planner_prompt", prompt_text)
span.set_attribute("param.planner_prompt.trainable", "true")
```

**Code (experimental):**
```python
import inspect
source = inspect.getsource(planner_node)
span.set_attribute("param.__code_planner", source)
span.set_attribute("param.__code_planner.trainable", "true")
```

The opto adapter extracts these as ParameterNodes for optimization. Code parameters enable the optimizer to modify function implementations via hot-patching.

### Dynamic Parameter Discovery

**Challenge**: Automatically discover all trainable parameters without hardcoding.

**Solution**: Extract semantic names from OTEL parameter node names:
```python
# Automatically discovered from spans:
# run0/0/planner_prompt:0 -> planner_prompt
# run0/0/__code_planner:0 -> __code_planner
# run0/0/executor_prompt:0 -> executor_prompt
```

This enables:
- No hardcoded parameter lists needed
- Automatic adaptation to any agent configuration
- Support for both prompt and code parameters
- Works with any number of optimizable components

## Optimization Strategy

**OptoPrime with Best Iteration Tracking:**
1. **Baseline**: Run with default prompts/code, establish baseline score
2. **Iterative Loop**:
   - Run queries with current prompts and code
   - Calculate iteration score (average across queries)
   - **If score improves**: Save current prompts and code as best
   - Convert OTLP → TraceJSON → Trace nodes
   - Backpropagate feedback to parameters (prompts + code)
   - Generate improved prompts/code via OptoPrime.step()
   - Apply updates: prompts (template strings), code (hot-patch functions)
   - Update current templates and functions for next iteration
3. **Restoration**: Restore prompts and code from best-scoring iteration
4. **Display**: Show progression and colored diffs for all changes

**Why it works:**
- Tracks best across all iterations (handles score fluctuations)
- Restores optimal prompts even if later iterations degrade
- Validation catches non-reproducible scores
- Colored diffs show actual prompt improvements

## Troubleshooting

### Import Error
Ensure you're in the repo root:
```bash
cd /path/to/Trace
python examples/JSON_OTEL_trace_optim_demo_LANGGRAPH.py
```

### LLM API Error
Check credentials:
```bash
echo $OPENAI_API_KEY  # Should print your key
# OR
cat OAI_CONFIG_LIST   # Should show valid config
```

Configure if needed:
```bash
export OPENAI_API_KEY=sk-...
```

### Missing Dependencies
```bash
pip install wikipedia requests opentelemetry-sdk opentelemetry-api langgraph
```

### Slow Execution
Reduce iterations or queries:
```python
NUM_ITERATIONS = 1  # Quick test
TEST_QUERIES = TEST_QUERIES[:1]  # Single query
```

### No Optimization Occurring
Check `OPTIMIZABLE` configuration:
```python
OPTIMIZABLE = ["planner", "executor", ""]  # Should include agent names
```

### Validation Score Differs from Best
This is **normal** and expected due to:
- LLM non-determinism (even with same prompts)
- Different test queries in validation
- Small sample size (3 queries)
- Score fluctuation typically <0.1

**Warning threshold**: 0.05 (shown if diff > 5%)

### "NO CHANGE" in Final Diffs
This indicates prompts weren't actually updated. Check debug output:
```
🔍 DEBUG: Parameter mapping:  # Shows param names
🔍 DEBUG: Updates dict keys:  # Shows which keys in updates
   ✅ Updated current_planner_tmpl  # Confirms updates
```

If debug shows updates but diff shows no change, the mapping might be wrong.

## Known Limitations

### Score Variability
- LLM responses are non-deterministic
- Scores can fluctuate ±0.1-0.2 between runs
- Best iteration tracking mitigates this
- Validation score may differ from recorded best score

### Evaluation Limitations
- Uses 3 metrics (answer_relevance, groundedness, plan_quality)
- Evaluator prompt not currently optimized (fixed evaluation criteria)
- No ground truth comparison for automatic validation
- Score interpretation depends on evaluator LLM quality and judgment

### Graph Structure
- Fixed graph topology (can't optimize which agents to call)
- All queries follow same agent sequence
- No conditional branching based on query type

### Optimization
- Fresh optimizer per iteration (no cross-iteration memory)
- No automatic hyperparameter tuning
- Requires manual configuration of iterations/queries
- No early stopping on convergence

### Retrieval
- Wikipedia: Simple search (no advanced ranking)
- Wikidata: Basic entity search (no SPARQL queries)
- No caching (repeated queries re-fetch)
- Network errors cause iteration failures

## Performance Expectations

**Baseline** (3 queries, default prompts):
- Score: ~0.50-0.60 (depends on LLM and queries)
- Time: ~2-4s per query
- Varies significantly based on query complexity

**After 5 iterations**:
- Score: ~0.70-0.80 (+40-60% improvement typical)
- Time: Similar or slightly faster
- Best iteration usually 1-3 (not always the last)
- Code optimization can add 10-15% improvement over prompts alone

**Score improvements vary widely** based on:
- Initial prompt quality
- Query difficulty
- LLM capability
- Random seed/temperature

**Note**: High initial scores (>0.7) leave less room for improvement.

## Differences from Other Demos

This demo differs from other OTEL optimization examples in the repo:

| Feature | This Demo | Other Demos |
|---------|-----------|-------------|
| **Framework** | LangGraph StateGraph | Custom graph or simpler flow |
| **Flow Control** | Command-based routing | Direct function calls |
| **Retrieval** | Wikipedia + Wikidata | Wikipedia only or none |
| **Score Tracking** | Best iteration with restoration | Final iteration only |
| **Diff Display** | Colored unified diff | Text comparison or none |
| **Span Linking** | Sequential parent-child | Simple tracing |
| **Iterations** | 5 (configurable) | 10 (various) |
| **Metrics** | 3 detailed metrics (relevance, groundedness, plan) | Various |
| **Code Optimization** | Yes (experimental) | No |

## References

- **Trace Framework**: https://github.com/microsoft/Trace
- **OptoPrime**: `opto/optimizers/optoprime.py`
- **OTEL Adapter**: `opto/trace/io/otel_adapter.py`
- **TGJ Ingest**: `opto/trace/io/tgj_ingest.py`
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **OpenTelemetry**: https://opentelemetry.io/

## License

See repository root for license information.
