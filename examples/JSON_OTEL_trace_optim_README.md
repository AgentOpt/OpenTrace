# OTEL + Trace + OptoPrimeV2 Demo

**End-to-end optimization of research agent prompts using OpenTelemetry tracing, Trace framework, and OptoPrimeV2**

## Quick Start

```bash
# Install dependencies
pip install wikipedia requests opentelemetry-sdk opentelemetry-api

# Set LLM API key (use gpt-5-nano for cost-effective testing)
# Run demo (10 optimization iterations by default)
python examples/otel_trace_optoprime_demo.py
```

## Overview

This demo implements a **mini research graph** (`planner → executor → {Wikipedia, Wikidata} → synthesizer`) that demonstrates:
- **Trainable prompts** via OTEL span attributes
- **10 iterative optimization rounds** with progressive improvement tracking
- **5-metric quality assessment** (relevance, groundedness, adherence, efficiency, consistency)
- **Per-agent performance tracking** (planner, executor, retrieval, synthesizer, judge)
- **Mode-B optimization** using OptoPrimeV2 with history-aware prompt generation

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Baseline  │────>│ Optimization │────>│   Results   │
│   Run       │     │   Loop (10x) │     │   & Table   │
└─────────────┘     └──────────────┘     └─────────────┘
      │                     │                     │
      v                     v                     v
 Capture OTEL          OTLP → TGJ           Display all
 Trainable Params      Backprop             metrics in
 Evaluate (5 metrics)  OptoPrimeV2          compact table
```

**Flow:**
1. **Baseline**: Run queries with initial prompts, capture OTEL traces, evaluate
2. **Iterative Loop** (×10): Convert traces → Backprop feedback → Generate improved prompts → Validate
3. **Results**: Display progression, final prompts, comprehensive metrics table

## Features

| Feature | Description |
|---------|-------------|
| **Iterative Optimization** | 10 configurable rounds showing progressive improvement |
| **Multi-Metric Tracking** | 5 quality metrics + LLM calls + execution time |
| **Per-Agent Breakdown** | Track calls to planner, executor, retrieval, synthesizer, judge |
| **Prompt Evolution** | Display COMPLETE initial vs final prompts (full text) |
| **Comprehensive Table** | All metrics in one view with averages across queries |
| **Per-Query Breakdown** | Individual query scores across all iterations |
| **Per-Prompt Metrics** | Separate quality tracking for planner vs executor prompts |
| **Free APIs** | Wikipedia & Wikidata (only LLM requires credentials) |
| **History-Aware** | OptoPrimeV2 uses memory for better candidates |

## Sample Output

### Baseline
```
Query 1: score=0.683 | LLM calls=4 | time=2.34s
         Relevance=0.70 | Grounded=0.68 | Adherence=0.67
         Agent calls: Plan=1 Exec=2 Retr=2 Synth=1 Judge=1
```

### Final Results
```
📈 Score Progression:
   Baseline:      0.700
   Iteration 1:   0.783  (Δ +0.083)
   Iteration 2:   0.818  (Δ +0.035)
   ...
   Iteration 10:  0.871  (Δ +0.002)

🎯 Overall: +0.171 (+24.4%) improvement
```

### Comprehensive Metrics Table

The demo outputs all metrics in a single table:

```
====================================================================================================
Iter    Score  Δ Score   LLM  Time(s)   Plan  Exec  Retr  Synth  Judge
----------------------------------------------------------------------------------------------------
Base    0.700             4.0     2.31    1.0   2.0   2.0    1.0    1.0
1       0.783   +0.083    4.0     2.28    1.0   2.0   2.0    1.0    1.0
2       0.818   +0.035    4.0     2.25    1.0   2.0   2.0    1.0    1.0
3       0.835   +0.017    4.0     2.23    1.0   2.0   2.0    1.0    1.0
4       0.846   +0.011    4.0     2.22    1.0   2.0   2.0    1.0    1.0
5       0.854   +0.008    4.0     2.21    1.0   2.0   2.0    1.0    1.0
6       0.859   +0.005    4.0     2.20    1.0   2.0   2.0    1.0    1.0
7       0.863   +0.004    4.0     2.19    1.0   2.0   2.0    1.0    1.0
8       0.867   +0.004    4.0     2.18    1.0   2.0   2.0    1.0    1.0
9       0.869   +0.002    4.0     2.18    1.0   2.0   2.0    1.0    1.0
10      0.871   +0.002    4.0     2.17    1.0   2.0   2.0    1.0    1.0
====================================================================================================

💡 Note: Plan/Exec/Retr/Synth/Judge columns show similar values across iterations because
   the graph structure (which agents are called) remains constant. Only the prompt quality
   improves through optimization, leading to better scores without changing the call pattern.
```

**Columns:**
- **Iter**: Iteration number (Base = baseline)
- **Score**: Average quality score (0-1) across 5 metrics (averaged across all queries)
- **Δ Score**: Change from previous iteration
- **LLM**: Total LLM API calls per query
- **Time(s)**: Average execution time per query
- **Plan/Exec/Retr/Synth/Judge**: Average calls per agent type (constant as graph structure doesn't change)

### Per-Query Score Breakdown

The demo also displays individual query progression:

```
📊 PER-QUERY SCORE BREAKDOWN
====================================================================================================

🔍 Query 1: Summarize the causes and key events of the French Revolu...
Iter       Score        Δ  Relevance  Grounded  Adherence
--------------------------------------------------------------------------------
Baseline    0.683              0.70      0.68      0.67
Iter 1      0.765    +0.082     0.78      0.76      0.75
Iter 2      0.802    +0.037     0.82      0.80      0.79
...
Iter 10     0.864    +0.002     0.88      0.86      0.85
```

This shows how each query improves independently across iterations, with 3 of the 5 quality metrics displayed.

### Per-Prompt Quality Metrics

The demo tracks individual prompt contributions:

```
📊 PER-PROMPT QUALITY METRICS
====================================================================================================

This shows how each trainable prompt contributes to overall quality:
  • Planner quality → measured by 'plan_adherence' metric
  • Executor quality → measured by 'execution_efficiency' metric
  • Overall quality → average of all 5 metrics

Iter       Overall   Planner   Executor   Planner Δ   Executor Δ
----------------------------------------------------------------------------------------------------
Baseline     0.700     0.670      0.650
Iter 1       0.783     0.750      0.720       +0.080       +0.070
...
```

This answers "which prompts are being optimized and how much do they contribute?"

## Key Metrics Tracked

### Quality Metrics (per query, 0-1 scale)
1. **Answer Relevance**: How well the answer addresses the query
2. **Groundedness**: Factual accuracy based on retrieved context
3. **Plan Adherence**: How well the execution followed the plan
4. **Execution Efficiency**: Optimal use of agents and steps
5. **Logical Consistency**: Internal coherence of the answer

### Efficiency Metrics
- **LLM Calls**: Total API calls (planner + executors + synthesizer + judge)
- **Execution Time**: End-to-end latency per query
- **Agent Breakdown**: Calls per agent type for optimization analysis

## Files

```
examples/
├── otel_trace_optoprime_demo.py       # Main demo (10 iterations)
├── README_OTEL_DEMO.md                # This file
├── DEMO_OUTPUT_SAMPLE.txt             # Sample full output
└── __init__.py                        # Module marker

tests/
└── test_otel_trace_optoprime_demo.py  # 20 comprehensive tests
```

## Running the Demo

### Standard Run
```bash
python examples/otel_trace_optoprime_demo.py
```

### As Python Module
```bash
python -m examples.otel_trace_optoprime_demo
```

### Customize Iterations
Edit `NUM_OPTIMIZATION_ITERATIONS` in `main()`:
```python
NUM_OPTIMIZATION_ITERATIONS = 5  # Fewer iterations
# or
NUM_OPTIMIZATION_ITERATIONS = 20  # More refinement
```

## Testing

```bash
# Run all 20 tests
python -m pytest tests/test_otel_trace_optoprime_demo.py -v

# Test specific component
python -m pytest tests/test_otel_trace_optoprime_demo.py::TestOTLPToTraceConversion -v

# With coverage
python -m pytest tests/test_otel_trace_optoprime_demo.py --cov=examples.otel_trace_optoprime_demo
```

**Test Coverage:**
- OTEL infrastructure (2 tests)
- OTLP→TGJ→Trace conversion (3 tests)
- Wikipedia/Wikidata tools (3 tests)
- LLM wrappers (2 tests)
- Prompt generation (2 tests)
- Graph execution (1 test)
- Optimization pipeline (2 tests)
- Integration (1 test)
- Edge cases (2 tests)
- Metrics (2 tests)

✅ **All 20 tests passing**

## Technical Details

### Data Classes

**RunOutput**
```python
@dataclass
class RunOutput:
    final_answer: str
    contexts: List[str]
    otlp_payload: Dict[str, Any]
    feedback_text: str
    score: float                        # Average of 5 metrics
    llm_calls: int                      # Total LLM API calls
    execution_time: float               # Seconds
    agent_metrics: Optional[AgentMetrics]  # Per-agent breakdown
```

**AgentMetrics**
```python
@dataclass
class AgentMetrics:
    planner_calls: int
    executor_calls: int
    retrieval_calls: int       # Wikipedia + Wikidata
    synthesizer_calls: int
    judge_calls: int
```

### Key Functions

- `run_graph_once()`: Execute research graph with tracing
- `ingest_runs_as_trace()`: Convert OTLP → TGJ → Trace nodes
- `mode_b_optimize()`: OptoPrimeV2 with history-aware generation
- `print_metrics_table()`: Display comprehensive results table

### OTEL Span Attributes

Trainable parameters are captured as:
```python
span.set_attribute("param.planner_prompt", prompt_text)
span.set_attribute("param.planner_prompt.trainable", "True")
```

The adapter extracts these into ParameterNodes for optimization.

## Optimization Strategy

**Mode-B (History-Aware):**
1. Generate 2 prompt candidates using OptoPrimeV2 memory
2. Judge candidates against aggregated feedback (no re-execution)
3. Select best via Pareto scoring across 5 metrics
4. Validate on query batch
5. Repeat for N iterations

**Why it works:**
- History prevents repeating failed attempts
- Rich feedback (5 metrics + reasons) guides improvements
- Pareto scoring balances trade-offs
- Validation ensures real improvement

## Troubleshooting

**Import Error**: Ensure you're in the repo root
```bash
cd /path/to/Trace
python examples/otel_trace_optoprime_demo.py
```

**LLM API Error**: Check credentials
```bash
echo $OPENAI_API_KEY  # Should print your key
```

**Slow Execution**: Reduce iterations or queries
```python
NUM_OPTIMIZATION_ITERATIONS = 3
subjects = subjects[:1]  # Only 1 query
```

## Performance Expectations

**Baseline** (3 queries, no optimization):
- Score: ~0.65-0.75
- Time: ~2.3s per query
- LLM calls: 4 per query

**After 10 iterations**:
- Score: ~0.85-0.90 (+15-25% improvement)
- Time: ~2.2s per query (slight speedup)
- LLM calls: 4 per query (consistent)

**Total runtime**: ~5-10 minutes (3 queries × 11 runs × ~2.5s + optimization overhead)

## References

- **Trace Framework**: https://github.com/microsoft/Trace
- **OptoPrimeV2**: `opto/optimizers/optoprime_v2.py`
- **OTEL Adapter**: `opto/trace/io/otel_adapter.py`
- **TGJ Ingest**: `opto/trace/io/tgj_ingest.py`
- **OpenTelemetry**: https://opentelemetry.io/

## License

See repository root for license information.
