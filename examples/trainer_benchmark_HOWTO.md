# Trace Benchmark Trainer - HOWTO Guide

## Overview

The Trace Benchmark Trainer is a comprehensive system for running optimization algorithms on algorithmic tasks derived from the [LLM4AD (Large Language Models for Algorithm Design)](https://github.com/Opti### Examples of Analysis Workflows

### Quick Task Evaluation
```bash
# Test a new optimization approach on a simple task
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task circle_packing --algos PrioritySearch --ps-steps 1
```

### Algorithm Comparison Study
```bash
# Compare all algorithms on multiple related tasks
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task "optimization_tsp_construct,optimization_knapsack_construct,optimization_set_cover_construct" --algos PrioritySearch,GEPA-Base,GEPA-UCB,GEPA-Beam --ps-steps 2 --gepa-iters 2 --threads 4
```

### Performance Profiling
```bash
# Detailed performance analysis with extended runtime
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task optimization_job_shop_scheduling --algos GEPA-UCB --gepa-iters 5 --gepa-train-bs 2 --threads 4 --eval-kwargs '{"timeout_seconds": 300}'
```oject. This system enables systematic evaluation and comparison of different optimization approaches on diverse algorithmic challenges.

### What it does

The benchmark trainer:
- **Runs optimization algorithms**: Supports PrioritySearch, GEPA-Base, GEPA-UCB, and GEPA-Beam algorithms
- **Evaluates performance**: Uses self-contained task evaluators derived from LLM4AD
- **Provides multiple outputs**: Console display, CSV results, TensorBoard logs for analysis
- **Supports parallel execution**: Multi-task and multi-algorithm runs with timeout protection
- **Enables comparison**: Systematic benchmarking across algorithms and tasks

### Key Features

- **60 benchmark tasks** covering optimization, machine learning, and scientific discovery
- **Timeout protection** prevents hanging on difficult tasks
- **Comprehensive logging** with CSV export and TensorBoard integration
- **Multi-task support** for batch evaluation
- **Self-contained tasks** with no external dependencies

## Quick Start

### Basic Usage

Run a single task with default PrioritySearch algorithm:
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task circle_packing
```

### Command Structure

```bash
python examples/trainers_benchmark.py --tasks <task_directory> --task <task_name(s)> [OPTIONS]
```

## Main Commands and Variations

### 1. Single Task, Single Algorithm

**Basic run with default settings:**
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task circle_packing
```

**With custom PrioritySearch parameters:**
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task circle_packing --ps-steps 2 --ps-batches 2
```

**With timeout and thread control:**
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task circle_packing --threads 4 --eval-kwargs '{"timeout_seconds": 60}'
```

### 2. Single Task, Multiple Algorithms

**Compare all algorithms on one task:**
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task circle_packing --algos PrioritySearch,GEPA-Base,GEPA-UCB,GEPA-Beam
```

**Compare specific algorithms with custom settings:**
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task optimization_knapsack_construct --algos PrioritySearch,GEPA-Beam --ps-steps 2 --gepa-iters 2
```

**Run with detailed GEPA configuration:**
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task online_bin_packing_local --algos GEPA-UCB,GEPA-Beam --gepa-train-bs 2 --gepa-pareto-subset 3 --threads 4
```

### 3. Multiple Tasks, Multiple Algorithms

**Batch evaluation on related tasks:**
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task "circle_packing,optimization_knapsack_construct,optimization_tsp_construct" --algos PrioritySearch,GEPA-Beam
```

**Comprehensive benchmark run:**
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task "circle_packing,machine_learning_acrobot,optimization_knapsack_construct" --algos PrioritySearch,GEPA-UCB,GEPA-Beam --ps-steps 2 --gepa-iters 2 --threads 4
```

**Production benchmark with full configuration:**
```bash
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task "optimization_tsp_construct,optimization_set_cover_construct,optimization_bp_1d_construct" --algos PrioritySearch,GEPA-Base,GEPA-UCB,GEPA-Beam --ps-steps 3 --gepa-iters 2 --gepa-train-bs 2 --threads 4 --eval-kwargs '{"timeout_seconds": 120}'
```

## Output Formats

### 1. Console Display
Real-time progress with:
- Task loading status
- Algorithm execution progress  
- Performance scores and timing
- Error messages and timeouts
- Final summary table

### 2. CSV Export (`./results/results_YYYYMMDD_HHMMSS.csv`)
Structured data with columns:
- `timestamp`: Execution timestamp
- `task`: Task name
- `algo`: Algorithm name
- `parameters`: JSON configuration used
- `time`: Execution time in seconds
- `score`: Final performance score
- `initial_params`: Starting code/parameters
- `final_params`: Optimized code/parameters
- `log_dir`: TensorBoard log directory

### 3. TensorBoard Logs (`./logs/<task>/<algorithm>/<timestamp>/`)
Interactive visualization with:
- Training curves and metrics
- Parameter evolution over time
- Algorithm-specific performance data
- Comparative analysis across runs

**Note**: For multi-task runs, logs are organized as `./logs/<task1>/`, `./logs/<task2>/`, etc.

## Available Benchmark Tasks

The system includes **60 self-contained benchmark tasks** organized by domain:

| Category | Tasks | Examples |
|----------|-------|----------|
| **Optimization - Basic** | 18 tasks | `circle_packing`, `online_bin_packing_local` |
| **Optimization - Constructive** | 15 tasks | `optimization_tsp_construct`, `optimization_knapsack_construct`, `optimization_set_cover_construct` |
| **Optimization - CO-Bench** | 21 tasks | `optimization_travelling_salesman_problem`, `optimization_job_shop_scheduling`, `optimization_container_loading` |
| **Machine Learning** | 5 tasks | `machine_learning_acrobot`, `machine_learning_pendulum`, `machine_learning_moon_lander` |
| **Scientific Discovery** | 1 task | `science_discovery_ode_1d` |

### Task Categories Detail

**Optimization - Basic:**
- `circle_packing`: Pack circles in unit square
- `online_bin_packing_local`: Online bin packing heuristics
- `optimization_admissible_set`: Admissible set priority
- `optimization_online_bin_packing`: Online bin packing strategies

**Optimization - Constructive Heuristics:**
- `optimization_tsp_construct`: TSP node selection
- `optimization_knapsack_construct`: Knapsack item selection  
- `optimization_set_cover_construct`: Set cover subset selection
- `optimization_bp_1d_construct`: 1D bin packing assignment
- `optimization_vrptw_construct`: Vehicle routing with time windows

**Optimization - CO-Bench (Complex):**
- `optimization_travelling_salesman_problem`: Complete TSP solving
- `optimization_job_shop_scheduling`: Job shop scheduling
- `optimization_container_loading`: 3D container packing
- `optimization_maximal_independent_set`: Graph MIS problem
- `optimization_flow_shop_scheduling`: Flow shop optimization

**Machine Learning Control:**
- `machine_learning_acrobot`: Acrobot control optimization
- `machine_learning_pendulum`: Pendulum control strategies
- `machine_learning_moon_lander`: Lunar lander control
- `machine_learning_car_mountain`: Mountain car problem

**Scientific Discovery:**
- `science_discovery_ode_1d`: ODE system discovery

## Command Line Parameters

### Required Parameters
- `--tasks`: Path to benchmark tasks directory (e.g., `examples/benchmark_tasks`)
- `--task`: Task name(s), comma-separated for multiple tasks

### Algorithm Selection
- `--algos`: Comma-separated algorithm list (default: `PrioritySearch`)
  - Options: `PrioritySearch`, `GEPA-Base`, `GEPA-UCB`, `GEPA-Beam`

### Performance Tuning
- `--threads`: Number of threads (default: 2)
- `--optimizer-kwargs`: JSON dict for optimizer configuration
- `--eval-kwargs`: JSON dict for evaluator parameters (e.g., timeout)

### PrioritySearch Parameters
- `--ps-steps`: Search steps (default: 3)
- `--ps-batches`: Batch size (default: 2) 
- `--ps-candidates`: Candidate count (default: 3)
- `--ps-proposals`: Proposal count (default: 3)
- `--ps-mem-update`: Memory update frequency (default: 2)

### GEPA Algorithm Parameters
- `--gepa-iters`: Search iterations (default: 3)
- `--gepa-train-bs`: Training batch size (default: 2)
- `--gepa-merge-every`: Merge frequency (default: 2)
- `--gepa-pareto-subset`: Pareto subset size (default: 3)

## Updating/Re-creating Tasks from LLM4AD

To update the benchmark tasks from the latest LLM4AD repository:

### 1. Clone/Update LLM4AD Repository

```bash
git clone https://github.com/Optima-CityU/LLM4AD.git
cd LLM4AD
git pull  # if already cloned
```

### 2. Convert Tasks to Benchmark Format

**Convert all available tasks:**
```bash
python examples/convert_llm4ad_benchmark.py --llm4ad-root /path/to/LLM4AD --out examples/benchmark_tasks
```

**Convert specific task families:**
```bash
python examples/convert_llm4ad_benchmark.py --llm4ad-root /path/to/LLM4AD --out examples/benchmark_tasks --select "circle_packing,optimization,machine_learning"
```

**Convert only the two core tasks (minimal set):**
```bash
python examples/convert_llm4ad_benchmark.py --llm4ad-root /path/to/LLM4AD --out examples/benchmark_tasks --select "circle_packing,science_discovery/ode_1d"
```

### 3. Validate Converted Tasks

```bash
python examples/trainers_benchmark_tasks_validation.py --tasks examples/benchmark_tasks --task circle_packing
```

### 4. Check Task Inventory

```bash
python -c "import json; print(json.dumps([t['key'] for t in json.load(open('examples/benchmark_tasks/index.json'))], indent=2))"
```

## Troubleshooting

### Common Issues

**Task hangs during execution:**
- Increase timeout: `--eval-kwargs '{"timeout_seconds": 120}'`
- Reduce complexity: Lower `--ps-steps` or `--gepa-iters`

**Out of memory errors:**
- Reduce `--threads` parameter
- Lower batch sizes: `--ps-batches` or `--gepa-train-bs`

**Task not found:**
- Check task name spelling in `examples/benchmark_tasks/index.json`
- Use partial matching: `optimization_tsp` matches `optimization_tsp_construct`

**Import errors:**
- Ensure Trace (opto) is properly installed: `pip install -e .`
- Verify benchmark tasks are properly converted

### Performance Tips

- **Parallel execution**: Use `--threads 4-8` for faster results
- **Batch processing**: Run multiple related tasks together
- **Timeout tuning**: Set appropriate timeouts based on task complexity
- **Algorithm selection**: Start with PrioritySearch for quick results, use GEPA for thorough optimization

## Examples of Analysis Workflows

### Quick Task Evaluation
```bash
# Test a new optimization approach on a simple task
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task circle_packing --algos PrioritySearch --ps-steps 3
```

### Algorithm Comparison Study
```bash
# Compare all algorithms on multiple related tasks
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task "optimization_tsp_construct,optimization_knapsack_construct,optimization_set_cover_construct" --algos PrioritySearch,GEPA-Base,GEPA-UCB,GEPA-Beam --threads 6
```

### Performance Profiling
```bash
# Detailed performance analysis with extended runtime
python examples/trainers_benchmark.py --tasks examples/benchmark_tasks --task optimization_job_shop_scheduling --algos GEPA-UCB --gepa-iters 10 --gepa-train-bs 4 --threads 8 --eval-kwargs '{"timeout_seconds": 300}'
```

The results can then be analyzed using the CSV output for statistical analysis or TensorBoard logs for detailed performance visualization.