#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''validate_benchmark_tasks.py
Quick validation script to test all benchmark tasks with minimal resources.

This script:
1. Tests if each task can be loaded and built
2. Runs 1 optimization step with PrioritySearch 
3. Times each task with a short timeout
4. Identifies which tasks work and can be optimized quickly
'''

import os
import sys
import time
import signal
import traceback
import importlib.util
import threading
import argparse
from pathlib import Path
from contextlib import contextmanager

# Add current directory to path for imports
sys.path.append('.')
sys.path.append('./examples/benchmark_tasks')

from opto.features.priority_search import PrioritySearch as SearchAlgorithm
from opto import trainer


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def run_with_timeout(task_func, timeout_seconds=5):
    """Run a task function with timeout using threading."""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = task_func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Timeout occurred - we can't actually kill the thread, but we can return timeout error
        raise TimeoutError(f"Task timed out after {timeout_seconds} seconds")
    
    if exception[0] is not None:
        raise exception[0]
    
    return result[0]


@contextmanager
def timeout_context(seconds):
    """Context manager for timeout using threading (fallback)"""
    def timeout_handler():
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def load_benchmark_task(task_dir: Path):
    '''Load an benchmark task module from its directory with path isolation.'''
    init_file = task_dir / '__init__.py'
    if not init_file.exists():
        raise FileNotFoundError(f"No __init__.py found in {task_dir}")
    
    # Save current sys.path to restore later
    original_path = sys.path.copy()
    
    try:
        # Clear sys.path and add only the task directory and essential paths
        sys.path.clear()
        sys.path.extend([
            str(task_dir),  # Task directory first for local imports
            '.',  # Current directory
        ])
        # Add back essential system paths, but exclude any benchmark_tasks paths to prevent conflicts
        original_path_filtered = [p for p in original_path if 'benchmark_tasks' not in p]
        sys.path.extend(original_path_filtered)
        
        # Create unique module name to avoid conflicts
        module_name = f"benchmark_task_{task_dir.name}_{hash(str(task_dir))}"
        
        # Clear any cached modules that might cause conflicts
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('get_instance') or k.startswith('benchmark_task_')]
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)
        
        spec = importlib.util.spec_from_file_location(module_name, str(init_file))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod  # Add to sys.modules to avoid import issues
        spec.loader.exec_module(mod)
        return mod
        
    finally:
        # Restore original sys.path
        sys.path.clear()
        sys.path.extend(original_path)


def _load_task_internal(task_name, task_dir):
    """Internal function for loading task (for multiprocessing)"""
    print(f"    Loading task module...")
    mod = load_benchmark_task(task_dir)
    
    print(f"    Building trace problem...")
    problem = mod.build_trace_problem()
    
    # Get initial score
    print(f"    Getting initial evaluation...")
    guide = problem['guide']
    param = problem['param']
    initial_code = param.data
    task_desc = problem['train_dataset']['inputs'][0]
    info = problem['train_dataset']['infos'][0]
    
    score, feedback = guide.get_feedback(task_desc, initial_code, info)
    
    return {
        'status': 'SUCCESS',
        'initial_score': score,
        'entry_function': problem['metadata']['entry'],
        'benchmark': problem['metadata']['benchmark'],
        'feedback_preview': feedback[:100] + '...' if len(feedback) > 100 else feedback
    }


def test_task_loading(task_name, task_dir):
    """Test if a task can be loaded and built"""
    try:
        # Use timeout for robust task loading
        result = run_with_timeout(lambda: _load_task_internal(task_name, task_dir), 5)
        return result
    except TimeoutError as e:
        return {
            'status': 'FAILED',
            'error': f'Task loading timed out after 5s',
            'error_type': 'TimeoutError'
        }
    except Exception as e:
        return {
            'status': 'FAILED',
            'error': str(e),
            'error_type': type(e).__name__
        }


def _optimize_task_internal(task_name, task_dir):
    """Internal function for optimization (for multiprocessing)"""
    print(f"      Loading for optimization...")
    mod = load_benchmark_task(task_dir)
    problem = mod.build_trace_problem()
    
    print(f"      Setting up optimization...")
    param = problem['param']
    guide = problem['guide']
    ds = problem['train_dataset']
    opt_kwargs = problem.get('optimizer_kwargs', {})
    
    # Minimal PrioritySearch parameters
    params = dict(
        guide=guide,
        train_dataset=ds,
        score_range=[-10, 10],
        num_epochs=1,
        num_steps=1,  # Just 1 step
        batch_size=1,
        num_batches=1,  # Just 1 batch
        verbose=False,
        num_candidates=2,  # Minimal candidates
        num_proposals=2,   # Minimal proposals
        memory_update_frequency=2,
        optimizer_kwargs=opt_kwargs,
        num_threads=1,
    )
    
    print(f"      Running optimization...")
    start_time = time.time()
    trainer.train(model=param, algorithm=SearchAlgorithm, **params)
    elapsed = time.time() - start_time
    
    # Get final score
    print(f"      Getting final score...")
    final_code = getattr(param, 'data', None)
    final_score, _ = guide('', final_code, ds['infos'][0])
    
    return {
        'status': 'OPTIMIZED',
        'optimization_time': elapsed,
        'final_score': final_score,
        'can_optimize': True
    }


def test_task_optimization(task_name, task_dir, max_time=5):
    """Test if a task can run optimization with minimal resources"""
    try:
        # Use timeout for robust optimization testing
        result = run_with_timeout(lambda: _optimize_task_internal(task_name, task_dir), max_time)
        return result
            
    except TimeoutError as e:
        return {
            'status': 'TIMEOUT',
            'optimization_time': max_time,
            'can_optimize': False,
            'error': f'Optimization timed out after {max_time}s'
        }
    except Exception as e:
        return {
            'status': 'OPT_FAILED',
            'can_optimize': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


def pick_benchmark_task(tasks_dir: Path, task_key: str) -> Path:
    '''
    Resolve an benchmark task directory by fuzzy key.
    '''
    cands = [p for p in tasks_dir.iterdir() if p.is_dir()]
    # exact
    for p in cands:
        if p.name == task_key:
            return p
    # substring
    for p in cands:
        if task_key in p.name:
            return p
    raise FileNotFoundError(f'No benchmark task matching: {task_key} in {tasks_dir}')


def main():
    ap = argparse.ArgumentParser(description='Validate benchmark LLM4AD tasks.')
    ap.add_argument('--tasks', type=str, default='./examples/benchmark_tasks', help='Folder with benchmark task directories')
    ap.add_argument('--task', type=str, help='Specific task key(s) to test, comma-separated (e.g., "circle_packing" or "optimization_bp_2d_construct,optimization_set_cover_construct")')
    args = ap.parse_args()
    
    # Threading-based timeout doesn't need multiprocessing setup
    
    tasks_dir = Path(args.tasks)
    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        return
    
    # Filter tasks based on --task parameter
    if args.task:
        task_keys = [key.strip() for key in args.task.split(',') if key.strip()]
        task_dirs = []
        for task_key in task_keys:
            try:
                task_dir = pick_benchmark_task(tasks_dir, task_key)
                task_dirs.append(task_dir)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        
        if not task_dirs:
            print("No valid tasks found!")
            return
        
        print(f"Testing {len(task_dirs)} specific task(s): {[d.name for d in task_dirs]}")
    else:
        task_dirs = [d for d in tasks_dir.iterdir() if d.is_dir()]
        print(f"Found {len(task_dirs)} benchmark tasks to validate")
    
    results = {}
    working_tasks = []
    optimizable_tasks = []
    
    for i, task_dir in enumerate(task_dirs, 1):
        task_name = task_dir.name
        print(f"\\n[{i}/{len(task_dirs)}] Testing {task_name}...")
        
        try:
            # Test loading (has its own robust timeout)
            load_result = test_task_loading(task_name, task_dir)
            results[task_name] = load_result
            
            print(f"  Loading: {load_result['status']}")
            if load_result['status'] == 'SUCCESS':
                print(f"    Entry: {load_result['entry_function']}")
                print(f"    Initial score: {load_result['initial_score']}")
                working_tasks.append(task_name)
                
                # Test optimization for all working tasks, including those with -inf scores
                # The updated llm4ad_loader should handle -inf more gracefully
                opt_result = test_task_optimization(task_name, task_dir)
                results[task_name].update(opt_result)
                print(f"  Optimization: {opt_result['status']}")
                if opt_result['status'] == 'OPTIMIZED':
                    print(f"    Time: {opt_result['optimization_time']:.2f}s")
                    print(f"    Final score: {opt_result['final_score']}")
                    optimizable_tasks.append(task_name)
                elif opt_result['status'] in ['TIMEOUT', 'OPT_FAILED']:
                    print(f"    Error: {opt_result.get('error', 'Unknown')}")
                
                # Mark as optimizable if it completed without major errors
                if opt_result['status'] in ['OPTIMIZED']:
                    results[task_name]['can_optimize'] = True
                else:
                    results[task_name]['can_optimize'] = False
                    
            else:
                print(f"    Error: {load_result['error']}")
                    
        except KeyboardInterrupt:
            print(f"\\nKeyboard interrupt - stopping validation")
            break
        except Exception as e:
            print(f"  UNEXPECTED ERROR: {e}")
            results[task_name] = {
                'status': 'FAILED', 
                'error': f'Unexpected error: {str(e)}',
                'error_type': type(e).__name__
            }
    
    # Summary
    print(f"\\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {len(task_dirs)}")
    print(f"Successfully loaded: {len(working_tasks)}")
    print(f"Can optimize quickly: {len(optimizable_tasks)}")
    
    print(f"\\nWORKING TASKS ({len(working_tasks)}):")
    for task in working_tasks:
        result = results[task]
        score = result['initial_score']
        print(f"  {task}: {result['entry_function']} (score: {score})")
    
    print(f"\\nQUICKLY OPTIMIZABLE TASKS ({len(optimizable_tasks)}):")
    for task in optimizable_tasks:
        result = results[task]
        print(f"  {task}: {result['optimization_time']:.2f}s (final: {result['final_score']})")
    
    print(f"\\nFAILED TASKS ({len(task_dirs) - len(working_tasks)}):")
    failed_tasks = [name for name, result in results.items() if result['status'] == 'FAILED']
    error_summary = {}
    for task in failed_tasks:
        error_type = results[task].get('error_type', 'Unknown')
        if error_type not in error_summary:
            error_summary[error_type] = []
        error_summary[error_type].append(task)
    
    for error_type, tasks in error_summary.items():
        print(f"  {error_type} ({len(tasks)}): {', '.join(tasks[:3])}{'...' if len(tasks) > 3 else ''}")
    
    # Save detailed results
    import json
    with open('benchmark_tasks_validation.json', 'w') as f:
        # Convert any non-serializable values
        serializable_results = {}
        for task, result in results.items():
            serializable_result = {}
            for k, v in result.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    serializable_result[k] = v
                else:
                    serializable_result[k] = str(v)
            serializable_results[task] = serializable_result
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\\nDetailed results saved to benchmark_tasks_validation.json")


if __name__ == '__main__':
    main()