import os
import sys

if os.environ.get("TRACE_BENCH_SMOKE") == "1":
    sys.exit(0)

import re
import string
import numpy as np
import time
from opto.trace.utils import dedent
from priority_search_on_convex_fn import Rosenbrock, SixHumpCamel, RewardGuide

# ============ TESTING code =============
import numpy as np
from opto import trace
from opto.features.priority_search import PrioritySearch as SearchAlgorithm
from opto.features.gepa.gepa_algorithms import GEPAAlgorithmBase, GEPAUCBSearch, GEPABeamPareto
from typing import Any
from opto import trainer
from typing import Tuple


def run_algorithm_comparison():
    """Compare PrioritySearch vs GEPA algorithms on optimization tasks using trainer.train API."""
    
    # Test on both Rosenbrock and SixHumpCamel
    envs = [
        ("Rosenbrock", Rosenbrock(horizon=20, seed=42)),
        ("SixHumpCamel", SixHumpCamel(horizon=20, seed=42))
    ]
    
    results = {}
    
    for env_name, env in envs:
        print(f"\n{'='*60}")
        print(f"Testing on {env_name}")
        print(f"{'='*60}")
        
        # Reset environment and get initial instruction
        instruction = env.reset()
        initial_input = instruction.split("\n")[0].strip()
        
        # Prepare train dataset 
        train_dataset = dict(inputs=[None], infos=[None])
        
        # Setup guide
        guide = RewardGuide(env)
        
        optimizer_kwargs = {'objective': "You have a task of guessing two numbers. You should make sure your guess minimizes y.", 'memory_size': 10}
        # Configure algorithms to test
        algorithms = [
            # PrioritySearch baseline
            {
                'name': 'PrioritySearch',
                'algorithm': SearchAlgorithm,
                'params': {
                    'guide': guide,
                    'train_dataset': train_dataset,
                    'score_range': [-10, 10],
                    'num_epochs': 1,
                    'num_steps': 3,
                    'batch_size': 1,
                    'num_batches': 2,
                    'verbose': False,
                    'num_candidates': 4,
                    'num_proposals': 4,
                    'memory_update_frequency': 2,
                    'optimizer_kwargs': optimizer_kwargs
                }
            },
            
            # GEPA algorithms  
            {
                'name': 'GEPA-Base',
                'algorithm': GEPAAlgorithmBase,
                'params': {
                    'guide': guide,
                    'train_dataset': train_dataset,
                    'validate_dataset': train_dataset,
                    'num_iters': 3,  # More iterations for better exploration
                    'train_batch_size': 2,  # Larger batch size
                    'merge_every': 2,  # Merge more frequently
                    'pareto_subset_size': 4,  # Larger Pareto subset
                    'num_threads': 2,
                    'optimizer_kwargs': optimizer_kwargs
                }
            },
            
            {
                'name': 'GEPA-UCB',
                'algorithm': GEPAUCBSearch,
                'params': {
                    'guide': guide,
                    'train_dataset': train_dataset,
                    'num_search_iterations': 3,  # More search iterations
                    'train_batch_size': 2,  # Larger batch size
                    'merge_every': 2,  # Merge more frequently  
                    'pareto_subset_size': 4,  # Larger Pareto subset
                    'num_threads': 2,
                    'optimizer_kwargs': optimizer_kwargs
                }
            },
            
            {
                'name': 'GEPA-Beam',
                'algorithm': GEPABeamPareto,
                'params': {
                    'guide': guide,
                    'train_dataset': train_dataset,
                    'validate_dataset': train_dataset,
                    'num_search_iterations': 3,  # More search iterations
                    'train_batch_size': 2,  # Larger batch size  
                    'merge_every': 2,  # Merge more frequently
                    'pareto_subset_size': 4,  # Larger Pareto subset
                    'num_threads': 2,
                    'optimizer_kwargs': optimizer_kwargs
                }
            }
        ]
        
        env_results = {}
        
        for algo_config in algorithms:
            name = algo_config['name']
            algorithm = algo_config['algorithm']
            params = algo_config['params']
            
            print(f"\nRunning {name}...")
            
            # Reset environment for each algorithm
            env.reset()
            
            # Create fresh trainable parameter for each algorithm
            param = trace.node(initial_input, description='Input x into the hidden function to get y.', trainable=True)
            
            # Time the algorithm
            start_time = time.time()
            
            try:
                print(f"  Initial parameter value: {param.data if hasattr(param, 'data') else param}")
                # Use trainer.train API consistently for all algorithms
                result = trainer.train(
                    model=param,
                    algorithm=algorithm,
                    **params
                )
                print(f"  Training result type: {type(result)}, value: {result}")
                
                # Get final parameter value and calculate score
                final_guess = str(param.data) if hasattr(param, 'data') else str(param)
                print(f"  Final parameter value: {final_guess}")
                x, _ = env.text_extract(final_guess)
                if x is not None:
                    # Get the function value directly, score = -function_value (higher is better)
                    final_score = -env.callable_func(x)
                    print(f"  Extracted coordinates: {x}, Function value: {env.callable_func(x)}")
                else:
                    final_score = -10.0  # penalty for invalid output
                    print(f"  Failed to extract coordinates from: {final_guess}")
                
                end_time = time.time()
                runtime = end_time - start_time
                
                env_results[name] = {
                    'score': final_score,
                    'runtime': runtime,
                    'success': final_score > -5.0
                }
                
                print(f"  ✓ {name}: Score={final_score:.4f}, Runtime={runtime:.2f}s")
                
            except Exception as e:
                print(f"  ✗ {name} failed with error: {e}")
                env_results[name] = {
                    'score': -10.0,
                    'runtime': float('inf'),
                    'success': False
                }
        
        results[env_name] = env_results
    
    # Analyze and display results
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for env_name, env_results in results.items():
        print(f"\n{env_name}:")
        print("-" * 40)
        
        priority_search_score = env_results.get("PrioritySearch", {}).get('score', -10.0)
        priority_search_time = env_results.get("PrioritySearch", {}).get('runtime', float('inf'))
        
        print(f"  PrioritySearch (baseline): {priority_search_score:.4f} (time: {priority_search_time:.2f}s)")
        
        gepa_wins = 0
        draws = 0
        for algo_name, result in env_results.items():
            if algo_name.startswith("GEPA"):
                improvement = result['score'] - priority_search_score
                time_ratio = result['runtime'] / priority_search_time if priority_search_time > 0 else float('inf')
                
                # Since scores are -function_value, higher scores = better performance (closer to optimal)
                if abs(improvement) < 1e-6:
                    status = "→ SAME"
                    draws += 1
                elif improvement > 0:
                    status = "✓ BETTER"
                    gepa_wins += 1
                else:
                    status = "✗ WORSE"
                
                print(f"  {algo_name:12}: {result['score']:7.4f} (improvement: {improvement:+6.4f}) "
                      f"(time: {result['runtime']:5.2f}s, ratio: {time_ratio:.2f}x) {status}")

        print(f"  → Results: {gepa_wins} GEPA wins // {draws} draws // PrioritySearch wins: {len(env_results)-1-gepa_wins-draws}")

    return results


if __name__ == "__main__":
    results = run_algorithm_comparison()
