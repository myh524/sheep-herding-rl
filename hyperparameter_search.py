#!/usr/bin/env python3
"""
Hyperparameter Search Script for PPO Sheep Herding

Supports grid search and random search for optimal hyperparameters
"""

import argparse
import os
import json
import numpy as np
import subprocess
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Search settings
    parser.add_argument('--search_type', type=str, default='random', choices=['grid', 'random'],
                        help='Type of hyperparameter search: grid or random')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='Number of trials for random search')
    parser.add_argument('--max_parallel', type=int, default=2,
                        help='Maximum number of parallel trials')
    
    # Default training settings
    parser.add_argument('--env_name', type=str, default='sheep_herding')
    parser.add_argument('--scenario_name', type=str, default='fast_train')
    parser.add_argument('--num_sheep', type=int, default=6)
    parser.add_argument('--num_herders', type=int, default=3)
    parser.add_argument('--episode_length', type=int, default=100)
    parser.add_argument('--num_env_steps', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=1)
    
    # Hyperparameter search space
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--lr_max', type=float, default=1e-3)
    parser.add_argument('--gae_lambda_min', type=float, default=0.9)
    parser.add_argument('--gae_lambda_max', type=float, default=0.99)
    parser.add_argument('--entropy_coef_min', type=float, default=0.001)
    parser.add_argument('--entropy_coef_max', type=float, default=0.1)
    parser.add_argument('--clip_param_min', type=float, default=0.1)
    parser.add_argument('--clip_param_max', type=float, default=0.3)
    parser.add_argument('--kl_coef_min', type=float, default=0.1)
    parser.add_argument('--kl_coef_max', type=float, default=1.0)
    
    args = parser.parse_args()
    return args


def generate_grid_search_space(args):
    """Generate grid search space"""
    search_space = {
        'lr': [1e-4, 3e-4, 5e-4],
        'gae_lambda': [0.95, 0.98],
        'entropy_coef': [0.01, 0.05],
        'clip_param': [0.2, 0.25],
        'kl_coef': [0.2, 0.5],
        'network_architecture': ['mlp', 'improved_mlp']
    }
    
    # Generate all combinations
    import itertools
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    return combinations


def generate_random_search_space(args):
    """Generate random search space"""
    combinations = []
    
    for _ in range(args.num_trials):
        combo = {
            'lr': np.exp(np.random.uniform(np.log(args.lr_min), np.log(args.lr_max))),
            'gae_lambda': np.random.uniform(args.gae_lambda_min, args.gae_lambda_max),
            'entropy_coef': np.exp(np.random.uniform(np.log(args.entropy_coef_min), np.log(args.entropy_coef_max))),
            'clip_param': np.random.uniform(args.clip_param_min, args.clip_param_max),
            'kl_coef': np.random.uniform(args.kl_coef_min, args.kl_coef_max),
            'network_architecture': np.random.choice(['mlp', 'improved_mlp'])
        }
        combinations.append(combo)
    
    return combinations


def run_trial(args, trial_config, trial_id):
    """Run a single trial with given hyperparameters"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'trial_{trial_id}_{timestamp}'
    
    # Build command
    cmd = [
        'python', 'train_ppo.py',
        '--env_name', args.env_name,
        '--scenario_name', args.scenario_name,
        '--num_sheep', str(args.num_sheep),
        '--num_herders', str(args.num_herders),
        '--episode_length', str(args.episode_length),
        '--num_env_steps', str(args.num_env_steps),
        '--seed', str(args.seed),
        '--experiment_name', exp_name,
        '--use_entropy_decay',
        '--use_cosine_lr',
        '--use_kl_penalty',
    ]
    
    # Add hyperparameters from trial config
    for key, value in trial_config.items():
        if key == 'lr':
            cmd.extend(['--lr', str(value)])
        elif key == 'gae_lambda':
            cmd.extend(['--gae_lambda', str(value)])
        elif key == 'entropy_coef':
            cmd.extend(['--initial_entropy_coef', str(value)])
        elif key == 'clip_param':
            cmd.extend(['--clip_param', str(value)])
        elif key == 'kl_coef':
            cmd.extend(['--kl_coef', str(value)])
        elif key == 'network_architecture':
            cmd.extend(['--network_architecture', str(value)])
    
    print(f"\nRunning trial {trial_id} with config: {trial_config}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the trial
    result_dir = os.path.join('results', args.env_name, args.scenario_name, 'ppo', f'seed{args.seed}')
    os.makedirs(result_dir, exist_ok=True)
    
    # Save trial config
    config_file = os.path.join(result_dir, f'trial_{trial_id}_config.json')
    with open(config_file, 'w') as f:
        json.dump(trial_config, f, indent=2)
    
    # Run the command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process, trial_id, exp_name, config_file


def monitor_trials(processes):
    """Monitor running trials and collect results"""
    results = []
    
    while processes:
        for proc_info in list(processes):
            process, trial_id, exp_name, config_file = proc_info
            
            # Check if process has finished
            returncode = process.poll()
            if returncode is not None:
                processes.remove(proc_info)
                
                # Collect stdout and stderr
                stdout, stderr = process.communicate()
                
                # Find the result directory
                result_dir = None
                for line in stdout.split('\n'):
                    if 'run_dir' in line or 'Results saved to' in line:
                        result_dir = line.split(':', 1)[1].strip()
                        break
                
                # Extract performance metrics
                metrics = extract_metrics(stdout)
                
                # Save results
                trial_result = {
                    'trial_id': trial_id,
                    'experiment_name': exp_name,
                    'returncode': returncode,
                    'metrics': metrics,
                    'stdout': stdout,
                    'stderr': stderr,
                    'config_file': config_file,
                    'result_dir': result_dir
                }
                
                results.append(trial_result)
                print(f"Trial {trial_id} completed with returncode {returncode}")
                print(f"Metrics: {metrics}")
        
        time.sleep(5)
    
    return results


def extract_metrics(stdout):
    """Extract performance metrics from stdout"""
    metrics = {}
    
    # Look for the last episode's metrics
    lines = stdout.split('\n')
    for line in reversed(lines):
        if 'episode:' in line and 'avg_reward:' in line:
            # Parse the line
            parts = line.strip().split(' | ')
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        if '.' in value:
                            metrics[key] = float(value)
                        else:
                            metrics[key] = int(value)
                    except:
                        metrics[key] = value
            break
    
    return metrics


def main():
    args = parse_args()
    
    print(f"Starting hyperparameter search: {args.search_type}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Maximum parallel trials: {args.max_parallel}")
    
    # Generate search space
    if args.search_type == 'grid':
        search_space = generate_grid_search_space(args)
    else:
        search_space = generate_random_search_space(args)
    
    print(f"Generated {len(search_space)} trials")
    
    # Run trials in parallel
    processes = []
    results = []
    
    for i, trial_config in enumerate(search_space):
        # Wait if we have reached max parallel trials
        while len(processes) >= args.max_parallel:
            time.sleep(10)
            # Check for completed processes
            completed = [p for p in processes if p[0].poll() is not None]
            for proc_info in completed:
                processes.remove(proc_info)
                # Collect results
                process, trial_id, exp_name, config_file = proc_info
                stdout, stderr = process.communicate()
                metrics = extract_metrics(stdout)
                results.append({
                    'trial_id': trial_id,
                    'metrics': metrics,
                    'config': trial_config
                })
        
        # Start new trial
        process_info = run_trial(args, trial_config, i)
        processes.append(process_info)
        time.sleep(2)  # Give some time for process to start
    
    # Wait for all remaining trials to complete
    if processes:
        remaining_results = monitor_trials(processes)
        results.extend(remaining_results)
    
    # Analyze results
    print("\n=== Hyperparameter Search Results ===")
    
    if results:
        # Sort by average reward
        results.sort(key=lambda x: x['metrics'].get('avg_reward', -float('inf')), reverse=True)
        
        # Save all results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'hyperparameter_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"All results saved to: {results_file}")
        
        # Print top 5 trials
        print("\nTop 5 trials:")
        for i, result in enumerate(results[:5]):
            print(f"\nRank {i+1}:")
            print(f"Trial ID: {result['trial_id']}")
            print(f"Average Reward: {result['metrics'].get('avg_reward', 'N/A')}")
            print(f"Config: {result.get('config', 'N/A')}")
    else:
        print("No results to analyze")


if __name__ == '__main__':
    main()
