import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import random
from collections import defaultdict

# Import RL components - modify these imports based on your actual module structure
import sys
sys.path.append('/Users/hwimalasooriya/Documents/GitHub/Intelligent-resource-management-system-POC-/main scripts/')
from RLv5 import EnhancedQAgent, EnhancedCloudEnvironment, rule_based_action

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def benchmark_rl_model():
    """Benchmark RL model against a rule-based baseline"""
    
    # Define model and data paths
    rl_model_path = '../models/RLv5/enhanced_cloud_q_best.pkl'
    
    data_paths = {
        'ec2': '../../../dataset scripts/reduced_ec2_data.csv',
        'rds': '../../../dataset scripts/reduced_rds_data.csv',
        'ecs': '../../../dataset scripts/reduced_ecs_data.csv'
    }
    
    print("Starting RL model benchmarking...")
    
    # 1. Load RL agent
    print("\nLoading RL model...")
    try:
        agent = EnhancedQAgent()
        success = agent.load_model(rl_model_path)
        if success:
            print("Successfully loaded RL model")
        else:
            print("Failed to load RL model, using a new agent instead")
    except Exception as e:
        print(f"Error loading RL model: {e}")
        print("Using a new agent instead")
        agent = EnhancedQAgent()
    
    # 2. Load data and setup environment
    print("\nLoading data and setting up environment...")
    data = {}
    
    for service, path in data_paths.items():
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                
                # Convert timestamp if it exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df = df.set_index('timestamp').sort_index()
                
                data[service] = df
                print(f"Loaded {service} data with {len(df)} rows")
            else:
                print(f"Data file {path} does not exist")
        except Exception as e:
            print(f"Error loading {service} data: {e}")
    
    # Create environment
    try:
        env = EnhancedCloudEnvironment(data)
        print("Environment setup complete")
    except Exception as e:
        print(f"Error setting up environment: {e}")
        print("Using default environment")
        env = EnhancedCloudEnvironment({})
    
    # 3. Run evaluation
    print("\nRunning benchmark evaluation...")
    num_episodes = 3
    steps_per_episode = 50
    
    results = {
        'rl': {
            'rewards': [],
            'resources': [],
            'cpu': []
        },
        'rule_based': {
            'rewards': [],
            'resources': [],
            'cpu': []
        }
    }
    
    # Test RL agent
    print("\nTesting RL agent...")
    for episode in range(num_episodes):
        print(f"  Episode {episode+1}/{num_episodes}...")
        
        # Reset environment
        state = env.reset()
        
        episode_reward = 0
        episode_resources = {s: [] for s in env.services}
        episode_cpu = {s: [] for s in env.services}
        
        for step in range(steps_per_episode):
            # Store current state
            for service in env.services:
                episode_resources[service].append(env.current_resources[service])
                episode_cpu[service].append(env.current_cpu[service])
            
            # Select action using RL agent
            # Use epsilon=0 to always select the best action
            original_epsilon = agent.epsilon
            agent.epsilon = 0
            actions = agent.select_action(state)
            agent.epsilon = original_epsilon
            
            # Take action
            next_state, reward, done, _ = env.step(actions)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Store episode results
        results['rl']['rewards'].append(episode_reward)
        results['rl']['resources'].append(episode_resources)
        results['rl']['cpu'].append(episode_cpu)
        
        print(f"    RL Reward: {episode_reward:.2f}")
    
    # Test rule-based approach
    print("\nTesting rule-based approach...")
    for episode in range(num_episodes):
        print(f"  Episode {episode+1}/{num_episodes}...")
        
        # Reset environment
        state = env.reset()
        
        episode_reward = 0
        episode_resources = {s: [] for s in env.services}
        episode_cpu = {s: [] for s in env.services}
        
        for step in range(steps_per_episode):
            # Store current state
            for service in env.services:
                episode_resources[service].append(env.current_resources[service])
                episode_cpu[service].append(env.current_cpu[service])
            
            # Select action using rule-based approach
            actions = {}
            for i, service in enumerate(env.services):
                actions[i] = rule_based_action(
                    env.current_cpu,
                    env.current_resources,
                    i,
                    env.services
                )
            
            # Take action
            next_state, reward, done, _ = env.step(actions)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Store episode results
        results['rule_based']['rewards'].append(episode_reward)
        results['rule_based']['resources'].append(episode_resources)
        results['rule_based']['cpu'].append(episode_cpu)
        
        print(f"    Rule-based Reward: {episode_reward:.2f}")
    
    # 4. Calculate metrics
    print("\nCalculating metrics...")
    
    # Overall reward metrics
    rl_avg_reward = np.mean(results['rl']['rewards'])
    rb_avg_reward = np.mean(results['rule_based']['rewards'])
    improvement = ((rl_avg_reward - rb_avg_reward) / abs(rb_avg_reward)) * 100 if rb_avg_reward != 0 else 0
    
    metrics = {
        'rl_avg_reward': rl_avg_reward,
        'rule_based_avg_reward': rb_avg_reward,
        'improvement_percentage': improvement
    }
    
    print(f"  RL Average Reward: {rl_avg_reward:.2f}")
    print(f"  Rule-based Average Reward: {rb_avg_reward:.2f}")
    print(f"  Reward Improvement: {improvement:.2f}%")
    
    # Resource allocation comparison
    service_metrics = {}
    
    for service in env.services:
        # For each service, get average resource allocation
        rl_resources = []
        rb_resources = []
        
        for episode in range(num_episodes):
            rl_episode_resources = results['rl']['resources'][episode][service]
            rb_episode_resources = results['rule_based']['resources'][episode][service]
            
            # Calculate average resource allocation for this episode
            rl_avg_resources = np.mean(rl_episode_resources) if rl_episode_resources else 0
            rb_avg_resources = np.mean(rb_episode_resources) if rb_episode_resources else 0
            
            rl_resources.append(rl_avg_resources)
            rb_resources.append(rb_avg_resources)
        
        # Calculate average across episodes
        avg_rl_resources = np.mean(rl_resources)
        avg_rb_resources = np.mean(rb_resources)
        resource_diff = avg_rl_resources - avg_rb_resources
        
        service_metrics[service] = {
            'avg_rl_resources': avg_rl_resources,
            'avg_rb_resources': avg_rb_resources,
            'resource_difference': resource_diff
        }
        
        print(f"  {service.upper()} Average Resources:")
        print(f"    RL: {avg_rl_resources:.2f}")
        print(f"    Rule-based: {avg_rb_resources:.2f}")
        print(f"    Difference: {resource_diff:.2f}")
    
    # 5. Create visualizations
    print("\nCreating visualizations...")
    
    # Reward comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['RL', 'Rule-based'], [rl_avg_reward, rb_avg_reward])
    plt.title('Average Reward Comparison')
    plt.ylabel('Average Reward')
    plt.grid(axis='y', alpha=0.3)
    
    # Add text with improvement percentage
    plt.text(0.5, min(rl_avg_reward, rb_avg_reward) - 1, 
             f"Improvement: {improvement:.2f}%", 
             horizontalalignment='center')
    
    plt.tight_layout()
    plt.savefig('rl_reward_comparison.png')
    plt.close()
    
    # Resource allocation comparison
    plt.figure(figsize=(12, 6))
    
    # For each service, plot average resource allocation
    x = np.arange(len(env.services))
    width = 0.35
    
    rl_resources = [service_metrics[s]['avg_rl_resources'] for s in env.services]
    rb_resources = [service_metrics[s]['avg_rb_resources'] for s in env.services]
    
    plt.bar(x - width/2, rl_resources, width, label='RL')
    plt.bar(x + width/2, rb_resources, width, label='Rule-based')
    
    plt.xlabel('Service')
    plt.ylabel('Average Resource Allocation')
    plt.title('Resource Allocation Comparison')
    plt.xticks(x, env.services)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rl_resource_comparison.png')
    plt.close()
    
    # Resource allocation over time for each service
    for service in env.services:
        plt.figure(figsize=(14, 8))
        
        # Plot CPU utilization
        plt.subplot(2, 1, 1)
        
        # Get average CPU utilization across episodes
        rl_cpu = np.mean([np.array(results['rl']['cpu'][episode][service]) 
                         for episode in range(num_episodes)], axis=0)
        rb_cpu = np.mean([np.array(results['rule_based']['cpu'][episode][service]) 
                         for episode in range(num_episodes)], axis=0)
        
        plt.plot(rl_cpu, 'b-', label='RL CPU', alpha=0.7)
        plt.plot(rb_cpu, 'r-', label='Rule-based CPU', alpha=0.7)
        plt.title(f'{service.upper()} - CPU Utilization')
        plt.ylabel('CPU Utilization (%)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot resource allocation
        plt.subplot(2, 1, 2)
        
        # Get average resource allocation across episodes
        rl_resources = np.mean([np.array(results['rl']['resources'][episode][service]) 
                               for episode in range(num_episodes)], axis=0)
        rb_resources = np.mean([np.array(results['rule_based']['resources'][episode][service]) 
                               for episode in range(num_episodes)], axis=0)
        
        plt.plot(rl_resources, 'b-', label='RL Resources', alpha=0.7)
        plt.plot(rb_resources, 'r-', label='Rule-based Resources', alpha=0.7)
        plt.title(f'{service.upper()} - Resource Allocation')
        plt.xlabel('Step')
        plt.ylabel('Resource Count')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{service}_comparison.png')
        plt.close()
    
    # 6. Save results
    # Combine all results
    all_results = {
        'episode_results': results,
        'metrics': metrics,
        'service_metrics': service_metrics
    }
    
    with open('rl_benchmark_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create summary dataframe
    summary_data = {
        'RL Average Reward': [rl_avg_reward],
        'Rule-based Average Reward': [rb_avg_reward],
        'Improvement (%)': [improvement]
    }
    
    for service in env.services:
        summary_data[f'{service} RL Resources'] = [service_metrics[service]['avg_rl_resources']]
        summary_data[f'{service} Rule-based Resources'] = [service_metrics[service]['avg_rb_resources']]
        summary_data[f'{service} Resource Difference'] = [service_metrics[service]['resource_difference']]
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('rl_benchmark_summary.csv', index=False)
    
    print("\nBenchmarking complete. Results saved to:")
    print("- rl_benchmark_results.pkl")
    print("- rl_benchmark_summary.csv")
    print("- rl_reward_comparison.png")
    print("- rl_resource_comparison.png")
    for service in env.services:
        print(f"- {service}_comparison.png")
    
    return all_results

if __name__ == "__main__":
    benchmark_rl_model()