import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_cpu_pattern(days, base_pattern='normal', min_val=5, max_val=95):
    """
    Generate a synthetic CPU utilization pattern.
    
    Args:
        days: Number of days of data to generate
        base_pattern: Pattern type ('normal', 'spiky', 'cyclic')
        min_val: Minimum CPU value
        max_val: Maximum CPU value
    
    Returns:
        List of CPU values at 5-minute intervals
    """
    points_per_day = 24 * 12  # 5-minute intervals
    total_points = days * points_per_day
    
    if base_pattern == 'normal':
        # Normal business hours pattern with weekends
        cpu_values = []
        for day in range(days):
            # Determine if it's a weekend
            is_weekend = (day % 7) >= 5
            
            for hour in range(24):
                # Business hours: 8am-6pm on weekdays
                if 8 <= hour < 18 and not is_weekend:
                    # Higher utilization during business hours
                    base = np.random.uniform(40, 70)
                elif 18 <= hour < 22 and not is_weekend:
                    # Medium utilization in evening
                    base = np.random.uniform(20, 50)
                else:
                    # Low utilization at night and weekends
                    base = np.random.uniform(5, 25)
                
                # Add 12 points for each hour (5-minute intervals)
                for _ in range(12):
                    # Add some random noise
                    noise = np.random.normal(0, 5)
                    value = max(min_val, min(max_val, base + noise))
                    cpu_values.append(value)
    
    elif base_pattern == 'spiky':
        # Pattern with occasional spikes
        cpu_values = []
        for day in range(days):
            for hour in range(24):
                # Decide if this hour will have a spike
                has_spike = random.random() < 0.1  # 10% chance of spike
                
                for minute in range(0, 60, 5):  # 5-minute intervals
                    if has_spike and 15 <= minute < 30:
                        # Create a spike for 15 minutes
                        base = np.random.uniform(75, 95)
                    else:
                        # Normal utilization
                        base = np.random.uniform(10, 40)
                    
                    # Add some random noise
                    noise = np.random.normal(0, 3)
                    value = max(min_val, min(max_val, base + noise))
                    cpu_values.append(value)
    
    elif base_pattern == 'cyclic':
        # Sine wave pattern with daily cycles
        times = np.linspace(0, 2*np.pi*days, total_points)
        
        # Create base sine wave
        base_cpu = (np.sin(times) + 1) / 2  # Normalized to [0, 1]
        
        # Scale to desired range
        cpu_values = base_cpu * (max_val - min_val) + min_val
        
        # Add noise
        noise = np.random.normal(0, 5, total_points)
        cpu_values = np.clip(cpu_values + noise, min_val, max_val)
        
        # Add weekly pattern (lower on weekends)
        for i in range(total_points):
            day_of_week = (i // points_per_day) % 7
            if day_of_week >= 5:  # Weekend
                cpu_values[i] *= 0.7  # 30% reduction on weekends
    
    return cpu_values

def generate_correlated_metric(cpu_values, correlation=0.7, min_val=0, max_val=100):
    """
    Generate a metric correlated with CPU utilization.
    
    Args:
        cpu_values: List of CPU utilization values
        correlation: Correlation coefficient with CPU (0-1)
        min_val: Minimum value for the metric
        max_val: Maximum value for the metric
    
    Returns:
        List of correlated metric values
    """
    n = len(cpu_values)
    
    # Normalize CPU values to [0, 1]
    cpu_norm = (np.array(cpu_values) - min(cpu_values)) / (max(cpu_values) - min(cpu_values))
    
    # Generate random values
    random_values = np.random.uniform(0, 1, n)
    
    # Create correlated values as a weighted average
    correlated = correlation * cpu_norm + (1 - correlation) * random_values
    
    # Scale to desired range
    scaled_values = correlated * (max_val - min_val) + min_val
    
    return scaled_values

def generate_task_count(cpu_values, max_tasks=10):
    """
    Generate ECS task count based on CPU utilization (integer values)
    """
    # Base the number of tasks on CPU utilization
    normalized_cpu = np.array(cpu_values) / 100.0
    
    # Generate task counts with positive correlation to CPU
    # Higher CPU utilization generally means more tasks
    task_counts = np.round(normalized_cpu * max_tasks).astype(int)
    
    # Ensure at least 1 task
    task_counts = np.maximum(task_counts, 1)
    
    return task_counts

def generate_connections(cpu_values, base_connections=50, max_connections=500):
    """
    Generate database connection counts for RDS
    """
    # Base connections has some correlation with CPU
    normalized_cpu = np.array(cpu_values) / 100.0
    
    # Generate connection counts with some correlation to CPU
    connections = base_connections + normalized_cpu * (max_connections - base_connections)
    
    # Add some random variation
    noise = np.random.normal(0, max_connections * 0.1, len(cpu_values))
    connections = np.maximum(connections + noise, 1)  # Ensure at least 1 connection
    
    return connections

def generate_aws_metrics(days=30, output_dir="data"):
    """
    Generate synthetic AWS metrics for EC2, RDS, and ECS services.
    
    Args:
        days: Number of days of data to generate
        output_dir: Directory to save CSV files
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp series (5-minute intervals)
    start_date = datetime(2023, 8, 1)
    timestamps = [start_date + timedelta(minutes=5*i) for i in range(days * 24 * 12)]
    
    # Generate data for each service type
    generate_ec2_data(timestamps, output_dir)
    generate_rds_data(timestamps, output_dir)
    generate_ecs_data(timestamps, output_dir)
    
    print(f"Generated synthetic data for {days} days ({len(timestamps)} data points)")
    print(f"Files saved in '{output_dir}' directory")

def generate_ec2_data(timestamps, output_dir):
    """Generate EC2 metrics data"""
    # Generate CPU patterns with balanced distribution
    cpu_normal = generate_cpu_pattern(len(timestamps) // (24 * 12), 'normal')
    cpu_spiky = generate_cpu_pattern(len(timestamps) // (24 * 12), 'spiky')
    cpu_cyclic = generate_cpu_pattern(len(timestamps) // (24 * 12), 'cyclic')
    
    # Ensure we have low, medium, and high utilization samples
    # by adjusting some of the values
    low_indices = np.random.choice(len(cpu_normal), size=len(cpu_normal)//3, replace=False)
    medium_indices = np.random.choice(len(cpu_normal), size=len(cpu_normal)//3, replace=False)
    high_indices = np.random.choice(len(cpu_normal), size=len(cpu_normal)//3, replace=False)
    
    for idx in low_indices:
        cpu_normal[idx] = np.random.uniform(5, 20)
    
    for idx in medium_indices:
        cpu_normal[idx] = np.random.uniform(30, 60)
    
    for idx in high_indices:
        cpu_normal[idx] = np.random.uniform(70, 95)
    
    # Combine patterns
    pattern_indices = np.random.choice([0, 1, 2], size=len(timestamps), p=[0.6, 0.2, 0.2])
    cpu_values = []
    
    for i, timestamp in enumerate(timestamps):
        if pattern_indices[i] == 0:
            idx = i % len(cpu_normal)
            cpu_values.append(cpu_normal[idx])
        elif pattern_indices[i] == 1:
            idx = i % len(cpu_spiky)
            cpu_values.append(cpu_spiky[idx])
        else:
            idx = i % len(cpu_cyclic)
            cpu_values.append(cpu_cyclic[idx])
    
    # Generate correlated metrics
    memory_utilization = generate_correlated_metric(cpu_values, 0.7, 5, 95)
    disk_writeops = generate_correlated_metric(cpu_values, 0.5, 10, 5000)
    network_in = generate_correlated_metric(cpu_values, 0.6, 100, 10000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'EC2_CPUUtilization': cpu_values,
        'EC2_MemoryUtilization': memory_utilization,
        'EC2_DiskWriteOps': disk_writeops,
        'EC2_NetworkIn': network_in
    })
    
    # Save to CSV
    df.to_csv(f"{output_dir}/ec2_metrics.csv", index=False)
    print(f"EC2 metrics saved to {output_dir}/ec2_metrics.csv")

def generate_rds_data(timestamps, output_dir):
    """Generate RDS metrics data"""
    # Generate CPU patterns with balanced distribution
    cpu_normal = generate_cpu_pattern(len(timestamps) // (24 * 12), 'normal')
    cpu_spiky = generate_cpu_pattern(len(timestamps) // (24 * 12), 'spiky')
    
    # Ensure we have low, medium, and high utilization samples
    low_indices = np.random.choice(len(cpu_normal), size=len(cpu_normal)//3, replace=False)
    medium_indices = np.random.choice(len(cpu_normal), size=len(cpu_normal)//3, replace=False)
    high_indices = np.random.choice(len(cpu_normal), size=len(cpu_normal)//3, replace=False)
    
    for idx in low_indices:
        cpu_normal[idx] = np.random.uniform(5, 20)
    
    for idx in medium_indices:
        cpu_normal[idx] = np.random.uniform(30, 60)
    
    for idx in high_indices:
        cpu_normal[idx] = np.random.uniform(70, 95)
    
    # Combine patterns
    pattern_indices = np.random.choice([0, 1], size=len(timestamps), p=[0.7, 0.3])
    cpu_values = []
    
    for i, timestamp in enumerate(timestamps):
        if pattern_indices[i] == 0:
            idx = i % len(cpu_normal)
            cpu_values.append(cpu_normal[idx])
        else:
            idx = i % len(cpu_spiky)
            cpu_values.append(cpu_spiky[idx])
    
    # Generate correlated metrics
    # RDS FreeableMemory is inversely related to CPU
    inverse_cpu = [100 - cpu for cpu in cpu_values]
    freeable_memory = generate_correlated_metric(inverse_cpu, 0.6, 1000, 10000)
    connections = generate_connections(cpu_values)
    write_iops = generate_correlated_metric(cpu_values, 0.8, 10, 2000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'RDS_CPUUtilization': cpu_values,
        'RDS_FreeableMemory': freeable_memory,
        'RDS_DatabaseConnections': connections,
        'RDS_WriteIOPS': write_iops
    })
    
    # Save to CSV
    df.to_csv(f"{output_dir}/rds_metrics.csv", index=False)
    print(f"RDS metrics saved to {output_dir}/rds_metrics.csv")

def generate_ecs_data(timestamps, output_dir):
    """Generate ECS metrics data"""
    # Generate CPU patterns with balanced distribution
    cpu_normal = generate_cpu_pattern(len(timestamps) // (24 * 12), 'normal')
    cpu_cyclic = generate_cpu_pattern(len(timestamps) // (24 * 12), 'cyclic')
    
    # Ensure we have low, medium, and high utilization samples
    low_indices = np.random.choice(len(cpu_normal), size=len(cpu_normal)//3, replace=False)
    medium_indices = np.random.choice(len(cpu_normal), size=len(cpu_normal)//3, replace=False)
    high_indices = np.random.choice(len(cpu_normal), size=len(cpu_normal)//3, replace=False)
    
    for idx in low_indices:
        cpu_normal[idx] = np.random.uniform(5, 20)
    
    for idx in medium_indices:
        cpu_normal[idx] = np.random.uniform(30, 60)
    
    for idx in high_indices:
        cpu_normal[idx] = np.random.uniform(70, 95)
    
    # Combine patterns
    pattern_indices = np.random.choice([0, 1], size=len(timestamps), p=[0.6, 0.4])
    cpu_values = []
    
    for i, timestamp in enumerate(timestamps):
        if pattern_indices[i] == 0:
            idx = i % len(cpu_normal)
            cpu_values.append(cpu_normal[idx])
        else:
            idx = i % len(cpu_cyclic)
            cpu_values.append(cpu_cyclic[idx])
    
    # Generate correlated metrics
    memory_utilization = generate_correlated_metric(cpu_values, 0.8, 10, 90)
    task_count = generate_task_count(cpu_values)
    network_in = generate_correlated_metric(cpu_values, 0.6, 100, 8000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'ECS_CPUUtilization': cpu_values,
        'ECS_MemoryUtilization': memory_utilization,
        'ECS_RunningTaskCount': task_count,
        'ECS_NetworkIn': network_in
    })
    
    # Save to CSV
    df.to_csv(f"{output_dir}/ecs_metrics.csv", index=False)
    print(f"ECS metrics saved to {output_dir}/ecs_metrics.csv")

if __name__ == "__main__":
    print("Generating synthetic AWS metrics data...")
    # Generate 30 days of metrics data at 5-minute intervals
    generate_aws_metrics(days=150)