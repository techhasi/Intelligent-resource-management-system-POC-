import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Import necessary classes from your LSTM module
import sys
sys.path.append('/Users/hwimalasooriya/Documents/GitHub/Intelligent-resource-management-system-POC-/main scripts/')
from LSTMv6 import LSTM

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def benchmark_lstm_models():
    """Benchmark LSTM models by extracting their metrics and comparing to simple baselines"""
    
    # Define model and data paths
    model_paths = {
        'ec2': '/Users/hwimalasooriya/Documents/GitHub/Intelligent-resource-management-system-POC-/models/ec2_lstm_model.pth',
        'rds': '/Users/hwimalasooriya/Documents/GitHub/Intelligent-resource-management-system-POC-/models/rds_lstm_model.pth',
        'ecs': '/Users/hwimalasooriya/Documents/GitHub/Intelligent-resource-management-system-POC-/models/ecs_lstm_model.pth'
    }
    
    test_data_paths = {
        'ec2': '/Users/hwimalasooriya/Documents/GitHub/Intelligent-resource-management-system-POC-/dataset scripts/reduced_ec2_data.csv',
        'rds': '/Users/hwimalasooriya/Documents/GitHub/Intelligent-resource-management-system-POC-/dataset scripts/reduced_rds_data.csv',
        'ecs': '/Users/hwimalasooriya/Documents/GitHub/Intelligent-resource-management-system-POC-/dataset scripts/reduced_ecs_data.csv'
    }
    
    all_results = {}
    model_metrics = {}
    
    # 1. Extract LSTM model metrics from saved models
    print("Extracting metrics from LSTM models...")
    for service, model_path in model_paths.items():
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, weights_only=False)
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    model_metrics[service] = metrics
                    print(f"  {service.upper()} - MAE: {metrics.get('mae_unscaled', 'N/A')}, "
                          f"RMSE: {metrics.get('rmse_unscaled', 'N/A')}")
                else:
                    print(f"  No metrics found in {service} model checkpoint")
            else:
                print(f"  Model file not found: {model_path}")
        except Exception as e:
            print(f"  Error loading {service} model: {e}")
    
    # 2. Create simple baselines on test data
    print("\nComputing baseline metrics on test data...")
    baseline_metrics = {}
    
    for service, data_path in test_data_paths.items():
        try:
            if os.path.exists(data_path):
                # Load data
                df = pd.read_csv(data_path)
                target_col = f"{service.upper()}_CPUUtilization"
                
                if target_col in df.columns:
                    # Clean data by removing NaN values
                    data = df[target_col].dropna().values
                    
                    if len(data) < 100:
                        print(f"  Not enough data for {service} after removing NaNs")
                        continue
                    
                    # Create training and test sets (80/20 split)
                    split_idx = int(len(data) * 0.8)
                    train_data = data[:split_idx]
                    test_data = data[split_idx:]
                    
                    # Simple baselines
                    # 1. Predict the mean of the training data
                    mean_baseline = np.full_like(test_data, np.mean(train_data))
                    
                    # 2. Predict the median of the training data
                    median_baseline = np.full_like(test_data, np.median(train_data))
                    
                    # 3. Predict the last value
                    last_value_baseline = np.roll(test_data, 1)
                    last_value_baseline[0] = train_data[-1]
                    
                    # 4. Predict the previous value (lag-1)
                    previous_value_baseline = np.roll(test_data, 1)
                    previous_value_baseline[0] = test_data[0]
                    
                    # Calculate metrics
                    baselines = {
                        'Mean': mean_baseline,
                        'Median': median_baseline,
                        'Last Training Value': last_value_baseline,
                        'Previous Value': previous_value_baseline
                    }
                    
                    metrics = {}
                    for name, predictions in baselines.items():
                        mae = mean_absolute_error(test_data, predictions)
                        mse = mean_squared_error(test_data, predictions)
                        rmse = np.sqrt(mse)
                        
                        metrics[name] = {
                            'MAE': mae,
                            'RMSE': rmse
                        }
                        
                        print(f"  {service.upper()} - {name} Baseline - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                    
                    baseline_metrics[service] = metrics
                    
                    # Create visualizations
                    plot_baseline_comparison(service, test_data, baselines)
                else:
                    print(f"  Target column {target_col} not found in {data_path}")
            else:
                print(f"  Data file not found: {data_path}")
        except Exception as e:
            print(f"  Error processing {service} test data: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. Compare LSTM with baselines
    print("\nComparing LSTM models with baselines...")
    
    comparison_results = {}
    
    for service in model_metrics.keys():
        if service in baseline_metrics:
            lstm_metrics = model_metrics[service]
            baselines = baseline_metrics[service]
            
            # Find best baseline
            best_baseline = min(baselines.items(), key=lambda x: x[1]['MAE'])
            best_baseline_name = best_baseline[0]
            best_baseline_mae = best_baseline[1]['MAE']
            
            if 'mae_unscaled' in lstm_metrics:
                lstm_mae = lstm_metrics['mae_unscaled']
                improvement = (best_baseline_mae - lstm_mae) / best_baseline_mae * 100
                
                comparison = {
                    'LSTM MAE': lstm_mae,
                    'Best Baseline': best_baseline_name,
                    'Best Baseline MAE': best_baseline_mae,
                    'Improvement (%)': improvement
                }
                
                comparison_results[service] = comparison
                
                print(f"  {service.upper()} - LSTM vs {best_baseline_name}:")
                print(f"    LSTM MAE: {lstm_mae:.4f}")
                print(f"    {best_baseline_name} MAE: {best_baseline_mae:.4f}")
                print(f"    Improvement: {improvement:.2f}%")
            else:
                print(f"  {service.upper()} - No MAE metric found in LSTM model")
    
    # 4. Create summary visualizations
    create_summary_visualization(model_metrics, baseline_metrics, comparison_results)
    
    # 5. Save results
    all_results = {
        'lstm_metrics': model_metrics,
        'baseline_metrics': baseline_metrics,
        'comparison': comparison_results
    }
    
    with open('lstm_benchmark_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # Create a summary dataframe and save as CSV
    summary_data = []
    
    for service in comparison_results.keys():
        comparison = comparison_results[service]
        row = {
            'Service': service,
            'LSTM MAE': comparison['LSTM MAE'],
            'Best Baseline': comparison['Best Baseline'],
            'Baseline MAE': comparison['Best Baseline MAE'],
            'Improvement (%)': comparison['Improvement (%)']
        }
        summary_data.append(row)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('lstm_benchmark_summary.csv', index=False)
        print("\nSummary saved to lstm_benchmark_summary.csv")
    
    return all_results

def plot_baseline_comparison(service, actuals, baselines):
    """Plot comparison of baseline predictions vs actuals"""
    plt.figure(figsize=(12, 8))
    
    # Plot actual values
    plt.plot(actuals, 'k-', label='Actual', linewidth=2)
    
    # Plot only a subset of points for clarity (100 points)
    sample_size = min(100, len(actuals))
    indices = np.linspace(0, len(actuals)-1, sample_size, dtype=int)
    
    # Plot baseline predictions
    for name, predictions in baselines.items():
        plt.plot(indices, predictions[indices], '--', label=f'{name}', alpha=0.7)
    
    plt.title(f'{service.upper()} - Baseline Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('CPU Utilization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{service}_baseline_comparison.png')
    plt.close()
    
    # Create a bar chart of MAE values
    plt.figure(figsize=(10, 6))
    
    # Calculate MAE for each baseline
    mae_values = []
    names = []
    
    for name, predictions in baselines.items():
        mae = mean_absolute_error(actuals, predictions)
        mae_values.append(mae)
        names.append(name)
    
    plt.bar(range(len(mae_values)), mae_values)
    plt.xticks(range(len(mae_values)), names, rotation=45)
    plt.title(f'{service.upper()} - Baseline MAE Comparison')
    plt.ylabel('Mean Absolute Error')
    plt.tight_layout()
    plt.savefig(f'{service}_baseline_mae_comparison.png')
    plt.close()

def create_summary_visualization(model_metrics, baseline_metrics, comparison_results):
    """Create summary visualizations comparing LSTM and baselines"""
    # Only proceed if we have comparison results
    if not comparison_results:
        return
    
    # Create a bar chart of improvement percentages
    plt.figure(figsize=(10, 6))
    
    services = list(comparison_results.keys())
    improvements = [comparison_results[s]['Improvement (%)'] for s in services]
    
    plt.bar(range(len(services)), improvements)
    plt.xticks(range(len(services)), services)
    plt.title('LSTM Improvement Over Best Baseline (%)')
    plt.ylabel('Improvement (%)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Add zero line
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_improvement_summary.png')
    plt.close()
    
    # Create a comparison of MAE values
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(services))
    width = 0.35
    
    lstm_maes = [model_metrics[s].get('mae_unscaled', 0) for s in services]
    baseline_maes = [comparison_results[s]['Best Baseline MAE'] for s in services]
    
    plt.bar(x - width/2, lstm_maes, width, label='LSTM')
    plt.bar(x + width/2, baseline_maes, width, label='Best Baseline')
    
    plt.xlabel('Service')
    plt.ylabel('Mean Absolute Error')
    plt.title('LSTM vs Best Baseline MAE Comparison')
    plt.xticks(x, services)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_vs_baseline_mae.png')
    plt.close()

if __name__ == "__main__":
    print("Starting LSTM model benchmarking...")
    results = benchmark_lstm_models()
    print("\nBenchmarking complete. Results saved to:")
    print("- lstm_benchmark_results.pkl")
    print("- lstm_benchmark_summary.csv")
    print("- lstm_improvement_summary.png")
    print("- lstm_vs_baseline_mae.png")