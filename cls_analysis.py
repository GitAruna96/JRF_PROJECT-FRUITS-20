import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def analyze_experiment(csv_path):
    """
    Analyzes a single experiment CSV file and creates plots.
    """
    # Load results
    df = pd.read_csv("C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\fruits_dataset\\paper_experiments\\results\\swin_t_fruits_cls_results.csv")
    model_name = os.path.basename(csv_path).split('_')[0]  # Extract model name from filename
    
    # Create output directory for plots
    plot_dir = os.path.join(os.path.dirname(csv_path), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Final Timing Analysis
    total_time = df['Cumulative_Time_s'].iloc[-1]
    avg_epoch_time = df['Epoch_Time_s'].mean()
    best_accuracy = df['Test_Accuracy'].max()
    
    print(f"\n--- {model_name} Timing Analysis ---")
    print(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f} seconds")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    
    # 2. Create Accuracy/Loss Curves Plot
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Train_Accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(df['Epoch'], df['Test_Accuracy'], 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Accuracy Curves\nBest Test: {best_accuracy:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Train_Loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(df['Epoch'], df['Test_Loss'], 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'{model_name}_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {plot_path}")
    
    # 3. Create Timing Analysis Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Epoch_Time_s'], 'g-o', markersize=4, linewidth=2)
    plt.axhline(y=avg_epoch_time, color='r', linestyle='--', 
                label=f'Average: {avg_epoch_time:.2f}s/epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title(f'{model_name} - Time per Epoch\nTotal: {total_time/60:.1f} minutes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    timing_path = os.path.join(plot_dir, f'{model_name}_timing_analysis.png')
    plt.savefig(timing_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved timing analysis to {timing_path}")
    
    return df

def analyze_all_experiments(results_dir):
    """
    Analyzes all CSV files in the results directory and creates comparison plots.
    """
    # Find all CSV result files
    csv_files = glob.glob(os.path.join(results_dir, '*_fruits_cls_results*.csv'))
    
    if not csv_files:
        print(f"No result files found in {results_dir}")
        return
    
    # Analyze each experiment
    all_results = {}
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Analyzing: {os.path.basename(csv_file)}")
        print(f"{'='*50}")
        
        df = analyze_experiment(csv_file)
        model_name = os.path.basename(csv_file).split('_')[0]
        all_results[model_name] = df
    
    # Create comparison plot (when you have multiple methods)
    if len(all_results) > 1:
        plt.figure(figsize=(10, 6))
        for model_name, df in all_results.items():
            plt.plot(df['Epoch'], df['Test_Accuracy'], '-', label=model_name, linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Comparison: Test Accuracy by Method')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        comparison_path = os.path.join(results_dir, 'plots', 'method_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved method comparison to {comparison_path}")

if __name__ == '__main__':
    # Analyze all experiments in the results directory
    results_directory = './results'  # Change this if your results are elsewhere
    analyze_all_experiments(results_directory)