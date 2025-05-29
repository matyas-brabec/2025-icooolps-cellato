#!/usr/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_csv(csv_path):
    """Process the CSV file and extract relevant data, finding best y_block_size for each case"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Process baseline implementations
    baseline_df = df[df['reference_impl'] == 'baseline'].copy()
    
    # Process standard implementations
    standard_df = df[(df['traverser'] == 'simple') & 
                    (df['evaluator'] == 'standard') & 
                    (df['layout'] == 'standard')].copy()
    
    # Find best y_block_size for each automaton and device
    best_baseline = []
    best_standard = []
    
    # Group by automaton and device to find the best y_block_size
    for (automaton, device), group in baseline_df.groupby(['automaton', 'device']):
        if len(group) > 1:  # There are multiple block sizes to choose from
            best_row = group.loc[group['average_time_per_cell_ns'].idxmin()]
            best_baseline.append(best_row)
            print(f"Best y_block_size for {automaton} on {device} (baseline): {best_row['cuda_block_size_y']} with {best_row['average_time_per_cell_ns']:.6f} ns/cell")
        else:  # Only one row (likely CPU which doesn't use block sizes)
            best_baseline.append(group.iloc[0])
    
    for (automaton, device), group in standard_df.groupby(['automaton', 'device']):
        if len(group) > 1:  # There are multiple block sizes to choose from
            best_row = group.loc[group['average_time_per_cell_ns'].idxmin()]
            best_standard.append(best_row)
            print(f"Best y_block_size for {automaton} on {device} (standard): {best_row['cuda_block_size_y']} with {best_row['average_time_per_cell_ns']:.6f} ns/cell")
        else:  # Only one row (likely CPU which doesn't use block sizes)
            best_standard.append(group.iloc[0])
    
    # Convert lists back to dataframes
    baseline_df = pd.DataFrame(best_baseline)
    standard_df = pd.DataFrame(best_standard)
    
    # Add implementation column for plotting
    baseline_df['implementation'] = 'baseline'
    standard_df['implementation'] = 'standard'
    
    # Combine dataframes for plotting
    plot_df = pd.concat([baseline_df, standard_df])
    
    return plot_df

def create_plot(df, output_path):
    """Create a side-by-side plot comparing CPU and CUDA implementations"""
    # Get unique automata for the x-axis
    # automata = df['automaton'].unique()
    automata = ['game-of-life', 'forest-fire', 'wire', 'greenberg-hastings']
    
    # Use publication-quality settings
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Computer Modern Roman'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300
    })
    
    # Set up the figure with two subplots side by side
    fig, (ax_cpu, ax_cuda) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Define consistent colors for baseline and standard implementations
    baseline_color = '#1f77b4'  # Blue
    standard_color = '#ff7f0e'  # Orange
    
    # Define hatching patterns for better distinction in print
    baseline_hatch = ''  # No hatching
    standard_hatch = '//'  # Diagonal lines
    
    # Set width of bars
    bar_width = 0.35
    
    # Set positions of bars on X axis
    index = np.arange(len(automata))
    
    # Plot CPU data
    cpu_data = df[df['device'] == 'CPU']
    baseline_bars = []
    standard_bars = []
    
    for i, automaton in enumerate(automata):
        # Filter data for this automaton
        auto_data = cpu_data[cpu_data['automaton'] == automaton]
        
        # Plot baseline and standard bars
        baseline = auto_data[auto_data['implementation'] == 'baseline']
        standard = auto_data[auto_data['implementation'] == 'standard']
        
        if not baseline.empty:
            bar = ax_cpu.bar(index[i] - bar_width/2, baseline['average_time_per_cell_ns'].values[0], 
                    bar_width, color=baseline_color, hatch=baseline_hatch, 
                    edgecolor='black', linewidth=1)
            if i == 0:
                baseline_bars.append(bar)
                
        if not standard.empty:
            bar = ax_cpu.bar(index[i] + bar_width/2, standard['average_time_per_cell_ns'].values[0], 
                    bar_width, color=standard_color, hatch=standard_hatch, 
                    edgecolor='black', linewidth=1)
            if i == 0:
                standard_bars.append(bar)
    
    # Plot CUDA data
    cuda_data = df[df['device'] == 'CUDA']
    for i, automaton in enumerate(automata):
        # Filter data for this automaton
        auto_data = cuda_data[cuda_data['automaton'] == automaton]
        
        # Plot baseline and standard bars
        baseline = auto_data[auto_data['implementation'] == 'baseline']
        standard = auto_data[auto_data['implementation'] == 'standard']
        
        if not baseline.empty:
            bar = ax_cuda.bar(index[i] - bar_width/2, baseline['average_time_per_cell_ns'].values[0], 
                     bar_width, color=baseline_color, hatch=baseline_hatch, 
                     edgecolor='black', linewidth=1)
            if i == 0:
                baseline_bars.append(bar)
                
        if not standard.empty:
            bar = ax_cuda.bar(index[i] + bar_width/2, standard['average_time_per_cell_ns'].values[0], 
                     bar_width, color=standard_color, hatch=standard_hatch, 
                     edgecolor='black', linewidth=1)
            if i == 0:
                standard_bars.append(bar)
    
    # Customize the plots
    ax_cpu.set_ylabel('Time per cell (ns)', fontweight='bold')
    ax_cuda.set_ylabel('Time per cell (ns)', fontweight='bold')
    
    ax_cpu.set_xticks(index)
    ax_cuda.set_xticks(index)
    
    # Format x-axis with automata names
    pretty_names = {
        'game-of-life': 'Game of Life',
        'forest-fire': 'Forest Fire',
        'wire': 'Wire World',
        'greenberg-hastings': 'Greenberg-Hastings'
    }
    ax_cpu.set_xticklabels([pretty_names.get(a, a) for a in automata], rotation=30, ha='right')
    ax_cuda.set_xticklabels([pretty_names.get(a, a) for a in automata], rotation=30, ha='right')
    
    # Add light grid lines
    ax_cpu.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax_cuda.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add device labels to identify the plots
    ax_cpu.text(0.5, 0.95, 'CPU', transform=ax_cpu.transAxes, 
                horizontalalignment='center', fontsize=16, fontweight='bold')
    ax_cuda.text(0.5, 0.95, 'CUDA', transform=ax_cuda.transAxes, 
                 horizontalalignment='center', fontsize=16, fontweight='bold')
    
    # Add a single legend for both subplots
    fig.legend(
        [baseline_bars[0], standard_bars[0]], 
        ['Baseline', 'Standard'], 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor='black'
    )
    
    # Adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend below
    
    # Save as both PNG and PDF for publication
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Plot saved to {output_path} and {output_path.replace('.png', '.pdf')}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_baseline_vs_standard.py <csv_file>")
        return 1
    
    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"Error: File {csv_path} does not exist")
        return 1
    
    # Determine output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(script_dir, '../../results'))
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, 'baseline_vs_standard_comparison.png')
    
    # Process data and create plot
    df = process_csv(csv_path)
    create_plot(df, output_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
