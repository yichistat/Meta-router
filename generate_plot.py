#!/usr/bin/env python3
"""
Specialized script to generate plot from PC50_varstd_RF_100G_individual_runs.csv
- Directly reads the specified CSV file
- Generates PC50_varstd_RF_100G_plot.png
- Uses MEAN instead of MEDIAN
- Clean style with no captions/titles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set clean style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (8, 8),
    'lines.linewidth': 3,
    'axes.grid': True,
    'grid.alpha': 0.3
})

def create_plot():
    """Create efficiency gain plot from PC50_varstd_RF_100G_individual_runs.csv"""
    
    # Hardcoded file paths
    csv_file = "PC50_varstd_RF_100G_individual_runs.csv"
    output_file = "PC50_varstd_RF_100G_plot.png"
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File not found: {csv_file}")
        return
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} records from {csv_file}")
        print(f"📊 Approaches found: {list(df['approach'].unique())}")
        
        # Get unique approaches (excluding Random if present)
        approaches = [a for a in df['approach'].unique() if a != 'Random']
        ratios = sorted(df['ratio'].unique())
        print(f"📈 Ratios: {ratios}")
        
        # Calculate mean efficiency gains for each approach at each ratio
        mean_results = {}
        
        for approach in approaches:
            approach_data = df[df['approach'] == approach]
            mean_gains = []
            valid_ratios = []
            
            for ratio in ratios:
                ratio_data = approach_data[approach_data['ratio'] == ratio]
                if len(ratio_data) > 0:
                    mean_gain = ratio_data['efficiency_gain'].mean()
                    mean_gains.append(mean_gain)
                    valid_ratios.append(ratio)
            
            mean_results[approach] = {
                'ratios': np.array(valid_ratios),
                'efficiency_gains': np.array(mean_gains)
            }
        
        # Create plot
        plt.figure(figsize=(8, 8))
        
        # Colors and styles for approaches
        colors = ['#E74C3C', '#3498DB', '#9B59B6', '#2ECC71', '#F39C12', '#808080']
        linestyles = ['-', '-', '--', '--', '-.', ':']
        markers = ['o', 's', 'v', 'p', 'h', 'D']
        
        # Plot curves for each approach
        for i, approach in enumerate(approaches):
            if approach in mean_results:
                ratios_data = mean_results[approach]['ratios']
                gains = mean_results[approach]['efficiency_gains'] / (0.03523782 * 500)  # Normalize
                
                plt.plot(ratios_data, gains,
                        color=colors[i % len(colors)],
                        linestyle=linestyles[i % len(linestyles)],
                        marker=markers[i % len(markers)],
                        markevery=max(1, len(ratios_data)//8),
                        markersize=8,
                        alpha=0.8)
        
        plt.xlabel('Primary Model Usage Ratio')
        plt.ylabel('Efficiency Gain (Mean)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0.2, 0.8)  # Focus on 0.2 to 0.8
        
        # Determine y-axis limits based on data
        all_gains = []
        for approach in approaches:
            if approach in mean_results:
                all_gains.extend(mean_results[approach]['efficiency_gains'])
        
        if all_gains:
            y_min = min(min(all_gains) * 1.1 / (0.03523782 * 500), -0.05)
            y_max = max(all_gains) * 1.05 / (0.03523782 * 500)
            plt.ylim(y_min, y_max)
        
        # Add reference line at y=0 (Random level)
        plt.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
        
        # Add subtle background
        plt.gca().set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"🎯 Plot saved to: {output_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Generating plot from PC50_varstd_RF_100G_individual_runs.csv")
    create_plot()
    print("✨ Done!")