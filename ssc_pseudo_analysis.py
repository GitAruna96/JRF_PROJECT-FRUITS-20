import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =================================================================
# SETUP AND DATA LOADING (Using the path from your previous context)
# =================================================================
csv_path = 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\fruits_dataset\\paper_experiments\\results\\swin_t_ssc_pseudo_results.csv'
plot_dir = './results/plots_ieee_final_big'
os.makedirs(plot_dir, exist_ok=True)

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"WARNING: CSV file not found at {csv_path}. Generating dummy data for demonstration.")
    # --- DUMMY DATA GENERATION for 50 Epochs ---
    epochs = np.arange(1, 51)
    test_acc_base = 63.2
    train_acc_base = 85.0
    
    test_accuracy = [70, 75, 80, 85, test_acc_base] # Initial SSC-only phase
    test_accuracy.extend([test_acc_base * (0.95 - (i * 0.01)) + np.random.rand() for i in range(5)]) 
    test_accuracy.extend([50 + (i * 0.5) + np.random.rand() * 2 for i in range(20)]) 
    test_accuracy.extend([53 + np.random.rand() * 1 for i in range(20)]) 
    test_accuracy = test_accuracy[:50]
    
    train_accuracy = [80, 85, 90, 93, 95] 
    train_accuracy.extend([train_acc_base + (i * 0.1) + np.random.rand() for i in range(45)]) 
    train_accuracy = train_accuracy[:50]
    
    epoch_times_ssc = np.full(5, 71.1) + np.random.uniform(-0.5, 0.5, 5)
    epoch_times_pseudo = np.full(45, 96.3) + np.random.uniform(-0.5, 0.5, 45)
    epoch_times = np.concatenate([epoch_times_ssc, epoch_times_pseudo])
    
    data = {
        'Epoch': epochs,
        'Test_Accuracy': test_accuracy,
        'Train_Accuracy': train_accuracy,
        'Train_Loss': np.logspace(0, -2, 50) * 0.1, # CLS Loss
        'SSC_Loss': np.logspace(0, -2, 50) * 0.5,
        'Pseudo_Loss': [0] * 5 + np.logspace(0.5, -0.5, 45).tolist(),
        'Epoch_Time_s': epoch_times,
        'Cumulative_Time_s': np.cumsum(epoch_times)
    }
    df = pd.DataFrame(data)

# Recalculate key metrics
best_epoch_idx = df['Test_Accuracy'].idxmax()
best_acc = df['Test_Accuracy'].max()
worst_epoch_idx = df['Test_Accuracy'].idxmin()
worst_acc = df['Test_Accuracy'].min()
best_epoch = best_epoch_idx + 1
worst_epoch = worst_epoch_idx + 1

avg_ssc_time = df['Epoch_Time_s'].iloc[:5].mean()
avg_pseudo_time = df['Epoch_Time_s'].iloc[5:].mean()
time_increase = ((avg_pseudo_time - avg_ssc_time) / avg_ssc_time) * 100

# =================================================================
# FINAL IEEE STYLE CONFIGURATION (TEXT SIZE BIGGER)
# =================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,                    # BIGGER Base font size
    "axes.titlesize": 15,               # BIGGER Title size
    "axes.labelsize": 13,               # BIGGER Axis label size
    "xtick.labelsize": 11,              # BIGGER Tick label size
    "ytick.labelsize": 11,
    "legend.fontsize": 11,              # BIGGER Legend size
    "axes.linewidth": 1.2,              # Thicker axes lines
    "lines.linewidth": 3.0,             # Much THICKER lines
    "lines.markersize": 8,              # Larger markers
    "grid.linestyle": ':',
    "grid.alpha": 0.6,
    "grid.linewidth": 0.8,
})

def save_figure(fig, name):
    fig.savefig(f'{plot_dir}/{name}.png', dpi=600, bbox_inches='tight')
    fig.savefig(f'{plot_dir}/{name}.pdf', dpi=600, bbox_inches='tight')
    plt.close(fig)

# =================================================================
# 1. ENHANCED ACCURACY / COLLAPSE PLOT (Double-Column, Big Text)
# =================================================================
fig1, ax1 = plt.subplots(figsize=(7.0, 4.0)) 
mark_interval = 5 

ax1.plot(df['Epoch'], df['Test_Accuracy'], 'r-o', 
         markevery=mark_interval, 
         markerfacecolor='red', markeredgecolor='darkred',
         label='Target Accuracy (Test)')
ax1.plot(df['Epoch'], df['Train_Accuracy'], 'b-s', 
         markevery=mark_interval,
         markerfacecolor='blue', markeredgecolor='darkblue',
         label='Source Accuracy (Train)')

ax1.axvline(x=5.5, color='darkred', linestyle='--', linewidth=1.8, alpha=0.8, label='PL Activated')

ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_title('Catastrophic Collapse: SSC + Pseudo-Labeling Performance (FO→FP)')
ax1.legend(loc='lower center', ncol=3, frameon=True, edgecolor='black', framealpha=0.9,
           bbox_to_anchor=(0.5, -0.28)) 

ax1.grid(True, which='major')
ax1.set_xticks(np.arange(0, len(df['Epoch']) + 1, 5))
ax1.set_xlim(0.5, len(df) + 0.5)
ax1.set_ylim(15, 100)

# Enhanced Annotations
ax1.annotate(f'Best: {best_acc:.1f}% (E{best_epoch})', 
             xy=(best_epoch, best_acc), xytext=(best_epoch + 1, best_acc + 10),
             arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6, alpha=0.8),
             fontsize=11, color='green', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec='green', lw=1.0))

ax1.annotate(f'Collapse: {worst_acc:.1f}% (E{worst_epoch})', 
             xy=(worst_epoch, worst_acc), xytext=(worst_epoch + 15, worst_acc - 5),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6, alpha=0.8),
             fontsize=11, color='red', ha='left', va='center', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec='red', lw=1.0))

plt.tight_layout(rect=[0, 0.15, 1, 1]) 
save_figure(fig1, '1.ssc_pseudo_collapse_ieee_big')

# =================================================================
# 2. ENHANCED LOSS COMPONENTS PLOT (Double-Column, Big Text)
# =================================================================
fig2, ax2 = plt.subplots(figsize=(7.0, 4.0)) 

# Distinct line styles for clear differentiation
ax2.plot(df['Epoch'], df['Train_Loss'], 'g-', label='CLS Loss', linewidth=3.0)
ax2.plot(df['Epoch'], df['SSC_Loss'], 'b--', label='SSC Loss', linewidth=3.0)
ax2.plot(df['Epoch'], df['Pseudo_Loss'], 'r:', label='Pseudo Loss', linewidth=3.0)
ax2.axvline(x=5.5, color='darkred', linestyle='--', linewidth=1.8, alpha=0.8)

ax2.set_xlabel('Epoch', fontweight='bold')
ax2.set_ylabel('Loss Value', fontweight='bold')
ax2.set_title('Loss Components: SSC + Pseudo-Labeling Training Dynamics')

ax2.set_yscale('log')  
ax2.legend(loc='upper right', frameon=True, edgecolor='black', framealpha=0.9, ncol=1)
ax2.grid(True)
ax2.set_xticks(np.arange(0, len(df['Epoch']) + 1, 5))
ax2.set_xlim(0.5, len(df) + 0.5)

ax2.text(6.0, ax2.get_ylim()[1]*0.5, 'PL Activated', color='darkred', 
         fontsize=11, ha='left', va='top', transform=ax2.transData, fontweight='bold')

plt.tight_layout()
save_figure(fig2, '2.loss_components_ieee_big')

# =================================================================
# 3. ENHANCED TIMING ANALYSIS PLOT (Double-Column, Big Text)
# =================================================================
fig3, ax3 = plt.subplots(figsize=(7.0, 4.0)) 
width = 0.8 

# Colors: Dark Blue for SSC-only, Red for SSC+Pseudo
bar_colors = ['#1f77b4'] * 5 + ['#d62728'] * (len(df) - 5) 

ax3.bar(df['Epoch'], df['Epoch_Time_s'], width=width, 
        color=bar_colors, 
        edgecolor='black', 
        linewidth=0.5, 
        alpha=0.9)

ax3.axhline(y=avg_ssc_time, color='green', linestyle='--', linewidth=2.5,
            label=f'SSC-only avg: {avg_ssc_time:.1f}s')
ax3.axhline(y=avg_pseudo_time, color='orange', linestyle='-', linewidth=2.5,
            label=f'SSC+Pseudo avg: {avg_pseudo_time:.1f}s')

ax3.set_xticks(np.arange(0, len(df['Epoch']) + 1, 5))
ax3.tick_params(axis='x', which='minor', length=2, color='gray') 

ax3.set_xlabel('Epoch', fontweight='bold')
ax3.set_ylabel('Time per Epoch (seconds)', fontweight='bold')
ax3.set_title(f'Computational Overhead of Pseudo-Labeling')

ax3.legend(loc='upper left', frameon=True, edgecolor='black', framealpha=0.9)
ax3.grid(axis='y', linestyle='--', alpha=0.6) 
ax3.set_xlim(0.5, len(df) + 0.5)
ax3.set_ylim(0, 110) # Adjusted Y-axis for better fit

# BOLD, prominent overhead annotation
ax3.text(0.98, 0.98, 
         f'Overhead: +{time_increase:.1f}%', 
         transform=ax3.transAxes, 
         fontsize=12, 
         color='darkred', 
         ha='right', 
         va='top', 
         fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9, ec='darkred', lw=1.5))

ax3.text(5.5, 5, 'PL Activated', 
         color='darkred', 
         fontsize=11, 
         ha='left', 
         va='bottom', 
         rotation=90,
         fontweight='bold')

plt.tight_layout()
save_figure(fig3, '3.timing_overhead_ieee_big')

print("\n" + "="*70)
print(f"ALL IEEE-ENHANCED FIGURES (BIG TEXT) SAVED SUCCESSFULLY! Check folder: {plot_dir}")
print("="*70)