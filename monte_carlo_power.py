import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from itertools import combinations
from tqdm import tqdm
import os

print("Loading evaluation matrix...")
df = pd.read_csv("results/evaluation_matrix_gemma4_e4b_8bit.csv")

if 'Question_ID' in df.columns:
    df_data = df.drop(columns='Question_ID')
else:
    df_data = df

sample_sizes = sample_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000]
n_iterations = 10000

alpha_omnibus = 0.05
prompts = df_data.columns.tolist()
pair_names = list(combinations(prompts, 2))
n_pairs = len(pair_names)

power_results = {n: {pair: 0 for pair in pair_names} for n in sample_sizes}

np.random.seed(42)

print(f"Running Pairwise Monte Carlo Simulation ({n_iterations} iterations per N)...")
for n in tqdm(sample_sizes):
    
    for _ in range(n_iterations):
        sample = df_data.sample(n=n, replace=True)
        
        p_vals_with_labels = []
        for col1, col2 in pair_names:
            b = sum((sample[col1] == 1) & (sample[col2] == 0))
            c = sum((sample[col1] == 0) & (sample[col2] == 1))
            
            if (b + c) > 0:
                p_val = binomtest(min(b, c), b + c, 0.5).pvalue
            else:
                p_val = 1.0
                
            p_vals_with_labels.append((p_val, (col1, col2)))
            
        p_vals_with_labels.sort(key=lambda x: x[0])
        
        for k, (p_val, pair) in enumerate(p_vals_with_labels):
            threshold = alpha_omnibus / (n_pairs - k)
            
            if p_val < threshold:
                power_results[n][pair] += 1
            else:
                break

print("\n--- SIMULATED POWER BY PAIR ---")
print(f"{'Pair':<35} | " + " | ".join([f"N={n:<3}" for n in sample_sizes]))
print("-" * 100)

for pair in pair_names:
    row_str = f"{pair[0][:15]} vs {pair[1][:15]:<15} | "
    for n in sample_sizes:
        power = power_results[n][pair] / n_iterations
        row_str += f"{power*100:>5.1f}% | "
    print(row_str)

plt.figure(figsize=(12, 8))

colors = plt.cm.tab10(np.linspace(0, 1, len(pair_names)))

for idx, pair in enumerate(pair_names):
    ps = [power_results[n][pair] / n_iterations for n in sample_sizes]
    
    if max(ps) > 0.05:
        label_name = f"{pair[0]} vs {pair[1]}"
        plt.plot(sample_sizes, ps, marker='o', markersize=4, linewidth=2, color=colors[idx], label=label_name)
    else:
        plt.plot(sample_sizes, ps, linestyle=':', linewidth=1, color='gray', alpha=0.5)

plt.axhline(y=0.80, color='red', linestyle='--', linewidth=2.5, label='80% Target Power')

plt.title('Holm-Bonferroni Power by Prompt Pair (Detecting Specific Differences)', fontsize=14, fontweight='bold')
plt.xlabel('Sample Size (N Questions)', fontsize=12)
plt.ylabel('Statistical Power', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.7)

plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=10)

plt.tight_layout()

os.makedirs("results", exist_ok=True)
plot_filename = "results/monte_carlo_power_curve_detailed.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"\nSuccess! Detailed power curves saved to '{plot_filename}'")