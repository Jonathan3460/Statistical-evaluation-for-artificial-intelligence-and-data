import os
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests


print("Loading evaluation matrix...")
df = pd.read_csv("results/evaluation_matrix_pilot.csv")

if 'Question_ID' in df.columns:
    df_data = df.drop(columns='Question_ID')
else:
    df_data = df

sample_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000]
n_iterations = 10000

alpha_omnibus = 0.05
prompts = df_data.columns.tolist()
pair_names = list(combinations(prompts, 2))
n_pairs = len(pair_names)
k_prompts = len(prompts)

power_results = {n: {pair: 0 for pair in pair_names} for n in sample_sizes}
omnibus_power = {n: 0 for n in sample_sizes}

np.random.seed(42)

data_matrix = df_data.values
n_total_questions = data_matrix.shape[0]

print(
    f"Running Strict Monte Carlo Simulation ({n_iterations} iterations per N)...")
for n in tqdm(sample_sizes):

    for _ in range(n_iterations):
        indices = np.random.randint(0, n_total_questions, size=n)
        sample = data_matrix[indices, :]

        q_result = cochrans_q(sample)

        if pd.isna(q_result.pvalue) or q_result.pvalue >= alpha_omnibus:
            continue

        omnibus_power[n] += 1

        p_raw = []
        for col_name_1, col_name_2 in pair_names:
            idx1 = prompts.index(col_name_1)
            idx2 = prompts.index(col_name_2)

            vec1 = sample[:, idx1]
            vec2 = sample[:, idx2]

            b00 = np.sum((vec1 == 0) & (vec2 == 0))
            b01 = np.sum((vec1 == 0) & (vec2 == 1))
            b10 = np.sum((vec1 == 1) & (vec2 == 0))
            b11 = np.sum((vec1 == 1) & (vec2 == 1))

            table = [[b00, b01],
                     [b10, b11]]

            discordant = b10 + b01
            use_exact = discordant < 25

            test = mcnemar(
                table,
                exact=use_exact,
                correction=not use_exact
            )

            p_raw.append(test.pvalue)

        reject, _, _, _ = multipletests(
            p_raw,
            alpha=alpha_omnibus,
            method="holm"
        )

        for k, pair in enumerate(pair_names):
            if reject[k]:
                power_results[n][pair] += 1

print("\n--- SIMULATED OMNIBUS POWER (COCHRAN'S Q) ---")
for n in sample_sizes:
    print(f"N={n:<4}: {(omnibus_power[n]/n_iterations)*100:.1f}%")

print("\n--- SIMULATED PAIRWISE POWER (McNemar + Holm) ---")
print(f"{'Pair':<35} | " + " | ".join([f"N={n:<3}" for n in sample_sizes]))
print("-" * 100)

for pair in pair_names:
    row_str = f"{pair[0][:15]} vs {pair[1][:15]:<15} | "
    for n in sample_sizes:
        power = power_results[n][pair] / n_iterations
        row_str += f"{power*100:>5.1f}% | "
    print(row_str)

# Plotting setup
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(pair_names)))

for idx, pair in enumerate(pair_names):
    ps = [power_results[n][pair] / n_iterations for n in sample_sizes]

    if max(ps) > 0.05:
        label_name = f"{pair[0]} vs {pair[1]}"
        plt.plot(sample_sizes, ps, marker='o', markersize=4,
                 linewidth=2, color=colors[idx], label=label_name)
    else:
        plt.plot(sample_sizes, ps, linestyle=':',
                 linewidth=1, color='gray', alpha=0.5)

plt.axhline(y=0.80, color='red', linestyle='--',
            linewidth=2.5, label='80% Target Power')

plt.title('Holm-Bonferroni Power by Prompt Pair (Conditional on Cochran\'s Q)',
          fontsize=14, fontweight='bold')
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
