import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("results/evaluation_matrix_experiment.csv")
if 'Question_ID' in df.columns:
    df = df.set_index('Question_ID')

prompts = df.columns.tolist()
n_iterations = 10000

np.random.seed(42)
data_matrix = df.values
n_samples = len(df)

print("Running 10,000 bootstrap iterations...")
indices = np.random.randint(0, n_samples, size=(n_iterations, n_samples))
bootstrapped_means = data_matrix[indices].mean(axis=1)

plot_data = pd.DataFrame(bootstrapped_means, columns=prompts)

sorted_prompts = plot_data.mean().sort_values().index.tolist()
plot_data = plot_data[sorted_prompts]

print("\n--- Bootstrapped 95% Confidence Intervals ---")
for prompt in sorted_prompts:
    means_array = plot_data[prompt].values
    mean_val = np.mean(means_array)
    lower_bound = np.percentile(means_array, 2.5)
    upper_bound = np.percentile(means_array, 97.5)
    print(
        f"{prompt}: {mean_val*100:.1f}% [95% CI: {lower_bound*100:.1f}% - {upper_bound*100:.1f}%]")

# Generate Visualization
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.violinplot(data=plot_data, inner="quartile", palette="viridis")
plt.title('Stability of AI Accuracy: Bootstrapped Distributions by Prompt Style\n(10,000 Resamples)', fontsize=14, pad=15)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Prompt Style', fontsize=12)
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
plt.xticks(rotation=15, fontsize=11)
plt.tight_layout()
plt.savefig('bootstrapped_variance_plot.png', dpi=300)
