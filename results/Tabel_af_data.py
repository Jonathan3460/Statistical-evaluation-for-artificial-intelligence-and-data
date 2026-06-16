import pandas as pd
from itertools import combinations
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

# ==========================================
# Load data
# ==========================================

df = pd.read_csv("results/evaluation_matrix_experiment.csv")

data = df.drop(columns=["Question_ID"])

# Accuracy for each prompt condition
accuracies = data.mean()

# ==========================================
# Cochran's Q Test
# ==========================================

q_result = cochrans_q(data.values)

print("\n========== COCHRAN'S Q TEST ==========")
print(f"Q statistic = {q_result.statistic:.4f}")
print(f"p-value     = {q_result.pvalue:.6e}")

alpha = 0.05

if q_result.pvalue >= alpha:
    print("\nNo significant difference detected.")
    quit()

print("\nSignificant difference detected.")
print("Proceeding with pairwise McNemar tests...\n")

# ==========================================
# Pairwise McNemar Tests
# ==========================================

results = []

for model1, model2 in combinations(data.columns, 2):

    a = ((data[model1] == 1) & (data[model2] == 1)).sum()
    b = ((data[model1] == 1) & (data[model2] == 0)).sum()
    c = ((data[model1] == 0) & (data[model2] == 1)).sum()
    d = ((data[model1] == 0) & (data[model2] == 0)).sum()

    contingency_table = [[a, b],
                         [c, d]]

    test = mcnemar(
        contingency_table,
        exact=False,
        correction=True
    )

    results.append({
        "Comparison": f"{model1} vs {model2}",
        "Accuracy A": accuracies[model1],
        "Accuracy B": accuracies[model2],
        "McNemar p-value": test.pvalue
    })

results_df = pd.DataFrame(results)

# ==========================================
# Holm-Bonferroni
# ==========================================

reject, p_adjusted, _, _ = multipletests(
    results_df["McNemar p-value"],
    alpha=0.05,
    method="holm"
)

results_df["Holm-adjusted p-value"] = p_adjusted
results_df["Significant"] = reject

results_df["Significant"] = results_df["Significant"].map(
    {True: "Yes", False: "No"}
)

# Sort by adjusted p-value
results_df = results_df.sort_values(
    by="Holm-adjusted p-value"
)

# Round values
results_df["Accuracy A"] = results_df["Accuracy A"].round(3)
results_df["Accuracy B"] = results_df["Accuracy B"].round(3)

# ==========================================
# Print Results
# ==========================================

print("\n========== MCNEMAR + HOLM-BONFERRONI ==========\n")

print(
    results_df.to_string(index=False)
)

# ==========================================
# Save CSV
# ==========================================

results_df.to_csv(
    "results/pairwise_mcnemar_holm_table.csv",
    index=False
)

print("\nSaved:")
print("results/pairwise_mcnemar_holm_table.csv")

# ==========================================
# Generate LaTeX table
# ==========================================

latex_table = results_df.to_latex(
    index=False,
    float_format="%.4g",
    caption="Pairwise McNemar tests with Holm--Bonferroni correction.",
    label="tab:mcnemar_holm"
)

print("\n========== LATEX TABLE ==========\n")
print(latex_table)