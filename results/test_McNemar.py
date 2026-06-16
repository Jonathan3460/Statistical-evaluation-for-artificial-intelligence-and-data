import pandas as pd
from itertools import combinations
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

# ==========================================
# Load data
# ==========================================

df = pd.read_csv("results/evaluation_matrix_experiment.csv")

data = df.drop(columns=["Question_ID"])

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
    print("McNemar tests not required.")
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

    table = [[a, b],
             [c, d]]

    test = mcnemar(
        table,
        exact=False,
        correction=True
    )

    results.append({
        "Comparison": f"{model1} vs {model2}",
        "Statistic": test.statistic,
        "p_raw": test.pvalue,
        "b": b,
        "c": c
    })

results_df = pd.DataFrame(results)

# ==========================================
# Holm-Bonferroni Correction
# ==========================================

reject, p_adjusted, _, _ = multipletests(
    results_df["p_raw"],
    alpha=0.05,
    method="holm"
)

results_df["p_holm"] = p_adjusted
results_df["Reject_H0"] = reject

results_df = results_df.sort_values("p_raw")

# ==========================================
# Print Results
# ==========================================

print("========== MCNEMAR + HOLM-BONFERRONI ==========\n")

print(
    results_df[
        [
            "Comparison",
            "Statistic",
            "p_raw",
            "p_holm",
            "Reject_H0"
        ]
    ].to_string(index=False)
)

# ==========================================
# Save Results
# ==========================================

results_df.to_csv(
    "results/mcnemar_holm_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/mcnemar_holm_results.csv")