from itertools import combinations
from pathlib import Path

import pandas as pd

from statsmodels.stats.contingency_tables import (
    cochrans_q,
    mcnemar
)

from statsmodels.stats.multitest import multipletests


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path(
    "results/evaluation_matrix_experiment.csv"
)

ALPHA = 0.05


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

data = df.drop(columns=["Question_ID"])

prompt_columns = data.columns.tolist()


# -----------------------------
# Cochran's Q assumptions
# -----------------------------
k = len(prompt_columns)

informative_rows = data.nunique(axis=1) > 1
n = informative_rows.sum()

print("\n========== COCHRAN'S Q ASSUMPTIONS ==========")
print(f"Prompt conditions (k): {k}")
print(f"Informative questions (n): {n}")
print(f"n * k = {n * k}")

if n >= 4:
    print("PASS: n >= 4")
else:
    print("WARNING: n < 4")

if n * k >= 24:
    print("PASS: n*k >= 24")
else:
    print("WARNING: n*k < 24")


# -----------------------------
# Cochran's Q test
# -----------------------------
q_result = cochrans_q(data.values)

print("\n========== COCHRAN'S Q TEST ==========")
print(f"Q statistic = {q_result.statistic:.4f}")
print(f"p-value     = {q_result.pvalue:.6g}")

if q_result.pvalue < ALPHA:
    print("\nReject H0")
    print("At least one prompt style differs.")
else:
    print("\nFail to reject H0")
    print("No significant differences detected.")


# -----------------------------
# Pairwise McNemar tests
# -----------------------------
results = []

for prompt_a, prompt_b in combinations(prompt_columns, 2):

    a = data[prompt_a]
    b = data[prompt_b]

    table = pd.crosstab(a, b)

    table = table.reindex(
        index=[0, 1],
        columns=[0, 1],
        fill_value=0
    )

    # Discordant pairs
    b10 = table.loc[1, 0]  # A correct, B wrong
    b01 = table.loc[0, 1]  # A wrong, B correct

    discordant = b10 + b01

    # Exact McNemar for small samples
    use_exact = discordant < 25

    test = mcnemar(
        table,
        exact=use_exact,
        correction=not use_exact
    )

    results.append({
        "Comparison": f"{prompt_a} vs {prompt_b}",
        "Accuracy A": round(a.mean(), 3),
        "Accuracy B": round(b.mean(), 3),
        "Discordant": discordant,
        "Method": "Exact" if use_exact else "Chi-square",
        "p_raw": test.pvalue
    })


results = pd.DataFrame(results)


# -----------------------------
# Holm-Bonferroni correction
# -----------------------------
reject, p_holm, _, _ = multipletests(
    results["p_raw"],
    alpha=ALPHA,
    method="holm"
)

results["Holm p"] = p_holm
results["Significant"] = reject


# -----------------------------
# Formatting
# -----------------------------
results = results.sort_values("Holm p")

results["Holm p"] = results["Holm p"].apply(
    lambda p: "<0.0001" if p < 0.0001 else f"{p:.4f}"
)

results["Significant"] = results["Significant"].map(
    {True: "Yes", False: "No"}
)


# -----------------------------
# Final results table
# -----------------------------
print("\n========== MCNEMAR + HOLM-BONFERRONI ==========\n")

print(
    results[
        [
            "Comparison",
            "Accuracy A",
            "Accuracy B",
            "Discordant",
            "Method",
            "Holm p",
            "Significant"
        ]
    ].to_string(index=False)
)