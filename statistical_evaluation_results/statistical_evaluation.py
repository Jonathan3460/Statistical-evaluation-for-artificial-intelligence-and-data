"""
Statistical evaluation of prompt sensitivity in matched binary MMLU results.

Input:
    evaluation_matrix_experiment.csv
    One row per question, one binary column per prompt condition.

Analyses:
    1. Descriptive accuracy per prompt
    2. Cochran's Q omnibus test for >=3 matched binary conditions
    3. Pairwise exact McNemar tests with Holm-Bonferroni correction
    4. Paired non-parametric bootstrap confidence intervals for:
       - each prompt accuracy
       - each pairwise accuracy difference

Outputs:
    results/accuracy_summary.csv
    results/cochran_q_result.txt
    results/pairwise_mcnemar_holm.csv
    results/bootstrap_pairwise_ci.csv
    results/statistical_evaluation_report.md
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest
from statsmodels.stats.multitest import multipletests


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("/mnt/data/evaluation_matrix_experiment.csv")
OUTPUT_DIR = Path("/mnt/data/statistical_evaluation_results")

ID_COLUMNS = {"Question_ID", "question_id", "id", "question", "Question"}
ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 10_000
RANDOM_SEED = 42


# -----------------------------
# Helper functions
# -----------------------------
def load_binary_matrix(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load the experiment matrix and identify binary prompt columns."""
    data = pd.read_csv(path)

    prompt_columns = [
        col for col in data.columns
        if col not in ID_COLUMNS and set(data[col].dropna().unique()).issubset({0, 1})
    ]

    if len(prompt_columns) < 3:
        raise ValueError(
            "Cochran's Q requires at least 3 matched binary prompt columns. "
            f"Detected columns: {prompt_columns}"
        )

    if data[prompt_columns].isna().any().any():
        missing = data[prompt_columns].isna().sum()
        raise ValueError(
            "Missing values found in prompt columns. "
            "Either fix parsing or decide on an exclusion rule.\n"
            f"{missing[missing > 0]}"
        )

    return data, prompt_columns


def cochran_q_test(binary_matrix: np.ndarray) -> dict[str, float]:
    """
    Cochran's Q test for matched binary data.

    Rows are questions/blocks.
    Columns are prompt conditions/treatments.

    Q = (k - 1) * (k * sum_j(C_j^2) - T^2) / (k*T - sum_i(R_i^2))

    where:
        k   = number of prompt conditions
        C_j = total correct for prompt j
        R_i = total correct for question i across prompts
        T   = total correct responses
    """
    x = np.asarray(binary_matrix, dtype=int)
    n_questions, k_prompts = x.shape

    column_totals = x.sum(axis=0)
    row_totals = x.sum(axis=1)
    total_correct = column_totals.sum()

    denominator = k_prompts * total_correct - np.sum(row_totals ** 2)
    if denominator == 0:
        raise ValueError(
            "Cochran's Q denominator is zero. "
            "This happens if there is no within-question variation across prompts."
        )

    statistic = (
        (k_prompts - 1)
        * (k_prompts * np.sum(column_totals ** 2) - total_correct ** 2)
        / denominator
    )
    degrees_of_freedom = k_prompts - 1
    p_value = chi2.sf(statistic, degrees_of_freedom)

    return {
        "n_questions": n_questions,
        "n_prompts": k_prompts,
        "statistic": statistic,
        "df": degrees_of_freedom,
        "p_value": p_value,
    }


def exact_mcnemar_from_vectors(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """
    Exact two-sided McNemar test using only discordant pairs.

    b01 = A incorrect, B correct
    b10 = A correct, B incorrect

    Under H0, discordant outcomes are equally likely in either direction.
    The exact p-value is a two-sided binomial test with p=0.5.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)

    a_correct_b_wrong = int(np.sum((a == 1) & (b == 0)))
    a_wrong_b_correct = int(np.sum((a == 0) & (b == 1)))
    discordant = a_correct_b_wrong + a_wrong_b_correct

    if discordant == 0:
        p_value = 1.0
    else:
        p_value = binomtest(
            k=min(a_correct_b_wrong, a_wrong_b_correct),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "A_correct_B_wrong": a_correct_b_wrong,
        "A_wrong_B_correct": a_wrong_b_correct,
        "discordant": discordant,
        "p_value_raw": p_value,
    }


def pairwise_mcnemar_holm(data: pd.DataFrame, prompt_columns: list[str]) -> pd.DataFrame:
    """Run all pairwise exact McNemar tests and apply Holm-Bonferroni correction."""
    rows = []

    for prompt_a, prompt_b in combinations(prompt_columns, 2):
        a = data[prompt_a].to_numpy(dtype=int)
        b = data[prompt_b].to_numpy(dtype=int)

        test = exact_mcnemar_from_vectors(a, b)
        rows.append({
            "prompt_A": prompt_a,
            "prompt_B": prompt_b,
            "accuracy_A": a.mean(),
            "accuracy_B": b.mean(),
            "accuracy_difference_A_minus_B": a.mean() - b.mean(),
            **test,
        })

    results = pd.DataFrame(rows)

    reject, adjusted_p, _, _ = multipletests(
        results["p_value_raw"],
        alpha=ALPHA,
        method="holm",
    )

    results["p_value_holm"] = adjusted_p
    results["significant_holm_0.05"] = reject

    return results.sort_values(["p_value_holm", "prompt_A", "prompt_B"]).reset_index(drop=True)


def bootstrap_accuracy_ci(
    data: pd.DataFrame,
    prompt_columns: list[str],
    n_iterations: int,
    seed: int,
    alpha: float,
) -> pd.DataFrame:
    """Paired bootstrap confidence intervals for prompt accuracies."""
    rng = np.random.default_rng(seed)
    x = data[prompt_columns].to_numpy(dtype=int)
    n = x.shape[0]

    bootstrap_means = np.empty((n_iterations, len(prompt_columns)))

    for i in range(n_iterations):
        sample_indices = rng.integers(0, n, size=n)
        bootstrap_means[i, :] = x[sample_indices, :].mean(axis=0)

    lower = np.quantile(bootstrap_means, alpha / 2, axis=0)
    upper = np.quantile(bootstrap_means, 1 - alpha / 2, axis=0)

    return pd.DataFrame({
        "prompt": prompt_columns,
        "accuracy": x.mean(axis=0),
        "ci_lower": lower,
        "ci_upper": upper,
    }).sort_values("accuracy", ascending=False).reset_index(drop=True)


def bootstrap_pairwise_difference_ci(
    data: pd.DataFrame,
    prompt_columns: list[str],
    n_iterations: int,
    seed: int,
    alpha: float,
) -> pd.DataFrame:
    """
    Paired bootstrap confidence intervals for accuracy differences.

    Rows/questions are resampled as complete matched blocks, preserving the
    dependence between prompt outcomes for the same question.
    """
    rng = np.random.default_rng(seed)
    x = data[prompt_columns].to_numpy(dtype=int)
    n = x.shape[0]

    rows = []

    for prompt_a, prompt_b in combinations(prompt_columns, 2):
        idx_a = prompt_columns.index(prompt_a)
        idx_b = prompt_columns.index(prompt_b)

        observed_difference = x[:, idx_a].mean() - x[:, idx_b].mean()
        bootstrap_differences = np.empty(n_iterations)

        for i in range(n_iterations):
            sample_indices = rng.integers(0, n, size=n)
            sample = x[sample_indices, :]
            bootstrap_differences[i] = sample[:, idx_a].mean() - sample[:, idx_b].mean()

        rows.append({
            "prompt_A": prompt_a,
            "prompt_B": prompt_b,
            "accuracy_difference_A_minus_B": observed_difference,
            "ci_lower": np.quantile(bootstrap_differences, alpha / 2),
            "ci_upper": np.quantile(bootstrap_differences, 1 - alpha / 2),
        })

    return pd.DataFrame(rows)


def format_p(p: float) -> str:
    """Readable p-value formatting."""
    if p < 0.001:
        return f"{p:.3e}"
    return f"{p:.4f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, prompt_columns = load_binary_matrix(DATA_PATH)
    x = data[prompt_columns].to_numpy(dtype=int)

    # Descriptive accuracy table
    accuracy_summary = bootstrap_accuracy_ci(
        data=data,
        prompt_columns=prompt_columns,
        n_iterations=BOOTSTRAP_ITERATIONS,
        seed=RANDOM_SEED,
        alpha=ALPHA,
    )
    accuracy_summary.to_csv(OUTPUT_DIR / "accuracy_summary.csv", index=False)

    # Cochran's Q
    q_result = cochran_q_test(x)
    with open(OUTPUT_DIR / "cochran_q_result.txt", "w", encoding="utf-8") as f:
        f.write("Cochran's Q omnibus test\n")
        f.write(f"N questions: {q_result['n_questions']}\n")
        f.write(f"Prompt conditions: {q_result['n_prompts']}\n")
        f.write(f"Q statistic: {q_result['statistic']:.6f}\n")
        f.write(f"df: {q_result['df']}\n")
        f.write(f"p-value: {q_result['p_value']:.12g}\n")

    # Pairwise McNemar + Holm
    mcnemar_results = pairwise_mcnemar_holm(data, prompt_columns)
    mcnemar_results.to_csv(OUTPUT_DIR / "pairwise_mcnemar_holm.csv", index=False)

    # Pairwise bootstrap CI
    pairwise_ci = bootstrap_pairwise_difference_ci(
        data=data,
        prompt_columns=prompt_columns,
        n_iterations=BOOTSTRAP_ITERATIONS,
        seed=RANDOM_SEED + 1,
        alpha=ALPHA,
    )
    pairwise_ci.to_csv(OUTPUT_DIR / "bootstrap_pairwise_ci.csv", index=False)

    # Merge inferential and CI results for easier reporting
    combined_pairwise = mcnemar_results.merge(
        pairwise_ci,
        on=["prompt_A", "prompt_B", "accuracy_difference_A_minus_B"],
        how="left",
    )

    # Markdown report
    report_lines = []
    report_lines.append("# Statistical evaluation of prompt conditions\n")
    report_lines.append("## Descriptive accuracy\n")
    report_lines.append(accuracy_summary.to_markdown(index=False, floatfmt=".4f"))
    report_lines.append("\n## Cochran's Q omnibus test\n")
    report_lines.append(
        f"Cochran's Q({int(q_result['df'])}) = {q_result['statistic']:.3f}, "
        f"p = {format_p(q_result['p_value'])}."
    )

    if q_result["p_value"] < ALPHA:
        report_lines.append(
            "\nConclusion: reject H0. At least one prompt condition has a different probability of a correct answer."
        )
    else:
        report_lines.append(
            "\nConclusion: do not reject H0. There is not sufficient evidence that the prompt conditions differ."
        )

    report_lines.append("\n## Pairwise exact McNemar tests with Holm-Bonferroni correction\n")
    report_lines.append(combined_pairwise.to_markdown(index=False, floatfmt=".4f"))

    significant = combined_pairwise[combined_pairwise["significant_holm_0.05"]]
    report_lines.append("\n## Holm-significant pairwise differences\n")
    if significant.empty:
        report_lines.append("No pairwise differences remained significant after Holm-Bonferroni correction.")
    else:
        for _, row in significant.iterrows():
            direction = row["prompt_A"] if row["accuracy_difference_A_minus_B"] > 0 else row["prompt_B"]
            report_lines.append(
                f"- {row['prompt_A']} vs {row['prompt_B']}: "
                f"difference A-B = {row['accuracy_difference_A_minus_B']:.4f}, "
                f"95% bootstrap CI [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
                f"Holm p = {format_p(row['p_value_holm'])}. "
                f"Higher accuracy: {direction}."
            )

    (OUTPUT_DIR / "statistical_evaluation_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # Console output
    print("\nPrompt columns:", ", ".join(prompt_columns))
    print("\nAccuracy summary:")
    print(accuracy_summary.to_string(index=False))
    print("\nCochran's Q:")
    print(
        f"Q({int(q_result['df'])}) = {q_result['statistic']:.3f}, "
        f"p = {format_p(q_result['p_value'])}"
    )
    print("\nPairwise McNemar + Holm:")
    print(combined_pairwise.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
