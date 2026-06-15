"""
Prompt Sensitivity Visualisation Script
========================================
Generates four report-ready figures for the MMLU prompt sensitivity experiment.

Figures produced:
    1. Bar chart with bootstrapped 95% confidence intervals  (main figure)
    2. Heatmap of the binary evaluation matrix
    3. Pairwise agreement matrix between prompt styles
    4. Bootstrap distribution plot (overlapping density curves)

Usage:
    - Place your evaluation CSV alongside this script, or change DATA_PATH below.
    - The CSV can be wide format (one column per prompt) or long format
      (columns: question_id, prompt_style, correct).
    - Run:  python visualisation.py
    - All figures are saved as PNG (300 dpi) and PDF in ./figures/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde

# ── Configuration ────────────────────────────────────────────────────────────

DATA_PATH      = r"C:\Users\victo\Desktop\Programmering eksamen\Statistical-evaluation-for-artificial-intelligence-and-data\results\evaluation_matrix_experiment.csv"
OUTPUT_DIR     = Path(r"C:\Users\victo\Desktop\Programmering eksamen\Statistical-evaluation-for-artificial-intelligence-and-data\figures")
N_BOOTSTRAP    = 5000          # number of bootstrap resamples
RANDOM_SEED    = 42
CI_LEVEL       = 0.95          # confidence interval level
FIGURE_DPI     = 300

# Colour palette — one distinct colour per prompt style (colour-blind safe)
PROMPT_COLOURS = {
    "Baseline":      "#2C6E9E",   # steel blue
    "Noise":         "#E07B39",   # burnt orange
    "Structured":    "#3BAA72",   # sage green
    "Yes_Man":       "#C0392B",   # red  (worst performer)
    "Zero_Shot_CoT": "#7B4EA0",   # purple (best performer)
}

# Clean display labels for axes / legends
PROMPT_LABELS = {
    "Baseline":      "Baseline",
    "Noise":         "Noise",
    "Structured":    "Structured",
    "Yes_Man":       "Yes-Man",
    "Zero_Shot_CoT": "Zero-Shot CoT",
}

# Global matplotlib style — clean, academic
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         FIGURE_DPI,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_wide(path: str) -> pd.DataFrame:
    """
    Load and normalise the evaluation matrix.
    Accepts wide format (one column per prompt) or long format
    (columns: question_id, prompt_style, correct).
    Returns a wide DataFrame with Question_ID as index.
    """
    df = pd.read_csv(path)

    # Detect long format: must have exactly these three columns
    long_cols = {"question_id", "prompt_style", "correct"}
    if long_cols.issubset({c.lower() for c in df.columns}):
        df.columns = [c.lower() for c in df.columns]
        df = df.pivot(index="question_id", columns="prompt_style", values="correct")
        df.columns.name = None
        df = df.reset_index()

    # Normalise the ID column name
    id_col = [c for c in df.columns if "id" in c.lower() or "question" in c.lower()]
    if id_col:
        df = df.rename(columns={id_col[0]: "Question_ID"})
        df = df.set_index("Question_ID")
    else:
        df.index.name = "Question_ID"

    return df.astype(int)


df = load_wide(DATA_PATH)
prompt_styles = df.columns.tolist()

print(f"Loaded {len(df)} questions × {len(prompt_styles)} prompt styles")
print(f"Prompt styles: {prompt_styles}\n")


# ── Bootstrap utility ─────────────────────────────────────────────────────────

def bootstrap_ci(data: np.ndarray, n_resamples: int = N_BOOTSTRAP,
                 ci: float = CI_LEVEL) -> tuple[float, float, np.ndarray]:
    """
    Non-parametric bootstrap confidence interval for the mean of binary data.

    Parameters
    ----------
    data       : 1-D array of 0/1 values
    n_resamples: number of bootstrap iterations
    ci         : confidence level (default 0.95)

    Returns
    -------
    lower, upper : CI bounds
    boot_means   : full array of bootstrap means (for distribution plots)
    """
    n = len(data)
    # Draw all resamples at once for speed
    indices     = rng.integers(0, n, size=(n_resamples, n))
    boot_means  = data[indices].mean(axis=1)
    alpha       = (1 - ci) / 2
    lower       = np.percentile(boot_means, 100 * alpha)
    upper       = np.percentile(boot_means, 100 * (1 - alpha))
    return lower, upper, boot_means


# Compute stats for every prompt style
stats = {}
for ps in prompt_styles:
    arr             = df[ps].values.astype(float)
    lo, hi, boots   = bootstrap_ci(arr)
    stats[ps] = {
        "mean":       arr.mean(),
        "ci_lower":   lo,
        "ci_upper":   hi,
        "ci_width":   hi - lo,
        "boots":      boots,
    }

# Print summary table
print("=" * 60)
print(f"{'Prompt Style':<18} {'Mean':>6} {'Lower CI':>9} {'Upper CI':>9} {'CI Width':>9}")
print("-" * 60)
for ps, s in stats.items():
    label = PROMPT_LABELS.get(ps, ps)
    print(f"{label:<18} {s['mean']:>6.3f} {s['ci_lower']:>9.3f} "
          f"{s['ci_upper']:>9.3f} {s['ci_width']:>9.3f}")
print("=" * 60, "\n")


# ── Figure 1: Bar chart with bootstrapped CIs ─────────────────────────────────

def save(fig: plt.Figure, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}", dpi=FIGURE_DPI,
                    bbox_inches="tight")
    print(f"  Saved: {name}.png / .pdf")


fig1, ax1 = plt.subplots(figsize=(8, 5))

labels  = [PROMPT_LABELS.get(ps, ps) for ps in prompt_styles]
means   = [stats[ps]["mean"]     for ps in prompt_styles]
lowers  = [stats[ps]["ci_lower"] for ps in prompt_styles]
uppers  = [stats[ps]["ci_upper"] for ps in prompt_styles]
colours = [PROMPT_COLOURS.get(ps, "#777777") for ps in prompt_styles]

x      = np.arange(len(prompt_styles))
err_lo = [m - lo for m, lo in zip(means, lowers)]
err_hi = [hi - m for m, hi in zip(means, uppers)]

bars = ax1.bar(
    x, means,
    color=colours, alpha=0.85, width=0.55,
    yerr=[err_lo, err_hi],
    capsize=5, error_kw={"linewidth": 1.5, "capthick": 1.5, "color": "black"},
    zorder=3,
)

# Annotate each bar with its mean value
for bar, mean in zip(bars, means):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(err_hi) * 0.15,
        f"{mean:.1%}",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold"
    )

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Accuracy (proportion correct)")
ax1.set_xlabel("Prompt Style")
ax1.set_title(
    "LLM Accuracy by Prompt Style\n"
    f"(N = {len(df)} questions, error bars = {int(CI_LEVEL*100)}% bootstrapped CI, "
    f"B = {N_BOOTSTRAP:,})",
    pad=12
)
ax1.set_ylim(0, 1.08)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

# Reference line at chance level (4 options → 25%)
ax1.axhline(0.25, linestyle=":", color="grey", linewidth=1, label="Chance (25%)")
ax1.legend(loc="lower right")

fig1.tight_layout()
save(fig1, "fig1_bar_bootstrap_ci")
plt.close(fig1)


# ── Figure 2: Heatmap of evaluation matrix ────────────────────────────────────

# Sort questions by total number of correct answers (most correct at top)
df_sorted = df.copy()
df_sorted["total_correct"] = df_sorted.sum(axis=1)
df_sorted = df_sorted.sort_values("total_correct", ascending=False).drop(
    columns="total_correct"
)

fig2, ax2 = plt.subplots(figsize=(7, 10))

cmap = sns.color_palette(["#C0392B", "#2C6E9E"], as_cmap=True)  # red=wrong, blue=correct

sns.heatmap(
    df_sorted,
    ax=ax2,
    cmap=cmap,
    cbar=True,
    xticklabels=[PROMPT_LABELS.get(ps, ps) for ps in prompt_styles],
    yticklabels=False,          # 400 rows — labels would be unreadable
    linewidths=0,
    vmin=0, vmax=1,
)

# Custom colour bar labels
cbar = ax2.collections[0].colorbar
cbar.set_ticks([0.25, 0.75])
cbar.set_ticklabels(["Incorrect (0)", "Correct (1)"])

ax2.set_title(
    "Evaluation Matrix — Correctness by Question and Prompt Style\n"
    "(sorted by total correct answers, highest at top)",
    pad=10
)
ax2.set_xlabel("Prompt Style")
ax2.set_ylabel(f"Questions (N = {len(df)}, sorted)")

fig2.tight_layout()
save(fig2, "fig2_heatmap_evaluation_matrix")
plt.close(fig2)


# ── Figure 3: Pairwise agreement matrix ───────────────────────────────────────

n_ps    = len(prompt_styles)
agree   = np.zeros((n_ps, n_ps))

for i, ps1 in enumerate(prompt_styles):
    for j, ps2 in enumerate(prompt_styles):
        # Agreement = proportion of questions where both gave the same answer
        agree[i, j] = (df[ps1] == df[ps2]).mean()

agree_df = pd.DataFrame(
    agree,
    index=[PROMPT_LABELS.get(ps, ps) for ps in prompt_styles],
    columns=[PROMPT_LABELS.get(ps, ps) for ps in prompt_styles],
)

fig3, ax3 = plt.subplots(figsize=(6, 5))

sns.heatmap(
    agree_df,
    ax=ax3,
    annot=True, fmt=".1%",
    cmap="Blues",
    vmin=0.5, vmax=1.0,
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Agreement (proportion of questions)"},
)

ax3.set_title(
    "Pairwise Agreement Between Prompt Styles\n"
    "(proportion of questions with identical correctness)",
    pad=10
)
ax3.set_xlabel("Prompt Style")
ax3.set_ylabel("Prompt Style")
ax3.tick_params(axis="x", rotation=30)
ax3.tick_params(axis="y", rotation=0)

fig3.tight_layout()
save(fig3, "fig3_agreement_matrix")
plt.close(fig3)


# ── Figure 4: Bootstrap distribution plot ────────────────────────────────────

fig4, ax4 = plt.subplots(figsize=(8, 5))

for ps in prompt_styles:
    boot_means = stats[ps]["boots"]
    colour     = PROMPT_COLOURS.get(ps, "#777777")
    label      = PROMPT_LABELS.get(ps, ps)

    # Fit a kernel density estimate for a smooth distribution curve
    kde    = gaussian_kde(boot_means, bw_method="scott")
    x_grid = np.linspace(boot_means.min() - 0.02, boot_means.max() + 0.02, 400)
    y_kde  = kde(x_grid)

    ax4.plot(x_grid, y_kde, color=colour, linewidth=2, label=label)
    ax4.fill_between(x_grid, y_kde, alpha=0.12, color=colour)

    # Mark the observed mean
    ax4.axvline(stats[ps]["mean"], color=colour, linewidth=1,
                linestyle="--", alpha=0.7)

ax4.set_xlabel("Bootstrapped Accuracy")
ax4.set_ylabel("Density")
ax4.set_title(
    f"Bootstrap Distribution of Accuracy per Prompt Style\n"
    f"(B = {N_BOOTSTRAP:,} resamples, N = {len(df)} questions; dashed lines = observed mean)",
    pad=10
)
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax4.legend(title="Prompt Style", loc="upper left")

fig4.tight_layout()
save(fig4, "fig4_bootstrap_distributions")
plt.close(fig4)

print("\nAll figures saved to:", OUTPUT_DIR)


# ── Interpretation guide ──────────────────────────────────────────────────────
print("""
════════════════════════════════════════════════════════════
HOW TO INTERPRET EACH FIGURE IN THE REPORT
════════════════════════════════════════════════════════════

Figure 1 — Bar chart with bootstrapped CIs (main figure)
  The height of each bar is the observed accuracy for that prompt style.
  Error bars show the empirical 95% CI from bootstrapping: if two bars'
  error bars do not overlap, their difference is likely significant.
  The wide gap between Yes-Man and Zero-Shot CoT is your headline finding.

Figure 2 — Evaluation matrix heatmap
  Each row is a question; each column is a prompt style. Blue = correct,
  red = incorrect. Questions are sorted so the easiest (most prompts
  correct) appear at the top. Rows that are mixed (some blue, some red)
  are questions where prompt phrasing changed the outcome — these are
  the direct evidence of prompt sensitivity.

Figure 3 — Pairwise agreement matrix
  Each cell shows what fraction of questions two prompt styles answered
  identically (both correct or both incorrect). High values (dark blue)
  indicate prompt styles that behave similarly. Low values reveal which
  pairs diverge most — particularly useful when comparing Yes-Man against
  the rest.

Figure 4 — Bootstrap distribution plot
  Each curve shows the sampling distribution of accuracy for one prompt
  style, estimated by resampling. Narrow curves indicate stable estimates;
  wide curves indicate uncertainty. Non-overlapping curves confirm that
  the accuracy difference between prompts is unlikely to be due to chance.
════════════════════════════════════════════════════════════
""")
