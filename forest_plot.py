"""
Forest Plot — Prompt Sensitivity Experiment
=============================================
Generates a publication-ready forest plot showing the bootstrapped 95%
confidence interval for each prompt style's accuracy, sorted from best
to worst. This is the standard format in statistical literature for
comparing multiple estimates on a single axis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH      = r"C:\Users\victo\Desktop\Programmering eksamen\Statistical-evaluation-for-artificial-intelligence-and-data\results\evaluation_matrix_experiment.csv"
OUTPUT_DIR     = Path(r"C:\Users\victo\Desktop\Programmering eksamen\Statistical-evaluation-for-artificial-intelligence-and-data\figures")
N_BOOTSTRAP = 5000
RANDOM_SEED = 42
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

PROMPT_LABELS = {
    "Baseline":      "Baseline",
    "Noise":         "Noise",
    "Structured":    "Structured",
    "Yes_Man":       "Yes-Man",
    "Zero_Shot_CoT": "Zero-Shot CoT",
}

PROMPT_COLOURS = {
    "Baseline":      "#2C6E9E",
    "Noise":         "#E07B39",
    "Structured":    "#3BAA72",
    "Yes_Man":       "#C0392B",
    "Zero_Shot_CoT": "#7B4EA0",
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.grid":          False,
})

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH).set_index("Question_ID")
prompt_styles = df.columns.tolist()
N = len(df)

# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(data, n_resamples=N_BOOTSTRAP, ci=0.95):
    n = len(data)
    indices    = rng.integers(0, n, size=(n_resamples, n))
    boot_means = data[indices].mean(axis=1)
    alpha      = (1 - ci) / 2
    lower      = np.percentile(boot_means, 100 * alpha)
    upper      = np.percentile(boot_means, 100 * (1 - alpha))
    return lower, upper

stats = {}
for ps in prompt_styles:
    arr       = df[ps].values.astype(float)
    lo, hi    = bootstrap_ci(arr)
    stats[ps] = {"mean": arr.mean(), "lower": lo, "upper": hi}

# Sort best → worst accuracy (top of plot = best)
sorted_prompts = sorted(prompt_styles, key=lambda ps: stats[ps]["mean"], reverse=True)

# ── Build forest plot ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

y_positions = np.arange(len(sorted_prompts))

for i, ps in enumerate(sorted_prompts):
    s      = stats[ps]
    colour = PROMPT_COLOURS[ps]
    y      = y_positions[i]

    # Horizontal CI line
    ax.hlines(y, s["lower"], s["upper"],
              color=colour, linewidth=2.5, zorder=2)

    # Diamond marker for the point estimate (classic forest plot style)
    ax.plot(s["mean"], y,
            marker="D", markersize=9,
            color=colour, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8)

    # Numeric annotation: mean (95% CI) on the right side
    ax.text(
        1.01, y,
        f"{s['mean']:.1%}  [{s['lower']:.1%}, {s['upper']:.1%}]",
        va="center", ha="left",
        transform=ax.get_yaxis_transform(),
        fontsize=9.5, color=colour
    )

# Reference line at Baseline mean for easy comparison
baseline_mean = stats["Baseline"]["mean"]
ax.axvline(baseline_mean, color="#2C6E9E", linewidth=1,
           linestyle="--", alpha=0.5, zorder=1)
ax.text(baseline_mean, len(sorted_prompts) - 0.05,
        "Baseline", color="#2C6E9E", fontsize=8,
        ha="center", va="bottom", alpha=0.7)

# Reference line at chance level
ax.axvline(0.25, color="grey", linewidth=1,
           linestyle=":", alpha=0.6, zorder=1)
ax.text(0.25, -0.6, "Chance\n(25%)",
        color="grey", fontsize=8, ha="center", va="top", alpha=0.7)

# Y-axis: prompt style labels
ax.set_yticks(y_positions)
ax.set_yticklabels(
    [PROMPT_LABELS[ps] for ps in sorted_prompts],
    fontsize=11
)
ax.tick_params(axis="y", length=0)   # hide tick marks, keep labels

# Subtle horizontal guide lines per row
for y in y_positions:
    ax.axhline(y, color="lightgrey", linewidth=0.6, zorder=0)

# X-axis formatting
ax.set_xlim(0.2, 1.0)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_xlabel("Accuracy (proportion correct)", labelpad=8)

ax.set_title(
    f"Forest Plot — LLM Accuracy by Prompt Style\n"
    f"(N = {N} questions, diamonds = mean, lines = 95% bootstrapped CI, "
    f"B = {N_BOOTSTRAP:,})",
    pad=12
)

# Expand right margin so the annotation text fits
fig.subplots_adjust(right=0.62)

# ── Save ──────────────────────────────────────────────────────────────────────

for ext in ("png", "pdf"):
    path = OUTPUT_DIR / f"fig5_forest_plot.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")

plt.close(fig)

print("""
How to read this figure in your report:
  Each row is one prompt style, sorted from highest to lowest accuracy.
  The diamond marks the observed mean accuracy; the horizontal line spans
  the 95% bootstrapped confidence interval. Intervals that do not overlap
  indicate a statistically meaningful difference — confirmed formally by
  the Cochran's Q and McNemar tests. The dashed blue line marks the
  Baseline accuracy as a visual reference point.
""")
