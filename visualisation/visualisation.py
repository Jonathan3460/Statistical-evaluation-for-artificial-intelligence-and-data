import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("results/evaluation_matrix_experiment.csv")
OUTPUT_DIR = Path("figures")

N_BOOTSTRAP = 10000
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
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

df = pd.read_csv(DATA_PATH).set_index("Question_ID")
prompt_styles = df.columns.tolist()
N = len(df)


def bootstrap_ci(data, n_resamples=N_BOOTSTRAP, ci=0.95):
    n = len(data)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = data[indices].mean(axis=1)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, 100 * alpha)
    upper = np.percentile(boot_means, 100 * (1 - alpha))
    return lower, upper


stats = {}
for ps in prompt_styles:
    arr = df[ps].values.astype(float)
    lo, hi = bootstrap_ci(arr)
    stats[ps] = {"mean": arr.mean(), "lower": lo, "upper": hi}


# ── Figure 3: Agreement matrix ────────────────────────────────────────────────

n_ps = len(prompt_styles)
agree = np.zeros((n_ps, n_ps))
for i, ps1 in enumerate(prompt_styles):
    for j, ps2 in enumerate(prompt_styles):
        agree[i, j] = (df[ps1] == df[ps2]).mean()

agree_df = pd.DataFrame(
    agree,
    index=[PROMPT_LABELS[ps] for ps in prompt_styles],
    columns=[PROMPT_LABELS[ps] for ps in prompt_styles],
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

for ext in ("png", "pdf"):
    fig3.savefig(
        OUTPUT_DIR / f"fig3_agreement_matrix.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig3)
print("Saved: fig3_agreement_matrix")


# ── Figure 5: Forest plot ─────────────────────────────────────────────────────

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.grid":         False,
})

sorted_prompts = sorted(
    prompt_styles, key=lambda ps: stats[ps]["mean"], reverse=True)
y_positions = np.arange(len(sorted_prompts))

fig5, ax5 = plt.subplots(figsize=(8, 4.5))

for i, ps in enumerate(sorted_prompts):
    s = stats[ps]
    colour = PROMPT_COLOURS[ps]
    y = y_positions[i]

    ax5.hlines(y, s["lower"], s["upper"],
               color=colour, linewidth=2.5, zorder=2)
    ax5.plot(s["mean"], y, marker="D", markersize=9, color=colour, zorder=3,
             markeredgecolor="white", markeredgewidth=0.8)
    ax5.text(
        1.01, y,
        f"{s['mean']:.1%}  [{s['lower']:.1%}, {s['upper']:.1%}]",
        va="center", ha="left",
        transform=ax5.get_yaxis_transform(),
        fontsize=9.5, color=colour
    )

baseline_mean = stats["Baseline"]["mean"]
ax5.axvline(baseline_mean, color="#2C6E9E", linewidth=1,
            linestyle="--", alpha=0.5, zorder=1)
ax5.text(baseline_mean, len(sorted_prompts) - 0.05, "Baseline",
         color="#2C6E9E", fontsize=8, ha="center", va="bottom", alpha=0.7)

ax5.axvline(0.25, color="grey", linewidth=1,
            linestyle=":", alpha=0.6, zorder=1)
ax5.text(0.25, -0.6, "Chance\n(25%)", color="grey", fontsize=8,
         ha="center", va="top", alpha=0.7)

ax5.set_yticks(y_positions)
ax5.set_yticklabels([PROMPT_LABELS[ps] for ps in sorted_prompts], fontsize=11)
ax5.tick_params(axis="y", length=0)

for y in y_positions:
    ax5.axhline(y, color="lightgrey", linewidth=0.6, zorder=0)

ax5.set_xlim(0.2, 1.0)
ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax5.set_xlabel("Accuracy (proportion correct)", labelpad=8)
ax5.set_title(
    f"Forest Plot — LLM Accuracy by Prompt Style\n"
    f"(N = {N} questions, diamonds = mean, lines = 95% bootstrapped CI, "
    f"B = {N_BOOTSTRAP:,})",
    pad=12
)
fig5.subplots_adjust(right=0.62)

for ext in ("png", "pdf"):
    fig5.savefig(
        OUTPUT_DIR / f"fig5_forest_plot.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig5)
print("Saved: fig5_forest_plot")
print("Done. Figures saved to:", OUTPUT_DIR)
