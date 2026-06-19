from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "results" / "evaluation_matrix_experiment.csv"
OUTPUT_DIR = Path(__file__).resolve().parent

N_BOOTSTRAP = 10000
RANDOM_SEED = 42

PROMPT_LABELS = {
    "Baseline": "Baseline",
    "Noise": "Noise",
    "Structured": "Structured",
    "Yes_Man": "Yes-Man",
    "Zero_Shot_CoT": "Zero-Shot CoT",
}

PROMPT_COLOURS = {
    "Baseline": "#2C6E9E",
    "Noise": "#E07B39",
    "Structured": "#3BAA72",
    "Yes_Man": "#C0392B",
    "Zero_Shot_CoT": "#7B4EA0",
}


def save_figure(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df = pd.read_csv(DATA_PATH).set_index("Question_ID")
prompts = df.columns.tolist()
n_questions = len(df)
matrix = df.to_numpy(dtype=float)

rng = np.random.default_rng(RANDOM_SEED)
indices = rng.integers(0, n_questions, size=(N_BOOTSTRAP, n_questions))
boot_means = matrix[indices, :].mean(axis=1)
ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5], axis=0)

stats = pd.DataFrame(
    {
        "mean": matrix.mean(axis=0),
        "lower": ci_lower,
        "upper": ci_upper,
    },
    index=prompts,
)

ordered = stats.sort_values("mean", ascending=False)
y_positions = np.arange(len(ordered))

fig, ax = plt.subplots(figsize=(8, 4.5))
for y, (prompt, row) in zip(y_positions, ordered.iterrows()):
    colour = PROMPT_COLOURS[prompt]

    ax.hlines(y, row["lower"], row["upper"], color=colour, linewidth=2.5)
    ax.plot(
        row["mean"],
        y,
        marker="D",
        markersize=9,
        color=colour,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    ax.text(
        1.01,
        y,
        f"{row['mean']:.1%}  [{row['lower']:.1%}, {row['upper']:.1%}]",
        va="center",
        ha="left",
        transform=ax.get_yaxis_transform(),
        fontsize=9.5,
        color=colour,
    )

baseline_mean = stats.loc["Baseline", "mean"]
ax.axvline(
    baseline_mean,
    color=PROMPT_COLOURS["Baseline"],
    linewidth=1,
    linestyle="--",
    alpha=0.5,
)

ax.axvline(0.25, color="grey", linewidth=1, linestyle=":", alpha=0.6)

ax.set_yticks(y_positions)
ax.set_yticklabels([PROMPT_LABELS[prompt] for prompt in ordered.index])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0.2, 1.0)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Accuracy", labelpad=8)
ax.set_title(
    "Prompt Style Accuracy\n"
    f"95% Bootstrap Confidence Intervals (N = {n_questions})",
    pad=12,
)
ax.grid(axis="x", linestyle=":", alpha=0.35)
ax.set_axisbelow(True)

for y in y_positions:
    ax.axhline(y, color="lightgrey", linewidth=0.6, zorder=0)

fig.subplots_adjust(right=0.62)
save_figure(fig, "forest_plot")

print("Done. Figures saved to:", OUTPUT_DIR)
