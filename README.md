# Statistical Evaluation of Prompt Fragility

This repository contains the code and data artifacts for a repeated-measures
experiment on prompt fragility in a local Gemma4 model. The same 400 MMLU
multiple-choice questions are evaluated under five prompt styles:
Baseline, Noise, Structured, Yes-Man, and Zero-Shot CoT.

## Files

- `download_mmlu.py` downloads and samples the MMLU pilot and experiment data.
- `experiment.py` runs the prompt experiment through Ollama.
- `grader.py` parses raw model responses and creates binary evaluation matrices.
- `stat_evaluation.py` runs Cochran's Q and gated pairwise McNemar tests.
- `monte_carlo_power.py` estimates power from the pilot data.
- `visualisation/visualisation.py` creates the bootstrap accuracy forest plot.
- `data/` contains the sampled MMLU questions.
- `results/` contains raw model outputs, evaluation matrices, and result figures.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiment script expects Ollama to be installed locally and the model
`gemma4:e4b-it-q8_0` to be available.

## Workflow

```bash
python download_mmlu.py
python experiment.py
python grader.py
python monte_carlo_power.py
python stat_evaluation.py
python visualisation/visualisation.py
```

The saved CSV files in `results/` allow the statistical analysis to be inspected
without rerunning the local LLM experiment.
