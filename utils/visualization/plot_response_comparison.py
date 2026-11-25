#!/usr/bin/env python3
"""Compare overall MMMU accuracy and cumulative extraction counts across seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "artifacts" / "response"
MODELS = {"Baseline": ("baseline", "scores_ours_summary.json"),
          "KRAdapter": ("KRAdpater", "scores_custom_scoring_summary.json")}
COLORS = {"Baseline": "#59758f", "KRAdapter": "#e28243"}


def load_runs(root: Path) -> dict[str, dict[str, dict]]:
    runs = {}
    for model, (directory, filename) in MODELS.items():
        runs[model] = {}
        for path in sorted((root / directory).glob(f"seed*/{filename}")):
            data = json.loads(path.read_text(encoding="utf-8"))
            n = data["num_questions"]
            methods = data["extraction_methods"]
            if n <= 0 or sum(methods.values()) != n:
                raise ValueError(f"Invalid extraction counts or question count: {path}")
            runs[model][path.parent.name] = data
        if len(runs[model]) < 2:
            raise ValueError(f"At least two seeds are needed for sample SD: {model}")
    if set(runs['Baseline']) != set(runs['KRAdapter']):
        raise ValueError("Baseline and KRAdapter must have matching seeds")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Response results directory")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()
    output = args.output or args.root / "overall_extraction_comparison.png"
    runs = load_runs(args.root)
    seeds = sorted(runs["Baseline"])
    methods = ["final_answer_pattern", "truncated", "leading_option", "open_final_answer"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1, 1.65]}, layout="constrained")
    fig.suptitle("MMMU response comparison", fontsize=17, fontweight="bold")
    max_total = max(sum(run["num_questions"] for run in model.values()) for model in runs.values())
    for i, (model, model_runs) in enumerate(runs.items()):
        values = np.array([model_runs[s]["overall_micro"] * 100 for s in seeds])
        mean, sd = values.mean(), values.std(ddof=1)
        ax.bar(i, mean, color=COLORS[model], width=.56, yerr=sd, capsize=5, error_kw={"lw": 1.5})
        ax.scatter(np.full(len(values), i), values, color="#25313b", s=24, zorder=4)
        ax.text(i, mean + sd + 1.1, f"{mean:.2f}% ± {sd:.2f}", ha="center", fontsize=11, fontweight="bold")
        for j, method in enumerate(methods):
            count = sum(model_runs[s]["extraction_methods"].get(method, 0) for s in seeds)
            y = j + (i - .5) * .32
            bx.barh(y, count, height=.28, color=COLORS[model],
                    label=model if j == 0 else None)
            bx.text(count + max_total * .01, y, f"{count:,}", va="center", fontsize=9)
    ax.set_xticks(range(len(runs)), list(runs))
    accuracy_values = [np.array([model[s]["overall_micro"] * 100 for s in seeds])
                       for model in runs.values()]
    lower = min(min(v.min(), v.mean() - v.std(ddof=1)) for v in accuracy_values)
    upper = max(max(v.max(), v.mean() + v.std(ddof=1)) for v in accuracy_values)
    padding = max((upper - lower) * 0.35, 1.0)
    ax.set_ylim(max(0, lower - padding), min(100, upper + padding * 2))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall accuracy", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    bx.set_yticks(range(len(methods)), [m.replace('_', ' ') for m in methods])
    bx.invert_yaxis()
    bx.set_xlim(0, max_total * 1.05)
    bx.set_xlabel("Number of responses across all seeds")
    bx.set_title("Answer extraction methods", loc="left", fontweight="bold")
    bx.grid(axis="x", alpha=.2); bx.set_axisbelow(True)
    bx.legend(loc="lower right", frameon=False)
    fig.text(.5, -.01, f"Overall: mean ± sample SD across {len(seeds)} seeds ({', '.join(seeds)}); dots show individual seed accuracy. Extraction: total counts.", ha="center", fontsize=9, color="#525c65")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    for model, model_runs in runs.items():
        values = np.array([model_runs[s]["overall_micro"] * 100 for s in seeds])
        print(f"{model}: {values.mean():.2f}% ± {values.std(ddof=1):.2f} pp")


if __name__ == "__main__":
    main()
