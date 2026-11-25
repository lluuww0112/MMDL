#!/usr/bin/env python3
"""Plot output token counts for correctly scored responses, by model and seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "artifacts" / "response"
MODELS = {"Baseline": ("baseline", "scores_ours.jsonl"),
          "KRAdapter": ("KRAdpater", "scores_custom_scoring.jsonl")}
COLORS = {"Baseline": "#59758f", "KRAdapter": "#e28243"}


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"Duplicate ID in {path}: {row['id']}")
            rows[row["id"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "correct_response_tokens_comparison.png"
    results = {}
    for name, (directory, score_file) in MODELS.items():
        seeds = {}
        for score_path in sorted((args.root / directory).glob(f"seed*/{score_file}")):
            scores = read_jsonl(score_path)
            predictions = read_jsonl(score_path.parent / "predictions.jsonl")
            if scores.keys() != predictions.keys():
                raise ValueError(f"Prediction and score IDs differ in {score_path.parent}")
            correct = [predictions[id_]["num_output_tokens"] for id_, score in scores.items()
                       if score["hit"] == 1]
            if not correct:
                raise ValueError(f"No correct answers in {score_path}")
            seeds[score_path.parent.name] = (float(np.mean(correct)), len(correct))
        if len(seeds) < 2:
            raise ValueError(f"At least two seeds required: {name}")
        results[name] = seeds
    if set(results['Baseline']) != set(results['KRAdapter']):
        raise ValueError("Models must have matching seed names")

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    all_values = [value for seeds in results.values() for value, _ in seeds.values()]
    offset = max(all_values) * .025
    for i, (name, seeds) in enumerate(results.items()):
        values = np.array([v[0] for v in seeds.values()])
        mean, sd = values.mean(), values.std(ddof=1)
        ax.bar(i, mean, color=COLORS[name], width=.58, yerr=sd, capsize=6,
               error_kw={"linewidth": 1.5}, zorder=2)
        ax.scatter(np.linspace(i-.09, i+.09, len(values)), values, s=35,
                   color="#263746", zorder=3)
        ax.text(i, mean + sd + offset, f"{mean:,.0f} ± {sd:,.0f}", ha="center",
                va="bottom", fontsize=12, fontweight="bold")
        print(f"{name}: mean {mean:.2f}, sample SD {sd:.2f}; " +
              ', '.join(f'{seed}: {value:.2f} tokens (n={n})' for seed, (value, n) in seeds.items()))
    ax.set_xticks(range(len(results)), list(results))
    ax.set_ylim(0, max(all_values) * 1.23)
    ax.set_ylabel("Output tokens per correct response")
    ax.set_title("Response length on correctly answered questions", fontweight="bold")
    ax.yaxis.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, -.015, "Mean ± sample SD across seeds; dots show seed means.\n"
             "Each model is evaluated on its own correctly answered questions.",
             ha="center", fontsize=9, color="#525c65")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
