#!/usr/bin/env python3
"""Compare MMMU subject accuracies from custom scoring summary files.

Examples:
    python utils/visualization/compare_score.py \
        --summary results/custom_vllm_mmmu/qkv_epoch2,results/custom_vllm_mmmu/kv_epoch2 \
        --legend KRAdapter_QKV,KRAdapter_KV

Each --summary item may be a directory containing
``scores_custom_scoring_summary.json`` or the JSON file itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

# A display server is not available on most experiment machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_FILENAME = "scores_custom_scoring_summary.json"
DEFAULT_OUTPUT = Path("utils/visualization/artifacts/compare_score/subject_performance_comparison.png")


def comma_separated(value: str) -> list[str]:
    """Split a comma-separated CLI value and reject empty entries."""
    values = [item.strip() for item in value.split(",")]
    if not values or any(not item for item in values):
        raise argparse.ArgumentTypeError("provide comma-separated, non-empty values")
    return values


def resolve_summary_path(value: str) -> Path:
    """Resolve a supplied result directory or a summary JSON file."""
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / SUMMARY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Summary file not found: {path}. Supply a directory containing "
            f"{SUMMARY_FILENAME} or the JSON file itself."
        )
    return path


def load_subject_scores(path: Path) -> dict[str, float]:
    """Read and validate the subject_acc mapping in one summary file."""
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    subject_acc = data.get("subject_acc")
    if not isinstance(subject_acc, dict) or not subject_acc:
        raise ValueError(f"{path} must contain a non-empty 'subject_acc' object")

    scores: dict[str, float] = {}
    for subject, score in subject_acc.items():
        if not isinstance(score, (int, float)):
            raise ValueError(f"Invalid accuracy for '{subject}' in {path}: {score!r}")
        scores[str(subject)] = float(score)
    return scores


def display_subject(subject: str) -> str:
    return subject.replace("_", " ")


def create_plot(
    scores_by_run: Sequence[dict[str, float]],
    legends: Sequence[str],
    baseline_scores: dict[str, float],
    output_path: Path,
    title: str | None,
) -> None:
    """Create a line chart that compares retention against baseline by subject."""
    subjects = sorted(baseline_scores)
    positions = np.arange(len(subjects))

    figure_height = 8
    fig, axis = plt.subplots(figsize=(18, figure_height), constrained_layout=True)

    colors = plt.get_cmap("tab10").colors
    for index, (scores, legend) in enumerate(zip(scores_by_run, legends)):
        values = np.array(
            [
                scores.get(subject, np.nan) / baseline_scores[subject] * 100
                if baseline_scores[subject] != 0
                else np.nan
                for subject in subjects
            ]
        )
        axis.plot(
            positions,
            values,
            label=legend,
            color=colors[index % len(colors)],
            marker="o",
            markersize=5,
            linewidth=2,
        )

    axis.axhline(100, color="dimgray", linestyle="--", linewidth=1, alpha=0.7)

    axis.set_xticks(positions, [display_subject(subject) for subject in subjects])
    axis.tick_params(axis="x", rotation=60, labelsize=9)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.set_xlim(-0.5, len(subjects) - 0.5)
    axis.set_xlabel("Subject")
    axis.set_ylabel("Performance retention (%)")
    axis.set_title(title or "MMMU Subject Performance Retention")
    axis.yaxis.grid(True, linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(title="Model", loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-subject performance retention against a baseline summary file."
    )
    parser.add_argument(
        "--summary",
        required=True,
        type=comma_separated,
        metavar="PATH[,PATH,...]",
        help="Result directories or summary JSON paths, separated by commas.",
    )
    parser.add_argument(
        "--legend",
        required=True,
        type=comma_separated,
        metavar="NAME[,NAME,...]",
        help="Legend labels corresponding to --summary, separated by commas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PNG output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--title", default=None, help="Optional chart title.")
    parser.add_argument(
        "--baseline",
        type=comma_separated,
        required=True,
        metavar="PATH",
        help="Baseline result directory or summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.summary) != len(args.legend):
        raise SystemExit(
            "--summary and --legend must contain the same number of comma-separated values "
            f"(got {len(args.summary)} summaries and {len(args.legend)} legends)."
        )

    summary_paths = [resolve_summary_path(value) for value in args.summary]
    scores_by_run = [load_subject_scores(path) for path in summary_paths]
    output_path = args.output.expanduser()
    baseline_paths = [resolve_summary_path(value) for value in args.baseline]
    if len(baseline_paths) != 1:
        raise SystemExit("--baseline accepts exactly one summary path for retention comparison.")
    baseline_scores = load_subject_scores(baseline_paths[0])

    create_plot(
        scores_by_run,
        args.legend,
        baseline_scores,
        output_path,
        args.title,
    )

    print(f"Saved comparison plot: {output_path}")
    for path, legend in zip(summary_paths, args.legend):
        print(f"  {legend}: {path}")
    print(f"  baseline: {baseline_paths[0]}")


if __name__ == "__main__":
    main()
