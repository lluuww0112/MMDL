#!/usr/bin/env python3
"""Paired, subject-stratified bootstrap confidence intervals for MMMU scores.

The same sampled question IDs are used for every model within each bootstrap
replicate. Each MMMU subject is bootstrapped independently, then subject-level
accuracies are averaged with equal weight. This preserves response pairing and
prevents subjects with more questions from dominating the confidence interval.

By default this performs every target-seed versus comparison-seed pairing
between ``utils/visualization/artifacts/response/KRAdpater`` and
``utils/visualization/artifacts/response/baseline``. With three seeds each,
this produces a 3-by-3 cross-seed bootstrap matrix and individual outputs:

    python utils/visualization/paired_stratified_bootstrap.py

The output directory contains a JSON summary, raw bootstrap differences (for
additional plots), and a PNG box plot.  A positive difference means that the
target has higher accuracy than the comparison model.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

# The experiment host usually has no display server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TARGET = Path("utils/visualization/artifacts/response/KRAdpater")
DEFAULT_COMPARATORS = (Path("utils/visualization/artifacts/response/baseline"),)
DEFAULT_COMPARATOR_LABELS = ("baseline",)
DEFAULT_OUTPUT_DIR = Path("utils/visualization/artifacts/paired_stratified_bootstrap")
SCORE_FILENAMES = ("scores_custom_scoring.jsonl", "scores_ours.jsonl")
DEFAULT_N_RESAMPLES = 100_000
DEFAULT_SAMPLES_PER_SUBJECT = 30


@dataclass(frozen=True)
class Score:
    """One scored MMMU response, keyed by its benchmark question ID."""

    question_id: str
    subject: str
    hit: int


def resolve_score_path(value: str | Path) -> Path:
    """Accept a score JSONL file or a result directory containing one."""
    path = Path(value).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for filename in SCORE_FILENAMES:
            candidate = path / filename
            if candidate.is_file():
                return candidate
    names = ", ".join(SCORE_FILENAMES)
    raise FileNotFoundError(f"Could not find a score JSONL at {path} (looked for: {names}).")


def seed_sort_key(seed_name: str) -> tuple[int, int | str]:
    """Sort conventional ``seed<number>`` names by their numeric suffix."""
    suffix = seed_name.removeprefix("seed")
    return (0, int(suffix)) if suffix.isdigit() else (1, seed_name)


def discover_seed_score_paths(root: Path) -> dict[str, Path]:
    """Find direct score files in ``seed*`` children of a response-model root."""
    root = root.expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Seed response directory does not exist: {root}")
    paths: dict[str, Path] = {}
    for child in sorted(root.glob("seed*"), key=lambda path: seed_sort_key(path.name)):
        if not child.is_dir():
            continue
        try:
            paths[child.name] = resolve_score_path(child)
        except FileNotFoundError:
            continue
    if not paths:
        raise FileNotFoundError(f"No score JSONL files were found in seed* directories below {root}.")
    return paths


def resolve_seed_runs(target: Path, comparisons: Sequence[Path]) -> list[tuple[str, str, Path, list[Path]]]:
    """Resolve one run or the Cartesian product of response-model seeds.

    A target path that directly contains a score file retains the original
    one-run behavior. Otherwise every target seed is compared against every
    seed shared by the comparison model directories.
    """
    try:
        target_path = resolve_score_path(target)
    except FileNotFoundError:
        target_path = None

    if target_path is not None:
        return [("single_run", "single_run", target_path, [resolve_score_path(path) for path in comparisons])]

    target_by_seed = discover_seed_score_paths(target)
    comparisons_by_seed = [discover_seed_score_paths(path) for path in comparisons]
    comparison_seeds = set(comparisons_by_seed[0])
    for paths in comparisons_by_seed[1:]:
        comparison_seeds &= set(paths)
    if not comparison_seeds:
        raise ValueError("No common seed directories with score JSONL files were found across comparisons.")

    return [
        (
            target_seed,
            comparison_seed,
            target_by_seed[target_seed],
            [paths[comparison_seed] for paths in comparisons_by_seed],
        )
        for target_seed in sorted(target_by_seed, key=seed_sort_key)
        for comparison_seed in sorted(comparison_seeds, key=seed_sort_key)
    ]


def load_scores(path: Path) -> dict[str, Score]:
    """Load the ID, subject, and binary correctness fields from score JSONL."""
    scores: dict[str, Score] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            try:
                question_id = str(row["id"])
                subject = str(row["subject"])
                hit = int(row["hit"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number} needs id, subject, and integer hit fields.") from error
            if not question_id or not subject or hit not in (0, 1):
                raise ValueError(f"{path}:{line_number} has an invalid id, subject, or binary hit value.")
            if question_id in scores:
                raise ValueError(f"Duplicate question ID '{question_id}' in {path}.")
            scores[question_id] = Score(question_id, subject, hit)
    if not scores:
        raise ValueError(f"No scores were found in {path}.")
    return scores


def validate_paired_scores(target: dict[str, Score], comparison: dict[str, Score], label: str) -> None:
    """Ensure comparison scores can be paired one-to-one with target scores."""
    target_ids, comparison_ids = set(target), set(comparison)
    if target_ids != comparison_ids:
        missing = sorted(target_ids - comparison_ids)
        extra = sorted(comparison_ids - target_ids)
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} target ID(s), e.g. {missing[:3]}")
        if extra:
            parts.append(f"has {len(extra)} extra ID(s), e.g. {extra[:3]}")
        raise ValueError(f"'{label}' cannot be paired with the target: " + "; ".join(parts))
    subject_mismatches = [
        question_id
        for question_id in target_ids
        if target[question_id].subject != comparison[question_id].subject
    ]
    if subject_mismatches:
        question_id = subject_mismatches[0]
        raise ValueError(
            f"Subject mismatch for ID '{question_id}' in '{label}': "
            f"target={target[question_id].subject!r}, comparison={comparison[question_id].subject!r}."
        )


def group_ids_by_subject(scores: dict[str, Score]) -> dict[str, list[str]]:
    """Return stable, deterministic question-ID strata."""
    groups: dict[str, list[str]] = {}
    for question_id, score in scores.items():
        groups.setdefault(score.subject, []).append(question_id)
    return {subject: sorted(question_ids) for subject, question_ids in sorted(groups.items())}


def macro_accuracy(scores: dict[str, Score]) -> float:
    """Return the equally weighted mean of per-subject accuracies."""
    strata = group_ids_by_subject(scores)
    return float(np.mean([np.mean([scores[question_id].hit for question_id in ids]) for ids in strata.values()]))


def paired_stratified_bootstrap(
    target: dict[str, Score],
    comparisons: Sequence[dict[str, Score]],
    n_resamples: int,
    samples_per_subject: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    """Return macro target-minus-comparison differences and their bootstrap draws.

    Each subject is sampled independently with replacement the same number of
    times, then per-subject accuracies are averaged equally in every replicate.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")
    strata = group_ids_by_subject(target)
    if samples_per_subject < 1:
        raise ValueError("samples_per_subject must be at least 1.")
    stratum_draw_sizes = {subject: samples_per_subject for subject in strata}
    target_hits = np.zeros(len(target), dtype=np.int8)
    comparison_hits = np.zeros((len(comparisons), len(target)), dtype=np.int8)
    stratum_indices: list[tuple[str, np.ndarray]] = []
    offset = 0
    for subject, question_ids in strata.items():
        count = len(question_ids)
        indices = np.arange(offset, offset + count)
        stratum_indices.append((subject, indices))
        for position, question_id in zip(indices, question_ids):
            target_hits[position] = target[question_id].hit
            for comparison_index, comparison in enumerate(comparisons):
                comparison_hits[comparison_index, position] = comparison[question_id].hit
        offset += count

    observed_target_sum = 0.0
    observed_comparison_sums = np.zeros(len(comparisons), dtype=float)
    sampled_target_accuracy_sums = np.zeros(n_resamples, dtype=float)
    sampled_comparison_accuracy_sums = np.zeros((len(comparisons), n_resamples), dtype=float)
    rng = np.random.default_rng(seed)
    for subject, indices in stratum_indices:
        observed_target_sum += target_hits[indices].mean()
        for comparison_index in range(len(comparisons)):
            observed_comparison_sums[comparison_index] += comparison_hits[comparison_index, indices].mean()

        # This exact index matrix is reused across models: paired bootstrap.
        sampled = rng.choice(indices, size=(n_resamples, stratum_draw_sizes[subject]), replace=True)
        sampled_target_accuracy_sums += target_hits[sampled].mean(axis=1)
        for comparison_index in range(len(comparisons)):
            sampled_comparison_accuracy_sums[comparison_index] += comparison_hits[comparison_index, sampled].mean(axis=1)

    num_subjects = len(stratum_indices)
    observed = (observed_target_sum - observed_comparison_sums) / num_subjects
    bootstrap_differences = (sampled_target_accuracy_sums[None, :] - sampled_comparison_accuracy_sums) / num_subjects
    return (
        observed,
        bootstrap_differences.T,
        {subject: len(ids) for subject, ids in strata.items()},
        stratum_draw_sizes,
    )


def paired_overall_bootstrap(
    target: dict[str, Score],
    comparisons: Sequence[dict[str, Score]],
    n_resamples: int,
    samples_per_resample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    """Bootstrap all questions as one paired stratum for micro accuracy CIs."""
    if n_resamples < 1 or samples_per_resample < 1:
        raise ValueError("n_resamples and samples_per_resample must be at least 1.")
    question_ids = sorted(target)
    target_hits = np.array([target[question_id].hit for question_id in question_ids], dtype=np.int8)
    comparison_hits = np.array(
        [[comparison[question_id].hit for question_id in question_ids] for comparison in comparisons], dtype=np.int8
    )
    observed = target_hits.mean() - comparison_hits.mean(axis=1)
    rng = np.random.default_rng(seed)
    # One draw is shared by every model, preserving response-level pairing.
    sampled = rng.integers(0, len(question_ids), size=(n_resamples, samples_per_resample))
    sampled_target_accuracy = target_hits[sampled].mean(axis=1)
    sampled_comparison_accuracy = np.array(
        [comparison_hits[index, sampled].mean(axis=1) for index in range(len(comparisons))]
    )
    return (
        observed,
        (sampled_target_accuracy[None, :] - sampled_comparison_accuracy).T,
        {"all_questions": len(question_ids)},
        {"all_questions": samples_per_resample},
    )


def percentile_interval(samples: np.ndarray, confidence_level: float) -> tuple[float, float]:
    """Compute the two-sided percentile bootstrap confidence interval."""
    alpha = 1 - confidence_level
    lower, upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def write_distribution_csv(path: Path, labels: Sequence[str], samples: np.ndarray) -> None:
    """Write raw accuracy differences, one bootstrap replicate per row."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bootstrap_iteration", *labels])
        for iteration, row in enumerate(samples, start=1):
            writer.writerow([iteration, *[f"{value:.10f}" for value in row]])


def create_box_plot(
    samples: np.ndarray,
    observed: np.ndarray,
    intervals: Sequence[tuple[float, float]],
    labels: Sequence[str],
    target_label: str,
    output_path: Path,
    n_resamples: int,
    metric_label: str,
    title: str,
) -> None:
    """Visualize bootstrap target-minus-comparison accuracy differences."""
    fig, axis = plt.subplots(figsize=(max(10, 3.6 * len(labels)), 7), constrained_layout=True)
    positions = np.arange(1, len(labels) + 1)
    boxplot = axis.boxplot(
        [samples[:, index] * 100 for index in range(len(labels))],
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.6},
    )
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(labels)))
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for position, point, (lower, upper) in zip(positions, observed, intervals):
        axis.errorbar(
            position,
            point * 100,
            yerr=[[max(0, (point - lower) * 100)], [max(0, (upper - point) * 100)]],
            fmt="o",
            color="#b2182b",
            markersize=6,
            capsize=5,
            linewidth=1.8,
            zorder=4,
        )
        axis.annotate(
            f"95% CI [{lower * 100:+.2f}, {upper * 100:+.2f}] pp",
            (position, upper * 100),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.axhline(0, color="dimgray", linestyle="--", linewidth=1.2, alpha=0.8)
    axis.set_xticks(positions, [f"{target_label}\n−\n{label}" for label in labels])
    axis.set_ylabel(f"{metric_label} difference (percentage points)")
    axis.set_title(f"{title} ({n_resamples:,} resamples)")
    axis.text(
        0.5,
        -0.20,
        "Each replicate averages per-subject accuracies equally. Boxes: bootstrap distribution (outliers hidden); red point/error bar: observed difference and percentile 95% CI.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
    )
    axis.yaxis.grid(True, linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def seed_matrix_values(
    summaries: Sequence[dict[str, object]], comparison_index: int
) -> tuple[list[str], list[str], np.ndarray]:
    """Return target-seed rows, comparison-seed columns, and observed pp deltas."""
    target_seeds = sorted({str(summary["target_experiment_seed"]) for summary in summaries}, key=seed_sort_key)
    comparison_seeds = sorted({str(summary["comparison_experiment_seed"]) for summary in summaries}, key=seed_sort_key)
    values = np.full((len(target_seeds), len(comparison_seeds)), np.nan)
    target_positions = {seed: index for index, seed in enumerate(target_seeds)}
    comparison_positions = {seed: index for index, seed in enumerate(comparison_seeds)}
    for summary in summaries:
        target_seed = str(summary["target_experiment_seed"])
        comparison_seed = str(summary["comparison_experiment_seed"])
        comparisons = summary["comparisons"]
        assert isinstance(comparisons, list)
        comparison = comparisons[comparison_index]
        assert isinstance(comparison, dict)
        values[target_positions[target_seed], comparison_positions[comparison_seed]] = float(
            comparison["observed_difference_percentage_points"]
        )
    return target_seeds, comparison_seeds, values


def write_seed_matrix_csv(path: Path, target_seeds: Sequence[str], comparison_seeds: Sequence[str], values: np.ndarray) -> None:
    """Write observed target-minus-comparison accuracy differences as a grid."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target_seed / comparison_seed", *comparison_seeds])
        for target_seed, row in zip(target_seeds, values):
            writer.writerow([target_seed, *[f"{value:.10f}" for value in row]])


def create_seed_matrix_plot(
    path: Path,
    target_seeds: Sequence[str],
    comparison_seeds: Sequence[str],
    values: np.ndarray,
    target_label: str,
    comparison_label: str,
    metric_label: str,
) -> None:
    """Plot a labeled target-seed by comparison-seed difference matrix."""
    maximum = float(np.nanmax(np.abs(values)))
    limit = maximum if maximum > 0 else 1.0
    fig, axis = plt.subplots(figsize=(max(6, 1.6 * len(comparison_seeds) + 3), max(5, 1.2 * len(target_seeds) + 3)), constrained_layout=True)
    image = axis.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(comparison_seeds)), comparison_seeds)
    axis.set_yticks(range(len(target_seeds)), target_seeds)
    axis.set_xlabel(f"{comparison_label} seed")
    axis.set_ylabel(f"{target_label} seed")
    axis.set_title(f"{metric_label} difference: {target_label} − {comparison_label}")
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, f"{value:+.2f}", ha="center", va="center", fontsize=10)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.88)
    colorbar.set_label("Observed difference (percentage points)")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_cross_seed_boxplot_matrix(
    output_path: Path,
    target_seeds: Sequence[str],
    comparison_seeds: Sequence[str],
    bootstrap_samples: dict[tuple[str, str], np.ndarray],
    observed_values: np.ndarray,
    comparison_index: int,
    target_label: str,
    comparison_label: str,
    metric_label: str,
) -> None:
    """Render one paired-bootstrap distribution boxplot in every seed-grid cell."""
    distributions = [
        bootstrap_samples[(target_seed, comparison_seed)][:, comparison_index] * 100
        for target_seed in target_seeds
        for comparison_seed in comparison_seeds
    ]
    all_values = np.concatenate([*distributions, observed_values.ravel()])
    lower, upper = float(all_values.min()), float(all_values.max())
    padding = max(0.5, (upper - lower) * 0.10)
    figure, axes = plt.subplots(
        len(target_seeds),
        len(comparison_seeds),
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=(3.2 * len(comparison_seeds), 2.9 * len(target_seeds)),
        constrained_layout=True,
    )
    for row_index, target_seed in enumerate(target_seeds):
        for column_index, comparison_seed in enumerate(comparison_seeds):
            axis = axes[row_index, column_index]
            samples = bootstrap_samples[(target_seed, comparison_seed)][:, comparison_index] * 100
            boxplot = axis.boxplot(
                [samples],
                positions=[1],
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.4},
            )
            boxplot["boxes"][0].set_facecolor("#80b1d3")
            boxplot["boxes"][0].set_alpha(0.9)
            axis.plot(1, observed_values[row_index, column_index], "o", color="#b2182b", markersize=5, zorder=4)
            axis.axhline(0, color="dimgray", linestyle="--", linewidth=1.0, alpha=0.8)
            axis.set_title(f"{target_seed} → {comparison_seed}", fontsize=10)
            axis.set_xlim(0.5, 1.5)
            axis.set_ylim(lower - padding, upper + padding)
            axis.set_xticks([])
            if column_index == 0:
                axis.set_ylabel("Δ (pp)")
            axis.yaxis.grid(True, linestyle="--", alpha=0.25)
            axis.set_axisbelow(True)
    figure.suptitle(f"{metric_label} paired-bootstrap distributions: {target_label} − {comparison_label}", fontsize=13)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help=f"Target result directory or score JSONL (default: {DEFAULT_TARGET}).")
    parser.add_argument(
        "--target-label",
        default="KRAdapter QKV Epoch 2",
        help="Display name for the target model.",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        action="append",
        default=None,
        help="Comparison model directory, seed directory, or score JSONL. Model directories use every target-seed × comparison-seed pair.",
    )
    parser.add_argument(
        "--comparison-label",
        action="append",
        default=None,
        help="Display name matching one --comparison. Repeat in the same order.",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=DEFAULT_N_RESAMPLES,
        help=f"Bootstrap repetitions (default: {DEFAULT_N_RESAMPLES}).",
    )
    parser.add_argument(
        "--samples-per-subject",
        type=int,
        default=DEFAULT_SAMPLES_PER_SUBJECT,
        help=f"Paired draws with replacement from each subject in every bootstrap repetition (default: {DEFAULT_SAMPLES_PER_SUBJECT}).",
    )
    parser.add_argument("--seed", type=int, default=42, help="NumPy random seed for reproducibility.")
    parser.add_argument(
        "--bootstrap-unit",
        choices=("subject", "overall"),
        default="subject",
        help="'subject' bootstraps each subject then equally averages subject accuracies; 'overall' bootstraps all questions together for micro accuracy.",
    )
    parser.add_argument("--confidence-level", type=float, default=0.95, help="Two-sided percentile CI level (default: 0.95).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Directory for JSON, CSV, and PNG outputs (default: {DEFAULT_OUTPUT_DIR}).")
    return parser.parse_args()


def run_analysis(
    args: argparse.Namespace,
    target_experiment_seed: str,
    comparison_experiment_seed: str,
    target_path: Path,
    comparison_paths: Sequence[Path],
    labels: Sequence[str],
    output_dir: Path,
) -> tuple[dict[str, object], np.ndarray]:
    """Run one paired bootstrap analysis and save its cross-seed artifacts."""
    target_scores = load_scores(target_path)
    comparison_scores = [load_scores(path) for path in comparison_paths]
    for comparison, label in zip(comparison_scores, labels):
        validate_paired_scores(target_scores, comparison, label)

    if args.bootstrap_unit == "subject":
        observed, samples, stratum_sizes, stratum_draw_sizes = paired_stratified_bootstrap(
            target_scores, comparison_scores, args.n_resamples, args.samples_per_subject, args.seed
        )
        metric = "macro accuracy difference (equal mean of per-subject accuracies; target - comparison)"
        metric_label, plot_title = "Macro accuracy", "Per-subject paired bootstrap"
    else:
        observed, samples, stratum_sizes, stratum_draw_sizes = paired_overall_bootstrap(
            target_scores,
            comparison_scores,
            args.n_resamples,
            args.samples_per_subject * len(group_ids_by_subject(target_scores)),
            args.seed,
        )
        metric = "micro accuracy difference (all questions pooled; target - comparison)"
        metric_label, plot_title = "Micro accuracy", "Overall paired bootstrap"

    intervals = [percentile_interval(samples[:, index], args.confidence_level) for index in range(len(labels))]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bootstrap_accuracy_differences.csv"
    json_path = output_dir / "bootstrap_summary.json"
    plot_path = output_dir / "paired_stratified_bootstrap_boxplot.png"
    write_distribution_csv(csv_path, labels, samples)
    create_box_plot(
        samples, observed, intervals, labels, args.target_label, plot_path,
        args.n_resamples, metric_label, plot_title,
    )

    target_macro_accuracy = macro_accuracy(target_scores)
    target_micro_accuracy = float(np.mean([score.hit for score in target_scores.values()]))
    comparisons_summary = []
    for index, (label, path, comparison, interval) in enumerate(zip(labels, comparison_paths, comparison_scores, intervals)):
        lower, upper = interval
        comparisons_summary.append(
            {
                "label": label,
                "score_path": str(path),
                "comparison_macro_accuracy": macro_accuracy(comparison),
                "comparison_micro_accuracy": float(np.mean([score.hit for score in comparison.values()])),
                "observed_difference": float(observed[index]),
                "observed_difference_percentage_points": float(observed[index] * 100),
                "bootstrap_standard_error": float(np.std(samples[:, index], ddof=1)),
                "confidence_interval": {"level": args.confidence_level, "lower": lower, "upper": upper},
                "confidence_interval_percentage_points": {"lower": lower * 100, "upper": upper * 100},
                "ci_excludes_zero": bool(lower > 0 or upper < 0),
            }
        )
    summary: dict[str, object] = {
        "target_experiment_seed": target_experiment_seed,
        "comparison_experiment_seed": comparison_experiment_seed,
        "method": "paired percentile bootstrap",
        "bootstrap_unit": args.bootstrap_unit,
        "metric": metric,
        "stratification_field": "subject" if args.bootstrap_unit == "subject" else None,
        "paired_unit": "MMMU question ID",
        "n_resamples": args.n_resamples,
        "samples_per_subject": args.samples_per_subject,
        "samples_per_resample": int(sum(stratum_draw_sizes.values())),
        "bootstrap_random_seed": args.seed,
        "num_questions": len(target_scores),
        "num_strata": len(stratum_sizes),
        "stratum_sizes": stratum_sizes,
        "stratum_draw_sizes": stratum_draw_sizes,
        "target": {
            "label": args.target_label,
            "score_path": str(target_path),
            "macro_accuracy": target_macro_accuracy,
            "micro_accuracy": target_micro_accuracy,
        },
        "comparisons": comparisons_summary,
        "artifacts": {"bootstrap_differences_csv": str(csv_path), "boxplot_png": str(plot_path)},
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    displayed_accuracy = target_macro_accuracy if args.bootstrap_unit == "subject" else target_micro_accuracy
    print(
        f"[{target_experiment_seed} -> {comparison_experiment_seed}] Target {metric_label.lower()}: "
        f"{args.target_label} = {displayed_accuracy:.4%} ({len(target_scores)} questions)"
    )
    print(
        f"  Bootstrap: {args.n_resamples:,} repetitions × {args.samples_per_subject:,} paired draws per subject "
        f"({sum(stratum_draw_sizes.values()):,} total per repetition; random seed={args.seed})"
    )
    for item in comparisons_summary:
        ci = item["confidence_interval_percentage_points"]
        print(
            f"  vs {item['label']}: Δ={item['observed_difference_percentage_points']:+.2f} pp, "
            f"{args.confidence_level:.0%} CI [{ci['lower']:+.2f}, {ci['upper']:+.2f}] pp, "
            f"excludes 0={item['ci_excludes_zero']}"
        )
    print(f"Saved summary: {json_path}")
    return summary, samples


def main() -> int:
    args = parse_args()
    if args.n_resamples < 2:
        raise SystemExit("--n-resamples must be at least 2.")
    if args.samples_per_subject < 1:
        raise SystemExit("--samples-per-subject must be at least 1.")
    if not 0 < args.confidence_level < 1:
        raise SystemExit("--confidence-level must be between 0 and 1.")

    comparison_values = args.comparison if args.comparison is not None else list(DEFAULT_COMPARATORS)
    labels = args.comparison_label if args.comparison_label is not None else list(DEFAULT_COMPARATOR_LABELS)
    if len(comparison_values) != len(labels):
        raise SystemExit("Supply exactly one --comparison-label for each --comparison.")

    runs = resolve_seed_runs(args.target, comparison_values)
    output_root = args.output_dir.expanduser()
    cross_seed_grid = len(runs) > 1
    summaries: list[dict[str, object]] = []
    bootstrap_samples: dict[tuple[str, str], np.ndarray] = {}
    for target_seed, comparison_seed, target_path, comparison_paths in runs:
        output_dir = output_root
        if cross_seed_grid:
            output_dir = output_root / f"{target_seed}__vs__{comparison_seed}"
        summary, samples = run_analysis(
            args, target_seed, comparison_seed, target_path, comparison_paths, labels, output_dir
        )
        summaries.append(summary)
        bootstrap_samples[(target_seed, comparison_seed)] = samples

    if cross_seed_grid:
        output_root.mkdir(parents=True, exist_ok=True)
        matrix_artifacts = []
        metric_label = "Macro accuracy" if args.bootstrap_unit == "subject" else "Micro accuracy"
        for comparison_index, label in enumerate(labels):
            suffix = "" if len(labels) == 1 else f"_{comparison_index + 1}"
            csv_path = output_root / f"cross_seed_observed_difference_matrix{suffix}.csv"
            plot_path = output_root / f"cross_seed_observed_difference_matrix{suffix}.png"
            target_seeds, comparison_seeds, values = seed_matrix_values(summaries, comparison_index)
            write_seed_matrix_csv(csv_path, target_seeds, comparison_seeds, values)
            create_seed_matrix_plot(
                plot_path, target_seeds, comparison_seeds, values, args.target_label, label, metric_label
            )
            boxplot_path = output_root / f"cross_seed_bootstrap_boxplot_matrix{suffix}.png"
            create_cross_seed_boxplot_matrix(
                boxplot_path,
                target_seeds,
                comparison_seeds,
                bootstrap_samples,
                values,
                comparison_index,
                args.target_label,
                label,
                metric_label,
            )
            matrix_artifacts.append(
                {"label": label, "csv": str(csv_path), "png": str(plot_path), "boxplot_matrix_png": str(boxplot_path)}
            )

        index_path = output_root / "all_seeds_summary.json"
        index = {
            "target_root": str(args.target),
            "comparison_roots": [str(path) for path in comparison_values],
            "cross_seed_matrix_artifacts": matrix_artifacts,
            "analyses": summaries,
        }
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved all-pairs summary: {index_path}")
        for artifact in matrix_artifacts:
            print(f"Saved cross-seed matrix: {artifact['csv']}")
            print(f"Saved cross-seed matrix plot: {artifact['png']}")
            print(f"Saved cross-seed boxplot matrix: {artifact['boxplot_matrix_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
