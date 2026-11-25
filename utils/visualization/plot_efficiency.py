#!/usr/bin/env python3
"""Plot efficiency measurements for baseline and adapter variants.

The script finds the newest CSV for each supported method in an efficiency
result directory and produces four line charts against image-token count:

* estimated merged-inference TFLOPs,
* decoding latency per token,
* prefill latency,
* peak memory on a selected GPU device.

For MMMU-backed measurements, the x-axis uses the actual image-token count
reported for each benchmark question. Older synthetic CSVs remain supported.

Example:
    python utils/visualization/plot_efficiency.py

    python utils/visualization/plot_efficiency.py \
        --input-dir results/efficiency \
        --output-dir results/efficiency/plots \
        --device-index 2
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

# Experiment servers generally do not have a display server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("results/efficiency")
METHOD_ORDER = ("baseline", "qkv", "kv", "qkvlora", "kvlora")
METHOD_LABELS = {
    "baseline": "Baseline",
    "qkv": "QKV",
    "kv": "KV",
    "qkvlora": "QKV-LoRA",
    "kvlora": "KV-LoRA",
}
COLORS = {
    "baseline": "#4c4c4c",
    "qkv": "#1f77b4",
    "kv": "#ff7f0e",
    "qkvlora": "#2ca02c",
    "kvlora": "#d62728",
}


@dataclass(frozen=True)
class Measurement:
    """One row in an efficiency CSV, normalized to plotting values."""

    image_tokens: int
    tflops: float
    decoding_ms_per_token: float
    prefill_ms: float
    device_memory_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create four efficiency plots from results/efficiency CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing measurement CSV files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG files (default: <input-dir>/plots).",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=2,
        help="Index in peak_memory_mb_per_gpu to plot (default: 2).",
    )
    return parser.parse_args()


def classify_method(path: Path) -> str | None:
    """Infer a canonical method name from a measurement CSV filename."""
    name = path.stem.lower()
    if "baseline" in name:
        return "baseline"
    if "qkv" in name and "lora" in name:
        return "qkvlora"
    if "kv" in name and "lora" in name:
        return "kvlora"
    if "qkv" in name:
        return "qkv"
    if "kv" in name:
        return "kv"
    return None


def newest_csvs_by_method(input_dir: Path) -> dict[str, Path]:
    """Return the newest timestamp-named CSV for every known method."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Efficiency result directory not found: {input_dir}")

    candidates: dict[str, list[Path]] = {method: [] for method in METHOD_ORDER}
    for path in input_dir.glob("*.csv"):
        # Per-question MMMU measurements accompany the binned CSV and should
        # not replace the cost-range summaries selected for plotting.
        if path.stem.endswith("_samples"):
            continue
        method = classify_method(path)
        if method is not None:
            candidates[method].append(path)

    selected = {
        method: max(paths, key=lambda item: item.name)
        for method, paths in candidates.items()
        if paths
    }
    missing = [METHOD_LABELS[method] for method in METHOD_ORDER if method not in selected]
    if missing:
        raise FileNotFoundError(
            "Missing efficiency CSVs for: " + ", ".join(missing) + f" in {input_dir}"
        )
    return selected


def required_float(row: dict[str, str], column: str, path: Path, row_number: int) -> float:
    """Read one required numeric CSV value with an actionable error message."""
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{path}:{row_number}: '{column}' must be a numeric value"
        ) from error


def image_token_count(row: dict[str, str], path: Path, row_number: int) -> int:
    """Use actual MMMU image tokens, falling back to legacy synthetic budgets."""
    value = row.get("actual_image_tokens") or row.get("requested_image_patches")
    try:
        return int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}:{row_number}: an image-token count is required "
            "(actual_image_tokens or requested_image_patches)"
        ) from error


def device_memory(row: dict[str, str], device_index: int, path: Path, row_number: int) -> float:
    """Extract a single device's peak memory from the CSV's list-like column."""
    try:
        values = ast.literal_eval(row["peak_memory_mb_per_gpu"])
    except (KeyError, SyntaxError, ValueError) as error:
        raise ValueError(
            f"{path}:{row_number}: invalid 'peak_memory_mb_per_gpu' value"
        ) from error
    if not isinstance(values, (list, tuple)) or device_index >= len(values) or device_index < 0:
        raise ValueError(
            f"{path}:{row_number}: device index {device_index} is unavailable in {values!r}"
        )
    try:
        return float(values[device_index])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}:{row_number}: device {device_index} memory is not numeric: "
            f"{values[device_index]!r}"
        ) from error


def load_measurements(path: Path, device_index: int) -> list[Measurement]:
    """Load one result CSV and sort it by requested image-patch budget."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Efficiency CSV is empty: {path}")

    measurements = [
        Measurement(
            image_tokens=image_token_count(row, path, row_number),
            tflops=required_float(row, "total_tflops_vllm_merged_estimate", path, row_number),
            decoding_ms_per_token=required_float(row, "decoding_ms_per_token", path, row_number),
            prefill_ms=required_float(row, "prefill_ms", path, row_number),
            device_memory_mb=device_memory(row, device_index, path, row_number),
        )
        for row_number, row in enumerate(rows, start=2)
    ]
    return sorted(measurements, key=lambda item: item.image_tokens)


def plot_metric(
    data: dict[str, list[Measurement]],
    value: Callable[[Measurement], float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Draw one metric with a line for each efficiency method."""
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for method in METHOD_ORDER:
        rows = data[method]
        axis.plot(
            [row.image_tokens for row in rows],
            [value(row) for row in rows],
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker="o",
            linewidth=2,
            markersize=5,
        )

    axis.set_title(title)
    axis.set_xlabel("Actual image tokens")
    axis.set_ylabel(ylabel)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(title="Method")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device_index < 0:
        raise SystemExit("--device-index must be non-negative")

    input_dir = args.input_dir.expanduser()
    output_dir = (args.output_dir or input_dir / "plots").expanduser()
    csv_paths = newest_csvs_by_method(input_dir)
    data = {
        method: load_measurements(csv_paths[method], args.device_index)
        for method in METHOD_ORDER
    }

    plots = (
        (
            "tflops.png",
            lambda row: row.tflops,
            "TFLOPs",
            "TFLOPs",
        ),
        (
            "decoding_per_token.png",
            lambda row: row.decoding_ms_per_token,
            "Decoding Latency per Token",
            "Milliseconds per token",
        ),
        ("prefill.png", lambda row: row.prefill_ms, "Prefill Latency", "Milliseconds"),
        (
            f"memory_device_{args.device_index}.png",
            lambda row: row.device_memory_mb,
            f"Peak Memory (Device {args.device_index})",
            "Memory (MiB)",
        ),
    )
    for filename, value, title, ylabel in plots:
        output_path = output_dir / filename
        plot_metric(data, value, title, ylabel, output_path)
        print(f"Saved: {output_path}")

    print("CSV files used:")
    for method in METHOD_ORDER:
        print(f"  {METHOD_LABELS[method]}: {csv_paths[method]}")


if __name__ == "__main__":
    main()
