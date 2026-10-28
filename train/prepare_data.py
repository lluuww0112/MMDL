#!/usr/bin/env python3
"""MMFineReason cache and COCONUT-row helpers.

Run this module once to download the complete Hugging Face dataset into the
standard Hugging Face cache. ``train.dataloader`` imports the helpers below at
training time, reads that cache, and converts rows directly to COCONUT inputs.
No Dataset Viewer API or duplicate JSON/image export is used.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


DATASET_NAME = "OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking"
DATASET_URI = f"hf://{DATASET_NAME}"
TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
THINKING_FIELD = "qwen3vl_235b_thinking_response"
IMAGE_MARKER = re.compile(r"^\s*<image>\s*", flags=re.IGNORECASE)
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")
_NUMBERED_STEP_BOUNDARY = re.compile(
    r"(?m)(?=^\s*(?:(?:step\s*)?\d+[.)]|[A-Za-z][.)]|[-*•])\s+)"
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z]|\d+[.)]))")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face token file not found: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Invalid .env entry on line {line_number}: expected KEY=VALUE.")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def hf_token(project_root: Path | None = None) -> str:
    root = _project_root() if project_root is None else project_root
    values = _read_dotenv(root / ".env")
    for name in TOKEN_ENV_NAMES:
        token = values.get(name, "").strip()
        if token:
            return token
    raise ValueError(f"{root / '.env'} must define one of: {', '.join(TOKEN_ENV_NAMES)}.")


def load_mmfinereason_dataset(*, cache_dir: str | Path | None = None, force_redownload: bool = False,
                              local_files_only: bool = False) -> Any:
    """Return the MMFineReason train split from cache, or download it when allowed."""
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as error:
        raise RuntimeError("Install the 'datasets' package before preparing MMFineReason.") from error
    if local_files_only and force_redownload:
        raise ValueError("local_files_only and force_redownload cannot both be enabled.")
    kwargs: dict[str, Any] = {"split": "train"}
    if local_files_only:
        kwargs["download_config"] = DownloadConfig(local_files_only=True)
    else:
        kwargs["token"] = hf_token()
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    if force_redownload:
        kwargs["download_mode"] = "force_redownload"
    return load_dataset(DATASET_NAME, **kwargs)


def separate_steps(value: Any) -> list[str]:
    """Extract deterministic reasoning steps from a teacher thinking response.

    MMFineReason's ``answer`` column is a short final answer, not a chain of
    thought.  Its Qwen teacher response is free-form, so it has no explicit
    step field. Preserve the teacher's strongest boundaries first: blank-line
    paragraphs and line-start numbered/bulleted items. When neither exists,
    fall back to sentence boundaries so a whole rationale never becomes a
    single COCONUT step. An empty ``<think>`` block is valid for a direct-answer
    example and deliberately returns no reasoning steps.
    """
    if value is None:
        return []
    if not isinstance(value, str):
        raise TypeError(f"{THINKING_FIELD} must be a string.")
    text = value.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if not text:
        return []

    paragraphs = [part.strip() for part in _PARAGRAPH_BOUNDARY.split(text) if part.strip()]
    steps = [
        part.strip()
        for paragraph in paragraphs
        for part in _NUMBERED_STEP_BOUNDARY.split(paragraph)
        if part.strip()
    ]
    # Some responses are a single unformatted paragraph.  Only then use a
    # sentence fallback; otherwise the author-provided paragraph structure is
    # the more meaningful reasoning-step boundary.
    if len(steps) == 1:
        steps = [part.strip() for part in _SENTENCE_BOUNDARY.split(steps[0]) if part.strip()]
    return steps


def coconut_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one decoded HF row to the schema consumed by train.dataloader."""
    question, answer, image = row.get("question"), row.get("answer"), row.get("image")
    if not isinstance(question, str) or not question.strip():
        raise TypeError("Dataset question must be a non-empty string.")
    if not isinstance(answer, str) or not answer.strip():
        raise TypeError("Dataset answer must be a non-empty string.")
    if image is None:
        raise ValueError("Dataset row does not contain an image.")
    # Stage 0 needs the original, unsplit teacher response. Prefer an
    # explicitly supplied value, then the teacher response used for the
    # reasoning trace, and finally the answer-only label for direct answers.
    original_answer = row.get("original_answer", row.get(THINKING_FIELD, answer))
    if not isinstance(original_answer, str) or not original_answer.strip():
        original_answer = answer
    return {
        "image": image,
        "question": IMAGE_MARKER.sub("", question).strip(),
        "answer": answer.strip(),
        "original_answer": original_answer.strip(),
        "steps": separate_steps(original_answer),
    }


def iter_coconut_rows(dataset: Any, *, count: int | None = None, seed: int = 0,
                      predicate: Callable[[dict[str, Any]], bool] | None = None) -> Iterator[dict[str, Any]]:
    """Yield valid COCONUT rows, randomly sampled without exporting files.

    A finite ``count`` is sampled in random index order. Rows lacking required
    annotations are skipped, and the iterator keeps reading until it yields the
    requested number of valid examples.
    """
    total = len(dataset)
    if count is not None:
        if count <= 0:
            raise ValueError("count must be positive when set.")
        if count > total:
            raise ValueError(f"Requested {count} samples, but the dataset has only {total} rows.")
        indices = list(range(total))
        random.Random(seed).shuffle(indices)
    else:
        indices = range(total)
    yielded = 0
    for index in indices:
        try:
            converted = coconut_row(dataset[index])
        except (TypeError, ValueError):
            continue
        if predicate is not None and not predicate(converted):
            continue
        yield converted
        yielded += 1
        if count is not None and yielded == count:
            return
    if count is not None:
        raise RuntimeError(f"Only {yielded} rows have all required non-empty COCONUT fields.")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the full MMFineReason train split into the Hugging Face cache.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Optional Hugging Face cache root (default: the standard HF cache).")
    parser.add_argument("--force-redownload", action="store_true", help="Re-download cached dataset files.")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    dataset = load_mmfinereason_dataset(cache_dir=args.cache_dir, force_redownload=args.force_redownload)
    print(f"Cached {DATASET_NAME} train split: {len(dataset)} rows")
    print(f"Use config.data.train_path: {DATASET_URI}")


if __name__ == "__main__":
    main()
