#!/usr/bin/env python3
"""Run a custom checkpoint on MMMU test and save answers keyed by question ID.

Accepts the same arguments as eval/custom/mmmu.py. The test split does not
require answer labels; raw completions are retained in predictions.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.qwen.custom_scoring import (extract_final_answer, extract_leading_option,
                                      extract_open_final, parse_multi_choice_response)


def submission_answer(prediction: dict[str, Any]) -> str:
    response = str(prediction["response"]).split("</think>")[-1].strip()
    choices = prediction["choices"]
    if prediction["question_type"] == "multiple-choice" and choices:
        labels = list(choices)
        return (extract_final_answer(response, labels)
                or extract_leading_option(response, labels)
                or parse_multi_choice_response(response, labels, choices)
                or response)
    return extract_open_final(response) or response


def write_submission(predictions: list[dict[str, Any]], output_path: Path) -> Path:
    answers: dict[str, str] = {}
    for prediction in predictions:
        question_id = prediction["id"]
        if question_id in answers:
            raise ValueError(f"Duplicate MMMU question ID: {question_id}")
        answers[question_id] = submission_answer(prediction)
    path = output_path / "submission.json"
    path.write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    from eval.custom.mmmu import main

    raise SystemExit(main(split="test"))
