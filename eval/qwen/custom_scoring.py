#!/usr/bin/env python3
"""Deterministic MMMU scoring adapted from feature/soogguang's score_ours.py."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FINAL_ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{\s*(?:\\text\{\s*)?\(?([A-Z])(?![A-Za-z0-9])"),
    re.compile(
        r"(?i:\b(?:final\s+answer|correct\s+answer|correct\s+option|correct\s+choice|answer))"
        r"(?:\s*(?i:choice|option))?\s*(?i:is)?\s*[:：]?"
        r"[^\w]{0,12}(?i:option\s*|choice\s*)?[\(\[\*]*([A-Z])(?![A-Za-z0-9])"
    ),
]
LEADING_OPTION = re.compile(r"^\s*[\*\(\[]*([A-Z])[\)\]\*]*(?:[\.):]|\s*$)")
ANSWER_LINE = re.compile(r"(?i)\b(?:final\s+answer|answer)\s*\**\s*[:：]\s*(.*)")
OPTION_LINE = re.compile(r"^([A-Z])\.\s*(.*)$", re.MULTILINE)


def extract_final_answer(response: str, all_choices: Sequence[str]) -> str | None:
    best: tuple[int, str] | None = None
    for pattern in FINAL_ANSWER_PATTERNS:
        for match in pattern.finditer(response):
            choice = match.group(1)
            if choice in all_choices and (best is None or match.start(1) > best[0]):
                best = (match.start(1), choice)
    return best[1] if best else None


def extract_leading_option(response: str, all_choices: Sequence[str]) -> str | None:
    match = LEADING_OPTION.match(response)
    return match.group(1) if match and match.group(1) in all_choices else None


def parse_multi_choice_response(response: str, all_choices: Sequence[str], index2ans: Mapping[str, str]) -> str | None:
    """MMMU official parser with its random fallback removed."""
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = f" {response} "
    index_answer, answer_with_brackets, candidates = True, False, []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            answer_with_brackets = True
    if not candidates:
        candidates = [choice for choice in all_choices if f" {choice} " in response]
    if not candidates and len(response.split()) > 5:
        for choice, answer in index2ans.items():
            if answer.lower() in response.lower():
                candidates.append(choice)
                index_answer = False
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if index_answer:
        key = (lambda choice: response.rfind(f"({choice})")) if answer_with_brackets else (lambda choice: response.rfind(f" {choice} "))
    else:
        key = lambda choice: response.lower().rfind(index2ans[choice].lower())
    return max(candidates, key=key)


def _is_number(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(value: str) -> list[str | float]:
    value = value.strip()
    if _is_number(value):
        return [round(float(value.replace(",", "")), 2)]
    value = value.lower()
    return [f" {value}", f"{value} "] if len(value) == 1 else [value]


def _extract_numbers(value: str) -> list[str]:
    return (re.findall(r"-?\b\d{1,3}(?:,\d{3})+\b", value)
            + re.findall(r"-?\d+(?:\.\d+)?[eE][+-]?\d+", value)
            + re.findall(r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])", value))


def parse_open_response(response: str) -> list[str | float]:
    response = response.strip().strip(".").lower()
    sections = re.split(r"\.\s(?=[A-Z])|\n", response)
    selected = []
    for index, section in enumerate(sections):
        indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
        if index == len(sections) - 1:
            indicators.append("=")
        matches = [section.split(indicator)[-1].strip() for indicator in indicators if indicator in section]
        if matches:
            shortest = min(matches, key=len)
            if shortest not in [":", ",", ".", "!", "?", ";", "'"]:
                selected.append(shortest)
    selected = selected or [response]
    parsed = selected.copy()
    for section in selected:
        parsed.extend(_extract_numbers(section))
    return list(set(item for value in parsed for item in normalize_str(value)))


def eval_open(gold: str | list[str], predictions: Sequence[str | float]) -> bool:
    answers = gold if isinstance(gold, list) else [gold]
    normalized = [item for answer in answers for item in normalize_str(str(answer))]
    return any(
        (isinstance(prediction, str) and any(isinstance(answer, str) and answer in prediction for answer in normalized))
        or prediction in normalized
        for prediction in predictions
    )


def _last_boxed(response: str) -> str | None:
    start = response.rfind("\\boxed{")
    if start < 0:
        return None
    content_start, depth = start + len("\\boxed{"), 1
    for index in range(content_start, len(response)):
        depth += {"{": 1, "}": -1}.get(response[index], 0)
        if depth == 0:
            return response[content_start:index]
    return None


def extract_open_final(response: str) -> str | None:
    boxed = _last_boxed(response)
    if boxed and boxed.strip():
        return boxed
    lines = [line for line in response.splitlines() if line.strip()]
    for index in range(len(lines) - 1, -1, -1):
        match = ANSWER_LINE.search(lines[index])
        if match:
            rest = match.group(1).strip(" *")
            return rest or (lines[index + 1] if index + 1 < len(lines) else None)
    return None


def parse_open_final(final: str) -> list[str | float]:
    final = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", final)
    final = re.sub(r"\\text\{([^{}]*)\}", r"\1", final)
    final = re.sub(r"\\[,;!quad ]+|[$*`]", " ", final).strip()
    parsed = normalize_str(final)
    fraction = re.compile(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")
    for numerator, denominator in fraction.findall(final):
        if float(denominator) != 0:
            parsed.append(round(float(numerator) / float(denominator), 2))
    for number in _extract_numbers(fraction.sub(" ", final)):
        parsed.extend(normalize_str(number))
    return list(set(parsed))


def score_one(prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Score one prediction using the feature/soogguang custom rules."""
    if prediction.get("finish_reason") == "length":
        return {"extracted": None, "gt": prediction["answer"], "method": "truncated", "hit": 0}
    response = str(prediction["response"]).split("</think>")[-1].strip()
    if prediction["question_type"] == "multiple-choice" and prediction["choices"]:
        choices = prediction["choices"]
        all_choices = list(choices)
        option, method = extract_final_answer(response, all_choices), "final_answer_pattern"
        if option is None:
            option, method = extract_leading_option(response, all_choices), "leading_option"
        if option is None:
            option, method = parse_multi_choice_response(response, all_choices, choices), "mmmu_parser"
        if option is None:
            method = "failed"
        return {"extracted": option, "gt": prediction["answer"], "method": method,
                "hit": int(option is not None and option == prediction["answer"])}
    final = extract_open_final(response)
    parsed, method = (parse_open_final(final), "open_final_answer") if final is not None else (parse_open_response(response), "mmmu_open")
    return {"extracted": sorted({str(value) for value in parsed})[:10], "gt": prediction["answer"],
            "method": method, "hit": int(eval_open(prediction["answer"], parsed))}


def _response_from_log(log: Mapping[str, Any]) -> str:
    response: Any = log.get("filtered_resps", log.get("resps"))
    if isinstance(response, (str, bytes)):
        return str(response)
    if not isinstance(response, Sequence) or isinstance(response, (str, bytes)) or not response:
        raise ValueError("lmms-eval sample has no response in filtered_resps or resps.")
    response = response[0]
    while isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
        response = response[0] if response else ""
    return str(response)


def _prediction_from_log(log: Mapping[str, Any]) -> dict[str, Any]:
    doc = log.get("doc")
    if not isinstance(doc, Mapping):
        return _prediction_from_jsonl_log(log)
    if not isinstance(doc, Mapping):
        raise ValueError("lmms-eval sample is missing its document.")
    question_type = str(doc["question_type"])
    options = doc.get("options", [])
    if isinstance(options, str):
        options = ast.literal_eval(options)
    choices = {chr(ord("A") + index): str(option) for index, option in enumerate(options)} if question_type == "multiple-choice" else {}
    answer = doc["answer"]
    if question_type == "open" and isinstance(answer, str) and answer.startswith("[") and answer.endswith("]"):
        try:
            parsed_answer = ast.literal_eval(answer)
        except (SyntaxError, ValueError):
            pass
        else:
            if isinstance(parsed_answer, list):
                answer = [str(value) for value in parsed_answer]
    identifier = str(doc.get("id", log.get("doc_id", "")))
    match = re.match(r"^[^_]+_(.+)_\d+$", identifier)
    return {"id": identifier, "subject": match.group(1) if match else None,
            "question_type": question_type, "choices": choices, "answer": answer,
            "response": _response_from_log(log), "finish_reason": log.get("finish_reason")}


def _prediction_from_jsonl_log(log: Mapping[str, Any]) -> dict[str, Any]:
    """Convert lmms-eval's legacy ``*_samples_*.jsonl`` row.

    Older lmms-eval releases omit ``doc`` from sample rows.  The MMMU metric
    payload preserves the id, subject, type and answer; the input retains the
    lettered options needed by the deterministic fallback parser.
    """
    metric = log.get("mmmu_acc")
    if not isinstance(metric, Mapping):
        raise ValueError("lmms-eval sample is missing both doc and mmmu_acc metadata.")
    question_type = str(metric["question_type"])
    answer: str | list[str] = metric.get("answer", log.get("target"))
    if question_type == "open" and isinstance(answer, str) and answer.startswith("[") and answer.endswith("]"):
        try:
            parsed_answer = ast.literal_eval(answer)
        except (SyntaxError, ValueError):
            pass
        else:
            if isinstance(parsed_answer, list):
                answer = [str(value) for value in parsed_answer]
    choices = {match.group(1): match.group(2) for match in OPTION_LINE.finditer(str(log.get("input", "")))}
    if question_type == "multiple-choice" and not choices:
        raise ValueError(f"Could not recover multiple-choice options for {metric.get('id')!r} from its sample input.")
    return {
        "id": str(metric["id"]),
        "subject": metric.get("subdomain"),
        "question_type": question_type,
        "choices": choices if question_type == "multiple-choice" else {},
        "answer": answer,
        "response": _response_from_log(log),
        "finish_reason": log.get("finish_reason"),
    }


def _load_lmms_logs(log_path: Path) -> list[Mapping[str, Any]]:
    if log_path.suffix == ".jsonl":
        with log_path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    with log_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    logs = payload.get("logs") if isinstance(payload, Mapping) else None
    if not isinstance(logs, list):
        raise ValueError(f"{log_path} is not an lmms-eval task sample log.")
    return logs


def write_lmms_scores(log_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Score an lmms-eval task log and write custom-scoring artifacts beside it."""
    logs = _load_lmms_logs(log_path)
    predictions = [_prediction_from_log(log) for log in logs]
    scores = [score_one(prediction) for prediction in predictions]
    subjects: dict[str, list[int]] = defaultdict(list)
    for prediction, score in zip(predictions, scores):
        if prediction["subject"]:
            subjects[prediction["subject"]].append(score["hit"])
    subject_acc = {name: sum(hits) / len(hits) for name, hits in sorted(subjects.items())}
    summary = {
        "source_log": str(log_path), "num_questions": len(scores), "num_correct": sum(score["hit"] for score in scores),
        "overall_micro": sum(score["hit"] for score in scores) / len(scores) if scores else None,
        "overall_macro": sum(subject_acc.values()) / len(subject_acc) if subject_acc else None,
        "subject_acc": subject_acc, "subject_n": {name: len(hits) for name, hits in sorted(subjects.items())},
        "extraction_methods": dict(Counter(score["method"] for score in scores)),
    }
    scores_path = log_path.with_name("scores_custom_scoring.jsonl")
    summary_path = log_path.with_name("scores_custom_scoring_summary.json")
    with scores_path.open("w", encoding="utf-8") as file:
        for prediction, score in zip(predictions, scores):
            file.write(json.dumps({"id": prediction["id"], "subject": prediction["subject"], **score}, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return scores_path, summary_path, summary
