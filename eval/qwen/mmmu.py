#!/usr/bin/env python3
"""Native vLLM MMMU-val runner with subject-level automatic batching."""

from __future__ import annotations
import argparse
import ast
import json
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from custom_scoring import score_one

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATASET = "MMMU/MMMU"
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
SUBJECTS = (
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
    "Finance", "Geography", "History", "Literature", "Manage", "Marketing", "Materials", "Math",
    "Mechanical_Engineering", "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
)
DTYPES = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16",
          "float16": "float16", "fp32": "float32", "float32": "float32", "auto": "auto"}


def config_defaults(path: Path) -> dict:
    import yaml
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    model, evaluation = config.get("model", {}), config.get("evaluation", {})
    generation, image = config.get("generation", {}), config.get("image", {})
    return {
        "model_path": model.get("path", MODEL),
        "revision": model.get("revision", MODEL_REVISION),
        "dtype": model.get("dtype", "bf16"),
        "gpu_util": model.get("vllm_gpu_utilization", model.get("gpu_memory_utilization", 0.9)),
        "max_model_len": model.get("max_model_len"),
        "max_num_seqs": model.get("max_num_seqs", 64),
        "tensor_parallel_size": model.get("tensor_parallel_size", 1),
        "enforce_eager": model.get("enforce_eager", False),
        "batch_size": evaluation.get("batch_size", 1),
        "max_new_tokens": generation.get("max_new_tokens", 16384),
        "do_sample": generation.get("do_sample", True),
        "temperature": generation.get("temperature", 0.7),
        "top_p": generation.get("top_p", 0.8),
        "top_k": generation.get("top_k", 20),
        "repetition_penalty": generation.get("repetition_penalty", 1.0),
        "presence_penalty": generation.get("presence_penalty", 1.5),
        "seed": generation.get("seed", 3407),
        "min_pixels": image.get("min_pixels"),
        "max_pixels": image.get("max_pixels"),
    }


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=ROOT / "config" / "qwen.yaml")
    selected, _ = pre.parse_known_args()
    values = config_defaults(selected.config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=selected.config)
    parser.add_argument("--model-path", default=values["model_path"])
    parser.add_argument("--model-revision", default=values["revision"])
    parser.add_argument("--dataset-path", default=DATASET)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--backend", "--inference-backend", choices=("vllm",), default="vllm")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=values["dtype"])
    parser.add_argument("--vllm-gpu-memory-utilization", "--vllm-gpu-util", type=float, default=values["gpu_util"])
    parser.add_argument("--vllm-max-model-len", type=int, default=values["max_model_len"])
    parser.add_argument("--vllm-max-num-seqs", type=int, default=values["max_num_seqs"])
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=values["tensor_parallel_size"])
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=values["enforce_eager"])
    parser.add_argument("--vllm-arg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--batch-size", type=int, default=values["batch_size"],
                        help="Kept for compatibility; vLLM batches all requests in each subject.")
    parser.add_argument("--limit-per-subject", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=values["max_new_tokens"])
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=values["do_sample"])
    parser.add_argument("--temperature", type=float, default=values["temperature"])
    parser.add_argument("--top-p", type=float, default=values["top_p"])
    parser.add_argument("--top-k", type=int, default=values["top_k"])
    parser.add_argument("--repetition-penalty", type=float, default=values["repetition_penalty"])
    parser.add_argument("--presence-penalty", type=float, default=values["presence_penalty"])
    parser.add_argument("--seed", type=int, default=values["seed"])
    parser.add_argument("--min-pixels", type=int, default=values["min_pixels"])
    parser.add_argument("--max-pixels", type=int, default=values["max_pixels"])
    parser.add_argument("--output-path", type=Path, default=ROOT / "results" / "vllm_mmmu" / "qwen3-vl-4b-instruct_mmmu_val")
    return parser.parse_args()


def extra_vllm_args(raw_values: list[str]) -> dict:
    result = {}
    for raw in raw_values:
        key, separator, value = raw.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"--vllm-arg must be KEY=VALUE; got {raw!r}.")
        try:
            result[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            result[key] = value
    return result


def records_for_subject(args: argparse.Namespace, subject: str) -> list[dict]:
    from datasets import load_dataset
    kwargs = {"split": "validation"}
    if args.dataset_path == DATASET:
        kwargs["revision"] = args.dataset_revision
    records = []
    for row in load_dataset(args.dataset_path, subject, **kwargs):
        raw_options = row.get("options")
        options = raw_options if isinstance(raw_options, list) else ast.literal_eval(raw_options) if raw_options else []
        qtype = str(row["question_type"])
        answer = str(row["answer"]).strip()
        if qtype != "multiple-choice" and answer.startswith("[") and answer.endswith("]"):
            try:
                answer = [str(item) for item in ast.literal_eval(answer)]
            except (SyntaxError, ValueError):
                pass
        records.append({
            "id": str(row["id"]), "subject": subject, "question": str(row["question"]),
            "question_type": qtype, "answer": answer,
            "choices": {string.ascii_uppercase[index]: str(value) for index, value in enumerate(options)},
            "images": [row[f"image_{index}"] for index in range(1, 8) if row.get(f"image_{index}") is not None],
        })
    return records[:args.limit_per_subject] if args.limit_per_subject else records


def input_for_record(record: dict, processor, min_pixels, max_pixels) -> dict:
    from qwen_vl_utils import process_vision_info
    content = []
    for image in record["images"]:
        item = {"type": "image", "image": image}
        if min_pixels is not None:
            item["min_pixels"] = min_pixels
        if max_pixels is not None:
            item["max_pixels"] = max_pixels
        content.append(item)
    prompt = f"Question: {record['question']}\n"
    if record["choices"]:
        prompt += "Options:\n" + "".join(f"{key}. {value}\n" for key, value in record["choices"].items())
        prompt += "Please select the correct answer from the options above."
    messages = [{"role": "user", "content": [*content, {"type": "text", "text": prompt.rstrip()}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, return_video_metadata=True,
    )
    multimodal = {}
    if images is not None:
        multimodal["image"] = images
    if videos is not None:
        multimodal["video"] = videos
    return {"prompt": text, "multi_modal_data": multimodal, "mm_processor_kwargs": kwargs}


def write_scores(predictions: list[dict], output_path: Path) -> tuple[Path, Path, dict]:
    scores = [score_one(prediction) for prediction in predictions]
    by_subject = defaultdict(list)
    for prediction, score in zip(predictions, scores):
        by_subject[prediction["subject"]].append(int(score["hit"]))
    subject_acc = {name: sum(hits) / len(hits) for name, hits in sorted(by_subject.items())}
    summary = {
        "source_log": str(output_path / "predictions.jsonl"), "num_questions": len(scores),
        "num_correct": sum(int(score["hit"]) for score in scores),
        "overall_micro": sum(int(score["hit"]) for score in scores) / len(scores) if scores else None,
        "overall_macro": sum(subject_acc.values()) / len(subject_acc) if subject_acc else None,
        "subject_acc": subject_acc, "subject_n": {name: len(hits) for name, hits in sorted(by_subject.items())},
        "extraction_methods": dict(Counter(score["method"] for score in scores)),
    }
    scores_path, summary_path = output_path / "scores_custom_scoring.jsonl", output_path / "scores_custom_scoring_summary.json"
    with scores_path.open("w", encoding="utf-8") as handle:
        for prediction, score in zip(predictions, scores):
            handle.write(json.dumps({"id": prediction["id"], "subject": prediction["subject"], **score}, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return scores_path, summary_path, summary


def main() -> int:
    args = parse_args()
    if not args.vllm_max_model_len:
        raise ValueError("Set model.max_model_len in qwen.yaml or pass --vllm-max-model-len.")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        raise ValueError("--vllm-gpu-memory-utilization must be in (0, 1].")
    if args.vllm_max_num_seqs < 1 or args.vllm_tensor_parallel_size < 1 or args.limit_per_subject < 0:
        raise ValueError("Invalid vLLM or limit setting.")
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    output_path = args.output_path.expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    revision = args.model_revision if args.model_path == MODEL else None
    processor = AutoProcessor.from_pretrained(args.model_path, revision=revision)
    llm = LLM(
        model=args.model_path, revision=revision, tokenizer_revision=revision, dtype=DTYPES[args.dtype],
        seed=args.seed, tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization, max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs, enforce_eager=args.vllm_enforce_eager,
        limit_mm_per_prompt={"image": 7}, **extra_vllm_args(args.vllm_arg),
    )
    predictions, timing = [], {}
    prediction_path = output_path / "predictions.jsonl"
    total_started = time.monotonic()
    with prediction_path.open("w", encoding="utf-8") as handle:
        for subject in SUBJECTS:
            records = records_for_subject(args, subject)
            inputs = [input_for_record(record, processor, args.min_pixels, args.max_pixels) for record in records]
            temperature = args.temperature if args.do_sample else 0.0
            params = [SamplingParams(
                temperature=temperature, top_p=args.top_p, top_k=args.top_k,
                repetition_penalty=args.repetition_penalty, presence_penalty=args.presence_penalty,
                max_tokens=args.max_new_tokens, seed=args.seed * 100000 + index,
            ) for index in range(len(records))]
            started = time.monotonic()
            outputs = llm.generate(inputs, sampling_params=params)
            timing[subject] = time.monotonic() - started
            for record, output in zip(records, outputs):
                completion = output.outputs[0]
                prediction = {
                    "id": record["id"], "subject": subject, "question_type": record["question_type"],
                    "choices": record["choices"], "answer": record["answer"], "response": completion.text,
                    "finish_reason": completion.finish_reason, "num_output_tokens": len(completion.token_ids),
                    "num_prompt_tokens": len(output.prompt_token_ids or []),
                }
                predictions.append(prediction)
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[vLLM] {subject:40s} {len(records):3d} samples {timing[subject]:7.1f}s")
    scores_path, summary_path, summary = write_scores(predictions, output_path)
    (output_path / "predictions_meta.json").write_text(json.dumps({
        "args": vars(args), "num_questions": len(predictions), "timing_sec_by_subject": timing,
        "total_sec": time.monotonic() - total_started,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    macro = summary["overall_macro"]
    print(f"custom_scoring macro accuracy: {macro * 100:.2f}%" if macro is not None else "custom_scoring macro accuracy: n/a")
    print(f"Saved predictions: {prediction_path}")
    print(f"Saved custom scoring: {scores_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
