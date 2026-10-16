#!/usr/bin/env python3
"""Evaluate a KRAdapter + COCONUT checkpoint on MMMU through vLLM 0.11."""

from __future__ import annotations

import argparse
from functools import partial
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.custom.checkpoint import (as_mapping, checkpoint_stage, latent_token_count,
                                    load_training_config, resolve_target)
from eval.custom.loader import LOAD_FORMAT
from eval.custom.model import coconut_hf_override, register
from eval.qwen.mmmu import (DATASET, DTYPES, MODEL, SUBJECTS,
                            config_defaults as base_config_defaults, extra_vllm_args,
                            input_for_record, records_for_subject, write_scores)
from model.conversation import get_conversation
from train.checkpoint import CheckpointManager

START_LATENT = "<|start-latent|>"
LATENT = "<|latent|>"
END_LATENT = "<|end-latent|>"


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=ROOT / "config" / "qwen.yaml")
    selected, _ = pre.parse_known_args()
    values = base_config_defaults(selected.config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=selected.config)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--model-path", default=values["model_path"])
    parser.add_argument("--model-revision", default=values["revision"])
    parser.add_argument("--dataset-path", default=DATASET)
    parser.add_argument("--dataset-revision", default=values["dataset_revision"])
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=values["dtype"])
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=values["gpu_util"])
    parser.add_argument("--vllm-max-model-len", type=int, default=values["max_model_len"])
    parser.add_argument("--vllm-max-num-seqs", type=int, default=values["max_num_seqs"])
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=values["tensor_parallel_size"])
    parser.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction,
                        default=values["enforce_eager"])
    parser.add_argument("--vllm-arg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--limit-per-subject", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=values["max_new_tokens"])
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=values["do_sample"])
    parser.add_argument("--temperature", type=float, default=values["temperature"])
    parser.add_argument("--top-p", type=float, default=values["top_p"])
    parser.add_argument("--top-k", type=int, default=values["top_k"])
    parser.add_argument("--repetition-penalty", type=float, default=values["repetition_penalty"])
    parser.add_argument("--presence-penalty", type=float, default=values["presence_penalty"])
    parser.add_argument("--seed", type=int, default=values["seed"])
    parser.add_argument("--sample", choices=("global", "per_request"), default=values["sample"])
    parser.add_argument("--conversation", default=values["conversation"])
    parser.add_argument("--min-pixels", type=int, default=values["min_pixels"])
    parser.add_argument("--max-pixels", type=int, default=values["max_pixels"])
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    if (args.limit_per_subject < 0 or args.max_new_tokens <= 0
            or args.vllm_max_num_seqs < 1 or args.vllm_tensor_parallel_size < 1):
        parser.error("--limit-per-subject must be non-negative and --max-new-tokens must be positive.")
    if not args.vllm_max_model_len:
        parser.error("Set model.max_model_len in qwen.yaml or pass --vllm-max-model-len.")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        parser.error("--vllm-gpu-memory-utilization must be in (0, 1].")
    return args


def _prepare_processor(model_path: str, revision: str | None, directory: Path) -> tuple[Any, dict[str, int]]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, revision=revision)
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": [START_LATENT, LATENT, END_LATENT]})
    ids = {
        "start_latent_id": tokenizer.convert_tokens_to_ids(START_LATENT),
        "latent_token_id": tokenizer.convert_tokens_to_ids(LATENT),
        "end_latent_id": tokenizer.convert_tokens_to_ids(END_LATENT),
    }
    if len(set(ids.values())) != 3:
        raise RuntimeError("COCONUT marker tokens did not receive distinct tokenizer IDs.")
    directory.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(directory)
    return processor, ids


def _with_latent_markers(request: dict[str, Any], latent_count: int) -> dict[str, Any]:
    request = dict(request)
    if latent_count == 0:
        return request
    # The three marker strings are tokenizer special tokens.  Do not insert
    # whitespace between latent markers: coconut_latent_layout requires them
    # to occupy contiguous token positions.
    request["prompt"] = request["prompt"] + START_LATENT + (LATENT * latent_count) + END_LATENT
    return request


def main() -> int:
    args = parse_args()
    target = resolve_target(args.target)
    train_config = load_training_config(target)
    adapter = as_mapping(train_config.get("adapter", {}), "checkpoint config.adapter")
    coconut = as_mapping(train_config.get("coconut", {}), "checkpoint config.coconut")
    checkpoint = CheckpointManager.load(target, map_location="cpu")
    stage = checkpoint_stage(checkpoint)
    latent_count = latent_token_count(coconut, stage)
    output_path = (args.output_path or ROOT / "results" / "custom_vllm_mmmu" / target.stem).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    extras = extra_vllm_args(args.vllm_arg)
    fixed = {"worker_cls", "load_format", "model_loader_extra_config", "hf_overrides"}
    collision = fixed.intersection(extras)
    if collision:
        raise ValueError(f"COCONUT's initial vLLM path fixes {sorted(collision)}; remove these --vllm-arg values.")

    register()
    tokenizer_dir = output_path / "coconut_tokenizer"
    processor, token_ids = _prepare_processor(args.model_path, args.model_revision, tokenizer_dir)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_path,
        revision=args.model_revision,
        tokenizer=str(tokenizer_dir),
        tokenizer_revision=None,
        dtype=DTYPES[args.dtype],
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        enforce_eager=args.vllm_enforce_eager,
        worker_cls="eval.custom.runner.CoconutGPUWorker",
        load_format=LOAD_FORMAT,
        model_loader_extra_config={"target": str(target), "adapter": adapter, **token_ids},
        hf_overrides=partial(coconut_hf_override, vocab_size=len(processor.tokenizer)),
        limit_mm_per_prompt={"image": 7},
        **extras,
    )

    conversation = get_conversation(args.conversation)
    predictions: list[dict[str, Any]] = []
    timing: dict[str, float] = {}
    prediction_path = output_path / "predictions.jsonl"
    started_total = time.monotonic()
    seed_index = 0
    with prediction_path.open("w", encoding="utf-8") as handle:
        for subject in SUBJECTS:
            records = records_for_subject(args, subject)
            inputs = [_with_latent_markers(
                input_for_record(record, processor, args.min_pixels, args.max_pixels, conversation), latent_count)
                for record in records]
            params = [SamplingParams(
                temperature=args.temperature if args.do_sample else 0.0,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                max_tokens=args.max_new_tokens,
                seed=(args.seed * 100000 + seed_index + index)
                if args.sample == "per_request" else None,
            ) for index in range(len(records))]
            started = time.monotonic()
            outputs = llm.generate(inputs, sampling_params=params)
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
            seed_index += len(records)
            timing[subject] = time.monotonic() - started
            print(f"[COCONUT vLLM] {subject:40s} {len(records):3d} samples {timing[subject]:7.1f}s")

    scores_path, summary_path, summary = write_scores(predictions, output_path)
    (output_path / "predictions_meta.json").write_text(json.dumps({
        "target": str(target), "checkpoint_stage": stage, "latent_token_count": latent_count,
        "adapter": adapter,
        "vllm": {"max_num_seqs": args.vllm_max_num_seqs,
                 "tensor_parallel_size": args.vllm_tensor_parallel_size,
                 "enforce_eager": args.vllm_enforce_eager,
                 "coconut_segmented_prefill": True},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "timing_sec_by_subject": timing, "total_sec": time.monotonic() - started_total,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    macro = summary["overall_macro"]
    print(f"custom_scoring macro accuracy: {macro * 100:.2f}%" if macro is not None else "custom_scoring macro accuracy: n/a")
    print(f"Saved predictions: {prediction_path}")
    print(f"Saved custom scoring: {scores_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
