#!/usr/bin/env python3
"""Measure Qwen3-VL inference efficiency on MMMU Art Theory with vLLM.

The benchmark runs each validation example in MMMU's ``Art_Theory`` subject
with its original image(s) and question prompt, then writes one CSV row per
token budget under ``results/efficiency``. ``--bin`` creates inclusive,
evenly spaced fixed visual-token budgets. Each budget is measured with the
same real Art Theory images and questions, without generated image content.

``--model baseline`` measures the Qwen model named in ``--config``.  Any other
``--model`` value is a checkpoint file or a run directory containing a LoRA or
KRAdapter checkpoint; it is restored through the project's vLLM merge loader.
"""

from __future__ import annotations

import asyncio
import argparse
import csv
import functools
import json
import math
import random
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.custom.checkpoint import (as_mapping, checkpoint_stage, latent_token_count,
                                    load_training_config, resolve_target)
from eval.qwen.mmmu import (DATASET, DTYPES, MODEL, config_defaults,
                            records_for_subject)
from model.conversation import get_conversation
from train.checkpoint import CheckpointManager


IMAGE_PAD_TOKEN = "<|image_pad|>"
SUBJECT = "Art_Theory"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class AdapterProfile:
    """Adapter metadata used for parameter and unmerged-compute reporting."""

    kind: str
    parameter_count: int
    module_count: int
    # FLOPs to evaluate adapters in their original, unmerged PyTorch form for
    # one transformer-token forward.  vLLM merges the updates, so it incurs no
    # separate adapter kernel at inference time.
    source_flops_per_token: float
    materialization_flops_per_token: float


@dataclass(frozen=True)
class TransformerShape:
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="'baseline' or a LoRA/KRAdapter checkpoint/run directory.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "qwen.yaml",
                        help="Qwen generation/image configuration YAML.")
    parser.add_argument("--dataset-path", default=DATASET,
                        help="MMMU dataset name or local dataset path.")
    parser.add_argument("--dataset-revision", default=None,
                        help="Dataset revision; defaults to the revision in --config.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Measure at most this many Art Theory validation samples (0: all).")
    parser.add_argument("--bin", type=int, default=5,
                        help="Number of inclusive fixed visual-token budgets to measure (default: 5).")
    parser.add_argument("--max-tokens", type=int, default=10,
                        help="Maximum generated tokens per MMMU sample (default: 10).")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative.")
    if args.bin < 1:
        parser.error("--bin must be positive.")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive.")
    if not args.config.is_file():
        parser.error(f"--config was not found: {args.config}")
    return args


@functools.lru_cache(maxsize=None)
def _token_grid(token_count: int, image_factor: int) -> tuple[int, int] | None:
    """Return an exact, factor-aligned grid within Qwen's aspect-ratio limit."""
    if token_count < 1:
        return None
    for width_blocks in range(math.isqrt(token_count), 0, -1):
        if token_count % width_blocks:
            continue
        height_blocks = token_count // width_blocks
        if max(height_blocks, width_blocks) / min(height_blocks, width_blocks) < 200:
            return height_blocks * image_factor, width_blocks * image_factor
    return None


def _nearest_grid_budget(target: int, lower: int, upper: int, image_factor: int) -> int:
    """Find the closest total token budget that admits a legal one-image grid."""
    for distance in range(max(target - lower, upper - target) + 1):
        for candidate in (target - distance, target + distance):
            if lower <= candidate <= upper and _token_grid(candidate, image_factor) is not None:
                return candidate
    raise ValueError(f"No aspect-ratio-safe visual-token budget exists in [{lower}, {upper}].")


def patch_budgets(min_pixels: int, max_pixels: int, bins: int, *, image_factor: int) -> list[int]:
    """Return inclusive fixed visual-token budgets that have legal image grids."""
    unit = image_factor * image_factor
    lower, upper = math.ceil(min_pixels / unit), math.floor(max_pixels / unit)
    if lower <= 0 or lower > upper:
        raise ValueError("The configured image-pixel interval contains no valid visual-token budget.")
    if bins == 1 or lower == upper:
        return [_nearest_grid_budget(lower, lower, upper, image_factor)]
    budgets = [_nearest_grid_budget(
        round(lower + index * (upper - lower) / (bins - 1)), lower, upper, image_factor,
    ) for index in range(bins)]
    if len(set(budgets)) != len(budgets):
        raise ValueError("--bin creates duplicate legal visual-token budgets; reduce --bin.")
    return budgets


@functools.lru_cache(maxsize=None)
def _split_token_budget(token_budget: int, image_count: int, image_factor: int) -> tuple[int, ...]:
    """Split one exact total budget into aspect-ratio-safe per-image budgets."""
    if image_count < 1 or token_budget < image_count:
        raise ValueError(f"Visual-token budget {token_budget} is too small for {image_count} images.")
    legal = [tokens for tokens in range(1, token_budget + 1)
             if _token_grid(tokens, image_factor) is not None]
    legal_set = set(legal)

    @functools.lru_cache(maxsize=None)
    def partition(remaining: int, slots: int) -> tuple[int, ...] | None:
        if slots == 1:
            return (remaining,) if remaining in legal_set else None
        ideal = remaining / slots
        for tokens in sorted((value for value in legal if value < remaining),
                             key=lambda value: abs(value - ideal)):
            result = partition(remaining - tokens, slots - 1)
            if result is not None:
                return (tokens, *result)
        return None

    result = partition(token_budget, image_count)
    if result is None:
        raise ValueError(
            f"Cannot split {token_budget} visual tokens across {image_count} images "
            "while respecting Qwen's aspect-ratio limit.")
    return result


def fixed_token_input(record: Mapping[str, Any], processor: Any, token_budget: int,
                      image_factor: int, conversation: Any) -> dict[str, Any]:
    """Build an MMMU request with its total visual-token count fixed exactly."""
    from qwen_vl_utils import process_vision_info

    source_images = list(record["images"])
    if not source_images:
        raise ValueError(f"MMMU record {record['id']} has no image to resize to a visual-token budget.")
    content: list[dict[str, Any]] = []
    for image, tokens in zip(source_images, _split_token_budget(token_budget, len(source_images), image_factor)):
        dimensions = _token_grid(tokens, image_factor)
        assert dimensions is not None
        height, width = dimensions
        content.append({"type": "image", "image": image,
                        "resized_height": height, "resized_width": width})
    prompt = conversation.render(record, fallback=_legacy_prompt(record))
    messages: list[dict[str, Any]] = []
    if conversation.system_prompt:
        messages.append({"role": "system", "content": conversation.system_prompt})
    messages.append({"role": "user", "content": [*content, {"type": "text", "text": prompt}]})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, return_video_metadata=True,
    )
    multimodal: dict[str, Any] = {}
    if images is not None:
        multimodal["image"] = images
    if videos is not None:
        multimodal["video"] = videos
    return {"prompt": text, "multi_modal_data": multimodal, "mm_processor_kwargs": kwargs}


def _legacy_prompt(record: Mapping[str, Any]) -> str:
    prompt = f"Question: {record['question']}\n"
    choices = record["choices"]
    if choices:
        prompt += "Options:\n" + "".join(f"{key}. {value}\n" for key, value in choices.items())
        prompt += "Please select the correct answer from the options above."
    return prompt.rstrip()


def _field(config: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def transformer_shape(model_path: str, revision: str | None) -> TransformerShape:
    """Read only the HF config; loading the 4B weights is not required."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, revision=revision)
    text = _field(config, "text_config", default=config)
    hidden = int(_field(text, "hidden_size"))
    heads = int(_field(text, "num_attention_heads", "num_heads"))
    head_dim = int(_field(text, "head_dim", default=hidden // heads))
    return TransformerShape(
        hidden_size=hidden,
        intermediate_size=int(_field(text, "intermediate_size", "ffn_hidden_size")),
        layers=int(_field(text, "num_hidden_layers", "num_layers")),
        attention_heads=heads,
        kv_heads=int(_field(text, "num_key_value_heads", default=heads)),
        head_dim=head_dim,
        vocab_size=int(_field(text, "vocab_size", default=_field(config, "vocab_size"))),
    )


def _projection_shape(shape: TransformerShape, name: str) -> tuple[int, int]:
    kv_width = shape.kv_heads * shape.head_dim
    if name == "q_proj":
        return shape.hidden_size, shape.hidden_size
    if name in {"k_proj", "v_proj"}:
        return kv_width, shape.hidden_size
    if name == "o_proj":
        return shape.hidden_size, shape.hidden_size
    raise ValueError(f"Unsupported adapter projection: {name}")


def adapter_profile(checkpoint: Mapping[str, Any], adapter: Mapping[str, Any], shape: TransformerShape) -> AdapterProfile:
    """Count stored adapter tensors and estimate their original forward FLOPs."""
    from eval.custom.loader import _state_adapter_pairs, _state_lora_pairs

    kind = str(adapter.get("type", "none")).lower()
    if kind in {"", "none"}:
        return AdapterProfile("none", 0, 0, 0.0, 0.0)
    targets = adapter.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))
    if isinstance(targets, str):
        targets = (targets,)
    if not isinstance(targets, (list, tuple)) or not all(isinstance(name, str) for name in targets):
        raise ValueError("checkpoint adapter.target_modules must be a list of projection names.")
    state = checkpoint.get("model_state", {})
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint has no model_state mapping.")

    params = 0
    modules = 0
    source_flops = 0.0
    materialization_flops = 0.0
    if kind == "lora":
        pairs: Iterable[tuple[int, str, Any, Any]] = _state_lora_pairs(state, tuple(targets))
        for _, projection, a_weight, b_weight in pairs:
            out_features, in_features = _projection_shape(shape, projection)
            rank = int(a_weight.shape[0])
            modules += 1
            params += int(a_weight.numel() + b_weight.numel())
            # train.lora's forward materializes B @ A, then applies that dense
            # update.  This is reported separately from vLLM's merged runtime.
            formed = 2.0 * out_features * rank * in_features
            materialization_flops += formed
            source_flops += formed + 2.0 * out_features * in_features
    elif kind == "kradapter":
        pairs = _state_adapter_pairs(state, tuple(targets))
        for _, projection, weight, v_weight in pairs:
            out_features, in_features = _projection_shape(shape, projection)
            modules += 1
            params += int(weight.numel() + v_weight.numel())
            # KRAdapter.get_update forms one Khatri--Rao product per element,
            # then the unmerged wrapper performs a dense x @ update operation.
            formed = float(weight.shape[0] * weight.shape[1] * v_weight.shape[1])
            materialization_flops += formed
            source_flops += formed + 2.0 * out_features * in_features
    else:
        raise ValueError(f"Unsupported checkpoint adapter.type: {kind!r}.")
    return AdapterProfile(kind, params, modules, source_flops, materialization_flops)


def _adapter_profile_with_count(checkpoint: Mapping[str, Any], adapter: Mapping[str, Any], shape: TransformerShape) -> AdapterProfile:
    """Adapter profile with a module count (kept separate to avoid loading weights twice)."""
    profile = adapter_profile(checkpoint, adapter, shape)
    if profile.kind == "none":
        return profile
    from eval.custom.loader import _state_adapter_pairs, _state_lora_pairs

    targets = adapter.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))
    if isinstance(targets, str):
        targets = (targets,)
    state = checkpoint["model_state"]
    pairs = (_state_lora_pairs(state, tuple(targets)) if profile.kind == "lora"
             else _state_adapter_pairs(state, tuple(targets)))
    return AdapterProfile(profile.kind, profile.parameter_count, sum(1 for _ in pairs),
                          profile.source_flops_per_token, profile.materialization_flops_per_token)


def estimate_tflops(shape: TransformerShape, *, prompt_tokens: int, output_tokens: int,
                    adapter: AdapterProfile) -> dict[str, float]:
    """Estimate decoder FLOPs from actual vLLM prompt/output token counts.

    These are analytic decoder estimates, not hardware-counter values.  They
    include QKV/O, gated MLP, causal attention and sampled LM-head projections.
    Vision-encoder kernels and norms are intentionally excluded because vLLM
    does not expose their per-request operation counts in a stable API.
    """
    if prompt_tokens < 1 or output_tokens < 0:
        raise ValueError("Invalid prompt/output token count.")
    h, kv, layers = shape.hidden_size, shape.kv_heads * shape.head_dim, shape.layers
    token_forwards = prompt_tokens + max(output_tokens - 1, 0)
    # Q, K, V, O and gated FFN (gate/up/down) matrix multiplications.
    dense_per_token = 2.0 * layers * (h * (h + 2 * kv) + h * h + 3 * h * shape.intermediate_size)
    prefill_attention = 4.0 * layers * h * prompt_tokens * (prompt_tokens + 1) / 2
    decode_attention = 4.0 * layers * h * sum(prompt_tokens + offset for offset in range(1, output_tokens))
    lm_head = 2.0 * h * shape.vocab_size * output_tokens
    base = dense_per_token * token_forwards + prefill_attention + decode_attention + lm_head
    source_adapter = adapter.source_flops_per_token * token_forwards
    return {
        "decoder_tflops_estimate": base / 1e12,
        "adapter_tflops_unmerged_estimate": source_adapter / 1e12,
        "adapter_materialization_tflops_unmerged_estimate": (
            adapter.materialization_flops_per_token * token_forwards / 1e12),
        # The custom vLLM loader permanently adds the adapter update into base
        # projection weights before generation.
        "adapter_tflops_merged_runtime": 0.0,
        "total_tflops_unmerged_estimate": (base + source_adapter) / 1e12,
        "total_tflops_vllm_merged_estimate": base / 1e12,
    }


def _gpu_memory_mb() -> list[int] | None:
    """Return driver-reported memory for all visible GPUs, when nvidia-smi exists."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


class MemoryMonitor:
    """Sample total driver memory while vLLM workers execute a request."""

    def __init__(self) -> None:
        self.before = _gpu_memory_mb()
        self.peak = list(self.before) if self.before is not None else None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemoryMonitor":
        if self.peak is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            values = _gpu_memory_mb()
            if values is not None and self.peak is not None and len(values) == len(self.peak):
                self.peak = [max(old, new) for old, new in zip(self.peak, values)]

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        values = _gpu_memory_mb()
        if values is not None and self.peak is not None and len(values) == len(self.peak):
            self.peak = [max(old, new) for old, new in zip(self.peak, values)]

    def result(self) -> tuple[int | None, int | None, str]:
        if self.before is None or self.peak is None:
            return None, None, ""
        before, peak = sum(self.before), sum(self.peak)
        return peak, peak - before, json.dumps(self.peak)





def _actual_image_tokens(processor: Any, output: Any) -> int | None:
    ids = getattr(output, "prompt_token_ids", None)
    if not ids:
        return None
    token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if token_id is None or token_id < 0:
        return None
    return sum(int(value == token_id) for value in ids)



def summarize_fixed_budgets(rows: list[dict[str, Any]], budgets: list[int]) -> list[dict[str, Any]]:
    """Average independent Art Theory measurements for each fixed token budget."""
    numeric_fields = (
        "actual_image_tokens", "prompt_tokens", "generated_tokens", "end_to_end_ms",
        "end_to_end_ms_per_generated_token", "prefill_ms", "prefill_ms_per_prompt_token",
        "decoding_ms", "decoding_ms_per_token", "peak_memory_mb", "memory_delta_mb",
        "adapter_parameter_count", "adapter_module_count", "decoder_tflops_estimate",
        "adapter_tflops_unmerged_estimate", "adapter_materialization_tflops_unmerged_estimate",
        "adapter_tflops_merged_runtime", "total_tflops_unmerged_estimate",
        "total_tflops_vllm_merged_estimate",
    )
    result: list[dict[str, Any]] = []
    for budget in budgets:
        group = [row for row in rows if row["requested_visual_tokens"] == budget]
        if not group:
            raise RuntimeError(f"No measurements were recorded for fixed visual-token budget {budget}.")
        first = group[0]
        summary: dict[str, Any] = {
            "model": first["model"], "model_kind": first["model_kind"],
            "checkpoint": first["checkpoint"], "dtype": first["dtype"],
            "subject": SUBJECT, "requested_visual_tokens": budget,
            "sample_count": len(group), "max_tokens": first["max_tokens"],
            "adapter_type": first["adapter_type"],
        }
        for field in numeric_fields:
            values = [row[field] for row in group if row[field] is not None]
            summary[field] = sum(values) / len(values) if values else None
        per_gpu = [json.loads(row["peak_memory_mb_per_gpu"]) for row in group
                   if row["peak_memory_mb_per_gpu"]]
        summary["peak_memory_mb_per_gpu"] = json.dumps([
            sum(values[index] for values in per_gpu) / len(per_gpu)
            for index in range(len(per_gpu[0]))
        ]) if per_gpu and len({len(values) for values in per_gpu}) == 1 else ""
        result.append(summary)
    return result


async def stream_generate_with_timing(
    engine: Any,
    request: Mapping[str, Any],
    sampling_params: Any,
    request_id: str,
) -> tuple[Any, float | None]:
    """Generate one request and timestamp the first streamed completion token.

    RequestOutput.metrics is not populated by every vLLM deployment. The async
    engine instead yields request outputs as generation proceeds, letting this
    benchmark measure time-to-first-token independently of that optional
    metadata. Outputs are cumulative by default, so a non-empty token ID list
    identifies the first generated token even when an initial empty update is
    emitted.
    """
    final_output = None
    first_token_at: float | None = None
    async for output in engine.generate(request, sampling_params, request_id):
        final_output = output
        if first_token_at is None:
            completions = getattr(output, "outputs", ())
            if any(getattr(completion, "token_ids", ()) for completion in completions):
                first_token_at = time.perf_counter()
    if final_output is None:
        raise RuntimeError("vLLM async generation ended without a request output.")
    return final_output, first_token_at


def _model_label(raw: str, target: Path | None) -> str:
    if raw.lower() == "baseline":
        return "baseline"
    assert target is not None
    return _SAFE_NAME.sub("_", target.parent.name + "_" + target.stem).strip("_") or "adapter"


def _build_engine(args: argparse.Namespace, values: Mapping[str, Any], output_dir: Path) -> tuple[Any, Any, AdapterProfile, str, str | None, int]:
    """Create baseline or merged-adapter async vLLM and its processor/profile."""
    model_path = str(values["model_path"])
    revision = values["revision"] if model_path == MODEL else None
    shape = transformer_shape(model_path, revision)
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_kwargs: dict[str, Any] = {
        "model": model_path,
        "revision": revision,
        "dtype": DTYPES[args.dtype],
        "seed": int(values["seed"]),
        "gpu_memory_utilization": float(values["gpu_util"]),
        "max_model_len": values["max_model_len"],
        "max_num_seqs": int(values["max_num_seqs"]),
        "tensor_parallel_size": int(values["tensor_parallel_size"]),
        "enforce_eager": bool(values["enforce_eager"]),
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"image": 7},
    }

    def make_engine() -> Any:
        return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))

    is_baseline = args.model.lower() == "baseline"
    if is_baseline:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, revision=revision)
        engine_kwargs.update(tokenizer_revision=revision)
        return make_engine(), processor, AdapterProfile("none", 0, 0, 0.0, 0.0), model_path, revision, 0

    target = resolve_target(args.model)
    train_config = load_training_config(target)
    adapter = as_mapping(train_config.get("adapter", {}), "checkpoint config.adapter")
    checkpoint = CheckpointManager.load(target, map_location="cpu")
    profile = _adapter_profile_with_count(checkpoint, adapter, shape)
    coconut = as_mapping(train_config.get("coconut", {}), "checkpoint config.coconut")
    stage = checkpoint_stage(checkpoint)
    uses_coconut_tokens = not (stage == 0 and bool(coconut.get("stage0_plain_sft", False)))
    latent_count = latent_token_count(coconut, stage) if uses_coconut_tokens else 0

    from functools import partial
    from eval.custom.loader import LOAD_FORMAT
    from eval.custom.mmmu import _checkpoint_embedding_vocab_size, _prepare_processor
    from eval.custom.model import coconut_hf_override, register

    register()
    checkpoint_vocab_size = _checkpoint_embedding_vocab_size(checkpoint)
    tokenizer_dir = output_dir / f".{_model_label(args.model, target)}_tokenizer"
    processor, token_ids = _prepare_processor(
        model_path, revision, tokenizer_dir, checkpoint_vocab_size=checkpoint_vocab_size,
        use_coconut_tokens=uses_coconut_tokens,
    )
    engine_kwargs.update(
        tokenizer=str(tokenizer_dir),
        tokenizer_revision=None,
        worker_cls="eval.custom.runner.CoconutGPUWorker",
        load_format=LOAD_FORMAT,
        model_loader_extra_config={"target": str(target), "adapter": adapter,
                                   "uses_coconut_tokens": uses_coconut_tokens,
                                   "checkpoint_vocab_size": checkpoint_vocab_size, **(token_ids or {})},
        hf_overrides=partial(coconut_hf_override, vocab_size=checkpoint_vocab_size),
    )
    return make_engine(), processor, profile, model_path, revision, latent_count


async def async_main() -> int:
    args = parse_args()
    values = config_defaults(args.config)
    if not values["max_model_len"]:
        raise ValueError("config.model.max_model_len is required for vLLM.")
    if values["min_pixels"] is None or values["max_pixels"] is None:
        raise ValueError("config.image.min_pixels and config.image.max_pixels are required.")
    if not 0 < float(values["gpu_util"]) <= 1:
        raise ValueError("config model vLLM GPU utilization must be in (0, 1].")

    args.dataset_revision = args.dataset_revision or values["dataset_revision"]
    args.limit_per_subject = args.limit
    records = records_for_subject(args, SUBJECT)
    if not records:
        raise RuntimeError(f"MMMU validation subject {SUBJECT!r} contains no records.")
    conversation = get_conversation(values["conversation"])

    output_dir = ROOT / "results" / "efficiency"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = None if args.model.lower() == "baseline" else resolve_target(args.model)
    label = _model_label(args.model, target)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"{label}_{args.dtype}_{SUBJECT.lower()}_{timestamp}.csv"
    samples_path = csv_path.with_name(f"{csv_path.stem}_samples.csv")
    metadata_path = csv_path.with_suffix(".json")

    engine, processor, profile, model_path, revision, latent_count = _build_engine(args, values, output_dir)
    image_factor = int(processor.image_processor.patch_size) * int(
        getattr(processor.image_processor, "merge_size", 2))
    budgets = patch_budgets(int(values["min_pixels"]), int(values["max_pixels"]), args.bin,
                            image_factor=image_factor)
    shape = transformer_shape(model_path, revision)
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=int(values["seed"]),
        bad_words=(["<|start-latent|>", "<|latent|>", "<|end-latent|>"] if latent_count else []),
        output_kind=RequestOutputKind.CUMULATIVE,
    )

    def request_for(record: Mapping[str, Any], token_budget: int) -> dict[str, Any]:
        request = fixed_token_input(record, processor, token_budget, image_factor, conversation)
        if latent_count:
            request = dict(request)
            request["prompt"] += "<|start-latent|>" + ("<|latent|>" * latent_count) + "<|end-latent|>"
        return request

    print(f"[efficiency] MMMU {SUBJECT}: {len(records)} samples; fixed token budgets={budgets}", flush=True)
    rows: list[dict[str, Any]] = []
    try:
        # Heat/stabilize the GPU with one complete real-data pass at the largest workload.
        warmup_budget = budgets[-1]
        for index, record in enumerate(records, start=1):
            await stream_generate_with_timing(
                engine, request_for(record, warmup_budget), params, f"warmup-{uuid.uuid4().hex}",
            )
            print(f"[efficiency] warmup {index}/{len(records)}", flush=True)
        print("[efficiency] warmup complete; starting randomized measured pass", flush=True)

        work = [(budget, record) for budget in budgets for record in records]
        random.Random(int(values["seed"])).shuffle(work)
        for index, (token_budget, record) in enumerate(work, start=1):
            request = request_for(record, token_budget)
            with MemoryMonitor() as monitor:
                started = time.perf_counter()
                output, first_token_at = await stream_generate_with_timing(
                    engine, request, params, f"measurement-{index}-{uuid.uuid4().hex}")
                finished = time.perf_counter()
            actual_image_tokens = _actual_image_tokens(processor, output)
            if actual_image_tokens != token_budget:
                raise RuntimeError(
                    f"MMMU record {record['id']} requested {token_budget} visual tokens but vLLM produced "
                    f"{actual_image_tokens}; refusing to mix non-fixed measurements.")
            elapsed_seconds = finished - started
            completion = output.outputs[0]
            generated_tokens = len(completion.token_ids)
            prompt_tokens = len(output.prompt_token_ids or [])
            prefill_ms = None if first_token_at is None else max(0.0, (first_token_at - started) * 1000)
            prefill_ms_per_token = None if prefill_ms is None else prefill_ms / max(prompt_tokens, 1)
            decoding_ms = None if first_token_at is None or generated_tokens < 2 else max(
                0.0, (finished - first_token_at) * 1000)
            decoding_ms_per_token = None if decoding_ms is None else decoding_ms / (generated_tokens - 1)
            peak_memory, memory_delta, per_gpu = monitor.result()
            flops = estimate_tflops(shape, prompt_tokens=prompt_tokens, output_tokens=generated_tokens, adapter=profile)
            row: dict[str, Any] = {
                "model": label,
                "model_kind": "baseline" if profile.kind == "none" and target is None else profile.kind,
                "checkpoint": "" if target is None else str(target),
                "dtype": args.dtype, "subject": SUBJECT,
                "measurement_index": index, "requested_visual_tokens": token_budget,
                "question_id": record["id"], "question_type": record["question_type"],
                "num_images": len(record["images"]), "actual_image_tokens": actual_image_tokens,
                "prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens,
                "max_tokens": args.max_tokens, "end_to_end_ms": elapsed_seconds * 1000,
                "end_to_end_ms_per_generated_token": elapsed_seconds * 1000 / max(generated_tokens, 1),
                "prefill_ms": prefill_ms, "prefill_ms_per_prompt_token": prefill_ms_per_token,
                "decoding_ms": decoding_ms, "decoding_ms_per_token": decoding_ms_per_token,
                "peak_memory_mb": peak_memory, "memory_delta_mb": memory_delta,
                "peak_memory_mb_per_gpu": per_gpu, "adapter_type": profile.kind,
                "adapter_parameter_count": profile.parameter_count,
                "adapter_module_count": profile.module_count, **flops,
            }
            rows.append(row)
            print(
                f"[efficiency] {index}/{len(work)} requested={token_budget} actual={actual_image_tokens} "
                f"id={record['id']} prefill={prefill_ms_per_token!s} ms/token "
                f"decode={decoding_ms_per_token!s} ms/token peak={peak_memory} MiB", flush=True)
    finally:
        engine.shutdown()

    summaries = summarize_fixed_budgets(rows, budgets)
    with samples_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    metadata = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_path": model_path, "model_revision": revision,
        "dataset": {"path": args.dataset_path, "revision": args.dataset_revision,
                    "split": "validation", "subject": SUBJECT},
        "fixed_visual_token_budgets": budgets,
        "num_measured_samples": len(rows),
        "warmup": {"passes": 1, "samples": len(records), "visual_tokens": warmup_budget,
                   "measurement_excluded": True},
        "measurement_order": {"type": "seeded_shuffle", "seed": int(values["seed"])},
        "vllm": {"enable_prefix_caching": False},
        "raw_measurements_csv": str(samples_path),
        "memory_method": "nvidia-smi total used memory sampled every 20 ms across visible GPUs",
        "timing_method": (
            "prefill_ms is client-observed time-to-first-token from async stream submission; "
            "decoding_ms is client-observed time from the first generated token to stream completion."),
        "flop_method": (
            "analytic text-decoder estimate (QKV/O, gated MLP, causal attention and LM head); "
            "vision encoder and normalization are excluded. adapter_tflops_unmerged_estimate models "
            "the project's original unmerged adapter implementation; vLLM merges adapter updates, "
            "so adapter_tflops_merged_runtime is zero."),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved fixed-budget efficiency CSV: {csv_path}")
    print(f"Saved per-sample efficiency CSV: {samples_path}")
    print(f"Saved measurement metadata: {metadata_path}")
    return 0

def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
