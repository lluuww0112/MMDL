"""Run one COCONUT batch and report CUDA memory by training phase.

This is intentionally separate from the training entry point: it reuses the
same model, adapter, data, and gradient-checkpointing setup without writing a
checkpoint or advancing a training run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from model.model_loader import load_qwen3_vl
from train.coconut import (
    _configure_kradapter,
    _enable_special_token_embeddings,
    _freeze_vision_encoder,
    _keep_vision_encoder_in_eval_mode,
    _load_yaml_mapping,
    _section,
    _special_tokens,
    inject_qwen_image_settings,
)
from train.dataloader import CoconutDataCollator, load_coconut_dataset, make_training_features


def _mib(value: int | float) -> float:
    return round(float(value) / 1024**2, 1)


def _record_memory_history() -> bool:
    record = getattr(torch.cuda.memory, "_record_memory_history", None)
    if record is None:
        return False
    try:
        record(enabled="all", context="all", stacks="all", max_entries=100_000)
    except TypeError:
        record(enabled="all")
    return True


def _active_blocks() -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for segment in torch.cuda.memory_snapshot():
        for block in segment.get("blocks", []):
            if block.get("state") == "active_allocated":
                blocks.append({
                    "allocated_mib": _mib(block.get("size", 0)),
                    "requested_mib": _mib(block.get("requested_size", 0)),
                    "stack": (block.get("frames") or [])[-5:],
                })
    return sorted(blocks, key=lambda block: block["allocated_mib"], reverse=True)[:12]


def _report(label: str, output_dir: Path, dump_snapshot: bool) -> None:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    report: dict[str, Any] = {
        "phase": label,
        "allocated_mib": _mib(torch.cuda.memory_allocated()),
        "reserved_mib": _mib(torch.cuda.memory_reserved()),
        "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved()),
        "driver_free_mib": _mib(free),
        "driver_total_mib": _mib(total),
    }
    if dump_snapshot:
        report["largest_active_blocks"] = _active_blocks()
        torch.cuda.memory._dump_snapshot(str(output_dir / f"{label}.pickle"))
        (output_dir / f"{label}.summary.txt").write_text(
            torch.cuda.memory_summary(abbreviated=False), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def _enable_gradient_checkpointing(model: Any, enabled: bool) -> None:
    if not enabled:
        return
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one real COCONUT batch on CUDA.")
    parser.add_argument("--config", default="config/stage/0.yaml")
    parser.add_argument("--qwen-config", default="config/train_qwen.yaml")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0", help="Device after CUDA_VISIBLE_DEVICES filtering.")
    parser.add_argument("--output-dir", default="profile_vram")
    parser.add_argument("--skip-backward", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    output_dir = Path(args.output_dir) / f"stage{args.stage}-{datetime.now():%Y%m%d-%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)
    memory_history = _record_memory_history()
    try:
        config = inject_qwen_image_settings(_load_yaml_mapping(args.config, "training config"), args.qwen_config)
        model_cfg, data_cfg = _section(config, "model"), _section(config, "data")
        coconut_cfg, training_cfg = _section(config, "coconut"), _section(config, "training")
        adapter_cfg, image_cfg = _section(config, "adapter"), _section(config, "image")
        components = load_qwen3_vl(
            model_cfg.get("path"), revision=model_cfg.get("revision"), device=str(device), device_map=None,
            torch_dtype=model_cfg.get("dtype", "bfloat16"),
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        )
        model, processor = components.model, components.processor
        tokens = _special_tokens(model, processor.tokenizer)
        adapters = _configure_kradapter(model, adapter_cfg)
        _freeze_vision_encoder(model)
        special_embedding = _enable_special_token_embeddings(model, tokens) if adapters else None
        _enable_gradient_checkpointing(model, bool(training_cfg.get("gradient_checkpointing", False)))
        model.train()
        _keep_vision_encoder_in_eval_mode(model)
        _report("01_model_ready", output_dir, memory_history)

        dataset = load_coconut_dataset(
            data_cfg["train_path"], processor, min_pixels=int(image_cfg["min_pixels"]),
            max_pixels=int(image_cfg["max_pixels"]), max_samples=data_cfg.get("max_train_samples"),
            seed=int(data_cfg.get("sample_seed", config.get("seed", 42))),
            min_len=data_cfg.get("min_len"), max_len=data_cfg.get("max_len"),
        )
        features = make_training_features(dataset, stage=args.stage, config=coconut_cfg, tokens=tokens)
        if not 0 <= args.sample_index < len(features):
            raise IndexError(f"sample index must be in [0, {len(features) - 1}]")
        batch = CoconutDataCollator(processor.tokenizer.pad_token_id, tokens.latent_id)([features[args.sample_index]])
        batch.pop("idx", None)
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        print(json.dumps({
            "input_shape": list(batch["input_ids"].shape),
            "supervised_tokens": int((batch["labels"] != -100).sum().item()),
            "pixel_values_shape": list(batch["pixel_values"].shape),
            "image_grid_thw": batch["image_grid_thw"].tolist(),
            "adapter_layers": len(adapters),
        }, ensure_ascii=False), flush=True)
        _report("02_batch_on_gpu", output_dir, memory_history)

        trainable = [
            parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter is not special_embedding and "visual" not in name.split(".")
        ]
        groups: list[dict[str, Any]] = [{"params": trainable}]
        if special_embedding is not None:
            groups.append({"params": [special_embedding], "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(groups, lr=float(training_cfg.get("lr", 2e-5)),
                                      weight_decay=float(training_cfg.get("weight_decay", 0.01)))
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        _report("03_forward", output_dir, memory_history)
        if not args.skip_backward:
            output.loss.backward()
            _report("04_backward", output_dir, memory_history)
            optimizer.step()
            _report("05_optimizer_step", output_dir, memory_history)
    finally:
        if memory_history:
            torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
