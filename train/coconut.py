"""Staged COCONUT training entry point for the project Qwen3-VL architecture."""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import yaml

try:
    from .checkpoint import CheckpointManager
    from .dataloader import (CoconutDataCollator, CoconutSpecialTokens, END_LATENT_TOKEN,
                             LATENT_TOKEN, START_LATENT_TOKEN, load_coconut_dataset,
                             make_training_features)
except ImportError:  # Allows `python train/coconut.py ...` as well as -m.
    from checkpoint import CheckpointManager
    from dataloader import (CoconutDataCollator, CoconutSpecialTokens, END_LATENT_TOKEN,
                            LATENT_TOKEN, START_LATENT_TOKEN, load_coconut_dataset,
                            make_training_features)
from model.model_loader import load_qwen3_vl
from model.adapter import KRAdapter


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"config.{key} must be a mapping.")
    return dict(value)


def _set_submodule(root: Any, path: str, replacement: torch.nn.Module) -> None:
    """Replace the module at a dotted path, including paths through ModuleLists."""
    parent = root
    *parents, attribute = path.split(".")
    for part in parents:
        parent = parent[int(part)] if isinstance(parent, torch.nn.ModuleList) and part.isdigit() else getattr(parent, part)
    if isinstance(parent, torch.nn.ModuleList) and attribute.isdigit():
        parent[int(attribute)] = replacement
    else:
        setattr(parent, attribute, replacement)


def _configure_kradapter(model: Any, adapter_cfg: Mapping[str, Any]) -> list[str]:
    """Freeze the base model and replace configured Linear projections with KRAdapters."""
    adapter_type = str(adapter_cfg.get("type", "none")).lower()
    if adapter_type in {"", "none"}:
        return []
    if adapter_type != "kradapter":
        raise ValueError(f"Unsupported adapter.type: {adapter_type!r}; expected 'kradapter' or 'none'.")

    scaling = float(adapter_cfg.get("scaling", 2))
    target_modules = adapter_cfg.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))
    if isinstance(target_modules, str):
        target_modules = (target_modules,)
    if not isinstance(target_modules, (list, tuple)) or not all(isinstance(name, str) and name for name in target_modules):
        raise TypeError("config.adapter.target_modules must be a non-empty list of Linear module names.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    replaced: list[str] = []
    for name, module in tuple(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in target_modules:
            _set_submodule(model, name, KRAdapter(module, scaling=scaling))
            replaced.append(name)
    if not replaced:
        targets = ", ".join(target_modules)
        raise ValueError(f"KRAdapter did not find any target Linear modules: {targets}.")
    return replaced


def _distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    return rank, world_size, torch.device("cpu")


def _seed(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _special_tokens(model: Any, tokenizer: Any) -> CoconutSpecialTokens:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [START_LATENT_TOKEN, LATENT_TOKEN, END_LATENT_TOKEN]})
    model.resize_token_embeddings(len(tokenizer))
    ids = CoconutSpecialTokens(
        start_id=tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN),
        latent_id=tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        end_id=tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN),
    )
    # A stable initial embedding is preferable to a randomly initialised marker.
    embeddings = model.get_input_embeddings().weight.data
    reference = tokenizer.eos_token_id
    for token_id in (ids.start_id, ids.latent_id, ids.end_id):
        embeddings[token_id].copy_(embeddings[reference])
    model.configure_coconut(latent_token_id=ids.latent_id, start_latent_id=ids.start_id, end_latent_id=ids.end_id)
    return ids


def _load_state(model: Any, optimizer: Any, path: str, device: torch.device) -> tuple[int, int, int]:
    checkpoint = CheckpointManager.load(path, map_location=device)
    state = checkpoint["model_state"]
    # Meta's original checkpoints can be a base CausalLM state dict.  Our Qwen
    # subclass has the same keys, so non-strict loading supports this warm start.
    model.load_state_dict(state, strict=False)
    if checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    progress = checkpoint.get("training_state", {})
    return int(progress.get("stage", 0)), int(progress.get("epoch", -1)) + 1, int(progress.get("global_step", 0))


def _evaluate(model: Any, loader: DataLoader[Any], device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("idx", None)
            result = model(**{key: value.to(device) for key, value in batch.items()})
            losses.append(float(result.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(config: Mapping[str, Any], config_path: str | Path, qwen_config_path: str | Path | None = None) -> None:
    model_cfg, data_cfg, coconut_cfg, training_cfg = (_section(config, key) for key in ("model", "data", "coconut", "training"))
    adapter_cfg = _section(config, "adapter")
    image_cfg = _section(config, "image")
    train_path = data_cfg.get("train_path")
    if not train_path:
        raise ValueError("config.data.train_path is required; it must point to official COCONUT JSON data.")
    rank, world_size, device = _distributed()
    _seed(int(config.get("seed", 42)), rank)

    components = load_qwen3_vl(
        model_cfg.get("path"), revision=model_cfg.get("revision"), device=str(device),
        device_map=None, torch_dtype=model_cfg.get("dtype", "bfloat16"),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model, processor = components.model, components.processor
    tokenizer = processor.tokenizer
    tokens = _special_tokens(model, tokenizer)
    adapter_modules = _configure_kradapter(model, adapter_cfg)
    train_set = load_coconut_dataset(
        train_path, processor, min_pixels=int(image_cfg["min_pixels"]), max_pixels=int(image_cfg["max_pixels"]),
        conversation_name=str(data_cfg.get("conversation", "qwen3_vl_mmmu")), max_samples=data_cfg.get("max_train_samples"),
    )
    val_set = load_coconut_dataset(
        data_cfg["val_path"], processor, min_pixels=int(image_cfg["min_pixels"]), max_pixels=int(image_cfg["max_pixels"]),
        conversation_name=str(data_cfg.get("conversation", "qwen3_vl_mmmu")), max_samples=data_cfg.get("max_val_samples"),
    ) if data_cfg.get("val_path") else None
    collator = CoconutDataCollator(tokenizer.pad_token_id, tokens.latent_id)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after adapter configuration.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(training_cfg.get("lr", 2e-5)), weight_decay=float(training_cfg.get("weight_decay", 0.01)))
    start_stage, start_epoch, global_step = 0, 0, 0
    if training_cfg.get("resume_from"):
        start_stage, start_epoch, global_step = _load_state(model, optimizer, training_cfg["resume_from"], device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    manager = CheckpointManager(config.get("history_dir", "./history"), config.get("run_name", "qwen3-vl-coconut"), config_path, config)
    if rank == 0:
        run_dir = manager.prepare()
        if qwen_config_path is not None:
            shutil.copy2(qwen_config_path, run_dir / "qwen.yaml")
        print(f"Qwen image settings: min_pixels={image_cfg['min_pixels']}, max_pixels={image_cfg['max_pixels']}", flush=True)
        if adapter_modules:
            print(f"KRAdapter applied to {len(adapter_modules)} Linear layers: {', '.join(adapter_modules)}", flush=True)
    if dist.is_initialized():
        dist.barrier()

    batch_size = int(training_cfg.get("batch_size", 1))
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    best_loss = float("inf")
    for stage in range(start_stage, int(coconut_cfg.get("max_latent_stage", 0)) + 1):
        for epoch in range(start_epoch if stage == start_stage else 0, int(coconut_cfg.get("epochs_per_stage", 1))):
            features = make_training_features(train_set, stage=stage, config=coconut_cfg,
                                              tokens=tokens, seed=int(config.get("seed", 42)) + epoch, shuffle=True)
            sampler = DistributedSampler(features, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
            loader = DataLoader(features, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
                                num_workers=int(training_cfg.get("num_workers", 0)), collate_fn=collator)
            if sampler is not None:
                sampler.set_epoch(stage * 100000 + epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                batch.pop("idx", None)
                result = model(**{key: value.to(device) for key, value in batch.items()})
                (result.loss / accumulation).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            metrics: dict[str, float] = {}
            if val_set is not None and rank == 0:
                val_features = make_training_features(val_set, stage=stage, config=coconut_cfg, tokens=tokens)
                metrics["val_loss"] = _evaluate(model.module if hasattr(model, "module") else model,
                                                  DataLoader(val_features, batch_size=batch_size, collate_fn=collator), device)
            score = metrics.get("val_loss", float("inf"))
            if rank == 0:
                saved = manager.save(model=model, optimizer=optimizer, stage=stage, epoch=epoch,
                                     global_step=global_step, metrics=metrics, is_best=score <= best_loss)
                print(f"saved {saved} {metrics}", flush=True)
            best_loss = min(best_loss, score)
            if dist.is_initialized():
                dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def inject_qwen_image_settings(config: Mapping[str, Any], qwen_config_path: str | Path) -> dict[str, Any]:
    """Inject MMMU Qwen single-image bounds into the COCONUT runtime config."""
    with Path(qwen_config_path).open(encoding="utf-8") as handle:
        qwen_config = yaml.safe_load(handle)
    if not isinstance(qwen_config, Mapping):
        raise ValueError("The Qwen config must be a YAML mapping.")
    image = qwen_config.get("image", {})
    if not isinstance(image, Mapping):
        raise TypeError("qwen.yaml image must be a mapping.")
    min_pixels, max_pixels = image.get("min_pixels"), image.get("max_pixels")
    if not isinstance(min_pixels, int) or not isinstance(max_pixels, int) or min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("qwen.yaml image.min_pixels/max_pixels must be positive integers with min <= max.")
    merged = dict(config)
    merged["image"] = {"min_pixels": min_pixels, "max_pixels": max_pixels}
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with COCONUT continuous thoughts.")
    parser.add_argument("--config", default="config/train.yaml", help="Path to a COCONUT YAML configuration.")
    parser.add_argument("--qwen-config", default="config/qwen.yaml", help="MMMU Qwen YAML supplying single-image bounds.")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("The training config must be a YAML mapping.")
    train(inject_qwen_image_settings(config, args.qwen_config), args.config, args.qwen_config)


if __name__ == "__main__":
    main()
