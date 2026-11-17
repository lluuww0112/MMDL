"""LoRA stage-0 SFT for the project's Qwen3-VL COCONUT data format.

This entry point reuses COCONUT's ordinary SFT data path: each image/question
prompt is supervised against its intact ``original_answer``. It deliberately
does not register COCONUT latent tokens or run later curriculum stages.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments
import yaml

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

try:
    from .checkpoint import CheckpointManager
    from .coconut import (CoconutCheckpointCallback, CoconutTrainer, _enable_input_require_grads,
                          _freeze_vision_encoder, _load_state, _resolve_checkpoint_path,
                          _section, _set_submodule, _seed, inject_qwen_image_settings)
    from .dataloader import CoconutDataCollator, load_coconut_dataset, make_training_features
except ImportError:
    from checkpoint import CheckpointManager
    from coconut import (CoconutCheckpointCallback, CoconutTrainer, _enable_input_require_grads,
                         _freeze_vision_encoder, _load_state, _resolve_checkpoint_path,
                         _section, _set_submodule, _seed, inject_qwen_image_settings)
    from dataloader import CoconutDataCollator, load_coconut_dataset, make_training_features
from model.adapter import LoRA
from model.model_loader import load_qwen3_vl


_VISION_ENCODER_ATTRIBUTES = ("visual", "vision_model", "vision_tower", "vision_encoder")


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` with a trainable additive :class:`LoRA`."""

    def __init__(self, original_layer: nn.Linear, *, rank: int, alpha: float | None = None) -> None:
        if not isinstance(original_layer, nn.Linear):
            raise TypeError("LoRALinear can wrap only torch.nn.Linear layers.")
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRA(
            (original_layer.out_features, original_layer.in_features),
            rank=rank,
            alpha=alpha,
        )
        for parameter in self.original_layer.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original_layer(x)
        update = self.lora.get_update().to(dtype=x.dtype)
        return base + F.linear(x, update)


def _is_vision_encoder_path(name: str) -> bool:
    return any(part in _VISION_ENCODER_ATTRIBUTES for part in name.split("."))


def _configure_lora(model: Any, adapter_cfg: Mapping[str, Any]) -> list[str]:
    """Freeze the base model and add LoRA to configured Linear projections."""
    adapter_type = str(adapter_cfg.get("type", "")).lower()
    if adapter_type != "lora":
        raise ValueError(f"Unsupported adapter.type: {adapter_type!r}; expected 'lora'.")

    rank = adapter_cfg.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("config.adapter.rank must be a positive integer.")
    alpha = adapter_cfg.get("alpha", rank)
    if (not isinstance(alpha, (int, float)) or isinstance(alpha, bool)
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("config.adapter.alpha must be a finite positive number.")

    target_modules = adapter_cfg.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))
    if isinstance(target_modules, str):
        target_modules = (target_modules,)
    if (not isinstance(target_modules, (list, tuple))
            or not target_modules
            or not all(isinstance(name, str) and name for name in target_modules)):
        raise TypeError("config.adapter.target_modules must be a non-empty list of Linear module names.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    replaced: list[str] = []
    for name, module in tuple(model.named_modules()):
        if (not _is_vision_encoder_path(name)
                and isinstance(module, nn.Linear)
                and name.rsplit(".", 1)[-1] in target_modules):
            adapter = LoRALinear(module, rank=rank, alpha=float(alpha)).to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            _set_submodule(model, name, adapter)
            replaced.append(name)
    if not replaced:
        targets = ", ".join(target_modules)
        raise ValueError(f"LoRA did not find any target Linear modules: {targets}.")
    return replaced


def _lora_state_dict(model: Any) -> dict[str, torch.Tensor]:
    """Return only LoRA tensors, never their wrapped base-projection weights."""
    unwrapped = getattr(model, "module", model)
    state = unwrapped.state_dict()
    result: dict[str, torch.Tensor] = {}
    for module_name, module in unwrapped.named_modules():
        if isinstance(module, LoRALinear):
            for parameter_name in ("lora.A", "lora.B"):
                name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                value = state.get(name)
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"LoRA state is missing {name}.")
                result[name] = value
    if not result:
        raise ValueError("No LoRA tensors were found to save.")
    return result


class LoRACheckpointCallback(CoconutCheckpointCallback):
    """COCONUT checkpoint callback that saves just the LoRA factors."""

    def _save(self, state: Any, kwargs: Mapping[str, Any],
              metrics: Mapping[str, Any] | None = None) -> None:
        if not state.is_world_process_zero:
            return
        epoch = self._epoch_index(state)
        if self.last_saved_epoch == epoch:
            return
        values = dict(metrics or {})
        if "eval_loss" in values:
            values["val_loss"] = float(values.pop("eval_loss"))
        numeric_metrics = {
            name: float(value) for name, value in values.items()
            if isinstance(value, (int, float))
        }
        score = numeric_metrics.get("val_loss", float("inf"))
        saved = self.manager.save(
            model=kwargs["model"],
            optimizer=kwargs.get("optimizer"),
            stage=self.stage,
            epoch=epoch,
            global_step=self.global_step_offset + state.global_step,
            metrics=numeric_metrics,
            is_best=score <= self.best_loss,
            model_state=_lora_state_dict(kwargs["model"]),
        )
        self.best_loss = min(self.best_loss, score)
        self.last_saved_epoch = epoch
        print(f"saved {saved} {numeric_metrics}", flush=True)


def _validate_sft_config(coconut_cfg: Mapping[str, Any]) -> int:
    if not bool(coconut_cfg.get("stage0_plain_sft", False)):
        raise ValueError("LoRA training requires coconut.stage0_plain_sft: true.")
    max_stage = coconut_cfg.get("max_latent_stage", 0)
    if isinstance(max_stage, bool) or not isinstance(max_stage, int) or max_stage != 0:
        raise ValueError("LoRA SFT supports only coconut.max_latent_stage: 0.")
    epochs = coconut_cfg.get("epochs_per_stage", 1)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("config.coconut.epochs_per_stage must be a positive integer.")
    return epochs


def train(config: Mapping[str, Any], config_path: str | Path,
          qwen_config_path: str | Path, device_override: str | None = None) -> None:
    model_cfg, adapter_cfg, data_cfg, coconut_cfg, training_cfg, history_cfg, image_cfg = (
        _section(config, key)
        for key in ("model", "adapter", "data", "coconut", "training", "history", "image")
    )
    epochs = _validate_sft_config(coconut_cfg)
    train_path = data_cfg.get("train_path")
    if not isinstance(train_path, str) or not train_path:
        raise ValueError("config.data.train_path is required.")

    history_dir = config.get("history_dir", "./history")
    save_name = history_cfg.get("save_name")
    if not isinstance(save_name, str) or not save_name.strip():
        raise TypeError("config.history.save_name must be a non-empty string.")
    load_checkpoint = history_cfg.get("load_checkpoint") or training_cfg.get("resume_from")
    if load_checkpoint is not None and (not isinstance(load_checkpoint, str) or not load_checkpoint.strip()):
        raise TypeError("config.history.load_checkpoint must be a non-empty checkpoint path or run name when set.")
    resume_path = _resolve_checkpoint_path(history_dir, load_checkpoint) if load_checkpoint else None

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_override == "cpu":
        device = torch.device("cpu")
    elif device_override == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        device = torch.device("cuda", local_rank)
    elif device_override is None:
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device override: {device_override!r}")

    seed = int(config.get("seed", 42))
    sample_seed = data_cfg.get("sample_seed", seed)
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
        raise TypeError("config.data.sample_seed must be an integer.")
    _seed(seed)
    components = load_qwen3_vl(
        model_cfg.get("path"),
        revision=model_cfg.get("revision"),
        device=str(device),
        device_map=None,
        torch_dtype=model_cfg.get("dtype", "bf16"),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model, processor = components.model, components.processor
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    adapter_modules = _configure_lora(model, adapter_cfg)
    frozen_vision_modules = _freeze_vision_encoder(model)
    if bool(training_cfg.get("gradient_checkpointing", False)):
        _enable_input_require_grads(model)

    min_pixels, max_pixels = image_cfg.get("min_pixels"), image_cfg.get("max_pixels")
    if (not isinstance(min_pixels, int) or not isinstance(max_pixels, int)
            or min_pixels <= 0 or max_pixels < min_pixels):
        raise ValueError("config.image.min_pixels/max_pixels must be positive integers with min <= max.")
    train_set = load_coconut_dataset(
        train_path,
        processor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_samples=data_cfg.get("max_train_samples"),
        seed=sample_seed,
        max_len=data_cfg.get("max_len"),
        min_len=data_cfg.get("min_len"),
    )
    val_path = data_cfg.get("val_path")
    val_set = (
        load_coconut_dataset(
            val_path,
            processor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_samples=data_cfg.get("max_val_samples"),
            seed=sample_seed,
            max_len=data_cfg.get("max_len"),
            min_len=data_cfg.get("min_len"),
        )
        if val_path else None
    )
    train_features = make_training_features(train_set, stage=0, config=coconut_cfg, tokens=None)
    val_features = (
        make_training_features(val_set, stage=0, config=coconut_cfg, tokens=None)
        if val_set is not None else None
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA parameters remain after configuration.")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training_cfg.get("lr", 2e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )

    start_epoch, global_step = 0, 0
    if resume_path is not None:
        stage, start_epoch, global_step, optimizer_state = _load_state(model, str(resume_path), device, None)
        if stage != 0:
            raise ValueError("LoRA SFT can resume only a stage-0 checkpoint.")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    manager = CheckpointManager(history_dir, save_name, config_path, config)
    if rank == 0:
        run_dir = manager.prepare()
        shutil.copy2(qwen_config_path, run_dir / "qwen.yaml")
        print(f"Checkpoint output run: {run_dir}", flush=True)
        print(f"LoRA applied to {len(adapter_modules)} Linear layers: {', '.join(adapter_modules)}", flush=True)
        print(f"SFT examples: train={len(train_set)}, validation={len(val_set) if val_set else 0}", flush=True)
        if frozen_vision_modules:
            print(f"Frozen vision encoder modules: {', '.join(frozen_vision_modules)}", flush=True)
    if start_epoch >= epochs:
        if rank == 0:
            print(f"All {epochs} LoRA SFT epochs are already complete; nothing to train.", flush=True)
        return

    dtype = str(model_cfg.get("dtype", "bf16")).lower()
    bf16 = torch.cuda.is_available() and dtype in {"bf16", "bfloat16"}
    fp16 = torch.cuda.is_available() and dtype in {"fp16", "float16", "half"}
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
    if bf16 and fp16 or accumulation <= 0 or not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("Invalid precision, gradient_accumulation_steps, or warmup_ratio.")

    trainer_args = TrainingArguments(
        output_dir=str(manager.run_dir / "trainer_stage00"),
        num_train_epochs=epochs - start_epoch,
        per_device_train_batch_size=int(training_cfg.get("batch_size", 1)),
        per_device_eval_batch_size=int(training_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=accumulation,
        learning_rate=float(training_cfg.get("lr", 2e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(training_cfg.get("num_workers", 0)),
        remove_unused_columns=False,
        eval_strategy="epoch" if val_features is not None else "no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        lr_scheduler_type=str(training_cfg.get("lr_scheduler_type", "constant")),
        warmup_ratio=warmup_ratio,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
        report_to="none",
        seed=seed,
        data_seed=sample_seed,
    )
    callback = LoRACheckpointCallback(
        manager=manager,
        stage=0,
        epoch_offset=start_epoch,
        global_step_offset=global_step,
        has_evaluation=val_features is not None,
        tokens=None,
    )
    trainer = CoconutTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_features,
        eval_dataset=val_features,
        data_collator=CoconutDataCollator(processor.tokenizer.pad_token_id, latent_id=None),
        optimizers=(optimizer, None),
        callbacks=[callback],
    )
    trainer.train()
    if trainer.is_world_process_zero():
        manager.mark_stage_completed(
            stage=0,
            epoch=epochs - 1,
            global_step=global_step + trainer.state.global_step,
        )
        print("completed LoRA stage-0 plain SFT", flush=True)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("The training config must be a YAML mapping.")
    return dict(config)


def _direct_device(value: str | None) -> str | None:
    if value is None or value in {"cuda", "cpu"}:
        return value
    match = re.fullmatch(r"cuda:(\d+)", value)
    if match is None:
        raise ValueError("--device must be cuda, cuda:N, or cpu.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("--device cuda:N is supported only for a one-process launch.")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before --device cuda:N could select a GPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = match.group(1)
    return "cuda"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with LoRA stage-0 plain SFT.")
    parser.add_argument("--config", default="config/train_lora.yaml", help="Path to a LoRA SFT YAML configuration.")
    parser.add_argument("--qwen-config", default="config/qwen.yaml", help="Qwen YAML supplying single-image bounds.")
    parser.add_argument("--load-checkpoint", help="Checkpoint file or history run directory to resume from.")
    parser.add_argument("--save-name", help="New history run name for checkpoints written by this run.")
    parser.add_argument("--device", default=None, help="Direct-execution device: cuda, cuda:N, or cpu.")
    args = parser.parse_args()

    config_path, qwen_config_path = Path(args.config), Path(args.qwen_config)
    if not qwen_config_path.is_file():
        raise FileNotFoundError(f"Qwen config was not found: {qwen_config_path}")
    config = _load_yaml(config_path)
    history = _section(config, "history")
    if args.load_checkpoint is not None:
        history["load_checkpoint"] = args.load_checkpoint
    if args.save_name is not None:
        history["save_name"] = args.save_name
    config["history"] = history
    train(
        inject_qwen_image_settings(config, qwen_config_path),
        config_path,
        qwen_config_path,
        device_override=_direct_device(args.device),
    )


if __name__ == "__main__":
    main()
