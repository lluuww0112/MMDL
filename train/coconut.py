"""Staged COCONUT training entry point for the project Qwen3-VL architecture."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
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


_VISION_ENCODER_ATTRIBUTES = ("visual", "vision_model", "vision_tower", "vision_encoder")


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"config.{key} must be a mapping.")
    return dict(value)


def _resolve_checkpoint_path(history_dir: str | Path, checkpoint: str) -> Path:
    """Resolve an explicit checkpoint path or a previous history run directory."""
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = Path(history_dir) / path
    # The resolved path is passed from ``main`` into ``train`` for a second
    # validation/loading step.  Canonicalise it here so a relative history_dir
    # is never prefixed twice (``history/history/...``).
    path = path.resolve()
    if path.is_dir():
        metadata_path = path / "metadata.json"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            latest = metadata.get("latest_checkpoint") if isinstance(metadata, Mapping) else None
            if isinstance(latest, str) and (path / latest).is_file():
                return path / latest
        checkpoints = sorted(path.glob("checkpoint_stage*_epoch*.pt"))
        if checkpoints:
            return checkpoints[-1]
        best = path / "best.pt"
        if best.is_file():
            return best
        raise FileNotFoundError(f"No checkpoint files found in history run: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


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


def _is_vision_encoder_path(name: str) -> bool:
    """Return whether a named module/parameter belongs to a vision encoder."""
    return any(part in _VISION_ENCODER_ATTRIBUTES for part in name.split("."))


def _freeze_vision_encoder(model: Any) -> list[str]:
    """Keep all vision-encoder parameters and mode fixed during COCONUT training."""
    frozen: list[str] = []
    for name, module in model.named_children():
        if name not in _VISION_ENCODER_ATTRIBUTES:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
        frozen.append(name)
    return frozen


def _keep_vision_encoder_in_eval_mode(model: Any) -> None:
    """``model.train()`` is recursive, so restore the frozen encoder's mode."""
    for name, module in model.named_children():
        if name in _VISION_ENCODER_ATTRIBUTES:
            module.eval()


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
        if not _is_vision_encoder_path(name) and isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in target_modules:
            # ``KRAdapter`` initializes its new parameters in fp32 by default.
            # Keep them in the wrapped projection's dtype so a bf16 base model
            # also trains its adapter weights in bf16.
            adapter = KRAdapter(module, scaling=scaling).to(device=module.weight.device, dtype=module.weight.dtype)
            _set_submodule(model, name, adapter)
            replaced.append(name)
    if not replaced:
        targets = ", ".join(target_modules)
        raise ValueError(f"KRAdapter did not find any target Linear modules: {targets}.")
    return replaced


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _data_sample_seed(config: Mapping[str, Any], data_cfg: Mapping[str, Any]) -> int:
    """Return the deterministic seed used when selecting a training subset."""
    seed = data_cfg.get("sample_seed", config.get("seed", 42))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("config.data.sample_seed must be an integer.")
    return seed


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


def _enable_special_token_embeddings(model: Any, tokens: CoconutSpecialTokens) -> torch.nn.Parameter:
    """Train only COCONUT's start/end latent-marker embedding rows.

    PyTorch exposes ``requires_grad`` per parameter, rather than per embedding
    row. We therefore unfreeze the embedding matrix and mask its gradients so
    only the two marker rows can change. ``<|latent|>`` is a positional
    placeholder whose embedding is replaced by a hidden state before use, so
    its learned input vector is unnecessary. The caller must put this parameter
    in an optimizer group with no weight decay; otherwise AdamW would still
    modify every frozen row through decoupled weight decay.
    """
    embedding = model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 2:
        raise TypeError("The model input embedding must expose a rank-2 weight Parameter.")
    token_ids = (tokens.start_id, tokens.end_id)
    if any(token_id < 0 or token_id >= weight.shape[0] for token_id in token_ids):
        raise ValueError("COCONUT special-token IDs are outside the input embedding table.")

    allowed_rows = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    allowed_rows[list(token_ids)] = 1
    weight.requires_grad_(True)
    weight.register_hook(
        lambda gradient: gradient * allowed_rows.to(device=gradient.device, dtype=gradient.dtype).unsqueeze(1)
    )
    return weight


def _load_state(model: Any, path: str, device: torch.device) -> tuple[int, int, int, Any | None]:
    """Load model state and return checkpoint progress plus its optimizer state.

    The caller decides whether the optimizer belongs to the stage being run.
    A completed stage must start the following stage with a fresh optimizer.
    """
    checkpoint = CheckpointManager.load(path, map_location=device)
    state = checkpoint["model_state"]
    # Meta's original checkpoints can be a base CausalLM state dict.  Our Qwen
    # subclass has the same keys, so non-strict loading supports this warm start.
    model.load_state_dict(state, strict=False)
    progress = checkpoint.get("training_state", {})
    return (
        int(progress.get("stage", 0)),
        int(progress.get("epoch", -1)) + 1,
        int(progress.get("global_step", 0)),
        checkpoint.get("optimizer_state"),
    )


def _load_yaml_mapping(path: str | Path, description: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"The {description} must be a YAML mapping.")
    return dict(config)


def _last_completed_stage(checkpoint_path: str | Path) -> int | None:
    """Read the completed-stage marker written beside a project checkpoint."""
    metadata_path = Path(checkpoint_path).parent / "metadata.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {metadata_path}")
    # A history directory resolves to ``latest_checkpoint``.  It may contain a
    # partially completed next stage, so use the last completed marker in that
    # case.  For an explicitly selected older/best checkpoint, fall back to
    # the checkpoint's own training state instead of pairing it with unrelated
    # run-level metadata.
    known_checkpoints = {metadata.get("latest_checkpoint"), metadata.get("last_completed_checkpoint")}
    if checkpoint_path.name not in known_checkpoints:
        return None
    stage = metadata.get("last_completed_stage")
    if stage is None:
        return None
    if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0:
        raise ValueError(f"Checkpoint metadata has invalid last_completed_stage: {stage!r}")
    return stage


class CoconutTrainer(Trainer):
    """Trainer adapter for COCONUT's auxiliary batch fields and frozen vision tower."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # CoconutQwen3VL accepts ``**kwargs`` for vision inputs. Trainer would
        # otherwise also forward its internal ``num_items_in_batch`` argument,
        # which Qwen then passes to the base model where it is not recognised.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                     num_items_in_batch: torch.Tensor | int | None = None) -> Any:
        inputs = dict(inputs)
        inputs.pop("idx", None)
        # Trainer calls model.train() immediately before this method. Restore
        # the intentionally frozen vision tower before its forward pass.
        _keep_vision_encoder_in_eval_mode(getattr(model, "module", model))
        outputs = model(**inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise ValueError("COCONUT model did not return a training loss.")
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model: Any, inputs: dict[str, Any], prediction_loss_only: bool,
                        ignore_keys: list[str] | None = None) -> tuple[torch.Tensor | None, None, None]:
        # Validation only needs loss; avoid gathering COCONUT's large logits.
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach().mean(), None, None


class CoconutCheckpointCallback(TrainerCallback):
    """Write project-compatible checkpoints at the end of each Trainer epoch."""

    def __init__(self, *, manager: CheckpointManager, stage: int, epoch_offset: int,
                 global_step_offset: int, has_evaluation: bool) -> None:
        self.manager = manager
        self.stage = stage
        self.epoch_offset = epoch_offset
        self.global_step_offset = global_step_offset
        self.has_evaluation = has_evaluation
        self.best_loss = float("inf")
        self.last_saved_epoch: int | None = None

    def _epoch_index(self, state: Any) -> int:
        # Trainer stores completed epochs as 1.0, 2.0, ...; our checkpoints
        # use zero-based epoch indices and may continue an interrupted stage.
        completed_epoch = max(0, math.ceil(float(state.epoch or 0)) - 1)
        return self.epoch_offset + completed_epoch

    def _save(self, state: Any, kwargs: Mapping[str, Any], metrics: Mapping[str, Any] | None = None) -> None:
        if not state.is_world_process_zero:
            return
        epoch = self._epoch_index(state)
        if self.last_saved_epoch == epoch:
            return
        metrics = dict(metrics or {})
        if "eval_loss" in metrics:
            metrics["val_loss"] = float(metrics.pop("eval_loss"))
        numeric_metrics = {name: float(value) for name, value in metrics.items() if isinstance(value, (int, float))}
        score = numeric_metrics.get("val_loss", float("inf"))
        saved = self.manager.save(
            model=kwargs["model"], optimizer=kwargs.get("optimizer"), stage=self.stage, epoch=epoch,
            global_step=self.global_step_offset + state.global_step, metrics=numeric_metrics,
            is_best=score <= self.best_loss,
        )
        self.best_loss = min(self.best_loss, score)
        self.last_saved_epoch = epoch
        print(f"saved {saved} {numeric_metrics}", flush=True)

    def on_epoch_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        if not self.has_evaluation:
            self._save(state, kwargs)
        return control

    def on_evaluate(self, args: TrainingArguments, state: Any, control: Any,
                    metrics: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        self._save(state, kwargs, metrics)
        return control


def train(config: Mapping[str, Any], config_path: str | Path, qwen_config_path: str | Path | None = None) -> None:
    model_cfg, data_cfg, coconut_cfg, training_cfg = (_section(config, key) for key in ("model", "data", "coconut", "training"))
    adapter_cfg = _section(config, "adapter")
    history_cfg = _section(config, "history")
    image_cfg = _section(config, "image")
    history_dir = config.get("history_dir", "./history")
    save_name = history_cfg.get("save_name", config.get("run_name", "qwen3-vl-coconut"))
    if not isinstance(save_name, str) or not save_name.strip():
        raise TypeError("config.history.save_name must be a non-empty string.")
    load_checkpoint = history_cfg.get("load_checkpoint") or training_cfg.get("resume_from")
    if load_checkpoint is not None and (not isinstance(load_checkpoint, str) or not load_checkpoint.strip()):
        raise TypeError("config.history.load_checkpoint must be a non-empty checkpoint path or run name when set.")
    train_path = data_cfg.get("train_path")
    if not train_path:
        raise ValueError("config.data.train_path is required; it must point to official COCONUT JSON data.")
    # Trainer/Accelerate owns process-group setup and DDP wrapping.  Select a
    # local device only for the initial model load; Trainer moves/wraps it.
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    sample_seed = _data_sample_seed(config, data_cfg)
    _seed(int(config.get("seed", 42)))

    components = load_qwen3_vl(
        model_cfg.get("path"), revision=model_cfg.get("revision"), device=str(device),
        device_map=None, torch_dtype=model_cfg.get("dtype", "bfloat16"),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model, processor = components.model, components.processor
    tokenizer = processor.tokenizer
    tokens = _special_tokens(model, tokenizer)
    adapter_modules = _configure_kradapter(model, adapter_cfg)
    frozen_vision_modules = _freeze_vision_encoder(model)
    special_embedding_parameter = _enable_special_token_embeddings(model, tokens) if adapter_modules else None
    train_set = load_coconut_dataset(
        train_path, processor, min_pixels=int(image_cfg["min_pixels"]), max_pixels=int(image_cfg["max_pixels"]),
        max_samples=data_cfg.get("max_train_samples"), seed=sample_seed,
    )
    val_set = load_coconut_dataset(
        data_cfg["val_path"], processor, min_pixels=int(image_cfg["min_pixels"]), max_pixels=int(image_cfg["max_pixels"]),
        max_samples=data_cfg.get("max_val_samples"), seed=sample_seed,
    ) if data_cfg.get("val_path") else None
    collator = CoconutDataCollator(tokenizer.pad_token_id, tokens.latent_id)

    trainable_parameters = [parameter for name, parameter in model.named_parameters()
                            if (parameter.requires_grad
                                and parameter is not special_embedding_parameter
                                and not _is_vision_encoder_path(name))]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after adapter configuration.")
    optimizer_groups: list[dict[str, Any]] = [{"params": trainable_parameters}]
    if special_embedding_parameter is not None:
        # A zero gradient alone does not freeze AdamW rows because its decoupled
        # weight decay updates them. Keep this whole matrix decay-free.
        optimizer_groups.append({"params": [special_embedding_parameter], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(optimizer_groups, lr=float(training_cfg.get("lr", 2e-5)),
                                  weight_decay=float(training_cfg.get("weight_decay", 0.01)))
    checkpoint_stage, checkpoint_epoch, global_step = 0, 0, 0
    checkpoint_optimizer_state: Any | None = None
    resume_path: Path | None = None
    if load_checkpoint:
        resume_path = _resolve_checkpoint_path(history_dir, load_checkpoint)
        checkpoint_stage, checkpoint_epoch, global_step, checkpoint_optimizer_state = _load_state(
            model, str(resume_path), device
        )
    completed_stage = _last_completed_stage(resume_path) if resume_path is not None else None
    # New runs begin at stage 0.  A completed checkpoint advances exactly one
    # stage; if interrupted mid-stage, its metadata remains unchanged and the
    # latest checkpoint resumes that same stage.
    stage = (completed_stage + 1) if completed_stage is not None else (checkpoint_stage if resume_path is not None else 0)
    max_stage = int(coconut_cfg.get("max_latent_stage", 0))
    if stage > max_stage:
        if rank == 0:
            print(f"All COCONUT stages through {max_stage} are already complete; nothing to train.", flush=True)
        return
    # Carry optimizer moments only when continuing an interrupted epoch of the
    # same stage. Advancing from a completed stage always uses a fresh AdamW.
    restoring_optimizer = resume_path is not None and stage == checkpoint_stage and checkpoint_optimizer_state is not None
    if restoring_optimizer:
        optimizer.load_state_dict(checkpoint_optimizer_state)
    start_epoch = checkpoint_epoch if resume_path is not None and stage == checkpoint_stage else 0

    manager = CheckpointManager(history_dir, save_name, config_path, config)
    if rank == 0:
        run_dir = manager.prepare()
        if qwen_config_path is not None:
            shutil.copy2(qwen_config_path, run_dir / "qwen.yaml")
        print(f"Qwen image settings: min_pixels={image_cfg['min_pixels']}, max_pixels={image_cfg['max_pixels']}", flush=True)
        print(f"Checkpoint output run: {run_dir}", flush=True)
        if resume_path is not None:
            print(f"Resumed from checkpoint: {resume_path}", flush=True)
            if restoring_optimizer:
                print(f"Restored optimizer state for interrupted stage {stage}.", flush=True)
            else:
                print(f"Initialized a fresh optimizer for stage {stage}; did not carry state from stage {checkpoint_stage}.", flush=True)
        print(f"Training COCONUT stage {stage} with Hugging Face Trainer", flush=True)
        if stage == 0:
            print("Stage 0: supervised on unsplit original_answer (no latent reasoning slots).", flush=True)
        else:
            latent_count = stage * int(coconut_cfg.get("c_thought", 1))
            print(
                f"Stage {stage}: replaces reasoning steps 1..{stage} with "
                f"{latent_count} sequential latent reasoning slots; supervises the remaining steps and answer.",
                flush=True,
            )
        if data_cfg.get("max_train_samples") is not None:
            print(f"Randomly sampling {data_cfg['max_train_samples']} training examples with seed {sample_seed}.", flush=True)
        if adapter_modules:
            print(f"KRAdapter applied to {len(adapter_modules)} Linear layers: {', '.join(adapter_modules)}", flush=True)
            print("Only <|start-latent|> and <|end-latent|> input-embedding rows are trainable.", flush=True)
        if frozen_vision_modules:
            print(f"Frozen vision encoder modules: {', '.join(frozen_vision_modules)}", flush=True)

    batch_size = int(training_cfg.get("batch_size", 1))
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    epochs_per_stage = int(coconut_cfg.get("epochs_per_stage", 1))
    if epochs_per_stage <= 0:
        raise ValueError("config.coconut.epochs_per_stage must be positive.")
    if start_epoch >= epochs_per_stage:
        if rank == 0:
            manager.mark_stage_completed(stage=stage, epoch=epochs_per_stage - 1, global_step=global_step)
            print(f"completed COCONUT stage {stage}", flush=True)
        return

    # The curriculum is fixed for a stage. Trainer's sampler supplies the
    # deterministic per-epoch shuffle and handles distributed sharding.
    train_features = make_training_features(train_set, stage=stage, config=coconut_cfg, tokens=tokens)
    val_features = (
        make_training_features(val_set, stage=stage, config=coconut_cfg, tokens=tokens)
        if val_set is not None else None
    )
    dtype = str(model_cfg.get("dtype", "bfloat16")).lower()
    bf16 = torch.cuda.is_available() and dtype in {"bf16", "bfloat16"}
    fp16 = torch.cuda.is_available() and dtype in {"fp16", "float16", "half"}
    if bf16 and fp16:
        raise ValueError("model.dtype cannot enable both bf16 and fp16 Trainer modes.")
    trainer_args = TrainingArguments(
        output_dir=str(manager.run_dir / f"trainer_stage{stage:02d}"),
        num_train_epochs=epochs_per_stage - start_epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
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
        lr_scheduler_type="constant",
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
        report_to="none",
        seed=int(config.get("seed", 42)),
        data_seed=sample_seed,
    )
    checkpoint_callback = CoconutCheckpointCallback(
        manager=manager, stage=stage, epoch_offset=start_epoch, global_step_offset=global_step,
        has_evaluation=val_features is not None,
    )
    trainer = CoconutTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_features,
        eval_dataset=val_features,
        data_collator=collator,
        optimizers=(optimizer, None),
        callbacks=[checkpoint_callback],
    )
    trainer.train()
    final_global_step = global_step + trainer.state.global_step
    if trainer.is_world_process_zero():
        manager.mark_stage_completed(stage=stage, epoch=epochs_per_stage - 1, global_step=final_global_step)
        print(f"completed COCONUT stage {stage}", flush=True)


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
    parser.add_argument("--qwen-config", default="config/train_qwen.yaml", help="Training Qwen YAML supplying single-image bounds.")
    parser.add_argument("--load-checkpoint", help="Checkpoint file or history run directory to resume from.")
    parser.add_argument("--save-name", help="New history run name used for checkpoints written by this training run.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = _load_yaml_mapping(config_path, "training config")
    bootstrap_history = _section(config, "history")
    requested_checkpoint = args.load_checkpoint or bootstrap_history.get("load_checkpoint")
    if requested_checkpoint is not None:
        if not isinstance(requested_checkpoint, str) or not requested_checkpoint.strip():
            raise TypeError("config.history.load_checkpoint must be a non-empty checkpoint path or run name when set.")
        resume_path = _resolve_checkpoint_path(config.get("history_dir", "./history"), requested_checkpoint)
    history = _section(config, "history")
    if requested_checkpoint is not None:
        # Checkpoints restore model/optimizer state and completed-stage metadata
        # only. Keep this invocation's YAML as the source of all settings.
        history["load_checkpoint"] = str(resume_path)
    if args.save_name is not None:
        history["save_name"] = args.save_name
    config["history"] = history
    train(inject_qwen_image_settings(config, args.qwen_config), config_path, args.qwen_config)


if __name__ == "__main__":
    main()
