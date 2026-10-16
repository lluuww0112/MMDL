"""Checkpoint discovery and KRAdapter merge helpers for vLLM COCONUT runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def resolve_target(target: str | Path) -> Path:
    """Resolve a checkpoint file or a resumable history run directory."""
    target = Path(target).expanduser()
    if target.is_file():
        return target.resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Target checkpoint or run directory was not found: {target}")
    metadata_path = target / "metadata.json"
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        latest = metadata.get("latest_checkpoint") if isinstance(metadata, Mapping) else None
        if isinstance(latest, str) and (target / latest).is_file():
            return (target / latest).resolve()
    checkpoints = sorted(target.glob("checkpoint_stage*_epoch*.pt"))
    if checkpoints:
        return checkpoints[-1].resolve()
    if (target / "best.pt").is_file():
        return (target / "best.pt").resolve()
    raise FileNotFoundError(f"No checkpoint files found in target run directory: {target}")


def load_training_config(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).parent / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is required to restore COCONUT settings.")
    with path.open(encoding="utf-8") as handle:
        return as_mapping(yaml.safe_load(handle) or {}, f"checkpoint config {path}")


def checkpoint_stage(payload: Mapping[str, Any]) -> int:
    state = as_mapping(payload.get("training_state", {}), "checkpoint training_state")
    stage = int(state.get("stage", 0))
    if stage < 0:
        raise ValueError(f"Checkpoint has invalid stage: {stage}")
    return stage


def latent_token_count(coconut_config: Mapping[str, Any], stage: int) -> int:
    c_thought = int(coconut_config.get("c_thought", 1))
    max_stage = int(coconut_config.get("max_latent_stage", stage))
    if c_thought < 0 or max_stage < 0:
        raise ValueError("Checkpoint has invalid coconut.c_thought or coconut.max_latent_stage.")
    blocks = max_stage if bool(coconut_config.get("pad_latent_to_max", False)) else min(stage, max_stage)
    return blocks * c_thought


def kradapter_update(weight: Any, v_weight: Any, *, out_features: int, in_features: int, scaling: float) -> Any:
    """Return the exact additive ``(out, in)`` KRAdapter projection update."""
    if weight.ndim != 2 or v_weight.ndim != 2 or weight.shape[0] != v_weight.shape[0]:
        raise ValueError("Invalid KRAdapter weight/v_weight shapes.")
    import torch

    update = (weight.unsqueeze(1) * v_weight.unsqueeze(-1)).reshape(weight.shape[0], -1)
    # Keep this orientation identical to ``model.adapter.KRAdapter.get_update``.
    if min(range(2), key=lambda index: (out_features, in_features)[index]) != min(range(2), key=lambda index: update.shape[index]):
        update = update.T
    update = update[:out_features, :in_features]
    if tuple(update.shape) != (out_features, in_features):
        raise ValueError(
            f"KRAdapter update shape {tuple(update.shape)} cannot cover projection "
            f"shape {(out_features, in_features)}."
        )
    return update.to(dtype=torch.float32) * float(scaling)
