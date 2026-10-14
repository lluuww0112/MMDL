"""Saving and loading resumable COCONUT checkpoints under ``history/``."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _torch() -> Any:
    import torch
    return torch


@dataclass
class CheckpointManager:
    """Persist weights, optimizer state, metrics, and the exact run config."""
    history_dir: str | Path
    run_name: str
    config_path: str | Path
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        name = self.run_name.strip().replace("/", "_")
        if not name:
            raise ValueError("run_name must not be empty.")
        self.run_dir = Path(self.history_dir) / name

    def prepare(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        source, destination = Path(self.config_path), self.run_dir / "config.yaml"
        if source.is_file():
            shutil.copy2(source, destination)
        else:
            import yaml
            destination.write_text(yaml.safe_dump(dict(self.config), sort_keys=False), encoding="utf-8")
        metadata = self.run_dir / "metadata.json"
        if not metadata.exists():
            metadata.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
        return self.run_dir

    def path_for(self, stage: int, epoch: int) -> Path:
        return self.run_dir / f"checkpoint_stage{stage:02d}_epoch{epoch:04d}.pt"

    def save(self, *, model: Any, optimizer: Any | None, stage: int, epoch: int,
             global_step: int, metrics: Mapping[str, float] | None = None,
             is_best: bool = False) -> Path:
        self.prepare()
        unwrapped = getattr(model, "module", model)
        payload = {
            "format_version": 1,
            "model_state": unwrapped.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "training_state": {"stage": stage, "epoch": epoch, "global_step": global_step},
            "metrics": dict(metrics or {}),
        }
        output, temporary = self.path_for(stage, epoch), self.path_for(stage, epoch).with_suffix(".tmp")
        _torch().save(payload, temporary)
        temporary.replace(output)
        if is_best:
            shutil.copy2(output, self.run_dir / "best.pt")
        return output

    @staticmethod
    def load(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
        """Support both new checkpoints and Meta COCONUT raw state dicts."""
        payload = _torch().load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Checkpoint {path} does not contain a mapping.")
        if "model_state" in payload:
            return dict(payload)
        return {"model_state": dict(payload), "optimizer_state": None, "training_state": {}}

    def latest(self) -> Path | None:
        matches = sorted(self.run_dir.glob("checkpoint_stage*_epoch*.pt"))
        return matches[-1] if matches else None
