"""vLLM loader for a base Qwen3-VL model plus a COCONUT checkpoint."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from eval.custom.checkpoint import kradapter_update, resolve_target
from train.checkpoint import CheckpointManager


LOAD_FORMAT = "coconut_kradapter"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def _state_adapter_pairs(state: Mapping[str, Any], target_modules: tuple[str, ...]
                         ) -> Iterable[tuple[int, str, Any, Any]]:
    """Yield complete adapter pairs, accepting project and DDP state key forms."""
    pairs: dict[tuple[int, str], tuple[Any, Any]] = {}
    for key, weight in state.items():
        if not key.endswith(".weight") or key.endswith("original_layer.weight"):
            continue
        prefix = key[:-len(".weight")]
        match = _LAYER_RE.search(prefix)
        if match is None or match.group(2) not in target_modules:
            continue
        v_weight = state.get(f"{prefix}.v_weight")
        if v_weight is None:
            raise ValueError(f"KRAdapter checkpoint is missing {prefix}.v_weight.")
        location = int(match.group(1)), match.group(2)
        if location in pairs:
            raise ValueError(f"KRAdapter checkpoint has duplicate adapter tensors for layer {location[0]} {location[1]}.")
        pairs[location] = weight, v_weight
    for (layer_idx, projection), (weight, v_weight) in sorted(pairs.items()):
        yield layer_idx, projection, weight, v_weight


def _embedding_tensor(state: Mapping[str, Any]) -> Any | None:
    matches = [value for key, value in state.items() if key.endswith("embed_tokens.weight")]
    return matches[-1] if matches else None


def _copy_marker_rows(model: Any, state: Mapping[str, Any], token_ids: Mapping[str, int]) -> None:
    """Restore trained marker rows, or use EOS rows for an explicit dummy run."""
    import torch

    source = _embedding_tensor(state)
    embedding = model.language_model.model.embed_tokens.weight
    eos_id = getattr(model.config, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(getattr(model.config, "text_config", None), "eos_token_id", None)
    if eos_id is None:
        raise ValueError("Qwen config does not provide an EOS token ID for COCONUT marker initialization.")
    eos_id = int(eos_id)
    with torch.no_grad():
        for token_id in token_ids.values():
            if source is not None and source.shape[0] > token_id:
                embedding[token_id].copy_(source[token_id].to(device=embedding.device, dtype=embedding.dtype))
            else:
                embedding[token_id].copy_(embedding[eos_id])
        lm_head = getattr(model.language_model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight") and lm_head.weight is not embedding:
            for token_id in token_ids.values():
                lm_head.weight[token_id].copy_(lm_head.weight[eos_id])


def _merge_kradapter(model: Any, state: Mapping[str, Any], adapter_config: Mapping[str, Any]) -> int:
    """Merge project KRAdapter weights into vLLM's TP=1 Qwen projections."""
    adapter_type = str(adapter_config.get("type", "none")).lower()
    if adapter_type in {"", "none"}:
        return 0
    if adapter_type != "kradapter":
        raise ValueError(f"Unsupported adapter.type: {adapter_type!r}.")
    scaling = float(adapter_config.get("scaling", 2))
    targets = adapter_config.get("target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))
    if isinstance(targets, str):
        targets = (targets,)
    if not isinstance(targets, (list, tuple)) or not all(isinstance(item, str) for item in targets):
        raise ValueError("adapter.target_modules must be a list of projection names.")
    targets = tuple(targets)
    supported = {"q_proj", "k_proj", "v_proj", "o_proj"}
    unsupported = set(targets).difference(supported)
    if unsupported:
        raise ValueError(f"vLLM KRAdapter loader does not support target_modules: {sorted(unsupported)}.")
    layers = model.language_model.model.layers
    expected = {(index, projection) for index in range(len(layers)) for projection in targets}
    pairs = {(layer_idx, projection): (weight, v_weight)
             for layer_idx, projection, weight, v_weight in _state_adapter_pairs(state, targets)}
    missing = sorted(expected.difference(pairs))
    unexpected = sorted(set(pairs).difference(expected))
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} expected adapter pair(s), e.g. {missing[:3]}")
        if unexpected:
            detail.append(f"unexpected {len(unexpected)} adapter pair(s), e.g. {unexpected[:3]}")
        raise ValueError("KRAdapter checkpoint/config mismatch: " + "; ".join(detail))
    merged = 0
    for (layer_idx, projection), (weight, v_weight) in pairs.items():
        attention = layers[layer_idx].self_attn
        if projection == "o_proj":
            destination = attention.o_proj.weight
            update = kradapter_update(weight, v_weight, out_features=destination.shape[0],
                                      in_features=destination.shape[1], scaling=scaling)
            destination.add_(update.to(device=destination.device, dtype=destination.dtype))
        else:
            destination = attention.qkv_proj.weight
            output_size = attention.q_size if projection == "q_proj" else attention.kv_size
            update = kradapter_update(weight, v_weight, out_features=output_size,
                                      in_features=destination.shape[1], scaling=scaling)
            offset = {"q_proj": 0, "k_proj": attention.q_size,
                      "v_proj": attention.q_size + attention.kv_size}[projection]
            destination[offset:offset + output_size].add_(update.to(device=destination.device,
                                                                      dtype=destination.dtype))
        merged += 1
    return merged


def register() -> None:
    """Register the loader lazily after vLLM is importable."""
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader(LOAD_FORMAT)
    class CoconutKRAdapterLoader(BaseModelLoader):
        def __init__(self, load_config: Any) -> None:
            super().__init__(load_config)
            extra = dict(load_config.model_loader_extra_config)
            try:
                self.target = resolve_target(extra.pop("target"))
                self.token_ids = {name: int(extra.pop(name)) for name in
                                  ("start_latent_id", "latent_token_id", "end_latent_id")}
            except KeyError as error:
                raise ValueError(f"{LOAD_FORMAT} requires model_loader_extra_config.{error.args[0]}.") from error
            self.adapter_config = dict(extra.pop("adapter", {}))
            # Keep DefaultModelLoader's input validation for its supported options.
            base_config = dataclasses.replace(load_config, load_format="auto",
                                              model_loader_extra_config=extra)
            self.base_loader = DefaultModelLoader(base_config)

        def download_model(self, model_config: Any) -> None:
            self.base_loader.download_model(model_config)

        def _base_weights_with_marker_rows(self, model: Any, model_config: Any) -> Iterable[tuple[str, Any]]:
            """Resize padded base token/lm-head weights to the COCONUT vocab."""
            import torch

            target_vocab = max(self.token_ids.values()) + 1
            eos_id = getattr(model.config, "eos_token_id", None)
            if eos_id is None:
                eos_id = getattr(getattr(model.config, "text_config", None), "eos_token_id", None)
            if eos_id is None:
                raise ValueError("Qwen config does not provide an EOS token ID for COCONUT vocab padding.")
            eos_id = int(eos_id)
            for name, tensor in self.base_loader.get_all_weights(model_config, model):
                if name.endswith(("embed_tokens.weight", "lm_head.weight")) and tensor.ndim == 2:
                    if tensor.shape[0] > target_vocab:
                        # Qwen config pads vocab rows beyond tokenizer length;
                        # training resize_token_embeddings removes that padding.
                        tensor = tensor[:target_vocab]
                    elif tensor.shape[0] < target_vocab:
                        if eos_id >= tensor.shape[0]:
                            raise ValueError("Base model EOS token is outside its embedding weight.")
                        missing = target_vocab - tensor.shape[0]
                        tensor = torch.cat((tensor, tensor[eos_id:eos_id + 1].expand(missing, -1)), dim=0)
                yield name, tensor

        def load_weights(self, model: Any, model_config: Any) -> None:
            import torch

            # DefaultModelLoader expects source and destination vocabulary sizes
            # to be equal. COCONUT adds exactly three marker rows, so extend
            # base embedding/lm-head tensors with the EOS row before mapping.
            model.load_weights(self._base_weights_with_marker_rows(model, model_config))
            payload = CheckpointManager.load(self.target, map_location="cpu")
            state = payload.get("model_state", {})
            if not isinstance(state, Mapping):
                raise ValueError(f"COCONUT checkpoint {self.target} has no model_state mapping.")
            with torch.no_grad():
                _copy_marker_rows(model, state, self.token_ids)
                merged = _merge_kradapter(model, state, self.adapter_config)
            model.coconut_latent_token_id = self.token_ids["latent_token_id"]
            model.coconut_start_latent_token_id = self.token_ids["start_latent_id"]
            model.coconut_end_latent_token_id = self.token_ids["end_latent_id"]
            model.coconut_merged_kradapter_count = merged

    return None
