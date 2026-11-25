#!/usr/bin/env python3
"""Visualize MMMU text-to-image attention for a saved Qwen3-VL checkpoint.

The saved maps are computed from the selected decoder layers' Q/K projections.
For every text query token after the image placeholders, logits are normalized
*only across visual keys*.  The resulting distributions are averaged over
heads and then over query tokens, which gives one probability map per source
image.  This is intentionally different from the model's ordinary causal
attention softmax (whose denominator also includes text tokens).

Example:
    python eval/check_attention.py \
        --target history/stage1 \
        --subject Art \
        --layer 1,2 \
        --config config/qwen.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.custom.checkpoint import checkpoint_stage, latent_token_count, load_training_config, resolve_target
from eval.qwen.mmmu import DATASET, SUBJECTS, _legacy_prompt, config_defaults, records_for_subject
from model.conversation import get_conversation
from model.model_loader import load_qwen3_vl
from train.checkpoint import CheckpointManager


START_LATENT = "<|start-latent|>"
LATENT = "<|latent|>"
END_LATENT = "<|end-latent|>"
DISPLAY_CLIP_PERCENTILE = 95.0


def _layers(values: list[str]) -> list[int]:
    """Accept both ``--layer 1 2`` and ``--layer 1,2``."""
    result: list[int] = []
    for value in values:
        for part in value.split(","):
            try:
                layer = int(part)
            except ValueError as error:
                raise argparse.ArgumentTypeError(f"Invalid layer index: {part!r}") from error
            if layer < 0:
                raise argparse.ArgumentTypeError("Layer indices must be non-negative.")
            if layer not in result:
                result.append(layer)
    if not result:
        raise argparse.ArgumentTypeError("At least one --layer index is required.")
    return result


def _config_mapping(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Inference config must be a mapping: {path}")
    return dict(value)


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=ROOT / "config" / "qwen.yaml")
    selected, _ = pre.parse_known_args()
    values = config_defaults(selected.config)
    config = _config_mapping(selected.config)
    model_config = config.get("model", {})
    if not isinstance(model_config, Mapping):
        raise TypeError("config.model must be a mapping.")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=Path, required=True, help="Checkpoint file or run directory under history/.")
    parser.add_argument("--subject", required=True, choices=SUBJECTS, help="MMMU validation subject to inspect.")
    parser.add_argument("--layer", nargs="+", required=True, metavar="INDEX",
                        help="LLM layer index/indices, e.g. --layer 1 2 or --layer 1,2.")
    parser.add_argument("--config", type=Path, default=selected.config,
                        help="Qwen inference YAML (model, prompt, and image settings).")
    parser.add_argument("--model-path", default=values["model_path"])
    parser.add_argument("--model-revision", default=values["revision"])
    parser.add_argument("--dataset-path", default=DATASET)
    parser.add_argument("--dataset-revision", default=values["dataset_revision"])
    parser.add_argument("--dtype", default=values["dtype"], choices=("auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"))
    parser.add_argument("--device", default=model_config.get("device"),
                        help="Device holding the complete model; defaults to cuda:0 when available.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of subject samples (0 = all).")
    parser.add_argument("--query-chunk-size", type=int, default=64,
                        help="Text-query rows evaluated at a time while extracting attention.")
    parser.add_argument("--output-path", type=Path,
                        help="Default: results/check_attention/<checkpoint>/<subject>.")
    args = parser.parse_args()
    args.layer = _layers(args.layer)
    if args.limit < 0 or args.query_chunk_size < 1:
        parser.error("--limit must be non-negative and --query-chunk-size must be positive.")
    # Reuse eval.qwen.mmmu.records_for_subject without changing its public
    # evaluation CLI contract.
    args.limit_per_subject = args.limit
    return args


def _load_checkpoint_model(args: argparse.Namespace) -> tuple[Any, Any, Path, int, int, set[int]]:
    """Restore the same base model, adapter, and latent-token state as training."""
    import torch

    target = resolve_target(args.target)
    train_config = load_training_config(target)
    checkpoint = CheckpointManager.load(target, map_location="cpu")
    stage = checkpoint_stage(checkpoint)
    coconut = train_config.get("coconut", {})
    adapter = train_config.get("adapter", {})
    model_train_config = train_config.get("model", {})
    if not isinstance(coconut, Mapping) or not isinstance(adapter, Mapping) or not isinstance(model_train_config, Mapping):
        raise TypeError("Checkpoint train.yaml must contain mapping-valued model, adapter, and coconut sections.")

    use_plain_sft = stage == 0 and bool(coconut.get("stage0_plain_sft", False))
    use_coconut_tokens = not use_plain_sft
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    attn_implementation = model_train_config.get("attn_implementation", "sdpa")
    components = load_qwen3_vl(
        args.model_path,
        revision=args.model_revision,
        device=device,
        device_map=None,
        torch_dtype=args.dtype,
        attn_implementation=attn_implementation,
    )
    model, processor = components.model, components.processor

    tokens = None
    if use_coconut_tokens:
        # These functions are intentionally reused from training so that marker
        # IDs and the COCONUT feedback forward exactly match the checkpoint.
        from train.coconut import _special_tokens

        tokens = _special_tokens(model, processor.tokenizer)

    adapter_type = str(adapter.get("type", "none")).lower()
    if adapter_type == "kradapter":
        from train.coconut import _configure_kradapter

        _configure_kradapter(model, adapter)
    elif adapter_type == "lora":
        from train.lora import _configure_lora

        _configure_lora(model, adapter)
    elif adapter_type not in {"", "none"}:
        raise ValueError(f"Unsupported adapter.type in {target.parent / 'train.yaml'}: {adapter_type!r}")

    # _load_model_state also handles full checkpoints and vocabulary rows whose
    # shape differs from a base Qwen checkpoint.
    from train.coconut import _load_model_state, _restore_special_token_state

    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint {target} has no model_state mapping.")
    _load_model_state(model, state)
    special_state = checkpoint.get("special_token_state")
    if special_state is not None:
        if not isinstance(special_state, Mapping):
            raise TypeError("Checkpoint special_token_state must be a mapping.")
        _restore_special_token_state(model, special_state, tokens)
    model.eval()

    latent_count = latent_token_count(coconut, stage) if use_coconut_tokens else 0
    control_ids = set()
    if tokens is not None:
        control_ids = {int(tokens.start_id), int(tokens.latent_id), int(tokens.end_id)}
    return model, processor, target, stage, latent_count, control_ids


def _messages_and_prompt(record: dict[str, Any], processor: Any, args: argparse.Namespace, latent_count: int,
                         inference: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    """Build the MMMU input exactly as eval.qwen.mmmu, plus COCONUT controls."""
    conversation = get_conversation(str(inference["conversation"]))
    content: list[dict[str, Any]] = []
    for image in record["images"]:
        item: dict[str, Any] = {"type": "image", "image": image}
        if inference["min_pixels"] is not None:
            item["min_pixels"] = inference["min_pixels"]
        if inference["max_pixels"] is not None:
            item["max_pixels"] = inference["max_pixels"]
        content.append(item)
    question_text = conversation.render(record, fallback=_legacy_prompt(record))
    messages: list[dict[str, Any]] = []
    if conversation.system_prompt:
        messages.append({"role": "system", "content": conversation.system_prompt})
    messages.append({"role": "user", "content": [*content, {"type": "text", "text": question_text}]})
    model_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if latent_count:
        model_text += START_LATENT + (LATENT * latent_count) + END_LATENT
    return messages, model_text, question_text


def _processor_inputs(messages: list[dict[str, Any]], model_text: str, processor: Any) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    images, videos, vision_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    encoded = processor(
        text=[model_text], images=images, videos=videos, padding=True,
        return_tensors="pt", **vision_kwargs,
    )
    return dict(encoded)


class VisualOnlyAttention:
    """Capture selected layers without materializing their full attention maps."""

    def __init__(self, *, layer_indices: list[int], visual_positions: Any, query_positions: Any,
                 chunk_size: int) -> None:
        self.layer_indices = layer_indices
        self.visual_positions = visual_positions
        self.query_positions = query_positions
        self.chunk_size = chunk_size
        self.results: dict[int, Any] = {}
        self._lengths: dict[int, int] = {}

    def hook(self, layer_index: int):
        def capture(module: Any, positional: tuple[Any, ...], keyword: dict[str, Any]) -> None:
            import torch
            from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

            hidden_states = keyword.get("hidden_states", positional[0] if positional else None)
            position_embeddings = keyword.get("position_embeddings")
            if hidden_states is None or position_embeddings is None or hidden_states.ndim != 3:
                return
            sequence_length = int(hidden_states.shape[1])
            # A COCONUT forward has shorter prefix passes before its final full
            # pass. Keeping only the longest pass yields the final contextualized
            # sequence and works for ordinary non-COCONUT checkpoints as well.
            if sequence_length < self._lengths.get(layer_index, -1):
                return
            visual = self.visual_positions[self.visual_positions < sequence_length]
            query = self.query_positions[self.query_positions < sequence_length]
            if visual.numel() == 0 or query.numel() == 0:
                return

            input_shape = hidden_states.shape[:-1]
            shaped = (*input_shape, -1, module.head_dim)
            query_states = module.q_norm(module.q_proj(hidden_states).view(shaped)).transpose(1, 2)
            key_states = module.k_norm(module.k_proj(hidden_states).view(shaped)).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            # Qwen3-VL uses grouped-query attention.  Match eager_attention_forward
            # so every query head sees its corresponding repeated KV head.
            key_states = repeat_kv(key_states, module.num_key_value_groups)

            # The MMMU image placeholders precede its question. Retain only
            # causal text rows with every selected visual key in their past.
            query = query[query >= int(visual.max())]
            if query.numel() == 0:
                return
            visual_keys = key_states[:, :, visual, :]
            total = torch.zeros(visual.numel(), device=hidden_states.device, dtype=torch.float32)
            for start in range(0, int(query.numel()), self.chunk_size):
                query_chunk = query[start:start + self.chunk_size]
                query_vectors = query_states[:, :, query_chunk, :]
                logits = torch.matmul(query_vectors.float(), visual_keys.float().transpose(2, 3)) * module.scaling
                # The required normalization: denominator is visual keys only.
                probabilities = torch.softmax(logits, dim=-1)
                total += probabilities.mean(dim=1).sum(dim=(0, 1))
            self.results[layer_index] = (total / query.numel()).detach().cpu()
            self._lengths[layer_index] = sequence_length
        return capture


def _split_image_maps(scores: Any, visual_positions: Any, image_grid_thw: Any, spatial_merge_size: int) -> list[Any]:
    """Restore one [height, width] map per original image placeholder run."""
    import torch

    grids = image_grid_thw.detach().cpu().tolist()
    expected = [int(t) * (int(height) // spatial_merge_size) * (int(width) // spatial_merge_size)
                for t, height, width in grids]
    if sum(expected) != int(scores.numel()) or len(grids) == 0:
        raise ValueError(
            "Visual placeholder count does not match image_grid_thw; cannot map attention scores back to images "
            f"({int(scores.numel())} placeholders, expected {sum(expected)})."
        )
    maps = []
    offset = 0
    for (frames, height, width), count in zip(grids, expected):
        grid_h, grid_w = int(height) // spatial_merge_size, int(width) // spatial_merge_size
        values = scores[offset:offset + count].reshape(int(frames), grid_h, grid_w)
        maps.append(values.mean(dim=0))  # MMMU images are normally one frame; average videos defensively.
        offset += count
    return maps


def _image_array(image: Any) -> Any:
    import numpy as np

    if hasattr(image, "convert"):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=-1)
    return array


def _save_visualization(path: Path, *, record: dict[str, Any], question_text: str, layer: int,
                        image_maps: list[Any], query_tokens: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    images = record["images"]
    if len(images) != len(image_maps):
        raise ValueError(f"Record {record['id']} has {len(images)} images but {len(image_maps)} attention maps.")
    all_values = np.concatenate([attention.numpy().ravel() for attention in image_maps])
    display_max = float(np.percentile(all_values, DISPLAY_CLIP_PERCENTILE))
    if display_max <= 0:
        display_max = float(all_values.max()) if all_values.size else 1.0
    prompt_lines = textwrap.wrap(question_text, width=135)
    header_height = max(1.4, 0.20 * (len(prompt_lines) + 3))
    figure = plt.figure(figsize=(17, header_height + 3.6 * len(images)), constrained_layout=True)
    grid = figure.add_gridspec(len(images) + 1, 3, height_ratios=[header_height] + [3] * len(images))
    text_axis = figure.add_subplot(grid[0, :])
    text_axis.axis("off")
    displayed = "\n".join(prompt_lines)
    text_axis.text(
        0.0, 1.0,
        f"MMMU {record['subject']} | id={record['id']} | LLM layer {layer}\n"
        f"Question presented to the model ({len(query_tokens)} text queries averaged):\n{displayed}",
        va="top", ha="left", fontsize=9,
    )
    heat_artist = None
    heat_axes = []
    for index, (image, attention) in enumerate(zip(images, image_maps), start=1):
        original = _image_array(image)
        original_axis = figure.add_subplot(grid[index, 0])
        heat_axis = figure.add_subplot(grid[index, 1])
        overlay_axis = figure.add_subplot(grid[index, 2])
        original_axis.imshow(original)
        original_axis.set_title(f"Image {index}: original")
        heat_artist = heat_axis.imshow(attention.numpy(), cmap="magma", vmin=0.0, vmax=display_max,
                                        interpolation="bilinear")
        heat_axis.set_title(f"Text → visual attention (display clipped at p{DISPLAY_CLIP_PERCENTILE:g})")
        overlay_axis.imshow(original)
        overlay_axis.imshow(attention.numpy(), cmap="magma", vmin=0.0, vmax=display_max, alpha=0.58,
                            interpolation="bilinear", extent=(0, original.shape[1], original.shape[0], 0))
        overlay_axis.set_title("Original + attention overlay")
        heat_axes.append(heat_axis)
        for axis in (original_axis, heat_axis, overlay_axis):
            axis.set_xticks([])
            axis.set_yticks([])
    assert heat_artist is not None
    figure.colorbar(heat_artist, ax=heat_axes, shrink=0.83, pad=0.02,
                    label="attention probability (visual-only softmax; display p95-clipped)")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "record"


def main() -> int:
    args = parse_args()

    import numpy as np
    import torch

    inference = config_defaults(args.config)
    model, processor, target, stage, latent_count, control_ids = _load_checkpoint_model(args)
    layers = model.model.language_model.layers
    invalid_layers = [layer for layer in args.layer if layer >= len(layers)]
    if invalid_layers:
        raise ValueError(f"Layer indices {invalid_layers} are out of range for this {len(layers)}-layer model.")

    output_path = args.output_path or ROOT / "results" / "check_attention" / target.stem / args.subject
    output_path = output_path.expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    records = records_for_subject(args, args.subject)
    if args.limit:
        records = records[:args.limit]
    device = next(model.parameters()).device
    spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
    report: dict[str, Any] = {
        "target": str(target), "subject": args.subject, "layers": args.layer,
        "checkpoint_stage": stage, "latent_token_count": latent_count,
        "normalization": "softmax independently over visual key tokens, then mean over heads and text query tokens",
        "visualization": {"colormap": "magma", "display_clip_percentile": DISPLAY_CLIP_PERCENTILE},
        "records": [],
    }

    for number, record in enumerate(records, start=1):
        messages, model_text, question_text = _messages_and_prompt(record, processor, args, latent_count, inference)
        encoded = _processor_inputs(messages, model_text, processor)
        encoded = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in encoded.items()}
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))[0].bool()
        visual_positions = ((input_ids == int(model.config.image_token_id)) & attention_mask).nonzero(as_tuple=False).flatten()
        if visual_positions.numel() == 0:
            raise ValueError(f"MMMU record {record['id']} has no visual placeholder tokens.")
        # MMMU places all images before the actual question. Query positions are
        # ordinary prompt tokens after the final visual token, excluding latent
        # marker controls when this checkpoint uses COCONUT.
        positions = torch.arange(input_ids.shape[0], device=device)
        query_positions = positions[(attention_mask) & (input_ids != int(model.config.image_token_id))]
        if control_ids:
            query_positions = query_positions[~torch.isin(input_ids[query_positions], torch.tensor(sorted(control_ids), device=device))]
        query_positions = query_positions[query_positions >= int(visual_positions.max())]
        if query_positions.numel() == 0:
            raise ValueError(f"MMMU record {record['id']} has no text query tokens after its image placeholders.")

        collector = VisualOnlyAttention(
            layer_indices=args.layer, visual_positions=visual_positions, query_positions=query_positions,
            chunk_size=args.query_chunk_size,
        )
        with ExitStack() as hooks:
            for layer in args.layer:
                handle = layers[layer].self_attn.register_forward_pre_hook(collector.hook(layer), with_kwargs=True)
                hooks.callback(handle.remove)
            with torch.inference_mode():
                model(**encoded, use_cache=False)
        missing = [layer for layer in args.layer if layer not in collector.results]
        if missing:
            raise RuntimeError(f"No attention capture from layer(s) {missing} for MMMU record {record['id']}.")

        token_strings = processor.tokenizer.convert_ids_to_tokens(input_ids[query_positions].detach().cpu().tolist())
        record_name = _safe_name(str(record["id"]))
        saved_layers: list[dict[str, Any]] = []
        for layer, scores in collector.results.items():
            maps = _split_image_maps(scores, visual_positions, encoded["image_grid_thw"], spatial_merge_size)
            image_path = output_path / f"{number:03d}_{record_name}_layer{layer:03d}.png"
            array_path = output_path / f"{number:03d}_{record_name}_layer{layer:03d}.npz"
            _save_visualization(image_path, record=record, question_text=question_text, layer=layer,
                                image_maps=maps, query_tokens=token_strings)
            arrays: dict[str, Any] = {
                "visual_positions": visual_positions.detach().cpu().numpy(),
                "query_positions": query_positions.detach().cpu().numpy(),
                "query_token_ids": input_ids[query_positions].detach().cpu().numpy(),
                "visual_attention": scores.numpy(),
            }
            arrays.update({f"image_{index}_attention": attention.numpy() for index, attention in enumerate(maps, start=1)})
            np.savez_compressed(array_path, **arrays)
            saved_layers.append({"layer": layer, "figure": image_path.name, "attention": array_path.name})
        report["records"].append({
            "id": record["id"], "question": record["question"], "model_prompt": model_text,
            "visual_token_count": int(visual_positions.numel()), "query_token_count": int(query_positions.numel()),
            "query_tokens": token_strings, "outputs": saved_layers,
        })
        print(f"[{number:3d}/{len(records):3d}] {record['id']} -> "
              + ", ".join(item["figure"] for item in saved_layers), flush=True)

    metadata = output_path / "metadata.json"
    metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved attention figures and arrays: {output_path}")
    print(f"Saved metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
