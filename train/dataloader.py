"""Single-image Qwen3-VL data pipeline for staged COCONUT training.

Each record keeps Meta COCONUT's reasoning fields and adds exactly one image:
``{"image": "path-or-url", "question": str, "answer": str, "steps": [str]}``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100
START_LATENT_TOKEN = "<|start-latent|>"
LATENT_TOKEN = "<|latent|>"
END_LATENT_TOKEN = "<|end-latent|>"


@dataclass(frozen=True)
class CoconutSpecialTokens:
    start_id: int
    latent_id: int
    end_id: int


@dataclass
class TokenizedExample:
    index: int
    question: list[int]
    steps: list[list[int]]
    answer: list[int]
    vision_inputs: dict[str, torch.Tensor]


class CoconutDataset(Dataset[TokenizedExample]):
    def __init__(self, examples: Sequence[TokenizedExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenizedExample:
        return self.examples[index]


def _image_source(value: Any, data_dir: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("image must be one non-empty local path or URL string.")
    if "://" in value or Path(value).is_absolute():
        return value
    return str((data_dir / value).resolve())


def _messages(row: Mapping[str, Any], image: str, min_pixels: int, max_pixels: int, conversation_name: str) -> list[dict[str, Any]]:
    """Use the same Qwen user-message/template contract as MMMU evaluation."""
    from model.conversation import get_conversation

    conversation = get_conversation(conversation_name)
    prompt = conversation.render(row, fallback=str(row["question"]))
    messages: list[dict[str, Any]] = []
    if conversation.system_prompt:
        messages.append({"role": "system", "content": conversation.system_prompt})
    messages.append({"role": "user", "content": [
        {"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels},
        {"type": "text", "text": prompt},
    ]})
    return messages


def _encode_qwen_prompt(processor: Any, messages: list[dict[str, Any]]) -> tuple[list[int], dict[str, torch.Tensor]]:
    """Create the exact Qwen prompt ids and vision tensors used by the model."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as error:
        raise RuntimeError("Single-image COCONUT training requires qwen_vl_utils.") from error
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    if videos is not None:
        raise ValueError("COCONUT VLM training supports exactly one image and no videos.")
    encoded = processor(text=[text], images=images, padding=False, return_tensors="pt")
    input_ids = encoded.pop("input_ids")[0].tolist()
    encoded.pop("attention_mask", None)
    encoded.pop("token_type_ids", None)
    vision = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    if "pixel_values" not in vision or "image_grid_thw" not in vision:
        raise ValueError("Qwen processor did not return pixel_values and image_grid_thw for the image.")
    return input_ids, vision


def load_coconut_dataset(path: str | Path, processor: Any, *, min_pixels: int, max_pixels: int,
                         conversation_name: str = "qwen3_vl_mmmu", max_samples: int | None = None) -> CoconutDataset:
    """Load official COCONUT reasoning data augmented with one image per sample."""
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("min_pixels/max_pixels must be positive with min_pixels <= max_pixels.")
    data_path = Path(path)
    with data_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError("COCONUT data must be a non-empty JSON list.")
    raw = raw if max_samples is None else raw[:max_samples]
    tokenizer, eos = processor.tokenizer, processor.tokenizer.eos_token_id
    examples: list[TokenizedExample] = []
    for index, row in enumerate(raw):
        required = {"image", "question", "answer", "steps"}
        if not isinstance(row, Mapping) or required.difference(row):
            raise ValueError(f"Example {index} must contain image, question, answer, and steps.")
        if not isinstance(row["question"], str) or not isinstance(row["answer"], str):
            raise TypeError(f"Example {index} question and answer must be strings.")
        if not isinstance(row["steps"], list) or not all(isinstance(step, str) for step in row["steps"]):
            raise TypeError(f"Example {index} steps must be a list of strings.")
        image = _image_source(row["image"], data_path.parent)
        prompt_ids, vision = _encode_qwen_prompt(processor, _messages(row, image, min_pixels, max_pixels, conversation_name))
        examples.append(TokenizedExample(
            index=index, question=prompt_ids,
            steps=[list(tokenizer.encode(step + "\n", add_special_tokens=False)) for step in row["steps"]],
            answer=list(tokenizer.encode("### " + row["answer"], add_special_tokens=False)) + [eos],
            vision_inputs=vision,
        ))
    if not examples:
        raise ValueError("max_samples excluded every data example.")
    return CoconutDataset(examples)


def _value(config: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def make_training_features(dataset: Iterable[TokenizedExample], *, stage: int, config: Mapping[str, Any] | Any,
                           tokens: CoconutSpecialTokens, seed: int = 0, shuffle: bool = False) -> list[dict[str, Any]]:
    """Replace whole CoT steps with learnable continuous-thought positions."""
    if stage < 0:
        raise ValueError("stage must be non-negative.")
    c_thought, max_stage = int(_value(config, "c_thought", 1)), int(_value(config, "max_latent_stage", stage))
    if c_thought < 0 or max_stage < 0:
        raise ValueError("Invalid COCONUT curriculum values.")
    rng, output = random.Random(seed), []
    for item in dataset:
        selected = stage
        skipped = len(item.steps) if selected > max_stage else min(selected, len(item.steps), max_stage)
        prefix = list(item.question) + [tokens.start_id] + [tokens.latent_id] * (skipped * c_thought) + [tokens.end_id]
        target = [token for step in item.steps[skipped:] for token in step] + list(item.answer)
        ids = prefix + target
        output.append({"input_ids": ids, "labels": [IGNORE_INDEX] * len(prefix) + target,
                       "attention_mask": [1] * len(ids), "position_ids": list(range(len(ids))),
                       "vision_inputs": item.vision_inputs, "idx": item.index})
    if shuffle:
        rng.shuffle(output)
    return output


def make_generation_features(dataset: Iterable[TokenizedExample], *, stage: int, config: Mapping[str, Any] | Any,
                             tokens: CoconutSpecialTokens) -> list[dict[str, Any]]:
    c_thought, max_stage = int(_value(config, "c_thought", 1)), int(_value(config, "max_latent_stage", stage))
    output = []
    for item in dataset:
        count = max_stage if bool(_value(config, "pad_latent_to_max", False)) else min(stage, max_stage, len(item.steps))
        ids = list(item.question) + [tokens.start_id] + [tokens.latent_id] * (count * c_thought) + [tokens.end_id]
        output.append({"input_ids": ids, "attention_mask": [1] * len(ids), "position_ids": list(range(len(ids))),
                       "vision_inputs": item.vision_inputs, "idx": item.index})
    return output


@dataclass
class CoconutDataCollator:
    """Pad token sequences and concatenate Qwen single-image vision tensors."""
    pad_token_id: int
    latent_id: int
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        rows = [dict(feature) for feature in features]
        positions = [row["input_ids"].index(self.latent_id) for row in rows if self.latent_id in row["input_ids"]]
        target = max(positions) if positions else None
        for row in rows:
            ids = list(row["input_ids"])
            left = target - ids.index(self.latent_id) if target is not None and self.latent_id in ids else 0
            row["input_ids"] = [self.pad_token_id] * left + ids
            row["attention_mask"] = [0] * left + list(row.get("attention_mask", [1] * len(ids)))
            row["position_ids"] = [0] * left + list(row.get("position_ids", range(len(ids))))
            if "labels" in row:
                row["labels"] = [self.label_pad_token_id] * left + list(row["labels"])
        width, batch = max(len(row["input_ids"]) for row in rows), {}
        for key, fill in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("position_ids", 0), ("labels", self.label_pad_token_id)):
            if key in rows[0]:
                batch[key] = torch.tensor([list(row[key]) + [fill] * (width - len(row[key])) for row in rows], dtype=torch.long)
        vision_keys = rows[0]["vision_inputs"].keys()
        if any(row["vision_inputs"].keys() != vision_keys for row in rows):
            raise ValueError("Every single-image sample must expose the same Qwen vision tensors.")
        for key in vision_keys:
            batch[key] = torch.cat([row["vision_inputs"][key] for row in rows], dim=0)
        if "idx" in rows[0]:
            batch["idx"] = torch.tensor([row["idx"] for row in rows], dtype=torch.long)
        return batch


get_dataset = load_coconut_dataset
get_cot_latent_dataset = make_training_features
get_question_latent_dataset = make_generation_features
MyCollator = CoconutDataCollator
