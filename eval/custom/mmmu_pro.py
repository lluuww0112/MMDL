#!/usr/bin/env python3
"""Evaluate a custom checkpoint on MMMU-Pro with the existing vLLM runner.

Use --subset to select standard (10 options), standard (4 options), or vision.
Test answers and, when labels are available, scores are saved.
"""

from __future__ import annotations

import ast
import re
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_MARKER = re.compile(r"<image\s+(\d+)>")
SUBSETS = ("standard (10 options)", "standard (4 options)", "vision")


def load_pro_records(args: Any) -> dict[str, list[dict[str, Any]]]:
    """Load the Pro configuration once and group examples by subject."""
    from datasets import load_dataset

    kwargs = {"split": "test"}
    if args.dataset_revision:
        kwargs["revision"] = args.dataset_revision
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_dataset(args.dataset_path, args.subset, **kwargs):
        subject = str(row["subject"])
        if args.limit_per_subject and len(grouped[subject]) >= args.limit_per_subject:
            continue
        # Vision rows contain only the rendered question image and no option text.
        # Keep letter labels for answer extraction and scoring.
        options = ([f"Option {label} (in image)" for label in string.ascii_uppercase[:10]]
                   if args.subset == "vision" else row.get("options") or [])
        if isinstance(options, str):
            options = ast.literal_eval(options)
        if not isinstance(options, (list, tuple)) or not 1 <= len(options) <= 26:
            raise ValueError(f"Invalid options for {row['id']}: {options!r}")
        options = [str(option) for option in options]
        answer = row.get("answer")
        images = ({1: row["image"]} if args.subset == "vision" else
                  {i: row[f"image_{i}"] for i in range(1, 8) if row.get(f"image_{i}") is not None})
        if args.subset == "vision" and images[1] is None:
            raise ValueError(f"Missing vision image for {row['id']}")
        grouped[subject].append({
            "id": str(row["id"]), "subject": subject,
            "question": str(row.get("question") or ""), "question_type": "multiple-choice",
            "options": options, "choices": dict(zip(string.ascii_uppercase, options)),
            "answer": str(answer).strip() if answer is not None else None,
            "images_by_index": images, "vision": args.subset == "vision",
        })
    return dict(sorted(grouped.items()))


def input_for_pro_record(record: dict, processor: Any, min_pixels: int | None,
                         max_pixels: int | None, conversation: Any) -> dict:
    """Interleave image references and text in the order they appear in the prompt."""
    from qwen_vl_utils import process_vision_info

    def image_item(image: Any) -> dict:
        item = {"type": "image", "image": image}
        if min_pixels is not None:
            item["min_pixels"] = min_pixels
        if max_pixels is not None:
            item["max_pixels"] = max_pixels
        return item

    if record["vision"]:
        content = [image_item(record["images_by_index"][1]),
                   {"type": "text", "text": "Answer the question in the image with the correct option letter."}]
    else:
        choices = "\n".join(f"{key}. {value}" for key, value in record["choices"].items())
        fallback = f"Question: {record['question']}\nOptions:\n{choices}\nPlease select the correct answer from the options above."
        prompt = conversation.render(record, fallback=fallback)
        content = []
        last = 0
        referenced = set()
        for match in IMAGE_MARKER.finditer(prompt):
            if match.start() > last:
                content.append({"type": "text", "text": prompt[last:match.start()]})
            index = int(match.group(1))
            if index not in record["images_by_index"]:
                raise ValueError(f"Missing image_{index} for {record['id']}")
            content.append(image_item(record["images_by_index"][index]))
            referenced.add(index)
            last = match.end()
        if last < len(prompt):
            content.append({"type": "text", "text": prompt[last:]})
        # Some records contain an image without an explicit marker.
        content = [image_item(image) for index, image in record["images_by_index"].items()
                   if index not in referenced] + content

    messages = []
    if conversation.system_prompt:
        messages.append({"role": "system", "content": conversation.system_prompt})
    messages.append({"role": "user", "content": content})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, return_video_metadata=True,
    )
    multimodal = {}
    if images is not None:
        multimodal["image"] = images
    if videos is not None:
        multimodal["video"] = videos
    return {"prompt": text, "multi_modal_data": multimodal, "mm_processor_kwargs": kwargs}


if __name__ == "__main__":
    from eval.custom.mmmu import main

    raise SystemExit(main(pro=True))
