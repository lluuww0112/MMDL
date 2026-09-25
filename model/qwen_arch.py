"""A small, explicit inference interface for Qwen3-VL.

``Qwen3VLArchitecture`` owns the model together with the matching processor.
It converts conversational text/image/video inputs into the tensors expected by
Transformers and decodes only the tokens generated after the prompt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Qwen3VLInputError(ValueError):
    """Raised when a conversation cannot be represented as a Qwen3-VL prompt."""


class _PresencePenaltyLogitsProcessor:
    """Apply the vLLM-style presence penalty to generated tokens only."""

    def __init__(self, penalty: float, prompt_width: int) -> None:
        self.penalty = penalty
        self.prompt_width = prompt_width

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        generated_ids = input_ids[:, self.prompt_width :]
        if generated_ids.numel() == 0:
            return scores

        import torch

        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated_ids, True)
        return scores - seen.to(dtype=scores.dtype) * self.penalty


def _as_media_list(media: Any | Sequence[Any] | None) -> list[Any]:
    if media is None:
        return []
    if isinstance(media, (str, Path, Mapping)):
        return [media]
    return list(media)


def _media_content(kind: str, item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        content = dict(item)
        if content.get("type", kind) != kind:
            raise Qwen3VLInputError(
                f"Expected a {kind!r} content item, got {content.get('type')!r}."
            )
        content["type"] = kind
        return content

    value = str(item) if isinstance(item, Path) else item
    return {"type": kind, kind: value}


@dataclass
class Qwen3VLArchitecture:
    """Qwen3-VL model facade for constructing inputs and generating answers."""

    model: Any
    processor: Any

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> "Qwen3VLArchitecture":
        """Load a model through :func:`model.model_loader.load_qwen3_vl`."""

        from .model_loader import load_qwen3_vl

        components = load_qwen3_vl(*args, **kwargs)
        return cls(model=components.model, processor=components.processor)

    @staticmethod
    def build_messages(
        prompt: str,
        *,
        images: Any | Sequence[Any] | None = None,
        videos: Any | Sequence[Any] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build Qwen chat messages from a prompt and optional media.

        Image and video values may be file paths, URLs, PIL images, tensors, or
        pre-built Qwen content dictionaries. Local paths are handled by
        ``qwen_vl_utils.process_vision_info`` during input preparation.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise Qwen3VLInputError("prompt must be a non-empty string.")

        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise Qwen3VLInputError("system_prompt must be a non-empty string when set.")
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        content = [
            *(_media_content("image", image) for image in _as_media_list(images)),
            *(_media_content("video", video) for video in _as_media_list(videos)),
            {"type": "text", "text": prompt},
        ]
        messages.append({"role": "user", "content": content})
        return messages

    @staticmethod
    def _contains_media(messages: Sequence[Mapping[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                continue
            if any(isinstance(item, Mapping) and item.get("type") in {"image", "video"} for item in content):
                return True
        return False

    @staticmethod
    def _process_vision_info(messages: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as error:
            raise RuntimeError(
                "Multimodal Qwen3-VL inputs require qwen_vl_utils. "
                "Install the project requirements before passing images or videos."
            ) from error
        return process_vision_info(messages)

    def prepare_inputs(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
    ) -> Any:
        """Convert one Qwen conversation into a batch of model input tensors."""

        if not messages:
            raise Qwen3VLInputError("messages must contain at least one conversation turn.")

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if self._contains_media(messages):
            image_inputs, video_inputs = self._process_vision_info(messages)
            if image_inputs is not None:
                processor_kwargs["images"] = image_inputs
            if video_inputs is not None:
                processor_kwargs["videos"] = video_inputs

        inputs = self.processor(**processor_kwargs)
        # Qwen3-VL does not consume the BERT-style token type ids some processors
        # include in their batch encoding.
        inputs.pop("token_type_ids", None)
        return inputs

    def _input_device(self) -> Any | None:
        try:
            device = self.model.device
        except (AttributeError, StopIteration):
            try:
                device = next(self.model.parameters()).device
            except (AttributeError, StopIteration):
                return None
        return None if getattr(device, "type", None) == "meta" else device

    def _move_inputs(self, inputs: Any) -> Any:
        device = self._input_device()
        if device is not None and hasattr(inputs, "to"):
            return inputs.to(device)
        return inputs

    def generate_from_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int = 512,
        **generation_kwargs: Any,
    ) -> str:
        """Generate and decode one assistant response for ``messages``."""

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        inputs = self._move_inputs(self.prepare_inputs(messages))
        generation_kwargs = dict(generation_kwargs)
        presence_penalty = float(generation_kwargs.pop("presence_penalty", 0.0))
        if presence_penalty:
            generation_kwargs["logits_processor"] = [
                *generation_kwargs.get("logits_processor", []),
                _PresencePenaltyLogitsProcessor(presence_penalty, inputs["input_ids"].shape[1]),
            ]
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
        prompt_width = inputs["input_ids"].shape[1]
        generated_only = generated_ids[:, prompt_width:]
        return self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def generate(
        self,
        prompt: str,
        *,
        images: Any | Sequence[Any] | None = None,
        videos: Any | Sequence[Any] | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        **generation_kwargs: Any,
    ) -> str:
        """Generate an answer for a text, image, or video prompt."""

        messages = self.build_messages(
            prompt,
            images=images,
            videos=videos,
            system_prompt=system_prompt,
        )
        return self.generate_from_messages(
            messages,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )

    def forward(self, **model_inputs: Any) -> Any:
        """Delegate a training/evaluation forward pass to the underlying model."""

        model_inputs.pop("token_type_ids", None)
        return self.model(**model_inputs)
