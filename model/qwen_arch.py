"""A small, explicit inference interface for Qwen3-VL.

``Qwen3VLArchitecture`` owns the model together with the matching processor.
It converts conversational text/image/video inputs into the tensors expected by
Transformers and decodes only the tokens generated after the prompt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
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


# COCONUT uses a native Qwen3-VL subclass so checkpoints retain the ordinary
# Transformers model layout while the forward pass gains continuous thoughts.
import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.utils import ModelOutput


@dataclass
class CoconutQwen3VLOutput(ModelOutput):
    """Transformers-compatible output for Trainer/DDP/DataParallel gathering."""

    loss: Any | None = None
    logits: Any | None = None
    past_key_values: Any | None = None
    inputs_embeds: Any | None = None


@dataclass(frozen=True)
class CoconutLatentLayout:
    """Validated positions of COCONUT latent tokens in a batched prompt.

    Both the Transformers training model and the vLLM model plugin must split
    a prefill immediately before every latent token, then replace that token's
    embedding with the previous segment's final hidden state.  Keeping this
    validation independent of either runtime prevents the two implementations
    from silently using different COCONUT prompt layouts.
    """

    rows: tuple[tuple[int, ...], ...]
    first: int
    count: int


def coconut_latent_layout(input_ids: Any, latent_token_id: int) -> CoconutLatentLayout | None:
    """Validate and return the aligned contiguous latent-token layout.

    ``None`` means that the prompt has no latent tokens, so callers can take
    the ordinary Qwen forward path.  Every non-empty row must contain latents
    at the same positions; this is the contract enforced by
    ``CoconutDataCollator`` and required for a batched latent prefill.
    """
    if getattr(input_ids, "ndim", None) != 2:
        raise ValueError("COCONUT latent layout requires rank-2 input_ids.")
    locations = (input_ids == latent_token_id).nonzero(as_tuple=False)
    if locations.numel() == 0:
        return None
    rows = tuple(
        tuple(int(position) for row, position in locations.tolist() if row == batch)
        for batch in range(input_ids.shape[0])
    )
    if not all(rows):
        raise ValueError("COCONUT latent tokens must occur in every batch row.")
    reference = rows[0]
    if reference[0] == 0:
        raise ValueError("A COCONUT latent token cannot be the first sequence token.")
    if tuple(range(reference[0], reference[0] + len(reference))) != reference:
        raise ValueError("COCONUT latent tokens must be contiguous.")
    if any(row != reference for row in rows[1:]):
        raise ValueError("COCONUT latent tokens must be aligned across batch rows.")
    return CoconutLatentLayout(rows=rows, first=reference[0], count=len(reference))


class CoconutQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL model whose latent tokens feed back preceding hidden states."""

    latent_token_id: int | None = None
    start_latent_id: int | None = None
    end_latent_id: int | None = None

    def configure_coconut(self, *, latent_token_id: int, start_latent_id: int, end_latent_id: int) -> None:
        self.latent_token_id = int(latent_token_id)
        self.start_latent_id = int(start_latent_id)
        self.end_latent_id = int(end_latent_id)

    @contextmanager
    def _reuse_vision_features(self, extra: Mapping[str, Any]):
        """Cache one image-tower pass while preserving Qwen's normal forward.

        The base Qwen VLM forward is responsible for image-token injection and
        DeepStack feature routing. Reusing its already-computed feature tuple
        keeps that exact path intact while preventing COCONUT's latent segments
        from calling the vision encoder repeatedly.
        """
        pixel_values = extra.get("pixel_values")
        image_grid_thw = extra.get("image_grid_thw")
        if extra.get("pixel_values_videos") is not None or extra.get("video_grid_thw") is not None:
            raise ValueError("COCONUT training supports images only; video inputs cannot be reused here.")
        if pixel_values is None:
            yield
            return
        if image_grid_thw is None:
            raise ValueError("pixel_values requires image_grid_thw.")

        cached_features = self.model.get_image_features(pixel_values, image_grid_thw)
        original_get_image_features = self.model.get_image_features

        def get_cached_image_features(received_pixels: Any, received_grid: Any) -> Any:
            if received_pixels is not pixel_values or received_grid is not image_grid_thw:
                raise RuntimeError("COCONUT vision-feature cache received unexpected image inputs.")
            return cached_features

        self.model.get_image_features = get_cached_image_features
        try:
            yield
        finally:
            self.model.get_image_features = original_get_image_features

    def _forward_segment(self, embeds: Any, attention_mask: Any, position_ids: Any,
                         extra: Mapping[str, Any]) -> Any:
        """Run Qwen's standard VLM forward using the scoped vision-feature cache."""
        return super().forward(inputs_embeds=embeds, attention_mask=attention_mask,
                               position_ids=position_ids, use_cache=False,
                               output_hidden_states=True,
                               return_dict=True, **extra)

    def forward(self, input_ids: Any | None = None, attention_mask: Any | None = None,
                position_ids: Any | None = None, labels: Any | None = None,
                **kwargs: Any) -> CoconutQwen3VLOutput:
        """Run COCONUT latent feedback without a persistent KV cache.

        Each latent slot recomputes its full preceding prefix. This preserves
        causal context without retaining ``past_key_values`` between segments.
        """
        import torch
        from torch.nn import CrossEntropyLoss

        if input_ids is None:
            raise ValueError("COCONUT Qwen requires input_ids.")
        if self.latent_token_id is None:
            raise RuntimeError("Call configure_coconut() after adding special tokens.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        # Training must not inherit a caller's generation-cache request.
        forward_extra = dict(kwargs)
        forward_extra.pop("use_cache", None)
        forward_extra.pop("past_key_values", None)
        layout = coconut_latent_layout(input_ids, self.latent_token_id)
        if layout is None:
            output = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                     position_ids=position_ids, labels=labels, use_cache=False,
                                     return_dict=True, **forward_extra)
            return CoconutQwen3VLOutput(output.loss, output.logits)
        embeds = self.get_input_embeddings()(input_ids)
        # Intermediate forwards exist solely to produce the next latent state.
        # Do not retain their full-vocabulary logits: slices of a logits tensor
        # share its storage, so the previous implementation kept one complete
        # ``[batch, prefix_length, vocab_size]`` allocation per latent slot.
        # The final causal forward already yields identical logits for every
        # position because later latent embeddings cannot affect earlier tokens.
        with self._reuse_vision_features(forward_extra):
            for index in range(layout.count):
                end = layout.first + index
                output = self._forward_segment(embeds[:, :end, :], attention_mask[:, :end],
                                               position_ids[:, :end], forward_extra)
                updated = embeds.clone()
                for batch, row in enumerate(layout.rows):
                    updated[batch, row[index], :] = output.hidden_states[-1][batch, -1, :]
                embeds = updated

            output = self._forward_segment(embeds, attention_mask, position_ids, forward_extra)
        logits = output.logits
        loss = None
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(logits[:, :-1, :].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return CoconutQwen3VLOutput(loss, logits, None, embeds)

    @torch.no_grad()
    def generate(self, input_ids: Any, attention_mask: Any | None = None, max_new_tokens: int = 128,
                 **kwargs: Any) -> Any:
        """Greedy generation retaining continuous-thought feedback (batch size 1)."""
        import torch
        if input_ids.shape[0] != 1:
            raise ValueError("COCONUT generation currently supports batch_size=1.")
        if kwargs.pop("do_sample", False):
            raise NotImplementedError("Sampling is not implemented for COCONUT generation.")
        generated = input_ids
        output = self.forward(input_ids=generated, attention_mask=attention_mask, **kwargs)
        embeds = output.inputs_embeds
        for _ in range(max_new_tokens):
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if int(next_token.item()) == self.config.eos_token_id:
                break
            embeds = torch.cat([embeds, self.get_input_embeddings()(next_token)], dim=1)
            base = super().forward(inputs_embeds=embeds, use_cache=False, return_dict=True)
            output = CoconutQwen3VLOutput(None, base.logits, None, embeds)
        return generated
