"""lmms-eval adapter for project-specific Qwen3-VL checkpoints.

This adapter loads through an importable factory rather than calling
transformers.from_pretrained directly. Fine-tuned checkpoints can use the
default model.model_loader:load_qwen3_vl factory. A future soft-prompt or
architecture change only needs to provide a factory returning an object with
model and processor fields, or a (model, processor) tuple.
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm

from model.conversation import Conversation, get_conversation
from model.qwen_arch import Qwen3VLArchitecture


DEFAULT_MODEL_LOADER = "model.model_loader:load_qwen3_vl"


def _import_symbol(spec: str) -> Callable[..., Any]:
    """Resolve a loader written as package.module:callable."""

    module_name, separator, symbol_name = spec.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError(
            "model_loader must use the form 'package.module:callable'; "
            f"got {spec!r}."
        )

    loader = getattr(importlib.import_module(module_name), symbol_name, None)
    if not callable(loader):
        raise TypeError(f"model_loader {spec!r} does not resolve to a callable.")
    return loader


def _unpack_components(components: Any) -> tuple[Any, Any]:
    """Accept the project's component dataclass or a model/processor pair."""

    model = getattr(components, "model", None)
    processor = getattr(components, "processor", None)
    if model is None or processor is None:
        if isinstance(components, tuple) and len(components) == 2:
            model, processor = components
        else:
            raise TypeError(
                "A Qwen3-VL model loader must return Qwen3VLComponents or "
                "a (model, processor) tuple."
            )
    return model, processor


def _normalise_visuals(visuals: Any) -> list[Any]:
    if visuals is None:
        return []
    if isinstance(visuals, (str, bytes, Mapping)):
        return [visuals]
    if isinstance(visuals, Sequence):
        return list(visuals)
    return [visuals]


def _apply_image_pixel_limits(
    visuals: Sequence[Any], min_pixels: int | None, max_pixels: int | None
) -> list[Any]:
    """Attach Qwen image-resolution limits without changing task media loaders."""

    if min_pixels is None and max_pixels is None:
        return list(visuals)

    limited_visuals: list[Any] = []
    for visual in visuals:
        if isinstance(visual, Mapping):
            content = dict(visual)
        else:
            content = {"type": "image", "image": visual}
        if min_pixels is not None:
            content["min_pixels"] = min_pixels
        if max_pixels is not None:
            content["max_pixels"] = max_pixels
        limited_visuals.append(content)
    return limited_visuals


@register_model("project_qwen3_vl")
class ProjectQwen3VL(lmms):
    """Qwen3-VL lmms-eval interface with a replaceable model factory.

    This simple adapter supports image MMMU while keeping model loading
    independent from lmms-eval's built-in Qwen wrapper. To evaluate a modified
    architecture, provide model_loader=my_package.loader:load_soft_prompt_qwen3_vl
    through lmms-eval's model arguments.
    """

    is_simple = True

    def __init__(
        self,
        pretrained: str | None = None,
        *,
        model_loader: str = DEFAULT_MODEL_LOADER,
        batch_size: int | str = 1,
        device: str | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        attn_implementation: str | None = "sdpa",
        conversation: str = "qwen3_vl_mmmu",
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        seed: int | None = 3407,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **loader_kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size_per_gpu = int(batch_size)
        self.conversation: Conversation = get_conversation(conversation)
        self.max_new_tokens = int(max_new_tokens)
        self.default_generation_kwargs = {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
        }
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        if seed is not None:
            random.seed(seed)
            try:
                import torch

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass

        loader = _import_symbol(model_loader)
        components = loader(
            pretrained,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **loader_kwargs,
        )
        model, processor = _unpack_components(components)
        self.architecture = Qwen3VLArchitecture(model=model, processor=processor)

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        raise NotImplementedError("ProjectQwen3VL supports generate_until tasks only.")

    def generate_until(self, requests: list[Any]) -> list[str]:
        responses: list[str] = []
        progress = tqdm(
            total=len(requests),
            desc="Model Responding",
            unit="sample",
            dynamic_ncols=True,
        )
        try:
            for request in requests:
                context, generation_kwargs, doc_to_visual, doc_id, task, split = request.args
                document = self.task_dict[task][split][doc_id]
                visuals = _apply_image_pixel_limits(
                    _normalise_visuals(doc_to_visual(document)), self.min_pixels, self.max_pixels
                )

                generation_kwargs = dict(generation_kwargs)
                generation_kwargs.pop("until", None)
                # The project recipe is the source of truth.  In particular,
                # MMMU's lmms-eval task carries a 128-token task default, which
                # must not override the requested Qwen3-VL generation limit.
                generation_kwargs.pop("max_new_tokens", None)
                generation_kwargs.pop("max_gen_toks", None)
                max_new_tokens = self.max_new_tokens
                # The config is the evaluation recipe source of truth, rather
                # than task defaults supplied by lmms-eval.
                generation_kwargs.update(self.default_generation_kwargs)
                prompt = self.conversation.render(document, fallback=context)
                messages = self.architecture.build_messages(
                    prompt,
                    images=visuals,
                    system_prompt=self.conversation.system_prompt,
                )
                response = self.architecture.generate_from_messages(
                    messages,
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs,
                )
                self.cache_hook.add_partial("generate_until", (prompt, generation_kwargs), response)
                responses.append(response)
                progress.update(1)
        finally:
            progress.close()
        return responses

    def generate_until_multi_round(self, requests: list[Any]) -> list[str]:
        raise NotImplementedError("ProjectQwen3VL does not implement multi-round tasks.")

def _register_model_manifest() -> None:
    """Expose the adapter to lmms-eval 0.7+ model registry V2."""

    try:
        from lmms_eval.models import MODEL_REGISTRY_V2
        from lmms_eval.models.registry_v2 import ModelManifest
    except ImportError:
        return

    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(
            model_id="project_qwen3_vl",
            simple_class_path=f"{__name__}.ProjectQwen3VL",
        )
    )


_register_model_manifest()
