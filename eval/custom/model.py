"""vLLM Qwen3-VL model with COCONUT latent-feedback prefill support."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, replace
from typing import Any, Optional, Union

import torch

try:
    from vllm import ModelRegistry
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration as _VLLMQwen3VL
    from vllm.sequence import IntermediateTensors
except ImportError as error:  # pragma: no cover
    raise RuntimeError("eval.custom.model requires vLLM 0.11.") from error


def coconut_hf_override(config: Any, *, vocab_size: int | None = None) -> Any:
    """Select the custom model class and, when needed, marker-token rows."""
    config.architectures = ["CoconutQwen3VLForConditionalGeneration"]
    if vocab_size is not None:
        text_config = getattr(config, "text_config", config)
        text_config.vocab_size = int(vocab_size)
        if hasattr(config, "vocab_size"):
            config.vocab_size = int(vocab_size)
    return config


class CoconutQwen3VLForConditionalGeneration(_VLLMQwen3VL):
    """Qwen3-VL that feeds the previous hidden state into each latent token."""

    def __init__(self, *, vllm_config: Any, prefix: str = "model") -> None:
        """Keep the engine config for nested COCONUT forward contexts."""
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        # A paged KV-cache stores K/V tensors but not the final hidden state
        # needed to seed the next COCONUT latent. Keep that small, immutable
        # sidecar per vLLM prefix identity so a cache hit may begin at a latent
        # boundary. The runner key includes token IDs and multimodal hashes.
        self._coconut_prefix_states: OrderedDict[str, list[Optional[torch.Tensor]]] = OrderedDict()
        self._coconut_prefix_state_limit = 256

    def _remember_latent_state(self, cache_key: str, index: int, state: torch.Tensor) -> None:
        """Record the hidden state immediately preceding latent ``index``."""
        states = self._coconut_prefix_states.get(cache_key)
        if states is None:
            states = []
            self._coconut_prefix_states[cache_key] = states
        while len(states) <= index:
            states.append(None)
        states[index] = state.detach().clone()
        self._coconut_prefix_states.move_to_end(cache_key)
        while len(self._coconut_prefix_states) > self._coconut_prefix_state_limit:
            self._coconut_prefix_states.popitem(last=False)

    def _cached_latent_state(self, cache_key: str, index: int, *, device: torch.device) -> torch.Tensor:
        """Get the state that a prefix-cache hit omitted from this forward."""
        states = self._coconut_prefix_states.get(cache_key)
        if states is None or index >= len(states) or states[index] is None:
            raise RuntimeError(
                "COCONUT prefix-cache hit is missing its latent state sidecar. "
                "Restart the request without a cache hit or keep this engine process alive."
            )
        self._coconut_prefix_states.move_to_end(cache_key)
        state = states[index]
        assert state is not None
        return state if state.device == device else state.to(device=device, non_blocking=True)

    @staticmethod
    def _segment_attention_metadata(
        metadata: Any,
        *,
        start: int,
        end: int,
        sequence_start: int,
        slot_start: int = 0,
        request_index: int = 0,
    ) -> Any:
        """Build one-request metadata without changing its original KV slots."""
        field_names = {field.name for field in fields(metadata)}
        length = end - start
        sequence_end = sequence_start + end
        device = metadata.slot_mapping.device
        updates: dict[str, Any] = {
            "num_actual_tokens": length,
            "max_query_len": length,
            "query_start_loc": torch.tensor([0, length], dtype=torch.int32, device=device),
            "max_seq_len": sequence_end,
            "seq_lens": torch.tensor([sequence_end], dtype=torch.int32, device=device),
            "block_table": metadata.block_table[request_index:request_index + 1],
            "slot_mapping": metadata.slot_mapping[slot_start + start:slot_start + end],
        }
        # Segmented prefills have distinct queries, so cascade metadata from
        # the outer batch cannot be reused here.
        for name, value in {
            "use_cascade": False,
            "common_prefix_len": 0,
            "cu_prefix_query_lens": None,
            "prefix_kv_lens": None,
            "suffix_kv_lens": None,
            "scheduler_metadata": None,
            "prefix_scheduler_metadata": None,
        }.items():
            if name in field_names:
                updates[name] = value
        return replace(metadata, **{name: value for name, value in updates.items() if name in field_names})

    def _coconut_prefill(
        self,
        *,
        first_latent: int,
        latent_count: int,
        prefill_start: int,
        cache_key: str,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context, set_forward_context

        context = get_forward_context()
        if not isinstance(context.attn_metadata, dict):
            raise NotImplementedError("COCONUT vLLM prefill does not support microbatched attention metadata.")
        if inputs_embeds.size(0) != positions.shape[-1]:
            raise RuntimeError("COCONUT prefill received padded embeddings; use --enforce-eager.")

        prefill_end = prefill_start + inputs_embeds.size(0)
        latent_end = first_latent + latent_count
        if prefill_start > first_latent and prefill_start >= latent_end:
            # Every latent token is already represented in the prefix KV.
            return super().forward(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        # vLLM has encoded image/video exactly once before model.forward().
        # Preserve Qwen3-VL deepstack inputs and feed only matching slices.
        deepstack = None
        if self.use_deepstack and self.deepstack_input_embeds is not None:
            deepstack = torch.stack(
                [value[:inputs_embeds.size(0)].clone() for value in self.deepstack_input_embeds], dim=0)

        chunks: list[torch.Tensor] = []
        embeds = inputs_embeds.clone()

        def run_segment(segment_start: int, segment_end: int) -> torch.Tensor:
            if segment_end <= segment_start:
                raise RuntimeError("Invalid empty COCONUT prefill segment.")
            if deepstack is not None:
                self._set_deepstack_input_embeds(deepstack[:, segment_start:segment_end])
            metadata = {
                name: self._segment_attention_metadata(
                    value, start=segment_start, end=segment_end, sequence_start=prefill_start)
                for name, value in context.attn_metadata.items()
            }
            with set_forward_context(metadata, self.vllm_config,
                                     num_tokens=segment_end - segment_start):
                hidden = _VLLMQwen3VL.forward(self,
                    input_ids=input_ids[segment_start:segment_end] if input_ids is not None else None,
                    positions=positions[..., segment_start:segment_end],
                    inputs_embeds=embeds[segment_start:segment_end],
                    **kwargs,
                )
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError("COCONUT vLLM prefill requires pipeline_parallel_size=1.")
            return hidden

        if prefill_start < first_latent:
            first_local = min(first_latent - prefill_start, inputs_embeds.size(0))
            hidden = run_segment(0, first_local)
            chunks.append(hidden)
            # A chunk may terminate directly before the first latent. Its
            # final hidden state is then consumed by the next scheduler chunk.
            self._remember_latent_state(cache_key, 0, hidden[-1])
            if first_local < inputs_embeds.size(0):
                embeds[first_local] = hidden[-1]
            start = first_local
            latent_index = 0
        else:
            latent_index = prefill_start - first_latent
            start = 0
            embeds[0] = self._cached_latent_state(
                cache_key, latent_index, device=embeds.device)

        local_latent_count = min(latent_count, prefill_end - first_latent) - latent_index
        for latent_index in range(latent_index, latent_index + max(local_latent_count, 0)):
            hidden = run_segment(start, start + 1)
            chunks.append(hidden)
            self._remember_latent_state(cache_key, latent_index + 1, hidden[-1])
            start += 1
            if latent_index + 1 < latent_count:
                embeds[start] = hidden[-1]

        if start < inputs_embeds.size(0):
            hidden = run_segment(start, inputs_embeds.size(0))
            chunks.append(hidden)
        return torch.cat(chunks, dim=0)

    def _coconut_batched_prefill(
        self,
        *,
        requests: list[dict[str, Any]],
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Run each active request with its own attention/KV-cache mapping.

        vLLM flattens a scheduler batch, while COCONUT has a data dependency
        within each request's latent run. Requests remain independent, so
        execute their segments separately and concatenate in vLLM's original
        flattened order for sampling.
        """
        from vllm.forward_context import get_forward_context, set_forward_context

        context = get_forward_context()
        if not isinstance(context.attn_metadata, dict):
            raise NotImplementedError("COCONUT batched prefill does not support microbatches.")
        if inputs_embeds.size(0) != positions.shape[-1]:
            raise RuntimeError("COCONUT batched prefill received padded embeddings.")
        deepstack = None
        if self.use_deepstack and self.deepstack_input_embeds is not None:
            deepstack = torch.stack(
                [value[:inputs_embeds.size(0)].clone() for value in self.deepstack_input_embeds], dim=0)

        outputs: list[torch.Tensor] = []
        for descriptor in requests:
            start, end = int(descriptor["input_start"]), int(descriptor["input_end"])
            if end <= start:
                continue
            request_index = int(descriptor["request_index"])
            prefill_start = int(descriptor["prefill_start"])
            metadata = {
                name: self._segment_attention_metadata(
                    value,
                    start=0,
                    end=end - start,
                    sequence_start=prefill_start,
                    slot_start=start,
                    request_index=request_index,
                )
                for name, value in context.attn_metadata.items()
            }
            if deepstack is not None:
                self._set_deepstack_input_embeds(deepstack[:, start:end])
            request_input_ids = input_ids[start:end] if input_ids is not None else None
            request_positions = positions[..., start:end]
            request_embeds = inputs_embeds[start:end]
            with set_forward_context(metadata, self.vllm_config, num_tokens=end - start):
                if "first_latent" in descriptor:
                    hidden = self._coconut_prefill(
                        first_latent=int(descriptor["first_latent"]),
                        latent_count=int(descriptor["latent_count"]),
                        prefill_start=prefill_start,
                        cache_key=str(descriptor["cache_key"]),
                        input_ids=request_input_ids,
                        positions=request_positions,
                        inputs_embeds=request_embeds,
                        kwargs=kwargs,
                    )
                else:
                    hidden = _VLLMQwen3VL.forward(
                        self,
                        input_ids=request_input_ids,
                        positions=request_positions,
                        inputs_embeds=request_embeds,
                        **kwargs,
                    )
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError("COCONUT vLLM prefill requires pipeline_parallel_size=1.")
            outputs.append(hidden)
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        coconut_batch = kwargs.pop("coconut_batch", None)
        if coconut_batch is not None:
            if inputs_embeds is None:
                raise RuntimeError("COCONUT vLLM prefill requires precomputed Qwen3-VL embeddings.")
            if intermediate_tensors is not None:
                raise NotImplementedError("COCONUT vLLM prefill currently requires pipeline_parallel_size=1.")
            return self._coconut_batched_prefill(
                requests=list(coconut_batch),
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                kwargs=kwargs,
            )
        prefill_start = kwargs.pop("coconut_prefill_start", None)
        first_latent = kwargs.pop("coconut_first_latent", None)
        latent_count = kwargs.pop("coconut_latent_count", None)
        cache_key = kwargs.pop("coconut_cache_key", None)
        coconut_values = (prefill_start, first_latent, latent_count, cache_key)
        if any(value is not None for value in coconut_values):
            if any(value is None for value in coconut_values):
                raise ValueError("COCONUT vLLM prefill received incomplete prefix-cache metadata.")
            if inputs_embeds is None:
                raise RuntimeError("COCONUT vLLM prefill requires precomputed Qwen3-VL embeddings.")
            if intermediate_tensors is not None:
                raise NotImplementedError("COCONUT vLLM prefill currently requires pipeline_parallel_size=1.")
            return self._coconut_prefill(
                first_latent=int(first_latent),
                latent_count=int(latent_count),
                prefill_start=int(prefill_start),
                cache_key=str(cache_key),
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                kwargs=kwargs,
            )
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


def register() -> None:
    """Register the architecture and target checkpoint loader."""
    from eval.custom.loader import register as register_loader

    register_loader()
    ModelRegistry.register_model(
        "CoconutQwen3VLForConditionalGeneration",
        "eval.custom.model:CoconutQwen3VLForConditionalGeneration",
    )
