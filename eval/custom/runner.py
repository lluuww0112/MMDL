"""vLLM 0.11 runner support for COCONUT segmented prefills.

vLLM normally invokes a multimodal model once with one flattened prompt. A
COCONUT prompt must instead be invoked once before each latent token, with the
final hidden state of the preceding invocation used as that latent token's
embedding. This module adds the small amount of runner plumbing required by
the model implementation in :mod:`eval.custom.model`.

The runner supports normal vLLM scheduling, including chunked prefills. It
supplies absolute prompt positions and an identity shared with the model's
latent-state sidecar.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import torch

from model.qwen_arch import coconut_latent_layout

try:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker
except ImportError as error:  # pragma: no cover - exercised in a vLLM runtime.
    raise RuntimeError("eval.custom.runner requires vLLM 0.11.") from error


class CoconutGPUModelRunner(GPUModelRunner):
    """Attach COCONUT layout and prefix identity to the model forward."""

    @staticmethod
    def _coconut_cache_key(prompt_token_ids: list[int], mm_features: list[Any]) -> str:
        """Match the prompt identity used by a reusable COCONUT KV prefix."""
        digest = hashlib.sha256()
        digest.update(b"coconut-v1\0")
        for token_id in prompt_token_ids:
            digest.update(f"{int(token_id)},".encode())
        digest.update(b"\0")
        for feature in mm_features:
            digest.update(str(feature.identifier).encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def _preprocess(
        self,
        scheduler_output: Any,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        ubatch_slices: Any = None,
        num_tokens_after_padding: Any = None,
    ) -> tuple[Any, ...]:
        result = super()._preprocess(
            scheduler_output,
            intermediate_tensors=intermediate_tensors,
            ubatch_slices=ubatch_slices,
            num_tokens_after_padding=num_tokens_after_padding,
        )
        (num_scheduled_tokens, num_input_tokens, num_tokens_across_dp,
         input_ids, inputs_embeds, positions, intermediate_tensors,
         model_kwargs) = result

        # vLLM flattens each active request into this input. Preserve every
        # request's query span so the model can temporarily install the
        # corresponding one-row attention metadata while it materializes the
        # latent sequence. This allows subject-level `generate(list)` calls.
        latent_id = getattr(self.model, "coconut_latent_token_id", None)
        if latent_id is not None and num_scheduled_tokens:
            requests = []
            input_start = 0
            has_coconut_prefill = False
            for request_index, req_id in enumerate(self.input_batch.req_ids):
                assert req_id is not None
                request = self.requests[req_id]
                scheduled = int(scheduler_output.num_scheduled_tokens[req_id])
                input_end = input_start + scheduled
                prefill_start = int(self.input_batch.num_computed_tokens_cpu[request_index])
                descriptor: dict[str, Any] = {
                    "request_index": request_index,
                    "input_start": input_start,
                    "input_end": input_end,
                    "prefill_start": prefill_start,
                }
                prompt_token_ids = request.prompt_token_ids
                if prompt_token_ids is None:
                    raise RuntimeError("COCONUT vLLM prefill requires prompt token IDs.")
                layout = coconut_latent_layout(
                    torch.tensor(prompt_token_ids, dtype=torch.long).unsqueeze(0), int(latent_id))
                prefill_end = prefill_start + scheduled
                # Include a chunk ending exactly at the first latent: the
                # model stores the preceding state for the next scheduler
                # chunk. Chunks outside the latent run use normal forward.
                if (layout is not None and prefill_start < layout.first + layout.count
                        and prefill_end >= layout.first):
                    descriptor.update({
                        "first_latent": layout.first,
                        "latent_count": layout.count,
                        "cache_key": self._coconut_cache_key(prompt_token_ids, request.mm_features),
                    })
                    has_coconut_prefill = True
                requests.append(descriptor)
                input_start = input_end
            if has_coconut_prefill:
                model_kwargs = dict(model_kwargs)
                model_kwargs["coconut_batch"] = requests

        return (
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_across_dp,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
        )


class CoconutGPUWorker(Worker):
    """Use :class:`CoconutGPUModelRunner` after vLLM selects its CUDA device."""

    def init_device(self) -> None:
        # GPUWorker creates its runner only after selecting ``cuda:<rank>``.
        # Replacing it here, rather than in __init__, keeps the device valid.
        super().init_device()
        self.model_runner = CoconutGPUModelRunner(self.vllm_config, self.device)
