"""Loading utilities for the Qwen3-VL checkpoints used by this project.

The imports from :mod:`torch` and :mod:`transformers` intentionally happen only
inside ``load_qwen3_vl``. This keeps configuration inspection and command-line
tools usable on machines where the inference stack is not installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


DEFAULT_QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN3_VL_MODEL_PATH = (
    _PROJECT_ROOT / "weights" / "Qwen" / "Qwen3-VL-4B-Instruct"
)


@dataclass(frozen=True)
class Qwen3VLComponents:
    """The model and processor loaded from one Qwen3-VL checkpoint."""

    model: Any
    processor: Any
    model_source: str


def default_qwen3_vl_source() -> str:
    """Return the checked-out model directory, falling back to its Hub id.

    ``utils/hugging_face/download.py --target Qwen/Qwen3-VL-4B-Instruct``
    creates precisely the local directory checked here.
    """

    if DEFAULT_QWEN3_VL_MODEL_PATH.is_dir():
        return str(DEFAULT_QWEN3_VL_MODEL_PATH)
    return DEFAULT_QWEN3_VL_MODEL_ID


def resolve_model_source(model_source: str | Path | None = None) -> str:
    """Resolve a local model path while preserving valid Hub repository ids."""

    if model_source is None:
        return default_qwen3_vl_source()

    source = Path(model_source).expanduser()
    if source.is_dir():
        return str(source.resolve())

    source_text = str(model_source)
    looks_like_path = source_text.startswith((".", "/", "~")) or Path(source_text).suffix
    if looks_like_path:
        raise FileNotFoundError(f"Qwen3-VL checkpoint directory does not exist: {source}")
    return source_text


def _import_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Loading Qwen3-VL requires compatible torch and transformers packages. "
            "Install the versions listed in requirements.txt."
        ) from error
    return torch, AutoProcessor, Qwen3VLForConditionalGeneration


def _resolve_torch_dtype(torch: Any, torch_dtype: str | Any | None) -> str | Any:
    if torch_dtype is None:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if not isinstance(torch_dtype, str):
        return torch_dtype
    if torch_dtype == "auto":
        return "auto"

    dtype = getattr(torch, torch_dtype, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(
            "torch_dtype must be a torch.dtype, 'auto', or a torch dtype name "
            f"such as 'bfloat16'; got {torch_dtype!r}."
        )
    return dtype


def load_qwen3_vl(
    model_source: str | Path | None = None,
    *,
    device: str | None = None,
    device_map: str | dict[str, Any] | None = None,
    torch_dtype: str | Any | None = None,
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] | None = "sdpa",
    local_files_only: bool | None = None,
    revision: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    **from_pretrained_kwargs: Any,
) -> Qwen3VLComponents:
    """Load a Qwen3-VL conditional-generation model and its processor.

    The default source is the local 4B-Instruct checkpoint if it has been
    downloaded under ``weights/``; otherwise Hugging Face resolves the official
    model id. On CUDA, ``device_map`` defaults to ``"auto"`` when no explicit
    ``device`` was requested; CPU loading uses the normal single-device
    placement.
    """

    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose at most one of load_in_4bit and load_in_8bit.")
    if device is not None and device_map is not None:
        raise ValueError("Specify either device or device_map, not both.")

    torch, AutoProcessor, model_class = _import_runtime()
    source = resolve_model_source(model_source)
    resolved_dtype = _resolve_torch_dtype(torch, torch_dtype)

    if device is None and device_map is None and torch.cuda.is_available():
        device_map = "auto"

    common_kwargs: dict[str, Any] = {}
    if local_files_only is not None:
        common_kwargs["local_files_only"] = local_files_only
    if revision is not None:
        common_kwargs["revision"] = revision

    model_kwargs: dict[str, Any] = {
        **common_kwargs,
        "torch_dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
        **from_pretrained_kwargs,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    processor = AutoProcessor.from_pretrained(source, **common_kwargs)
    model = model_class.from_pretrained(source, **model_kwargs)
    if device is not None:
        model.to(device)
    model.eval()
    return Qwen3VLComponents(model=model, processor=processor, model_source=source)


def load_model(*args: Any, **kwargs: Any) -> Qwen3VLComponents:
    """Compatibility alias for :func:`load_qwen3_vl`."""

    return load_qwen3_vl(*args, **kwargs)
