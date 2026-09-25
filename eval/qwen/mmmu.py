#!/usr/bin/env python3
"""Run MMMU validation with the project's custom Qwen3-VL lmms-eval adapter.

The worker mode imports eval.model before lmms-eval starts. This registers
ProjectQwen3VL in the process that runs lmms-eval, avoiding changes to the
third-party lmms-eval installation.
"""

from __future__ import annotations

import argparse
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DEFAULT_LOADER = "model.model_loader:load_qwen3_vl"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "qwen.yaml"

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
    "auto": "auto",
}


def _load_hf_token_from_dotenv() -> bool:
    """Load ``HF_TOKEN`` from the repository .env without overriding the shell."""

    if os.environ.get("HF_TOKEN"):
        return False

    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.is_file():
        return False

    with dotenv_path.open(encoding="utf-8") as dotenv_file:
        for raw_line in dotenv_file:
            line = raw_line.strip()
            if line.startswith("export "):
                line = line.removeprefix("export ").lstrip()
            key, separator, raw_value = line.partition("=")
            if separator != "=" or key.strip() != "HF_TOKEN":
                continue

            try:
                values = shlex.split(raw_value, comments=True)
            except ValueError as error:
                raise ValueError(f"Invalid HF_TOKEN entry in {dotenv_path}.") from error
            if not values:
                return False
            os.environ["HF_TOKEN"] = values[0]
            return True
    return False


def _config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return section


def _load_config(path: Path) -> dict[str, Any]:
    """Load MMMU settings from a YAML config file as argparse defaults."""

    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("--config requires PyYAML. Install the project requirements first.") from error

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, Mapping):
        raise ValueError("The top level of --config must be a mapping.")

    model = _config_section(config, "model")
    evaluation = _config_section(config, "evaluation")
    generation = _config_section(config, "generation")
    image = _config_section(config, "image")
    return {
        "model_path": model.get("path"),
        "model_revision": model.get("revision", DEFAULT_MODEL_REVISION),
        "model_loader": model.get("loader", DEFAULT_LOADER),
        "device": model.get("device", "cuda:0"),
        "device_map": model.get("device_map"),
        "dtype": model.get("dtype", "bf16"),
        "batch_size": evaluation.get("batch_size", 1),
        "num_processes": evaluation.get("num_processes", 1),
        "main_process_port": evaluation.get("main_process_port", 12346),
        "do_sample": generation.get("do_sample", True),
        "temperature": generation.get("temperature", 0.7),
        "top_p": generation.get("top_p", 0.8),
        "top_k": generation.get("top_k", 20),
        "repetition_penalty": generation.get("repetition_penalty", 1.0),
        "presence_penalty": generation.get("presence_penalty", 1.5),
        "seed": generation.get("seed", 3407),
        "max_new_tokens": generation.get("max_new_tokens", 32768),
        "min_pixels": image.get("min_pixels"),
        "max_pixels": image.get("max_pixels"),
    }


def _run_lmms_worker() -> None:
    """Register the local model class, then dispatch lmms-eval in this process."""

    _load_hf_token_from_dotenv()
    sys.path.insert(0, str(REPO_ROOT))
    import eval.model  # noqa: F401  # registration side effect

    sys.argv = [sys.argv[0], *sys.argv[2:]]
    runpy.run_module("lmms_eval", run_name="__main__")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    config_path = config_args.config.expanduser()
    config_defaults = _load_config(config_path)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help=f"YAML configuration file (default: {DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--model-path",
        default=config_defaults["model_path"],
        help=(
            "Model Hub id or local checkpoint directory. When omitted, uses "
            f"{DEFAULT_MODEL} at revision {DEFAULT_MODEL_REVISION}."
        ),
    )
    parser.add_argument("--model-revision", default=config_defaults["model_revision"])
    parser.add_argument("--model-loader", default=config_defaults["model_loader"])
    parser.add_argument("--loader-arg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--device", default=config_defaults["device"], help="Single-device target (default: cuda:0).")
    parser.add_argument(
        "--dtype",
        default=config_defaults["dtype"],
        choices=sorted(_DTYPE_ALIASES),
        help="Model dtype (default: bf16).",
    )
    parser.add_argument("--batch-size", type=int, default=config_defaults["batch_size"], help="Evaluation batch size (default: 1).")
    parser.add_argument("--num-processes", type=int, default=config_defaults["num_processes"])
    parser.add_argument("--main-process-port", type=int, default=config_defaults["main_process_port"])
    parser.add_argument(
        "--device-map",
        default=config_defaults["device_map"],
        help="Optional Transformers device map (for example: auto). Overrides --device.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Deprecated alias for --dtype; when supplied, takes precedence.",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=config_defaults["max_new_tokens"])
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=config_defaults["do_sample"],
        help="Enable sampling; use --no-do-sample for greedy decoding.",
    )
    parser.add_argument("--temperature", type=float, default=config_defaults["temperature"])
    parser.add_argument("--top-p", type=float, default=config_defaults["top_p"])
    parser.add_argument("--top-k", type=int, default=config_defaults["top_k"])
    parser.add_argument("--repetition-penalty", type=float, default=config_defaults["repetition_penalty"])
    parser.add_argument("--presence-penalty", type=float, default=config_defaults["presence_penalty"])
    parser.add_argument("--seed", type=int, default=config_defaults["seed"])
    parser.add_argument("--min-pixels", type=int, default=config_defaults["min_pixels"])
    parser.add_argument("--max-pixels", type=int, default=config_defaults["max_pixels"])
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "results" / "lmms_eval" / "qwen3-vl-4b-instruct_mmmu_val",
    )
    return parser.parse_known_args()


def _model_args(args: argparse.Namespace) -> str:
    model_path = args.model_path or DEFAULT_MODEL
    torch_dtype = args.torch_dtype or _DTYPE_ALIASES[args.dtype]
    values = [
        f"pretrained={model_path}",
        f"model_loader={args.model_loader}",
        f"attn_implementation={args.attn_implementation}",
        f"max_new_tokens={args.max_new_tokens}",
        f"torch_dtype={torch_dtype}",
        f"do_sample={args.do_sample}",
        f"temperature={args.temperature}",
        f"top_p={args.top_p}",
        f"top_k={args.top_k}",
        f"repetition_penalty={args.repetition_penalty}",
        f"presence_penalty={args.presence_penalty}",
        f"seed={args.seed}",
    ]
    if not args.model_path or args.model_path == DEFAULT_MODEL:
        values.append(f"revision={args.model_revision}")
    if args.device_map:
        values.append(f"device_map={args.device_map}")
    else:
        values.append(f"device={args.device}")
    if args.min_pixels is not None:
        values.append(f"min_pixels={args.min_pixels}")
    if args.max_pixels is not None:
        values.append(f"max_pixels={args.max_pixels}")
    for value in args.loader_arg:
        if "=" not in value or value.startswith("="):
            raise ValueError(f"--loader-arg must be KEY=VALUE; got {value!r}.")
        values.append(value)
    return ",".join(values)


def main() -> int:
    args, extra_lmms_args = _parse_args()
    loaded_hf_token = _load_hf_token_from_dotenv()
    if args.batch_size < 1 or args.num_processes < 1:
        raise ValueError("--batch-size and --num-processes must be positive.")
    if not args.device:
        raise ValueError("--device must be non-empty.")
    if args.min_pixels is not None and args.min_pixels <= 0:
        raise ValueError("--min-pixels must be positive.")
    if args.max_pixels is not None and args.max_pixels <= 0:
        raise ValueError("--max-pixels must be positive.")
    if args.min_pixels and args.max_pixels and args.min_pixels > args.max_pixels:
        raise ValueError("--min-pixels cannot exceed --max-pixels.")
    if args.num_processes != 1:
        raise ValueError("This adapter supports one process; use --device-map auto for model parallelism.")

    accelerator = shutil.which("accelerate")
    if accelerator is None:
        raise RuntimeError("accelerate is required. Install the project requirements first.")

    if loaded_hf_token:
        print("Loaded HF_TOKEN from .env for Hugging Face downloads.")

    args.output_path.mkdir(parents=True, exist_ok=True)
    lmms_args = [
        "--model",
        "project_qwen3_vl",
        "--model_args",
        _model_args(args),
        "--tasks",
        "mmmu_val",
        "--batch_size",
        str(args.batch_size),
        "--log_samples",
        "--log_samples_suffix",
        "project-qwen3-vl-mmmu-val",
        "--output_path",
        str(args.output_path),
    ]
    command = [
        accelerator,
        "launch",
        "--num_processes",
        str(args.num_processes),
        "--main_process_port",
        str(args.main_process_port),
        str(Path(__file__).resolve()),
        "--_worker",
        *lmms_args,
        *extra_lmms_args,
    ]
    print("Running:", " ".join(command))
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        _run_lmms_worker()
    else:
        raise SystemExit(main())
