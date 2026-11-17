#!/usr/bin/env python3
"""Download a Hugging Face dataset repository into the configured Hub cache.

Examples:
    python data/download_data.py --target OpenDataArena/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking
    python data/download_data.py --target org/dataset --output-path /datasets/dataset
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root containing the project's .env file."""
    return Path(__file__).resolve().parents[1]


def hf_home() -> Path:
    """Return HF_HOME from the environment or the user's .bashrc configuration."""
    configured_path = os.environ.get("HF_HOME")
    if configured_path:
        return Path(configured_path).expanduser()

    bashrc_path = Path.home() / ".bashrc"
    if bashrc_path.is_file():
        for raw_line in bashrc_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line.startswith("export HF_HOME="):
                continue
            configured_path = line.split("=", maxsplit=1)[1].strip()
            if len(configured_path) >= 2 and configured_path[0] == configured_path[-1] and configured_path[0] in {"'", '"'}:
                configured_path = configured_path[1:-1]
            if configured_path:
                return Path(configured_path).expanduser()

    raise ValueError("Set HF_HOME in the environment or export it from ~/.bashrc before downloading.")


def load_hf_token(env_path: Path) -> str:
    """Read ``HF_TOKEN`` from a simple dotenv file without logging the value."""
    if not env_path.is_file():
        raise FileNotFoundError(f"Hugging Face token file not found: {env_path}")

    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key.strip() != "HF_TOKEN":
            continue

        token = value.strip()
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
            token = token[1:-1]
        if token:
            return token
        raise ValueError(f"HF_TOKEN is empty in {env_path} (line {line_number}).")

    raise ValueError(f"{env_path} must define a non-empty HF_TOKEN.")


def _readme_front_matter(readme_path: Path) -> dict[str, Any]:
    """Return the YAML metadata block at the start of a dataset README."""
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError("Install PyYAML to select a Hugging Face dataset configuration.") from error
            metadata = yaml.safe_load("\n".join(lines[1:index]))
            return metadata if isinstance(metadata, dict) else {}
    return {}


def _data_file_patterns(data_files: Any) -> list[str]:
    """Extract file patterns from a Hugging Face README ``data_files`` value."""
    if isinstance(data_files, str):
        return [data_files]
    if isinstance(data_files, list):
        return [pattern for item in data_files for pattern in _data_file_patterns(item)]
    if isinstance(data_files, dict):
        if "path" in data_files:
            return _data_file_patterns(data_files["path"])
        return [pattern for value in data_files.values() for pattern in _data_file_patterns(value)]
    return []


def subset_download_patterns(target: str, subset: str, token: str, cache_dir: Path) -> list[str]:
    """Return the repository patterns for a dataset config, including image assets."""
    normalized_subset = subset.strip().strip("/")
    if not normalized_subset or ".." in Path(normalized_subset).parts:
        raise ValueError("--subset must be a non-empty repository path or split name.")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Install the 'huggingface-hub' package before downloading data.") from error

    readme_path = Path(
        hf_hub_download(
            repo_id=target,
            filename="README.md",
            repo_type="dataset",
            token=token,
            cache_dir=cache_dir,
        )
    )
    configs = _readme_front_matter(readme_path).get("configs", [])
    if isinstance(configs, list):
        config_names = [config.get("config_name") for config in configs if isinstance(config, dict)]
        for config in configs:
            if not isinstance(config, dict) or config.get("config_name") != normalized_subset:
                continue
            patterns = _data_file_patterns(config.get("data_files"))
            if patterns:
                # Image-backed datasets often keep images outside the parquet
                # config files (for example VisualWebInstruct's images.zip).
                # Include conventional image archives/directories so the
                # selected subset remains usable after download.
                return [
                    ".gitattributes",
                    "README.md",
                    *patterns,
                    "images.zip",
                    "images/**",
                ]
            break
        if config_names:
            available_configs = ", ".join(repr(name) for name in config_names)
            raise ValueError(f"Unknown --subset {normalized_subset!r}. Available configurations: {available_configs}.")

    # Dataset repositories commonly store a subset either in a top-level or
    # data subdirectory, or as sharded files such as data/train-00000.parquet.
    return [
        ".gitattributes",
        "README.md",
        normalized_subset,
        f"{normalized_subset}/**",
        f"data/{normalized_subset}",
        f"data/{normalized_subset}/**",
        f"data/{normalized_subset}-*",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset repository into the configured Hugging Face cache."
    )
    parser.add_argument(
        "--target",
        required=True,
        metavar="HF_REPO_NAME",
        help="Hugging Face dataset repository, for example org/dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Directory to receive the dataset. Defaults to the HF_HOME Hub cache configured in ~/.bashrc.",
    )
    parser.add_argument(
        "--subset",
        help="Subset directory or split name to download, for example train or en."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target.strip()
    if not target or target.strip("/") == "":
        raise ValueError("--target must be a non-empty Hugging Face repository name.")

    token = load_hf_token(project_root() / ".env")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install the 'huggingface-hub' package before downloading data.") from error

    download_kwargs: dict[str, Any] = {
        "repo_id": target,
        "repo_type": "dataset",
        "token": token,
    }
    cache_dir = hf_home() / "hub"
    if args.subset is not None:
        download_kwargs["allow_patterns"] = subset_download_patterns(target, args.subset, token, cache_dir)
    if args.output_path is not None:
        output_path = args.output_path.expanduser().resolve()
        download_kwargs["local_dir"] = output_path
        print(f"Downloading {target}{f' subset {args.subset!r}' if args.subset else ''} to {output_path}")
    else:
        download_kwargs["cache_dir"] = cache_dir
        print(f"Downloading {target}{f' subset {args.subset!r}' if args.subset else ''} to Hugging Face cache: {cache_dir}")

    downloaded_path = snapshot_download(**download_kwargs)
    print(f"Download complete: {downloaded_path}")


if __name__ == "__main__":
    main()
