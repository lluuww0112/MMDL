"""Text-only long-CoT SFT for Qwen3-VL.

No vision tensors or image tokens are constructed. Checkpoints intentionally use
the project stage-0 KRAdapter format, so eval/custom/mmmu.py loads them without
any modification.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

# ``python train/text_only.py`` puts only ``train/`` on sys.path, whereas
# ``python -m train.text_only`` already has the repository root. Support both.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

try:
    from .checkpoint import CheckpointManager
    from .coconut import (CoconutCheckpointCallback, _configure_kradapter,
                          _enable_input_require_grads, _freeze_vision_encoder,
                          _load_state, _resolve_checkpoint_path, _section, _seed)
except ImportError:
    from checkpoint import CheckpointManager
    from coconut import (CoconutCheckpointCallback, _configure_kradapter,
                         _enable_input_require_grads, _freeze_vision_encoder,
                         _load_state, _resolve_checkpoint_path, _section, _seed)
from model.model_loader import load_qwen3_vl

IGNORE_INDEX = -100


class TextOnlyDataset(Dataset[dict[str, list[int]]]):
    def __init__(self, examples: Sequence[dict[str, list[int]]]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class TextOnlyDataCollator:
    """Right-pad only text fields; image/video tensors cannot enter this batch."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        width = max(len(item["input_ids"]) for item in features)
        padding = {"input_ids": self.pad_token_id, "attention_mask": 0,
                   "position_ids": 0, "labels": IGNORE_INDEX}
        return {
            key: torch.tensor(
                [list(item[key]) + [fill] * (width - len(item[key])) for item in features],
                dtype=torch.long,
            )
            for key, fill in padding.items()
        }


class TextOnlyTrainer(Trainer):
    """Keep Trainer's internal bookkeeping kwargs out of Qwen's forward call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                     num_items_in_batch: torch.Tensor | int | None = None) -> Any:
        outputs = model(**inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise ValueError("Qwen3-VL did not return a training loss.")
        return (loss, outputs) if return_outputs else loss


def _parquet_files(path: str | Path) -> list[str]:
    source = Path(path)
    if source.is_file() and source.suffix == ".parquet":
        return [str(source)]
    if source.is_dir():
        files = sorted(source.rglob("*.parquet"))
        if files:
            return [str(file) for file in files]
    raise FileNotFoundError(f"Expected a parquet file or directory containing parquet files: {source}")


def _messages(value: Any, index: int) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"Example {index} needs prompt and assistant messages.")
    result: list[dict[str, str]] = []
    for message_index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TypeError(f"Example {index} message {message_index} must be a mapping.")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not role or not isinstance(content, str):
            raise TypeError(f"Example {index} message {message_index} must have textual role/content.")
        result.append({"role": role, "content": content})
    if result[-1]["role"] != "assistant":
        raise ValueError(f"Example {index} must end with an assistant response.")
    return result


def _tokenize(processor: Any, messages: list[dict[str, str]], index: int) -> dict[str, list[int]]:
    tokenizer = processor.tokenizer
    prompt_text = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Example {index} chat-template prefix mismatch.")
    target = full_ids[len(prompt_ids):]
    if not target:
        raise ValueError(f"Example {index} has no assistant target tokens.")
    input_ids = prompt_ids + target
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "position_ids": list(range(len(input_ids))),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target,
    }


def _metadata_token_count(row: Mapping[str, Any], index: int) -> int | None:
    """Return an optional dataset-provided target-token count.

    The s1K parquet exports ``num_tokens`` for its long-CoT assistant target.
    It lets a length cap reject expensive examples before running the tokenizer.
    Other parquet sources remain supported by returning ``None`` when the column
    is absent. The fully tokenized labels remain the final length check below.
    """
    value = row.get("num_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"Example {index} num_tokens must be a positive integer when present.")
    return value


def load_text_only_dataset(path: str | Path, processor: Any, *, max_samples: int | None,
                           seed: int, min_len: int | None = None,
                           max_len: int | None = None) -> TextOnlyDataset:
    for name, value in (("max_samples", max_samples), ("min_len", min_len), ("max_len", max_len)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"data.{name} must be a positive integer when set.")
    if min_len is not None and max_len is not None and min_len > max_len:
        raise ValueError("data.min_len cannot exceed data.max_len.")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Text-only SFT requires the `datasets` package from requirements.txt.") from error
    rows = load_dataset("parquet", data_files={"train": _parquet_files(path)}, split="train")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    examples: list[dict[str, list[int]]] = []
    for row_index in order:
        row = rows[row_index]
        metadata_len = _metadata_token_count(row, row_index)
        # Avoid tokenizing long s1K traces that are known to be excluded. Do
        # not use this for min_len: only the actual tokenizer is authoritative
        # for accepting short examples.
        if max_len is not None and metadata_len is not None and metadata_len > max_len:
            continue
        feature = _tokenize(processor, _messages(row.get("messages"), row_index), row_index)
        response_len = sum(token != IGNORE_INDEX for token in feature["labels"])
        if ((min_len is not None and response_len < min_len)
                or (max_len is not None and response_len > max_len)):
            continue
        examples.append(feature)
        if max_samples is not None and len(examples) >= max_samples:
            break
    if not examples:
        raise ValueError("No text-only examples remained after validation and length filtering.")
    return TextOnlyDataset(examples)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError("The training config must be a YAML mapping.")
    return dict(config)


def train(config: Mapping[str, Any], config_path: str | Path, qwen_config_path: str | Path,
          device_override: str | None = None) -> None:
    model_cfg, adapter_cfg, data_cfg, coconut_cfg, training_cfg, history_cfg = (
        _section(config, key) for key in ("model", "adapter", "data", "coconut", "training", "history")
    )
    if not bool(coconut_cfg.get("stage0_plain_sft", False)) or int(coconut_cfg.get("max_latent_stage", 0)) != 0:
        raise ValueError("Set coconut.stage0_plain_sft: true and coconut.max_latent_stage: 0.")
    if str(adapter_cfg.get("type", "")).lower() != "kradapter":
        raise ValueError("adapter.type must be kradapter for eval/custom/mmmu.py compatibility.")
    train_path = data_cfg.get("train_path")
    if not isinstance(train_path, str) or not train_path:
        raise ValueError("config.data.train_path must point to local s1K parquet data.")
    history_dir, save_name = config.get("history_dir", "./history"), history_cfg.get("save_name")
    if not isinstance(save_name, str) or not save_name.strip():
        raise TypeError("config.history.save_name must be a non-empty string.")
    load_checkpoint = history_cfg.get("load_checkpoint") or training_cfg.get("resume_from")
    if load_checkpoint is not None and (not isinstance(load_checkpoint, str) or not load_checkpoint.strip()):
        raise TypeError("config.history.load_checkpoint must be a non-empty checkpoint path or run name when set.")
    resume_path = _resolve_checkpoint_path(history_dir, load_checkpoint) if load_checkpoint else None

    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if device_override == "cpu":
        device = torch.device("cpu")
    elif device_override == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        device = torch.device("cuda", local_rank)
    elif device_override is None:
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device override: {device_override!r}")
    seed = int(config.get("seed", 42))
    sample_seed = data_cfg.get("sample_seed", seed)
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
        raise TypeError("config.data.sample_seed must be an integer.")
    _seed(seed)
    components = load_qwen3_vl(
        model_cfg.get("path"), revision=model_cfg.get("revision"), device=str(device), device_map=None,
        torch_dtype=model_cfg.get("dtype", "bf16"), attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model, processor = components.model, components.processor
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_modules = _configure_kradapter(model, adapter_cfg)
    frozen_vision_modules = _freeze_vision_encoder(model)
    if bool(training_cfg.get("gradient_checkpointing", False)) and not model.get_input_embeddings().weight.requires_grad:
        _enable_input_require_grads(model)
    train_set = load_text_only_dataset(train_path, processor, max_samples=data_cfg.get("max_train_samples"),
                                       seed=sample_seed, min_len=data_cfg.get("min_len"), max_len=data_cfg.get("max_len"))
    val_path = data_cfg.get("val_path")
    val_set = (load_text_only_dataset(val_path, processor, max_samples=data_cfg.get("max_val_samples"),
                                      seed=sample_seed, min_len=data_cfg.get("min_len"), max_len=data_cfg.get("max_len"))
               if val_path else None)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters remain after KRAdapter configuration.")
    optimizer = torch.optim.AdamW(trainable, lr=float(training_cfg.get("lr", 1e-5)),
                                  weight_decay=float(training_cfg.get("weight_decay", 0.01)))
    start_epoch, global_step = 0, 0
    if resume_path is not None:
        stage, start_epoch, global_step, optimizer_state = _load_state(model, str(resume_path), device, None)
        if stage != 0:
            raise ValueError("A text-only plain-SFT run can resume only a stage-0 checkpoint.")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    epochs = int(coconut_cfg.get("epochs_per_stage", 1))
    if epochs <= 0:
        raise ValueError("config.coconut.epochs_per_stage must be positive.")
    manager = CheckpointManager(history_dir, save_name, config_path, config)
    if rank == 0:
        run_dir = manager.prepare()
        shutil.copy2(qwen_config_path, run_dir / "qwen.yaml")
        print(f"Checkpoint output run: {run_dir}", flush=True)
        print(f"Text-only examples: train={len(train_set)}, validation={len(val_set) if val_set else 0}", flush=True)
        print(f"KRAdapter applied to {len(adapter_modules)} Linear layers: {', '.join(adapter_modules)}", flush=True)
        if frozen_vision_modules:
            print(f"Frozen unused vision encoder modules: {', '.join(frozen_vision_modules)}", flush=True)
    if start_epoch >= epochs:
        if rank == 0:
            print(f"All {epochs} text-only SFT epochs are already complete; nothing to train.", flush=True)
        return
    dtype = str(model_cfg.get("dtype", "bf16")).lower()
    bf16 = torch.cuda.is_available() and dtype in {"bf16", "bfloat16"}
    fp16 = torch.cuda.is_available() and dtype in {"fp16", "float16", "half"}
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    if bf16 and fp16 or accumulation <= 0:
        raise ValueError("Invalid precision or gradient_accumulation_steps.")
    trainer_args = TrainingArguments(
        output_dir=str(manager.run_dir / "trainer_stage00"), num_train_epochs=epochs - start_epoch,
        per_device_train_batch_size=int(training_cfg.get("batch_size", 1)),
        per_device_eval_batch_size=int(training_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=accumulation, learning_rate=float(training_cfg.get("lr", 1e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)), max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(training_cfg.get("num_workers", 0)), remove_unused_columns=False,
        eval_strategy="epoch" if val_set is not None else "no", save_strategy="no", logging_strategy="steps",
        logging_steps=int(training_cfg.get("logging_steps", 1)),
        lr_scheduler_type=str(training_cfg.get("lr_scheduler_type", "constant")),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)), bf16=bf16, fp16=fp16,
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
        report_to="none", seed=seed, data_seed=sample_seed,
    )
    callback = CoconutCheckpointCallback(manager=manager, stage=0, epoch_offset=start_epoch,
                                         global_step_offset=global_step, has_evaluation=val_set is not None, tokens=None)
    trainer = TextOnlyTrainer(model=model, args=trainer_args, train_dataset=train_set, eval_dataset=val_set,
                              data_collator=TextOnlyDataCollator(tokenizer.pad_token_id),
                              optimizers=(optimizer, None), callbacks=[callback])
    trainer.train()
    if trainer.is_world_process_zero():
        manager.mark_stage_completed(stage=0, epoch=epochs - 1, global_step=global_step + trainer.state.global_step)
        print("completed text-only plain SFT stage 0", flush=True)


def _direct_device(value: str | None) -> str | None:
    """Normalize direct-execution device syntax to Trainer's single-process view."""
    if value is None or value in {"cuda", "cpu"}:
        return value
    match = re.fullmatch(r"cuda:(\d+)", value)
    if match is None:
        raise ValueError("--device must be cuda, cuda:N, or cpu.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("--device cuda:N is supported only for a one-process launch.")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before --device cuda:N could select a GPU.")
    # Trainer sees this selected physical GPU as its local cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = match.group(1)
    return "cuda"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL with text-only s1K long-CoT SFT.")
    # Keep both the Python entrypoint names and train/run.sh's public names.
    parser.add_argument("--config", "--train-config", dest="config", default="config/train_text_only.yaml")
    parser.add_argument("--qwen-config", "--model-config", dest="qwen_config", default="config/qwen.yaml")
    parser.add_argument("--load-checkpoint", help="Checkpoint file or history run directory to resume from.")
    parser.add_argument("--save-name", help="New history run name for checkpoints written by this run.")
    parser.add_argument("--stage0-plain-sft", action="store_true",
                        help="Force COCONUT-compatible ordinary stage-0 SFT (the only text-only mode).")
    parser.add_argument("--device", default=None,
                        help="Direct-execution device: cuda, cuda:N, or cpu.")
    args = parser.parse_args()
    device_override = _direct_device(args.device)
    config_path, qwen_config_path = Path(args.config), Path(args.qwen_config)
    if not qwen_config_path.is_file():
        raise FileNotFoundError(f"Qwen config was not found: {qwen_config_path}")
    config = _load_yaml(config_path)
    history = _section(config, "history")
    if args.load_checkpoint is not None:
        history["load_checkpoint"] = args.load_checkpoint
    if args.save_name is not None:
        history["save_name"] = args.save_name
    config["history"] = history
    if args.stage0_plain_sft:
        coconut = _section(config, "coconut")
        coconut["stage0_plain_sft"] = True
        coconut["max_latent_stage"] = 0
        config["coconut"] = coconut
    train(config, config_path, qwen_config_path, device_override=device_override)


if __name__ == "__main__":
    main()
