"""Full fine-tuning (supervised) for the Qwen3 review model."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.ai_code_reviewer.finetuning.config import FinetuneConfig
from src.ai_code_reviewer.finetuning.data import encode_sft_example, load_jsonl_rows

logger = logging.getLogger(__name__)


@dataclass
class CausalSFTCollator:
    """Pad variable-length ``input_ids`` / ``labels`` for decoder-only causal LM training."""

    pad_token_id: int
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []
        for f in features:
            ids = list(f["input_ids"])
            attn = list(f["attention_mask"])
            lbl = list(f["labels"])
            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(attn + [0] * pad_len)
            batch_labels.append(lbl + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset_from_jsonl(
    path: Path,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dataset:
    rows = load_jsonl_rows(path)
    prompts: List[str] = []
    targets: List[Any] = []
    for row in rows:
        if "prompt" not in row:
            logger.warning("Skipping row without 'prompt' in %s", path)
            continue
        prompts.append(row["prompt"])
        targets.append(row.get("target"))

    if not prompts:
        raise ValueError(f"No usable rows in {path}")

    raw = Dataset.from_dict({"prompt": prompts, "target": targets})

    def _tokenize_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []
        for prompt, target in zip(batch["prompt"], batch["target"], strict=True):
            enc = encode_sft_example(prompt, target, tokenizer, max_seq_length)
            input_ids.append(enc["input_ids"])
            attention_mask.append(enc["attention_mask"])
            labels.append(enc["labels"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return raw.map(
        _tokenize_batch,
        batched=True,
        remove_columns=raw.column_names,
        desc=f"Tokenizing {path.name}",
    )


def parse_args(argv: Optional[List[str]] = None) -> FinetuneConfig:
    p = argparse.ArgumentParser(description="Full SFT for Qwen3 code review (JSONL prompt/target).")
    p.add_argument("--model_name", type=str, default=FinetuneConfig.model_name)
    p.add_argument("--output_dir", type=str, default=FinetuneConfig.output_dir)
    p.add_argument("--train_file", type=str, default=FinetuneConfig.train_file)
    p.add_argument("--validation_file", type=str, default="", help="Optional val JSONL; empty to skip eval.")
    p.add_argument("--max_seq_length", type=int, default=FinetuneConfig.max_seq_length)
    p.add_argument("--per_device_train_batch_size", type=int, default=FinetuneConfig.per_device_train_batch_size)
    p.add_argument("--per_device_eval_batch_size", type=int, default=FinetuneConfig.per_device_eval_batch_size)
    p.add_argument("--gradient_accumulation_steps", type=int, default=FinetuneConfig.gradient_accumulation_steps)
    p.add_argument("--learning_rate", type=float, default=FinetuneConfig.learning_rate)
    p.add_argument("--num_train_epochs", type=float, default=FinetuneConfig.num_train_epochs)
    p.add_argument("--warmup_ratio", type=float, default=FinetuneConfig.warmup_ratio)
    p.add_argument("--weight_decay", type=float, default=FinetuneConfig.weight_decay)
    p.add_argument("--max_grad_norm", type=float, default=FinetuneConfig.max_grad_norm)
    p.add_argument("--logging_steps", type=int, default=FinetuneConfig.logging_steps)
    p.add_argument("--save_steps", type=int, default=FinetuneConfig.save_steps)
    p.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluation every N steps; 0 means once per epoch (when validation_file is set).",
    )
    p.add_argument("--save_total_limit", type=int, default=FinetuneConfig.save_total_limit)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=FinetuneConfig.bf16)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=FinetuneConfig.gradient_checkpointing,
    )
    p.add_argument("--seed", type=int, default=FinetuneConfig.seed)
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataloader_num_workers", type=int, default=FinetuneConfig.dataloader_num_workers)
    p.add_argument("--report_to", type=str, default=FinetuneConfig.report_to)

    ns = p.parse_args(argv)
    eval_steps: Optional[int] = ns.eval_steps if ns.eval_steps > 0 else None

    val_path: Optional[str] = ns.validation_file.strip() or None

    return FinetuneConfig(
        model_name=ns.model_name,
        output_dir=ns.output_dir,
        train_file=ns.train_file,
        validation_file=val_path,
        max_seq_length=ns.max_seq_length,
        per_device_train_batch_size=ns.per_device_train_batch_size,
        per_device_eval_batch_size=ns.per_device_eval_batch_size,
        gradient_accumulation_steps=ns.gradient_accumulation_steps,
        learning_rate=ns.learning_rate,
        num_train_epochs=ns.num_train_epochs,
        warmup_ratio=ns.warmup_ratio,
        weight_decay=ns.weight_decay,
        max_grad_norm=ns.max_grad_norm,
        logging_steps=ns.logging_steps,
        save_steps=ns.save_steps,
        eval_steps=eval_steps,
        save_total_limit=ns.save_total_limit,
        bf16=ns.bf16,
        gradient_checkpointing=ns.gradient_checkpointing,
        seed=ns.seed,
        trust_remote_code=ns.trust_remote_code,
        dataloader_num_workers=ns.dataloader_num_workers,
        report_to=ns.report_to,
    )


def run_training(cfg: FinetuneConfig) -> None:
    _set_seed(cfg.seed)

    train_path = Path(cfg.train_file)
    if not train_path.is_file():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    logger.info("Loading tokenizer %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    logger.info("Loading model %s (dtype=%s)", cfg.model_name, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_ds = _build_dataset_from_jsonl(train_path, tokenizer, cfg.max_seq_length)

    eval_ds: Optional[Dataset] = None
    eval_strategy = "no"
    if cfg.validation_file:
        val_path = Path(cfg.validation_file)
        if val_path.is_file():
            eval_ds = _build_dataset_from_jsonl(val_path, tokenizer, cfg.max_seq_length)
            eval_strategy = "steps" if cfg.eval_steps else "epoch"
        else:
            logger.warning("Validation file not found, skipping eval: %s", val_path)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer pad_token_id is required for padding.")
    data_collator = CausalSFTCollator(pad_token_id=int(pad_id), pad_to_multiple_of=8)

    training_kwargs: Dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "bf16": cfg.bf16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "dataloader_num_workers": cfg.dataloader_num_workers,
        "report_to": cfg.report_to,
        "save_strategy": "steps",
        "remove_unused_columns": False,
        "seed": cfg.seed,
    }

    if eval_ds is not None:
        training_kwargs["eval_strategy"] = eval_strategy
        training_kwargs["load_best_model_at_end"] = eval_strategy == "steps"
        training_kwargs["metric_for_best_model"] = "loss"
        training_kwargs["greater_is_better"] = False
        if eval_strategy == "steps" and cfg.eval_steps:
            training_kwargs["eval_steps"] = cfg.eval_steps
    else:
        training_kwargs["eval_strategy"] = "no"

    args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training; train rows=%d", len(train_ds))
    trainer.train()
    logger.info("Saving model to %s", cfg.output_dir)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    cfg = parse_args(argv)
    run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
