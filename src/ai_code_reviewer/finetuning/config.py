from dataclasses import dataclass
from typing import Optional


@dataclass
class FinetuneConfig:
    """Hyperparameters for full fine-tuning of the reviewer (causal LM)."""

    model_name: str = "Qwen/Qwen3-1.7B"
    output_dir: str = "outputs/qwen3-review-sft"
    train_file: str = "data/train.jsonl"
    validation_file: Optional[str] = None
    max_seq_length: int = 16384
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: Optional[int] = 500
    save_total_limit: int = 3
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    trust_remote_code: bool = True
    dataloader_num_workers: int = 0
    report_to: str = "none"
