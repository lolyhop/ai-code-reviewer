"""Load JSONL exports and build tokenized examples for supervised fine-tuning."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizerBase

from ai_code_reviewer.utils import load_jsonl

logger = logging.getLogger(__name__)

load_jsonl_rows = load_jsonl


def _target_to_str(raw: Any) -> str:
    if raw is None:
        return '{"issues": []}'
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False)


def _build_chat_strings(
    prompt: str,
    target: str,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[str, str]:
    """Return (user_segment, full_conversation) text for Qwen chat template."""
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
    full = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return user_only, full


def _full_token_len(prompt: str, target: str, tokenizer: PreTrainedTokenizerBase) -> int:
    _, full = _build_chat_strings(prompt, target, tokenizer)
    return len(tokenizer.encode(full, add_special_tokens=False))


def _fit_prompt_suffix(
    prompt: str,
    target: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> str:
    """
    If the full chat exceeds ``max_seq_length`` tokens, keep a suffix of ``prompt``
    (end of diff / file) so the assistant ``target`` stays intact.
    """
    if _full_token_len(prompt, target, tokenizer) <= max_seq_length:
        return prompt

    lo, hi = 1, len(prompt)
    best = prompt[-1:] if hi else prompt
    while lo <= hi:
        mid = (lo + hi) // 2
        suffix = prompt[-mid:]
        if _full_token_len(suffix, target, tokenizer) <= max_seq_length:
            best = suffix
            lo = mid + 1
        else:
            hi = mid - 1

    p = best
    for _ in range(64):
        if _full_token_len(p, target, tokenizer) <= max_seq_length or len(p) == 0:
            break
        step = max(1, len(p) // 32)
        p = p[step:]
    return p


def _longest_common_prefix(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def encode_sft_example(
    prompt: str,
    target_raw: Any,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> Dict[str, List[int]]:
    """
    Tokenize one example; labels are ``-100`` on user tokens, token ids on assistant span.
    """
    target = _target_to_str(target_raw)
    prompt_fit = _fit_prompt_suffix(prompt, target, tokenizer, max_seq_length)

    user_only, full = _build_chat_strings(prompt_fit, target, tokenizer)
    prompt_ids = tokenizer.encode(user_only, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)

    if len(full_ids) > max_seq_length:
        drop = len(full_ids) - max_seq_length
        full_ids = full_ids[drop:]
        if drop <= len(prompt_ids):
            prompt_ids = prompt_ids[drop:]
        else:
            prompt_ids = []

    lcp = _longest_common_prefix(prompt_ids, full_ids)
    labels = [-100] * lcp + full_ids[lcp:]

    if len(labels) != len(full_ids):
        raise RuntimeError(
            f"Label length {len(labels)} != input length {len(full_ids)}"
        )

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
