from __future__ import annotations

import gzip
import logging
import os
import tempfile
from collections.abc import Callable, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import zstandard as zstd

logger = logging.getLogger(__name__)

_GZIP_MAGIC = b"\x1f\x8b"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def dataset_to_plain_dict(obj: Any) -> Any:
    """Recursively convert `defaultdict` and other mappings to plain `dict` / `list`.

    Args:
        obj:
            Arbitrary nested structure (from the EDA dataset).

    Returns:
        JSON-serializable structure with only `dict`, `list`, str, numbers, booleans,
        and `None`.
    """
    if isinstance(obj, dict):  # covers defaultdict subclass
        return {k: dataset_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dataset_to_plain_dict(x) for x in obj]
    return obj


def plain_dict_to_dataset(
    data: dict[str, Any],
    make_dataset: Callable[[], MutableMapping[str, Any]],
) -> MutableMapping[str, Any]:
    """Build a fresh `make_dataset()` tree from a plain JSON-loaded dict.

    Args:
        data:
            Dataset mapping `repo -> pr -> ...` as loaded from JSON.
        make_dataset:
            Factory returning an empty dataset (e.g. `make_dataset`).

    Returns:
        Nested `defaultdict` dataset compatible with the EDA pipeline.
    """
    out = make_dataset()
    _merge_plain_dict_into_target(out, data)
    return out


def _merge_plain_dict_into_target(
    target: MutableMapping[str, Any],
    source: dict[str, Any],
) -> None:
    """Merge JSON `source` into `target` (result of `make_dataset()`)."""
    for repo_name, pr_map in source.items():
        for pr_number, pr_entry in pr_map.items():
            tgt_pr = target[repo_name][pr_number]
            if tgt_pr["base_commit"] is None and pr_entry.get("base_commit"):
                tgt_pr["base_commit"] = pr_entry["base_commit"]
            commits = pr_entry.get("commits") or {}
            for snap, path_map in commits.items():
                for path, file_entry in path_map.items():
                    tgt_file = tgt_pr["commits"][snap][path]
                    for key, val in file_entry.items():
                        if key == "comments":
                            tgt_file["comments"] = list(val)
                        else:
                            tgt_file[key] = val


def _decompress_checkpoint(raw: bytes) -> bytes:
    """Decompress checkpoint bytes (zstd preferred; gzip for legacy files)."""
    if len(raw) >= 4 and raw[:4] == _ZSTD_MAGIC:
        return zstd.ZstdDecompressor().decompress(raw)
    if len(raw) >= 2 and raw[:2] == _GZIP_MAGIC:
        return gzip.decompress(raw)
    msg = "Checkpoint is not zstd- or gzip-compressed (unrecognized magic bytes)"
    raise ValueError(msg)


def load_dataset_checkpoint(
    path: Path | str,
    make_dataset: Callable[[], MutableMapping[str, Any]],
) -> MutableMapping[str, Any]:
    """Load a JSON checkpoint (zstd or legacy gzip) into a dataset tree.

    Args:
        path:
            Path to ``*.json.zst`` from :func:`save_dataset_checkpoint`, or legacy
            ``*.json.gz``.
        make_dataset:
            Dataset factory.

    Returns:
        Nested `defaultdict` dataset.

    Raises:
        ValueError:
            If the file is missing a top-level ``dataset`` object or uses an
            unknown compression format.
    """
    path = Path(path)
    raw = path.read_bytes()
    payload = _decompress_checkpoint(raw)
    wrapper: dict[str, Any] = orjson.loads(payload)

    raw_dataset = wrapper.get("dataset")
    if not isinstance(raw_dataset, dict):
        raise ValueError("Checkpoint missing 'dataset' object")

    return plain_dict_to_dataset(raw_dataset, make_dataset)


def save_dataset_checkpoint(
    dataset: MutableMapping[str, Any],
    path: Path | str,
) -> Path:
    """Atomically write the dataset as zstd-compressed JSON (orjson).

    Args:
        dataset:
            In-memory dataset (`defaultdict` or plain nested dicts).
        path:
            Destination path (typically ``*.json.zst``).

    Returns:
        `Path` to the written file.

    Raises:
        OSError:
            On filesystem errors during write or replace.
    """
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_to_plain_dict(dataset),
    }

    payload = orjson.dumps(wrapper, option=orjson.OPT_NON_STR_KEYS)
    compressed = zstd.ZstdCompressor().compress(payload)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".checkpoint_",
        suffix=".tmp",
        dir=final_path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as raw:
            raw.write(compressed)
        os.replace(tmp_name, final_path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            logger.exception("Could not remove temp checkpoint %s", tmp_name)
        raise

    logger.info("Wrote checkpoint -> %s", final_path)
    return final_path
