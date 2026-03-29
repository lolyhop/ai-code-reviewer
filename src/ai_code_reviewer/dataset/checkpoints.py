from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, MutableMapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_code_reviewer.dataset import config

logger = logging.getLogger(__name__)


def dataset_to_plain_dict(obj: Any) -> Any:
    """Recursively convert ``defaultdict`` and other mappings to plain ``dict`` / ``list``.

    Args:
        obj:
            Arbitrary nested structure (from the EDA dataset).

    Returns:
        JSON-serializable structure with only ``dict``, ``list``, str, numbers, booleans,
        and ``None``.
    """
    if isinstance(obj, defaultdict):
        return {k: dataset_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {k: dataset_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dataset_to_plain_dict(x) for x in obj]
    return obj


def plain_dict_to_dataset(
    data: dict[str, Any],
    make_dataset: Callable[[], MutableMapping[str, Any]],
) -> MutableMapping[str, Any]:
    """Build a fresh ``make_dataset()`` tree from a plain JSON-loaded dict.

    Args:
        data:
            Dataset mapping ``repo -> pr -> ...`` as loaded from JSON.
        make_dataset:
            Factory returning an empty dataset (e.g. ``make_dataset``).

    Returns:
        Nested ``defaultdict`` dataset compatible with the EDA pipeline.
    """
    out = make_dataset()
    _merge_plain_dict_into_target(out, data)
    return out


def _merge_plain_dict_into_target(
    target: MutableMapping[str, Any],
    source: dict[str, Any],
) -> None:
    """Merge JSON ``source`` into ``target`` (result of ``make_dataset()``)."""
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


def load_dataset_checkpoint(
    path: Path | str,
    make_dataset: Callable[[], MutableMapping[str, Any]],
) -> MutableMapping[str, Any]:
    """Load a gzip JSON checkpoint and return a dataset compatible with ``make_dataset()``.

    Args:
        path:
            Path to ``*.json.gz`` written by ``save_dataset_checkpoint``.
        make_dataset:
            Dataset factory.

    Returns:
        Nested ``defaultdict`` dataset.

    Raises:
        ValueError:
            If schema version is unsupported or the file is missing ``dataset``.
    """
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        wrapper: dict[str, Any] = json.load(f)

    raw = wrapper.get("dataset")
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint missing 'dataset' object")

    return plain_dict_to_dataset(raw, make_dataset)


def save_dataset_checkpoint(
    dataset: MutableMapping[str, Any],
    path: Path | str,
) -> Path:
    """Atomically write the dataset to the given ``path`` (gzip JSON).

    The caller chooses ``path`` (including filename and parent directory).

    Args:
        dataset:
            In-memory dataset (``defaultdict`` or plain nested dicts).
        path:
            Destination file path, typically ending in ``.json.gz``.

    Returns:
        ``Path`` to the written file.

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

    fd, tmp_name = tempfile.mkstemp(
        prefix=".checkpoint_",
        suffix=".json.gz.tmp",
        dir=final_path.parent,
    )
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                payload = json.dumps(
                    wrapper,
                    ensure_ascii=False,
                ).encode("utf-8")
                gz.write(payload)
        os.replace(tmp_name, final_path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            logger.exception("Could not remove temp checkpoint %s", tmp_name)
        raise

    logger.info("Wrote checkpoint -> %s", final_path)
    return final_path
