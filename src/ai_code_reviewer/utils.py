"""Shared utilities used across multiple modules."""

import json
import logging
import typing as tp
from pathlib import Path


logger = logging.getLogger(__name__)


def parse_json_response(raw: str) -> tp.Dict[str, tp.Any]:
    """Best-effort extraction of a JSON object from an LLM response.

    Handles markdown fences and leading/trailing text around the JSON body.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
    return {}


def load_jsonl(path: tp.Union[str, Path]) -> tp.List[tp.Dict[str, tp.Any]]:
    """Read a JSONL file and return a list of parsed dicts."""
    rows: tp.List[tp.Dict[str, tp.Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows
