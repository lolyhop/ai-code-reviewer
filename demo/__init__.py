"""Streamlit demo for the Automated Pull Request Reviewer.

This package provides a thin UI layer that reuses the existing
``ai_code_reviewer`` pipeline (prompt construction, local Qwen3 inference,
output parsing) and adds a small adapter for fetching live GitHub PR data.

Run with:

    streamlit run demo/app.py
"""

from __future__ import annotations

__all__ = [
    "adapters",
    "demo_data",
    "view",
]
