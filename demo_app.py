"""Compatibility shim for the legacy entrypoint.

The demo lives under :mod:`demo` now; this file just forwards to
:func:`demo.app.main` so existing `streamlit run demo_app.py` invocations
keep working.

Prefer:

    streamlit run demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from demo.app import main  # noqa: E402


if __name__ == "__main__":
    main()
