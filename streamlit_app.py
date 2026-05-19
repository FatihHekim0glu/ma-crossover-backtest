"""Streamlit Cloud deployment entry point.

Streamlit Cloud auto-detects ``streamlit_app.py`` (or ``app.py``) at the
repository root. This module exists solely to satisfy the canonical
``streamlit_app.py`` convention while delegating all behaviour to the
existing ``app.py``. It executes ``app.py`` under ``__name__ == "__main__"``
so any top-level Streamlit calls behave identically to a direct
``streamlit run app.py`` invocation.
"""

from __future__ import annotations

import runpy
from pathlib import Path

_APP_PATH: Path = Path(__file__).resolve().parent / "app.py"

runpy.run_path(str(_APP_PATH), run_name="__main__")
