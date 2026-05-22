"""Screenshot the live Streamlit dashboard via Playwright.

Used to capture proof-of-life imagery for the README and for QA reviews.
Waits for real content (the metrics dataframe + the plotly equity curve)
to render before taking the shot — not just page-load.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8502"
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # (1) Default dashboard at laptop viewport (1920x1080).
        ctx1 = browser.new_context(viewport={"width": 1920, "height": 1080})
        page1 = ctx1.new_page()
        page1.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page1.get_by_text("Headline metrics").wait_for(timeout=120_000)
        page1.wait_for_timeout(4000)
        above_fold = OUT_DIR / "dashboard_above_the_fold.png"
        page1.screenshot(path=str(above_fold), full_page=False)
        print(f"wrote {above_fold}  ({above_fold.stat().st_size:,} bytes)")

        # (2) Full dashboard at tall viewport.
        ctx2 = browser.new_context(viewport={"width": 1920, "height": 4000})
        page2 = ctx2.new_page()
        page2.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page2.get_by_text("Headline metrics").wait_for(timeout=120_000)
        page2.get_by_text("Cost sensitivity").wait_for(timeout=120_000)
        page2.wait_for_timeout(5000)
        full = OUT_DIR / "dashboard_full.png"
        page2.screenshot(path=str(full), full_page=False)
        print(f"wrote {full}  ({full.stat().st_size:,} bytes)")

        # (3) Walk-forward tab populated. This is the project's punchline
        # view — the IS-vs-OOS scatter that makes the no-skill conclusion
        # visceral.
        ctx3 = browser.new_context(viewport={"width": 1920, "height": 4500})
        page3 = ctx3.new_page()
        page3.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page3.get_by_text("Headline metrics").wait_for(timeout=120_000)
        page3.wait_for_timeout(3000)
        # Click the walk-forward tab heading (avoid Sweep tab).
        page3.get_by_role("tab", name="Walk-forward (the honest one)").click()
        page3.wait_for_timeout(1500)
        # Click "Run walk-forward" — it has key="run_wf" in app.py.
        page3.get_by_role("button", name="Run walk-forward").click()
        # The compute is non-trivial (~30s for 319 configs across ~10 folds);
        # wait for the OOS Sharpe metric label to appear.
        page3.get_by_text("Concatenated OOS Sharpe").wait_for(timeout=300_000)
        page3.wait_for_timeout(4000)
        wf = OUT_DIR / "dashboard_walk_forward.png"
        page3.screenshot(path=str(wf), full_page=False)
        print(f"wrote {wf}  ({wf.stat().st_size:,} bytes)")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
