---
name: Bug report
about: Report a defect in the moving-average crossover backtester
title: ""
labels: bug
assignees: ""
---

## Title

<!-- One-line summary of the bug. -->

## Expected behaviour

<!-- What did you expect to happen? -->

## Actual behaviour

<!-- What actually happened? Include error messages, stack traces, or
unexpected output. Paste tracebacks inside a fenced code block. -->

## Reproduction

Minimal steps to reproduce. Prefer one of:

- A CLI invocation:

  ```bash
  uv run ma-backtester ...
  ```

- Or a notebook cell (paste the exact cell contents and note which notebook
  it came from):

  ```python
  # notebooks/<name>.ipynb, cell N
  ...
  ```

Include the input parameters (ticker, date range, window sizes, etc.) and
any data file paths required.

## Environment

Please provide:

- Python version: `python --version`
- Operating system and version (e.g. Windows 11, macOS 14.5, Ubuntu 24.04)
- uv version: `uv --version`
- Relevant package versions:

  ```bash
  pip list | grep -E "pandas|numpy|streamlit|yfinance"
  ```

## Additional context

<!-- Anything else that may help: recent changes, data source quirks,
intermittent vs. deterministic, etc. -->
