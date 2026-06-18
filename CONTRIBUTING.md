# Contributing

Thanks for taking the time to contribute. This document covers the local
workflow used in this repository.

## Setup

Install the project and all development extras with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras --dev
```

This creates the project virtual environment and installs runtime, optional,
and development dependencies declared in `pyproject.toml`.

## Verify before pushing

Run the full local check suite before opening a pull request. All four
commands must succeed.

- Tests must be green:

  ```bash
  uv run pytest -q
  ```

- Lint:

  ```bash
  uv run ruff check src tests app.py
  ```

- Format check (no rewrites; fails if files would be reformatted):

  ```bash
  uv run ruff format --check src tests app.py
  ```

- Static type checking:

  ```bash
  uv run pyright src tests app.py
  ```

If any of these fail, fix the issue locally rather than pushing and relying
on CI to surface it.

## Notebooks

The `.ipynb` files under `notebooks/` are **generated artefacts**. They are
produced from `notebooks/_generate.py`, which is the source of truth.

If you need to change a notebook:

1. Edit `notebooks/_generate.py` (not the `.ipynb` directly).
2. Regenerate the notebooks:

   ```bash
   uv run python notebooks/_generate.py
   ```

3. Commit both the updated generator and the regenerated `.ipynb` files in
   the same commit so they stay in sync.

Directly hand-edited notebooks will diverge from the generator and be
overwritten on the next regeneration.

## Adding a strategy variant

The strategy layer is intentionally small. To add a new variant:

- `StrategyConfig` is the entry point for configuring a strategy run. Any new
  strategy should accept its parameters via a `StrategyConfig` (extend it or
  add a sibling config as appropriate).
- A strategy function must follow the signature:

  ```python
  def my_strategy(close, config) -> signal:
      ...
  ```

  That is, it takes a `close` price series and a config object, and returns
  a `signal` series.
- Pipe the resulting signal through `signal_to_position` to convert it into
  positions, then through `run_backtest` to produce the backtest result.

Keep new strategies self-contained and composable with the existing
`signal -> position -> backtest` pipeline rather than introducing parallel
execution paths.

## Code style

- **Type hints are required** on all new and modified functions, methods,
  and module-level signatures.
- `ruff format` is the canonical formatter. Configure your editor to run
  `ruff format` on save so diffs stay minimal.
- Commit messages use plain
  [Conventional Commits](https://www.conventionalcommits.org/) subjects,
  e.g. `feat: add EMA crossover variant`, `fix: handle empty close series`,
  `refactor: extract signal_to_position helper`.
- Keep commits, PR descriptions, code comments, and every other repository
  artefact free of automated tool attribution, extra co-author trailers, and
  generated-with markers.
