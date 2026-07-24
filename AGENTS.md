# AGENTS.md

## Cursor Cloud specific instructions

This repo is the **WNBA Player Props PMF Model** — a Python batch data/ML pipeline
and CLI toolset (no web server, no database, no message queue). State lives in
Parquet/JSON files on disk; production scheduling is done via GitHub Actions cron.
See `README.md` for the pipeline architecture and per-script usage.

### Environment / dependencies
- Pure Python project. The update script runs `pip install -e ".[dev]"`, which
  installs into the user site (`~/.local`). No system services to start.
- Console entry points (`pytest`, `ruff`, `typer`, `optuna`) land in
  `~/.local/bin`, which is **not** always on `PATH`. Invoke tools as
  `python -m pytest` / `python -m ruff` to avoid `command not found`.
- The VM has Python 3.12; CI uses 3.11 and `pyproject.toml` requires `>=3.10`.
  Everything runs fine on 3.12.

### Test / lint / build
- Tests (no external services or API keys needed — `tests/conftest.py` builds
  synthetic sklearn model artifacts): `python -m pytest tests/ -q --ignore=tests/test_elite_projection_gate.py`
  (CI ignores that one file). ~1,200 tests, runs in ~90s.
- Lint: `python -m ruff check src scripts`. The repo currently has many
  pre-existing ruff findings; **CI does not gate on ruff** — the CI gate is
  `actionlint` (workflow YAML) + `pytest` (see `.github/workflows/ci.yml`).
- There is no build step. `make compile` just runs `python -m compileall`.

### Running the pipeline
- A real end-to-end data run (pull → features → train → predict → edges) needs
  `BDL_API_KEY` (Ball Don't Lie, GOAT tier) — see `.env.example`. That secret is
  not required for tests.
- Without the API key you can still exercise the production prediction path on
  synthetic inputs: build model artifacts the way `tests/conftest.py` does
  (`MinutesModel` / `StatRateModel` / `ZINBStatModel` + `feature_manifest.json` +
  a stage-4 config), then run `scripts/predict_today.py --no-calibration
  --features-wide <slate.parquet> --model-dir <dir> --config <cfg> --raw-props
  <market.parquet> --out-dir <out>` to produce PMFs, projections, and market
  edges.

### Static frontend (odds-scanner)
- `tools/odds-scanner/` is a static site (plain HTML/CSS/JS, no build) that
  fetches prediction JSON via `fetch()` on **relative** URLs, so `file://`
  won't work — serve it (e.g. `python -m http.server` from
  `tools/odds-scanner/`) and open `index.html`.
