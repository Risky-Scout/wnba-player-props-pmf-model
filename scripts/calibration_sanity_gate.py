"""Calibration sanity gate (extracted from pregame_initial inline heredoc).

Incident (run #309, 29642662780): the inline gate printed "All sanity gates passed"
and then the process died with exit 134 —

    terminate called without an active exception
    Aborted (core dumped)

i.e. a C++ ``std::terminate`` (SIGABRT) during interpreter/native-threadpool TEARDOWN,
AFTER the gate had already decided PASS. pandas 3.x + pyarrow 25 spin up Arrow's native
CPU/IO thread pools when reading parquet; a non-deterministic teardown race can abort at
exit (which is why #308 passed and #309 failed on the same commit 9dde686).

Root-cause repair (does NOT weaken or remove the gate):
  1. Single-thread the native numerical/Arrow pools BEFORE importing them, so the pools
     tear down cleanly (env vars are read at import time).
  2. On the validated PASS path, flush and ``os._exit(0)`` to bypass the native
     interpreter teardown entirely — a teardown race can never flip a decided PASS to 134.
Explicit gate FAILURES still exit nonzero (as they must).
"""
from __future__ import annotations

import os

# ── Must run BEFORE numpy/pyarrow/pandas import (env is read at import time). ──────
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ARROW_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # avoid TBB/OMP teardown abort
os.environ.setdefault("PYTHONFAULTHANDLER", "1")

import json  # noqa: E402
import sys  # noqa: E402
import pathlib  # noqa: E402

import typer  # noqa: E402

app = typer.Typer(add_completion=False)


def _clean_exit(code: int) -> None:
    """Flush and bypass native teardown so a threadpool abort can't corrupt the exit code."""
    try:
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os._exit(code)


def run_gate(cal_dir: str, edges: str, bias_floor: float = 0.50,
             mean_edge_floor: float = -20.0) -> int:
    """Return 0 (pass) or 1 (fail). Pure — no process exit — so it is unit-testable."""
    import pandas as pd

    bias_path = pathlib.Path(cal_dir) / "bias_corrections.json"
    if bias_path.exists():
        bias = json.loads(bias_path.read_text())
        vals = {k: float(v) for k, v in bias.items() if isinstance(v, (int, float))}
        for stat, val in vals.items():
            if val < bias_floor:
                print(f"GATE FAIL: bias_corrections[{stat}]={val:.3f} below {bias_floor:.2f} floor")
                return 1
        if vals:
            print(f"Gate 1 PASS: bias_corrections min={min(vals.values()):.3f}")
    else:
        print("Gate 1 SKIP: bias_corrections.json not found")

    edges_path = pathlib.Path("artifacts/edge_report/publishable_edges.parquet")
    if not edges_path.exists():
        edges_path = pathlib.Path(edges)
    if edges_path.exists():
        df = pd.read_parquet(edges_path)
        if "edge_pp" in df.columns and len(df) > 0:
            mean_edge = float(df["edge_pp"].mean())
            print(f"Gate 2: mean_edge_pp={mean_edge:.2f} across {len(df)} rows")
            if mean_edge < mean_edge_floor:
                print(f"DEPLOY BLOCKED: mean_edge={mean_edge:.1f}pp — calibration mismatch.")
                return 1
            print(f"Gate 2 PASS: mean edge {mean_edge:.1f}pp above {mean_edge_floor:.0f}pp threshold")
        else:
            print("Gate 2 SKIP: no edge_pp column or empty parquet (abstain OK)")
    else:
        print("Gate 2 SKIP: publishable_edges.parquet not found")

    print("All sanity gates passed — deployment approved")
    return 0


@app.command()
def main(cal_dir: str = typer.Option("artifacts/models/calibration"),
         edges: str = typer.Option("deliveries/tonight/publishable_edges.parquet")) -> None:
    code = run_gate(cal_dir, edges)
    _clean_exit(code)


if __name__ == "__main__":
    try:
        app(standalone_mode=False)
    except SystemExit as exc:  # typer/click may raise SystemExit before our clean exit
        _clean_exit(int(exc.code) if isinstance(exc.code, int) else 1)
