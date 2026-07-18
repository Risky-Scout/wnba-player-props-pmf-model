# Incident: pregame_initial run #309 (29642662780) — exit 134

## Comparison
| field | #308 (29642444874) PASS | #309 (29642662780) FAIL |
|---|---|---|
| commit | 9dde686 | 9dde686 (identical) |
| failing step | — | "Sanity gate — block deployment if calibration is catastrophically wrong" |
| gate logic outcome | pass | **pass** (printed "All sanity gates passed — deployment approved") |
| process exit | 0 | **134** |

## Exact mechanism (from full private logs)
The gate step emitted, in order:
```
11:33:53.776  All sanity gates passed — deployment approved
11:33:53.776  terminate called without an active exception
11:33:58.033  line 36: 2826 Aborted (core dumped) python - <<'EOF'
##[error]Process completed with exit code 134.
```
Exit 134 = 128 + SIGABRT(6). `terminate called without an active exception` is a C++
`std::terminate()` — raised here during **interpreter / native-threadpool TEARDOWN**,
AFTER the gate had already decided PASS and printed it. It is NOT an assertion, malformed
parquet, incompatible artifact, or memory the gate logic touched (the gate completed).

The gate subprocess (`python - <<EOF`) imports `pandas` (3.0.3) and reads parquet, which
spins up Arrow's native CPU/IO thread pools (pyarrow 25.0.0). A non-deterministic teardown
race in that native pool aborts at process exit — which is exactly why run #308 passed and
run #309 failed on the **same commit** and same inputs (thread-scheduling dependent).

Category: **native library threadpool teardown abort** (Arrow/OpenMP `std::terminate`),
not an application assertion.

## Repair (gate NOT weakened or removed)
`scripts/calibration_sanity_gate.py` (replaces the inline heredoc):
1. Sets `OMP/OPENBLAS/MKL/NUMEXPR/ARROW/NUMBA_NUM_THREADS=1` and
   `NUMBA_THREADING_LAYER=workqueue` **before** importing numpy/pandas/pyarrow, so the
   native pools are single-threaded and tear down cleanly.
2. On the validated PASS path, flushes and `os._exit(0)` — bypassing the native
   interpreter teardown so a teardown race can never flip a decided PASS to 134.
3. Explicit gate FAILURES still `return 1` → exit 1 (unchanged blocking behavior).
The workflow step runs with `PYTHONFAULTHANDLER=1`, `PYTHONUNBUFFERED=1` and the thread caps.

## Regression test
`tests/test_p3_sanity_gate_incident.py`: runs the gate 20× on a PASS fixture asserting
exit 0 every time (never 134), asserts the FAIL path exits 1, asserts thread limits precede
heavy imports + `os._exit`, and asserts the workflow uses the hardened script (no heredoc).
