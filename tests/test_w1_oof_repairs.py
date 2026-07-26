"""Phase 5: W1 OOF must be trustworthy before the expensive run.

These are fail-closed source/contract regressions (the full OOF run needs the feature data,
which is unblocked separately in Phase 3):

  5.1 no global position encoder fitted on all dates (fold-safe, train-only, unknown-safe);
  5.3 --strict-baseline makes prior_only / failed_model_fit FATAL and requires all 7 props.
"""
from __future__ import annotations

from pathlib import Path

SRC = (Path(__file__).resolve().parent.parent / "scripts" / "build_oof_pmfs.py").read_text()


def test_no_global_encoder_leak():
    # 5.1: the global encoder fit + per-fold override (a future-category leak) must be gone.
    assert "global_pos_encoder" not in SRC
    assert "encode_features(wide, model_cols, fit_encoder=True)" not in SRC
    assert "Fold-safe encoding" in SRC


def test_strict_baseline_flag_and_fatal_paths():
    # 5.3: strict-baseline exists and converts prior_only/failed into FATAL aborts.
    assert '"--strict-baseline"' in SRC
    assert "if strict_baseline:" in SRC
    assert "may not emit prior_only" in SRC
    assert "may not convert a fit failure" in SRC
    # Completeness gate: 100% model_oof + all direct props.
    assert "must be 100% model_oof" in SRC
    assert "missing direct prop" in SRC


def test_encoder_uses_unknown_value():
    # The per-fold encoder maps unseen validation categories to explicit UNKNOWN (-1).
    training = (Path(__file__).resolve().parent.parent / "src" / "wnba_props_model"
                / "models" / "training.py").read_text()
    assert 'handle_unknown="use_encoded_value"' in training
    assert "unknown_value=-1" in training
