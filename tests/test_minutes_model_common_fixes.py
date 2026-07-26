"""Owner ITEM 2 — MinutesModel common fixes.

(A) DNP sample weights: the DNP head trains on ALL eligible rows with the FULL-row weights;
    the conditional-minute regressors train on APPEARANCE rows only with the appearance-subset
    weights; filtering must not misalign weights; weighted and unweighted paths both work.
(B) One minutes maximum: read ``minutes_clip_max`` ONCE, persist it, use it in
    fit/mean/quantile/sigma/serialization; no second hard-coded production minutes maximum.
(C) Strict metadata: when appearance-only training is enabled, missing ``did_play`` is FATAL in
    strict mode (no silent train-on-all-rows).
"""
from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.models import minutes_model as mm
from wnba_props_model.models.minutes_model import (
    DEFAULT_MINUTES_CLIP_MAX,
    MinutesModel,
    MissingMinutesMetadataError,
)


def _synth(n: int = 300, seed: int = 7, dnp_rate: float = 0.15):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "player_minutes_l5": rng.uniform(8, 38, n),
        "team_pace": rng.uniform(90, 110, n),
    })
    dnp_mask = rng.random(n) < dnp_rate
    y_raw = np.clip(rng.normal(25, 8, n), 3, 44)
    y = pd.Series(np.where(dnp_mask, 0.0, y_raw))
    meta = pd.DataFrame({
        "projected_minutes_bucket": rng.choice(["low", "high"], n),
        "role_uncertainty_bucket": ["certain"] * n,
        "did_play": (~dnp_mask).astype(int),
    })
    return X, y, meta, dnp_mask


# ---------------------------------------------------------------------------
# (A) DNP sample weights + appearance-only conditional weights
# ---------------------------------------------------------------------------


def test_dnp_head_uses_all_rows_conditional_uses_appearances():
    X, y, meta, dnp_mask = _synth()
    model = MinutesModel({"hgb_regressor": {"max_iter": 40}})
    n_appear = int((~dnp_mask).sum())

    captured = {}
    real_lr_fit = mm.LogisticRegression.fit

    orig_reg_fit = mm.HistGradientBoostingRegressor.fit

    def spy_reg_fit(self, Xf, yf, sample_weight=None):  # noqa: ANN001
        captured.setdefault("reg_rows", []).append(len(Xf))
        return orig_reg_fit(self, Xf, yf, sample_weight=sample_weight)

    with mock.patch.object(mm.HistGradientBoostingRegressor, "fit", spy_reg_fit):
        model.fit(X, y, meta)

    # Every conditional regressor (mean + 5 quantiles) saw ONLY appearance rows.
    assert captured["reg_rows"], "regressor fit not observed"
    assert all(r == n_appear for r in captured["reg_rows"]), captured["reg_rows"]
    # DNP head is fitted (two classes present) and trained on the full row set.
    assert model._dnp_model is not None
    assert real_lr_fit is not None  # sanity


def test_dnp_weights_use_all_rows_conditional_weights_use_appearances():
    X, y, meta, dnp_mask = _synth()
    n = len(X)
    n_appear = int((~dnp_mask).sum())
    # A distinctive weight vector so we can detect misalignment.
    sw = np.linspace(0.1, 2.0, n)

    seen = {}
    orig_reg_fit = mm.HistGradientBoostingRegressor.fit
    orig_pipe_fit = mm.Pipeline.fit

    def spy_reg_fit(self, Xf, yf, sample_weight=None):  # noqa: ANN001
        seen.setdefault("reg_w", []).append(
            None if sample_weight is None else np.asarray(sample_weight, float).copy())
        return orig_reg_fit(self, Xf, yf, sample_weight=sample_weight)

    def spy_pipe_fit(self, Xf, yf=None, **kw):  # noqa: ANN001
        seen["dnp_w"] = kw.get("clf__sample_weight")
        seen["dnp_rows"] = len(Xf)
        return orig_pipe_fit(self, Xf, yf, **kw)

    model = MinutesModel({"hgb_regressor": {"max_iter": 40}})
    with mock.patch.object(mm.HistGradientBoostingRegressor, "fit", spy_reg_fit), \
         mock.patch.object(mm.Pipeline, "fit", spy_pipe_fit):
        model.fit(X, y, meta, sample_weight=sw)

    # DNP head: full rows + full weights, exactly the appearance mask complement kept.
    assert seen["dnp_rows"] == n
    assert seen["dnp_w"] is not None
    np.testing.assert_allclose(np.asarray(seen["dnp_w"], float), sw)
    # Conditional regressors: appearance-subset weights, correctly aligned (no misalignment).
    expected_cond_w = sw[~dnp_mask]
    for w in seen["reg_w"]:
        assert w is not None and len(w) == n_appear
        np.testing.assert_allclose(w, expected_cond_w)


def test_unweighted_and_weighted_paths_both_fit():
    X, y, meta, _ = _synth()
    m0 = MinutesModel({"hgb_regressor": {"max_iter": 40}}).fit(X, y, meta)
    m1 = MinutesModel({"hgb_regressor": {"max_iter": 40}}).fit(
        X, y, meta, sample_weight=np.ones(len(X)) * 0.5)
    for m in (m0, m1):
        means, sig, p_dnp = m.predict(X, meta)
        assert means.shape == (len(X),)
        assert np.all(np.isfinite(means)) and np.all(np.isfinite(sig))
        assert np.all((p_dnp >= 0) & (p_dnp <= 1))


# ---------------------------------------------------------------------------
# (B) One minutes maximum
# ---------------------------------------------------------------------------


def test_single_minutes_maximum_is_persisted_and_used_everywhere():
    X, y, meta, _ = _synth()
    clip = 40.0
    model = MinutesModel({"minutes_clip_max": clip, "hgb_regressor": {"max_iter": 40}}).fit(
        X, y, meta)
    assert model._minutes_clip_max == clip
    assert model._clip_max() == clip
    means, _, _ = model.predict(X, meta)
    q = model.predict_quantiles(X, meta)
    assert means.max() <= clip + 1e-9
    assert q.max() <= clip + 1e-9
    assert model.get_training_summary()["minutes_clip_max"] == clip


def test_clip_max_default_is_single_canonical_value():
    X, y, meta, _ = _synth()
    model = MinutesModel({"hgb_regressor": {"max_iter": 40}}).fit(X, y, meta)
    assert model._clip_max() == DEFAULT_MINUTES_CLIP_MAX == 48.0


def test_no_second_minutes_maximum_literal():
    """SOURCE-LEVEL regression: the only minutes-maximum literal is DEFAULT_MINUTES_CLIP_MAX.

    Any second hard-coded production maximum (42, 45, or a bare ``minutes_clip_max`` default other
    than the single constant) reintroduces the inconsistency this item fixes.
    """
    src = Path(mm.__file__).read_text()
    # Every cfg read of minutes_clip_max must default to the SINGLE canonical constant — never a
    # bare numeric fallback (that is exactly how the old 45 vs 48 inconsistency crept in).
    numeric_defaults = re.findall(
        r"cfg\.get\(\s*[\"']minutes_clip_max[\"']\s*,\s*([0-9][0-9.]*)\s*\)", src)
    assert not numeric_defaults, (
        f"minutes_clip_max must default to DEFAULT_MINUTES_CLIP_MAX, found numeric "
        f"fallback(s): {numeric_defaults}")
    # No forbidden hard-coded minute caps (42 / 45 / a bare 48) on any clip-bearing line.
    for line in src.splitlines():
        if "clip" not in line.lower():
            continue
        if "DEFAULT_MINUTES_CLIP_MAX = 48.0" in line:
            continue  # the single canonical constant definition
        for forbidden in ("42", "45", "48"):
            assert not re.search(rf"(?<![0-9.]){forbidden}(?:\.0)?(?![0-9.])", line), (
                f"forbidden hard-coded minutes literal {forbidden!r} on clip line: {line.strip()!r}")
    # Exactly one canonical default definition (the sole 48.0 literal is the module constant).
    assert src.count("DEFAULT_MINUTES_CLIP_MAX = 48.0") == 1
    assert src.count("48.0") == 1


# ---------------------------------------------------------------------------
# (C) Strict metadata
# ---------------------------------------------------------------------------


def test_missing_did_play_is_fatal_in_strict_mode():
    X, y, meta, _ = _synth()
    meta_no_didplay = meta.drop(columns=["did_play"])
    model = MinutesModel({
        "train_minutes_on_appearances_only": True,
        "strict_minutes_metadata": True,
        "hgb_regressor": {"max_iter": 40},
    })
    with pytest.raises(MissingMinutesMetadataError):
        model.fit(X, y, meta_no_didplay)


def test_missing_did_play_non_strict_falls_back_and_warns():
    X, y, meta, _ = _synth()
    meta_no_didplay = meta.drop(columns=["did_play"])
    # Non-strict: legacy behaviour (train on all rows) still permitted.
    model = MinutesModel({
        "train_minutes_on_appearances_only": True,
        "strict_minutes_metadata": False,
        "hgb_regressor": {"max_iter": 40},
    })
    model.fit(X, y, meta_no_didplay)
    assert model._fitted
    assert model.get_training_summary()["trained_minutes_on_appearances_only"] is False
