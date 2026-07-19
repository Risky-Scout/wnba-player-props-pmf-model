"""Forecasting-quality diagnostics for discrete PMF predictions (P2 Phase 3).

Pure, deterministic functions to evaluate whether the PMF forecasting model is
trustworthy independent of any betting result: bias, MAE, RMSE, CRPS, PIT
uniformity, central-interval coverage (with binomial-compatibility tests),
calibration error by probability bucket, tail calibration, and a per-stat launch
gate. Coverage is judged by whether nominal coverage lies inside the binomial
uncertainty interval, not by the point estimate alone.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np


def pmf_to_array(pmf_json) -> np.ndarray:
    if isinstance(pmf_json, str):
        s = pmf_json.strip()
        if not s:
            return np.array([])
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return np.array([])
    else:
        obj = pmf_json
    if obj is None:
        return np.array([])
    if isinstance(obj, dict):
        n = max(int(k) for k in obj) + 1
        a = np.zeros(n)
        for k, v in obj.items():
            a[int(k)] = float(v)
        return a
    return np.asarray(obj, dtype=float)


def _cdf(pmf: np.ndarray) -> np.ndarray:
    return np.clip(np.cumsum(pmf), 0.0, 1.0)


def crps_discrete(pmf: np.ndarray, y: int) -> float:
    """CRPS for a discrete distribution on 0..K: sum_k (CDF(k) - 1{y<=k})^2."""
    if pmf.size == 0:
        return float("nan")
    cdf = _cdf(pmf)
    k = np.arange(len(pmf))
    step = (k >= y).astype(float)
    return float(np.sum((cdf - step) ** 2))


def mid_pit(pmf: np.ndarray, y: int) -> float:
    """Mid-PIT value: CDF(y-1) + 0.5 * P(y). NOT uniform for discrete outcomes — retained
    for diagnostics only; DO NOT gate on its uniformity (use randomized_pit)."""
    if pmf.size == 0 or y < 0:
        return float("nan")
    below = float(pmf[:y].sum()) if y > 0 else 0.0
    at = float(pmf[y]) if y < len(pmf) else 0.0
    return below + 0.5 * at


def _stable_uniform(seed_key: str) -> float:
    """Deterministic pseudo-random V in [0,1) from a stable hash of the key, so the
    randomized PIT is reproducible across runs but not degenerate."""
    h = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def randomized_pit(pmf: np.ndarray, y: int, seed_key: str) -> float:
    """Randomized PIT for a discrete forecast: U = F(y-1) + V * P(Y=y), with V a stable
    per-observation uniform. U is Uniform(0,1) under correct calibration (Czado et al.)."""
    if pmf.size == 0 or y < 0:
        return float("nan")
    below = float(pmf[:y].sum()) if y > 0 else 0.0
    at = float(pmf[y]) if y < len(pmf) else 0.0
    v = _stable_uniform(seed_key)
    return below + v * at


def ks_uniform(u: np.ndarray) -> tuple[float, float]:
    """One-sample KS statistic of u against Uniform(0,1) + an asymptotic p-value."""
    u = np.sort(np.asarray(u, dtype=float))
    n = len(u)
    if n == 0:
        return float("nan"), float("nan")
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - u)
    d_minus = np.max(u - (i - 1) / n)
    d = float(max(d_plus, d_minus))
    # Kolmogorov asymptotic survival function
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(max(p, 0.0), 1.0))


OVERFLOW_FLOOR = 1e-6  # explicit tail probability for out-of-support outcomes


def clustered_pit_deviation(u: np.ndarray, clusters: np.ndarray, n_boot: int = 500,
                            seed: int = 13, grid_pts: int = 21):
    """Game-date block-bootstrap assessment of randomized-PIT non-uniformity.

    Resamples whole clusters (game dates) — preserving all correlated within-date
    observations — and computes the max deviation of the PIT empirical CDF from Uniform on
    a fixed grid for each resample. Returns (observed_dev, clustered_mean_dev, band95_hi).
    Duplicating correlated observations within one date cannot inflate nominal n here
    because whole dates are the resampling unit."""
    u = np.asarray(u, dtype=float); clusters = np.asarray(clusters)
    grid = np.linspace(0, 1, grid_pts)

    def _dev(vals):
        if len(vals) == 0:
            return float("nan")
        ecdf = np.searchsorted(np.sort(vals), grid, side="right") / len(vals)
        return float(np.max(np.abs(ecdf - grid)))

    observed = _dev(u)
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return observed, observed, observed
    rng = np.random.default_rng(seed)
    by = {c: u[clusters == c] for c in uniq}
    devs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([by[c] for c in pick])
        devs.append(_dev(vals))
    return observed, float(np.mean(devs)), float(np.percentile(devs, 95))


def log_score(pmf: np.ndarray, y: int, overflow_floor: float = OVERFLOW_FLOOR) -> float:
    """Negative log predictive probability at the realized count (lower is better).

    An outcome BEYOND stored support is scored through an explicit overflow probability
    (``overflow_floor``) rather than returning NaN — so every observation contributes a
    finite proper score and out-of-support misses cannot silently disappear from the mean."""
    if pmf.size == 0 or y < 0:
        return float("nan")
    p = float(pmf[y]) if y < len(pmf) else 0.0
    return -math.log(max(p, overflow_floor))


def validate_pmf(pmf: np.ndarray, y: int, tol: float = 1e-6) -> tuple[bool, str]:
    """PMF integrity check. Returns (ok, reason). Fails on non-finite/negative mass,
    mass not summing to 1 within tol, or empty support. `support_miss` (y beyond support)
    is returned as a distinct reason so it can be counted (not silently NaN'd)."""
    if pmf.size == 0:
        return False, "empty_support"
    if not np.all(np.isfinite(pmf)):
        return False, "nonfinite_mass"
    if np.any(pmf < -tol):
        return False, "negative_mass"
    s = float(pmf.sum())
    if abs(s - 1.0) > 1e-3:
        return False, f"mass_sum_{s:.4f}"
    if y >= len(pmf):
        return True, "support_miss"   # valid PMF but outcome beyond stored support
    return True, "ok"


def central_interval(pmf: np.ndarray, cover: float) -> tuple[int, int]:
    """Smallest central interval [lo,hi] with coverage >= `cover` (equal-tailed)."""
    cdf = _cdf(pmf)
    alpha = (1.0 - cover) / 2.0
    lo = int(np.searchsorted(cdf, alpha))
    hi = int(np.searchsorted(cdf, 1.0 - alpha))
    lo = min(lo, len(pmf) - 1)
    hi = min(hi, len(pmf) - 1)
    return lo, hi


def interval_residual(pmf: np.ndarray, actual: int, nominal: float) -> tuple[float, float, float, int]:
    """Discrete interval calibration primitive. Returns (contained_mass, hit, residual, width)
    where contained_mass is the PMF mass actually inside the (inclusive integer) central
    interval requested at ``nominal``, hit=1 if the outcome is inside, and residual = hit -
    contained_mass. A calibrated model has E[residual] ~ 0 REGARDLESS of the requested
    nominal, because inclusive integer intervals legitimately contain more mass than the
    continuous nominal (comparing empirical inclusion to the nominal directly is invalid)."""
    lo, hi = central_interval(pmf, nominal)
    contained = float(pmf[lo:hi + 1].sum())
    hit = 1.0 if lo <= actual <= hi else 0.0
    return contained, hit, hit - contained, hi - lo


def matched_mass_width(pmf: np.ndarray, target_mass: float) -> int:
    """Width of the smallest central interval whose contained mass >= target_mass — used
    for matched-mass sharpness comparisons (compare widths at the SAME contained mass)."""
    cdf = _cdf(pmf)
    alpha = (1.0 - target_mass) / 2.0
    lo = int(np.searchsorted(cdf, alpha))
    hi = int(np.searchsorted(cdf, 1.0 - alpha))
    lo = min(lo, len(pmf) - 1); hi = min(hi, len(pmf) - 1)
    # expand until contained mass reaches target
    while float(pmf[lo:hi + 1].sum()) < target_mass - 1e-9 and (lo > 0 or hi < len(pmf) - 1):
        if lo > 0:
            lo -= 1
        if hi < len(pmf) - 1:
            hi += 1
    return hi - lo


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def _ece(pred: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, list]:
    """Expected calibration error over pooled (predicted prob, binary outcome)."""
    if len(pred) == 0:
        return float("nan"), []
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(pred, bins) - 1, 0, n_bins - 1)
    ece, table = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = float(pred[m].mean()); acc = float(y[m].mean()); w = int(m.sum())
        ece += (w / len(pred)) * abs(conf - acc)
        table.append({"bin": b, "n": w, "pred": round(conf, 4), "obs": round(acc, 4)})
    return float(ece), table


@dataclass
class StatForecastResult:
    stat: str = ""
    n: int = 0
    bias: float = float("nan")
    bias_se: float = float("nan")
    bias_ok: bool = False
    mae: float = float("nan")
    rmse: float = float("nan")
    crps: float = float("nan")
    log_score: float = float("nan")
    pit_ks_stat: float = float("nan")       # randomized-PIT KS statistic vs Uniform (DIAGNOSTIC)
    pit_ks_p: float = float("nan")          # asymptotic KS p-value (DIAGNOSTIC)
    pit_clustered_dev: float = float("nan")  # game-date block-bootstrap mean PIT-ECDF deviation (GATE)
    pit_clustered_band: float = float("nan")
    pit_mid_ece: float = float("nan")       # DIAGNOSTIC ONLY (not gated)
    coverage: dict = field(default_factory=dict)   # {nominal: {...two-sided...}}
    calib_ece_pooled: float = float("nan")  # DIAGNOSTIC ONLY (dependent thresholds)
    line_level: dict = field(default_factory=dict)  # separate, real-line calibration
    n_dates: int = 0
    support_miss: int = 0
    invalid_pmf: int = 0
    crps_vs_baseline: float = float("nan")   # model - baseline (<=0 is good)
    log_vs_baseline: float = float("nan")
    sharpness_ratio: float = float("nan")    # model width / baseline width (<=1.15 ok)
    # three independent per-stat statuses (a stat can forecast without betting eligibility)
    forecast_allowed: bool = False
    market_comparison_allowed: bool = False
    betting_recommendation_allowed: bool = False
    passed: bool = False                      # alias of forecast_allowed
    reasons: list = field(default_factory=list)


def _clustered_coverage_ci(hits: np.ndarray, clusters: np.ndarray,
                           n_boot: int = 1000, seed: int = 7) -> tuple[float, float]:
    """95% CI for a coverage proportion, resampling whole game-date clusters."""
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    by = {c: hits[clusters == c] for c in uniq}
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([by[c] for c in pick])
        boot.append(vals.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (float(lo), float(hi))


def line_level_threshold_calibration(rows, min_lines: int = 150) -> dict:
    """Threshold calibration at REAL historical market lines — ONE observation per
    (game,player,stat,line), NOT every integer threshold. `rows` needs columns
    p_over (model P(Y>line)) and over_outcome (1 if actual>line, NaN on push).
    Requires at least ``min_lines`` (committed spec = 150) genuine line observations."""
    import pandas as pd  # local import
    df = pd.DataFrame(rows) if not hasattr(rows, "columns") else rows
    if df.empty or "p_over" not in df or "over_outcome" not in df:
        return {"available": False, "reason": "no real market lines"}
    d = df.dropna(subset=["p_over", "over_outcome"])
    if len(d) < min_lines:
        return {"available": False, "reason": f"only {len(d)} lines (<{min_lines})"}
    p = d["p_over"].astype(float).clip(1e-6, 1 - 1e-6).values
    y = d["over_outcome"].astype(float).values
    brier = float(np.mean((p - y) ** 2))
    ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    # logistic calibration slope/intercept via IRLS-free closed-ish fit (numpy)
    try:
        from numpy.polynomial import polynomial as _  # noqa
        logit = np.log(p / (1 - p))
        # simple 1D logistic regression by Newton's method
        b0, b1 = 0.0, 1.0
        for _ in range(50):
            z = b0 + b1 * logit
            mu = 1 / (1 + np.exp(-z))
            w = np.clip(mu * (1 - mu), 1e-6, None)
            g0 = np.sum(mu - y); g1 = np.sum((mu - y) * logit)
            h00 = np.sum(w); h01 = np.sum(w * logit); h11 = np.sum(w * logit * logit)
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-9:
                break
            b0 -= (h11 * g0 - h01 * g1) / det
            b1 -= (h00 * g1 - h01 * g0) / det
        slope, intercept = float(b1), float(b0)
    except Exception:
        slope, intercept = float("nan"), float("nan")
    ece, table = _ece(p, y)
    return {"available": True, "n_lines": int(len(d)), "brier": brier, "log_loss": ll,
            "calibration_slope": slope, "calibration_intercept": intercept,
            "reliability_ece": ece, "reliability": table}


def evaluate_stat(df, *, coverage_levels=(0.5, 0.8, 0.9),
                  material_bias_frac: float = 0.25,
                  under_tol: float = 0.05, over_tol: float = 0.07,
                  ks_p_min: float = 0.01, min_n: int = 300, min_dates: int = 25,
                  lines=None, baseline: dict | None = None,
                  sharpness_max_ratio: float = 1.15, min_real_lines: int = 150,
                  pit_envelope: float = 0.10) -> StatForecastResult:
    """Corrected forecasting gate (P3 Defect 3). df needs pmf_json, actual_outcome and
    (for the seed/cluster) game_id, player_id, stat, model_version, game_date.

    Mathematically valid gates:
      * MATERIAL bias: |bias| <= material_bias_frac * RMSE.
      * DISCRETE calibration: randomized PIT (deterministic seeded V) vs Uniform via KS;
        FAIL if KS p < ks_p_min. Midpoint PIT is diagnostic only — never gated.
      * TWO-SIDED coverage (50/80/90): FAIL if empirical is materially below
        (nominal - under_tol, overconfident) OR materially above (nominal + over_tol,
        over-dispersed) AND the game-date-clustered CI excludes nominal. A nominal-50%
        interval covering ~78% therefore FAILS.
      * Pooled per-threshold ECE is reported as DIAGNOSTIC ONLY (dependent thresholds).
      * Line-level threshold calibration (real lines) reported separately when provided.
    """
    r = StatForecastResult(stat=str(df["stat"].iloc[0]) if len(df) else "", n=int(len(df)))
    if df.empty:
        r.reasons.append("no rows"); return r
    pmfs = [pmf_to_array(p) for p in df["pmf_json"]]
    y_all = df["actual_outcome"].astype(float).values
    gid = df["game_id"].astype(str).values if "game_id" in df else np.array([""] * len(df))
    pid = df["player_id"].astype(str).values if "player_id" in df else np.array([""] * len(df))
    mver = df["model_version"].astype(str).values if "model_version" in df else np.array([""] * len(df))
    gdate = df["game_date"].astype(str).values if "game_date" in df else gid
    means = np.array([float((np.arange(len(p)) * p).sum()) if p.size else np.nan for p in pmfs])
    valid = ~np.isnan(means) & ~np.isnan(y_all)
    idx = np.where(valid)[0]
    pmfs = [pmfs[i] for i in idx]
    y = y_all[idx].astype(int)
    means = means[idx]; gid = gid[idx]; pid = pid[idx]; mver = mver[idx]; gdate = gdate[idx]
    r.n = int(len(y)); r.n_dates = int(len(np.unique(gdate)))
    if r.n == 0:
        r.reasons.append("no valid rows"); return r
    stat = r.stat
    mhash = df["model_hash"].astype(str).values[idx] if "model_hash" in df else mver
    chash = df["calibration_hash"].astype(str).values[idx] if "calibration_hash" in df else mver

    # PMF integrity BEFORE scoring — count invalid PMFs and support misses (an
    # out-of-support outcome is scored via an explicit overflow prob in log_score, so it
    # can never silently become NaN and drop out of the mean).
    for p, yi in zip(pmfs, y):
        ok, reason = validate_pmf(p, int(yi))
        if not ok:
            r.invalid_pmf += 1
        elif reason == "support_miss":
            r.support_miss += 1

    err = means - y
    r.bias = float(err.mean()); r.mae = float(np.abs(err).mean())
    r.rmse = float(np.sqrt((err ** 2).mean()))
    r.bias_se = float(err.std(ddof=1) / math.sqrt(r.n)) if r.n > 1 else float("nan")
    r.bias_ok = bool(abs(r.bias) <= material_bias_frac * r.rmse)
    r.crps = float(np.mean([crps_discrete(p, int(yi)) for p, yi in zip(pmfs, y)]))
    _ls = [log_score(p, int(yi)) for p, yi in zip(pmfs, y)]
    r.log_score = float(np.nanmean(_ls))

    # Randomized, deterministically-seeded PIT vs Uniform (KS). Seed includes model +
    # calibration hashes so the V draw is reproducible and artifact-specific.
    u = np.array([randomized_pit(p, int(yi), f"{g}|{pl}|{stat}|{mh}|{ch}")
                  for p, yi, g, pl, mh, ch in zip(pmfs, y, gid, pid, mhash, chash)])
    _uv = ~np.isnan(u)
    r.pit_ks_stat, r.pit_ks_p = ks_uniform(u[_uv])
    # Clustered PIT (game-date block bootstrap) is the GATE; raw KS above is diagnostic.
    _obs, r.pit_clustered_dev, r.pit_clustered_band = clustered_pit_deviation(u[_uv], gdate[_uv])
    _midpit = np.array([mid_pit(p, int(yi)) for p, yi in zip(pmfs, y)])
    _midpit = _midpit[~np.isnan(_midpit)]
    _hist, _ = np.histogram(_midpit, bins=np.linspace(0, 1, 11))
    r.pit_mid_ece = float(np.abs(_hist / max(len(_midpit), 1) - 0.1).sum() / 2.0)

    # CORRECTED discrete interval calibration (residual gate). For each row compare
    # empirical inclusion to the interval's OWN contained PMF mass; gate on the game-date-
    # clustered CI of the mean residual. Over-dispersion => residual<0; under-dispersion
    # => residual>0; a calibrated (even inclusive-integer) interval => residual~0.
    residual_tol = under_tol  # committed practical tolerance on |mean residual|
    for cl in coverage_levels:
        res = np.zeros(r.n); hits = np.zeros(r.n); masses = np.zeros(r.n); widths = []
        for j, (p, yi) in enumerate(zip(pmfs, y)):
            cm, hit, rr, w = interval_residual(p, int(yi), cl)
            res[j] = rr; hits[j] = hit; masses[j] = cm; widths.append(w)
        mean_res = float(res.mean())
        clo, chi = _clustered_coverage_ci(res, gdate)   # clustered CI of the mean residual
        ci_contains_zero = bool((clo == clo) and (chi == chi) and (clo <= 0.0 <= chi))
        within_tol = abs(mean_res) <= residual_tol
        r.coverage[str(cl)] = {
            "nominal": cl,
            "contained_mass": round(float(masses.mean()), 4),
            "empirical_inclusion": round(float(hits.mean()), 4),
            "residual": round(mean_res, 4),
            "residual_ci_lo": None if clo != clo else round(clo, 4),
            "residual_ci_hi": None if chi != chi else round(chi, 4),
            "ci_contains_zero": ci_contains_zero, "within_tol": bool(within_tol),
            "mean_width": round(float(np.mean(widths)), 3),
            "median_width": round(float(np.median(widths)), 3),
            # residual = empirical_inclusion - contained_mass:
            #   > +tol  => outcomes land inside MORE than claimed => OVER-dispersed
            #   < -tol  => outcomes escape MORE than claimed => UNDER-dispersed (overconfident)
            "materially_over": bool(mean_res > residual_tol),
            "materially_under": bool(mean_res < -residual_tol),
            # PRACTICAL EQUIVALENCE (ROPE): pass when the clustered residual CI lies within
            # the asymmetric tolerance band [-under_tol, +over_tol]. Do NOT additionally
            # require the CI to contain exactly zero (that is an over-strict significance test).
            "interval_equivalent": bool((clo == clo) and (chi == chi)
                                        and (clo >= -under_tol) and (chi <= over_tol)),
            "fail": bool(not ((clo == clo) and (chi == chi)
                              and (clo >= -under_tol) and (chi <= over_tol))),
        }

    # Pooled per-threshold ECE — DIAGNOSTIC ONLY (dependent thresholds; not gated).
    preds, outs = [], []
    for p, yi in zip(pmfs, y):
        cdf = _cdf(p)
        for k in range(1, len(p)):
            preds.append(1.0 - cdf[k - 1]); outs.append(1.0 if yi >= k else 0.0)
    r.calib_ece_pooled, _ = _ece(np.array(preds), np.array(outs))

    # Proper-score & sharpness vs the preregistered seasonal-player empirical baseline.
    if baseline:
        b_crps = baseline.get("crps"); b_log = baseline.get("log_score")
        if b_crps is not None:
            r.crps_vs_baseline = r.crps - float(b_crps)
        if b_log is not None:
            r.log_vs_baseline = r.log_score - float(b_log)
        # Matched-mass sharpness: compare interval widths at the SAME contained mass (0.8).
        b_mw = baseline.get("matched_width_80")
        if b_mw and float(b_mw) > 0:
            m_mw = float(np.mean([matched_mass_width(p, 0.8) for p in pmfs]))
            r.sharpness_ratio = m_mw / float(b_mw)

    # Line-level real-market calibration (separate; >=150 genuine lines required).
    if lines is not None:
        r.line_level = line_level_threshold_calibration(lines, min_lines=min_real_lines)

    # ---- forecast gate (determines forecast_allowed) ----
    if r.invalid_pmf > 0:
        r.reasons.append(f"{r.invalid_pmf} invalid PMFs (integrity failure)")
    if r.support_miss > 0:
        r.reasons.append(f"{r.support_miss} out-of-support outcomes (insufficient upper support)")
    if r.n < min_n:
        r.reasons.append(f"insufficient sample: {r.n} rows (<{min_n})")
    if r.n_dates < min_dates:
        r.reasons.append(f"insufficient coverage: {r.n_dates} game-dates (<{min_dates})")
    if not r.bias_ok:
        r.reasons.append(f"material bias {r.bias:+.2f} > {material_bias_frac:.2f}·RMSE ({r.rmse:.2f})")
    # Clustered randomized-PIT vs the FROZEN practical envelope (raw KS is diagnostic only).
    if not (r.pit_clustered_dev == r.pit_clustered_dev and r.pit_clustered_dev <= pit_envelope):
        r.reasons.append(f"clustered randomized-PIT deviation {r.pit_clustered_dev:.3f} > "
                         f"envelope {pit_envelope} (KS diag p={r.pit_ks_p:.3f})")
    for cl in coverage_levels:
        c = r.coverage[str(cl)]
        if c["fail"]:
            r.reasons.append(f"{int(cl*100)}% interval not practically-equivalent "
                             f"(residual {c['residual']:+.3f}, CI [{c['residual_ci_lo']},{c['residual_ci_hi']}] "
                             f"outside [-{under_tol},{over_tol}]; contained {c['contained_mass']:.2f} "
                             f"vs inclusion {c['empirical_inclusion']:.2f})")
    # Proper scores MUST affect passage (no worse than baseline).
    if baseline:
        if r.crps_vs_baseline == r.crps_vs_baseline and r.crps_vs_baseline > 0:
            r.reasons.append(f"CRPS {r.crps:.3f} worse than baseline by {r.crps_vs_baseline:+.3f}")
        if r.log_vs_baseline == r.log_vs_baseline and r.log_vs_baseline > 0:
            r.reasons.append(f"log score {r.log_score:.3f} worse than baseline by {r.log_vs_baseline:+.3f}")
        if r.sharpness_ratio == r.sharpness_ratio and r.sharpness_ratio > sharpness_max_ratio:
            r.reasons.append(f"80% interval too broad: sharpness ratio {r.sharpness_ratio:.2f} > {sharpness_max_ratio}")
    else:
        r.reasons.append("no preregistered baseline supplied (proper-score/sharpness gate cannot pass)")

    r.forecast_allowed = len(r.reasons) == 0
    r.passed = r.forecast_allowed

    # ---- market/betting eligibility (independent of forecast_allowed) ----
    ll = r.line_level
    if ll.get("available"):
        slope = ll.get("calibration_slope", float("nan"))
        ece = ll.get("reliability_ece", float("nan"))
        slope_ok = (slope == slope and 0.8 <= slope <= 1.25)
        ece_ok = (ece == ece and ece <= 0.06)
        r.market_comparison_allowed = bool(ll.get("n_lines", 0) >= min_real_lines and ece_ok)
        r.betting_recommendation_allowed = bool(r.market_comparison_allowed and slope_ok and r.forecast_allowed)
    return r
