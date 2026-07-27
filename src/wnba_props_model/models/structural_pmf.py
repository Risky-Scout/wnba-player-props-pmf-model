"""Pure structural repair PMFs (owner item 7): opportunity × conversion, no market inputs.

The direct count model regresses the final stat (points/rebounds/threes) in one shot. The
*structural* repair candidates instead decompose the stat into interpretable, physically-coherent
components and convolve them back into a count PMF:

    PTS  = 2*makes_2P + 3*makes_3P + 1*makes_FT
    REB  = OREB + DREB
    FG3M = makes_3P

For each component we model the ATTEMPT (opportunity) volume and the per-attempt CONVERSION
separately:

  * opportunity volume  : a count PMF for attempts, mean = shrunk per-minute attempt rate ×
                          projected active minutes × a shared usage/pace multiplier;
  * conversion          : a per-attempt success probability from hierarchical empirical-Bayes
                          shrinkage (league → role → role×position → team → player, attempts as
                          the exposure/denominator), so thin-sample players shrink to their
                          role/position/team priors instead of overfitting;
  * makes               : Binomial(attempts, p_convert), marginalized over the attempt PMF;
  * shared usage latent : all component attempt means scale by ONE per-game usage multiplier drawn
                          from a small grid, so component volumes co-vary (a high-usage night lifts
                          2PA, 3PA and FTA together) instead of being independent.

Everything here is a PURE function of the model's own historical box-score aggregates and the
minutes projection. NO market-derived input (spread/total/implied-total/odds/consensus/CLV/line)
can enter: :func:`assert_no_market_inputs` fails closed on any forbidden column name, reusing the
``drop_forbidden_market_columns`` resolver.

All estimators are fit on TRAINING rows only (nested rolling-origin discipline); the caller passes
a train-window frame to :meth:`StructuralRepairModel.fit` and a validation frame to
:meth:`StructuralRepairModel.build_active_pmf_matrix`, so no future information enters the
transforms, priors, shrinkage or convolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom as _binom

from wnba_props_model.models.pmf_utils import (
    dispersion_from_moments,
    negbinom_pmf_batch,
    poisson_pmf_batch,
)
from wnba_props_model.models.pure_model_contract import (
    MarketLeakageError,
    drop_forbidden_market_columns,
    is_forbidden_market_field,
)

_EPS = 1e-9

# Candidate IDs registered in the repair ladder (consumed by evaluate_pure_oof).
STRUCTURAL_CANDIDATE_IDS: dict[str, str] = {
    "pts": "S_pts_opportunity_conversion",
    "reb": "S_reb_oreb_dreb_opportunity",
    "fg3m": "S_fg3m_3pa_hurdle_shrunk_conversion",
}
SUPPORTED_PROPS: tuple[str, ...] = ("pts", "reb", "fg3m")


def assert_no_market_inputs(columns: Sequence[str], *, context: str = "structural_pmf") -> None:
    """Fail closed if ANY column the structural model would read is market-derived."""
    _, dropped = drop_forbidden_market_columns(columns)
    if dropped:
        raise MarketLeakageError(
            f"{context}: structural repair may not read market-derived column(s): {sorted(dropped)}")


# ---------------------------------------------------------------------------
# Count-PMF primitives (all pure; each returns a 1-D array summing to ~1)
# ---------------------------------------------------------------------------

def attempts_pmf(mean: float, cap: int, r: float | None) -> np.ndarray:
    """Opportunity (attempt) count PMF for a single mean: NegBinom when ``r`` is finite, else
    Poisson. Returns a length ``cap+1`` array."""
    mean = float(max(mean, _EPS))
    mus = np.array([mean], dtype=float)
    mat = negbinom_pmf_batch(mus, r, cap) if r is not None else poisson_pmf_batch(mus, cap)
    return mat[0]


def hurdle_attempts_pmf(p_zero: float, positive_mean: float, cap: int, r: float | None) -> np.ndarray:
    """Zero-aware attempt PMF: mass ``p_zero`` at 0, the rest a (truncated>0-renormalized) count
    PMF with the given positive mean. Used for 3PA where many players attempt zero threes."""
    p_zero = float(np.clip(p_zero, 0.0, 1.0))
    base = attempts_pmf(positive_mean, cap, r)
    pos = base.copy()
    pos[0] = 0.0
    s = pos.sum()
    if s <= _EPS:
        out = np.zeros(cap + 1)
        out[0] = 1.0
        return out
    pos = pos / s
    out = pos * (1.0 - p_zero)
    out[0] += p_zero
    return out


def binomial_makes_pmf(att_pmf: np.ndarray, p_make: float) -> np.ndarray:
    """PMF of makes = Binomial(A, p) marginalized over the attempt PMF ``att_pmf``.

    makes support is 0..len(att_pmf)-1 (a game with A attempts yields at most A makes).
    """
    p = float(np.clip(p_make, 0.0, 1.0))
    A = att_pmf.size
    k = np.arange(A)
    # B[a, m] = P(makes=m | attempts=a) = Binom(m; a, p); upper-triangular (m<=a).
    B = _binom.pmf(k[np.newaxis, :], k[:, np.newaxis], p)
    B = np.nan_to_num(B, nan=0.0)
    makes = att_pmf @ B  # (A,) · (A, A) -> (A,)
    s = makes.sum()
    return makes / s if s > _EPS else makes


def scale_support(pmf: np.ndarray, k: int) -> np.ndarray:
    """Spread ``pmf`` over the value grid {0, k, 2k, ...} (points from 2P/3P makes)."""
    if k == 1:
        return pmf
    out = np.zeros((pmf.size - 1) * k + 1, dtype=float)
    out[:: k] = pmf
    return out


def convolve_pmfs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convolution of two independent count PMFs (sum of the two variables)."""
    out = np.convolve(a, b)
    s = out.sum()
    return out / s if s > _EPS else out


def truncate_pmf(pmf: np.ndarray, cap: int) -> np.ndarray:
    """Truncate to support 0..cap, piling any tail mass onto ``cap`` (mass-preserving)."""
    if pmf.size <= cap + 1:
        out = np.zeros(cap + 1)
        out[: pmf.size] = pmf
    else:
        out = pmf[: cap + 1].copy()
        out[cap] += pmf[cap + 1:].sum()
    s = out.sum()
    return out / s if s > _EPS else out


# ---------------------------------------------------------------------------
# Hierarchical empirical-Bayes rate (attempts as exposure)
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalRate:
    """Nested empirical-Bayes ratio estimator numerator/denominator, shrinking each level toward
    its coarser parent with ``kappa`` pseudo-observations. ``key_levels`` are CUMULATIVE prefixes
    (coarse → fine), e.g. [(role,), (role,position), (role,position,team_id),
    (role,position,team_id,player_id)]. Attempts (or minutes) are the denominator/exposure."""

    key_levels: list[tuple[str, ...]]
    kappa: float = 20.0
    global_rate: float = 0.0
    _levels: list[dict[tuple, float]] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, num_col: str, den_col: str) -> "HierarchicalRate":
        num = pd.to_numeric(df[num_col], errors="coerce").fillna(0.0).to_numpy(float)
        den = pd.to_numeric(df[den_col], errors="coerce").fillna(0.0).to_numpy(float)
        self.global_rate = float(num.sum() / max(den.sum(), _EPS))
        work = df.copy()
        work["__num__"], work["__den__"] = num, den
        self._levels = []
        prev: dict[tuple, float] = {(): self.global_rate}
        prev_keys: tuple[str, ...] = ()
        for keys in self.key_levels:
            grp = work.groupby(list(keys), dropna=False)[["__num__", "__den__"]].sum()
            level: dict[tuple, float] = {}
            for key_tuple, row in grp.iterrows():
                kt = key_tuple if isinstance(key_tuple, tuple) else (key_tuple,)
                parent = kt[: len(prev_keys)]
                parent_rate = prev.get(parent, self.global_rate)
                level[kt] = float((row["__num__"] + self.kappa * parent_rate)
                                  / (row["__den__"] + self.kappa))
            self._levels.append(level)
            prev, prev_keys = level, keys
        return self

    def predict_one(self, row: Any) -> float:
        """Finest available shrunk rate for a row (falls back up the hierarchy, then global)."""
        for li in range(len(self.key_levels) - 1, -1, -1):
            keys = self.key_levels[li]
            kt = tuple(_get(row, k) for k in keys)
            r = self._levels[li].get(kt)
            if r is not None:
                return float(r)
        return float(self.global_rate)


def _get(row: Any, key: str):
    if isinstance(row, dict):
        return row.get(key)
    return row[key] if key in getattr(row, "index", []) else getattr(row, key, None)


def default_key_levels() -> list[tuple[str, ...]]:
    return [
        ("role_bucket",),
        ("role_bucket", "position"),
        ("role_bucket", "position", "team_id"),
        ("role_bucket", "position", "team_id", "player_id"),
    ]


# ---------------------------------------------------------------------------
# Per-prop structural PMF builders (single row)
# ---------------------------------------------------------------------------

def _usage_grid(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    block = cfg.get("structural_repair", {}) or {}
    grid = np.asarray(block.get("usage_grid", [0.85, 1.0, 1.15]), dtype=float)
    w = np.asarray(block.get("usage_weights", [0.25, 0.50, 0.25]), dtype=float)
    w = w / w.sum()
    return grid, w


def build_points_pmf(minutes: float, rates: dict[str, float], caps: dict[str, int],
                     usage_grid: np.ndarray, usage_weights: np.ndarray, cap: int) -> np.ndarray:
    """Structural points PMF: shared-usage marginalized convolution of 2P/3P/FT component makes."""
    out = np.zeros(cap + 1)
    for u, w in zip(usage_grid, usage_weights):
        a2 = attempts_pmf(rates["fg2a_per_min"] * minutes * u, caps["fg2a"], rates["r_fg2a"])
        a3 = attempts_pmf(rates["fg3a_per_min"] * minutes * u, caps["fg3a"], rates["r_fg3a"])
        aft = attempts_pmf(rates["fta_per_min"] * minutes * u, caps["fta"], rates["r_fta"])
        m2 = scale_support(binomial_makes_pmf(a2, rates["p2"]), 2)
        m3 = scale_support(binomial_makes_pmf(a3, rates["p3"]), 3)
        mft = scale_support(binomial_makes_pmf(aft, rates["pft"]), 1)
        pts = convolve_pmfs(convolve_pmfs(m2, m3), mft)
        out += w * truncate_pmf(pts, cap)
    s = out.sum()
    return out / s if s > _EPS else out


def build_points_pmf_fallback(minutes: float, rates: dict[str, float], caps: dict[str, int],
                              usage_grid: np.ndarray, usage_weights: np.ndarray,
                              cap: int) -> np.ndarray:
    """Fallback structural points PMF when total FG-made / FT-made columns are unavailable.

    Decomposes points into 3P scoring and NON-3P scoring (2P + FT combined), each of which needs
    only columns that always exist:

        3P points     = 3 * Binomial(3PA, p3)                    (3PA from fg3a rate, p3 shrunk)
        non-3P points = a count PMF fit directly on realized (actual_pts - 3*fg3m) per minute

    The two independent components are convolved under the shared usage latent. This lets pts be
    supported on the production feature matrix (which carries actual_pts + fg3m + fg3a but NOT
    total fgm/ftm), instead of abstaining."""
    out = np.zeros(cap + 1)
    for u, w in zip(usage_grid, usage_weights):
        a3 = attempts_pmf(rates["fg3a_per_min"] * minutes * u, caps["fg3a"], rates["r_fg3a"])
        pts3 = scale_support(binomial_makes_pmf(a3, rates["p3"]), 3)
        non3 = attempts_pmf(rates["non3p_pts_per_min"] * minutes * u,
                            caps["non3p_pts"], rates["r_non3p_pts"])
        pts = convolve_pmfs(pts3, non3)
        out += w * truncate_pmf(pts, cap)
    s = out.sum()
    return out / s if s > _EPS else out


def build_reb_pmf(minutes: float, rates: dict[str, float], caps: dict[str, int],
                  usage_grid: np.ndarray, usage_weights: np.ndarray, cap: int) -> np.ndarray:
    """Structural rebounds PMF: OREB + DREB opportunity volumes convolved, shared usage latent."""
    out = np.zeros(cap + 1)
    for u, w in zip(usage_grid, usage_weights):
        o = attempts_pmf(rates["oreb_per_min"] * minutes * u, caps["oreb"], rates["r_oreb"])
        d = attempts_pmf(rates["dreb_per_min"] * minutes * u, caps["dreb"], rates["r_dreb"])
        reb = convolve_pmfs(o, d)
        out += w * truncate_pmf(reb, cap)
    s = out.sum()
    return out / s if s > _EPS else out


def build_fg3m_pmf(minutes: float, rates: dict[str, float], caps: dict[str, int],
                   usage_grid: np.ndarray, usage_weights: np.ndarray, cap: int) -> np.ndarray:
    """Structural FG3M PMF: zero-aware 3PA opportunity × shrunk 3P conversion → Binomial makes.

    Building makes from a Binomial over the attempt distribution yields a SHARPER, better-specified
    makes PMF than a marginal NB fit on makes directly (repairing the FG3M full-PMF sharpness /
    forecast-certification failure)."""
    out = np.zeros(cap + 1)
    for u, w in zip(usage_grid, usage_weights):
        a3 = hurdle_attempts_pmf(rates["fg3a_p_zero"], rates["fg3a_per_min"] * minutes * u,
                                 caps["fg3a"], rates["r_fg3a"])
        makes = binomial_makes_pmf(a3, rates["p3"])
        out += w * truncate_pmf(makes, cap)
    s = out.sum()
    return out / s if s > _EPS else out


# ---------------------------------------------------------------------------
# StructuralRepairModel: train-only fit, validation-frame predict
# ---------------------------------------------------------------------------

_DEFAULT_COLS = {
    "minutes": "actual_minutes", "fgm": "actual_fgm", "fga": "actual_fga",
    "fg3m": "actual_fg3m", "fg3a": "actual_fg3a", "ftm": "actual_ftm", "fta": "actual_fta",
    "oreb": "actual_oreb", "dreb": "actual_dreb", "pts": "actual_pts",
}

# Candidate column names probed at fit time (in order) so the model works whether the feature
# matrix stores realized per-game volumes BARE (fga/fg3a/fta/oreb/dreb) or actual_-prefixed, and
# whatever the settled-outcome prefix is. This is the ROOT-CAUSE fix for the 0aec00c0 all-NULL
# structural output: the default map was actual_*-only, so production's bare volume columns were
# never found and every prop silently abstained. Resolution is train-time only (predictions read
# context keys + minutes, never box-score volumes), so no future information enters.
_COL_ALIASES: dict[str, list[str]] = {
    "minutes": ["actual_minutes", "minutes"],
    "fgm": ["fgm", "actual_fgm"],
    "fga": ["fga", "actual_fga"],
    "fg3m": ["fg3m", "actual_fg3m"],
    "fg3a": ["fg3a", "actual_fg3a"],
    "ftm": ["ftm", "actual_ftm"],
    "fta": ["fta", "actual_fta"],
    "oreb": ["oreb", "actual_oreb"],
    "dreb": ["dreb", "actual_dreb"],
    "pts": ["actual_pts", "pts"],
}


@dataclass
class StructuralRepairModel:
    """Pure, train-only structural repair models for pts/reb/fg3m. Abstains (prop unsupported)
    when the required box-score columns are absent — tracking is never a blocker."""

    cfg: dict[str, Any]
    cols: dict[str, str] = field(default_factory=dict)
    kappa_rate: float = 30.0
    kappa_conv: float = 40.0
    supported: set[str] = field(default_factory=set)
    pts_mode: str | None = None  # "full" (2P/3P/FT) | "fallback" (3P + non-3P) | None
    _rates: dict[str, HierarchicalRate] = field(default_factory=dict)
    _scalars: dict[str, float] = field(default_factory=dict)
    _caps: dict[str, int] = field(default_factory=dict)
    _resolved: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        block = self.cfg.get("structural_repair", {}) or {}
        self.cols = {**_DEFAULT_COLS, **(block.get("columns", {}) or {})}
        self.kappa_rate = float(block.get("kappa_rate", self.kappa_rate))
        self.kappa_conv = float(block.get("kappa_conversion", self.kappa_conv))
        # Fail closed if any configured column name is market-derived.
        assert_no_market_inputs(list(self.cols.values()), context="StructuralRepairModel.cols")

    def _resolve(self, df: pd.DataFrame, key: str) -> str | None:
        """First existing column for a logical field: the configured name (if present), else the
        alias list. Returns None when none are present (that component/prop then abstains)."""
        configured = self.cols.get(key)
        if configured and configured in df.columns:
            return configured
        for cand in _COL_ALIASES.get(key, []):
            if cand in df.columns:
                return cand
        return None

    def _resolve_all(self, df: pd.DataFrame) -> None:
        self._resolved = {k: c for k in _COL_ALIASES if (c := self._resolve(df, k)) is not None}
        # Market-leakage guard on whatever we actually resolved to.
        assert_no_market_inputs(list(self._resolved.values()), context="StructuralRepairModel.resolved")

    def _has(self, *keys: str) -> bool:
        return all(k in self._resolved for k in keys)

    def _col(self, key: str) -> str:
        return self._resolved[key]

    def fit(self, train_wide: pd.DataFrame, cfg: dict[str, Any]) -> "StructuralRepairModel":
        df = train_wide.copy()
        self._resolve_all(df)
        if "minutes" not in self._resolved:
            return self  # cannot form per-minute rates -> everything abstains
        mcol = self._col("minutes")
        df = df[pd.to_numeric(df[mcol], errors="coerce").fillna(0.0) > 0.0].copy()
        if df.empty:
            return self
        if "role_bucket" not in df.columns and "player_minutes_mean_l5" in df.columns:
            from wnba_props_model.features.role_buckets import add_ex_ante_role_bucket  # noqa: PLC0415
            df = add_ex_ante_role_bucket(df, minutes_col="player_minutes_mean_l5")
        for c in ("role_bucket", "position", "team_id", "player_id"):
            if c not in df.columns:
                df[c] = "unknown"
            df[c] = df[c].astype(str)
        levels = default_key_levels()
        m = pd.to_numeric(df[mcol], errors="coerce").fillna(0.0)
        df["_min"] = m

        def _num(key):
            return pd.to_numeric(df[self._col(key)], errors="coerce").fillna(0.0)

        # ---- PTS: prefer full 2P/3P/FT decomposition; else 3P + non-3P fallback ----
        if self._has("fgm", "fga", "fg3m", "fg3a", "ftm", "fta"):
            fga, fgm = _num("fga"), _num("fgm")
            fg3a, fg3m = _num("fg3a"), _num("fg3m")
            fta, ftm = _num("fta"), _num("ftm")
            df["_fg2a"] = np.clip(fga - fg3a, 0.0, None)
            df["_fg2m"] = np.clip(fgm - fg3m, 0.0, None)
            self._fit_rate("fg2a_per_min", df, "_fg2a", "_min", levels, self.kappa_rate)
            self._fit_rate("fg3a_per_min", df.assign(_fg3a=fg3a), "_fg3a", "_min", levels, self.kappa_rate)
            self._fit_rate("fta_per_min", df.assign(_fta=fta), "_fta", "_min", levels, self.kappa_rate)
            self._fit_rate("p2", df, "_fg2m", "_fg2a", levels, self.kappa_conv)
            self._fit_rate("p3", df.assign(_m=fg3m, _a=fg3a), "_m", "_a", levels, self.kappa_conv)
            self._fit_rate("pft", df.assign(_m=ftm, _a=fta), "_m", "_a", levels, self.kappa_conv)
            self._caps["fg2a"] = _cap_for(df["_fg2a"])
            self._caps["fg3a"] = _cap_for(fg3a)
            self._caps["fta"] = _cap_for(fta)
            self._scalars["r_fg2a"] = _disp(df["_fg2a"])
            self._scalars["r_fg3a"] = _disp(fg3a)
            self._scalars["r_fta"] = _disp(fta)
            self.pts_mode = "full"
            self.supported.add("pts")
        elif self._has("pts", "fg3m", "fg3a"):
            # Fallback: 3P points (3PA × p3 binomial × 3) + non-3P points fit directly on
            # realized (actual_pts - 3*fg3m). Needs only actual_pts + fg3m + fg3a + minutes.
            fg3a, fg3m = _num("fg3a"), _num("fg3m")
            pts = _num("pts")
            df["_non3p_pts"] = np.clip(pts - 3.0 * fg3m, 0.0, None)
            self._fit_rate("fg3a_per_min", df.assign(_fg3a=fg3a), "_fg3a", "_min", levels, self.kappa_rate)
            self._fit_rate("p3", df.assign(_m=fg3m, _a=fg3a), "_m", "_a", levels, self.kappa_conv)
            self._fit_rate("non3p_pts_per_min", df, "_non3p_pts", "_min", levels, self.kappa_rate)
            self._caps["fg3a"] = _cap_for(fg3a)
            self._caps["non3p_pts"] = _cap_for(df["_non3p_pts"])
            self._scalars["r_fg3a"] = _disp(fg3a)
            self._scalars["r_non3p_pts"] = _disp(df["_non3p_pts"])
            self.pts_mode = "fallback"
            self.supported.add("pts")

        # ---- REB components (OREB / DREB) ----
        if self._has("oreb", "dreb"):
            oreb, dreb = _num("oreb"), _num("dreb")
            self._fit_rate("oreb_per_min", df.assign(_o=oreb), "_o", "_min", levels, self.kappa_rate)
            self._fit_rate("dreb_per_min", df.assign(_d=dreb), "_d", "_min", levels, self.kappa_rate)
            self._caps["oreb"] = _cap_for(oreb)
            self._caps["dreb"] = _cap_for(dreb)
            self._scalars["r_oreb"] = _disp(oreb)
            self._scalars["r_dreb"] = _disp(dreb)
            self.supported.add("reb")

        # ---- FG3M (shared with PTS 3P rate but adds a hurdle zero-rate) ----
        if self._has("fg3m", "fg3a"):
            fg3a, fg3m = _num("fg3a"), _num("fg3m")
            if "fg3a_per_min" not in self._rates:
                self._fit_rate("fg3a_per_min", df.assign(_fg3a=fg3a), "_fg3a", "_min", levels, self.kappa_rate)
            if "p3" not in self._rates:
                self._fit_rate("p3", df.assign(_m=fg3m, _a=fg3a), "_m", "_a", levels, self.kappa_conv)
            # Hurdle zero-rate P(3PA==0) shrunk by role/position (game-level zero inflation).
            df["_z"] = (fg3a <= 0).astype(float)
            df["_one"] = 1.0
            self._fit_rate("fg3a_p_zero", df, "_z", "_one", levels, self.kappa_conv)
            self._caps.setdefault("fg3a", _cap_for(fg3a))
            self._scalars.setdefault("r_fg3a", _disp(fg3a))
            self.supported.add("fg3m")
        return self

    def _fit_rate(self, name, df, num, den, levels, kappa):
        self._rates[name] = HierarchicalRate(levels, kappa).fit(df, num, den)

    # ---- prediction --------------------------------------------------------

    def _row_rates(self, prop: str, row: Any) -> dict[str, float]:
        r = {k: v for k, v in self._scalars.items()}
        if prop == "pts":
            names = (("fg2a_per_min", "fg3a_per_min", "fta_per_min", "p2", "p3", "pft")
                     if self.pts_mode == "full"
                     else ("fg3a_per_min", "p3", "non3p_pts_per_min"))
            for nm in names:
                r[nm] = self._rates[nm].predict_one(row)
        elif prop == "reb":
            for nm in ("oreb_per_min", "dreb_per_min"):
                r[nm] = self._rates[nm].predict_one(row)
        elif prop == "fg3m":
            for nm in ("fg3a_per_min", "p3", "fg3a_p_zero"):
                r[nm] = self._rates[nm].predict_one(row)
        return r

    def build_active_pmf_matrix(self, prop: str, aligned_wide: pd.DataFrame,
                                min_means: np.ndarray, cap: int) -> np.ndarray | None:
        """Structural ACTIVE (conditional-on-appearance) PMF per row, or None when unsupported."""
        if prop not in self.supported:
            return None
        grid, w = _usage_grid(self.cfg)
        caps = self._caps
        n = len(aligned_wide)
        mat = np.zeros((n, cap + 1))
        ctx = aligned_wide.copy()
        for c in ("role_bucket", "position", "team_id", "player_id"):
            ctx[c] = ctx[c].astype(str) if c in ctx.columns else "unknown"
        rows = ctx.to_dict("records")
        pts_builder = build_points_pmf if self.pts_mode == "full" else build_points_pmf_fallback
        builder = {"pts": pts_builder, "reb": build_reb_pmf, "fg3m": build_fg3m_pmf}[prop]
        for i, row in enumerate(rows):
            minutes = float(max(min_means[i], _EPS))
            rr = self._row_rates(prop, row)
            mat[i] = builder(minutes, rr, caps, grid, w, cap)
        return mat


def _cap_for(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mx = float(s.max()) if len(s) else 0.0
    return int(min(80, max(6, np.ceil(mx) + 4)))


def _disp(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(s) < 5:
        return None
    mean, var = float(s.mean()), float(s.var(ddof=1))
    return dispersion_from_moments(mean, var)
