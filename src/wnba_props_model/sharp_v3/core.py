"""Sharp v3 modeling core: fail-closed loading, stat feature contracts, chronological folds,
active-conditional count PMFs, no-vig market conversion, and proper-scoring metrics.

Sportsbook data never enters the PURE feature path here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wnba_props_model.opportunity.pmf_builders import poisson_or_nbinom_pmf

REPO = Path(__file__).resolve().parents[3]
FEATURES = REPO / "data/recovered_v2/modeling/wnba_pregame_features_t12.parquet"
TARGETS = REPO / "data/recovered_v2/modeling/wnba_player_targets.parquet"
MANIFEST = REPO / "artifacts/sharp_v3/PRIVATE_INPUT_MANIFEST.json"

ID_COLS = ["game_id", "player_id", "game_date", "season", "team_id", "opponent_team_id"]
TIER_A = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
# Same-game LABEL columns (from the targets table). These are OUTCOMES and must NEVER enter any
# estimator matrix. Excluded from every feature contract to prevent same-game target leakage.
LABEL_COLS = ["participation", "actual_minutes", "pts", "reb", "ast", "fg3m", "stl", "blk",
              "turnover", "stocks", "pts_ast", "pts_reb", "reb_ast", "pts_reb_ast"]
HARD_CAP = {"pts": 70, "reb": 35, "ast": 25, "fg3m": 15, "stl": 12, "blk": 12, "turnover": 15}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_verified() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features + targets; FAIL CLOSED if hashes drift from the private manifest."""
    if not MANIFEST.exists():
        raise FileNotFoundError("PRIVATE_INPUT_MANIFEST.json missing — run sharp_v3_preserve_inputs.py")
    man = json.loads(MANIFEST.read_text())["inputs"]
    for key, path in (("pregame_features_t12", FEATURES), ("player_targets", TARGETS)):
        want = man[key]["sha256"]
        got = _sha256(path)
        if got != want:
            raise ValueError(f"HASH MISMATCH for {key}: manifest={want[:12]} disk={got[:12]} (fail-closed)")
    f = pd.read_parquet(FEATURES)
    t = pd.read_parquet(TARGETS)
    f["game_date"] = pd.to_datetime(f["game_date"])
    df = f.merge(t, on=["game_id", "player_id"], suffixes=("", "_tgt"))
    return f, df


# ---- stat-specific compact feature contracts (domain core) ----
_AVAIL_MIN = ("minutes", "rest", "cumulative", "days_since", "game_number", "usage",
              "form", "season_zscore", "participation")


def stat_feature_contract(stat: str, all_cols: list[str]) -> list[str]:
    """Domain-core contract: same-stat lagged features + availability/minutes/usage/recency.
    Excludes ids, targets, and any same-game actuals (features are pregame T-1.2 by construction)."""
    cols = []
    forbidden = set(ID_COLS) | set(LABEL_COLS)
    for c in all_cols:
        lc = c.lower()
        if c in forbidden or c.endswith("_tgt"):
            continue                       # never let a same-game outcome column be a feature
        if stat in lc or any(k in lc for k in _AVAIL_MIN):
            cols.append(c)
    # de-dup, stable order
    seen = set(); out = []
    for c in cols:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def feature_schema_hash(cols: list[str]) -> str:
    return hashlib.sha256("\n".join(cols).encode()).hexdigest()[:16]


# ---- chronological folds ----
@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str          # exclusive upper bound for training (expanding window)
    eval_start: str
    eval_end: str
    is_holdout: bool = False


DEV_FOLDS = [
    Fold("dev_2024_h1", "2024-05-01", "2024-05-01", "2024-06-30"),
    Fold("dev_2024_h2", "2024-07-01", "2024-07-01", "2024-09-30"),
    Fold("dev_2025_h1", "2025-05-01", "2025-05-01", "2025-06-30"),
    Fold("dev_2025_h2", "2025-07-01", "2025-07-01", "2025-10-31"),
]
HOLDOUT = Fold("holdout_2026", "2026-01-01", "2026-01-01", "2026-12-31", is_holdout=True)


def split(df: pd.DataFrame, fold: Fold) -> tuple[pd.Index, pd.Index]:
    d = df["game_date"]
    train = df.index[d < pd.Timestamp(fold.train_end)]
    ev = df.index[(d >= pd.Timestamp(fold.eval_start)) & (d <= pd.Timestamp(fold.eval_end))]
    return train, ev


# ---- active-conditional count PMF (NB2 with conditional-residual dispersion) ----
def residual_dispersion_r(y: np.ndarray, mu: np.ndarray) -> float | None:
    y = np.asarray(y, float); mu = np.asarray(mu, float)
    num = float(np.mean((y - mu) ** 2 - mu)); den = float(np.mean(mu ** 2))
    if den <= 1e-9 or num <= 1e-9:
        return None
    return float(np.clip(den / num, 0.3, 500.0))


def count_pmf(mean: float, r: float | None, cap: int) -> np.ndarray:
    for c in (cap, cap * 2, cap * 3, 300):
        try:
            return poisson_or_nbinom_pmf(max(float(mean), 1e-6), r, maximum_cap=int(c))
        except ValueError:
            continue
    return poisson_or_nbinom_pmf(max(float(mean), 1e-6), None, maximum_cap=300)


# ---- metrics ----
def nll(pmf_rows: list[np.ndarray], y: np.ndarray) -> float:
    ll = []
    for p, yi in zip(pmf_rows, y):
        yi = int(min(max(yi, 0), len(p) - 1))
        ll.append(np.log(max(p[yi], 1e-12)))
    return float(-np.mean(ll))


def crps_discrete(pmf_rows: list[np.ndarray], y: np.ndarray) -> float:
    vals = []
    for p, yi in zip(pmf_rows, y):
        cdf = np.cumsum(p)
        k = np.arange(len(p))
        heavi = (k >= yi).astype(float)
        vals.append(float(np.sum((cdf - heavi) ** 2)))
    return float(np.mean(vals))


def pit_values(pmf_rows: list[np.ndarray], y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomized PIT for discrete distributions."""
    out = []
    for p, yi in zip(pmf_rows, y):
        yi = int(min(max(yi, 0), len(p) - 1))
        cdf = np.cumsum(p)
        lo = cdf[yi - 1] if yi > 0 else 0.0
        out.append(lo + rng.random() * p[yi])
    return np.asarray(out)


def prob_over(pmf: np.ndarray, line: float) -> float:
    k = np.arange(len(pmf))
    return float(pmf[k > line].sum())


def prob_push(pmf: np.ndarray, line: float) -> float:
    return float(pmf[int(line)]) if float(line).is_integer() and int(line) < len(pmf) else 0.0


def ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    probs = np.asarray(probs); outcomes = np.asarray(outcomes)
    if len(probs) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(probs)) * abs(probs[m].mean() - outcomes[m].mean())
    return float(tot)


# ---- no-vig market conversion ----
def american_to_prob(a: float) -> float:
    a = float(a)
    if not np.isfinite(a) or (-100 < a < 100):
        return float("nan")
    return (100.0 / (a + 100.0)) if a > 0 else (abs(a) / (abs(a) + 100.0))


def no_vig_over(over_odds: float, under_odds: float) -> float:
    po = american_to_prob(over_odds); pu = american_to_prob(under_odds)
    if not (np.isfinite(po) and np.isfinite(pu)) or (po + pu) <= 0:
        return float("nan")
    return float(po / (po + pu))
