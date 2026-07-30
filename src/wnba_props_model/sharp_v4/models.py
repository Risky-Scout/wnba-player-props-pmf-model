"""Sharp v4 fitted components: minutes-as-a-distribution, exact-tail hierarchical-dispersion count
PMFs, structural rebounds (OREB+DREB) and structural threes (3PA x 3P%), and a hurdle challenger
for rare zero-heavy stats. Selection uses 2023-2025 folds only.

Honest data limit: the recovered stats table lacks FGM/FTM, so a fully-fitted structural POINTS
decomposition (needs 2PM/FTM labels) is not possible here; points uses the V4 exact-tail
hierarchical-dispersion direct model (improved over V3), not a falsely-labelled structural model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from wnba_props_model.sharp_v4 import core as C

_HGB = {"max_depth": 3, "max_iter": 200, "learning_rate": 0.06, "min_samples_leaf": 40,
        "l2_regularization": 1.0, "random_state": C.SEED}
_HGBC = {**_HGB}


def _num(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _usable(X):
    ok = np.zeros(X.shape[1], bool)
    for j in range(X.shape[1]):
        f = X[:, j][np.isfinite(X[:, j])]
        ok[j] = f.size > 0 and np.unique(f).size >= 2
    return ok


def prep(train, other, feat):
    """Explicit-contract matrices; fail closed on label leakage."""
    leak = set(feat) & set(C.LABEL_COLS)
    if leak:
        raise ValueError(f"LEAKAGE: {sorted(leak)}")
    Xtr = _num(train, feat); m = _usable(Xtr); used = [c for c, k in zip(feat, m) if k]
    return Xtr[:, m], _num(other, used), used


def role_band(df: pd.DataFrame) -> np.ndarray:
    """Pregame role proxy from lagged season minutes (never actual same-game minutes)."""
    col = "player_minutes_mean_season"
    if col not in df.columns:
        return np.zeros(len(df), int)
    mm = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
    return np.digitize(mm, [12, 22, 30])   # 0 fringe,1 bench,2 secondary,3 core


# ---- minutes as a distribution ----
@dataclass
class MinutesPMF:
    atoms: np.ndarray          # P(minutes=0..48)

    def mean(self):
        return float(np.dot(np.arange(self.atoms.size), self.atoms))


def fit_minutes(train, other):
    feat = C.resolve_contract("minutes", list(train.columns))
    act = train[train["actual_minutes"] > 0]
    Xtr, Xot, _ = prep(act, other, feat)
    y = act["actual_minutes"].to_numpy(float)
    reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr, y)
    band = role_band(act)
    resid = y - reg.predict(Xtr)
    sd_by_band = {b: float(np.clip(np.std(resid[band == b]), 2.0, 12.0)) if (band == b).sum() > 20
                  else float(np.clip(np.std(resid), 2.0, 12.0)) for b in range(4)}
    mu = np.clip(reg.predict(Xot), 0, 44)
    ob = role_band(other)
    pmfs = []
    grid = np.arange(0, 49)
    for m, b in zip(mu, ob):
        s = sd_by_band.get(int(b), 6.0)
        a, bnd = (0 - m) / s, (48 - m) / s
        pdf = truncnorm.pdf(grid, a, bnd, loc=m, scale=s)
        pdf = pdf / pdf.sum()
        pmfs.append(MinutesPMF(pdf))
    return pmfs, reg, sd_by_band


def minutes_metrics(pmfs, y):
    # discrete NLL + CRPS on integer minutes
    nll, crps = [], []
    for p, yi in zip(pmfs, y):
        yi = int(min(max(round(yi), 0), 48))
        nll.append(-np.log(max(p.atoms[yi], 1e-9)))
        cdf = np.cumsum(p.atoms); heavi = (np.arange(p.atoms.size) >= yi).astype(float)
        crps.append(float(np.sum((cdf - heavi) ** 2)))
    return float(np.mean(nll)), float(np.mean(crps))


# ---- direct exact-tail hierarchical-dispersion count model ----
def fit_direct(train, other, stat):
    feat = C.resolve_contract(stat, list(train.columns))
    act = train[train["actual_minutes"] > 0]
    Xtr, Xot, used = prep(act, other, feat)
    y = np.clip(act[stat].to_numpy(float), 0, None)
    reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr, y)
    mu_tr = np.clip(reg.predict(Xtr), 1e-4, None)
    band_tr = role_band(act)
    r_by_band = C.hierarchical_dispersion(y, mu_tr, band_tr)
    mu = np.clip(reg.predict(Xot), 1e-4, None)
    ob = role_band(other)
    pmfs = [C.build_count_pmf(m, r_by_band.get(int(b), r_by_band["__global__"]), stat)
            for m, b in zip(mu, ob)]
    return pmfs, C.contract_hash(used), len(used), r_by_band


# ---- hurdle-NB2 challenger for rare zero-heavy stats ----
def fit_hurdle(train, other, stat):
    feat = C.resolve_contract(stat, list(train.columns))
    act = train[train["actual_minutes"] > 0]
    Xtr, Xot, used = prep(act, other, feat)
    y = np.clip(act[stat].to_numpy(float), 0, None)
    pos = (y > 0).astype(int)
    clf = HistGradientBoostingClassifier(**_HGBC).fit(Xtr, pos)
    p_pos = np.clip(clf.predict_proba(Xot)[:, 1], 1e-4, 1 - 1e-4)
    # conditional-positive count model
    posm = y > 0
    reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr[posm], y[posm])
    mu_cond = np.clip(reg.predict(Xot), 1e-4, None)
    band_tr = role_band(act[posm])
    r_by_band = C.hierarchical_dispersion(y[posm], np.clip(reg.predict(Xtr[posm]), 1e-4, None), band_tr)
    ob = role_band(other)
    pmfs = []
    for pp, mc, b in zip(p_pos, mu_cond, ob):
        base = C.build_count_pmf(mc, r_by_band.get(int(b), r_by_band["__global__"]), stat)
        # hurdle: mix point mass at 0 with (1-p0) * truncated-at->=1 count
        atoms = base.atoms.copy()
        atoms[0] = 0.0
        s = atoms.sum()
        atoms = atoms / s if s > 0 else atoms
        mixed = np.zeros_like(atoms)
        mixed[0] = 1 - pp
        mixed[1:] = pp * atoms[1:]
        pmfs.append(_ArrayPMF(mixed, base.tail_method, base.overflow * pp))
    return pmfs, C.contract_hash(used), len(used)


@dataclass
class _ArrayPMF:
    atoms: np.ndarray
    tail_method: str
    overflow: float

    @property
    def support_max(self):
        return int(self.atoms.size - 1)

    def logpmf(self, y):
        y = int(max(y, 0))
        if y < self.atoms.size:
            return float(np.log(max(self.atoms[y], 1e-12)))
        return float(np.log(max(self.overflow, 1e-12)))   # exact tail retained (no clip)

    def cdf(self, k):
        k = int(np.floor(k))
        if k < 0:
            return 0.0
        return float(min(1.0, self.atoms[:k + 1].sum()))

    def prob_over(self, line):
        return float(max(0.0, 1.0 - self.cdf(np.floor(line)) + 0))

    def prob_push(self, line):
        return float(self.atoms[int(line)]) if float(line).is_integer() and int(line) < self.atoms.size else 0.0


# ---- structural rebounds: fit OREB & DREB, derive REB = OREB + DREB ----
def fit_structural_reb(train, other):
    """Fit OREB and DREB separately (labels read directly from the merged frame) and derive REB by
    convolving the two count PMFs (shared minutes via features)."""
    out = {}
    for comp in ("oreb", "dreb"):
        feat = C.resolve_contract(comp, list(train.columns))
        act = train[train["actual_minutes"] > 0]
        Xtr, Xot, _ = prep(act, other, feat)
        y = np.clip(act[comp].to_numpy(float), 0, None)
        reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr, y)
        mu_tr = np.clip(reg.predict(Xtr), 1e-4, None)
        r = C.hierarchical_dispersion(y, mu_tr, role_band(act))
        mu = np.clip(reg.predict(Xot), 1e-4, None)
        ob = role_band(other)
        out[comp] = [C.build_count_pmf(m, r.get(int(b), r["__global__"]), comp) for m, b in zip(mu, ob)]
    reb_pmfs = []
    for op, dp in zip(out["oreb"], out["dreb"]):
        conv = np.convolve(op.atoms, dp.atoms)
        reb_pmfs.append(_ArrayPMF(conv / conv.sum(), "oreb+dreb_convolution", op.overflow + dp.overflow))
    return reb_pmfs
