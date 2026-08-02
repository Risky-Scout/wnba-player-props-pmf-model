"""Sharp v5 models: regulation+overtime minutes PMF and minutes-propagated stat distributions.

Every direct stat PMF is an analytic MIXTURE over the minutes PMF:
    P(Y=y|active,X) = sum_m P(Y=y|minutes=m,X) * P(minutes=m|active,X)
so wider minutes uncertainty widens the stat PMF and its tails (proven by test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.ensemble import HistGradientBoostingRegressor

from wnba_props_model.sharp_v4 import core as V4C
from wnba_props_model.sharp_v5.distribution import CountDistribution, MixtureDistribution

_HGB = {"max_depth": 3, "max_iter": 200, "learning_rate": 0.06, "min_samples_leaf": 40,
        "l2_regularization": 1.0, "random_state": 20260730}
REG_MAX = 40         # regulation minutes support 0..40
OT_MAX = 8           # overtime adds up to 8 minutes


def role_band(df: pd.DataFrame) -> np.ndarray:
    """Pregame role proxy from lagged season minutes (never same-game actual minutes)."""
    col = "player_minutes_mean_season"
    if col not in df.columns:
        return np.zeros(len(df), int)
    mm = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
    return np.digitize(mm, [12, 22, 30])


def _num(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _usable(X):
    ok = np.zeros(X.shape[1], bool)
    for j in range(X.shape[1]):
        f = X[:, j][np.isfinite(X[:, j])]
        ok[j] = f.size > 0 and np.unique(f).size >= 2
    return ok


def prep(train, other, feat):
    leak = set(feat) & set(V4C.LABEL_COLS)
    if leak:
        raise ValueError(f"LEAKAGE: {sorted(leak)}")
    Xtr = _num(train, feat); m = _usable(Xtr); used = [c for c, k in zip(feat, m) if k]
    return Xtr[:, m], _num(other, used), used


def minutes_pmf_rows(train, other, contract: list[str]):
    """Regulation minutes PMF (0..40) + separate overtime mixture. Returns a list of atom arrays
    over 0..(40+OT_MAX). Role-band dispersion; observed minutes never clipped in scoring."""
    act = train[train["actual_minutes"] > 0]
    Xtr, Xot, _ = prep(act, other, contract)
    y = np.clip(act["actual_minutes"].to_numpy(float), 0, None)
    reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr, np.clip(y, 0, REG_MAX))
    band = role_band(act)
    resid = np.clip(y, 0, REG_MAX) - reg.predict(Xtr)
    sd_by = {b: float(np.clip(np.std(resid[band == b]), 2.0, 11.0)) if (band == b).sum() > 20
             else float(np.clip(np.std(resid), 2.0, 11.0)) for b in range(4)}
    # overtime probability (empirical: fraction of active games with minutes>40, by band)
    ot_p_by = {b: float(np.mean(y[band == b] > REG_MAX)) if (band == b).sum() > 20 else float(np.mean(y > REG_MAX))
               for b in range(4)}
    mu = np.clip(reg.predict(Xot), 0, REG_MAX)
    ob = role_band(other)
    grid = np.arange(0, REG_MAX + OT_MAX + 1)
    out = []
    for m, b in zip(mu, ob):
        s = sd_by.get(int(b), 6.0)
        a, bnd = (0 - m) / s, (REG_MAX - m) / s
        reg_pdf = truncnorm.pdf(np.arange(REG_MAX + 1), a, bnd, loc=m, scale=s)
        reg_pdf = reg_pdf / reg_pdf.sum()
        atoms = np.zeros(grid.size)
        p_ot = ot_p_by.get(int(b), 0.05)
        atoms[:REG_MAX + 1] = reg_pdf * (1 - p_ot)
        # overtime: shift a portion of upper regulation mass into 41..48 (extra OT minutes)
        if p_ot > 0:
            ot_tail = np.zeros(grid.size)
            hi = reg_pdf.copy()
            for j in range(1, OT_MAX + 1):
                ot_tail[REG_MAX + j] = hi[max(REG_MAX - 5, 0):].sum() / OT_MAX
            s2 = ot_tail.sum()
            if s2 > 0:
                atoms[REG_MAX + 1:] = (ot_tail[REG_MAX + 1:] / s2) * p_ot
        atoms = atoms / atoms.sum()
        out.append(atoms)
    return out, sd_by, ot_p_by


def stat_mixture_rows(train, other, stat, minutes_atoms_list):
    """Minutes-propagated stat PMF: per-minute rate lambda from features, then mixture over the
    minutes PMF. Component for minutes m is NB2(mean=lambda*m, r)."""
    contract = V4C.resolve_contract(stat, list(train.columns))
    act = train[train["actual_minutes"] > 0]
    Xtr, Xot, used = prep(act, other, contract)
    minutes_tr = np.clip(act["actual_minutes"].to_numpy(float), 1.0, None)
    y = np.clip(act[stat].to_numpy(float), 0, None)
    rate = y / minutes_tr                                   # per-minute rate target
    reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr, rate)
    lam_tr = np.clip(reg.predict(Xtr), 1e-6, None)
    mu_tr = lam_tr * minutes_tr                             # implied conditional mean
    r_by = V4C.hierarchical_dispersion(y, mu_tr, role_band(act))
    lam = np.clip(reg.predict(Xot), 1e-6, None)
    ob = role_band(other)
    r_rows = np.array([r_by.get(int(b), r_by["__global__"]) or np.nan for b in ob])
    return lam, r_rows, V4C.contract_hash(used), len(used), r_by


def build_mixture(lam_i, r, matoms):
    """Convenience: build the MixtureDistribution object for one row (used in live pricing)."""
    idx = np.where(matoms > 1e-4)[0]
    comps = [CountDistribution(max(lam_i * m, 1e-6), None if (r is None or np.isnan(r)) else r) for m in idx]
    return MixtureDistribution(comps, matoms[idx])
