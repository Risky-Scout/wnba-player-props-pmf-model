"""Fitted V6 components: participation, minutes, game environment, direct-stat PMFs.

All estimators live here so production inference never imports sharp_v3/v4/v5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson, truncnorm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from wnba_props_model.sharp_v6.contracts import (
    EMERGENCY_CAP,
    SEED,
    TIER_A,
    contract_hash,
    prep_matrices,
    resolve_contract,
    role_band,
)
from wnba_props_model.sharp_v6.distribution import (
    CountDistribution,
    HurdleDistribution,
    MixtureDistribution,
    TabularDistribution,
)

_HGB = {
    "max_depth": 3, "max_iter": 200, "learning_rate": 0.06, "min_samples_leaf": 40,
    "l2_regularization": 1.0, "random_state": SEED,
}
_HGBC = {**_HGB}
REG_MAX = 40
OT_MAX = 8
TEAM_REG_MINUTES = 200.0
TEAM_Q1_MINUTES = 50.0


def hierarchical_dispersion(y: np.ndarray, mu: np.ndarray, group: np.ndarray, shrink: float = 50.0) -> dict:
    def _r(yy, mm):
        num = float(np.mean((yy - mm) ** 2 - mm))
        den = float(np.mean(mm ** 2))
        if den <= 1e-9 or num <= 1e-9:
            return None
        return float(np.clip(den / num, 0.3, 500.0))

    g_global = _r(y, mu)
    phi_global = (1.0 / g_global) if g_global else 0.0
    out = {"__global__": g_global}
    for gv in np.unique(group):
        m = group == gv
        if m.sum() < 5:
            out[int(gv)] = g_global
            continue
        rg = _r(y[m], mu[m])
        phi_g = (1.0 / rg) if rg else phi_global
        n = int(m.sum())
        phi_shrunk = (n * phi_g + shrink * phi_global) / (n + shrink)
        out[int(gv)] = (1.0 / phi_shrunk) if phi_shrunk > 1e-9 else g_global
    return out


# ---- Participation ----
@dataclass
class ParticipationModel:
    feature_cols: list[str]
    classifier: Any
    calibrator: IsotonicRegression | None
    method: str
    feature_hash: str

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.clip(self.classifier.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 1e-4, 1 - 1e-4)


def fit_participation(train: pd.DataFrame, feat_cols: list[str] | None = None) -> ParticipationModel:
    feat = feat_cols or resolve_contract("participation", list(train.columns))
    y = train["participation"].fillna(False).astype(bool).to_numpy().astype(int)
    X, _, used = prep_matrices(train, train, feat)
    clf = HistGradientBoostingClassifier(**_HGBC).fit(X, y)
    raw = np.clip(clf.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    # chronological half-split isotonic (no silent identity)
    mid = len(train) // 2
    order = np.argsort(pd.to_datetime(train["game_date"]).to_numpy())
    cal_idx, hold_idx = order[:mid], order[mid:]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(raw[cal_idx], y[cal_idx])
    # select isotonic vs identity by Brier on holdout
    p_iso = np.clip(iso.predict(raw[hold_idx]), 1e-4, 1 - 1e-4)
    p_id = raw[hold_idx]
    b_iso = float(np.mean((p_iso - y[hold_idx]) ** 2))
    b_id = float(np.mean((p_id - y[hold_idx]) ** 2))
    use_iso = b_iso <= b_id
    return ParticipationModel(
        feature_cols=used, classifier=clf,
        calibrator=iso if use_iso else None,
        method="isotonic" if use_iso else "identity",
        feature_hash=contract_hash(used),
    )


# ---- Minutes (discrete) + joint team allocation ----
@dataclass
class MinutesModel:
    feature_cols: list[str]
    regressor: Any
    sd_by_band: dict
    ot_p_by_band: dict
    feature_hash: str
    family: str = "role_band_truncnorm_mixture"


def fit_minutes(train: pd.DataFrame) -> MinutesModel:
    feat = resolve_contract("minutes", list(train.columns))
    act = train[train["actual_minutes"] > 0]
    Xtr, _, used = prep_matrices(act, act, feat)
    y = np.clip(act["actual_minutes"].to_numpy(float), 0, None)
    reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr, np.clip(y, 0, REG_MAX))
    band = role_band(act)
    resid = np.clip(y, 0, REG_MAX) - reg.predict(Xtr)
    sd_by = {
        int(b): float(np.clip(np.std(resid[band == b]), 2.0, 11.0)) if (band == b).sum() > 20
        else float(np.clip(np.std(resid), 2.0, 11.0))
        for b in range(4)
    }
    ot_p_by = {
        int(b): float(np.mean(y[band == b] > REG_MAX)) if (band == b).sum() > 20
        else float(np.mean(y > REG_MAX))
        for b in range(4)
    }
    return MinutesModel(used, reg, sd_by, ot_p_by, contract_hash(used))


def _player_minutes_atoms(mu: float, sd: float, p_ot: float) -> np.ndarray:
    grid = np.arange(0, REG_MAX + OT_MAX + 1)
    a, bnd = (0 - mu) / sd, (REG_MAX - mu) / sd
    reg_pdf = truncnorm.pdf(np.arange(REG_MAX + 1), a, bnd, loc=mu, scale=sd)
    reg_pdf = reg_pdf / max(reg_pdf.sum(), 1e-12)
    atoms = np.zeros(grid.size)
    atoms[: REG_MAX + 1] = reg_pdf * (1 - p_ot)
    if p_ot > 0:
        ot_tail = np.zeros(grid.size)
        for j in range(1, OT_MAX + 1):
            ot_tail[REG_MAX + j] = reg_pdf[max(REG_MAX - 5, 0) :].sum() / OT_MAX
        s2 = ot_tail.sum()
        if s2 > 0:
            atoms[REG_MAX + 1 :] = (ot_tail[REG_MAX + 1 :] / s2) * p_ot
    return atoms / atoms.sum()


def _frame_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Align a frame to a frozen feature contract; missing columns become NaN (HGB-native)."""
    out = df.reindex(columns=cols)
    return out.apply(pd.to_numeric, errors="coerce")


def predict_minutes_means(model: MinutesModel, slate: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _frame_features(slate, model.feature_cols).to_numpy(float)
    mu = np.clip(model.regressor.predict(X), 0, REG_MAX)
    bands = role_band(slate)
    sds = np.array([model.sd_by_band.get(int(b), 6.0) for b in bands])
    ot = np.array([model.ot_p_by_band.get(int(b), 0.05) for b in bands])
    return mu, sds, ot


def allocate_team_minutes(mu: np.ndarray, target: float = TEAM_REG_MINUTES) -> np.ndarray:
    """Joint soft allocation so team expected regulation minutes equal `target`."""
    mu = np.clip(np.asarray(mu, float), 0.0, None)
    s = mu.sum()
    if s <= 1e-9:
        return np.full(mu.shape, target / max(len(mu), 1))
    return mu * (target / s)


def minutes_pmf_rows(
    model: MinutesModel,
    slate: pd.DataFrame,
    *,
    reconcile_teams: bool = True,
) -> list[np.ndarray]:
    mu, sds, ot = predict_minutes_means(model, slate)
    if reconcile_teams and "game_id" in slate.columns and "team_id" in slate.columns:
        mu_adj = mu.copy()
        pos = pd.Series(np.arange(len(slate)), index=slate.index)
        for (_, _), idx in slate.groupby(["game_id", "team_id"]).groups.items():
            ii = pos.loc[list(idx)].to_numpy()
            mu_adj[ii] = allocate_team_minutes(mu[ii], TEAM_REG_MINUTES)
        mu = mu_adj
    return [_player_minutes_atoms(float(m), float(s), float(p)) for m, s, p in zip(mu, sds, ot)]


def q1_minutes_pmf_rows(model: MinutesModel, slate: pd.DataFrame) -> list[np.ndarray]:
    """Q1 minutes: scale regulation means to team total 50, support 0..15."""
    mu, sds, _ = predict_minutes_means(model, slate)
    mu_q1 = mu * (TEAM_Q1_MINUTES / TEAM_REG_MINUTES)
    if "game_id" in slate.columns and "team_id" in slate.columns:
        pos = pd.Series(np.arange(len(slate)), index=slate.index)
        for (_, _), idx in slate.groupby(["game_id", "team_id"]).groups.items():
            ii = pos.loc[list(idx)].to_numpy()
            mu_q1[ii] = allocate_team_minutes(mu_q1[ii], TEAM_Q1_MINUTES)
    out = []
    grid = np.arange(0, 16)
    for m, s in zip(mu_q1, sds * 0.35):
        s = max(float(s), 1.0)
        a, bnd = (0 - m) / s, (15 - m) / s
        pdf = truncnorm.pdf(grid, a, bnd, loc=m, scale=s)
        pdf = pdf / max(pdf.sum(), 1e-12)
        out.append(pdf)
    return out


# ---- Game environment ----
@dataclass
class GameEnvironmentModel:
    """Shared pregame game-state means fitted from team box aggregates."""
    feature_cols: list[str]
    targets: dict[str, Any]  # name -> regressor
    residual_cov: dict[str, float]
    feature_hash: str
    status: str = "FITTED"


GAME_ENV_TARGETS = [
    "possessions", "pace", "team_fga", "team_fg3a", "team_fta",
    "team_misses", "reb_opportunities", "assist_env", "tov_env", "p_ot",
]


def _game_env_labels(stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    s = stats.copy()
    s["minutes"] = pd.to_numeric(s.get("minutes", s.get("actual_minutes", 0)), errors="coerce").fillna(0)
    for c in ("fga", "fg3a", "fta", "oreb", "dreb", "ast", "tov", "turnover", "pts", "reb"):
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce").fillna(0)
    if "tov" in s.columns and "turnover" not in s.columns:
        s["turnover"] = s["tov"]
    # team game aggregates
    gcols = ["game_id", "team_id"]
    agg = s.groupby(gcols).agg(
        team_fga=("fga", "sum") if "fga" in s.columns else ("pts", "count"),
        team_fg3a=("fg3a", "sum") if "fg3a" in s.columns else ("pts", "count"),
        team_fta=("fta", "sum") if "fta" in s.columns else ("pts", "count"),
        team_ast=("ast", "sum") if "ast" in s.columns else ("pts", "count"),
        team_tov=("turnover", "sum") if "turnover" in s.columns else ("pts", "count"),
        team_reb=("reb", "sum") if "reb" in s.columns else ("pts", "count"),
        team_pts=("pts", "sum") if "pts" in s.columns else ("pts", "count"),
        team_minutes=("minutes", "sum"),
    ).reset_index()
    if "fga" in s.columns and "fg3a" in s.columns:
        # misses ≈ FGA - (approx FGM from pts rough); use fga - fg3a*0.35 as proxy misses if no fgm
        pass
    agg["team_misses"] = np.clip(agg["team_fga"] * 0.55, 0, None)  # league-ish miss rate prior, refined below
    if "fgm" in s.columns:
        fgm = s.groupby(gcols)["fgm"].sum().reset_index()
        agg = agg.merge(fgm, on=gcols, how="left")
        agg["team_misses"] = np.clip(agg["team_fga"] - agg["fgm"].fillna(0), 0, None)
    # possessions proxy: 0.5 * (FGA + 0.44*FTA - OREB + TOV) * 2 sides approximated via team
    oreb = s.groupby(gcols)["oreb"].sum().reset_index() if "oreb" in s.columns else None
    if oreb is not None:
        agg = agg.merge(oreb, on=gcols, how="left")
        agg["possessions"] = agg["team_fga"] + 0.44 * agg["team_fta"] - agg["oreb"].fillna(0) + agg["team_tov"]
    else:
        agg["possessions"] = agg["team_fga"] + 0.44 * agg["team_fta"] + agg["team_tov"]
    agg["pace"] = agg["possessions"]  # per-team possessions as pace proxy
    agg["reb_opportunities"] = agg["team_misses"] + agg.get("oreb", pd.Series(0, index=agg.index)).fillna(0)
    agg["assist_env"] = agg["team_ast"]
    agg["tov_env"] = agg["team_tov"]
    # OT from minutes > 200 team or game scores - use team minutes > 200
    agg["p_ot"] = (agg["team_minutes"] > 200.5).astype(float)
    return agg


def fit_game_environment(train_features: pd.DataFrame, stats: pd.DataFrame, games: pd.DataFrame) -> GameEnvironmentModel:
    labels = _game_env_labels(stats, games)
    # one row per game-team from features (use first player row for team context features)
    team_feat = train_features.sort_values("game_date").groupby(["game_id", "team_id"], as_index=False).first()
    feat = resolve_contract("game_environment", list(team_feat.columns))
    if not feat:
        feat = [c for c in team_feat.columns if c.startswith(("opp_", "team_", "is_home", "player_rest"))]
        feat = [c for c in feat if c not in ("team_id",)][:40]
    merged = team_feat.merge(labels, on=["game_id", "team_id"], how="inner")
    X, _, used = prep_matrices(merged, merged, feat)
    targets = {}
    resid_var = {}
    for name in GAME_ENV_TARGETS:
        if name not in merged.columns:
            continue
        y = pd.to_numeric(merged[name], errors="coerce").to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 50:
            continue
        reg = HistGradientBoostingRegressor(**_HGB).fit(X[ok], y[ok])
        pred = reg.predict(X[ok])
        targets[name] = reg
        resid_var[name] = float(np.var(y[ok] - pred))
    if not targets:
        raise RuntimeError("game environment fit produced no targets")
    return GameEnvironmentModel(used, targets, resid_var, contract_hash(used), status="FITTED")


def predict_game_environment(model: GameEnvironmentModel, slate: pd.DataFrame) -> pd.DataFrame:
    team = slate.groupby(["game_id", "team_id"], as_index=False).first()
    X = _frame_features(team, model.feature_cols).to_numpy(float)
    out = team[["game_id", "team_id"]].copy()
    for name, reg in model.targets.items():
        out[name] = reg.predict(X)
    # shared OT probability per game = mean of team p_ot
    if "p_ot" in out.columns:
        got = out.groupby("game_id")["p_ot"].transform("mean").clip(0, 1)
        out["p_ot_shared"] = got
    return out


# ---- Direct stat mixture models ----
@dataclass
class StatMixtureModel:
    stat: str
    feature_cols: list[str]
    rate_regressor: Any
    r_by_band: dict
    family: str
    feature_hash: str
    hurdle_clf: Any | None = None


def fit_stat_mixture(train: pd.DataFrame, stat: str, family: str = "nb2") -> StatMixtureModel:
    feat = resolve_contract(stat, list(train.columns))
    act = train[train["actual_minutes"] > 0]
    Xtr, _, used = prep_matrices(act, act, feat)
    minutes_tr = np.clip(act["actual_minutes"].to_numpy(float), 1.0, None)
    y = np.clip(act[stat].to_numpy(float), 0, None)
    rate = y / minutes_tr
    reg = HistGradientBoostingRegressor(**_HGB).fit(Xtr, rate)
    lam_tr = np.clip(reg.predict(Xtr), 1e-6, None)
    mu_tr = lam_tr * minutes_tr
    r_by = hierarchical_dispersion(y, mu_tr, role_band(act))
    hurdle_clf = None
    chosen = family
    if family in ("hurdle_nb2", "auto"):
        pos = (y > 0).astype(int)
        if pos.mean() < 0.85 and pos.sum() > 30:
            hurdle_clf = HistGradientBoostingClassifier(**_HGBC).fit(Xtr, pos)
            chosen = "hurdle_nb2"
        elif family == "auto":
            chosen = "nb2"
    return StatMixtureModel(stat, used, reg, r_by, chosen, contract_hash(used), hurdle_clf)


def mix_atoms(lam: float, r: float | None, matoms: np.ndarray, K: int) -> tuple[np.ndarray, float]:
    idx = np.where(matoms > 1e-4)[0]
    w = matoms[idx]
    means = np.clip(lam * idx, 1e-6, None)
    k = np.arange(K + 1)
    if r is None or (isinstance(r, float) and np.isnan(r)):
        comp = poisson.pmf(k[:, None], means[None, :])
    else:
        p = r / (r + means)
        comp = nbinom.pmf(k[:, None], r, p[None, :])
    atoms = comp @ w
    s = float(atoms.sum())
    if s > 0:
        atoms = atoms / s
    return atoms, float(max(0.0, 1.0 - s))


def predict_stat_atoms(
    model: StatMixtureModel,
    slate: pd.DataFrame,
    minutes_atoms: list[np.ndarray],
) -> list[tuple[np.ndarray, float]]:
    X = _frame_features(slate, model.feature_cols).to_numpy(float)
    lam = np.clip(model.rate_regressor.predict(X), 1e-6, None)
    bands = role_band(slate)
    K = EMERGENCY_CAP.get(model.stat, 60)
    out = []
    p_pos = None
    if model.hurdle_clf is not None:
        p_pos = np.clip(model.hurdle_clf.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    for i in range(len(slate)):
        r = model.r_by_band.get(int(bands[i]), model.r_by_band.get("__global__"))
        a, ovf = mix_atoms(float(lam[i]), None if r is None else float(r), minutes_atoms[i], K)
        if p_pos is not None:
            a = a.copy()
            a[0] = 0.0
            s = a.sum()
            if s > 0:
                a = a / s
            mixed = np.zeros_like(a)
            mixed[0] = 1 - p_pos[i]
            mixed[1:] = p_pos[i] * a[1:]
            a = mixed
            ovf = ovf * float(p_pos[i])
            tot = a.sum() + ovf
            if tot > 0:
                a = a / tot
                ovf = ovf / tot
        out.append((a, ovf))
    return out


# ---- Structural shooting ----
@dataclass
class ShootingModel:
    attempts: dict[str, StatMixtureModel]  # fg2a, fg3a, fta
    makes_rate: dict[str, Any]  # conditional make probability models
    feature_hash: str
    status: str = "FITTED"
    n_train: int = 0


def fit_shooting(train: pd.DataFrame, shoot_labels: pd.DataFrame) -> ShootingModel:
    """Fit attempt volumes + conditional make rates; identities enforced at draw/PMF build."""
    m = train.merge(shoot_labels, on=["game_id", "player_id"], how="inner", suffixes=("", "_sh"))
    if len(m) < 200:
        raise RuntimeError(f"insufficient shooting labels for fit: {len(m)}")
    m["fg2a"] = np.clip(m["fga"] - m["fg3a"], 0, None)
    m["fg2m"] = np.clip(m["fgm"] - m["fg3m"], 0, None)
    m["actual_minutes"] = pd.to_numeric(m["actual_minutes"], errors="coerce").fillna(0)
    attempts = {}
    for stat in ("fg2a", "fg3a", "fta"):
        # temporarily expose label column named as stat for fit_stat_mixture
        tmp = m.copy()
        tmp[stat] = pd.to_numeric(tmp[stat], errors="coerce").fillna(0)
        attempts[stat] = fit_stat_mixture(tmp, stat, family="nb2")
    makes = {}
    for attempt, make in (("fg2a", "fg2m"), ("fg3a", "fg3m"), ("fta", "ftm")):
        act = m[(m["actual_minutes"] > 0) & (m[attempt] > 0)]
        feat = resolve_contract("pts", list(act.columns))
        X, _, used = prep_matrices(act, act, feat)
        rate = (act[make] / act[attempt]).clip(0, 1).to_numpy(float)
        makes[make] = {
            "regressor": HistGradientBoostingRegressor(**_HGB).fit(X, rate),
            "feature_cols": used,
            "mean_rate": float(rate.mean()),
        }
    return ShootingModel(attempts, makes, contract_hash(sorted(attempts["fg2a"].feature_cols)), n_train=len(m))


def structural_points_pmf(
    shoot: ShootingModel,
    slate: pd.DataFrame,
    minutes_atoms: list[np.ndarray],
    n_sims: int = 400,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Monte Carlo structural PTS PMF preserving PTS = 2*2PM + 3*3PM + FTM."""
    rng = rng or np.random.default_rng(SEED)
    att = {k: predict_stat_atoms(v, slate, minutes_atoms) for k, v in shoot.attempts.items()}
    make_p = {}
    for make, spec in shoot.makes_rate.items():
        X = _frame_features(slate, spec["feature_cols"]).to_numpy(float)
        make_p[make] = np.clip(spec["regressor"].predict(X), 0.05, 0.95)
    K = EMERGENCY_CAP["pts"]
    out = []
    for i in range(len(slate)):
        samples = []
        for _ in range(n_sims):
            def draw(atoms_ovf):
                a, ovf = atoms_ovf
                p = np.concatenate([a, [ovf]])
                p = p / p.sum()
                k = rng.choice(len(p), p=p)
                return int(k) if k < len(a) else int(len(a))
            a2 = draw(att["fg2a"][i]); a3 = draw(att["fg3a"][i]); aft = draw(att["fta"][i])
            m2 = rng.binomial(a2, float(make_p["fg2m"][i]))
            m3 = rng.binomial(a3, float(make_p["fg3m"][i]))
            mft = rng.binomial(aft, float(make_p["ftm"][i]))
            samples.append(2 * m2 + 3 * m3 + mft)
        samples = np.asarray(samples, int)
        atoms = np.bincount(np.clip(samples, 0, K), minlength=K + 1).astype(float)
        atoms = atoms / atoms.sum()
        ovf = float((samples > K).mean())
        if ovf > 0:
            atoms = atoms * (1 - ovf)
        out.append((atoms, ovf))
    return out


# ---- Rebounds structural ----
@dataclass
class ReboundModel:
    oreb: StatMixtureModel
    dreb: StatMixtureModel
    status: str = "FITTED"


def fit_rebounds(train: pd.DataFrame, shoot_labels: pd.DataFrame | None = None) -> ReboundModel:
    df = train.copy()
    if shoot_labels is not None and {"oreb", "dreb"}.issubset(shoot_labels.columns):
        df = df.drop(columns=[c for c in ("oreb", "dreb") if c in df.columns], errors="ignore")
        df = df.merge(shoot_labels[["game_id", "player_id", "oreb", "dreb"]], on=["game_id", "player_id"], how="left")
    if "oreb" not in df.columns or df["oreb"].isna().all():
        # fallback: split reb 30/70
        df["oreb"] = (pd.to_numeric(df["reb"], errors="coerce").fillna(0) * 0.28).round()
        df["dreb"] = np.clip(pd.to_numeric(df["reb"], errors="coerce").fillna(0) - df["oreb"], 0, None)
    else:
        # fill unmatched join rows from total reb split
        reb = pd.to_numeric(df["reb"], errors="coerce").fillna(0)
        o = pd.to_numeric(df["oreb"], errors="coerce")
        d = pd.to_numeric(df["dreb"], errors="coerce")
        miss = o.isna() | d.isna()
        df.loc[miss, "oreb"] = (reb[miss] * 0.28).round()
        df.loc[miss, "dreb"] = np.clip(reb[miss] - df.loc[miss, "oreb"], 0, None)
        df["oreb"] = pd.to_numeric(df["oreb"], errors="coerce").fillna(0)
        df["dreb"] = pd.to_numeric(df["dreb"], errors="coerce").fillna(0)
    return ReboundModel(fit_stat_mixture(df, "oreb"), fit_stat_mixture(df, "dreb"))


def structural_reb_pmf(model: ReboundModel, slate: pd.DataFrame, minutes_atoms: list[np.ndarray], n_sims: int = 400, rng=None):
    rng = rng or np.random.default_rng(SEED)
    o = predict_stat_atoms(model.oreb, slate, minutes_atoms)
    d = predict_stat_atoms(model.dreb, slate, minutes_atoms)
    K = EMERGENCY_CAP["reb"]
    out = []
    for i in range(len(slate)):
        def draw(a_ovf):
            a, ovf = a_ovf
            p = np.concatenate([a, [max(ovf, 0)]])
            p = p / p.sum()
            k = rng.choice(len(p), p=p)
            return int(k) if k < len(a) else int(len(a))
        samples = np.array([draw(o[i]) + draw(d[i]) for _ in range(n_sims)])
        atoms = np.bincount(np.clip(samples, 0, K), minlength=K + 1).astype(float)
        atoms /= atoms.sum()
        ovf = float((samples > K).mean())
        if ovf > 0:
            atoms = atoms * (1 - ovf)
        out.append((atoms, ovf))
    return out


# ---- Calibration ----
@dataclass
class StatCalibrator:
    stat: str
    method: str  # identity | monotone_pit
    iso: IsotonicRegression | None
    pit_ks_before: float
    pit_ks_after: float


def fit_calibrator(stat: str, atoms_list: list[np.ndarray], y: np.ndarray, rng: np.random.Generator) -> StatCalibrator:
    """Monotone randomized-PIT recalibration; identity only when selected by holdout KS."""
    pits = []
    for a, yi in zip(atoms_list, y):
        yi = int(yi)
        lo = float(a[:yi].sum()) if yi > 0 else 0.0
        p = float(a[yi]) if yi < a.size else max(1e-12, 1 - float(a.sum()))
        pits.append(lo + rng.random() * p)
    pits = np.asarray(pits)
    order = np.argsort(pits)
    mid = len(pits) // 2
    train_i, hold_i = order[:mid], order[mid:]
    # map empirical PIT CDF -> uniform via isotonic on sorted PIT vs ranks
    u_train = pits[train_i]
    ranks = (np.arange(1, len(u_train) + 1)) / (len(u_train) + 1)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(np.sort(u_train), ranks)
    def ks(u):
        u = np.sort(u)
        n = len(u)
        return float(np.max(np.abs(np.arange(1, n + 1) / n - u))) if n else float("nan")
    ks_before = ks(pits[hold_i])
    ks_after = ks(iso.predict(pits[hold_i]))
    use = ks_after <= ks_before
    return StatCalibrator(stat, "monotone_pit" if use else "identity", iso if use else None, ks_before, ks_after if use else ks_before)


def apply_calibrator(cal: StatCalibrator, atoms: np.ndarray, overflow: float) -> tuple[np.ndarray, float]:
    """Apply CDF isotonic recalibration to a discrete PMF while preserving mass/monotonicity."""
    if cal.method == "identity" or cal.iso is None:
        return atoms, overflow
    cdf = np.cumsum(atoms)
    # recalibrate CDF values then differenced back to atoms
    cdf_c = np.clip(cal.iso.predict(np.clip(cdf, 1e-6, 1 - 1e-6)), 0, 1)
    # enforce monotone
    for i in range(1, len(cdf_c)):
        cdf_c[i] = max(cdf_c[i], cdf_c[i - 1])
    new = np.diff(cdf_c, prepend=0.0)
    new = np.clip(new, 0, None)
    mass = float(new.sum())
    target = float(max(0.0, 1.0 - overflow))
    if mass > 1e-12:
        new = new * (target / mass)
    else:
        new = atoms
    return new, overflow


# ---- Dependence (Gaussian copula on PIT residuals) ----
@dataclass
class DependenceModel:
    stats: list[str]
    corr: np.ndarray
    method: str = "gaussian_copula_psd"
    status: str = "FITTED"


def _psd_project(C: np.ndarray) -> np.ndarray:
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C2))
    C2 = C2 / np.outer(d, d)
    np.fill_diagonal(C2, 1.0)
    return C2


def fit_dependence(pit_by_stat: dict[str, np.ndarray], stats: list[str] | None = None) -> DependenceModel:
    stats = stats or [s for s in TIER_A if s in pit_by_stat]
    mats = [pit_by_stat[s] for s in stats]
    n = min(len(m) for m in mats)
    X = np.column_stack([m[:n] for m in mats])
    # gaussianize via inverse normal approx
    from scipy.stats import norm
    Z = norm.ppf(np.clip(X, 1e-6, 1 - 1e-6))
    C = np.corrcoef(Z, rowvar=False)
    C = _psd_project(np.asarray(C, float))
    return DependenceModel(stats, C)


@dataclass
class ModelBundle:
    """Frozen production bundle contents (in-memory)."""
    participation: ParticipationModel
    minutes: MinutesModel
    game_environment: GameEnvironmentModel
    stats: dict[str, StatMixtureModel]
    shooting: ShootingModel | None
    rebounds: ReboundModel | None
    calibrators: dict[str, StatCalibrator]
    dependence: DependenceModel | None
    contracts: dict
    meta: dict = field(default_factory=dict)
    selected_family: dict[str, str] = field(default_factory=dict)
