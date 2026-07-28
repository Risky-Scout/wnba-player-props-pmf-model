"""Full PTS decomposition (owner directive section 7): pure PMF path + bundle integration.

Proves:
  * verified reconstruction identity holds on the persisted labels (2*FG2M + 3*FG3M + FTM == PTS);
  * the convolved PTS PMF is normalized/nonneg and its mean == 2*E[2PM] + 3*E[3PM] + E[FTM];
  * the bundle builds a full-decomposition PTS PMF for reconstruction-grounded players and FALLS BACK
    to the diagnostic proxy for players without reconstruction labels (and entirely when labels absent).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wnba_props_model.opportunity.bundle import (
    CANDIDATE_PTS_DECOMP,
    OpportunityModelBundleV2,
)
from wnba_props_model.opportunity.pmf_builders import pmf_mean
from wnba_props_model.opportunity.pts_decomposition import (
    build_pts_pmf_for_minutes,
    build_pts_pmf_over_minutes,
    stretch_pmf,
)

_LABELS = Path("data/processed/pts_conversion_labels.parquet")
_VALIDATION = Path("artifacts/opportunity_v2/PTS_LABEL_VALIDATION.json")


# --- pure PMF primitives ----------------------------------------------------
def test_stretch_places_mass_at_multiples():
    out = stretch_pmf(np.array([0.5, 0.3, 0.2]), 3)
    assert out.size == 7  # (3-1)*3 + 1
    assert np.isclose(out[0], 0.5) and np.isclose(out[3], 0.3) and np.isclose(out[6], 0.2)
    assert np.isclose(out.sum(), 1.0)


def test_pts_pmf_normalized_and_nonneg():
    pmf = build_pts_pmf_for_minutes(
        30.0, rate_2pa=0.20, r_2pa=None, alpha2=50, beta2=50,
        rate_3pa=0.12, r_3pa=None, alpha3=35, beta3=65,
        rate_fta=0.10, r_fta=None, alpha_ft=80, beta_ft=20)
    assert np.all(pmf >= 0)
    assert np.isclose(pmf.sum(), 1.0, atol=1e-8)


def test_pts_pmf_mean_matches_points_identity():
    # E[PTS] = 2*E[2PM] + 3*E[3PM] + E[FTM]; E[makes] = E[attempts]*conversion.
    m = 30.0
    r2a, p2 = 0.20, 0.50   # E[2PA]=6.0 -> E[2PM]=3.0
    r3a, p3 = 0.12, 0.35   # E[3PA]=3.6 -> E[3PM]=1.26
    rfa, pft = 0.10, 0.80  # E[FTA]=3.0 -> E[FTM]=2.4
    expected = 2 * (r2a * m * p2) + 3 * (r3a * m * p3) + (rfa * m * pft)
    pmf = build_pts_pmf_for_minutes(
        m, rate_2pa=r2a, r_2pa=None, alpha2=p2 * 1e6, beta2=(1 - p2) * 1e6,
        rate_3pa=r3a, r_3pa=None, alpha3=p3 * 1e6, beta3=(1 - p3) * 1e6,
        rate_fta=rfa, r_fta=None, alpha_ft=pft * 1e6, beta_ft=(1 - pft) * 1e6)
    assert pmf_mean(pmf) == pytest.approx(expected, rel=1e-3)


def test_pts_pmf_over_minutes_averages():
    mins = np.array([10.0, 30.0])
    w = np.array([0.5, 0.5])
    kw = dict(rate_2pa=0.2, r_2pa=None, alpha2=50, beta2=50,
              rate_3pa=0.12, r_3pa=None, alpha3=35, beta3=65,
              rate_fta=0.1, r_fta=None, alpha_ft=80, beta_ft=20)
    avg = build_pts_pmf_over_minutes(mins, w, **kw)
    lo = build_pts_pmf_for_minutes(10.0, **kw)
    hi = build_pts_pmf_for_minutes(30.0, **kw)
    assert pmf_mean(avg) == pytest.approx(0.5 * pmf_mean(lo) + 0.5 * pmf_mean(hi), rel=1e-6)


# --- verified reconstruction identity on persisted labels -------------------
@pytest.mark.skipif(not _LABELS.exists(), reason="pts_conversion_labels.parquet not present")
def test_reconstruction_identity_holds_on_persisted_labels():
    lab = pd.read_parquet(_LABELS)
    recon_pts = 2 * lab["FG2M"] + 3 * lab["FG3M"] + lab["FTM"]
    assert (recon_pts == lab["pts"]).mean() == pytest.approx(1.0, abs=1e-9)
    assert (lab["FG2M"] >= 0).all() and (lab["FG2A"] >= lab["FG2M"]).all()
    assert (lab["FTM"] >= 0).all() and (lab["FTA"] >= lab["FTM"]).all()
    assert (lab["FGM"] >= lab["FG3M"]).all()


@pytest.mark.skipif(not _LABELS.exists(), reason="pts_conversion_labels.parquet not present")
def test_persisted_labels_carry_inferred_provenance():
    lab = pd.read_parquet(_LABELS)
    for col in ("reconstruction_method", "label_status", "confidence",
                "rounding_residual", "source_tag"):
        assert col in lab.columns, f"missing provenance column {col}"
    # only validated rows may feed conversion fits
    assert (lab["label_status"] == "validated").all()


@pytest.mark.skipif(not _VALIDATION.exists(), reason="PTS_LABEL_VALIDATION.json not present")
def test_pts_label_validation_reports_contested_cross_check():
    import json
    rep = json.loads(_VALIDATION.read_text())
    assert "contested_cross_check" in rep
    assert "source_hashes" in rep and rep["source_hashes"]
    assert "label_status_counts" in rep
    # inferred-label honesty is documented
    assert "INFERRED" in rep["honest_note"]


# --- bundle integration + proxy fallback ------------------------------------
def _frame(n_games=70, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    teams = [(10, 20), (30, 40)]
    for gi in range(n_games):
        home, away = teams[gi % len(teams)]
        gid = 2000 + gi
        date = pd.Timestamp("2026-05-01", tz="UTC") + pd.Timedelta(days=gi)
        for team, opp in ((home, away), (away, home)):
            for pj in range(8):
                pid = team * 100 + pj
                mins = float(np.clip(rng.normal(24, 6), 3, 38))
                fg3a = int(rng.poisson(max(0.12 * mins, 0.1)))
                fg3m = int(rng.binomial(fg3a, 0.35)) if fg3a > 0 else 0
                fg2a = int(rng.poisson(max(0.25 * mins, 0.1)))
                fg2m = int(rng.binomial(fg2a, 0.48)) if fg2a > 0 else 0
                fta = int(rng.poisson(max(0.10 * mins, 0.1)))
                ftm = int(rng.binomial(fta, 0.80)) if fta > 0 else 0
                pts = 2 * fg2m + 3 * fg3m + ftm
                did_play = pj < 7 or rng.uniform() < 0.6
                rows.append(dict(
                    game_id=gid, team_id=team, opponent_team_id=opp, player_id=pid,
                    game_date=date, prediction_cutoff_utc=str(date - pd.Timedelta(minutes=90)),
                    did_play=bool(did_play), minutes=mins if did_play else 0.0,
                    fga=fg2a + fg3a, fg3a=fg3a, fg3m=fg3m, fg2a=fg2a, fg2m=fg2m,
                    fta=fta, ftm=ftm, tov=int(rng.poisson(2)), pts=pts,
                    player_fg3a_per_min_ewma=max(rng.normal(0.12, 0.03), 0.0),
                    player_minutes_ewma=mins,
                    player_pts_per_min_ewma=max(rng.normal(0.5, 0.1), 0.05),
                    player_fga_per_min_ewma=max(rng.normal(0.37, 0.08), 0.05),
                    team_fg3a_ewma=float(rng.normal(24, 3)),
                    player_active_rate_ewma=float(np.clip(rng.normal(0.9, 0.1), 0, 1)),
                    position=["G", "F", "C"][pj % 3],
                    role_bucket=["starter", "bench"][0 if pj < 5 else 1],
                ))
    return pd.DataFrame(rows)


def _recon_from_frame(df, eligible_players):
    """Synthetic reconstruction labels (identity-consistent) for a subset of players only."""
    d = df[df["player_id"].isin(eligible_players) & df["did_play"]]
    return pd.DataFrame({
        "game_id": d["game_id"].to_numpy(), "player_id": d["player_id"].to_numpy(),
        "FG2M": d["fg2m"].to_numpy(), "FG2A": d["fg2a"].to_numpy(),
        "FG3M": d["fg3m"].to_numpy(), "FG3A": d["fg3a"].to_numpy(),
        "FTM": d["ftm"].to_numpy(), "FTA": d["fta"].to_numpy(),
    })


def test_full_decomposition_used_for_grounded_players_and_proxy_otherwise():
    df = _frame()
    eligible = sorted(df["player_id"].unique())[:8]
    recon = _recon_from_frame(df, eligible)
    bundle = OpportunityModelBundleV2({"minutes": {"deterministic_samples": 7}}).fit(
        df, df, pts_recon_labels=recon)
    assert bundle._pts_decomp_available is True

    val = df[df["game_id"] < 2004].copy()
    pred = bundle.predict_active_pmfs(val, None, ["pts"], candidate=CANDIDATE_PTS_DECOMP)
    assert (pred["candidate_id"] == CANDIDATE_PTS_DECOMP).all()
    # PMFs valid
    for js in pred["active_pmf_json"]:
        import json
        arr = np.array(json.loads(js))
        assert np.all(arr >= -1e-12) and np.isclose(arr.sum(), 1.0, atol=1e-6)
    # grounded players -> full decomposition; others -> proxy
    grounded = pred[pred["player_id"].isin(eligible)]
    ungrounded = pred[~pred["player_id"].isin(eligible)]
    assert (grounded["pts_construction"] == "full_decomposition").all()
    assert (ungrounded["pts_construction"] == "proxy").all()
    assert grounded["active_pmf_mean"].gt(0).all()


def test_pts_decomp_unavailable_without_recon_labels_falls_back():
    df = _frame()
    bundle = OpportunityModelBundleV2({"minutes": {"deterministic_samples": 7}}).fit(df, df)
    assert bundle._pts_decomp_available is False
    with pytest.raises(RuntimeError, match="reconstruction conversion labels"):
        bundle.predict_active_pmfs(df[df["game_id"] < 2002], None, ["pts"],
                                   candidate=CANDIDATE_PTS_DECOMP)
