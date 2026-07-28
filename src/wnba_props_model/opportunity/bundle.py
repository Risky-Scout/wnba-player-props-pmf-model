"""Opportunity V2 model bundle: assembles the canonical ACTIVE (conditional-on-play) prop PMF.

Tier-0 honest scope on this repository's data:
  * FG3M  -> true opportunity decomposition: 3PA count PMF x Beta(3P%) conversion, averaged over
             conditional-active minutes samples.
  * PTS   -> hybrid: (3 * FG3M) convolved with a direct per-minute non-3PT points count. The non-3PT
             component is a Tier-0 proxy (box lacks FGM/FTM makes), recorded in opportunity_data_tier.

Minutes are integrated by averaging PMFs with equal-probability minute weights; the active PMF is
built once and never multiplied by (1 - p_dnp). An optional availability mixture is stored separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .availability_model import AvailabilityModelV2
from .component_models import OpportunityRateModel
from .contracts import DATA_TIER_BOX
from .conversion_model import HierarchicalBetaConversionModel
from .minutes_distribution import ConditionalMinutesDistributionV2
from .pmf_builders import (
    convolve_pmfs,
    marginal_beta_binomial_pmf,
    pmf_mean,
    pmf_variance,
    poisson_or_nbinom_pmf,
    weighted_mix_pmfs,
)
from .pts_decomposition import build_pts_pmf_over_minutes
from .share_model import TeamOpportunityShareModel
from .team_environment import TeamEnvironmentModelV2

CANDIDATE_RAW = "OPP_V2_RAW"
# OPP_V2_TEAM_SHARE is the *advertised* game-specific candidate:
#   predicted team 3PA (TeamEnvironmentModelV2)
#   x predicted player 3PA share (TeamOpportunityShareModel)
#   x shrunk 3P conversion  ->  FG3M PMF
CANDIDATE_TEAM_SHARE = "OPP_V2_TEAM_SHARE"
# OPP_V2_PTS_DECOMP is the full PTS decomposition candidate (owner directive section 7):
#   PTS = stretch(2PM,2) (x) stretch(3PM,3) (x) FTM, where each makes-count is a Beta-Binomial of an
#   attempt-count opportunity distribution and a shrunk conversion. 2P%/FT% conversions are fit ONLY
#   from the verified tracking-based reconstruction labels; rows without that grounding fall back to
#   the diagnostic proxy and are tagged pts_construction="proxy".
CANDIDATE_PTS_DECOMP = "OPP_V2_PTS_DECOMP"
_DEFAULT_HIER = [("position",), ("position", "role_bucket"), ("team_id", "role_bucket"), ("player_id",)]

# strictly-prior team features used to predict tonight's team 3PA total.
_TEAM_FEATURE_COLS = ["team_fg3a_ewma"]
# strictly-prior player features used to predict player 3PA share log-ratio delta.
_SHARE_FEATURE_COLS = ["player_fg3a_per_min_ewma", "player_minutes_ewma"]


class BundleFitError(RuntimeError):
    """Raised in strict (certified) mode when a required submodel genuinely fails to fit.

    Replaces the previous broad ``except Exception: available=False`` swallowing so real defects
    surface instead of silently degrading to a non-certifiable state.
    """

    def __init__(self, component: str, cause: BaseException) -> None:
        super().__init__(f"strict fit failure in {component}: {type(cause).__name__}: {cause}")
        self.component = component
        self.cause = cause


def _stretch(pmf: np.ndarray, mult: int) -> np.ndarray:
    if mult == 1:
        return pmf
    out = np.zeros((pmf.size - 1) * mult + 1)
    out[::mult] = pmf
    return out


class OpportunityModelBundleV2:
    VERSION = "opportunity_v2_bundle_v1"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._minutes = ConditionalMinutesDistributionV2(
            minimum_minutes=self.config.get("minutes", {}).get("minimum_minutes", 0.5),
            maximum_minutes=self.config.get("minutes", {}).get("maximum_minutes", 60.0))
        self._availability = AvailabilityModelV2()
        self._fg3a_rate = OpportunityRateModel(source_tier=DATA_TIER_BOX)
        self._non3pt_rate = OpportunityRateModel(source_tier=DATA_TIER_BOX)
        self._conv_3p = HierarchicalBetaConversionModel()
        # game-specific team-opportunity + player-share components (candidate TEAM_SHARE)
        self._team_env = TeamEnvironmentModelV2(
            reconcile_possessions=bool(self.config.get("team_environment", {}).get("reconcile_possessions", True)))
        self._share_fg3a = TeamOpportunityShareModel(
            share_name="fg3a_share", target_numerator_col="fg3a",
            team_total_col="team_fg3a_actual", baseline_share_col="_fg3a_share_baseline",
            feature_columns=_SHARE_FEATURE_COLS)
        self._team_share_available = False
        # full PTS decomposition components (candidate PTS_DECOMP)
        self._fg2a_rate = OpportunityRateModel(source_tier=DATA_TIER_BOX)
        self._fta_rate = OpportunityRateModel(source_tier=DATA_TIER_BOX)
        self._conv_2p = HierarchicalBetaConversionModel()
        self._conv_ft = HierarchicalBetaConversionModel()
        self._pts_decomp_available = False
        self._pts_decomp_players: set = set()
        # strict (certified) mode: genuine submodel fit failures RAISE instead of silently degrading.
        # Diagnostic fallback (labeled non-certifiable) is allowed only when strict_mode is False.
        self._strict = bool(self.config.get("strict_mode", True))
        self._team_share_reason: str | None = None
        self._pts_decomp_reason: str | None = None
        self.non_certifiable_reasons: list[str] = []
        self._n_samples = int(self.config.get("minutes", {}).get("deterministic_samples", 21))
        self._hierarchy = _DEFAULT_HIER
        _ps = self.config.get("conversion", {}).get("prior_strength", {})
        self._prior_strength = float(_ps.get("fg3", 125.0))
        self._prior_strength_2p = float(_ps.get("fg2", 200.0))
        self._prior_strength_ft = float(_ps.get("ft", 60.0))
        self._pmf_cfg = self.config.get("pmf", {})
        self.training_cutoff_utc: str | None = None
        self.feature_schema_hash: str | None = None
        self.model_bundle_hash: str | None = None
        self._fitted = False

    # --- fit ---------------------------------------------------------------
    def fit(self, train_frame: pd.DataFrame, team_train_frame: pd.DataFrame,
            config: dict[str, Any] | None = None,
            pts_recon_labels: pd.DataFrame | None = None) -> "OpportunityModelBundleV2":
        if config:
            self.config.update(config)
        df = train_frame
        played = df["did_play"].astype(bool)

        self._minutes.fit(df, df["minutes"], played)
        self._availability.fit(df, played)

        self._fg3a_rate.fit(df, df["fg3a"], df["minutes"], played,
                            feature_columns=_present(df, ["player_fg3a_per_min_ewma",
                                                          "player_minutes_ewma", "team_fg3a_ewma"]))
        non3pt = pd.to_numeric(df["pts"], errors="coerce") - 3 * pd.to_numeric(df["fg3m"], errors="coerce")
        non3pt = non3pt.clip(lower=0)
        self._non3pt_rate.fit(df.assign(_non3pt=non3pt), df.assign(_non3pt=non3pt)["_non3pt"],
                              df["minutes"], played,
                              feature_columns=_present(df, ["player_pts_per_min_ewma",
                                                            "player_fga_per_min_ewma", "player_minutes_ewma"]))
        # 3P conversion trained on training-only actual makes/attempts (labels within the fold).
        conv_df = df.assign(_succ=pd.to_numeric(df["fg3m"], errors="coerce").fillna(0.0),
                            _att=pd.to_numeric(df["fg3a"], errors="coerce").fillna(0.0))
        conv_df = conv_df[conv_df["_att"] > 0]
        self._conv_3p.fit(conv_df, "_succ", "_att", self._hierarchy, self._prior_strength)

        # --- game-specific team-opportunity + player-share fit (candidate TEAM_SHARE) ---
        # team_train_frame is now genuinely consumed: aggregate it into team-game rows and
        # fit TeamEnvironmentModelV2 (team 3PA total) + TeamOpportunityShareModel (player 3PA share).
        self._fit_team_share(team_train_frame if team_train_frame is not None else df)

        # --- full PTS decomposition fit (candidate PTS_DECOMP) ---
        # 2P%/FT% conversions are fit ONLY on tracking-reconstruction labels merged into training rows.
        self._fit_pts_decomposition(df, pts_recon_labels)

        self.training_cutoff_utc = str(pd.to_datetime(df["prediction_cutoff_utc"], utc=True,
                                                       errors="coerce").max())
        self.feature_schema_hash = hashlib.sha256(
            json.dumps(sorted(df.columns.tolist())).encode()).hexdigest()[:16]
        self.model_bundle_hash = hashlib.sha256(
            (str(self._availability.model_hash) + str(self.feature_schema_hash)).encode()).hexdigest()[:16]
        self._fitted = True
        return self

    # --- team-opportunity / player-share plumbing --------------------------
    @staticmethod
    def _aggregate_team_frame(player_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player rows into one strictly-prior team-game row per (game_id, team_id).

        Targets (e.g. team_fg3a_actual) are within-fold labels; features (team_fg3a_ewma) are
        strictly prior-game EWMAs shared across a team-game, so no leakage is introduced.
        """
        played = player_df["did_play"].astype(bool) if "did_play" in player_df.columns else True
        d = player_df[played].copy() if not isinstance(played, bool) else player_df.copy()
        grp = d.groupby(["game_id", "team_id"], as_index=False)
        agg = grp.agg(
            fg3a=("fg3a", "sum"),
            fga=("fga", "sum") if "fga" in d.columns else ("fg3a", "sum"),
            fta=("fta", "sum") if "fta" in d.columns else ("fg3a", "sum"),
            turnovers=("tov", "sum") if "tov" in d.columns else ("fg3a", "sum"),
            team_fg3a_ewma=("team_fg3a_ewma", "first"),
        )
        # opponent linkage for reconciliation (best-effort; single team retained otherwise)
        if "opponent_team_id" in d.columns:
            opp = d.groupby(["game_id", "team_id"], as_index=False)["opponent_team_id"].first()
            agg = agg.merge(opp, on=["game_id", "team_id"], how="left")
        return agg

    def _fit_team_share(self, team_source: pd.DataFrame) -> None:
        team = self._aggregate_team_frame(team_source)
        # rename the aggregated actual as the environment target
        team_fit = team.rename(columns={"fg3a": "fg3a"})  # fg3a already the team total
        feat = [c for c in _TEAM_FEATURE_COLS if c in team_fit.columns]
        if not feat or len(team_fit) < 5 or team_fit["fg3a"].notna().sum() < 5:
            # data-availability skip (not a fit failure): unavailable but not an error in either mode.
            self._team_share_available = False
            self._team_share_reason = "insufficient team-game coverage for team-environment fit"
            return
        try:
            # TeamEnvironmentModelV2 only fits targets with >= _MIN_COVERAGE rows; with fewer rows
            # (unit tests / tiny folds) it will mark fg3a unavailable — handled below.
            self._team_env.fit(team_fit, feature_columns=feat)
            # player share training frame: baseline share from prior expected 3PA, team total label
            player = team_source.copy()
            player = player[player["did_play"].astype(bool)] if "did_play" in player.columns else player
            base = (pd.to_numeric(player.get("player_fg3a_per_min_ewma", 0.0), errors="coerce").fillna(0.0)
                    * pd.to_numeric(player.get("player_minutes_ewma", 0.0), errors="coerce").fillna(0.0))
            player = player.assign(_fg3a_share_baseline=base.clip(lower=0.0))
            team_tot = team_source.copy()
            team_tot = team_tot[team_tot["did_play"].astype(bool)] if "did_play" in team_tot.columns else team_tot
            tot = (team_tot.groupby(["game_id", "team_id"])["fg3a"].transform("sum"))
            player = player.assign(team_fg3a_actual=tot.reindex(player.index).to_numpy())
            if "p_active" not in player.columns:
                player = player.assign(p_active=1.0)
            self._share_fg3a.fit(player)
            self._team_share_available = bool(self._team_env.target_available.get("fg3a", False))
            if not self._team_share_available:
                self._team_share_reason = "team-environment fg3a target not available after fit"
        except Exception as e:
            if self._strict:
                raise BundleFitError("team_share", e) from e
            self._team_share_available = False
            self._team_share_reason = f"non_certifiable_fallback: {type(e).__name__}: {e}"
            self.non_certifiable_reasons.append(f"team_share: {self._team_share_reason}")

    def _fit_pts_decomposition(self, df: pd.DataFrame, recon: pd.DataFrame | None) -> None:
        """Fit 2PA/FTA opportunity rates (box) + 2P%/FT% conversions (tracking-reconstruction labels).

        The 2P% and FT% conversions require MAKE counts (FG2M, FTM) that only exist in the verified
        tracking-based reconstruction (``pts_conversion_labels``). If those labels are absent, the
        full-decomposition candidate is marked unavailable and callers fall back to the proxy.
        """
        if recon is None or len(recon) == 0:
            self._pts_decomp_available = False
            self._pts_decomp_reason = "no reconstruction labels supplied"
            return
        try:
            played = df["did_play"].astype(bool)
            fg2a = (pd.to_numeric(df["fga"], errors="coerce")
                    - pd.to_numeric(df["fg3a"], errors="coerce")).clip(lower=0)
            self._fg2a_rate.fit(df.assign(_fg2a=fg2a), df.assign(_fg2a=fg2a)["_fg2a"],
                                df["minutes"], played,
                                feature_columns=_present(df, ["player_fga_per_min_ewma",
                                                              "player_minutes_ewma"]))
            fta = pd.to_numeric(df["fta"], errors="coerce").fillna(0.0)
            self._fta_rate.fit(df.assign(_fta=fta), df.assign(_fta=fta)["_fta"],
                               df["minutes"], played,
                               feature_columns=_present(df, ["player_fga_per_min_ewma",
                                                             "player_minutes_ewma"]))
            # Merge reconstruction makes/attempts onto training rows (which carry the hierarchy cols).
            lab = recon.copy()
            key = ["game_id", "player_id"]
            need = key + ["FG2M", "FG2A", "FTM", "FTA"]
            merged = df.merge(lab[need], on=key, how="inner")
            if len(merged) < 30:
                self._pts_decomp_available = False
                self._pts_decomp_reason = f"insufficient reconstruction-matched rows ({len(merged)}<30)"
                return
            m2 = merged[pd.to_numeric(merged["FG2A"], errors="coerce") > 0].assign(
                _s=lambda x: pd.to_numeric(x["FG2M"], errors="coerce").fillna(0.0),
                _a=lambda x: pd.to_numeric(x["FG2A"], errors="coerce").fillna(0.0))
            mft = merged[pd.to_numeric(merged["FTA"], errors="coerce") > 0].assign(
                _s=lambda x: pd.to_numeric(x["FTM"], errors="coerce").fillna(0.0),
                _a=lambda x: pd.to_numeric(x["FTA"], errors="coerce").fillna(0.0))
            if len(m2) < 30 or len(mft) < 30:
                self._pts_decomp_available = False
                self._pts_decomp_reason = f"insufficient conversion rows (2P={len(m2)}, FT={len(mft)})"
                return
            self._conv_2p.fit(m2, "_s", "_a", self._hierarchy, self._prior_strength_2p)
            self._conv_ft.fit(mft, "_s", "_a", self._hierarchy, self._prior_strength_ft)
            # Only players grounded by reconstruction labels are full-decomposition eligible.
            self._pts_decomp_players = set(merged["player_id"].unique().tolist())
            self._pts_decomp_available = True
        except Exception as e:
            if self._strict:
                raise BundleFitError("pts_decomposition", e) from e
            self._pts_decomp_available = False
            self._pts_decomp_reason = f"non_certifiable_fallback: {type(e).__name__}: {e}"
            self.non_certifiable_reasons.append(f"pts_decomposition: {self._pts_decomp_reason}")

    # --- predict -----------------------------------------------------------
    def predict_active_pmfs(self, player_frame: pd.DataFrame, team_frame: pd.DataFrame | None,
                            props: Sequence[str], candidate: str = CANDIDATE_RAW) -> pd.DataFrame:
        if candidate == CANDIDATE_TEAM_SHARE:
            return self._predict_team_share(player_frame, team_frame, props)
        if candidate == CANDIDATE_PTS_DECOMP:
            return self._predict_pts_decomposition(player_frame, props)
        if not self._fitted:
            raise RuntimeError("OpportunityModelBundleV2.predict before fit")
        supported = [p for p in props if p in ("fg3m", "pts")]
        unsupported = [p for p in props if p not in ("fg3m", "pts")]
        if unsupported:
            raise ValueError(f"OpportunityModelBundleV2: props {unsupported} require Tier-2 data "
                             "not available (see DATA_AVAILABILITY_AUDIT.json)")
        df = player_frame.reset_index(drop=True)
        samples, weights = self._minutes.deterministic_samples(df, n_samples=self._n_samples)
        p_active = self._availability.predict_active_probability(df)
        conv_post = self._conv_3p.predict_posterior(df)

        # Per-minute rates (predict_for_minutes with minutes=1 returns the per-minute hazard).
        ones = np.ones(len(df))
        fg3a_rate = self._fg3a_rate.predict_for_minutes(df, ones).mean
        non3_rate = self._non3pt_rate.predict_for_minutes(df, ones).mean
        r_fg3a = self._fg3a_rate._dispersion_r
        r_non3 = self._non3pt_rate._dispersion_r

        tail = float(self._pmf_cfg.get("tail_tolerance", 1e-8))
        cap = int(self._pmf_cfg.get("maximum_support", 120))

        out_rows = []
        for i in range(len(df)):
            mins = samples[i]
            alpha, beta = conv_post[i].alpha, conv_post[i].beta
            for prop in supported:
                per_sample = []
                for m in mins:
                    if prop == "fg3m":
                        mu = max(fg3a_rate[i] * m, 1e-6)
                        att = poisson_or_nbinom_pmf(mu, r_fg3a, tail_tolerance=tail, maximum_cap=cap)
                        pmf = marginal_beta_binomial_pmf(att, alpha, beta)
                    else:  # pts hybrid: 3*FG3M + non-3pt count
                        mu3 = max(fg3a_rate[i] * m, 1e-6)
                        att3 = poisson_or_nbinom_pmf(mu3, r_fg3a, tail_tolerance=tail, maximum_cap=cap)
                        fg3m_pmf = marginal_beta_binomial_pmf(att3, alpha, beta)
                        mun = max(non3_rate[i] * m, 1e-6)
                        non3 = poisson_or_nbinom_pmf(mun, r_non3, tail_tolerance=tail, maximum_cap=cap)
                        pmf = convolve_pmfs(_stretch(fg3m_pmf, 3), non3)
                    per_sample.append(pmf)
                active = weighted_mix_pmfs(per_sample, weights)
                out_rows.append({
                    "game_id": df.at[i, "game_id"], "player_id": df.at[i, "player_id"],
                    "team_id": df.at[i, "team_id"],
                    "game_date": df.at[i, "game_date"] if "game_date" in df.columns else pd.NaT,
                    "prediction_cutoff_utc": df.at[i, "prediction_cutoff_utc"],
                    "stat": prop, "candidate_id": CANDIDATE_RAW,
                    "active_pmf_json": json.dumps([round(float(x), 10) for x in active]),
                    "active_pmf_mean": pmf_mean(active), "active_pmf_variance": pmf_variance(active),
                    "p_active": float(p_active[i]),
                    "opportunity_data_tier": DATA_TIER_BOX,
                    "conversion_mean": float(conv_post[i].mean),
                    "feature_schema_hash": self.feature_schema_hash,
                    "model_bundle_hash": self.model_bundle_hash,
                    "training_cutoff_utc": self.training_cutoff_utc,
                })
        return pd.DataFrame(out_rows)

    def _predict_team_share(self, player_frame: pd.DataFrame, team_frame: pd.DataFrame | None,
                            props: Sequence[str]) -> pd.DataFrame:
        """Game-specific FG3M candidate: team 3PA x player 3PA share x 3P conversion.

        Requires the team-share components to have fit (``_team_share_available``). ``team_frame``
        is genuinely consumed: if provided it is used directly for the team environment prediction,
        otherwise it is aggregated from ``player_frame``. Raises if the caller requests props other
        than fg3m under this candidate (PTS decomposition is a separate work item).
        """
        if not self._fitted:
            raise RuntimeError("OpportunityModelBundleV2.predict before fit")
        if not self._team_share_available:
            raise RuntimeError("OPP_V2_TEAM_SHARE unavailable: team-environment fg3a target had "
                               "insufficient coverage to fit (see TeamEnvironmentModelV2.target_available)")
        unsupported = [p for p in props if p != "fg3m"]
        if unsupported:
            raise ValueError(f"OPP_V2_TEAM_SHARE currently supports fg3m only; got {unsupported}")

        df = player_frame.reset_index(drop=True)
        team_src = team_frame if team_frame is not None else df
        team_pred_frame = self._aggregate_team_frame(team_src)
        # tonight's predicted team 3PA total per (game_id, team_id)
        env = self._team_env.predict(team_pred_frame)
        team_pred_frame = team_pred_frame.assign(team_fg3a_hat=env["fg3a"].to_numpy())
        team_hat = team_pred_frame.set_index(["game_id", "team_id"])["team_fg3a_hat"]

        # player 3PA share (sums to 1 within each team-game-cutoff; inactive players zeroed)
        base = (pd.to_numeric(df.get("player_fg3a_per_min_ewma", 0.0), errors="coerce").fillna(0.0)
                * pd.to_numeric(df.get("player_minutes_ewma", 0.0), errors="coerce").fillna(0.0))
        df = df.assign(_fg3a_share_baseline=base.clip(lower=0.0))
        p_active = self._availability.predict_active_probability(df)
        df = df.assign(p_active=p_active)
        if "prediction_cutoff_utc" not in df.columns:
            df = df.assign(prediction_cutoff_utc="NA")
        shares = self._share_fg3a.predict_team_normalized_shares(df)

        conv_post = self._conv_3p.predict_posterior(df)
        r_fg3a = self._fg3a_rate._dispersion_r
        tail = float(self._pmf_cfg.get("tail_tolerance", 1e-8))
        cap = int(self._pmf_cfg.get("maximum_support", 120))

        out_rows = []
        for i in range(len(df)):
            g, t = df.at[i, "game_id"], df.at[i, "team_id"]
            team_total = float(team_hat.get((g, t), np.nan))
            player_3pa_mean = max(team_total * float(shares[i]), 1e-6) if np.isfinite(team_total) else 1e-6
            att = poisson_or_nbinom_pmf(player_3pa_mean, r_fg3a, tail_tolerance=tail, maximum_cap=cap)
            alpha, beta = conv_post[i].alpha, conv_post[i].beta
            active = marginal_beta_binomial_pmf(att, alpha, beta)
            out_rows.append({
                "game_id": g, "player_id": df.at[i, "player_id"], "team_id": t,
                "game_date": df.at[i, "game_date"] if "game_date" in df.columns else pd.NaT,
                "prediction_cutoff_utc": df.at[i, "prediction_cutoff_utc"],
                "stat": "fg3m", "candidate_id": CANDIDATE_TEAM_SHARE,
                "active_pmf_json": json.dumps([round(float(x), 10) for x in active]),
                "active_pmf_mean": pmf_mean(active), "active_pmf_variance": pmf_variance(active),
                "p_active": float(p_active[i]),
                "team_fg3a_hat": team_total, "player_fg3a_share": float(shares[i]),
                "player_fg3a_mean": player_3pa_mean,
                "opportunity_data_tier": DATA_TIER_BOX,
                "conversion_mean": float(conv_post[i].mean),
                "feature_schema_hash": self.feature_schema_hash,
                "model_bundle_hash": self.model_bundle_hash,
                "training_cutoff_utc": self.training_cutoff_utc,
            })
        return pd.DataFrame(out_rows)

    def _predict_pts_decomposition(self, player_frame: pd.DataFrame,
                                   props: Sequence[str]) -> pd.DataFrame:
        """Full PTS decomposition candidate: 2PM/3PM/FTM convolution, with per-row proxy fallback.

        A row is full-decomposition eligible iff its player was grounded by tracking-reconstruction
        conversion labels in training (``_pts_decomp_players``). Ineligible rows fall back to the
        diagnostic proxy (3*FG3M convolved with a direct non-3PT points count) and are tagged
        ``pts_construction="proxy"`` so they cannot enter full-PTS candidate evidence.
        """
        if not self._fitted:
            raise RuntimeError("OpportunityModelBundleV2.predict before fit")
        if not self._pts_decomp_available:
            raise RuntimeError("OPP_V2_PTS_DECOMP unavailable: reconstruction conversion labels "
                               "(pts_recon_labels) were not provided/fit")
        unsupported = [p for p in props if p != "pts"]
        if unsupported:
            raise ValueError(f"OPP_V2_PTS_DECOMP supports pts only; got {unsupported}")

        df = player_frame.reset_index(drop=True)
        samples, weights = self._minutes.deterministic_samples(df, n_samples=self._n_samples)
        p_active = self._availability.predict_active_probability(df)
        ones = np.ones(len(df))
        rate_2pa = self._fg2a_rate.predict_for_minutes(df, ones).mean
        rate_3pa = self._fg3a_rate.predict_for_minutes(df, ones).mean
        rate_fta = self._fta_rate.predict_for_minutes(df, ones).mean
        non3_rate = self._non3pt_rate.predict_for_minutes(df, ones).mean
        r_2pa, r_3pa = self._fg2a_rate._dispersion_r, self._fg3a_rate._dispersion_r
        r_fta, r_non3 = self._fta_rate._dispersion_r, self._non3pt_rate._dispersion_r
        post2 = self._conv_2p.predict_posterior(df)
        post3 = self._conv_3p.predict_posterior(df)
        postft = self._conv_ft.predict_posterior(df)
        tail = float(self._pmf_cfg.get("tail_tolerance", 1e-8))
        cap = int(self._pmf_cfg.get("maximum_support", 120))

        out_rows = []
        for i in range(len(df)):
            mins = samples[i]
            eligible = df.at[i, "player_id"] in self._pts_decomp_players
            if eligible:
                active = build_pts_pmf_over_minutes(
                    mins, weights,
                    rate_2pa=float(rate_2pa[i]), r_2pa=r_2pa,
                    alpha2=float(post2[i].alpha), beta2=float(post2[i].beta),
                    rate_3pa=float(rate_3pa[i]), r_3pa=r_3pa,
                    alpha3=float(post3[i].alpha), beta3=float(post3[i].beta),
                    rate_fta=float(rate_fta[i]), r_fta=r_fta,
                    alpha_ft=float(postft[i].alpha), beta_ft=float(postft[i].beta),
                    tail_tolerance=tail, maximum_cap=cap)
                construction = "full_decomposition"
            else:
                per_sample = []
                for m in mins:
                    att3 = poisson_or_nbinom_pmf(max(rate_3pa[i] * m, 1e-6), r_3pa,
                                                 tail_tolerance=tail, maximum_cap=cap)
                    fg3m_pmf = marginal_beta_binomial_pmf(att3, post3[i].alpha, post3[i].beta)
                    non3 = poisson_or_nbinom_pmf(max(non3_rate[i] * m, 1e-6), r_non3,
                                                 tail_tolerance=tail, maximum_cap=cap)
                    per_sample.append(convolve_pmfs(_stretch(fg3m_pmf, 3), non3))
                active = weighted_mix_pmfs(per_sample, weights)
                construction = "proxy"
            out_rows.append({
                "game_id": df.at[i, "game_id"], "player_id": df.at[i, "player_id"],
                "team_id": df.at[i, "team_id"],
                "game_date": df.at[i, "game_date"] if "game_date" in df.columns else pd.NaT,
                "prediction_cutoff_utc": df.at[i, "prediction_cutoff_utc"],
                "stat": "pts", "candidate_id": CANDIDATE_PTS_DECOMP,
                "pts_construction": construction,
                "active_pmf_json": json.dumps([round(float(x), 10) for x in active]),
                "active_pmf_mean": pmf_mean(active), "active_pmf_variance": pmf_variance(active),
                "p_active": float(p_active[i]),
                "opportunity_data_tier": DATA_TIER_BOX,
                "conversion_2p_mean": float(post2[i].mean), "conversion_3p_mean": float(post3[i].mean),
                "conversion_ft_mean": float(postft[i].mean),
                "feature_schema_hash": self.feature_schema_hash,
                "model_bundle_hash": self.model_bundle_hash,
                "training_cutoff_utc": self.training_cutoff_utc,
            })
        return pd.DataFrame(out_rows)

    # --- persistence -------------------------------------------------------
    def save(self, directory: Path) -> dict[str, Any]:
        import joblib
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "opportunity_bundle_v2.joblib"
        joblib.dump(self, path)
        meta = {"version": self.VERSION, "model_bundle_hash": self.model_bundle_hash,
                "feature_schema_hash": self.feature_schema_hash,
                "training_cutoff_utc": self.training_cutoff_utc,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        json.dump(meta, open(directory / "opportunity_bundle_v2_meta.json", "w"), indent=2)
        return meta

    @classmethod
    def load(cls, directory: Path) -> "OpportunityModelBundleV2":
        import joblib
        return joblib.load(Path(directory) / "opportunity_bundle_v2.joblib")


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    present = [c for c in cols if c in df.columns]
    if not present:
        raise KeyError(f"OpportunityModelBundleV2: none of the required features present: {cols}")
    return present
