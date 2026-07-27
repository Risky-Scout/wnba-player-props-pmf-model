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

CANDIDATE_RAW = "OPP_V2_RAW"
_DEFAULT_HIER = [("position",), ("position", "role_bucket"), ("team_id", "role_bucket"), ("player_id",)]


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
        self._n_samples = int(self.config.get("minutes", {}).get("deterministic_samples", 21))
        self._hierarchy = _DEFAULT_HIER
        self._prior_strength = float(self.config.get("conversion", {}).get("prior_strength", {}).get("fg3", 125.0))
        self._pmf_cfg = self.config.get("pmf", {})
        self.training_cutoff_utc: str | None = None
        self.feature_schema_hash: str | None = None
        self.model_bundle_hash: str | None = None
        self._fitted = False

    # --- fit ---------------------------------------------------------------
    def fit(self, train_frame: pd.DataFrame, team_train_frame: pd.DataFrame,
            config: dict[str, Any] | None = None) -> "OpportunityModelBundleV2":
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

        self.training_cutoff_utc = str(pd.to_datetime(df["prediction_cutoff_utc"], utc=True,
                                                       errors="coerce").max())
        self.feature_schema_hash = hashlib.sha256(
            json.dumps(sorted(df.columns.tolist())).encode()).hexdigest()[:16]
        self.model_bundle_hash = hashlib.sha256(
            (str(self._availability.model_hash) + str(self.feature_schema_hash)).encode()).hexdigest()[:16]
        self._fitted = True
        return self

    # --- predict -----------------------------------------------------------
    def predict_active_pmfs(self, player_frame: pd.DataFrame, team_frame: pd.DataFrame | None,
                            props: Sequence[str]) -> pd.DataFrame:
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
