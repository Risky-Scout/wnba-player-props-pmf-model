"""WNBATeamScoreModel — Dixon-Coles inspired bivariate team score model.

Replaces the quantile-based GameTotalsModel with a proper parametric model:
  home_lambda = exp(home_attack + away_defense + home_advantage + log_pace_adj)
  away_lambda = exp(away_attack + home_defense + log_pace_adj)
  home_score ~ NegBinom(mu=home_lambda, r=r_home)
  away_score ~ NegBinom(mu=away_lambda, r=r_away)
  game_total_pmf = convolve_pmfs(home_pmf, away_pmf)  # exact convolution

References
----------
Dixon & Coles (1997) — "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market", Applied Statistics.

PenaltyBlog — FootballProbabilityGrid, bivariate Poisson model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from wnba_props_model.constants import DOMAIN_MAX
from wnba_props_model.models.market import fair_american
from wnba_props_model.models.simulation import convolve_pmfs, normalize_pmf

logger = logging.getLogger(__name__)

# Truncated domain for team score PMFs: WNBA scores rarely exceed 130
TEAM_SCORE_MAX = 130
GAME_TOTAL_MAX = DOMAIN_MAX.get("game_total", 260)

try:
    import penaltyblog
    _PB_AVAILABLE = True
except ImportError:
    _PB_AVAILABLE = False


def _negbinom_pmf(mu: float, r: float, domain: int = TEAM_SCORE_MAX) -> np.ndarray:
    """Negative binomial PMF with mean=mu, dispersion=r.

    Parameterization: P(Y=k) = C(k+r-1, k) * (r/(r+mu))^r * (mu/(r+mu))^k

    When r→∞, converges to Poisson(mu).
    """
    if r is None or r <= 0 or not np.isfinite(r):
        # Fall back to Poisson
        ks = np.arange(domain + 1)
        pmf = stats.poisson.pmf(ks, mu=max(mu, 1e-6))
        return normalize_pmf(pmf)

    ks = np.arange(domain + 1)
    p = r / (r + mu)
    pmf = np.exp(
        special.gammaln(ks + r) - special.gammaln(r) - special.gammaln(ks + 1)
        + r * np.log(p + 1e-15)
        + ks * np.log(1 - p + 1e-15)
    )
    return normalize_pmf(np.clip(pmf, 0, None))


def _fit_negbinom_dispersion(y: np.ndarray) -> float:
    """MLE estimate of NegBinom dispersion r from observed count data."""
    if len(y) < 5:
        return 20.0  # weak prior for small samples
    mu = float(y.mean())
    var = float(y.var())
    if var <= mu or mu <= 0:
        return 50.0  # near-Poisson
    r_hat = mu ** 2 / max(var - mu, 1e-6)
    return float(np.clip(r_hat, 1.0, 200.0))


class WNBATeamScoreModel:
    """Dixon-Coles NegBinom model for WNBA team scores.

    Parameters are estimated via numerical log-likelihood maximisation with
    time-decay weighting.  Attack/defence parameters are fit per team;
    a single home-court advantage parameter and pace adjustment are shared.

    Usage
    -----
    model = WNBATeamScoreModel()
    model.fit(games_df, xi=0.002)
    grid = model.predict("LV", "SEA")
    print(grid.total_over(163.5))
    """

    VERSION = "team_score_v1"

    def __init__(self) -> None:
        self._params: dict[str, float] = {}
        self._teams: list[str] = []
        self._r_home: float = 20.0
        self._r_away: float = 20.0
        self._fitted: bool = False
        self._train_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, games_df: pd.DataFrame, xi: float = 0.002) -> "WNBATeamScoreModel":
        """Fit the model on historical WNBA game data.

        Parameters
        ----------
        games_df : pd.DataFrame
            Must contain columns: home_team, away_team, home_score, away_score,
            game_date (datetime-like). Optionally: home_pace, away_pace,
            home_opp_points_roll5, away_opp_points_roll5.
        xi : float
            Dixon-Coles time-decay parameter (weight = exp(-xi * days_ago)).
            Larger xi → more weight on recent games. Typical: 0.002 – 0.005.
        """
        df = games_df.copy()
        df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        # Compute time-decay weights
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
        latest = df["game_date"].max()
        df["days_ago"] = (latest - df["game_date"]).dt.total_seconds() / 86400.0
        df["weight"] = np.exp(-xi * df["days_ago"])

        # Team list (sorted for reproducibility)
        all_teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
        self._teams = all_teams
        n_teams = len(all_teams)
        team_idx = {t: i for i, t in enumerate(all_teams)}

        # Pace adjustment: derive from rolling defensive pace proxy if available
        df["pace_adj"] = self._compute_pace_adj(df)

        # Fit dispersion parameters from training data
        self._r_home = _fit_negbinom_dispersion(df["home_score"].values)
        self._r_away = _fit_negbinom_dispersion(df["away_score"].values)

        # Parameter vector layout:
        # [attack_0..attack_N-1, defense_0..defense_N-1, home_advantage, log_base_score]
        # Identify constraint: attack_0 = 0 (reference team), defence_0 = 0
        # In practice, we drop last attack and last defense to avoid collinearity
        # and use: n_attack = n_teams - 1, n_defense = n_teams - 1, + home_adv + base
        n_attack = n_teams - 1
        n_defense = n_teams - 1
        n_params = n_attack + n_defense + 2  # +home_adv +log_base

        log_mean_home = float(np.log(max(df["home_score"].mean(), 1.0)))
        log_mean_away = float(np.log(max(df["away_score"].mean(), 1.0)))

        x0 = np.zeros(n_params)
        x0[-2] = 0.1   # home_advantage
        x0[-1] = log_mean_home  # log_base_score

        weights = df["weight"].values
        home_idx = df["home_team"].map(team_idx).values
        away_idx = df["away_team"].map(team_idx).values
        home_scores = df["home_score"].values
        away_scores = df["away_score"].values
        pace_adjs = df["pace_adj"].values

        def _neg_log_lik(x: np.ndarray) -> float:
            attack = np.concatenate([[0.0], x[:n_attack]])   # attack[0] = 0 (reference)
            defense = np.concatenate([[0.0], x[n_attack:n_attack + n_defense]])
            home_adv = x[-2]
            log_base = x[-1]

            log_home_lam = log_base + attack[home_idx] - defense[away_idx] + home_adv + pace_adjs
            log_away_lam = log_base + attack[away_idx] - defense[home_idx] + pace_adjs

            home_lam = np.clip(np.exp(log_home_lam), 1.0, TEAM_SCORE_MAX)
            away_lam = np.clip(np.exp(log_away_lam), 1.0, TEAM_SCORE_MAX)

            r_h = self._r_home
            r_a = self._r_away
            ll_home = self._negbinom_logpmf_vec(home_scores, home_lam, r_h)
            ll_away = self._negbinom_logpmf_vec(away_scores, away_lam, r_a)
            return -float((weights * (ll_home + ll_away)).sum())

        result = optimize.minimize(
            _neg_log_lik,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        x_opt = result.x
        attack = np.concatenate([[0.0], x_opt[:n_attack]])
        defense = np.concatenate([[0.0], x_opt[n_attack:n_attack + n_defense]])
        home_adv = float(x_opt[-2])
        log_base = float(x_opt[-1])

        self._params = {
            "home_advantage": home_adv,
            "log_base_score": log_base,
        }
        for i, t in enumerate(all_teams):
            self._params[f"attack_{t}"] = float(attack[i])
            self._params[f"defense_{t}"] = float(defense[i])

        self._fitted = True
        self._train_stats = {
            "n_games": int(len(df)),
            "n_teams": n_teams,
            "xi": xi,
            "converged": bool(result.success),
            "final_nll": float(result.fun),
            "r_home": self._r_home,
            "r_away": self._r_away,
        }
        logger.info(
            "WNBATeamScoreModel fit: n_games=%d, n_teams=%d, converged=%s, NLL=%.4f",
            len(df), n_teams, result.success, result.fun,
        )
        return self

    @staticmethod
    def _negbinom_logpmf_vec(k: np.ndarray, mu: np.ndarray, r: float) -> np.ndarray:
        """Vectorised NegBinom log-PMF."""
        p = r / (r + mu + 1e-12)
        return (
            special.gammaln(k + r) - special.gammaln(r) - special.gammaln(k + 1)
            + r * np.log(p + 1e-15)
            + k * np.log(1 - p + 1e-15)
        )

    @staticmethod
    def _compute_pace_adj(df: pd.DataFrame) -> np.ndarray:
        """Return per-game log pace adjustment (0 if features unavailable)."""
        adj = np.zeros(len(df))
        if "home_opp_points_roll5" in df.columns and "away_opp_points_roll5" in df.columns:
            home_pace = df["home_opp_points_roll5"].fillna(df["home_opp_points_roll5"].median())
            away_pace = df["away_opp_points_roll5"].fillna(df["away_opp_points_roll5"].median())
            league_avg = float(pd.concat([home_pace, away_pace]).mean())
            if league_avg > 0:
                adj = 0.5 * np.log(
                    (home_pace.values / league_avg) * (away_pace.values / league_avg) + 1e-9
                )
        return adj

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _get_lambda(self, home_team: str, away_team: str,
                    pace_adj: float = 0.0) -> tuple[float, float]:
        """Compute (home_lambda, away_lambda) from fitted parameters."""
        if not self._fitted:
            raise RuntimeError("WNBATeamScoreModel not fitted")

        p = self._params
        log_base = p["log_base_score"]
        home_adv = p["home_advantage"]

        home_att = p.get(f"attack_{home_team}", 0.0)
        home_def = p.get(f"defense_{home_team}", 0.0)
        away_att = p.get(f"attack_{away_team}", 0.0)
        away_def = p.get(f"defense_{away_team}", 0.0)

        if home_team not in self._teams:
            logger.warning("Unknown home team '%s'; using neutral parameters", home_team)
        if away_team not in self._teams:
            logger.warning("Unknown away team '%s'; using neutral parameters", away_team)

        log_home_lam = log_base + home_att - away_def + home_adv + pace_adj
        log_away_lam = log_base + away_att - home_def + pace_adj

        home_lam = float(np.clip(np.exp(log_home_lam), 1.0, TEAM_SCORE_MAX))
        away_lam = float(np.clip(np.exp(log_away_lam), 1.0, TEAM_SCORE_MAX))
        return home_lam, away_lam

    def predict(
        self,
        home_team: str,
        away_team: str,
        pace_adj: float = 0.0,
        game_id: int | str | None = None,
        game_date: str | None = None,
    ) -> "WNBATeamScorePMFGrid":
        """Predict score distributions for a game.

        Parameters
        ----------
        home_team, away_team : str
            Team abbreviations as stored during fit.
        pace_adj : float
            Log pace adjustment (computed externally or zero for neutral pace).
        game_id, game_date : optional metadata for the grid.
        """
        home_lam, away_lam = self._get_lambda(home_team, away_team, pace_adj)
        home_pmf = _negbinom_pmf(home_lam, self._r_home, domain=TEAM_SCORE_MAX)
        away_pmf = _negbinom_pmf(away_lam, self._r_away, domain=TEAM_SCORE_MAX)
        total_pmf = convolve_pmfs(home_pmf, away_pmf, domain_max=GAME_TOTAL_MAX)

        return WNBATeamScorePMFGrid(
            home_team=home_team,
            away_team=away_team,
            home_score_pmf=home_pmf,
            away_score_pmf=away_pmf,
            total_score_pmf=total_pmf,
            home_lambda=home_lam,
            away_lambda=away_lam,
            game_id=game_id,
            game_date=game_date,
        )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "WNBATeamScoreModel":
        return joblib.load(path)

    def get_training_summary(self) -> dict[str, Any]:
        return {**self._train_stats, "version": self.VERSION}


@dataclass
class WNBATeamScorePMFGrid:
    """Bivariate score distribution for a single WNBA game.

    All distributions are normalized PMF arrays (index = score value).
    Total score PMF is the exact convolution of home and away score PMFs.

    Properties
    ----------
    home_score_pmf : np.ndarray, domain [0, 130]
    away_score_pmf : np.ndarray, domain [0, 130]
    total_score_pmf : np.ndarray, domain [0, 260]

    Usage
    -----
    grid.total_over(163.5)        # P(total > 163.5)
    grid.team_total_over("LV", 85.5)
    grid.home_covers_spread(-4.5)
    grid.to_dict()                # full serializable output
    """

    home_team: str
    away_team: str
    home_score_pmf: np.ndarray
    away_score_pmf: np.ndarray
    total_score_pmf: np.ndarray
    home_lambda: float = 0.0
    away_lambda: float = 0.0
    game_id: int | str | None = None
    game_date: str | None = None

    def __post_init__(self) -> None:
        self.home_score_pmf = normalize_pmf(self.home_score_pmf)
        self.away_score_pmf = normalize_pmf(self.away_score_pmf)
        self.total_score_pmf = normalize_pmf(self.total_score_pmf)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prob_over(pmf: np.ndarray, line: float) -> float:
        k_min = int(np.floor(line)) + 1
        if k_min >= len(pmf):
            return 0.0
        return float(pmf[k_min:].sum())

    @staticmethod
    def _prob_under(pmf: np.ndarray, line: float) -> float:
        k_max = int(np.ceil(line)) - 1
        if k_max < 0:
            return 0.0
        k_max = min(k_max, len(pmf) - 1)
        return float(pmf[: k_max + 1].sum())

    @staticmethod
    def _push_prob(pmf: np.ndarray, line: float) -> float:
        frac = line % 1.0
        if abs(frac - 0.5) < 1e-9:
            return 0.0
        int_k = int(round(line))
        if int_k < 0 or int_k >= len(pmf):
            return 0.0
        return float(pmf[int_k])

    # ------------------------------------------------------------------
    # Public market methods
    # ------------------------------------------------------------------

    def total_over(self, line: float) -> float:
        """P(home_score + away_score > line)."""
        return self._prob_over(self.total_score_pmf, line)

    def total_under(self, line: float) -> float:
        """P(home_score + away_score < line)."""
        return self._prob_under(self.total_score_pmf, line)

    def total_push(self, line: float) -> float:
        """P(home_score + away_score == line)."""
        return self._push_prob(self.total_score_pmf, line)

    def home_covers_spread(self, spread: float) -> float:
        """P(home_score - away_score > spread).

        For a -4.5 spread: P(home wins by 5+).
        For a +4.5 spread: P(home wins or loses by fewer than 5).
        """
        h = self.home_score_pmf
        a = self.away_score_pmf
        domain_h = len(h)
        domain_a = len(a)

        prob = 0.0
        for i_h in range(domain_h):
            if h[i_h] < 1e-12:
                continue
            # need: i_h - i_a > spread → i_a < i_h - spread
            max_a = int(np.floor(i_h - spread)) - 1
            if max_a < 0:
                continue
            max_a = min(max_a, domain_a - 1)
            prob += h[i_h] * float(a[:max_a + 1].sum())
        return float(prob)

    def away_covers_spread(self, spread: float) -> float:
        """P(away_score - home_score > spread)."""
        return self.home_covers_spread(-spread)

    def team_total_over(self, team: str, line: float) -> float:
        """P(team_score > line) for a specific team."""
        if team == self.home_team:
            return self._prob_over(self.home_score_pmf, line)
        if team == self.away_team:
            return self._prob_over(self.away_score_pmf, line)
        raise ValueError(f"Team '{team}' not in this game ({self.home_team} vs {self.away_team})")

    def team_total_under(self, team: str, line: float) -> float:
        """P(team_score < line) for a specific team."""
        if team == self.home_team:
            return self._prob_under(self.home_score_pmf, line)
        if team == self.away_team:
            return self._prob_under(self.away_score_pmf, line)
        raise ValueError(f"Team '{team}' not in this game ({self.home_team} vs {self.away_team})")

    def expected_total(self) -> float:
        """E[home_score + away_score] = E[home] + E[away] (linearity of expectation)."""
        ks_h = np.arange(len(self.home_score_pmf))
        ks_a = np.arange(len(self.away_score_pmf))
        return float(np.dot(ks_h, self.home_score_pmf) + np.dot(ks_a, self.away_score_pmf))

    def expected_home_score(self) -> float:
        ks = np.arange(len(self.home_score_pmf))
        return float(np.dot(ks, self.home_score_pmf))

    def expected_away_score(self) -> float:
        ks = np.arange(len(self.away_score_pmf))
        return float(np.dot(ks, self.away_score_pmf))

    def to_dict(self, line_step: float = 0.5) -> dict[str, Any]:
        """Serializable dict with all game total markets at `line_step` increments."""
        home_mean = self.expected_home_score()
        away_mean = self.expected_away_score()
        total_mean = self.expected_total()

        markets: list[dict[str, Any]] = []
        line = max(100.0, total_mean - 20.0)
        while line <= min(GAME_TOTAL_MAX, total_mean + 20.0):
            p_over = self.total_over(line)
            p_under = self.total_under(line)
            p_push = self.total_push(line)
            markets.append({
                "line": line,
                "p_over": round(p_over, 6),
                "p_under": round(p_under, 6),
                "p_push": round(p_push, 6),
                "fair_over_american": round(fair_american(p_over), 1) if p_over > 0 else None,
                "fair_under_american": round(fair_american(p_under), 1) if p_under > 0 else None,
            })
            line += line_step

        # Team totals at common lines
        home_markets: list[dict] = []
        away_markets: list[dict] = []
        for tl in np.arange(max(60.0, home_mean - 10.0), min(TEAM_SCORE_MAX, home_mean + 10.0), line_step):
            p_o = self._prob_over(self.home_score_pmf, tl)
            p_u = self._prob_under(self.home_score_pmf, tl)
            home_markets.append({"line": round(tl, 1), "p_over": round(p_o, 6), "p_under": round(p_u, 6)})
        for tl in np.arange(max(60.0, away_mean - 10.0), min(TEAM_SCORE_MAX, away_mean + 10.0), line_step):
            p_o = self._prob_over(self.away_score_pmf, tl)
            p_u = self._prob_under(self.away_score_pmf, tl)
            away_markets.append({"line": round(tl, 1), "p_over": round(p_o, 6), "p_under": round(p_u, 6)})

        return {
            "game_id": self.game_id,
            "game_date": self.game_date,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_lambda": round(self.home_lambda, 4),
            "away_lambda": round(self.away_lambda, 4),
            "home_mean": round(home_mean, 3),
            "away_mean": round(away_mean, 3),
            "total_mean": round(total_mean, 3),
            "game_total_markets": markets,
            "home_team_total_markets": home_markets,
            "away_team_total_markets": away_markets,
        }

    def __repr__(self) -> str:
        return (
            f"WNBATeamScorePMFGrid("
            f"{self.home_team} vs {self.away_team}, "
            f"E[home]={self.expected_home_score():.1f}, "
            f"E[away]={self.expected_away_score():.1f})"
        )


# ---------------------------------------------------------------------------
# Weibull Copula Game Totals Model (F4)
# ---------------------------------------------------------------------------

class WNBAWeibullCopulaScoreModel:
    """Thin wrapper around PenaltyBlog's WeibullCopulaGoalsModel for WNBA.

    Requires ``penaltyblog`` package.  Gracefully falls back to returning None
    if the package is unavailable (caller should fall back to NegBinom model).
    """

    VERSION = "stage4_weibull_copula_v1"

    def __init__(self) -> None:
        self._inner = None
        self._teams: set[str] = set()

    def fit(self, df: pd.DataFrame, time_decay_xi: float = 0.002) -> "WNBAWeibullCopulaScoreModel":
        """Fit the Weibull Copula model.

        Args:
            df: DataFrame with columns: home_team, away_team, home_score, away_score,
                days_since (days back from today — for time decay).
            time_decay_xi: Dixon-Coles decay parameter (higher = faster decay).
        """
        if not _PB_AVAILABLE:
            logger.warning("penaltyblog not available — WNBAWeibullCopulaScoreModel not fitted")
            return self

        required = {"home_team", "away_team", "home_score", "away_score"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing columns for WeibullCopula fit: {missing}")

        # Compute time-decay weights. Explicit .copy() on both input and output prevents
        # "buffer source array is read-only" when penaltyblog operates on numpy views.
        if "days_since" in df.columns:
            weights = penaltyblog.models.dixon_coles_weights(
                df["days_since"].to_numpy(copy=True), xi=time_decay_xi
            ).copy()
        else:
            weights = None

        import numpy as _np  # noqa: PLC0415
        self._inner = penaltyblog.models.WeibullCopulaGoalsModel(
            goals_home=_np.array(df["home_score"].values, dtype=int, copy=True),
            goals_away=_np.array(df["away_score"].values, dtype=int, copy=True),
            teams_home=_np.array(df["home_team"].values, dtype=str, copy=True),
            teams_away=_np.array(df["away_team"].values, dtype=str, copy=True),
            weights=weights,
        )
        self._inner.fit()
        self._teams = set(df["home_team"].tolist()) | set(df["away_team"].tolist())
        logger.info(
            "WNBAWeibullCopulaScoreModel fitted on %d games, %d teams",
            len(df), len(self._teams),
        )
        return self

    def predict(
        self,
        home_team: str,
        away_team: str,
        max_score: int = TEAM_SCORE_MAX,
        game_id: Any = None,
        game_date: Any = None,
    ) -> "WNBATeamScorePMFGrid | None":
        """Predict WNBATeamScorePMFGrid for a game.

        Returns None if either team is unknown or model not fitted.
        """
        if self._inner is None:
            return None

        try:
            grid = self._inner.predict(home_team, away_team, max_goals=max_score)
        except Exception as exc:
            logger.warning(
                "WeibullCopula prediction failed for %s vs %s: %s",
                home_team, away_team, exc,
            )
            return None

        # Extract marginal score PMFs from FootballProbabilityGrid
        # home_goal_distribution() / away_goal_distribution() return arrays
        home_pmf = np.array(grid.home_goal_distribution())
        away_pmf = np.array(grid.away_goal_distribution())

        # Pad to max_score + 1 if needed
        if len(home_pmf) < max_score + 1:
            home_pmf = np.pad(home_pmf, (0, max_score + 1 - len(home_pmf)))
        if len(away_pmf) < max_score + 1:
            away_pmf = np.pad(away_pmf, (0, max_score + 1 - len(away_pmf)))

        home_pmf = normalize_pmf(home_pmf[:max_score + 1])
        away_pmf = normalize_pmf(away_pmf[:max_score + 1])
        game_total_pmf = normalize_pmf(convolve_pmfs(home_pmf, away_pmf))

        # Use the expected value as the lambda proxy
        ks_h = np.arange(len(home_pmf))
        ks_a = np.arange(len(away_pmf))
        home_lambda = float(np.dot(ks_h, home_pmf))
        away_lambda = float(np.dot(ks_a, away_pmf))

        return WNBATeamScorePMFGrid(
            home_team=home_team,
            away_team=away_team,
            home_score_pmf=home_pmf,
            away_score_pmf=away_pmf,
            total_score_pmf=game_total_pmf,
            home_lambda=home_lambda,
            away_lambda=away_lambda,
            game_id=game_id,
            game_date=game_date,
        )

    def can_predict(self, home_team: str, away_team: str) -> bool:
        """Return True if both teams were seen in training data."""
        return (self._inner is not None
                and home_team in self._teams
                and away_team in self._teams)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "WNBAWeibullCopulaScoreModel":
        return joblib.load(path)
