import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

_ROLE_SHRINKAGE_MULT = {
    'starter': 1.0, 'core': 1.2, 'rotation': 1.5,
    'bench': 2.0, 'fringe': 3.0, 'inactive_risk': 4.0,
}

_FALLBACK_MEANS = {
    'pts': 9.5, 'reb': 4.2, 'ast': 2.1, 'fg3m': 1.1,
    'stl': 0.9, 'blk': 0.5, 'turnover': 1.7,
}


class ArchetypeConditionedShrinkage:
    """
    Hierarchical Bayesian shrinkage with archetype-anchored priors.
    Replaces the single-prior James-Stein shrinkage (Stage 6C).

    Key improvements:
    - Per-archetype priors (K-means on SVD dims or stat profile)
    - Smooth alpha = k_eff / (k_eff + n) decay — no hard bypass threshold
    - Role-adjusted k: bench/fringe players get more shrinkage
    - Position-stratified priors for reb/blk/stl
    """

    def __init__(self, n_archetypes: int = 8):
        self.n_archetypes = n_archetypes
        self.archetype_map: dict = {}
        self.archetype_means: dict = {}
        self.archetype_k: dict = {}
        self.position_priors: dict = {}

    def _assign_archetypes(self, df: pd.DataFrame) -> np.ndarray:
        svd_cols = [c for c in df.columns if 'svd_dim' in c.lower()]
        if svd_cols:
            X = df[svd_cols].fillna(0).values
        else:
            profile_cols = [
                'player_pts_mean_season', 'player_reb_mean_season',
                'player_ast_mean_season', 'player_stl_mean_season',
                'player_blk_mean_season', 'player_fg3m_mean_season',
                'player_usage_rate_season',
            ]
            available = [c for c in profile_cols if c in df.columns]
            X = df[available].fillna(0).values

        X_scaled = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=min(self.n_archetypes, len(df)),
                    random_state=42, n_init=20)
        ids = km.fit_predict(X_scaled)

        if 'player_id' in df.columns:
            for pid in df['player_id'].unique():
                mask = (df['player_id'] == pid).values
                if mask.any():
                    self.archetype_map[str(pid)] = int(ids[mask][-1])
        return ids

    def fit(self, df: pd.DataFrame,
            stats: list = None) -> 'ArchetypeConditionedShrinkage':
        if stats is None:
            stats = ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'turnover']

        archetype_ids = self._assign_archetypes(df)
        df = df.copy()
        df['_archetype'] = archetype_ids

        for stat in stats:
            if stat not in df.columns or 'player_id' not in df.columns:
                continue

            # Use separate agg calls to avoid pandas dict-in-kwargs incompatibility
            grp = df.groupby('player_id')
            game_col = 'game_id' if 'game_id' in df.columns else stat
            n_games = grp[game_col].count().rename('n_games')
            total_stat = grp[stat].sum().rename('total_stat')
            archetype_col = grp['_archetype'].last().rename('archetype')
            player_stats = pd.concat([n_games, total_stat, archetype_col], axis=1).reset_index()
            player_stats['rate'] = (player_stats['total_stat'] /
                                    player_stats['n_games'].clip(lower=1))

            for arch in sorted(player_stats['archetype'].dropna().unique()):
                mask = ((player_stats['archetype'] == arch) &
                        (player_stats['n_games'] >= 3))
                rates = player_stats.loc[mask, 'rate'].dropna().values
                if len(rates) < 10:
                    rates = player_stats['rate'].dropna().values
                if len(rates) == 0:
                    continue
                mu = float(np.mean(rates))
                sigma2 = float(np.var(rates))
                beta = mu / sigma2 if sigma2 > 0 else 10.0
                self.archetype_means[(stat, int(arch))] = mu
                self.archetype_k[(stat, int(arch))] = beta

        if 'player_position' in df.columns:
            for stat in ['reb', 'blk', 'stl']:
                if stat not in df.columns:
                    continue
                for pos in df['player_position'].dropna().unique():
                    vals = df.loc[df['player_position'] == pos, stat].dropna()
                    if len(vals) > 20:
                        self.position_priors[(stat, str(pos))] = {
                            'mean': float(vals.mean()),
                            'std': float(vals.std()),
                        }
        return self

    def shrink_pmf(self, pmf: np.ndarray, player_id: str, stat: str,
                   n_games: int, role_bucket: str = 'starter',
                   cap: int = 60) -> np.ndarray:
        archetype = self.archetype_map.get(str(player_id), 0)
        k_base = self.archetype_k.get((stat, archetype), 10.0)
        role_mult = _ROLE_SHRINKAGE_MULT.get(role_bucket, 2.0)
        k_eff = k_base * role_mult
        alpha = k_eff / (k_eff + n_games)  # smooth decay, no hard cutoff

        if alpha < 0.01:
            return pmf

        prior_mean = self.archetype_means.get(
            (stat, archetype), _FALLBACK_MEANS.get(stat, 5.0))
        prior_pmf = np.array([poisson.pmf(k, prior_mean) for k in range(cap + 1)])
        prior_pmf /= prior_pmf.sum()

        # Align lengths
        n = min(len(pmf), len(prior_pmf))
        shrunk = alpha * prior_pmf[:n] + (1 - alpha) * pmf[:n]
        return shrunk / shrunk.sum()

    def save(self, path: str):
        joblib.dump(self.__dict__, path)

    @classmethod
    def load(cls, path: str) -> 'ArchetypeConditionedShrinkage':
        data = joblib.load(path)
        obj = cls()
        obj.__dict__.update(data)
        return obj
