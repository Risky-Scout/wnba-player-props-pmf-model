import numpy as np
from sklearn.isotonic import IsotonicRegression
import joblib


class VennAbersCalibrator:
    """
    Venn-Abers predictor for distribution-free calibration validity.
    Replaces single isotonic regression in the calibration chain.
    """

    def __init__(self):
        self.iso_0 = None
        self.iso_1 = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'VennAbersCalibrator':
        sort_idx = np.argsort(scores)
        s_sorted = scores[sort_idx]
        l_sorted = labels[sort_idx]

        self.iso_0 = IsotonicRegression(out_of_bounds='clip')
        self.iso_0.fit(np.append(s_sorted, s_sorted[-1]),
                       np.append(l_sorted, 0))

        self.iso_1 = IsotonicRegression(out_of_bounds='clip')
        self.iso_1.fit(np.append(s_sorted, s_sorted[-1]),
                       np.append(l_sorted, 1))
        return self

    def predict(self, scores: np.ndarray):
        p0 = self.iso_0.predict(scores)
        p1 = self.iso_1.predict(scores)
        denom = p1 + (1 - p0)
        p_cal = np.where(denom > 0, p1 / denom, scores)
        return p_cal, p0, p1

    def save(self, path: str):
        joblib.dump({'iso_0': self.iso_0, 'iso_1': self.iso_1}, path)

    @classmethod
    def load(cls, path: str) -> 'VennAbersCalibrator':
        data = joblib.load(path)
        obj = cls()
        obj.iso_0 = data['iso_0']
        obj.iso_1 = data['iso_1']
        return obj


class PerRoleVarianceCompressor:
    """
    Per-role variance compression. Replaces global compression factor.
    Uses direct ratio Var_model / Var_actual (NOT sqrt — sqrt under-corrects).
    """

    def __init__(self, min_samples: int = 50):
        self.role_factors: dict = {}
        self.player_factors: dict = {}
        self.min_samples = min_samples

    def fit(self, oof_preds: np.ndarray, actuals: np.ndarray,
            roles: np.ndarray, players: np.ndarray,
            stat: str = 'pts') -> 'PerRoleVarianceCompressor':
        for role in np.unique(roles):
            mask = roles == role
            if mask.sum() < self.min_samples:
                continue
            var_model = np.var(oof_preds[mask])
            var_actual = np.var(actuals[mask])
            if var_actual > 0:
                factor = var_model / var_actual  # direct ratio, NOT sqrt
                self.role_factors[(stat, str(role))] = float(np.clip(factor, 0.5, 2.0))

        for pid in np.unique(players):
            mask = players == pid
            if mask.sum() < 20:
                continue
            var_model = np.var(oof_preds[mask])
            var_actual = np.var(actuals[mask])
            if var_actual > 0:
                factor = var_model / var_actual
                self.player_factors[(stat, str(pid))] = float(np.clip(factor, 0.5, 2.0))

        return self

    def get_factor(self, stat: str, role: str, player_id: str = None) -> float:
        if player_id and (stat, str(player_id)) in self.player_factors:
            return self.player_factors[(stat, str(player_id))]
        return self.role_factors.get((stat, str(role)), 1.0)

    def compress_pmf(self, pmf: np.ndarray, factor: float) -> np.ndarray:
        """
        Variance compression via PMF-mean mixture. Does NOT use np.roll
        (np.roll wraps boundary mass incorrectly for count PMFs).
        """
        if abs(factor - 1.0) < 0.01:
            return pmf
        support = np.arange(len(pmf))
        pmf_mean = float(np.sum(support * pmf))
        mean_idx = min(int(round(pmf_mean)), len(pmf) - 1)

        if factor > 1.0:
            alpha = 1.0 - 1.0 / factor
            compressed = (1 - alpha) * pmf.copy()
            compressed[mean_idx] += alpha
        else:
            alpha = max(0.0, 1.0 - factor)
            compressed = pmf.copy()
            peak_idx = int(np.argmax(compressed))
            mass = alpha * compressed[peak_idx]
            compressed[peak_idx] -= mass
            neighbors = [i for i in [peak_idx - 1, peak_idx + 1]
                         if 0 <= i < len(compressed)]
            if neighbors:
                per_n = mass / len(neighbors)
                for n in neighbors:
                    compressed[n] += per_n

        total = compressed.sum()
        return compressed / total if total > 0 else pmf

    def save(self, path: str):
        joblib.dump(self.__dict__, path)

    @classmethod
    def load(cls, path: str) -> 'PerRoleVarianceCompressor':
        data = joblib.load(path)
        obj = cls()
        obj.__dict__.update(data)
        return obj
