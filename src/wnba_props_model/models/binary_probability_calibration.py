"""Fail-closed binary-probability calibration registry (PR 1A B5).

The interface is introduced now with IDENTITY as the only enabled implementation. No
binary calibrators are trained in PR 1A. When calibration is configured disabled, the
registry returns the input probability unchanged with status ``identity_disabled``.

When configured enabled, the registry is strictly fail-closed:
  * missing artifact                      -> fatal (CalibrationError)
  * artifact hash mismatch                -> fatal
  * unsupported artifact schema           -> fatal
  * wrong prop mapping                    -> fatal
  * undeclared role fallback              -> fatal
  * NaN / infinite output                 -> fatal
  * output outside [0, 1]                 -> fatal
  * silent return of the raw probability after any failure -> forbidden

There is intentionally no ``except: return raw`` path.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CALIBRATION_LINEAGE_VERSION = "binary-cal-v1"


class CalibrationError(RuntimeError):
    """Fatal, fail-closed binary-calibration error."""


@dataclass(frozen=True)
class CalibrationResult:
    p_calibrated: float
    calibration_status: str          # "identity_disabled" | "calibrated"
    calibrator_id: str | None
    calibrator_hash: str | None


@dataclass
class BinaryCalibrationRegistry:
    """Registry of per-(prop, role) binary calibrators with a fallback hierarchy.

    Parameters
    ----------
    enabled : master switch. When False, every apply() is identity.
    artifacts : mapping "prop|role" or "prop" -> {"path": str, "sha256": str}. Only
        consulted when enabled.
    allow_role_fallback_to_prop : when True, a missing (prop, role) may fall back to the
        prop-level calibrator. This must be DECLARED here; an undeclared fallback is fatal.
    """
    enabled: bool = False
    artifacts: dict[str, dict[str, str]] = field(default_factory=dict)
    allow_role_fallback_to_prop: bool = False
    _loaded: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def status(self) -> str:
        return "enabled" if self.enabled else "identity_disabled"

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _resolve_key(self, prop: str, role: str) -> str:
        role_key = f"{prop}|{role}"
        if role_key in self.artifacts:
            return role_key
        if prop in self.artifacts:
            if not self.allow_role_fallback_to_prop:
                raise CalibrationError(
                    f"undeclared role fallback for {role_key}: enable allow_role_fallback_to_prop "
                    "to permit the prop-level calibrator")
            return prop
        raise CalibrationError(f"no binary calibrator registered for {role_key} or {prop}")

    def _load(self, key: str) -> Any:
        if key in self._loaded:
            return self._loaded[key]
        import joblib  # noqa: PLC0415
        spec = self.artifacts[key]
        path = Path(spec["path"])
        if not path.exists():
            raise CalibrationError(f"calibration artifact missing: {path}")
        expected = spec.get("sha256")
        if not expected:
            raise CalibrationError(f"calibration artifact has no pinned sha256: {key}")
        actual = self._sha256(path)
        if actual != expected:
            raise CalibrationError(
                f"calibration artifact hash mismatch for {key}: "
                f"expected {expected[:12]} actual {actual[:12]}")
        try:
            model = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - surface as fatal, never silent-fallback
            raise CalibrationError(f"failed to load calibrator {key}: {exc}") from exc
        if not hasattr(model, "predict"):
            raise CalibrationError(f"unsupported calibrator schema for {key}: no predict()")
        self._loaded[key] = model
        return model

    def apply(self, prop: str, role: str, p_over: float) -> CalibrationResult:
        if not (isinstance(p_over, (int, float)) and math.isfinite(p_over)):
            raise CalibrationError(f"input probability not finite: {p_over!r}")
        if not (0.0 <= float(p_over) <= 1.0):
            raise CalibrationError(f"input probability outside [0,1]: {p_over}")

        if not self.enabled:
            return CalibrationResult(float(p_over), "identity_disabled", None, None)

        key = self._resolve_key(prop, role)
        model = self._load(key)
        try:
            out = float(model.predict([[float(p_over)]])[0])
        except Exception as exc:  # noqa: BLE001
            raise CalibrationError(f"calibrator predict failed for {key}: {exc}") from exc
        if not math.isfinite(out):
            raise CalibrationError(f"calibrator produced non-finite output for {key}: {out}")
        if not (0.0 <= out <= 1.0):
            raise CalibrationError(f"calibrator output outside [0,1] for {key}: {out}")
        cal_hash = self.artifacts[key].get("sha256")
        return CalibrationResult(out, "calibrated", key, cal_hash)

    # --- construction from a versioned policy file (enabled path only) ---
    @classmethod
    def from_policy(cls, policy_path: str | Path | None) -> "BinaryCalibrationRegistry":
        """Build from config/binary_calibration_policy_v1.json when present; otherwise a
        disabled identity registry. PR 1A ships no policy file, so this returns identity."""
        if policy_path is None:
            return cls(enabled=False)
        p = Path(policy_path)
        if not p.exists():
            return cls(enabled=False)
        cfg = json.loads(p.read_text())
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            artifacts=cfg.get("artifacts", {}),
            allow_role_fallback_to_prop=bool(cfg.get("allow_role_fallback_to_prop", False)),
        )
