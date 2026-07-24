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


@dataclass
class VennAbersBinaryCalibrationRegistry:
    """Binary-calibration registry backed by the existing per-(stat, role) Venn-Abers
    artifacts (``venn_abers_{stat}_{role}.pkl``). Implements the same ``apply`` contract as
    ``BinaryCalibrationRegistry`` so it plugs into build_probability_lineage's binary
    calibration stage - VA is applied BEFORE model_prob_over_final is created (never as a
    post-lineage mutation).

    Fail-closed: a NaN/out-of-range input or a calibrator that produces a NaN/out-of-range
    value is fatal. A missing (stat, role) calibrator uses the declared identity fallback
    only when ``allow_missing_calibrator_identity`` is True; when ``require`` is True a
    missing calibrator is fatal. Exceptions during predict are never swallowed into a raw
    fallback.
    """
    cal_dir: str | Path
    require: bool = False
    allow_missing_calibrator_identity: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def status(self) -> str:
        return "venn_abers"

    @staticmethod
    def _role_safe(role: str) -> str:
        return str(role).replace("/", "_").replace(" ", "_")

    def _artifact_path(self, prop: str, role: str) -> Path:
        return Path(self.cal_dir) / f"venn_abers_{prop}_{self._role_safe(role)}.pkl"

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def apply(self, prop: str, role: str, p_over: float) -> CalibrationResult:
        if not (isinstance(p_over, (int, float)) and math.isfinite(p_over)):
            raise CalibrationError(f"input probability not finite: {p_over!r}")
        if not (0.0 <= float(p_over) <= 1.0):
            raise CalibrationError(f"input probability outside [0,1]: {p_over}")

        path = self._artifact_path(prop, role)
        if not path.exists():
            if self.require:
                raise CalibrationError(f"required Venn-Abers calibrator missing: {path}")
            if self.allow_missing_calibrator_identity:
                return CalibrationResult(float(p_over), "identity_no_calibrator", None, None)
            raise CalibrationError(f"no Venn-Abers calibrator for {prop}|{role} and identity "
                                   "fallback not permitted")
        key = path.name
        cal = self._cache.get(key)
        if cal is None:
            from wnba_props_model.calibration.venn_abers import VennAbersCalibrator  # noqa: PLC0415
            try:
                cal = VennAbersCalibrator.load(str(path))
            except Exception as exc:  # noqa: BLE001 - fatal, never silent-fallback
                raise CalibrationError(f"failed to load VA calibrator {key}: {exc}") from exc
            self._cache[key] = cal
        try:
            import numpy as _np  # noqa: PLC0415
            pred = cal.predict(_np.array([float(p_over)]))
            p_cal_arr = pred[0] if isinstance(pred, tuple) else pred
            out = float(p_cal_arr[0]) if hasattr(p_cal_arr, "__len__") else float(p_cal_arr)
        except Exception as exc:  # noqa: BLE001
            raise CalibrationError(f"VA predict failed for {key}: {exc}") from exc
        if not math.isfinite(out):
            raise CalibrationError(f"VA calibrator produced non-finite output for {key}: {out}")
        if not (0.0 <= out <= 1.0):
            raise CalibrationError(f"VA calibrator output outside [0,1] for {key}: {out}")
        return CalibrationResult(out, "calibrated", key, self._sha256(path))


# ---------------------------------------------------------------------------
# W0.6: ONE required-mode policy resolver + per-prop registry
# ---------------------------------------------------------------------------

@dataclass
class PolicyBinaryCalibrationRegistry:
    """Per-prop binary-calibration registry driven by the W0.5 policy file
    ({props: {stat: {method, path, sha256}}}). Implements the same
    ``apply(prop, role, p_over) -> CalibrationResult`` contract as BinaryCalibrationRegistry
    so it plugs into build_probability_lineage. ONE instance is loaded by delivery, market
    comparison, proof, scoring, and OOF via ``load_binary_calibration_registry`` so the
    calibrated probability is identical everywhere.

    mode:
      * "disabled" -> always identity (passthrough).
      * "optional" -> identity when policy/prop-entry/artifact is missing (never fatal).
      * "required" -> missing policy, missing prop entry, missing/invalid artifact, or
                      sha256 mismatch is FATAL (certified operation).
    """
    props: dict = field(default_factory=dict)
    mode: str = "optional"
    policy_id: str | None = None
    policy_hash: str | None = None
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def status(self) -> str:
        return f"policy:{self.mode}"

    def _load_calibrator(self, spec: dict, key: str):
        if key in self._cache:
            return self._cache[key]
        import joblib  # noqa: PLC0415
        path = Path(spec["path"])
        if not path.exists():
            raise CalibrationError(f"calibrator artifact missing: {path}")
        expected = spec.get("sha256")
        if not expected:
            raise CalibrationError(f"calibrator has no pinned sha256: {key}")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected:
            raise CalibrationError(
                f"calibrator hash mismatch for {key}: expected {expected[:12]} actual {actual[:12]}")
        model = joblib.load(path)
        if not hasattr(model, "predict"):
            raise CalibrationError(f"calibrator {key} has no predict()")
        self._cache[key] = model
        return model

    def apply(self, prop: str, role: str, p_over: float) -> CalibrationResult:  # noqa: ARG002
        if not (isinstance(p_over, (int, float)) and math.isfinite(p_over)):
            raise CalibrationError(f"input probability not finite: {p_over!r}")
        if not (0.0 <= float(p_over) <= 1.0):
            raise CalibrationError(f"input probability outside [0,1]: {p_over}")
        if self.mode == "disabled":
            return CalibrationResult(float(p_over), "identity_disabled", None, None)

        entry = self.props.get(prop)
        if entry is None:
            if self.mode == "required":
                raise CalibrationError(f"required policy has no entry for prop {prop!r}")
            return CalibrationResult(float(p_over), "identity_no_policy_entry", None, None)

        method = entry.get("method", "identity")
        if method == "identity":
            return CalibrationResult(float(p_over), "identity", f"identity:{prop}", None)

        # Non-identity: load and apply the calibrator (fail-closed in required mode).
        try:
            model = self._load_calibrator(entry, f"{prop}:{method}")
            out = float(model.predict([[float(p_over)]])[0])
        except CalibrationError:
            if self.mode == "required":
                raise
            return CalibrationResult(float(p_over), "identity_calibrator_unavailable", None, None)
        except Exception as exc:  # noqa: BLE001
            if self.mode == "required":
                raise CalibrationError(f"calibrator predict failed for {prop}:{method}: {exc}") from exc
            return CalibrationResult(float(p_over), "identity_calibrator_error", None, None)
        if not math.isfinite(out) or not (0.0 <= out <= 1.0):
            raise CalibrationError(f"calibrator output invalid for {prop}:{method}: {out}")
        return CalibrationResult(out, "calibrated", f"{method}:{prop}", entry.get("sha256"))


def load_binary_calibration_registry(policy_path, mode: str = "required"):
    """The single loader used by delivery, market comparison, proof, scoring, and OOF.

    Returns a PolicyBinaryCalibrationRegistry with the given ``mode``. In "required" mode a
    missing policy file is FATAL; in "optional"/"disabled" a missing policy yields identity.
    """
    if mode not in ("disabled", "optional", "required"):
        raise ValueError(f"invalid mode {mode!r} (disabled|optional|required)")
    p = Path(policy_path) if policy_path else None
    if p is None or not p.exists():
        if mode == "required":
            raise CalibrationError(f"required binary-calibration policy not found: {policy_path}")
        return PolicyBinaryCalibrationRegistry(props={}, mode=mode)
    cfg = json.loads(p.read_text())
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return PolicyBinaryCalibrationRegistry(
        props=cfg.get("props", {}), mode=mode,
        policy_id=cfg.get("version"), policy_hash=h)
