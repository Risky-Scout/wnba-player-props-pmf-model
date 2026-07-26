#!/usr/bin/env python3
"""Stage 5 — Build strict walk-forward OOF PMFs.

Generates OOF PMFs using expanding-window chronological splits.
Each validation fold is predicted by a model trained exclusively on
game_date < fold_validation_start_date (strict temporal separation).

Usage:
    python3 scripts/build_oof_pmfs.py \\
      --features-wide data/processed/wnba_player_game_features_wide.parquet \\
      --features-long data/processed/wnba_player_game_features_long.parquet \\
      --manifest data/processed/feature_schema_manifest.json \\
      --config config/model/stage5_oof.yaml \\
      --out-dir data/oof \\
      --audit-out artifacts/audits/stage5_oof_audit.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wnba_props_model.features.feature_contract import assert_no_forbidden_features
from wnba_props_model.models.oof_engine import generate_oof_folds, make_prior_only_pmfs
from wnba_props_model.models.pure_model_contract import (
    assert_pure_feature_columns,
    assert_pure_model_config,
    is_pure_model,
    pure_forecast_provenance,
)
from wnba_props_model.models.training import train_fold, generate_fold_pmfs
from wnba_props_model.models.svd_bridge import SVDBridgeEstimator

app = typer.Typer(add_completion=False)

REQUIRED_PROPS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def _sha256_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p) -> "str | None":
    import hashlib
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fold_input_hash(fold: dict, *, data_hashes: dict, contract_hash: str, config_hash: str,
                     code_sha: str) -> str:
    """Deterministic hash over everything that MUST match for a checkpoint to be reusable."""
    payload = json.dumps({
        "fold_id": fold["fold_id"],
        "train_end": str(fold.get("train_end_date")),
        "val_start": str(fold["val_start_date"]), "val_end": str(fold["val_end_date"]),
        "data_hashes": data_hashes, "contract_hash": contract_hash,
        "config_hash": config_hash, "code_sha": code_sha,
        "encoder_policy": "fold_train_only_ordinal_unknown_-1",
    }, sort_keys=True, default=str)
    return _sha256_bytes(payload.encode())


def _checkpoint_paths(ckpt_dir: Path, fid) -> "tuple[Path, Path]":
    return ckpt_dir / f"fold_{fid}.json", ckpt_dir / f"fold_{fid}.parquet"


def _load_valid_checkpoint(ckpt_dir: Path, fid, input_hash: str):
    """Return the checkpoint's pmf DataFrame iff the meta exists, input_hash matches, the parquet
    exists and its output hash matches, the fit_status is model_oof, and all 7 props are present."""
    meta_p, data_p = _checkpoint_paths(ckpt_dir, fid)
    if not (meta_p.exists() and data_p.exists()):
        return None
    try:
        meta = json.loads(meta_p.read_text())
    except Exception:  # noqa: BLE001
        return None
    if meta.get("input_hash") != input_hash:
        return None
    if meta.get("fit_status") != "model_oof":
        return None
    if _sha256_file(data_p) != meta.get("output_hash"):
        return None
    df = pd.read_parquet(data_p)
    if set(df.get("stat", pd.Series(dtype=str)).unique()) < set(REQUIRED_PROPS):
        return None
    return df


def _write_checkpoint(ckpt_dir: Path, fid, *, pmf_frame, fold, input_hash: str,
                      data_hashes: dict, contract_hash: str, pit_hash: str, config_hash: str,
                      code_sha: str, encoder_hash: str, model_hashes: dict, fit_status: str) -> dict:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_p, data_p = _checkpoint_paths(ckpt_dir, fid)
    if data_p.exists():  # immutable: never rewrite a historical checkpoint
        raise FileExistsError(f"refusing to overwrite existing checkpoint {data_p}")
    tmp = data_p.with_suffix(".parquet.tmp")
    pmf_frame.to_parquet(tmp, index=False)
    import os as _os
    fd = _os.open(str(tmp), _os.O_RDONLY); _os.fsync(fd); _os.close(fd)
    _os.replace(tmp, data_p)
    output_hash = _sha256_file(data_p)
    rows_by_prop = {p: int((pmf_frame["stat"] == p).sum()) for p in REQUIRED_PROPS}
    pmf_ok = True
    try:
        pmf_ok = bool((pmf_frame["pmf_json"].map(lambda s: abs(sum(json.loads(s).values()) - 1.0) < 1e-6)).all())
    except Exception:  # noqa: BLE001
        pmf_ok = False
    meta = {
        "fold_id": fid, "train_date_range": [str(fold.get("train_start_date")), str(fold["train_end_date"])],
        "val_date_range": [str(fold["val_start_date"]), str(fold["val_end_date"])],
        "required_props": REQUIRED_PROPS, "data_hashes": data_hashes,
        "feature_contract_hash": contract_hash, "point_in_time_audit_hash": pit_hash,
        "code_sha": code_sha, "config_hash": config_hash, "encoder_hash": encoder_hash,
        "model_hashes": model_hashes, "rows_by_prop": rows_by_prop, "pmf_integrity_ok": pmf_ok,
        "output_hash": output_hash, "input_hash": input_hash, "fit_status": fit_status,
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
    }
    meta_p.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return meta


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


@app.command()
def build(
    features_wide: Path = typer.Option(
        Path("data/processed/wnba_player_game_features_wide.parquet"),
        "--features-wide",
    ),
    features_long: Path = typer.Option(
        Path("data/processed/wnba_player_game_features_long.parquet"),
        "--features-long",
    ),
    manifest_path: Path = typer.Option(
        Path("data/processed/feature_schema_manifest.json"),
        "--manifest",
    ),
    config_path: Path = typer.Option(
        Path("config/model/stage5_oof.yaml"), "--config"
    ),
    out_dir: Path = typer.Option(Path("data/oof"), "--out-dir"),
    audit_out: Path = typer.Option(
        Path("artifacts/audits/stage5_oof_audit.json"), "--audit-out"
    ),
    max_folds: int = typer.Option(
        0, "--max-folds",
        help="Limit the number of folds (0 = use all folds). Set to 2 for fast fallback run.",
    ),
    svd_bridge: str = typer.Option(
        "", "--svd-bridge",
        help="Path to trained SVDBridgeEstimator pkl. When provided, predicts SVD dims "
             "from leak-free features instead of dropping them from OOF folds.",
    ),
    strict_baseline: bool = typer.Option(
        False, "--strict-baseline",
        help="TRUSTED baseline (Phase 5.3): a prior_only or failed_model_fit fold is FATAL "
             "(the run aborts) instead of silently emitting prior PMFs. Leave off only for a "
             "clearly-labeled diagnostic run.",
    ),
    list_folds: bool = typer.Option(False, "--list-folds",
        help="Print the frozen fold list (id + train/val windows) as JSON and exit."),
    fold_id: int = typer.Option(-1, "--fold-id",
        help="Run ONLY this fold id (for matrix-parallel workflows). -1 = all folds."),
    checkpoint_dir: str = typer.Option("", "--checkpoint-dir",
        help="Directory for immutable per-fold checkpoints (enables --resume)."),
    resume: bool = typer.Option(False, "--resume",
        help="Reuse a fold's checkpoint when ALL its input hashes match (else recompute)."),
    verify_checkpoints: bool = typer.Option(False, "--verify-checkpoints",
        help="Verify every checkpoint's output hash against its parquet and exit."),
) -> None:
    t0 = time.time()
    print("=" * 70)
    print("Stage 5 — Walk-forward OOF PMF Generation")
    print("=" * 70)

    cfg: dict = yaml.safe_load(config_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)

    # Load SVD bridge estimator if provided
    svd_bridge_model: SVDBridgeEstimator | None = None
    if svd_bridge and Path(svd_bridge).exists():
        svd_bridge_model = SVDBridgeEstimator.load(svd_bridge)
        print(f"  SVD bridge loaded from {svd_bridge} "
              f"(dims: {list(svd_bridge_model.bridge_models.keys())})")
    else:
        if svd_bridge:
            print(f"  SVD bridge path '{svd_bridge}' not found — falling back to SVD drop")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading: {features_wide}")
    wide = pd.read_parquet(features_wide)
    wide["game_date"] = pd.to_datetime(wide["game_date"])
    print(f"  {len(wide):,} rows")

    print(f"Loading: {features_long}")
    long = pd.read_parquet(features_long)
    long["game_date"] = pd.to_datetime(long["game_date"])
    print(f"  {len(long):,} rows")

    print(f"Loading manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    model_cols: list[str] = manifest["model_feature_columns"]

    # OOF temporal leakage guard: SVD embeddings are computed on the full wide
    # table (all games in one pass) so they encode future-game information for
    # any given OOF fold.  When a bridge model is provided, SVD dims are predicted
    # from leak-free features instead; otherwise drop them entirely.
    _OOF_EXCLUDED_PREFIXES = ("player_svd_dim_",)
    _svd_cols_in_manifest = [c for c in model_cols
                              if any(c.startswith(p) for p in _OOF_EXCLUDED_PREFIXES)]
    if svd_bridge_model is not None and _svd_cols_in_manifest:
        # Keep SVD cols in model_cols — bridge will inject predicted values per fold
        print(f"  SVD bridge mode: {len(_svd_cols_in_manifest)} SVD cols kept "
              f"(bridge will predict them per fold)")
    else:
        _n_before = len(model_cols)
        model_cols = [c for c in model_cols
                      if not any(c.startswith(p) for p in _OOF_EXCLUDED_PREFIXES)]
        _n_excluded = _n_before - len(model_cols)
        if _n_excluded:
            print(f"  OOF leakage guard: excluded {_n_excluded} SVD embedding columns "
                  f"(fold-unsafe; used only in live Stage 4 predictions)")
    print(f"  {len(model_cols)} model feature columns (OOF-safe)")

    # Leakage guard
    assert_no_forbidden_features(model_cols)
    targets = manifest.get("target_columns", [])
    leaked = [c for c in model_cols if c in targets]
    if leaked:
        raise ValueError(f"Target columns in model_feature_cols: {leaked}")
    print("  Leakage guard: PASS")

    # ------------------------------------------------------------------
    # 1b. Fail-closed pure_forecast guard (STEP 3)
    # ------------------------------------------------------------------
    # The OOF build uses train_fold()/generate_fold_pmfs() (NOT pmf_engine.build_all_pmfs),
    # so the pure contract must be re-enforced HERE before any fold is trained. A pure OOF
    # config carries ZERO market weight/nudge and no market-derived feature column enters the
    # (OOF-safe) model_feature_columns; otherwise this aborts the run.
    assert_pure_model_config(cfg, context="build_oof_pmfs")
    _pure_mode = is_pure_model(cfg)
    pure_provenance: dict | None = None
    if _pure_mode:
        assert_pure_feature_columns(model_cols, context="build_oof_pmfs")
        pure_provenance = pure_forecast_provenance(cfg, model_cols)
        pure_provenance["config_path"] = str(config_path)
        pure_provenance["ordered_feature_list_count"] = len(model_cols)
        prov_out = audit_out.parent / "PURE_OOF_RUN_MANIFEST.json"
        prov_out.write_text(json.dumps(pure_provenance, indent=2, default=str) + "\n")
        print(f"  pure_forecast guard: PASS (information_contract=pure_forecast; "
              f"market_prior_lambda=0.0; CLV disabled)")
        print(f"    config_sha256={pure_provenance['config_sha256']}")
        print(f"    ordered_feature_list_sha256={pure_provenance['ordered_feature_list_sha256']}")
        print(f"  → wrote pure run manifest: {prov_out}")
    else:
        print("  pure_forecast guard: SKIPPED (config not marked pure_model)")

    # ------------------------------------------------------------------
    # 2. Fold-safe encoding (Phase 5.1)
    # ------------------------------------------------------------------
    # NO global encoder fitted on all dates: fitting on the full dataset leaks FUTURE
    # validation categories into the training encoding. Each fold fits its position encoder
    # on TRAINING dates only (inside train_fold); encode_features uses
    # OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), so an unseen
    # validation category maps to the explicit UNKNOWN code (-1) rather than a leaked ordinal.
    print("\nFold-safe encoding: per-fold train-only position encoders (no global leak).")

    # ------------------------------------------------------------------
    # 3. Generate folds
    # ------------------------------------------------------------------
    game_dates_all: list[date] = sorted(
        wide["game_date"].dt.date.unique().tolist()
    )

    # oof_first_val_date restricts VALIDATION windows to the current season.
    # Each fold still trains on all historical data (train_mask uses full wide).
    # Without this filter, 2022-2025 game dates generate ~30 extra folds that
    # push the OOF build past the 360-min GitHub Actions budget.
    first_val_str = cfg.get("oof_first_val_date")
    if first_val_str:
        first_val = date.fromisoformat(str(first_val_str))
        fold_game_dates = [d for d in game_dates_all if d >= first_val]
        print(f"\noof_first_val_date={first_val}: restricting fold windows to "
              f"{len(fold_game_dates)} dates (of {len(game_dates_all)} total)")
    else:
        fold_game_dates = game_dates_all

    folds = generate_oof_folds(fold_game_dates, cfg.get("validation_window_days", 14))
    print(f"\nFolds generated: {len(folds)}")
    if max_folds and max_folds > 0 and len(folds) > max_folds:
        # Keep the MOST RECENT folds — they use the largest training sets
        # and best represent current-season calibration targets.
        folds = folds[-max_folds:]
        print(f"  --max-folds {max_folds}: using last {len(folds)} folds (most recent data)")
    print(f"  First val window: {folds[0]['val_start_date']} – {folds[0]['val_end_date']}")
    print(f"  Last  val window: {folds[-1]['val_start_date']} – {folds[-1]['val_end_date']}")

    # ---- P7: frozen fold list / single-fold selection / checkpoint plumbing ----
    if list_folds:
        print(json.dumps([{"fold_id": f["fold_id"], "val_start": str(f["val_start_date"]),
                           "val_end": str(f["val_end_date"]),
                           "train_end": str(f.get("train_end_date"))} for f in folds],
                         indent=2, default=str))
        raise typer.Exit(0)
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if verify_checkpoints:
        if not ckpt_dir or not ckpt_dir.exists():
            print("[verify-checkpoints] no checkpoint dir", file=sys.stderr); raise typer.Exit(1)
        bad = 0
        for f in folds:
            meta_p, data_p = _checkpoint_paths(ckpt_dir, f["fold_id"])
            if not meta_p.exists():
                print(f"  fold {f['fold_id']}: MISSING"); continue
            meta = json.loads(meta_p.read_text())
            ok = data_p.exists() and _sha256_file(data_p) == meta.get("output_hash")
            print(f"  fold {f['fold_id']}: {'OK' if ok else 'HASH_MISMATCH'}")
            bad += 0 if ok else 1
        raise typer.Exit(1 if bad else 0)
    if fold_id >= 0:
        folds = [f for f in folds if f["fold_id"] == fold_id]
        if not folds:
            print(f"[fold-id] no fold {fold_id}", file=sys.stderr); raise typer.Exit(1)
        print(f"  --fold-id {fold_id}: running a single fold")

    # Reusability hashes shared by all folds this run.
    _data_hashes = {"features_wide": _sha256_file(features_wide),
                    "features_long": _sha256_file(features_long),
                    "manifest": _sha256_file(manifest_path)}
    _contract_hash = __import__("hashlib").sha256("\n".join(model_cols).encode()).hexdigest()
    _config_hash = _sha256_file(config_path)
    _code_sha = _git_commit() or "unknown"
    _pit_audit = REPO_ROOT / "artifacts" / "data_bootstrap" / "FEATURE_POINT_IN_TIME_AUDIT.json" \
        if (REPO_ROOT := Path(__file__).resolve().parent.parent) else None
    _pit_hash = _sha256_file(_pit_audit) if _pit_audit and _pit_audit.exists() else ""

    min_train = cfg.get("min_train_long_rows", 2000)

    # ------------------------------------------------------------------
    # 4. OOF loop
    # ------------------------------------------------------------------
    all_pmf_frames: list[pd.DataFrame] = []
    fold_records: list[dict] = []
    stats = cfg.get("stats", ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"])

    for fold in folds:
        fid = fold["fold_id"]
        val_start: date = fold["val_start_date"]

        # P7 resume: reuse a fully hash-valid checkpoint (skip retraining this fold).
        _input_hash = _fold_input_hash(fold, data_hashes=_data_hashes, contract_hash=_contract_hash,
                                       config_hash=_config_hash, code_sha=_code_sha)
        if resume and ckpt_dir is not None:
            _cached = _load_valid_checkpoint(ckpt_dir, fid, _input_hash)
            if _cached is not None:
                print(f"\n  Fold {fid:2d}: RESUMED from valid checkpoint ({len(_cached):,} rows)")
                all_pmf_frames.append(_cached)
                fold_records.append({"fold_id": fid, "fit_status": "model_oof",
                                     "error_message": "resumed_checkpoint",
                                     "validation_long_rows": int(len(_cached))})
                continue

        # STRICT temporal split: train on game_date < val_start (never <=)
        train_mask_wide = wide["game_date"].dt.date < val_start
        val_mask_wide   = wide["game_date"].dt.date.isin(fold["val_dates"])
        train_mask_long = long["game_date"].dt.date < val_start
        val_mask_long   = long["game_date"].dt.date.isin(fold["val_dates"])

        train_wide_df = wide[train_mask_wide].reset_index(drop=True)
        val_wide_df   = wide[val_mask_wide].reset_index(drop=True)
        train_long_df = long[train_mask_long].reset_index(drop=True)
        val_long_df   = long[val_mask_long].reset_index(drop=True)

        # Derive role_bucket for both splits — not stored in the features parquet.
        # Without it, role-stratified HGB predictions and per-role dispersion are
        # silently bypassed (role_series=None falls back to global model for all rows).
        from wnba_props_model.features.role_buckets import add_ex_ante_role_bucket as _add_rb  # noqa: PLC0415
        if "role_bucket" not in val_wide_df.columns and "player_minutes_mean_l5" in val_wide_df.columns:
            val_wide_df = _add_rb(val_wide_df, minutes_col="player_minutes_mean_l5")

        # SVD bridge: predict SVD dims from leak-free features, or drop them
        _svd_cols = [c for c in train_wide_df.columns if c.startswith("player_svd_dim_")]
        if _svd_cols and svd_bridge_model is not None:
            val_wide_df   = svd_bridge_model.predict(val_wide_df,   use_real_svd=False)
            train_wide_df = svd_bridge_model.predict(train_wide_df, use_real_svd=False)
        elif _svd_cols and svd_bridge_model is None:
            val_wide_df   = val_wide_df.drop(columns=_svd_cols, errors='ignore')
            train_wide_df = train_wide_df.drop(columns=_svd_cols, errors='ignore')

        n_train_long  = len(train_long_df)
        n_val_wide    = len(val_wide_df)
        n_train_games = int(train_wide_df["game_id"].nunique()) if len(train_wide_df) > 0 else 0

        fold_meta_base = {
            "fold_id":         fid,
            "train_start_date": fold.get("train_start_date"),
            "train_end_date":  fold["train_end_date"],
            "val_start_date":  val_start,
            "val_end_date":    fold["val_end_date"],
            "train_wide_rows": len(train_wide_df),
            "train_long_rows": n_train_long,
            "train_games":     n_train_games,
        }

        # Print fold header
        print(f"\n  Fold {fid:2d}: val={val_start}–{fold['val_end_date']}"
              f"  train_rows={n_train_long:,}  val_rows={n_val_wide}")

        if n_val_wide == 0:
            print("    → no validation rows, skipping")
            fold_records.append({**fold_meta_base,
                "fit_status": "skipped", "error_message": "no_val_rows",
                "validation_long_rows": 0})
            continue

        # --- Check eligibility ---
        if n_train_long < min_train:
            if strict_baseline:
                raise RuntimeError(
                    f"[strict-baseline] fold {fid}: insufficient training data "
                    f"({n_train_long} < {min_train}). A trusted baseline may not emit prior_only "
                    "PMFs; aborting (use a diagnostic run without --strict-baseline).")
            print(f"    → insufficient training data ({n_train_long} < {min_train}) → prior_only")
            fold_meta = {**fold_meta_base, "oof_prediction_type": "prior_only"}
            pmf_frame = make_prior_only_pmfs(val_wide_df, val_long_df, fold_meta, cfg)
            fold_records.append({**fold_meta_base,
                "fit_status": "prior_only", "error_message": "",
                "validation_long_rows": len(val_long_df)})
        else:
            try:
                # --- Train fold models (encoder fitted on TRAIN dates only; Phase 5.1) ---
                fold_model = train_fold(train_wide_df, train_long_df, model_cols, cfg)
                # NO global-encoder override: the fold keeps its train-only, unknown-safe encoder.

                fold_meta = {
                    **fold_meta_base,
                    "oof_prediction_type": "model_oof",
                    "train_stat_rows": fold_model.train_stat_rows,
                }

                pmf_frame = generate_fold_pmfs(
                    fold_model, val_wide_df, val_long_df, fold_meta, cfg
                )

                # W0.2: minutes-offset for ast/turnover uses the SAME shared rebuild as the
                # live delivery path (apply_minutes_offset_rebuild) so the PMF itself is rebuilt
                # at the adjusted mean (never a detached stat_mean shift) and OOF == live for
                # identical inputs. pmf_frame must carry a game-date-aligned index reset.
                from wnba_props_model.models.pmf_utils import apply_minutes_offset_rebuild
                from wnba_props_model.models.simulation import json_to_pmf, pmf_to_json
                pmf_frame = pmf_frame.reset_index(drop=True)
                apply_minutes_offset_rebuild(
                    pmf_frame, val_wide_df, to_json=pmf_to_json, from_json=json_to_pmf,
                    stats=("turnover", "ast"),
                )

                status = "model_oof"
                errmsg = ""
                print(f"    → model_oof  PMF rows={len(pmf_frame):,}")
                # P7: write an immutable, hash-keyed checkpoint for this fold.
                if ckpt_dir is not None:
                    try:
                        _enc = getattr(getattr(fold_model, "minutes_model", None), "_pos_encoder", None)
                        _enc_hash = _sha256_bytes(repr(getattr(_enc, "categories_", "")).encode())
                        _write_checkpoint(
                            ckpt_dir, fid, pmf_frame=pmf_frame, fold=fold, input_hash=_input_hash,
                            data_hashes=_data_hashes, contract_hash=_contract_hash,
                            pit_hash=_pit_hash, config_hash=_config_hash, code_sha=_code_sha,
                            encoder_hash=_enc_hash,
                            model_hashes={"train_stat_rows": getattr(fold_model, "train_stat_rows", None)},
                            fit_status="model_oof")
                        print(f"    → checkpoint written: {_checkpoint_paths(ckpt_dir, fid)[1].name}")
                    except FileExistsError:
                        print(f"    → checkpoint already exists for fold {fid} (immutable; kept)")
            except Exception as e:
                import traceback as _tb
                print(f"    → FAILED: {e}")
                _tb.print_exc()  # full stack trace so CI logs show exact cause
                if strict_baseline:
                    raise RuntimeError(
                        f"[strict-baseline] fold {fid}: model fit failed ({e}). A trusted "
                        "baseline may not convert a fit failure into a prior_only PMF; aborting."
                    ) from e
                fold_meta = {**fold_meta_base, "oof_prediction_type": "failed_model_fit"}
                pmf_frame = make_prior_only_pmfs(
                    val_wide_df, val_long_df, fold_meta, cfg, error_msg=str(e)
                )
                status = "failed_model_fit"
                errmsg = str(e)

            fold_records.append({**fold_meta_base,
                "fit_status": status, "error_message": errmsg,
                "validation_long_rows": len(val_long_df)})

        if not pmf_frame.empty:
            all_pmf_frames.append(pmf_frame)

    # ------------------------------------------------------------------
    # 5. Concatenate and validate
    # ------------------------------------------------------------------
    if not all_pmf_frames:
        raise ValueError("No OOF PMF frames generated — check data and config")

    print("\nConcatenating OOF frames...")
    oof_df = pd.concat(all_pmf_frames, ignore_index=True)
    print(f"  Total OOF rows: {len(oof_df):,}")

    # Phase 5.3/5.5: trusted-baseline completeness — no prior_only/failed rows and every
    # direct prop present. (Diagnostic runs skip this and remain explicitly labeled.)
    if strict_baseline:
        _bad_types = oof_df["oof_prediction_type"].isin(["prior_only", "failed_model_fit"])
        if bool(_bad_types.any()):
            raise RuntimeError(
                f"[strict-baseline] {int(_bad_types.sum())} OOF rows are prior_only/failed_model_fit "
                "— a trusted baseline must be 100% model_oof.")
        _required = set(cfg.get("stats", ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]))
        _present = set(oof_df["stat"].astype(str).unique())
        _missing = sorted(_required - _present)
        if _missing:
            raise RuntimeError(f"[strict-baseline] missing direct prop(s) in OOF: {_missing}")

    # Duplicate key check
    dup_count = oof_df.duplicated(subset=["player_id", "game_id", "stat"]).sum()
    if dup_count > 0:
        raise ValueError(f"Duplicate player × game × stat keys: {dup_count}")
    print(f"  Duplicate keys: 0 (PASS)")

    # PMF sum check
    sum_errors = oof_df.apply(
        lambda r: abs(sum(json.loads(r["pmf_json"]).values()) - 1.0), axis=1
    )
    max_err = float(sum_errors.max())
    invalid = int((sum_errors > 1e-6).sum())
    if invalid > 0:
        raise ValueError(f"{invalid} invalid OOF PMFs (sum error > 1e-6)")
    print(f"  Max PMF sum error: {max_err:.2e}  (PASS)")

    # W0.2 fail-closed: minutes-offset stats (ast/turnover) must have the offset baked INTO
    # the PMF (stat_mean == pmf_mean == mean(pmf_json)), never a detached mean shift.
    _offset_mask = oof_df["stat"].isin(["ast", "turnover"])
    if _offset_mask.any():
        _sm = oof_df.loc[_offset_mask, "stat_mean"].astype(float).to_numpy()
        _pm = oof_df.loc[_offset_mask, "pmf_mean"].astype(float).to_numpy()
        _dev = float(np.nanmax(np.abs(_sm - _pm)))
        if _dev > 1e-6:
            raise ValueError(
                f"AST/TOV OOF PMF parity FAILED: max|stat_mean - pmf_mean|={_dev:.3e} > 1e-6 "
                "(detached mean shift detected; the minutes offset must rebuild the PMF).")
        _json_mean = oof_df.loc[_offset_mask, "pmf_json"].map(
            lambda s: sum(int(k) * float(v) for k, v in json.loads(s).items())).to_numpy()
        _dev2 = float(np.nanmax(np.abs(_pm - _json_mean)))
        if _dev2 > 1e-6:
            raise ValueError(
                f"AST/TOV OOF PMF self-consistency FAILED: max|pmf_mean - mean(pmf_json)|="
                f"{_dev2:.3e} > 1e-6.")
        print(f"  AST/TOV PMF parity: max|stat_mean-pmf_mean|={_dev:.2e} (PASS)")

    # Forbidden field check
    for col in ["line", "over_odds", "under_odds", "book", "vendor", "sportsbook"]:
        if col in oof_df.columns:
            raise ValueError(f"Forbidden column '{col}' in OOF output")
    print("  Forbidden field check: PASS")

    # is_calibrated check
    if (oof_df["is_calibrated"] != False).any():  # noqa: E712
        raise ValueError("OOF rows with is_calibrated != False")
    print("  is_calibrated = False: PASS")

    # ------------------------------------------------------------------
    # 5b. Active-PMF lineage (STEP 4): derive the conditional-on-play PMF authoritatively
    # ------------------------------------------------------------------
    # ``pmf_json`` is the availability MIXTURE (DNP mass folded onto outcome 0, plus the
    # ast/turnover minutes-offset rebuild). The sportsbook binary probability must be settled
    # from the ACTIVE (conditional-on-appearance) PMF and p_dnp kept SEPARATE. We recover the
    # active PMF by inverting the DNP fold on the FINAL mixture (canonical recover_active_pmf),
    # so active ⊕ p_dnp is consistent with the delivered mixture for EVERY stat — never the
    # invalid model_prob_over_final/(1-p_dnp) post-hoc shortcut.
    from wnba_props_model.models.availability_pmf import recover_active_pmf  # noqa: PLC0415
    from wnba_props_model.models.simulation import pmf_to_json as _active_to_json  # noqa: PLC0415
    _pdnp_col = oof_df["p_dnp"].fillna(0.0).to_numpy(float) if "p_dnp" in oof_df.columns \
        else np.zeros(len(oof_df))
    _active_jsons, _active_means, _active_vars = [], [], []
    for _js, _d in zip(oof_df["pmf_json"].to_numpy(), _pdnp_col):
        _a = recover_active_pmf(_js, float(_d))
        _k = np.arange(_a.size, dtype=float)
        _mn = float(np.dot(_k, _a))
        _active_jsons.append(_active_to_json(_a))
        _active_means.append(_mn)
        _active_vars.append(float(np.dot(_k * _k, _a) - _mn * _mn))
    oof_df["active_pmf_json"] = _active_jsons
    oof_df["active_pmf_mean"] = _active_means
    oof_df["active_pmf_variance"] = _active_vars
    oof_df["availability_mixture_pmf_json"] = oof_df["pmf_json"].to_numpy()
    oof_df["availability_mixture_mean"] = oof_df["pmf_mean"].to_numpy()
    _mix_mean = oof_df["pmf_mean"].astype(float).to_numpy()
    _amn = np.asarray(_active_means, float)
    _bad_active = int(np.sum(_mix_mean > _amn + 1e-6))
    if _bad_active:
        raise ValueError(
            f"active-PMF lineage FAILED: {_bad_active} rows have mixture mean > active mean "
            "(folding DNP mass onto 0 must not raise the mean).")
    print(f"  active-PMF lineage: PASS (mean(active)-mean(mixture)="
          f"{float(np.mean(_amn - _mix_mean)):.4f}; p_dnp kept separate)")

    # ------------------------------------------------------------------
    # 5c. Persist contract + provenance fields on EVERY OOF row (owner item 2)
    # ------------------------------------------------------------------
    # Line-independent availability + provenance fields. The line-dependent
    # model_prob_over_settled_from_active_pmf / model_prob_over_final are persisted per scored
    # row by scripts/evaluate_pure_oof.py (a PMF has no line until joined to a market quote).
    oof_df["information_contract"] = cfg.get("information_contract")
    oof_df["market_probability_weight"] = float(cfg.get("market_probability_weight", 0.0) or 0.0)
    oof_df["market_prior_lambda"] = float(cfg.get("market_prior_lambda", 0.0) or 0.0)
    oof_df["model_hash"] = _code_sha
    oof_df["config_hash"] = _config_hash
    oof_df["feature_hash"] = _contract_hash
    # OOF is uncalibrated by contract (is_calibrated == False), so the calibrator identity is
    # the explicit sentinel; the enabled binary calibrator is fit + hashed downstream.
    oof_df["calibrator_hash"] = "uncalibrated_oof_is_calibrated_false"

    # ------------------------------------------------------------------
    # 5d. STRICT trusted-baseline gates (owner item 3): fail closed, never fall back
    # ------------------------------------------------------------------
    if strict_baseline:
        if oof_df["active_pmf_json"].isna().any():
            raise RuntimeError("[strict-baseline] OOF rows missing active_pmf_json "
                               "(active-PMF lineage incomplete).")
        if not _pure_mode or float(cfg.get("market_probability_weight", 0.0) or 0.0) != 0.0 \
                or float(cfg.get("market_prior_lambda", 0.0) or 0.0) != 0.0:
            raise RuntimeError("[strict-baseline] OOF config is not pure (market weight != 0); "
                               "a trusted pure baseline must carry zero market weight.")
        _te = pd.to_datetime(oof_df["fold_train_end_date"])
        _vs = pd.to_datetime(oof_df["fold_validation_start_date"])
        _leak = int((_te >= _vs).sum())
        if _leak:
            raise RuntimeError(f"[strict-baseline] {_leak} OOF rows with fold_train_end_date >= "
                               "fold_validation_start_date (temporal leakage).")
        _req = set(cfg.get("stats", REQUIRED_PROPS))
        _missing = sorted(_req - set(oof_df["stat"].astype(str).unique()))
        if _missing:
            raise RuntimeError(f"[strict-baseline] missing direct prop(s) in OOF: {_missing}")
        print("  strict-baseline gates: PASS (100% model_oof, pure, no leakage, active-PMF "
              "present, all props present)")

    # ------------------------------------------------------------------
    # 6. Write outputs
    # ------------------------------------------------------------------
    long_out = out_dir / "oof_player_stat_pmfs.parquet"
    oof_df.to_parquet(long_out, index=False)
    print(f"\nSaved long OOF PMFs: {long_out}")

    # Wide pivot
    wide_out = _build_wide_oof_table(oof_df, stats)
    wide_out.to_parquet(out_dir / "oof_player_stat_pmfs_wide.parquet", index=False)
    print(f"Saved wide OOF PMFs: {out_dir}/oof_player_stat_pmfs_wide.parquet"
          f"  ({len(wide_out):,} rows)")

    # Fold manifest
    fold_df = pd.DataFrame(fold_records)
    fold_df["created_at_utc"] = pd.Timestamp.utcnow()
    fold_df["stats_in_fold"] = json.dumps(stats)
    fold_df["train_wide_rows"] = fold_df.get("train_wide_rows", 0)
    fold_df["validation_wide_rows"] = fold_df.get("validation_long_rows", 0)
    fold_df.to_parquet(out_dir / "oof_fold_manifest.parquet", index=False)
    print(f"Saved fold manifest: {out_dir}/oof_fold_manifest.parquet")

    # ------------------------------------------------------------------
    # 7. Audit
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    n_model_oof  = int((oof_df["oof_prediction_type"] == "model_oof").sum())
    n_prior_only = int((oof_df["oof_prediction_type"] == "prior_only").sum())
    n_failed     = int((oof_df["oof_prediction_type"] == "failed_model_fit").sum())
    n_cal_elig   = int((oof_df["calibration_eligible"] == True).sum())  # noqa: E712
    low_adj      = int(oof_df.get("low_minutes_adjustment_count",
                        pd.Series([0]*len(oof_df))).sum())

    audit = {
        "stage": "stage5_oof",
        "elapsed_seconds": round(elapsed, 1),
        "git_commit": _git_commit(),
        "n_folds": len(folds),
        "n_fold_model_oof": sum(1 for r in fold_records if r["fit_status"] == "model_oof"),
        "n_fold_prior_only": sum(1 for r in fold_records if r["fit_status"] == "prior_only"),
        "n_fold_failed": sum(1 for r in fold_records if r["fit_status"] == "failed_model_fit"),
        "n_fold_skipped": sum(1 for r in fold_records if r["fit_status"] == "skipped"),
        "first_val_date": str(folds[0]["val_start_date"]),
        "last_val_date": str(folds[-1]["val_end_date"]),
        "oof_pmf_rows_total": int(len(oof_df)),
        "oof_pmf_rows_by_stat": {
            s: int((oof_df["stat"] == s).sum()) for s in stats
        },
        "n_model_oof_rows": n_model_oof,
        "n_prior_only_rows": n_prior_only,
        "n_failed_rows": n_failed,
        "n_calibration_eligible_rows": n_cal_elig,
        "duplicate_key_count": int(dup_count),
        "invalid_pmf_count": invalid,
        "max_pmf_sum_error": max_err,
        "is_calibrated_all_false": True,
        "pmf_source_correct": bool(
            (oof_df["pmf_source"] == cfg["pmf_source"]).all()
        ),
        "forbidden_feature_check": "PASS",
        "target_leakage_check": "PASS",
        "same_day_leakage_check": "PASS",
        "low_minutes_adjustment_count": low_adj,
        "model_feature_count": len(model_cols),
        "pure_model": bool(_pure_mode),
        "information_contract": cfg.get("information_contract"),
        "pure_forecast_provenance": pure_provenance,
        "active_pmf_lineage": bool("active_pmf_json" in oof_df.columns),
    }
    audit_out.write_text(json.dumps(audit, indent=2, default=str))
    print(f"\nSaved audit: {audit_out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Stage 5 OOF Generation Complete")
    print(f"  Elapsed:             {elapsed:.1f}s")
    print(f"  Folds:               {len(folds)}")
    print(f"  model_oof rows:      {n_model_oof:,}")
    print(f"  prior_only rows:     {n_prior_only:,}")
    print(f"  calibration_eligible:{n_cal_elig:,}")
    print(f"  OOF PMF rows:        {len(oof_df):,}")
    print(f"  Max sum error:       {max_err:.2e}")
    print(f"  Low-min adjustments: {low_adj:,}")
    print("=" * 70)


def _build_wide_oof_table(oof_df: pd.DataFrame, stats: list[str]) -> pd.DataFrame:
    """Pivot OOF long table to wide (one row per player × game)."""
    id_cols = ["game_id", "game_date", "season", "player_id", "player_name",
               "team_id", "team_abbreviation", "opponent_team_id",
               "actual_minutes", "fold_id", "fold_validation_start_date"]
    metric_cols = ["pmf_mean", "p0", "p_ge_1", "p_ge_5", "stat_mean",
                   "actual_outcome", "calibration_eligible"]

    available_id = [c for c in id_cols if c in oof_df.columns]
    # Ensure player_id and game_id are included (they're already in id_cols; avoid dups)
    for col in ("player_id", "game_id"):
        if col not in available_id:
            available_id.append(col)
    id_df = (oof_df[available_id]
             .drop_duplicates(subset=["player_id", "game_id"]))

    for stat in stats:
        sub = oof_df[oof_df["stat"] == stat]
        if sub.empty:
            continue
        metrics = {c: f"{stat}_{c}" for c in metric_cols if c in sub.columns}
        sub_piv = sub[["player_id", "game_id"] + list(metrics.keys())].rename(
            columns=metrics
        )
        id_df = id_df.merge(sub_piv, on=["player_id", "game_id"], how="left")

    return id_df.reset_index(drop=True)


if __name__ == "__main__":
    app()
