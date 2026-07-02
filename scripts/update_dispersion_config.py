#!/usr/bin/env python3
"""Auto-update dispersion_r_by_role in stage4_baseline.yaml from OOF data.

Computes r = mu**2 / max(var - mu, 1e-4) per (stat, role) from current OOF,
then updates the YAML config file in-place using ruamel.yaml to preserve comments.

Usage:
    python scripts/update_dispersion_config.py \
        --oof-path data/oof/oof_player_stat_pmfs.parquet \
        --config-path config/model/stage4_baseline.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ruamel.yaml import YAML
    _yaml = YAML()
    _yaml.preserve_quotes = True
    _yaml.default_flow_style = False
    _USE_RUAMEL = True
except ImportError:
    _USE_RUAMEL = False
    print("WARNING: ruamel.yaml not available — yaml.dump will be used (comments lost)")


def compute_per_role_r(oof: pd.DataFrame) -> dict:
    if "role_bucket" not in oof.columns:
        return {}
    results: dict = {}
    direct_stats = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]
    for stat in oof["stat"].unique():
        if stat not in direct_stats:
            continue
        stat_rows = oof[oof["stat"] == stat]
        for role in stat_rows["role_bucket"].unique():
            role_rows = stat_rows[stat_rows["role_bucket"] == role]
            if len(role_rows) < 30:
                continue
            if "pmf_mean" not in role_rows.columns or "pmf_variance" not in role_rows.columns:
                continue
            mu = float(role_rows["pmf_mean"].mean())
            var = float(role_rows["pmf_variance"].mean())
            if var <= mu + 1e-4:
                continue  # near-Poisson, no overdispersion to capture
            r_val = round(mu**2 / max(var - mu, 1e-4), 4)
            r_val = max(0.3, min(20.0, r_val))  # clamp to sensible range
            results.setdefault(stat, {})[role] = r_val
    return results


def update_yaml_config(yaml_path: Path, r_block: dict) -> None:
    if _USE_RUAMEL:
        from ruamel.yaml import YAML as _YAML
        yaml_inst = _YAML()
        yaml_inst.preserve_quotes = True
        yaml_inst.default_flow_style = False
        with open(yaml_path) as f:
            cfg = yaml_inst.load(f)
        cfg["dispersion_r_by_role"] = r_block
        with open(yaml_path, "w") as f:
            yaml_inst.dump(cfg, f)
    else:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        cfg["dispersion_r_by_role"] = r_block
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Updated dispersion_r_by_role in {yaml_path}")
    for stat, roles in r_block.items():
        for role, r in roles.items():
            print(f"  {stat}/{role}: r={r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-path", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    oof = pd.read_parquet(args.oof_path)
    if "calibration_eligible" in oof.columns:
        oof = oof[oof["calibration_eligible"] == True]  # noqa: E712

    r_values = compute_per_role_r(oof)
    if not r_values:
        print("ERROR: No per-role r values computed. Check OOF data.")
        sys.exit(1)

    update_yaml_config(Path(args.config_path), r_values)


if __name__ == "__main__":
    main()
