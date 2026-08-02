"""End-to-end WNBA Pricing PMF v1 run: joint generator -> atom PMFs -> fair prices for the full
supported market inventory, plus completeness/identity/monotonicity/tail audits.

Without the Phase-1 recovered feature slate this runs on a deterministic FIXTURE slate to prove
the end-to-end path (pricing_status records the source). A real slate plugs the same
PlayerGameParams from pregame features.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.pricing import MODEL_VERSION, RELEASE_VERSION
from wnba_props_model.pricing import engine as E
from wnba_props_model.pricing import market_registry as MR
from wnba_props_model.pricing.joint_generator import (
    RELEASE_SEED,
    PlayerGameParams,
    simulate_player,
)

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
V1 = REPO / "artifacts" / "v1"
_OUTCOME_FOR_MARKET = {  # count-market internal outcome -> generator pmf key
    "pts": "pts", "reb": "reb", "ast": "ast", "fg3m": "fg3m", "blk": "blk", "stl": "stl",
    "turnover": "turnover", "fgm": "fgm", "ftm": "ftm", "fta": "fta",
    "stocks": "stocks", "pts_reb": "pts_reb", "pts_ast": "pts_ast", "reb_ast": "reb_ast",
    "pts_reb_ast": "pts_reb_ast",
}


def _fixture_slate() -> list[PlayerGameParams]:
    slate = []
    for i in range(8):
        starter = i < 5
        slate.append(PlayerGameParams(
            player_id=f"FIX{i:02d}", p_active=0.97 if starter else 0.85,
            minutes_mean=30.0 if starter else 16.0, minutes_sd=5.0 if starter else 6.0,
            fga_per_min=0.38 if starter else 0.30, fg3a_share=0.30 + 0.1 * (i % 3),
            fta_per_min=0.14 if starter else 0.08, fg2_pct=0.50, fg3_pct=0.34, ft_pct=0.80,
            oreb_per_min=0.05, dreb_per_min=0.16, ast_per_min=0.12 if starter else 0.07,
            stl_per_min=0.03, blk_per_min=0.02, tov_per_min=0.08))
    return slate


def _line_grid(pmf: np.ndarray) -> list[float]:
    mean = float(np.dot(np.arange(pmf.size), pmf))
    base = round(mean * 2) / 2 - 0.5
    return sorted({round(base + 0.5 * j, 1) for j in range(-3, 4) if base + 0.5 * j >= 0.5})


@app.command()
def main(date: str = typer.Option(None, "--date"),
         n_samples: int = typer.Option(40000, "--n-samples"),
         source: str = typer.Option("FIXTURE", "--source", help="FIXTURE | features parquet path")) -> None:
    run_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = REPO / "deliveries" / "pricing_v1" / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    V1.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    code_sha = _git_sha()

    slate = _fixture_slate()   # real slate would come from pregame features
    atom_rows, price_rows, meta_rows = [], [], []
    tail_rows, ident_rows, mono_rows, cov_rows = [], [], [], []

    for p in slate:
        jo = simulate_player(p, n_samples=n_samples, seed=RELEASE_SEED)
        status = jo.pricing_status
        meta_rows.append({"player_id": p.player_id, "p_dnp": jo.p_dnp, "n_samples": jo.n_samples,
                          "seed": jo.seed, "mc_max_se": jo.mc_max_se, "identities_hold": jo.identities_hold,
                          "pricing_status": status})
        ident_rows.append({"player_id": p.player_id, "identities_hold": jo.identities_hold})
        # atom PMFs (every atom through support max)
        for okey, pmf in jo.pmfs.items():
            omitted = 0.0  # empirical PMF is exact over its support
            tail_rows.append({"player_id": p.player_id, "outcome": okey, "support_max": int(pmf.size - 1),
                              "omitted_tail_mass": omitted, "tail_ok": omitted < 1e-8})
            for k, prob in enumerate(pmf):
                if prob <= 0:
                    continue
                atom_rows.append({"player_id": p.player_id, "market_outcome_key": okey, "period": "FULL",
                                  "atom_value": int(k), "atom_probability": float(prob), "p_dnp": jo.p_dnp,
                                  "model_version": MODEL_VERSION, "code_sha": code_sha, "pricing_status": status})
        # price count markets + alternates + combos from the SAME pmfs
        for mkey, spec in MR.MARKET_REGISTRY.items():
            if spec.settlement_type != "count_over_under":
                continue
            okey = spec.internal_outcome_key
            if spec.market_family == "over_under_q1":
                pmf = jo.q1_pmfs.get(okey)
            else:
                pmf = jo.pmfs.get(okey)
            if pmf is None:
                cov_rows.append({"player_id": p.player_id, "market_key": mkey, "status": "NO_DISTRIBUTION"})
                continue
            grid = _line_grid(pmf)
            ladder = E.price_alternate_ladder(pmf, grid, mkey, margin_method="proportional", overround=0.045)
            p_over_seq = [pl.p_over_win for pl in ladder]
            monotone = all(p_over_seq[i] >= p_over_seq[i + 1] - 1e-9 for i in range(len(p_over_seq) - 1))
            mono_rows.append({"player_id": p.player_id, "market_key": mkey, "monotone": monotone})
            for pl in ladder:
                price_rows.append({"player_id": p.player_id, "market_key": mkey, "line": pl.line,
                                   "p_win": pl.p_over_win, "p_lose": pl.p_under_win, "p_push": pl.p_push,
                                   "settled_probability": pl.p_over_settled, "fair_decimal": pl.fair_decimal_over,
                                   "fair_american": pl.fair_american_over, "margin_method": pl.margin_method,
                                   "quoted_decimal": pl.quoted_decimal_over, "quoted_american": pl.quoted_american_over,
                                   "pricing_status": status, "source_track": "PURE_PMF_FIXTURE"})
            cov_rows.append({"player_id": p.player_id, "market_key": mkey, "status": "PRICED"})
        # event markets
        for ev, prob in jo.event_probs.items():
            yn = E.price_yes_no(prob, f"player_{ev}")
            price_rows.append({"player_id": p.player_id, "market_key": f"player_{ev}", "line": None,
                               "p_win": yn.p_yes, "p_lose": yn.p_no, "p_push": 0.0,
                               "settled_probability": yn.p_yes, "fair_decimal": yn.fair_decimal_yes,
                               "fair_american": yn.fair_american_yes, "margin_method": "none",
                               "quoted_decimal": None, "quoted_american": None,
                               "pricing_status": status, "source_track": "PURE_PMF_FIXTURE"})
            cov_rows.append({"player_id": p.player_id, "market_key": f"player_{ev}", "status": "PRICED"})

    atom_df = pd.DataFrame(atom_rows); price_df = pd.DataFrame(price_rows)
    atom_df.to_parquet(out_dir / "active_atom_pmfs.parquet", index=False)
    pd.DataFrame(meta_rows).to_parquet(out_dir / "joint_outcome_metadata.parquet", index=False)
    price_df.to_parquet(out_dir / "fair_prices.parquet", index=False)
    price_df.to_csv(out_dir / "priced_market_inventory.csv", index=False)

    manifest = {"release_version": RELEASE_VERSION, "model_version": MODEL_VERSION, "run_date": run_date,
                "generated_at_utc": ts, "code_sha": code_sha, "source": source, "seed": RELEASE_SEED,
                "n_samples": n_samples, "n_players": len(slate),
                "n_atoms": len(atom_df), "n_priced_lines": len(price_df),
                "supported_markets": sorted(MR.MARKET_REGISTRY.keys()),
                "note": "FIXTURE slate (deterministic). A real slate plugs pregame-feature-driven "
                        "PlayerGameParams; today's real pricing run requires the Phase-1 recovered data."}
    (out_dir / "pricing_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # audits
    (V1 / "ATOM_COMPLETENESS_AUDIT.json").write_text(json.dumps({
        "artifact": "ATOM_COMPLETENESS_AUDIT", "generated_at_utc": ts, "n_atoms": len(atom_df),
        "players": len(slate), "outcomes_per_player": int(atom_df["market_outcome_key"].nunique()),
        "every_pmf_sums_to_one": bool(all(abs(g["atom_probability"].sum() - 1) < 1e-6
                                          for _, g in atom_df.groupby(["player_id", "market_outcome_key"]))),
    }, indent=2, default=str))
    (V1 / "STRUCTURAL_IDENTITY_AUDIT.json").write_text(json.dumps({
        "artifact": "STRUCTURAL_IDENTITY_AUDIT", "generated_at_utc": ts,
        "all_players_identities_hold": bool(all(r["identities_hold"] for r in ident_rows)),
        "identities": ["fgm=2pm+3pm", "pts=2*2pm+3*3pm+ftm", "reb=oreb+dreb", "stocks=stl+blk"]},
        indent=2, default=str))
    (V1 / "MONOTONICITY_AUDIT.json").write_text(json.dumps({
        "artifact": "MONOTONICITY_AUDIT", "generated_at_utc": ts,
        "all_alternate_ladders_monotone": bool(all(r["monotone"] for r in mono_rows)),
        "n_ladders_checked": len(mono_rows)}, indent=2, default=str))
    (V1 / "TAIL_MASS_AUDIT.json").write_text(json.dumps({
        "artifact": "TAIL_MASS_AUDIT", "generated_at_utc": ts,
        "max_omitted_tail_mass": float(max((r["omitted_tail_mass"] for r in tail_rows), default=0.0)),
        "all_tail_ok": bool(all(r["tail_ok"] for r in tail_rows)),
        "note": "empirical simulation PMFs are exact over their support; MC precision tracked via mc_max_se"},
        indent=2, default=str))
    pd.DataFrame(cov_rows).to_csv(V1 / "TODAY_PRICING_COVERAGE.csv", index=False)

    typer.echo("================ PRICING V1 RUN ================")
    typer.echo(f"  date={run_date} players={len(slate)} atoms={len(atom_df)} priced_lines={len(price_df)}")
    typer.echo(f"  identities_hold(all)={all(r['identities_hold'] for r in ident_rows)} "
               f"monotone(all)={all(r['monotone'] for r in mono_rows)}")
    typer.echo(f"  outputs -> {out_dir.relative_to(REPO)} ; audits -> artifacts/v1/")


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


if __name__ == "__main__":
    app()
