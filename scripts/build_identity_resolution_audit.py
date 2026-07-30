"""Stage 7: identity-resolution audit + tracked alias table (offline, no API).

Reads the rebuilt atomic store (with identity_method) and reports resolution outcomes,
collisions, and multi-mapping. Writes a tracked alias table (auto-accepted deterministic
resolutions) and a tracked identity-resolution audit.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.data.identity_resolution import normalize_strict  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"


@app.command()
def main(
    store: str = typer.Option("data/atomic_quotes/atomic_quotes.parquet", "--store"),
    roster: str = typer.Option("data/recovered_v2/wnba_player_game_stats.parquet", "--roster"),
) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(store, columns=["event_id", "game_id", "player_id", "player_name",
                                        "identity_method", "eligibility_status"])
    roster_df = pd.read_parquet(roster)[["player_id", "player_name"]].dropna().drop_duplicates()
    canon = {str(pid): pn for pid, pn in zip(roster_df["player_id"], roster_df["player_name"])}

    method_counts = {str(k): int(v) for k, v in d["identity_method"].value_counts(dropna=False).to_dict().items()}
    resolved = d[d["player_id"].notna()]

    # alias table: auto-accepted deterministic (normalized_relaxed / approved_alias) resolutions
    alias_rows = []
    aset = d[d["identity_method"].isin(["normalized_relaxed", "approved_alias"])][
        ["player_name", "player_id", "identity_method"]].dropna().drop_duplicates()
    for _, r in aset.iterrows():
        pid = str(int(r["player_id"])) if pd.notna(r["player_id"]) else None
        alias_rows.append({
            "provider_player_name": r["player_name"],
            "normalized_provider_name": normalize_strict(r["player_name"]),
            "bdl_player_id": pid,
            "canonical_bdl_name": canon.get(pid),
            "resolution_method": r["identity_method"],
            "evidence": "deterministic normalization within game roster",
            "approval_status": "auto_accepted_deterministic",
        })

    # multi-mapping diagnostics
    name_to_ids = resolved.groupby(resolved["player_name"].map(normalize_strict))["player_id"].nunique()
    id_to_names = resolved.groupby(resolved["player_id"].astype(str))["player_name"].nunique()
    provider_name_multi_id = int((name_to_ids > 1).sum())
    bdl_id_multi_name = int((id_to_names > 1).sum())

    audit = {
        "artifact": "IDENTITY_RESOLUTION_AUDIT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows": int(len(d)),
        "identity_method_counts": method_counts,
        "resolved_before_repair_exact": method_counts.get("exact_roster_name", 0),
        "resolved_through_normalization_relaxed": method_counts.get("normalized_relaxed", 0),
        "resolved_through_approved_aliases": method_counts.get("approved_alias", 0),
        "unresolved_unmatched": method_counts.get("unmatched", 0),
        "collisions": method_counts.get("collision", 0),
        "provider_name_mapping_to_multiple_bdl_ids": provider_name_multi_id,
        "bdl_id_receiving_multiple_provider_names": bdl_id_multi_name,
        "n_alias_entries": len(alias_rows),
        "note": "Collisions are REFUSED (never forced). Multiple BDL ids for one normalized "
                "provider name across DIFFERENT games are legitimately different players; "
                "within a single game roster a collision blocks resolution.",
    }
    (AUD / "IDENTITY_RESOLUTION_AUDIT.json").write_text(json.dumps(audit, indent=2, default=str))
    (REPO / "config" / "player_alias_table_v2.json").write_text(
        json.dumps({"generated_at_utc": audit["generated_at_utc"], "aliases": alias_rows}, indent=2, default=str))

    typer.echo("================ IDENTITY RESOLUTION AUDIT ================")
    for k in ["exact_roster_name", "normalized_relaxed", "approved_alias", "unmatched", "collision"]:
        typer.echo(f"  {k:22s}= {method_counts.get(k, 0)}")
    typer.echo(f"  provider_name -> multiple BDL ids : {provider_name_multi_id}")
    typer.echo(f"  BDL id -> multiple provider names : {bdl_id_multi_name}")
    typer.echo(f"  alias entries written             : {len(alias_rows)}")
    typer.echo(f"  wrote {AUD}/IDENTITY_RESOLUTION_AUDIT.json + config/player_alias_table_v2.json")


if __name__ == "__main__":
    app()
