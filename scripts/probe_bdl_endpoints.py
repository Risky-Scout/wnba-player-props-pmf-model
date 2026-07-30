"""One-request BDL capability + play-by-play probes (Section B/C, L outputs 1-3).

Proves entitlement and pagination from REAL responses instead of local flags. Writes:
  artifacts/audits/BDL_ENDPOINT_AUDIT.json   per-endpoint capability probe
  artifacts/audits/BDL_PBP_PROBE.json        real /wnba/v1/plays behavior on one game
  artifacts/audits/BDL_FIELD_INVENTORY.json  actually-returned keys (box + advanced measure_types)

Never prints the API key.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from wnba_props_model.data.bdl_client import BDLClient  # noqa: E402

app = typer.Typer(add_completion=False)
REPO = Path(__file__).resolve().parent.parent
AUD = REPO / "artifacts" / "audits"


def _find_completed_game(client: BDLClient, season: int) -> dict | None:
    """Return one completed WNBA game dict (has a final status)."""
    try:
        games = client.list_endpoint("games", {"seasons": [season], "per_page": 100})
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"[probe] games fetch failed: {exc}", err=True)
        return None
    for g in games:
        status = str(g.get("status", "")).lower()
        if "final" in status or status == "post" or (g.get("home_team_score") or g.get("home_score")):
            return g
    return games[0] if games else None


@app.command()
def main(season: int = typer.Option(2024, "--season")) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    client = BDLClient()
    ts = datetime.now(timezone.utc).isoformat()

    # ---- 1. capability probe of each endpoint (real HTTP status) ----
    probe_params = {
        "teams": {}, "players": {"per_page": 1}, "games": {"seasons": [season], "per_page": 1},
        "player_stats": {"seasons": [season], "per_page": 1},
        "player_game_advanced_stats": {"seasons": [season], "per_page": 1},
        "team_game_advanced_stats": {"seasons": [season], "per_page": 1},
        "player_season_advanced_stats": {"season": season, "measure_type": "advanced", "per_mode": "per_game"},
        "team_season_advanced_stats": {"season": season, "measure_type": "advanced", "per_mode": "per_game"},
        "player_shot_locations": {"season": season, "distance_range": "by_zone", "per_mode": "per_game"},
        "team_shot_locations": {"season": season, "distance_range": "by_zone", "per_mode": "per_game"},
        "standings": {"season": season}, "player_injuries": {"per_page": 1},
    }
    endpoint_audit = {"generated_at_utc": ts, "season": season, "endpoints": {}}
    for name, params in probe_params.items():
        rec = client.probe(name, params)
        endpoint_audit["endpoints"][name] = rec
        typer.echo(f"[probe] {name:32s} http={rec['http_status']} rows={rec['n_rows']} "
                   f"meta={rec['has_meta']} next_cursor={rec['has_next_cursor']}")
    (AUD / "BDL_ENDPOINT_AUDIT.json").write_text(json.dumps(endpoint_audit, indent=2, default=str))

    # ---- 2. find a completed game and probe /wnba/v1/plays ----
    game = _find_completed_game(client, season)
    pbp = {"generated_at_utc": ts, "season": season, "game_id": None, "note": "no game found"}
    field_inv = {"generated_at_utc": ts, "season": season}
    if game is not None:
        gid = game.get("id")
        pbp = {"generated_at_utc": ts, "season": season, "game_id": gid,
               "home": (game.get("home_team") or {}).get("abbreviation"),
               "away": (game.get("visitor_team") or {}).get("abbreviation")}
        # base call (no pagination params)
        base = client.get_json("/wnba/v1/plays", {"game_id": gid})
        base_data = base.get("data", []) if isinstance(base, dict) else base
        orders = [p.get("order") for p in base_data if isinstance(p, dict) and p.get("order") is not None]
        # repeated-with-pagination-params call to test whether undocumented params change anything
        withp = client.get_json("/wnba/v1/plays", {"game_id": gid, "per_page": 5, "cursor": 1})
        withp_data = withp.get("data", []) if isinstance(withp, dict) else withp
        sample_keys = sorted(base_data[0].keys()) if base_data and isinstance(base_data[0], dict) else []
        pbp.update({
            "http_ok": True,
            "n_play_rows_base": len(base_data),
            "response_has_meta": bool(isinstance(base, dict) and base.get("meta")),
            "response_has_next_cursor": bool(isinstance(base, dict) and (base.get("meta") or {}).get("next_cursor")),
            "max_play_order": max(orders) if orders else None,
            "n_play_rows_with_per_page_5_cursor_1": len(withp_data),
            "per_page_param_changed_result": len(withp_data) != len(base_data),
            "play_row_keys": sample_keys,
            "has_player_id_field": "player_id" in sample_keys,
            "conclusion": (
                "NON_PAGINATED: full game returned in one call; per_page/cursor had no effect"
                if len(withp_data) == len(base_data) else
                "PAGINATION_SIGNAL: per_page/cursor changed the row count — investigate"),
        })
        typer.echo(f"[probe] plays game={gid}: rows={len(base_data)} meta={pbp['response_has_meta']} "
                   f"next_cursor={pbp['response_has_next_cursor']} has_player_id={pbp['has_player_id_field']} "
                   f"-> {pbp['conclusion']}")

        # ---- 3. field inventory: box score + advanced measure_types ----
        stats = client.list_endpoint("player_stats", {"game_ids": [gid], "per_page": 100})
        box_keys = sorted(stats[0].keys()) if stats else []
        field_inv["player_stats_row_keys"] = box_keys
        field_inv["player_stats_n_rows_for_game"] = len(stats)
        adv_inv = {}
        for mt in ["advanced", "usage", "four_factors", "scoring", "misc", "defense", "opponent", "base"]:
            try:
                rows = client.list_endpoint(
                    "player_season_advanced_stats",
                    {"season": season, "measure_type": mt, "per_mode": "per_game", "per_page": 1})
                keys = sorted((rows[0].get("stats") or {}).keys()) if rows and isinstance(rows[0].get("stats"), dict) else []
                adv_inv[mt] = {"n_rows": len(rows), "returned_stat_keys": keys}
            except Exception as exc:  # noqa: BLE001
                adv_inv[mt] = {"error": str(exc)[:200]}
        field_inv["player_season_advanced_by_measure_type"] = adv_inv

    (AUD / "BDL_PBP_PROBE.json").write_text(json.dumps(pbp, indent=2, default=str))
    (AUD / "BDL_FIELD_INVENTORY.json").write_text(json.dumps(field_inv, indent=2, default=str))
    typer.echo(f"[probe] wrote {AUD}/BDL_ENDPOINT_AUDIT.json, BDL_PBP_PROBE.json, BDL_FIELD_INVENTORY.json")


if __name__ == "__main__":
    app()
