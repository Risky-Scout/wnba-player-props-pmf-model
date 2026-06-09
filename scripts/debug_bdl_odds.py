"""Debug BDL WNBA game odds and player props endpoints.

Prints sanitized request/response details for three calls:
  1. Game odds by date
  2. Game odds by game_id
  3. Player props by game_id + vendors

Never prints the API key.

Usage:
    python3 scripts/debug_bdl_odds.py \\
        --date 2026-05-08 \\
        --game-id 24752 \\
        --vendors fanduel,draftkings
"""
from __future__ import annotations

import json
from typing import Any

import typer

app = typer.Typer(add_completion=False)

_SEP = "=" * 64


def _sanitize_url(url: str, api_key: str) -> str:
    return url.replace(api_key, "***API_KEY***")


def _sanitize(obj: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "..."
    if isinstance(obj, dict):
        return {
            k: _sanitize(v, depth + 1)
            for k, v in obj.items()
            if k.lower() not in ("api_key", "key", "token", "authorization", "auth")
        }
    if isinstance(obj, list):
        return [_sanitize(x, depth + 1) for x in obj[:3]]
    return obj


def _print_section(title: str, result: dict[str, Any], api_key: str) -> None:
    typer.echo(f"\n{_SEP}")
    typer.echo(f"{title}")
    typer.echo(_SEP)
    safe_url = _sanitize_url(result.get("url", ""), api_key)
    typer.echo(f"  URL:         {safe_url}")
    typer.echo(f"  HTTP status: {result.get('status_code')}")

    payload = result.get("json", {})
    data = payload.get("data", []) if isinstance(payload, dict) else []
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}

    typer.echo(f"  Row count:   {len(data)}")
    if meta:
        typer.echo(f"  Meta:        {meta}")

    if data:
        vendors = sorted({str(r.get("vendor", r.get("book", r.get("sportsbook", "?"))))
                          for r in data})
        typer.echo(f"  Vendors:     {vendors}")
        typer.echo(f"  First row keys: {list(data[0].keys())}")
        typer.echo("  First row (sanitized):")
        typer.echo(json.dumps(_sanitize(data[0]), indent=4, default=str))
        if len(data) > 1:
            typer.echo("  Second row keys:")
            typer.echo(f"    {list(data[1].keys())}")
    elif isinstance(payload, dict) and "error" in payload:
        typer.echo(f"  Error body:  {payload['error'][:500]}")
    else:
        typer.echo("  (empty data array — expected for live-only endpoints on completed games)")


@app.command()
def main(
    date: str = typer.Option("2026-05-08", help="Game date (YYYY-MM-DD) to test odds by date."),
    game_id: int = typer.Option(24752, help="Game ID to test odds by game_id and player props."),
    vendors: str = typer.Option("fanduel,draftkings", help="Comma-separated vendor list."),
    per_page: int = typer.Option(2, help="Rows per page (keep low for debug)."),
) -> None:
    from wnba_props_model.data.bdl_client import BDLClient, _array_params

    client = BDLClient()
    vendor_list = [v.strip() for v in vendors.split(",") if v.strip()]

    def _call(path: str, params: dict) -> dict:
        url = f"{client.base_url}{path}"
        query = _array_params(params)
        resp = client.session.get(url, params=query, timeout=client.timeout)
        try:
            body = resp.json()
        except Exception:
            body = {"error": resp.text[:500]}
        return {
            "url": str(resp.url),
            "status_code": resp.status_code,
            "json": body,
        }

    typer.echo(f"\nBDL WNBA Odds/Props Debug  (base_url: {client.base_url})")

    # ------------------------------------------------------------------
    # 1. Game odds by date
    # ------------------------------------------------------------------
    r1 = _call("/wnba/v1/odds", {"dates": [date], "per_page": per_page})
    _print_section(f"1. GAME ODDS BY DATE  (dates[]={date})", r1, client.api_key)

    # ------------------------------------------------------------------
    # 2. Game odds by game_id
    # ------------------------------------------------------------------
    r2 = _call("/wnba/v1/odds", {"game_ids": [game_id], "per_page": per_page})
    _print_section(f"2. GAME ODDS BY GAME_ID  (game_ids[]={game_id})", r2, client.api_key)

    # ------------------------------------------------------------------
    # 3. Player props by game_id + vendors
    # ------------------------------------------------------------------
    r3 = _call("/wnba/v1/odds/player_props", {"game_id": game_id, "vendors": vendor_list})
    _print_section(
        f"3. PLAYER PROPS  (game_id={game_id}, vendors={vendor_list})", r3, client.api_key
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    typer.echo(f"\n{_SEP}")
    typer.echo("SUMMARY")
    typer.echo(_SEP)
    for label, result in [
        ("Game odds by date", r1),
        ("Game odds by game_id", r2),
        ("Player props", r3),
    ]:
        status = result["status_code"]
        n = len(result.get("json", {}).get("data", []))
        typer.echo(f"  {label:30s} HTTP {status}  rows={n}")


if __name__ == "__main__":
    app()
