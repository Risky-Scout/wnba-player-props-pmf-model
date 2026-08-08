# Player 631 Root Cause — Kiana Williams

## Identity

| Field | Value |
|---|---|
| canonical_player_id | 631 |
| normalized_name | Kiana Williams |
| frozen slate team | PHX (team_id=10) |
| frozen slate opponent | NY |
| frozen game_id | 24970 (NY @ PHX) |
| identity audit status | ACCEPTED |
| identity reject_reason | current_team_mismatch_warning |
| wnba_players table team (at freeze) | LA |
| sportsbook name in frozen odds | (absent — no quotes for this player) |

## PHX versus LA

Verification warning:

`player 631 slate_team=PHX players_table_team=LA`

- The frozen **players table** recorded current team **LA**.
- The frozen **slate / feature row** (BDL-sourced) attached player 631 to **PHX** for game 24970 (NY @ PHX).
- Public roster evidence (June 19, 2026): Los Angeles Sparks signed Kiana Williams from a Phoenix Mercury developmental contract (offer-sheet / poach). Correct contemporaneous team after that transaction is **LA**, not PHX.

## Classification

| Hypothesis | Determination |
|---|---|
| Stale player table | **No** — players table LA matches the June 19 Sparks signing. |
| Trade / signing | **Yes (completed earlier)** — LA signing from PHX developmental deal (2026-06-19). |
| Alias / duplicate ID | **No** — single canonical id 631; no alternate slate identity for the same person. |
| Slate-construction problem | **Yes** — feature/slate construction still placed 631 on PHX for the Aug 1 Mercury game after she had moved to LA. |
| Wrong-player forecast risk | **Yes** — accepted forecast rows price Kiana Williams as a PHX participant in NY@PHX; LA is not on this slate. |

## Settlement recommendation

**EXCLUDE** player 631 accepted forecast rows from primary postgame settlement for this slate.

Rationale: material team/game identity defect in accepted rows. Do not repair the frozen prediction retroactively; keep PMFs as immutable evidence of what was forecast, but mark settlement status `VOID_IDENTITY_TEAM_MISMATCH` / exclude from certified metrics.

## Evidence sources (frozen only + public roster context)

- `PLAYER_IDENTITY_AUDIT.csv` row for 631
- `slate_2026-08-01.parquet` row for 631
- `PREGAME_VERIFICATION_SUMMARY.json` warning list
- Public contemporaneous reporting of the 2026-06-19 Sparks signing (context only; does not alter frozen atoms)
