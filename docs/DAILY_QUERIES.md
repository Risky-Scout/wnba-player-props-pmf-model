# Daily Query Reference
## WNBA Player Props PMF Model

All commands run from the repository root:
```bash
cd /path/to/wnba-player-props-pmf-model
```

Required environment variable:
```bash
export BDL_API_KEY="your_balldontlie_api_key"
```

---

## A. Pre-Game Probabilities

### A1. Full Daily Pipeline (Production — Run This Every Morning)

Runs every step from data pull through calibrated PMFs and edge report.

```bash
python scripts/run_daily_pipeline.py \
  --season 2026 \
  --game-date 2026-06-25 \
  --out-dir deliveries/today
```

**What it produces:**
- `deliveries/today/full_pmfs_wide.parquet` — Full PMF for every player-stat
- `deliveries/today/player_projections_2026-06-25.parquet` — Clean projections
- `deliveries/today/player_projections_2026-06-25.json` — Same in JSON
- `deliveries/today/edge_report.parquet` — All edges ≥ 4pp vs. market
- `deliveries/today/publishable_edges.parquet` — Highest-confidence edges

---

### A2. Fast Prediction Only (Skip Pull/Train — Use Cached Models)

Use this when models are already trained and you just need today's predictions.

```bash
python scripts/predict_today.py \
  --features-wide data/processed/wnba_player_game_features_wide.parquet \
  --model-dir artifacts/models/stage4_baseline \
  --cal-dir artifacts/models/calibration \
  --game-date 2026-06-25 \
  --out-dir deliveries/today
```

---

### A3. Predict With Full PMF Grid JSON (for Kalshi / Polymarket Half-Line Markets)

```bash
python scripts/predict_today.py \
  --features-wide data/processed/wnba_player_game_features_wide.parquet \
  --model-dir artifacts/models/stage4_baseline \
  --cal-dir artifacts/models/calibration \
  --game-date 2026-06-25 \
  --out-dir deliveries/today \
  --export-grids-json
```

**Additional output:**
- `deliveries/today/pmf_grids_2026-06-25.json` — Full PMF grids with P(over) at every 0.5-step line

---

### A4. Build Edge Report Against Market Lines

After predicting, compute edges vs. live BDL props:

```bash
python scripts/build_edge_report.py \
  --pmfs deliveries/today/full_pmfs_wide.parquet \
  --props data/raw/bdl/wnba_player_props.parquet \
  --out-dir deliveries/today \
  --edge-threshold 0.04
```

---

### A5. Apply Injury News and Re-Predict

When a player is ruled out before tip-off:

```bash
# Mark player out (automatically redistributes usage via UsageTransferMatrix)
python scripts/apply_injury_news.py \
  --player-id 434 \
  --status out \
  --game-date 2026-06-25

# If a player is limited (e.g., minutes cap of 20):
python scripts/apply_injury_news.py \
  --player-id 434 \
  --status limited \
  --minutes-cap 20 \
  --game-date 2026-06-25
```

---

### A6. Query Pre-Game Probabilities Programmatically (Python)

```python
import pandas as pd

# Load today's projections
proj = pd.read_parquet("deliveries/today/player_projections_2026-06-25.parquet")

# All props for a specific player (e.g., A'ja Wilson, player_id=434)
wilson = proj[proj["player_id"] == 434][
    ["player_name", "stat", "pmf_mean", "pmf_std", "p_over_line", "line",
     "edge", "model_p_over", "market_p_over_no_vig", "fair_american_over",
     "confidence_tier"]
]
print(wilson.to_string(index=False))

# All edges ≥ 4pp across all players for today
edges = pd.read_parquet("deliveries/today/publishable_edges.parquet")
edges_sorted = edges.sort_values("edge", ascending=False)
print(edges_sorted[["player_name", "stat", "line", "direction",
                     "edge", "model_p_over", "market_p_over_no_vig",
                     "fair_american_over"]].to_string(index=False))

# P(over) at a specific line for a specific player-stat
from wnba_props_model.models.simulation import json_to_pmf
import numpy as np

row = proj[(proj["player_id"] == 434) & (proj["stat"] == "pts")].iloc[0]
pmf = json_to_pmf(row["pmf_json"])
line = 22.5
p_over = float(pmf[int(line)+1:].sum())
print(f"P(A'ja Wilson pts > {line}) = {p_over:.4f} ({p_over*100:.1f}%)")
```

---

### A7. Export Betting Sheet (CSV)

```bash
python scripts/export_betting_sheet.py \
  --pmfs deliveries/today/full_pmfs_wide.parquet \
  --props data/raw/bdl/wnba_player_props.parquet \
  --out deliveries/today/betting_sheet_2026-06-25.csv \
  --min-edge 0.04
```

---

### A8. Run Quality Gates (Before Publishing)

```bash
python scripts/verify_gates.py production-gates \
  --oof-scored artifacts/models/calibration/oof_pmfs_scored.parquet \
  --odds-parquet data/processed/wnba_odds.parquet \
  --results-parquet data/clv_tracking/results.parquet
```

---

## B. Live In-Play Probabilities

### B1. Automatic Live Tracking (Production — Triggered by GitHub Actions)

This runs automatically via `live_tracker.yml` every 5 minutes during game windows.
To trigger manually on GitHub:

```
GitHub → Actions → "Live In-Play Tracker" → "Run workflow"
→ Optional: enter specific game_id (leave blank for auto-detect)
→ duration_minutes: 150  (covers full WNBA game + OT buffer)
```

---

### B2. Manual Live Tracking (Local — Step by Step)

**Step 1: Find today's active games**
```bash
python scripts/find_live_games.py --out-dir artifacts/live
# Output: artifacts/live/live_games.json
# Prints: has_games=true/false, game_ids=12345 67890
```

**Step 2: Run the live tracker**
```bash
python scripts/run_live_tracker.py \
  --game-ids 12345 67890 \
  --duration 150 \
  --poll-interval 15 \
  --out-dir artifacts/live \
  --projections deliveries/today/full_pmfs_wide.parquet
```

**What it produces (updated every 15 seconds):**
- `artifacts/live/live_edges_latest.json` — Current live edges (always overwritten)
- `artifacts/live/live_edges_{game_id}_{timestamp}.json` — Historical snapshots per poll
- `artifacts/live/live_session_summary.json` — End-of-session summary

---

### B3. Query Live Probabilities Programmatically (Python)

```python
import json

# Read current live edges
with open("artifacts/live/live_edges_latest.json") as f:
    live = json.load(f)

print(f"Game state: {live['game_states']}")
print(f"Bettable edges: {live['n_bettable']}/{live['n_total_edges']}")

# Top bettable edges right now
for edge in live["top_edges"][:10]:
    if edge["bettable"]:
        print(
            f"Player {edge['player_id']} {edge['stat']} "
            f"line={edge['line']} {edge['direction'].upper()} "
            f"edge={edge['edge_pp']:+.1f}pp  "
            f"model={edge['model_p_over']:.3f}  market={edge['market_p_over']:.3f}  "
            f"observed={edge['observed_count']} in {edge['elapsed_minutes']:.1f}min  "
            f"projected={edge['projected_total']:.1f}"
        )
```

---

### B4. Run Live Engine Directly (Single Game, Python API)

```python
from wnba_props_model.data.bdl_client import BDLClient
from wnba_props_model.live import (
    GammaPoissonLiveEngine,
    LiveEdgeCalculator,
    LiveGameOrchestrator,
)
from wnba_props_model.live.orchestrator import build_roster_lookup
import pandas as pd

# Initialize
client = BDLClient()  # reads BDL_API_KEY from environment
engine = GammaPoissonLiveEngine()
edge_calc = LiveEdgeCalculator(min_edge=0.04)
orch = LiveGameOrchestrator(client, engine, edge_calc, out_dir="artifacts/live")

# Load pre-game projections (from daily pipeline output)
pmfs = pd.read_parquet("deliveries/today/full_pmfs_wide.parquet")

# Build projections dict: {player_id: {stat: {mean, line}, projected_minutes}}
from scripts.run_live_tracker import _build_projections_dict
projections = _build_projections_dict(pmfs)

# Build roster lookup (maps "A. Wilson" → {player_id, team_id, team_side})
players = pd.read_parquet("data/processed/wnba_players.parquet")
roster = build_roster_lookup(players, home_team_id=5, away_team_id=3)

# Run one live cycle
game_id = 12345
edges, game_state = orch.run_game(game_id, projections, roster)

# Print live edges
for e in edges:
    if e["bettable"]:
        print(f"LIVE EDGE: {e}")
```

---

### B5. Compute Bayesian Posterior for a Single Player-Stat (Python)

```python
from wnba_props_model.live.bayesian_updater import GammaPoissonLiveEngine

engine = GammaPoissonLiveEngine()

# A'ja Wilson pts: pre-game mean=22.5, projected 30 minutes
# She has 8 pts in 12 minutes; line is 22.5
result = engine.compute_live_p_over(
    stat="pts",
    mean_per_game=22.5,        # pre-game projected mean
    projected_total_minutes=30.0,
    observed_count=8,           # pts so far
    elapsed_minutes=12.0,       # minutes played so far
    line=22.5,                  # prop line
)

print(f"P(over 22.5 pts) = {result['p_over']:.4f}")
print(f"P(under 22.5 pts) = {result['p_under']:.4f}")
print(f"Projected final total = {result['projected_total']:.1f}")
print(f"Full PMF: {result['pmf']}")
```

---

### B6. Read Live Edges as They Update (Shell Watch Loop)

```bash
# Watch live_edges_latest.json refresh every 20 seconds
watch -n 20 "python3 -c \"
import json
with open('artifacts/live/live_edges_latest.json') as f:
    d = json.load(f)
gs = list(d.get('game_states', {}).values())
if gs:
    g = gs[0]
    print(f'Q{g.get(\"period\",\"?\")} {g.get(\"clock\",\"?\")} | {g.get(\"home_score\",0)}-{g.get(\"away_score\",0)}')
print(f'Bettable: {d[\"n_bettable\"]}/{d[\"n_total_edges\"]}')
for e in d[\"top_edges\"][:5]:
    if e.get(\"bettable\"):
        print(f'  {e[\"stat\"]:8s} line={e[\"line\"]:5.1f} {e[\"direction\"].upper():5s} {e[\"edge_pp\"]:+.1f}pp  model={e[\"model_p_over\"]:.3f}')
\""
```

---

## C. Scoring and Monitoring

### C1. Score Yesterday's Predictions (After Games Finish)

```bash
python scripts/score_daily_predictions.py \
  --game-date 2026-06-24 \
  --pmfs deliveries/2026-06-24/full_pmfs_wide.parquet \
  --actuals data/processed/wnba_player_game_stats.parquet \
  --out-dir data/clv_tracking \
  --closing-lines data/raw/bdl/wnba_player_props.parquet
```

### C2. Check Calibration Drift

```bash
python scripts/check_calibration_drift.py \
  --scored-predictions data/clv_tracking/drift_window.parquet \
  --out artifacts/audits/drift_check_$(date +%Y-%m-%d).json
```

### C3. Generate CLV Report

```bash
python scripts/generate_clv_report.py \
  --results data/clv_tracking/results.parquet \
  --lookback 100
```

---

## D. Output Schema Reference

### Pre-game `player_projections_{date}.parquet`

| Column | Type | Description |
|---|---|---|
| `player_id` | int | BDL player ID |
| `player_name` | str | Full player name |
| `game_id` | int | BDL game ID |
| `game_date` | date | Game date |
| `stat` | str | pts / reb / ast / fg3m / stl / blk / turnover / stocks / pts_ast / pts_reb / reb_ast / pts_reb_ast |
| `pmf_mean` | float | Model's expected value |
| `pmf_std` | float | Standard deviation of distribution |
| `p_over_line` | float | P(stat > line) at market line |
| `line` | float | Market prop line |
| `edge` | float | Model P(over) − Shin no-vig market P(over) |
| `model_p_over` | float | Calibrated model probability |
| `market_p_over_no_vig` | float | Market's vig-free probability |
| `fair_american_over` | int | American odds at model's probability |
| `is_calibrated` | bool | Whether Beta/IDR calibration was applied |
| `confidence_tier` | str | standard / high_adversity |
| `pmf_json` | str | Full PMF as JSON dict {k: prob} |

### Live `live_edges_latest.json`

| Field | Description |
|---|---|
| `timestamp_utc` | ISO timestamp of this poll |
| `game_states` | Dict of game_id → {home_score, away_score, period, clock} |
| `n_total_edges` | Total props compared |
| `n_bettable` | Edges ≥ 4pp |
| `top_edges[].player_id` | BDL player ID |
| `top_edges[].stat` | Stat name |
| `top_edges[].line` | Current prop line |
| `top_edges[].direction` | "over" or "under" |
| `top_edges[].model_p_over` | Gamma-Poisson posterior P(over) |
| `top_edges[].market_p_over` | Vig-free market P(over) |
| `top_edges[].edge` | Edge in decimal (0.08 = 8pp) |
| `top_edges[].edge_pp` | Edge in percentage points |
| `top_edges[].observed_count` | Stat accumulated so far |
| `top_edges[].elapsed_minutes` | Minutes played so far |
| `top_edges[].projected_total` | Posterior mean for final total |
| `top_edges[].bettable` | True if \|edge\| ≥ min_edge threshold |
