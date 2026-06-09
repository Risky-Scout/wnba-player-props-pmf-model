# NBA → WNBA BALLDONTLIE API Feature Parity Map

Checked against the public BALLDONTLIE NBA and WNBA docs on 2026-05-30.

## Critical finding

WNBA is usable for a player-prop PMF model and a game-total model, but it is **not one-to-one identical** to the NBA data contract.

The major gap is lineups: NBA exposes `GET /v1/lineups`; WNBA docs do not list a lineup endpoint. The WNBA port therefore uses a fallback `expected_starter`/`recent_starter_rate5` lineup proxy and reserves the `lineup_confirmed` fields as zero/unknown unless a separate lineup feed is added.

## Endpoint map

| NBA feature | NBA BDL endpoint | WNBA BDL endpoint | One-to-one? | Model handling |
|---|---:|---:|---:|---|
| Teams | `/v1/teams` | `/wnba/v1/teams` | Yes | Team identity, conference, abbreviations |
| Players | `/v1/players` | `/wnba/v1/players` | Mostly | WNBA adds `position_abbreviation`, age in examples; normalize to common schema |
| Active players | `/v1/players/active` | `/wnba/v1/players/active` | Yes | Slate/player universe filter |
| Games | `/v1/games` | `/wnba/v1/games` | Mostly | WNBA dates are ISO datetimes; normalize UTC/date |
| Player game stats | `/v1/stats` | `/wnba/v1/player_stats` | Yes for core box stats | Labels for pts/reb/ast/fg3m/tov/stl/blk and minutes |
| Team game stats | NBA inferred from player stats / box scores | `/wnba/v1/team_stats` | WNBA stronger | Game total and opponent context features |
| Season averages | `/v1/season_averages` | `/wnba/v1/player_season_stats` | Partial | Use only for priors; avoid current-season leakage by date cutoff |
| Team season averages | `/v1/teams/season_averages` | `/wnba/v1/team_season_stats` | Partial | Pace/scoring/rebounding priors |
| Advanced player game stats | `/nba/v1/stats/advanced` | `/wnba/v1/player_game_advanced_stats` | Partial | Usage/pace/off-def rating where available |
| Advanced team game stats | NBA advanced endpoint can include game/team context | `/wnba/v1/team_game_advanced_stats` | Partial | Team pace/off-def rating context |
| Shot location | NBA v2 advanced/tracking has shooting categories | `/wnba/v1/player_shot_locations`, `/wnba/v1/team_shot_locations` | Not exact | Use WNBA shot zones for rim/three mix and block-opportunity proxies |
| Box scores | `/v1/box_scores`, `/v1/box_scores/live` | No documented WNBA analog | No | Reconstruct from `player_stats` + `games`; no live box score dependency |
| Lineups | `/v1/lineups` | No documented WNBA analog | No | Use expected lineup proxy; add external lineup adapter later |
| Injuries | `/v1/player_injuries` | `/wnba/v1/player_injuries` | Mostly | Player availability, team usage/rebound/assist vacated proxies |
| Standings | `/v1/standings` | `/wnba/v1/standings` | Yes | Strength/rest/playoff-context features |
| Leaders | `/v1/leaders` | No documented WNBA analog | No | Not required; derive from stats |
| Betting odds | `/v2/odds` | `/wnba/v1/odds` | Yes conceptually, different base path | Game total market comparison, no-vig baselines |
| Player props | `/v2/odds/player_props` | `/wnba/v1/odds/player_props` | Partial | WNBA supported O/U props: points, rebounds, assists, threes, PA, PR, RA, PRA |
| Play-by-play | `/v1/plays` | `/wnba/v1/plays` | Mostly | Possession/pace proxy and live model later |
| Contracts | `/v1/contracts/*` | No documented WNBA analog | No | Not required for pricing |

## Player prop market map

| WNBA BDL `prop_type` | Internal stat | Model support |
|---|---:|---|
| `points` | `pts` | Direct PMF |
| `rebounds` | `reb` | Direct PMF |
| `assists` | `ast` | Direct PMF |
| `threes` | `fg3m` | Direct PMF |
| `points_assists` | `pa` | Convolution from pts + ast |
| `points_rebounds` | `pr` | Convolution from pts + reb |
| `rebounds_assists` | `ra` | Convolution from reb + ast |
| `points_rebounds_assists` | `pra` | Convolution from pts + reb + ast |
| `double_double` | milestone | Stub only; needs joint stat dependency model |
| `triple_double` | milestone | Stub only; needs joint stat dependency model |

## NBA features that must be modified for WNBA

1. **Official lineup features**
   - NBA: confirmed starters/bench from BDL lineups.
   - WNBA: no documented BDL lineup endpoint. Use `expected_starter`, recent minutes, and injury feed until external lineups are added.

2. **Sparse props**
   - NBA props list includes blocks and steals.
   - WNBA props list in current BDL docs does not list steals/blocks as offered prop types, but the model still predicts stl/blk for `stocks`, player evaluation, and future market expansion.

3. **Game totals**
   - WNBA BDL odds endpoint has `total_value`, `total_over_odds`, `total_under_odds`.
   - Build a separate game-total PMF so the model can score total O/U markets against no-vig market probabilities.

4. **Sample-size discipline**
   - WNBA has fewer teams and games than NBA. Role-aware calibrators should shrink more aggressively to global, and superiority claims require the same UCB gate but will have fewer eligible stat×role cells early.
