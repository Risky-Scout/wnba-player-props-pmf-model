# Tracking / Hustle Capability Matrix

- tracking rows: 33,259 across 1,476 games (player grain: True, personId nunique=386)
- hustle rows: 171 (player grain: False, PLAYER_ID nunique=1) -> deferred: True

| feature | status | source | coverage |
|---|---|---|---|
| assist_opportunity_proxy | `DERIVABLE` | tracking(derived) | - |
| assists_tracked | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| box_outs | `UNAVAILABLE` | hustle | - |
| charges_drawn | `UNAVAILABLE` | hustle | - |
| contested_fga | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| contested_fgm | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| contested_shots | `UNAVAILABLE` | hustle | - |
| defended_at_rim_fga | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| deflections | `UNAVAILABLE` | hustle | - |
| distance | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| fg3m_attempts | `PROXY_ONLY` | box_score_3PA | - |
| free_throw_assists | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| loose_balls_recovered | `UNAVAILABLE` | hustle | - |
| passes | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| reb_chances_defensive | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| reb_chances_offensive | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| reb_chances_total | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| screen_assists | `UNAVAILABLE` | hustle | - |
| secondary_assists | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| speed | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
| touches | `DIRECTLY_AVAILABLE` | tracking | 1.0 |
