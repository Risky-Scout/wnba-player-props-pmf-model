"""Play-by-play player-attribution parser (owner directive step B).

BDL WNBA play-by-play carries player attribution ONLY inside the free-text ``text`` field (there is
no per-play player_id). This module converts normalized plays into per-player-per-game event counts:

    FGA_2/FGA_3, FGM_2/FGM_3, FG3M, assisted-makes (from "(NAME assists)"), OREB/DREB, STL, BLK, TOV,
    FTA/FTM, plus a usage/possession proxy (shot attempts + turnovers + 0.44*FTA).

Names are resolved to canonical player ids using the per-game box roster (the two teams that played),
which is the fail-closed, game-scoped identity strategy used elsewhere in the repo
(:mod:`wnba_props_model.data.identity_crosswalk`). Team rebounds / team turnovers (whose leading token
is a team, not a rostered player) are attributed to the team, not a player, and dropped from the
per-player table.

The core functions are pure and importable so they can be unit-tested against the official box
(:func:`reconcile_against_box`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .identity_crosswalk import normalize_name

# ---------------------------------------------------------------------------
# stat vocabulary
# ---------------------------------------------------------------------------
COUNT_COLS = [
    "fg2a", "fg3a", "fga", "fg2m", "fg3m", "fgm",
    "fta", "ftm", "oreb", "dreb", "reb",
    "ast", "stl", "blk", "tov", "pts", "poss_proxy",
]

_ASSIST_RE = re.compile(r"\(([^()]+?)\s+assists?\)", re.IGNORECASE)
_STEAL_RE = re.compile(r"\(([^()]+?)\s+steals?\)", re.IGNORECASE)
_BLOCK_SPLIT_RE = re.compile(r"\bblocks\b", re.IGNORECASE)


def _blank_counts() -> dict[str, float]:
    return {c: 0.0 for c in COUNT_COLS}


@dataclass
class GameRoster:
    """Normalized-name -> canonical player_id resolver scoped to a single game."""
    name_to_id: dict[str, int]
    ambiguous_names: set[str] = field(default_factory=set)

    @classmethod
    def from_box(cls, box_game: pd.DataFrame) -> "GameRoster":
        name_to_id: dict[str, int] = {}
        counts: dict[str, set[int]] = {}
        for _, r in box_game.iterrows():
            nm = normalize_name(r.get("player_name"))
            pid = r.get("player_id")
            if not nm or pd.isna(pid):
                continue
            counts.setdefault(nm, set()).add(int(pid))
        ambiguous = {nm for nm, ids in counts.items() if len(ids) > 1}
        for nm, ids in counts.items():
            if nm not in ambiguous:
                name_to_id[nm] = next(iter(ids))
        return cls(name_to_id=name_to_id, ambiguous_names=ambiguous)

    def leading_id(self, text_norm: str) -> int | None:
        """Return the player id whose normalized name is the longest token-prefix of text_norm."""
        best_id, best_len = None, 0
        for nm, pid in self.name_to_id.items():
            if not nm:
                continue
            if text_norm == nm or text_norm.startswith(nm + " "):
                if len(nm) > best_len:
                    best_id, best_len = pid, len(nm)
        return best_id

    def find_id(self, fragment: str) -> int | None:
        """Return the player id whose normalized name appears (longest match) in a fragment."""
        frag = normalize_name(fragment)
        best_id, best_len = None, 0
        for nm, pid in self.name_to_id.items():
            if not nm:
                continue
            if nm == frag or frag.startswith(nm + " ") or frag.endswith(" " + nm) or (" " + nm + " ") in (" " + frag + " "):
                if len(nm) > best_len:
                    best_id, best_len = pid, len(nm)
        return best_id


def _is_three(text: str, score_value) -> bool:
    t = text.lower()
    if "three point" in t or "three pointer" in t or "3-pt" in t or "3pt" in t:
        return True
    try:
        if float(score_value) == 3.0:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _strip_possessive(fragment: str) -> str:
    # "Pauline Astier 's" / "Lauren Betts's" -> "Pauline Astier" / "Lauren Betts"
    frag = re.sub(r"\s*'s\b", "", fragment)
    return frag.strip()


@dataclass
class ParseStats:
    n_plays: int = 0
    n_attributable: int = 0
    n_unmatched_actor: int = 0
    unmatched_examples: list[str] = field(default_factory=list)


def parse_game_plays(plays: pd.DataFrame, roster: GameRoster,
                     stats: ParseStats | None = None) -> dict[int, dict[str, float]]:
    """Parse one game's plays into {player_id: counts}. Pure, no I/O."""
    per_player: dict[int, dict[str, float]] = {}
    stats = stats or ParseStats()

    def bump(pid: int, key: str, amt: float = 1.0) -> None:
        if pid is None:
            return
        per_player.setdefault(pid, _blank_counts())[key] += amt

    for _, p in plays.iterrows():
        text = p.get("text") or p.get("description") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        etype = str(p.get("event_type") or "")
        tnorm_full = text.lower()
        text_norm = normalize_name(text)
        stats.n_plays += 1
        made = bool(p.get("scoring_play")) or (" makes " in f" {tnorm_full} ")
        score_value = p.get("score_value")

        handled = False

        # --- Blocked field goals: "Blocker blocks Shooter 's <shot>" -----------------
        if "free throw" not in tnorm_full and _BLOCK_SPLIT_RE.search(tnorm_full) and "rebound" not in tnorm_full:
            left, right = _BLOCK_SPLIT_RE.split(text, maxsplit=1)
            blocker = roster.leading_id(normalize_name(left))
            shooter = roster.find_id(_strip_possessive(right))
            three = _is_three(text, score_value)
            if shooter is not None:
                bump(shooter, "fga")
                bump(shooter, "fg3a" if three else "fg2a")
            if blocker is not None:
                bump(blocker, "blk")
            handled = shooter is not None or blocker is not None
            if handled:
                stats.n_attributable += 1
            else:
                stats.n_unmatched_actor += 1
                if len(stats.unmatched_examples) < 25:
                    stats.unmatched_examples.append(text)
            continue

        # --- Free throws --------------------------------------------------------------
        if "free throw" in tnorm_full:
            actor = roster.leading_id(text_norm)
            if actor is not None:
                bump(actor, "fta")
                if made:
                    bump(actor, "ftm")
                    bump(actor, "pts", 1.0)
                stats.n_attributable += 1
            else:
                stats.n_unmatched_actor += 1
                if len(stats.unmatched_examples) < 25:
                    stats.unmatched_examples.append(text)
            continue

        # --- Rebounds -----------------------------------------------------------------
        if "rebound" in tnorm_full:
            actor = roster.leading_id(text_norm)
            if actor is not None:
                if "offensive rebound" in tnorm_full:
                    bump(actor, "oreb"); bump(actor, "reb")
                elif "defensive rebound" in tnorm_full:
                    bump(actor, "dreb"); bump(actor, "reb")
                stats.n_attributable += 1
            # team rebounds have no rostered leading name: silently team-attributed (dropped)
            continue

        # --- Turnovers (incl. steals credited to the stealer) -------------------------
        if "turnover" in tnorm_full or etype.lower().endswith("turnover") or "traveling" in tnorm_full:
            actor = roster.leading_id(text_norm)
            if actor is not None:
                bump(actor, "tov")
                stats.n_attributable += 1
            sm = _STEAL_RE.search(text)
            if sm:
                stealer = roster.find_id(sm.group(1))
                if stealer is not None:
                    bump(stealer, "stl")
            continue

        # --- Field goals (makes / misses) ---------------------------------------------
        is_shot = (" makes " in f" {tnorm_full} ") or (" misses " in f" {tnorm_full} ")
        # exclude non-FG "makes/misses" already handled (free throws). Fouls have no makes/misses.
        if is_shot:
            actor = roster.leading_id(text_norm)
            three = _is_three(text, score_value)
            if actor is not None:
                bump(actor, "fga")
                bump(actor, "fg3a" if three else "fg2a")
                if made:
                    bump(actor, "fgm")
                    if three:
                        bump(actor, "fg3m"); bump(actor, "pts", 3.0)
                    else:
                        bump(actor, "fg2m"); bump(actor, "pts", 2.0)
                stats.n_attributable += 1
                # assisted make
                am = _ASSIST_RE.search(text)
                if am and made:
                    assister = roster.find_id(am.group(1))
                    if assister is not None:
                        bump(assister, "ast")
            else:
                stats.n_unmatched_actor += 1
                if len(stats.unmatched_examples) < 25:
                    stats.unmatched_examples.append(text)
            continue

    # possession/usage proxy per player: FGA + TOV + 0.44*FTA
    for pid, c in per_player.items():
        c["poss_proxy"] = c["fga"] + c["tov"] + 0.44 * c["fta"]
    return per_player


def parse_plays_to_player_game(plays: pd.DataFrame, box: pd.DataFrame,
                               ) -> tuple[pd.DataFrame, ParseStats]:
    """Parse many games' plays into a per-(game_id, player_id) count table.

    ``box`` supplies the per-game roster used for name resolution (columns:
    game_id, player_id, player_name[, team_id, game_date]).
    """
    stats = ParseStats()
    box = box.copy()
    box["game_id"] = pd.to_numeric(box["game_id"], errors="coerce")
    plays = plays.copy()
    plays["game_id"] = pd.to_numeric(plays["game_id"], errors="coerce")
    date_by_game = {}
    if "game_date" in box.columns:
        date_by_game = box.dropna(subset=["game_id"]).groupby("game_id")["game_date"].first().to_dict()

    rows: list[dict] = []
    for gid, gplays in plays.groupby("game_id"):
        box_game = box[box["game_id"] == gid]
        if box_game.empty:
            continue
        roster = GameRoster.from_box(box_game)
        per_player = parse_game_plays(gplays, roster, stats)
        team_by_pid = box_game.dropna(subset=["player_id"]).set_index("player_id")["team_id"].to_dict() \
            if "team_id" in box_game.columns else {}
        for pid, c in per_player.items():
            row = {"game_id": gid, "player_id": pid, "game_date": date_by_game.get(gid)}
            if pid in team_by_pid:
                row["team_id"] = team_by_pid[pid]
            row.update({k: float(v) for k, v in c.items()})
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["game_id", "player_id"]).reset_index(drop=True)
    return out, stats


def reconcile_against_box(parsed: pd.DataFrame, box: pd.DataFrame,
                          stat_map: dict[str, str] | None = None) -> dict:
    """Compare parsed per-player counts vs the official box for shared (game_id, player_id).

    Returns per-stat exact-match rate, mean absolute error, and the number of compared rows.
    """
    stat_map = stat_map or {
        "pts": "pts", "reb": "reb", "ast": "ast", "stl": "stl",
        "blk": "blk", "tov": "turnover", "fg3m": "fg3m", "fga": "fga",
        "fg3a": "fg3a", "fta": "fta",
    }
    p = parsed.copy()
    b = box.copy()
    for c in ("game_id", "player_id"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
        b[c] = pd.to_numeric(b[c], errors="coerce")
    # only compare players who actually played (box did_play) to avoid DNP noise
    if "did_play" in b.columns:
        b = b[b["did_play"].astype(bool)]
    merged = p.merge(b, on=["game_id", "player_id"], how="inner", suffixes=("_pbp", "_box"))
    report: dict = {"n_rows": int(len(merged)), "n_games": int(merged["game_id"].nunique()),
                    "per_stat": {}}

    def _resolve(col: str, side: str) -> str | None:
        # after a suffixed merge, a name present on BOTH sides becomes col+"_pbp"/col+"_box";
        # a name present on only one side keeps its bare name.
        suf = f"{col}_{side}"
        if suf in merged.columns:
            return suf
        if col in merged.columns:
            return col
        return None

    for parsed_col, box_col in stat_map.items():
        if parsed_col not in p.columns or box_col not in b.columns:
            continue
        ac = _resolve(parsed_col, "pbp")
        bc = _resolve(box_col, "box")
        if ac is None or bc is None:
            continue
        a = pd.to_numeric(merged[ac], errors="coerce").fillna(0.0)
        d = pd.to_numeric(merged[bc], errors="coerce").fillna(0.0)
        diff = (a - d).abs()
        report["per_stat"][parsed_col] = {
            "vs_box_col": box_col,
            "exact_match_rate": float((diff == 0).mean()),
            "within_1_rate": float((diff <= 1).mean()),
            "mean_abs_error": float(diff.mean()),
            "max_abs_error": float(diff.max()) if len(diff) else 0.0,
            "parsed_total": float(a.sum()),
            "box_total": float(d.sum()),
        }
    return report
