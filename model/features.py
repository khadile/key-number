"""One row per team per game: what that team should score against that opponent.

Every value comes from the ratings snapshot entering the game's week, so a
row never sees its own game or anything after it.
"""
import numpy as np
import polars as pl

from model.qb import QBModel
from model.ratings import Inputs, Params, all_snapshots

FEATURES = [
    "pass_off", "pass_def", "rush_off", "rush_def",
    "pass_sr_off", "pass_sr_def", "rush_sr_off", "rush_sr_def",
    "qb_adj", "proe_off", "pace_own", "pace_opp", "plays_own",
    "st_diff", "home", "rest_diff", "dome", "wind", "cold",
    "lg_pass_epa", "lg_rush_epa",
]

# Which features feed each bar of "where the margin comes from".
GROUPS = {
    "Passing game": ["pass_off", "pass_def", "pass_sr_off", "pass_sr_def"],
    "Running game": ["rush_off", "rush_def", "rush_sr_off", "rush_sr_def"],
    "Quarterback": ["qb_adj"],
    "Pace & tendency": ["proe_off", "pace_own", "pace_opp", "plays_own"],
    "Special teams": ["st_diff"],
    "Home field": ["home"],
    "Rest": ["rest_diff"],
}


def _previous_starters(games: pl.DataFrame) -> dict[tuple[str, str], str | None]:
    """(game_id, team) -> the QB who started that team's previous game this season."""
    long = pl.concat([
        games.select("game_id", "season", "week", team="home_team", qb="home_qb_id"),
        games.select("game_id", "season", "week", team="away_team", qb="away_qb_id"),
    ]).sort("team", "season", "week").with_columns(prev=pl.col("qb").shift(1).over("team", "season"))
    return {(g, t): q for g, t, q in zip(long["game_id"], long["team"], long["prev"])}


def build(inp: Inputs, games: pl.DataFrame, p: Params, as_of_publish: bool = False
          ) -> tuple[pl.DataFrame, dict, QBModel]:
    """Feature rows for every game.

    as_of_publish=True restricts game-specific facts to what is known on
    Tuesday of game week: last week's starting QB (week 1 keeps the listed
    starter, settled by then) and no game-day weather. Use it when comparing
    against Tuesday lines; the default uses the facts nflverse records.
    """
    weeks = sorted(set(zip(games["season"].to_list(), games["week"].to_list())))
    snaps = all_snapshots(inp, weeks, p)
    qbm = QBModel(inp, snaps, p)
    prev_qb = _previous_starters(games) if as_of_publish else {}

    qb_cache: dict = {}
    rows = []
    for g in games.iter_rows(named=True):
        key = (g["season"], g["week"])
        s = snaps[key]
        if key not in qb_cache:
            qb_cache[key] = (qbm.ratings(*key), qbm.team_mix(*key))
        qr, mix = qb_cache[key]
        h, a = inp.idx[g["home_team"]], inp.idx[g["away_team"]]
        outdoor = not g["dome"]
        if as_of_publish:
            wind, cold = (8.0 if outdoor else 0.0), 0.0  # Tuesday has no game-day weather
        else:
            wind = (g["wind"] if g["wind"] is not None else 8.0) if outdoor else 0.0
            cold = max(0.0, 40.0 - g["temp"]) / 10.0 if outdoor and g["temp"] is not None else 0.0
        rest = {h: g["home_rest"] or 7, a: g["away_rest"] or 7}
        home_qb, away_qb = g["home_qb_id"], g["away_qb_id"]
        if as_of_publish:
            home_qb = prev_qb.get((g["game_id"], g["home_team"])) or home_qb
            away_qb = prev_qb.get((g["game_id"], g["away_team"])) or away_qb
        for t, o, is_home, qb_id, pts in (
            (h, a, 1, home_qb, g["home_score"]),
            (a, h, 0, away_qb, g["away_score"]),
        ):
            rows.append({
                "game_id": g["game_id"], "season": g["season"], "week": g["week"],
                "team": inp.teams[t], "opp": inp.teams[o], "is_home": is_home,
                "points": pts,
                "pass_off": s["off_pass_epa"][t], "pass_def": s["def_pass_epa"][o],
                "rush_off": s["off_rush_epa"][t], "rush_def": s["def_rush_epa"][o],
                "pass_sr_off": s["off_pass_sr"][t], "pass_sr_def": s["def_pass_sr"][o],
                "rush_sr_off": s["off_rush_sr"][t], "rush_sr_def": s["def_rush_sr"][o],
                # No listed starter yet (lookahead weeks): assume the status quo.
                "qb_adj": qbm.rating_of(qr, qb_id) - mix[t] if qb_id else 0.0,
                "proe_off": s["off_neutral_proe"][t],
                "pace_own": s["off_neutral_sec_per_play"][t] + s["def_neutral_sec_per_play"][o],
                "pace_opp": s["off_neutral_sec_per_play"][o] + s["def_neutral_sec_per_play"][t],
                "plays_own": s["off_plays"][t] + s["def_plays"][o],
                "st_diff": s["st_net"][t] - s["st_net"][o],
                "home": float(is_home and not g["neutral_site"]),
                "rest_diff": float(np.clip(rest[t] - rest[o], -7, 7)) / 7,
                "dome": float(g["dome"]),
                "wind": float(wind),
                "cold": cold,
                "lg_pass_epa": s["mu_pass_epa"], "lg_rush_epa": s["mu_rush_epa"],
                "qb_id": qb_id,
            })
    return pl.DataFrame(rows), snaps, qbm
