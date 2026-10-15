"""Who plays: each game's quarterback, and the depth chart's order for the props roles.

nflverse's schedule names a QB for every game, but it is a label, not the box score. In weeks 1-5 of 2026,
10 of 158 team-games named the wrong one (Seattle weeks 3-5: Drew Lock, while Sam Darnold took every snap),
and it carries those labels into upcoming games (Seattle at Denver, week 6: Lock). The props roles had a
second gap: a player with no games this season gets a starter's prior share, so Jalen Milroe (no snaps all
season) ranked QB1 over Darnold and Irvin Charles (just off IR) WR2 over Cooper Kupp. (user, 2026-10-15)

- A played game's QB is its starter: the one who took the team's first dropback. Not the one with the most
  dropbacks: when a starter is hurt in the first quarter, his backup throws most of the passes, and a backtest
  that credited the backup would be using what happened after kickoff (CLAUDE.md rule 1). Who starts is known
  before kickoff.
- An upcoming game's QB is the depth chart's top quarterback who is on that week's active roster and not
  ruled Out or Doubtful. nflverse's depth charts are refreshed daily; on 2026-10-15 they were right in all
  three week-6 games where the schedule disagreed (SEA Darnold; CHI Caleb Williams, back from a hamstring;
  TB Baker Mayfield, back from a thumb, unless Friday's report rules him out).
- Props roles go to the depth chart's starters, ordered among themselves by projected usage.
"""
import polars as pl

from config import RAW

POS = {"FB": "RB"}   # props treat fullbacks as running backs


def latest_depth(season: int) -> pl.DataFrame:
    """The latest depth chart per team: team, gsis_id, position, dc_rank (1 = first at the position).
    Empty when the season's file is missing or in the older weekly format."""
    f = RAW / "depth_charts" / f"{season}.parquet"
    empty = pl.DataFrame(schema={"team": pl.String, "gsis_id": pl.String, "position": pl.String, "dc_rank": pl.Int64})
    if not f.exists():
        return empty
    d = pl.read_parquet(f)
    if not {"dt", "team", "gsis_id", "pos_abb", "pos_rank"} <= set(d.columns):
        return empty
    d = d.filter(pl.col("gsis_id").is_not_null() & (pl.col("dt") == pl.col("dt").max().over("team")))
    # a fullback counts as a running back for the props, but behind every listed running back: the FB1 is
    # not the RB1 (Seattle's Brady Russell)
    return (d.with_columns(position=pl.col("pos_abb").replace(POS),
                           pos_rank=pl.when(pl.col("pos_abb") == "FB").then(pl.col("pos_rank") + 10).otherwise(pl.col("pos_rank")))
            .group_by("team", "gsis_id", "position").agg(dc_rank=pl.col("pos_rank").min().cast(pl.Int64)))


def available(season: int, week: int) -> pl.DataFrame:
    """gsis_id, team of everyone on the week's active roster (the latest week published) and not ruled Out or
    Doubtful on that week's injury report."""
    r = pl.read_parquet(RAW / "rosters_weekly" / f"{season}.parquet")
    wk = week if r.filter(pl.col("week") == week).height else int(r["week"].max())
    r = r.filter((pl.col("week") == wk) & (pl.col("status") == "ACT")).select("gsis_id", "team")
    inj = RAW / "injuries" / f"{season}.parquet"
    if inj.exists():
        out = pl.read_parquet(inj).filter((pl.col("week") == week) & pl.col("report_status").is_in(["Out", "Doubtful"]))
        r = r.join(out.select("gsis_id"), on="gsis_id", how="anti")
    return r.unique()


def upcoming_qbs(season: int, week: int) -> pl.DataFrame:
    """team, qb_id: the depth chart's top available quarterback for each team this week."""
    d = latest_depth(season).filter(pl.col("position") == "QB")
    if not d.height:
        return pl.DataFrame(schema={"team": pl.String, "qb_id": pl.String})
    a = available(season, week)
    return (d.join(a, on=["gsis_id", "team"], how="semi").sort("dc_rank")
            .group_by("team", maintain_order=True).first().select("team", qb_id="gsis_id"))


def played_qbs(plays: pl.DataFrame) -> pl.DataFrame:
    """game_id, team, qb_id: each team's starting quarterback in each game played (its first dropback)."""
    return (plays.filter(pl.col("qb_id").is_not_null()).sort("game_id", "posteam", "play_id")
            .group_by("game_id", "posteam", maintain_order=True).first()
            .select("game_id", team="posteam", qb_id="qb_id"))


def fix_qbs(games: pl.DataFrame, plays: pl.DataFrame) -> pl.DataFrame:
    """The games table with each side's QB from the play-by-play (played games) or the depth chart (upcoming
    games of the current season); the schedule's label stays only where neither has an answer."""
    names = dict(pl.read_parquet(RAW / "players.parquet").select("gsis_id", "display_name").drop_nulls().iter_rows())
    played = played_qbs(plays)
    up = games.filter(pl.col("result").is_null())
    fixed = []
    if up.height:
        season = int(up["season"].min())
        for (wk,), _ in up.filter(pl.col("season") == season).group_by("week"):
            q = upcoming_qbs(season, int(wk))
            fixed.append(up.filter((pl.col("season") == season) & (pl.col("week") == wk)).select("game_id", "home_team", "away_team")
                         .join(q, left_on="home_team", right_on="team", how="left").rename({"qb_id": "home_dc"})
                         .join(q, left_on="away_team", right_on="team", how="left").rename({"qb_id": "away_dc"})
                         .select("game_id", "home_dc", "away_dc"))
    dc = pl.concat(fixed) if fixed else pl.DataFrame(schema={"game_id": pl.String, "home_dc": pl.String, "away_dc": pl.String})
    for side in ("home", "away"):
        games = (games.join(played.rename({"team": f"{side}_team", "qb_id": f"{side}_pbp"}), on=["game_id", f"{side}_team"], how="left")
                 .join(dc.select("game_id", f"{side}_dc"), on="game_id", how="left")
                 .with_columns(pl.coalesce(f"{side}_pbp", f"{side}_dc", f"{side}_qb_id").alias(f"{side}_qb_id"))
                 .with_columns(pl.when(pl.col(f"{side}_pbp").is_not_null() | pl.col(f"{side}_dc").is_not_null())
                               .then(pl.col(f"{side}_qb_id").replace_strict(names, default=None))
                               .otherwise(None).fill_null(pl.col(f"{side}_qb_name")).alias(f"{side}_qb_name"))
                 .drop(f"{side}_pbp", f"{side}_dc"))
    return games
