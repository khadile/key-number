"""Player and team game logs for the props model, built from nflverse.

Three tables, rebuilt by `python -m props.data` (and by the board build):

- team_games:   one row per team per game: plays, dropbacks, pass attempts,
                sacks, scrambles, designed runs and targets, plus the game's
                closing spread and total (nflverse), from the team's side.
- player_games: one row per player per game: targets, receptions, receiving
                yards, designed carries and their yards, scrambles, passing,
                offensive snaps, and shares of the team's targets and runs.
- yard_pools:   per-play yardage samples (yards per catch, per designed carry,
                per scramble, per completion) by position, from 2016-2021 only,
                so the simulation's shapes never see a holdout season.

Regular season only. Stats match how books settle props: passing yards are
gross (sacks not subtracted), rushing yards include scrambles, everything
includes overtime.
"""
import polars as pl

from config import CLEAN, FIRST_SEASON, RAW
from pipeline.clean import FRANCHISE

TEAM_GAMES = CLEAN / "props_team_games.parquet"
PLAYER_GAMES = CLEAN / "props_player_games.parquet"
YARD_POOLS = CLEAN / "props_yard_pools.parquet"
POOL_SEASONS = (2016, 2021)

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "qb_dropback",
            "pass_attempt", "rush_attempt", "sack", "qb_scramble", "qb_kneel", "qb_spike", "two_point_attempt",
            "complete_pass", "passer_player_id", "receiver_player_id", "rusher_player_id", "yards_gained",
            "passing_yards", "receiving_yards", "rushing_yards"]


def _fix(col: str) -> pl.Expr:
    return pl.col(col).replace(FRANCHISE)


def _pbp(seasons: list[int]) -> pl.DataFrame:
    frames = []
    for s in seasons:
        path = RAW / "pbp" / f"{s}.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path, columns=PBP_COLS))
    p = pl.concat(frames, how="diagonal_relaxed").filter(pl.col("season_type") == "REG")
    return p.with_columns(_fix("posteam"), _fix("defteam")).filter(
        pl.col("posteam").is_not_null() & (pl.col("two_point_attempt").fill_null(0) == 0))


def build_team_games(p: pl.DataFrame) -> pl.DataFrame:
    scrimmage = p.filter(pl.col("play_type").is_in(["pass", "run"]) & (pl.col("qb_kneel").fill_null(0) == 0)
                         & (pl.col("qb_spike").fill_null(0) == 0))
    t = scrimmage.group_by("game_id", "season", "week", "posteam", "defteam").agg(
        plays=pl.len(),
        dropbacks=pl.col("qb_dropback").fill_null(0).sum(),
        sacks=pl.col("sack").fill_null(0).sum(),
        scrambles=pl.col("qb_scramble").fill_null(0).sum(),
        pass_att=((pl.col("pass_attempt") == 1) & (pl.col("sack").fill_null(0) == 0)).sum(),
        targets=(pl.col("receiver_player_id").is_not_null() & (pl.col("pass_attempt") == 1)
                 & (pl.col("sack").fill_null(0) == 0)).sum(),
        designed_runs=((pl.col("rush_attempt") == 1) & (pl.col("qb_scramble").fill_null(0) == 0)).sum(),
        pass_yds=pl.col("passing_yards").fill_null(0).sum(),
    ).rename({"posteam": "team", "defteam": "opp"})
    g = pl.read_parquet(CLEAN / "games.parquet").select(
        "game_id", "home_team", "spread_line", "total_line", "result", "total", "kickoff_utc")
    t = t.join(g, on="game_id", how="left").with_columns(
        home=(pl.col("team") == pl.col("home_team")).cast(pl.Int8),
        # points this team is favored by (nflverse spread_line is the home team's)
        spread=pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("spread_line")).otherwise(-pl.col("spread_line")),
        margin=pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("result")).otherwise(-pl.col("result")),
    ).drop("home_team", "spread_line", "result")
    return t.sort("season", "week", "game_id", "team")


def build_player_games(p: pl.DataFrame, seasons: list[int], team_games: pl.DataFrame) -> pl.DataFrame:
    stats = []
    for s in seasons:
        path = RAW / "player_stats" / f"{s}.parquet"
        if path.exists():
            stats.append(pl.read_parquet(path))
    ps = (pl.concat(stats, how="diagonal_relaxed").filter(pl.col("season_type") == "REG")
          .select(pl.col("player_id").alias("gsis_id"), pl.col("player_display_name").alias("name"), "position",
                  "season", "week", "game_id", _fix("team").alias("team"), _fix("opponent_team").alias("opp"),
                  pl.col("completions").alias("cmp"), pl.col("attempts").alias("att"), "passing_yards",
                  "carries", "rushing_yards", "receptions", "targets", "receiving_yards"))
    # nflverse player_stats has game_id in recent releases; rebuild it where it is missing
    # Designed runs and scrambles per player, which player_stats lumps together.
    runs = p.filter(pl.col("rush_attempt") == 1).group_by("game_id", pl.col("rusher_player_id").alias("gsis_id")).agg(
        d_car=(pl.col("qb_scramble").fill_null(0) == 0).sum(),
        d_yds=pl.when(pl.col("qb_scramble").fill_null(0) == 0).then(pl.col("rushing_yards").fill_null(0)).otherwise(0).sum(),
        scr=(pl.col("qb_scramble").fill_null(0) == 1).sum(),
        scr_yds=pl.when(pl.col("qb_scramble").fill_null(0) == 1).then(pl.col("rushing_yards").fill_null(0)).otherwise(0).sum(),
    )
    ps = ps.join(runs, on=["game_id", "gsis_id"], how="left").with_columns(
        pl.col("d_car", "d_yds", "scr", "scr_yds").fill_null(0))
    # offensive snaps (PFR ids -> gsis)
    ids = pl.read_parquet(RAW / "players.parquet").select("gsis_id", "pfr_id").drop_nulls()
    snaps = []
    for s in seasons:
        path = RAW / "snap_counts" / f"{s}.parquet"
        if path.exists():
            snaps.append(pl.read_parquet(path).filter(pl.col("game_type") == "REG")
                         .select("game_id", pl.col("pfr_player_id").alias("pfr_id"), "team", "offense_snaps", "offense_pct"))
    if snaps:
        sn = (pl.concat(snaps).join(ids, on="pfr_id").filter(pl.col("offense_snaps") > 0)
              .select("game_id", "gsis_id", "offense_snaps", "offense_pct").unique(["game_id", "gsis_id"]))
        ps = ps.join(sn, on=["game_id", "gsis_id"], how="left")
        # A receiver who played but drew no target has no player_stats row; add those games
        # as zeros, or his target share would only count the games he was thrown to.
        meta = (pl.read_parquet(RAW / "players.parquet").select("gsis_id", "display_name", "position")
                .rename({"display_name": "name"}))
        sides = team_games.select("game_id", "season", "week", "team", "opp")
        played = (pl.concat(snaps).join(ids, on="pfr_id").filter(pl.col("offense_snaps") > 0)
                  .select("game_id", "gsis_id", "offense_snaps", "offense_pct", "team").unique(["game_id", "gsis_id"])
                  .with_columns(_fix("team")))
        extra = (played.join(ps.select("game_id", "gsis_id"), on=["game_id", "gsis_id"], how="anti")
                 .join(meta, on="gsis_id").filter(pl.col("position").is_in(["QB", "RB", "FB", "WR", "TE"]))
                 .join(sides, on=["game_id", "team"]))
        zero = {c: 0 for c in ("cmp", "att", "passing_yards", "carries", "rushing_yards", "receptions",
                               "targets", "receiving_yards", "d_car", "d_yds", "scr", "scr_yds")}
        extra = extra.with_columns(**{c: pl.lit(v) for c, v in zero.items()})
        ps = pl.concat([ps, extra.select(ps.columns)], how="vertical_relaxed")
    tg = team_games.select("game_id", "team", pl.col("targets").alias("team_targets"),
                           pl.col("designed_runs").alias("team_runs"), pl.col("dropbacks").alias("team_dropbacks"),
                           pl.col("plays").alias("team_plays"), pl.col("pass_att").alias("team_att"))
    ps = ps.join(tg, on=["game_id", "team"], how="left").with_columns(
        tgt_share=pl.col("targets") / pl.col("team_targets"),
        run_share=pl.col("d_car") / pl.col("team_runs"),
    )
    return ps.sort("season", "week", "game_id", "team", "gsis_id")


def build_yard_pools(p: pl.DataFrame) -> pl.DataFrame:
    """Per-play yardage samples by position group, from the tuning seasons only."""
    lo, hi = POOL_SEASONS
    p = p.filter(pl.col("season").is_between(lo, hi))
    pos = pl.read_parquet(RAW / "players.parquet").select("gsis_id", "position")
    catches = (p.filter(pl.col("complete_pass") == 1).select(pl.col("receiver_player_id").alias("gsis_id"),
                                                             yds=pl.col("receiving_yards").fill_null(0))
               .join(pos, on="gsis_id").filter(pl.col("position").is_in(["WR", "TE", "RB"]))
               .select(pool=pl.lit("catch_") + pl.col("position"), yds="yds"))
    runs = (p.filter((pl.col("rush_attempt") == 1) & (pl.col("qb_scramble").fill_null(0) == 0))
            .select(pl.col("rusher_player_id").alias("gsis_id"), yds=pl.col("rushing_yards").fill_null(0))
            .join(pos, on="gsis_id").filter(pl.col("position").is_in(["RB", "QB", "WR"]))
            .select(pool=pl.lit("run_") + pl.col("position"), yds="yds"))
    scr = p.filter(pl.col("qb_scramble") == 1).select(pool=pl.lit("scramble"), yds=pl.col("rushing_yards").fill_null(0))
    comp = p.filter((pl.col("complete_pass") == 1) & (pl.col("sack").fill_null(0) == 0)).select(
        pool=pl.lit("completion"), yds=pl.col("passing_yards").fill_null(0))
    return pl.concat([catches, runs, scr, comp]).with_columns(pl.col("yds").cast(pl.Float64))


def build(seasons: list[int] | None = None) -> None:
    import glob
    seasons = seasons or sorted(int(f[-12:-8]) for f in glob.glob(str(RAW / "pbp" / "*.parquet")))
    seasons = [s for s in seasons if s >= FIRST_SEASON]
    p = _pbp(seasons)
    tg = build_team_games(p)
    tg.write_parquet(TEAM_GAMES)
    pg = build_player_games(p, seasons, tg)
    pg.write_parquet(PLAYER_GAMES)
    build_yard_pools(p).write_parquet(YARD_POOLS)
    print(f"props data: {tg.height} team-games, {pg.height} player-games, seasons {seasons[0]}-{seasons[-1]}")


if __name__ == "__main__":
    build()
