"""Turn raw nflverse files into the three tables the model reads.

data/clean/plays.parquet       one row per scrimmage play (pass or designed run)
data/clean/team_games.parquet  one row per team per game (pace, tendency, ST)
data/clean/games.parquet       one row per game: teams, result, lines, context

    python -m pipeline.clean
"""
import polars as pl

from config import CLEAN, FIRST_SEASON, RAW

# pbp already uses current abbreviations for every season; schedules does not.
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Plays outside this win-probability band still count, at reduced weight.
# Uses plain wp, never vegas_wp (CLAUDE.md rule 1).
GARBAGE_LO, GARBAGE_HI, GARBAGE_WEIGHT = 0.10, 0.90, 0.25

PBP_COLS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
    "home_team", "away_team", "pass", "rush", "qb_dropback", "qb_scramble",
    "qb_kneel", "qb_spike", "special_teams_play", "two_point_attempt", "epa",
    "qb_epa", "success", "cpoe", "xpass", "pass_oe", "wp", "down", "qtr",
    "half_seconds_remaining", "game_seconds_remaining", "fixed_drive",
    "passer_id", "id", "sack", "interception", "fumble_lost", "no_huddle",
]


def _load_pbp() -> pl.DataFrame:
    files = sorted((RAW / "pbp").glob("*.parquet"))
    return pl.concat(
        [pl.read_parquet(f, columns=PBP_COLS) for f in files if int(f.stem) >= FIRST_SEASON],
        how="vertical_relaxed",
    )


def build_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    scrimmage = pbp.filter(
        ((pl.col("pass") == 1) | (pl.col("rush") == 1))
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("down").is_not_null()          # drops 2-pt tries
        & (pl.col("qb_kneel") != 1)
        & (pl.col("qb_spike") != 1)
    )
    return scrimmage.with_columns(
        dropback=(pl.col("qb_dropback") == 1),
        # Designed runs only; scrambles belong to the pass game.
        designed_run=(pl.col("rush") == 1) & (pl.col("qb_dropback") != 1),
        home=(pl.col("posteam") == pl.col("home_team")).cast(pl.Int8),
        qb_id=pl.when(pl.col("qb_dropback") == 1)
        .then(pl.coalesce("passer_id", "id"))
        .otherwise(None),
        neutral=pl.col("wp").is_between(0.20, 0.80)
        & (pl.col("down") <= 2)
        & (pl.col("half_seconds_remaining") > 120),
        g=pl.when(pl.col("wp").is_between(GARBAGE_LO, GARBAGE_HI))
        .then(1.0)
        .otherwise(GARBAGE_WEIGHT),
        success=pl.col("success").fill_null(0.0),
    ).select(
        "game_id", "play_id", "season", "week", "season_type", "posteam",
        "defteam", "home", "dropback", "designed_run", "qb_id", "epa", "qb_epa",
        "success", "cpoe", "xpass", "pass_oe", "neutral", "g", "wp",
        "game_seconds_remaining", "fixed_drive", "sack", "interception",
        "fumble_lost", "no_huddle",
    )


def build_team_games(pbp: pl.DataFrame, plays: pl.DataFrame) -> pl.DataFrame:
    # Seconds between consecutive snaps of the same drive, neutral plays only.
    paced = (
        plays.sort("game_id", "play_id")
        .with_columns(
            gap=pl.col("game_seconds_remaining").shift(1).over("game_id", "posteam", "fixed_drive")
            - pl.col("game_seconds_remaining")
        )
        .filter(pl.col("neutral") & pl.col("gap").is_between(1, 60))
    )
    pace = paced.group_by("game_id", "posteam").agg(neutral_sec_per_play=pl.col("gap").mean())

    offense = plays.group_by("game_id", "posteam").agg(
        plays=pl.len(),
        neutral_pass_rate=pl.col("dropback").filter(pl.col("neutral")).mean(),
        neutral_proe=pl.col("pass_oe").filter(pl.col("neutral")).mean() / 100,
        no_huddle_rate=pl.col("no_huddle").mean(),
    )

    st = pbp.filter((pl.col("special_teams_play") == 1) & pl.col("epa").is_not_null())
    st_for = st.group_by("game_id", "posteam").agg(st_for=pl.col("epa").sum())
    st_against = st.group_by("game_id", "defteam").agg(st_against=pl.col("epa").sum())

    return (
        offense.rename({"posteam": "team"})
        .join(pace.rename({"posteam": "team"}), on=["game_id", "team"], how="left")
        .join(st_for.rename({"posteam": "team"}), on=["game_id", "team"], how="left")
        .join(st_against.rename({"defteam": "team"}), on=["game_id", "team"], how="left")
        .with_columns(st_net_epa=pl.col("st_for").fill_null(0) - pl.col("st_against").fill_null(0))
        .drop("st_for", "st_against")
    )


def build_games() -> pl.DataFrame:
    s = pl.read_parquet(RAW / "schedules.parquet").filter(pl.col("season") >= FIRST_SEASON)
    for col in ("home_team", "away_team"):
        s = s.with_columns(pl.col(col).replace(FRANCHISE))
    # game_id keeps the historical abbreviations (2016_01_OAK_NO); pbp uses the
    # same ids, so it is left alone and only the team columns are mapped.
    s = s.with_columns(
        # gametime is Eastern; store kickoff as UTC so snapshots compare cleanly.
        kickoff_utc=pl.concat_str("gameday", pl.lit(" "), "gametime")
        .str.to_datetime("%Y-%m-%d %H:%M")
        .dt.replace_time_zone("America/New_York")
        .dt.convert_time_zone("UTC"),
        neutral_site=(pl.col("location") == "Neutral").cast(pl.Int8),
    )
    # Upcoming games have no roof yet; use the stadium's usual setting.
    usual = (
        s.filter(pl.col("roof").is_not_null())
        .group_by("stadium_id")
        .agg(usual_roof=pl.col("roof").mode().first())
    )
    s = s.join(usual, on="stadium_id", how="left").with_columns(
        roof=pl.coalesce("roof", "usual_roof", pl.lit("outdoors")),
    ).with_columns(dome=pl.col("roof").is_in(["dome", "closed"]).cast(pl.Int8))
    return s.select(
        "game_id", "season", "week", "game_type", "gameday", "gametime", "kickoff_utc",
        "away_team", "home_team", "away_score", "home_score", "result", "total",
        "overtime", "spread_line", "home_spread_odds", "away_spread_odds",
        "total_line", "over_odds", "under_odds", "home_moneyline", "away_moneyline",
        "away_rest", "home_rest", "div_game", "neutral_site", "roof", "dome",
        "surface", "temp", "wind", "away_qb_id", "home_qb_id", "away_qb_name",
        "home_qb_name", "away_coach", "home_coach", "stadium",
    )


def main() -> None:
    pbp = _load_pbp()
    plays = build_plays(pbp)
    team_games = build_team_games(pbp, plays)
    games = build_games()

    # Every pbp game must map to a schedule row, or ratings silently lose games.
    missing = set(plays["game_id"].unique()) - set(games["game_id"].unique())
    if missing:
        raise SystemExit(f"{len(missing)} pbp games missing from schedules, e.g. {sorted(missing)[:5]}")

    plays.write_parquet(CLEAN / "plays.parquet")
    team_games.write_parquet(CLEAN / "team_games.parquet")
    games.write_parquet(CLEAN / "games.parquet")
    print(f"plays {plays.height:,}  team_games {team_games.height:,}  games {games.height:,}")


if __name__ == "__main__":
    main()
