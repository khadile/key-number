"""Defensive stats for the Defense tab and each game's matchup panel.

Everything is computed from nflverse play-by-play, plus PFR pressures and FTN
blitz charting. "Rank" is always 1 = best defense (fewest allowed, or the
most of a good thing like sacks or takeaways), 32 = worst.
"""
import polars as pl

from config import RAW

POS_GROUP = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB", "QB": "QB"}


def _pbp(seasons: list[int]) -> pl.DataFrame:
    cols = ["game_id", "play_id", "season", "week", "season_type", "game_date", "posteam", "defteam",
            "pass", "rush", "down", "qb_dropback", "sack", "qb_hit", "pass_attempt", "rush_attempt",
            "complete_pass", "interception", "fumble_lost", "pass_touchdown", "rush_touchdown",
            "touchdown", "td_team", "passing_yards", "rushing_yards", "receiving_yards", "yards_gained",
            "epa", "success", "third_down_converted", "third_down_failed", "fixed_drive",
            "fixed_drive_result", "drive_inside20", "receiver_player_id", "rusher_player_id", "passer_player_id"]
    frames = [pl.read_parquet(RAW / "pbp" / f"{s}.parquet", columns=cols) for s in seasons
              if (RAW / "pbp" / f"{s}.parquet").exists()]
    p = pl.concat(frames, how="vertical_relaxed")
    return p.filter(pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null())


def _positions(season: int) -> pl.DataFrame:
    """gsis_id -> position group. Weekly rosters first (position that season), players table as fallback."""
    players = pl.read_parquet(RAW / "players.parquet", columns=["gsis_id", "position"])
    r = RAW / "rosters_weekly" / f"{season}.parquet"
    if r.exists():
        wk = (pl.read_parquet(r, columns=["gsis_id", "position"]).filter(pl.col("gsis_id").is_not_null())
              .group_by("gsis_id").agg(pl.col("position").mode().first()))
        players = wk.join(players, on="gsis_id", how="full", coalesce=True).select(
            "gsis_id", position=pl.coalesce("position", "position_right"))
    return players.with_columns(group=pl.col("position").replace_strict(POS_GROUP, default=None)).select("gsis_id", "group")


def _rank(df: pl.DataFrame, col: str, good_high: bool) -> pl.Expr:
    return pl.col(col).rank(method="min", descending=good_high).cast(pl.Int32).alias(f"{col}_rk")


# ---------------------------------------------------------------- Defense tab

def season_table(season: int) -> tuple[list[dict], dict]:
    """One row per defense for the season to date, with ranks, and league averages."""
    p = _pbp([season]).filter(pl.col("season_type") == "REG")
    scrim = p.filter(((pl.col("pass") == 1) | (pl.col("rush") == 1)) & pl.col("down").is_not_null())
    games = (pl.read_parquet(RAW / "schedules.parquet")
             .filter((pl.col("season") == season) & (pl.col("game_type") == "REG") & pl.col("result").is_not_null()))
    allowed = pl.concat([
        games.select(team="home_team", pa="away_score", game_id="game_id"),
        games.select(team="away_team", pa="home_score", game_id="game_id"),
    ]).with_columns(pl.col("team").replace({"OAK": "LV", "SD": "LAC", "STL": "LA"}))
    base = allowed.group_by("team").agg(g=pl.len(), pts=pl.col("pa").sum())

    d = scrim.group_by("defteam").agg(
        plays=pl.len(),
        pass_yds=(pl.col("passing_yards").fill_null(0) + pl.when(pl.col("sack") == 1).then(pl.col("yards_gained")).otherwise(0)).sum(),
        gross_pass_yds=pl.col("passing_yards").fill_null(0).sum(),
        rush_yds=pl.col("rushing_yards").fill_null(0).sum(),
        att=((pl.col("pass_attempt") == 1) & (pl.col("sack") != 1)).sum(),
        cmp=pl.col("complete_pass").sum(),
        carries=((pl.col("rush_attempt") == 1)).sum(),
        dropbacks=pl.col("qb_dropback").sum(),
        sacks=pl.col("sack").sum(),
        qb_hits=pl.col("qb_hit").sum(),
        ints=pl.col("interception").sum(),
        fum=pl.col("fumble_lost").sum(),
        pass_td=pl.col("pass_touchdown").sum(),
        rush_td=pl.col("rush_touchdown").sum(),
        epa=pl.col("epa").mean(),
        pass_epa=pl.col("epa").filter(pl.col("qb_dropback") == 1).mean(),
        rush_epa=pl.col("epa").filter(pl.col("qb_dropback") != 1).mean(),
        sr=pl.col("success").mean(),
        third_conv=pl.col("third_down_converted").sum(),
        third_fail=pl.col("third_down_failed").sum(),
        expl_pass=((pl.col("qb_dropback") == 1) & (pl.col("yards_gained") >= 20)).sum(),
        expl_rush=((pl.col("qb_dropback") != 1) & (pl.col("yards_gained") >= 10)).sum(),
        stuffs=((pl.col("rush_attempt") == 1) & (pl.col("qb_dropback") != 1) & (pl.col("yards_gained") <= 0)).sum(),
        designed_runs=((pl.col("rush_attempt") == 1) & (pl.col("qb_dropback") != 1)).sum(),
    ).rename({"defteam": "team"})

    # Red zone: opponent drives that reached the 20, and how many ended in a touchdown.
    rz = (scrim.filter(pl.col("drive_inside20") == 1)
          .group_by("defteam", "game_id", "fixed_drive")
          .agg(td=(pl.col("fixed_drive_result").first() == "Touchdown"))
          .group_by("defteam").agg(rz_drives=pl.len(), rz_td=pl.col("td").sum())
          .rename({"defteam": "team"}))
    # Touchdowns allowed of every kind (offense plus returns), as Fox counts them.
    tds = (p.filter((pl.col("touchdown") == 1) & pl.col("td_team").is_not_null())
           .with_columns(team=pl.when(pl.col("td_team") == pl.col("posteam")).then(pl.col("defteam")).otherwise(pl.col("posteam")))
           .group_by("game_id", "play_id", "team").agg()
           .group_by("team").agg(td_allowed=pl.len()))
    pfr = (pl.read_parquet(RAW / "pfr_def" / f"{season}.parquet")
           .filter(pl.col("game_type") == "REG")
           .group_by("team").agg(pressures=pl.col("def_pressures").sum(), hurries=pl.col("def_times_hurried").sum(),
                                 blitzes_pfr=pl.col("def_times_blitzed").sum()))
    ftn_path = RAW / "ftn_charting" / f"{season}.parquet"
    blitz = None
    if ftn_path.exists():
        ftn = pl.read_parquet(ftn_path, columns=["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers"]
                              ).with_columns(pl.col("nflverse_play_id").cast(pl.Int64))
        db = scrim.filter(pl.col("qb_dropback") == 1).select(
            "game_id", pl.col("play_id").cast(pl.Int64), "defteam")
        blitz = (db.join(ftn.rename({"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}),
                         on=["game_id", "play_id"], how="inner")
                 .group_by("defteam").agg(blitz_rate=(pl.col("n_blitzers") > 0).mean(),
                                          rushers=pl.col("n_pass_rushers").mean())
                 .rename({"defteam": "team"}))

    t = base.join(d, on="team").join(rz, on="team", how="left").join(tds, on="team", how="left").join(pfr, on="team", how="left")
    if blitz is not None:
        t = t.join(blitz, on="team", how="left")
    t = t.with_columns(pl.col("rz_drives", "rz_td", "td_allowed", "pressures").fill_null(0))
    t = t.with_columns(
        pts_pg=pl.col("pts") / pl.col("g"),
        yds_pg=(pl.col("pass_yds") + pl.col("rush_yds")) / pl.col("g"),
        pass_yds_pg=pl.col("pass_yds") / pl.col("g"),
        rush_yds_pg=pl.col("rush_yds") / pl.col("g"),
        ypa=pl.col("gross_pass_yds") / pl.col("att"),
        ypc=pl.col("rush_yds") / pl.col("carries"),
        cmp_pct=pl.col("cmp") / pl.col("att"),
        pass_td_pg=pl.col("pass_td") / pl.col("g"),
        rush_td_pg=pl.col("rush_td") / pl.col("g"),
        third_pct=pl.col("third_conv") / (pl.col("third_conv") + pl.col("third_fail")),
        rz_pct=pl.when(pl.col("rz_drives") > 0).then(pl.col("rz_td") / pl.col("rz_drives")),
        sacks_pg=pl.col("sacks") / pl.col("g"),
        sack_rate=pl.col("sacks") / pl.col("dropbacks"),
        pressure_rate=pl.col("pressures") / pl.col("dropbacks"),
        qb_hits_pg=pl.col("qb_hits") / pl.col("g"),
        takeaways=pl.col("ints") + pl.col("fum"),
        expl_pass_rate=pl.col("expl_pass") / pl.col("dropbacks"),
        expl_rush_rate=pl.col("expl_rush") / pl.col("designed_runs"),
        stuff_rate=pl.col("stuffs") / pl.col("designed_runs"),
    )
    pos = position_allowed(season, p)
    t = t.join(pos, on="team", how="left")

    # (column, higher is better for the defense?)
    ranked = [("pts_pg", False), ("yds_pg", False), ("pass_yds_pg", False), ("rush_yds_pg", False),
              ("ypa", False), ("ypc", False), ("cmp_pct", False), ("pass_td_pg", False), ("rush_td_pg", False),
              ("third_pct", False), ("rz_pct", False), ("epa", False), ("pass_epa", False), ("rush_epa", False),
              ("sr", False), ("sacks", True), ("sack_rate", True), ("pressures", True), ("pressure_rate", True),
              ("qb_hits_pg", True), ("takeaways", True), ("ints", True), ("expl_pass_rate", False),
              ("expl_rush_rate", False), ("stuff_rate", True), ("td_allowed", False)]
    if blitz is not None:
        ranked.append(("blitz_rate", True))  # ranked only for sorting; blitzing more is a style, shown as such
    ranked += [(c, False) for c in pos.columns if c != "team"]
    t = t.with_columns([_rank(t, c, good) for c, good in ranked])
    league = {c: float(t[c].mean()) for c, _ in ranked if t[c].dtype.is_numeric()}
    keep = ["team", "g"] + [c for c, _ in ranked] + [f"{c}_rk" for c, _ in ranked]
    rows = t.select([c for c in keep if c in t.columns]).sort("pts_pg").to_dicts()
    return rows, league


def position_allowed(season: int, p: pl.DataFrame | None = None) -> pl.DataFrame:
    """Per game allowed to each position group: receiving targets/yards/TDs, RB and QB
    production, and PPR fantasy points (1 per catch, 0.1 per yard, 6 per TD;
    QB 0.04 per pass yard, 4 per pass TD, -2 per interception)."""
    if p is None:
        p = _pbp([season]).filter(pl.col("season_type") == "REG")
    pos = _positions(season)
    g = p.select("game_id", "defteam").unique().group_by("defteam").agg(games=pl.len())

    tg = (p.filter(pl.col("receiver_player_id").is_not_null() & (pl.col("pass_attempt") == 1))
          .join(pos.rename({"gsis_id": "receiver_player_id"}), on="receiver_player_id", how="left")
          .filter(pl.col("group").is_in(["WR", "TE", "RB"]))
          .group_by("defteam", "group").agg(
              tgt=pl.len(), rec=pl.col("complete_pass").sum(),
              yds=pl.col("receiving_yards").fill_null(0).sum(), td=pl.col("pass_touchdown").sum()))
    ru = (p.filter(pl.col("rusher_player_id").is_not_null() & (pl.col("rush_attempt") == 1))
          .join(pos.rename({"gsis_id": "rusher_player_id"}), on="rusher_player_id", how="left")
          .filter(pl.col("group").is_in(["WR", "TE", "RB", "QB"]))
          .group_by("defteam", "group").agg(ryds=pl.col("rushing_yards").fill_null(0).sum(),
                                            rtd=pl.col("rush_touchdown").sum()))
    qb = (p.filter(pl.col("passer_player_id").is_not_null())
          .group_by("defteam").agg(pyds=pl.col("passing_yards").fill_null(0).sum(),
                                   ptd=pl.col("pass_touchdown").sum(), ints=pl.col("interception").sum())
          .with_columns(group=pl.lit("QB")))
    allg = pl.DataFrame({"group": ["QB", "RB", "WR", "TE"]})
    full = (g.join(allg, how="cross")
            .join(tg, on=["defteam", "group"], how="left")
            .join(ru, on=["defteam", "group"], how="left")
            .join(qb, on=["defteam", "group"], how="left")
            .fill_null(0))
    full = full.with_columns(
        fpts=(pl.col("rec") + 0.1 * (pl.col("yds") + pl.col("ryds")) + 6 * (pl.col("td") + pl.col("rtd"))
              + 0.04 * pl.col("pyds") + 4 * pl.col("ptd") - 2 * pl.col("ints")) / pl.col("games"),
        rec_yds=pl.col("yds") / pl.col("games"),
        rush_yds=pl.col("ryds") / pl.col("games"),
        tgts=pl.col("tgt") / pl.col("games"),
        tds=(pl.col("td") + pl.col("rtd") + pl.col("ptd")) / pl.col("games"),
        pass_yds=pl.col("pyds") / pl.col("games"),
    )
    wide = full.pivot(on="group", index="defteam", values=["fpts", "rec_yds", "rush_yds", "tgts", "tds", "pass_yds"])
    keep = ["defteam", "fpts_QB", "fpts_RB", "fpts_WR", "fpts_TE", "pass_yds_QB", "rush_yds_QB",
            "rush_yds_RB", "rec_yds_RB", "rec_yds_WR", "rec_yds_TE", "tgts_RB", "tgts_WR", "tgts_TE",
            "tds_QB", "tds_RB", "tds_WR", "tds_TE"]
    return wide.select(keep).rename({"defteam": "team"})


# ------------------------------------------------------- Game matchup panel

def last_n_window(as_of_season: int, n: int | None = None) -> dict[str, dict]:
    """Pass and run defense over each team's games this season (or its last n, if given),
    ranked of 32, with the same numbers over the last 3 games."""
    p = _pbp([as_of_season])
    scrim = p.filter(((pl.col("pass") == 1) | (pl.col("rush") == 1)) & pl.col("down").is_not_null())
    # Gross passing yards here (box-score style); the Defense tab uses yards net of sacks, as Fox does.
    per_game = scrim.group_by("defteam", "game_id", "game_date").agg(
        pass_yds=pl.col("passing_yards").fill_null(0).sum(),
        gross=pl.col("passing_yards").fill_null(0).sum(),
        att=((pl.col("pass_attempt") == 1) & (pl.col("sack") != 1)).sum(),
        pass_td=pl.col("pass_touchdown").sum(),
        ints=pl.col("interception").sum(),
        rush_yds=pl.col("rushing_yards").fill_null(0).sum(),
        rush_td=pl.col("rush_touchdown").sum(),
        carries=(pl.col("rush_attempt") == 1).sum(),
    ).sort("defteam", "game_date", "game_id")

    def window(k: int | None) -> pl.DataFrame:
        w = per_game if k is None else per_game.group_by("defteam").tail(k)
        return w.group_by("defteam").agg(
            games=pl.len(),
            pass_yds=pl.col("pass_yds").mean(), pass_td=pl.col("pass_td").mean(),
            ypa=pl.col("gross").sum() / pl.col("att").sum(), ints=pl.col("ints").mean(),
            rush_yds=pl.col("rush_yds").mean(), rush_td=pl.col("rush_td").mean(),
            ypc=pl.col("rush_yds").sum() / pl.col("carries").sum(), carries=pl.col("carries").mean(),
        )

    long, short = window(n), window(3)
    metrics = [("pass_yds", False), ("pass_td", False), ("ypa", False), ("ints", True),
               ("rush_yds", False), ("rush_td", False), ("ypc", False), ("carries", False)]
    long = long.with_columns([_rank(long, m, good) for m, good in metrics])
    out = {}
    for row in long.join(short, on="defteam", suffix="_l3").to_dicts():
        out[row["defteam"]] = {
            "games": row["games"],
            **{m: {"v": row[m], "rk": row[f"{m}_rk"], "l3": row[f"{m}_l3"]} for m, _ in metrics},
        }
    return out


def role_table(season: int) -> dict[str, list[dict]]:
    """What each defense allowed to opponents' WR1/WR2/WR3/TE1/RB1 this season.

    Roles come from usage: each opponent's receivers ranked by season targets
    within their own team and position. A record of what happened, not a
    forecast: how a defense does against a role barely carries from one half
    of a season to the next.
    """
    p = _pbp([season]).filter(pl.col("season_type") == "REG")
    pos = _positions(season)
    tg = (p.filter(pl.col("receiver_player_id").is_not_null() & (pl.col("pass_attempt") == 1))
          .join(pos.rename({"gsis_id": "receiver_player_id"}), on="receiver_player_id", how="left")
          .filter(pl.col("group").is_in(["WR", "TE", "RB"])))
    usage = (tg.group_by("posteam", "group", "receiver_player_id").agg(season_tgt=pl.len())
             .with_columns(rank=pl.col("season_tgt").rank("ordinal", descending=True).over("posteam", "group")))
    usage = usage.with_columns(role=pl.concat_str(pl.col("group"), pl.col("rank").cast(pl.Utf8))).filter(
        pl.col("role").is_in(["WR1", "WR2", "WR3", "TE1", "RB1"]))
    games = (tg.join(usage.select("posteam", "receiver_player_id", "role"), on=["posteam", "receiver_player_id"])
             .group_by("defteam", "role", "game_id").agg(
                 tgt=pl.len(), rec=pl.col("complete_pass").sum(), yds=pl.col("receiving_yards").fill_null(0).sum()))
    per = games.group_by("defteam", "role").agg(
        g=pl.len(), tgt=pl.col("tgt").sum(), rec=pl.col("rec").sum(), yds=pl.col("yds").sum())
    per = per.with_columns(ypg=pl.col("yds") / pl.col("g"), ypt=pl.col("yds") / pl.col("tgt"),
                           cmp=pl.col("rec") / pl.col("tgt"))
    league = per.group_by("role").agg(lg=pl.col("yds").sum() / pl.col("g").sum())
    per = per.join(league, on="role").with_columns(
        vs_lg=pl.col("ypg") - pl.col("lg"),
        rk=pl.col("ypg").rank("min").over("role").cast(pl.Int32),
        n=pl.col("ypg").count().over("role"))
    out: dict[str, list[dict]] = {}
    order = {"RB1": 0, "TE1": 1, "WR1": 2, "WR2": 3, "WR3": 4}
    for r in per.sort("defteam", pl.col("role").replace_strict(order)).to_dicts():
        out.setdefault(r["defteam"], []).append(
            {k: r[k] for k in ("role", "ypg", "vs_lg", "ypt", "cmp", "rk", "n", "g", "tgt")})
    return out
