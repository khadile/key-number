"""The NBA game model, rebuilt 2026-11-12 (user: "upgrade our model to better predict lines").

The team ratings of nba/model.py stay, as one input. What is new is who is on the floor, valued from the box
score, and a regression that weighs it all:

- Every player carries pre-game rates per minute (points, shots, threes, free throws, rebounds, assists, steals,
  blocks, turnovers, fouls, on-court plus-minus), exponentially weighted over the games he was available for
  (half-life `HL` games), shrunk toward last season's league rate by `M0` minutes, and expected minutes (recent
  minutes when available, half-life `HL_MIN` games, a dressed-but-unused game counting as zero).
- A team's features are those rates summed over the players AVAILABLE tonight at their expected minutes scaled to
  240. Available means played or dressed and not used (coach's decision): known before the tip. The old lineup
  adjustment used who got minutes, which leaks the game: a blowout empties both benches, and its disagreements
  with the close covered 46% on the holdout.
- Margin: a ridge on the team-rating margin, the home-minus-away player features and rest. Total: the same with
  sums, everything measured against the league's last `ENV_DAYS` game days, because the scoring level moves with
  the season and the year (rules, pace) faster than any fit on past seasons can follow.
- Weights are fitted on finished seasons only (`train`), kept in nba/model_weights.json, and the backtest fits
  them walk-forward: a season is always predicted from the seasons before it (nba/backtest.py).

Walk-forward, tuned on 2020-21 to 2022-23 and held out on 2023-24 to 2025-26 (mean absolute error, points):
margin 10.83 -> 10.53 and 11.26 -> 10.89 (the close 10.35, 10.66); total 14.71 -> 14.59 and 14.80 -> 14.57 (the
close 14.20, 14.29). About 60% of the distance to the closing line on margins, half on totals. It still does not
beat the close: where it disagrees with the closing spread by 3+ points its side covered 50.0% and 46.3%; on
totals 50.9% and 51.8%. A better number to read a game by, not a reason to bet against the line.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from nba.data import load
from nba.model import fit, team_games

STATS = ["pts", "fga", "fta", "fg3m", "fg3a", "orb", "drb", "ast", "stl", "blk", "tov", "pf", "pm"]
HL, M0, HL_MIN = 40.0, 400.0, 15.0      # tuned on 2020-21 to 2022-23 (flat between 25-70 games and 200-800 minutes)
ENV_DAYS = 12                           # league game days behind the totals' scoring level
LAM = 30.0
TOP = 13                                # players a live roster is cut to (a box score dresses 11 to 13)
WEIGHTS = Path(__file__).with_name("model_weights.json")
REST = ["b2b_h", "b2b_a", "rest_d"]
MARGIN_COLS = ["team_margin"] + [f"d_{s}" for s in STATS] + ["d_new", "d_start"] + REST
TOTAL_COLS = ["r_tm_total", "r_pace", "b2b_h", "b2b_a"] + [f"r_s_{s}" for s in STATS] + ["s_new", "s_start"]
ENV_COLS = ["total", "tm_total", "pace"] + [f"s_{s}" for s in STATS]


# ---------------------------------------------------------------- players

def player_box(seasons) -> pl.DataFrame:
    """One row per player per game he was available for (played, or dressed and unused), oldest first per player."""
    pb = load("player_box", seasons).filter(pl.col("season_type").is_in([2, 5]))
    f = lambda c: pl.col(c).cast(pl.Float64).fill_null(0.0)
    return (pb.select(pl.col("game_id").cast(pl.Int64), "season", pl.col("game_date").cast(pl.Date).alias("date"),
                      pl.col("athlete_id").cast(pl.Int64), pl.col("team_abbreviation").alias("team"),
                      f("minutes").alias("min"), f("points").alias("pts"), f("field_goals_attempted").alias("fga"),
                      f("free_throws_attempted").alias("fta"), f("three_point_field_goals_made").alias("fg3m"),
                      f("three_point_field_goals_attempted").alias("fg3a"), f("offensive_rebounds").alias("orb"),
                      f("defensive_rebounds").alias("drb"), f("assists").alias("ast"), f("steals").alias("stl"),
                      f("blocks").alias("blk"), f("turnovers").alias("tov"), f("fouls").alias("pf"),
                      pl.col("plus_minus").cast(pl.String).str.replace(r"^\+", "").cast(pl.Float64, strict=False).fill_null(0.0).alias("pm"),
                      pl.col("starter").fill_null(False).cast(pl.Float64).alias("start"),
                      (pl.col("did_not_play").fill_null(False)
                       & pl.col("reason").fill_null("").str.to_uppercase().str.starts_with("COACH")).alias("dnp_cd"))
              .filter((pl.col("min") > 0) | pl.col("dnp_cd")).unique(["game_id", "athlete_id"])
              .sort("athlete_id", "date", "game_id"))


def _league(p: pl.DataFrame) -> pl.DataFrame:
    """season -> the league's per-minute rates of the season before (its own for the first one): the prior a
    player with little history is shrunk toward, known before the season starts."""
    r = p.group_by("season").agg([(pl.col(s).sum() / pl.col("min").sum()).alias(f"lg_{s}") for s in STATS]).sort("season")
    nxt = r.with_columns(season=pl.col("season") + 1)
    first = r.head(1)
    return pl.concat([first, nxt.filter(pl.col("season") > first["season"][0])])


def player_rates(p: pl.DataFrame) -> pl.DataFrame:
    """`p` with each row's PRE-GAME numbers: r_<stat> per minute, exp_min, exp_start, m_eff (weighted minutes behind
    the rates). Nothing from the row's own game is used."""
    lam = 0.5 ** (1 / HL)
    n = pl.int_range(0, pl.len()).over("athlete_id")
    eff = (1 - lam ** n) / (1 - lam)
    ew = lambda c: pl.col(c).ewm_mean(half_life=HL, adjust=True).shift(1).over("athlete_id")
    lg = _league(p)
    last = lg.tail(1)
    out = p.join(lg, on="season", how="left").with_columns(m_eff=(ew("min") * eff).fill_null(0.0))
    out = out.with_columns([pl.col(f"lg_{s}").fill_null(last[f"lg_{s}"][0]) for s in STATS])
    out = out.with_columns([(((ew(s) * eff).fill_null(0.0) + pl.col(f"lg_{s}") * M0) / (pl.col("m_eff") + M0)).alias(f"r_{s}") for s in STATS])
    return out.with_columns(
        exp_min=pl.col("min").ewm_mean(half_life=HL_MIN, adjust=True).shift(1).over("athlete_id").fill_null(6.0),
        exp_start=pl.col("start").ewm_mean(half_life=HL_MIN, adjust=True).shift(1).over("athlete_id").fill_null(0.0)).drop([f"lg_{s}" for s in STATS])


def current(p: pl.DataFrame, today: date) -> pl.DataFrame:
    """Every player's numbers going into a game today: athlete_id, r_<stat>, exp_min, exp_start, m_eff."""
    p = p.filter(pl.col("date") < today)
    nxt = (p.group_by("athlete_id").agg(pl.col("team").last())
            .with_columns(game_id=pl.lit(-1, pl.Int64), date=pl.lit(today), dnp_cd=pl.lit(False), season=pl.lit(int(p["season"].max()), p.schema["season"]),
                          **{c: pl.lit(0.0) for c in ["min", "start"] + STATS}))
    both = pl.concat([p, nxt.select(p.columns)]).sort("athlete_id", "date", "game_id")
    # the synthetic row sorts last for its player (latest date; -1 only orders a same-day tie first, so move it)
    both = both.with_columns(_k=(pl.col("game_id") == -1)).sort("athlete_id", "date", "_k").drop("_k")
    r = player_rates(both).filter(pl.col("game_id") == -1)
    return r.select("athlete_id", *[f"r_{s}" for s in STATS], "exp_min", "exp_start", "m_eff")


def _sum(rows: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    rows = rows.with_columns(share=pl.col("exp_min") / pl.col("exp_min").sum().over(by) * 240.0)
    return rows.group_by(by).agg([(pl.col("share") * pl.col(f"r_{s}")).sum().alias(f"x_{s}") for s in STATS]
                                 + [(pl.col("share") * (pl.col("m_eff") < 200)).sum().alias("x_new"),
                                    (pl.col("share") * pl.col("exp_start")).sum().alias("x_start")])


def team_features(pr: pl.DataFrame) -> pl.DataFrame:
    """Per team-game: each stat's expected total over the available players at their expected minutes."""
    return _sum(pr, ["game_id", "team"])


def side_features(cur: pl.DataFrame, ids: list[int], prior: dict[str, float]) -> dict[str, float]:
    """The same for a live roster: the players not ruled out, cut to the `TOP` by expected minutes. A player with
    no history plays 6 minutes at the league's rates."""
    have = cur.filter(pl.col("athlete_id").is_in(ids))
    missing = [a for a in ids if a not in set(have["athlete_id"].to_list())]
    if missing:
        blank = pl.DataFrame({"athlete_id": missing, **{f"r_{s}": [prior[s]] * len(missing) for s in STATS},
                              "exp_min": [6.0] * len(missing), "exp_start": [0.0] * len(missing), "m_eff": [0.0] * len(missing)},
                             schema=have.schema)
        have = pl.concat([have, blank])
    have = have.sort("exp_min", descending=True).head(TOP).with_columns(k=pl.lit(1))
    if have.height == 0 or have["exp_min"].sum() <= 0:
        return {}
    return {c[2:]: float(v) for c, v in _sum(have, ["k"]).drop("k").row(0, named=True).items()}


# ---------------------------------------------------------------- the frame the weights are fitted on

def _homes(tg: pl.DataFrame) -> pl.DataFrame:
    real = tg.group_by("team").len().filter(pl.col("len") >= 40)["team"].to_list()   # the 30 clubs, not the All-Star sides
    t = tg.filter(pl.col("team").is_in(real) & pl.col("opp").is_in(real))
    return t.filter(pl.col("is_home") | (pl.col("home") == 0)).unique("game_id", keep="first").sort("date", "game_id")


def frame(seasons, dates: list[date] | None = None) -> pl.DataFrame:
    """One row per game with every pre-game input: the team ratings' projection (refit before each date), both
    sides' player features, rest, and the result. `dates`: only these game dates (the live build's last few)."""
    seasons = list(seasons)
    tg = team_games(range(min(seasons) - 2, max(seasons) + 1))
    homes = _homes(tg).filter(pl.col("season").is_in(seasons))
    if dates is not None:
        homes = homes.filter(pl.col("date").is_in(dates))
    rows = []
    for (season, d), g in homes.group_by(["season", "date"], maintain_order=True):
        r = fit(tg, d, season)
        if r is None:
            continue
        for x in g.iter_rows(named=True):
            p = r.project(x["team"], x["opp"], neutral=x["home"] == 0)
            if p is not None:
                rows.append({"game_id": x["game_id"], "season": season, "date": d, "home": x["team"], "away": x["opp"],
                             "team_margin": p[0], "tm_total": p[1], "pace": p[2],
                             "result": x["pts"] - x["opp_pts"], "total": x["pts"] + x["opp_pts"]})
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    rest = tg.select("game_id", "team", "rest")
    df = (df.join(rest.rename({"team": "home", "rest": "home_rest"}), on=["game_id", "home"], how="left")
            .join(rest.rename({"team": "away", "rest": "away_rest"}), on=["game_id", "away"], how="left"))
    tf = team_features(player_rates(player_box(range(min(seasons) - 2, max(seasons) + 1))))
    xs = [c for c in tf.columns if c.startswith("x_")]
    df = (df.join(tf.select("game_id", pl.col("team").alias("home"), *[pl.col(c).alias(f"h_{c[2:]}") for c in xs]), on=["game_id", "home"], how="left")
            .join(tf.select("game_id", pl.col("team").alias("away"), *[pl.col(c).alias(f"a_{c[2:]}") for c in xs]), on=["game_id", "away"], how="left"))
    return _derive(df).drop_nulls(["d_pts"])


def _derive(df: pl.DataFrame) -> pl.DataFrame:
    ks = STATS + ["new", "start"]
    return df.with_columns(
        [(pl.col(f"h_{k}") - pl.col(f"a_{k}")).alias(f"d_{k}") for k in ks] + [(pl.col(f"h_{k}") + pl.col(f"a_{k}")).alias(f"s_{k}") for k in ks]
        + [(pl.col("home_rest") == 1).cast(pl.Float64).alias("b2b_h"), (pl.col("away_rest") == 1).cast(pl.Float64).alias("b2b_a"),
           (pl.col("home_rest").clip(1, 4) - pl.col("away_rest").clip(1, 4)).cast(pl.Float64).alias("rest_d")])


def with_env(df: pl.DataFrame) -> pl.DataFrame:
    """Adds env_<col>: the league's mean over the last `ENV_DAYS` game days before each date (earlier dates only),
    and the totals' inputs measured against it (r_tm_total, r_pace, r_s_<stat>, y_rel)."""
    days = df.group_by("date").agg([pl.col(c).sum().alias(c) for c in ENV_COLS] + [pl.len().alias("n")]).sort("date")
    n = days["n"].to_numpy().astype(float)
    cum_n = np.r_[0.0, np.cumsum(n)]
    out = {}
    for c in ENV_COLS:
        cum = np.r_[0.0, np.cumsum(days[c].to_numpy())]
        i = np.arange(days.height)
        lo = np.maximum(0, i - ENV_DAYS)
        den = cum_n[i] - cum_n[lo]
        out[f"env_{c}"] = np.where(den > 0, (cum[i] - cum[lo]) / np.where(den > 0, den, 1.0), np.nan)
    env = days.select("date", *[pl.Series(k, v) for k, v in out.items()])
    return _relative(df.join(env, on="date", how="left"))


def _relative(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([(pl.col("tm_total") - pl.col("env_tm_total")).alias("r_tm_total"), (pl.col("pace") - pl.col("env_pace")).alias("r_pace")]
                           + [(pl.col(f"s_{s}") - pl.col(f"env_s_{s}")).alias(f"r_s_{s}") for s in STATS]
                           + ([(pl.col("total") - pl.col("env_total")).alias("y_rel")] if "total" in df.columns else []))


# ---------------------------------------------------------------- weights

def _ridge(X: np.ndarray, y: np.ndarray) -> dict:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    ym = float(y.mean())
    beta = np.linalg.solve(Z.T @ Z + LAM * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist(), "ym": ym}


def train(F: pl.DataFrame) -> dict:
    """Weights from a frame with its environment (`with_env`)."""
    T = F.filter(pl.col("env_total").is_not_nan() & pl.col("env_total").is_not_null())
    return {"margin": {"cols": MARGIN_COLS, **_ridge(F.select(MARGIN_COLS).to_numpy(), F["result"].to_numpy())},
            "total": {"cols": TOTAL_COLS, **_ridge(T.select(TOTAL_COLS).to_numpy(), T["y_rel"].to_numpy())},
            "games": F.height, "seasons": [int(F["season"].min()), int(F["season"].max())]}


def apply(w: dict, X: np.ndarray) -> np.ndarray:
    return w["ym"] + ((X - np.array(w["mu"])) / np.array(w["sd"])) @ np.array(w["beta"])


def predict(W: dict, F: pl.DataFrame) -> pl.DataFrame:
    """model_margin and model_total for a frame with its environment (the total falls back to the team ratings'
    where the league's recent level is unknown)."""
    m = apply(W["margin"], F.select(MARGIN_COLS).to_numpy())
    t = apply(W["total"], F.select(TOTAL_COLS).fill_nan(0.0).fill_null(0.0).to_numpy()) + F["env_total"].fill_nan(None).to_numpy()
    t = np.where(np.isnan(t.astype(float)), F["tm_total"].to_numpy(), t)
    # how much of the margin the players and the rest move, for the card (standardized coefficients x inputs)
    z = (F.select(MARGIN_COLS).to_numpy() - np.array(W["margin"]["mu"])) / np.array(W["margin"]["sd"])
    part = z * np.array(W["margin"]["beta"])
    is_rest = np.array([c in REST for c in MARGIN_COLS])
    is_pl = np.array([c.startswith("d_") for c in MARGIN_COLS])
    return F.with_columns(model_margin=pl.Series(m), model_total=pl.Series(t.astype(float)),
                          lineup_adj=pl.Series(part[:, is_pl].sum(1)), rest_adj=pl.Series(part[:, is_rest].sum(1)))


def load_weights() -> dict | None:
    return json.loads(WEIGHTS.read_text(encoding="utf-8")) if WEIGHTS.exists() else None


# ---------------------------------------------------------------- live

def live(games: list[dict], today: date, roster: pl.DataFrame, out_ids: set[int], sched: pl.DataFrame, season: int) -> dict[int, dict]:
    """Tonight's games: game_id -> margin, total and their parts. `games`: game_id, home, away, neutral, day;
    `roster`: athlete_id, team; `out_ids`: players ruled out; `sched`: every game's home, away, day, tip."""
    W = load_weights()
    tg = team_games(range(season - 2, season + 1))
    r = fit(tg, today, season)
    if r is None or W is None:
        return {}
    p = player_box(range(season - 2, season + 1))
    cur = current(p, today)
    lg = _league(p).tail(1)
    prior = {s: float(lg[f"lg_{s}"][0]) for s in STATS}
    prev: dict[str, list] = {}
    for g in sched.sort("tip").iter_rows(named=True):
        for t in (g["home"], g["away"]):
            prev.setdefault(t, []).append(g["day"])

    def rest(team: str, day: date) -> int:
        before = [d for d in prev.get(team, []) if d < day]
        return min((day - max(before)).days, 7) if before else 7

    # the league's recent level: the last ENV_DAYS game dates before today, each projected as it was that day
    homes = _homes(tg).filter(pl.col("date") < today)
    recent = homes["date"].unique().sort().tail(ENV_DAYS).to_list()
    R = frame(sorted({int(s) for s in homes.filter(pl.col("date").is_in(recent))["season"].unique()}), dates=recent) if recent else pl.DataFrame()
    env = {f"env_{c}": float(R[c].mean()) for c in ENV_COLS} if R.height else None
    rows, meta = [], {}
    for g in games:
        pr = r.project(g["home"], g["away"], neutral=g["neutral"])
        if pr is None:
            continue
        side = {}
        for k, team in (("h", g["home"]), ("a", g["away"])):
            ids = [a for a in roster.filter(pl.col("team") == team)["athlete_id"].to_list() if a not in out_ids]
            side[k] = side_features(cur, ids, prior)
        if not side["h"] or not side["a"]:
            continue
        hr, ar = rest(g["home"], g["day"]), rest(g["away"], g["day"])
        rows.append({"game_id": g["game_id"], "team_margin": pr[0], "tm_total": pr[1], "pace": pr[2], "home_rest": hr, "away_rest": ar,
                     **{f"h_{k}": v for k, v in side["h"].items()}, **{f"a_{k}": v for k, v in side["a"].items()},
                     **(env or {f"env_{c}": None for c in ENV_COLS})})
        meta[g["game_id"]] = (hr, ar)
    if not rows:
        return {}
    F = _relative(_derive(pl.DataFrame(rows, schema_overrides={f"env_{c}": pl.Float64 for c in ENV_COLS})))
    P = predict(W, F.with_columns(pl.col("env_total").fill_null(float("nan"))))
    out = {}
    for x in P.iter_rows(named=True):
        hr, ar = meta[x["game_id"]]
        out[x["game_id"]] = {"margin": x["model_margin"], "team_margin": x["team_margin"], "b2b": x["rest_adj"], "lineup": x["lineup_adj"],
                             "team_part": x["model_margin"] - x["lineup_adj"] - x["rest_adj"],
                             "total": x["model_total"], "pace": x["pace"], "pace_factor": x["pace"] / r.mu_p, "rest": {"home": hr, "away": ar}}
    return out
