"""Walk-forward tests of the props model.

    python -m props.backtest            # tune on 2018-2021, report the 2022-2025 holdout
    python -m props.backtest --market   # 2026 so far against real prop lines (needs the archive)

Projections for a game only use earlier games, and roles (WR1, RB1, ...) are
set from the pre-game projections of the players who suited up, never from
how the game went. Two baselines keep the model honest: the player's average
this season (last season's in week 1) and his last three games.
"""
import argparse
import itertools
import json

import numpy as np
import polars as pl

from config import REPORTS
from props.model import (STAT_COL, Dispersion, Params, Simulator, TeamModel, defense_frame,
                         player_frame, project, team_frame)

TUNE = (2018, 2021)
HOLD = (2022, 2025)
PARAMS_PATH = REPORTS / "props_params.json"
# which markets each tracked role is judged on
ROLE_MARKETS = {"QB1": ("pass_yds", "rush_yds"), "RB1": ("rush_yds", "receptions", "rec_yds"),
                "RB2": ("rush_yds", "receptions", "rec_yds"), "WR1": ("rec_yds", "receptions"),
                "WR2": ("rec_yds", "receptions"), "TE1": ("rec_yds", "receptions")}
HAT = {"pass_yds": "pass_yds_hat", "rush_yds": "rush_yds_hat", "rec_yds": "rec_yds_hat", "receptions": "rec_hat"}
RB2_MIN_SHARE = 0.30  # an RB2 is tracked only in a real timeshare


def assign_roles(x: pl.DataFrame) -> pl.DataFrame:
    """Rank each team's players in a game by projected usage: QB1, RB1-2, WR1-2, TE1.
    `x` holds only the players available for that game, so a teammate ruled out
    simply moves everyone below him up a spot."""
    usage = (pl.when(pl.col("position") == "QB").then(pl.col("att_hat"))
             .when(pl.col("position") == "RB").then(pl.col("car_hat") + pl.col("tgt_hat"))
             .otherwise(pl.col("tgt_hat")))
    x = x.with_columns(usage=usage).with_columns(
        rank=pl.col("usage").rank("ordinal", descending=True).over("game_id", "team", "position"))
    role = (pl.when((pl.col("position") == "QB") & (pl.col("rank") == 1)).then(pl.lit("QB1"))
            .when((pl.col("position") == "RB") & (pl.col("rank") == 1)).then(pl.lit("RB1"))
            .when((pl.col("position") == "RB") & (pl.col("rank") == 2) & (pl.col("rshare") >= RB2_MIN_SHARE)).then(pl.lit("RB2"))
            .when((pl.col("position") == "WR") & (pl.col("rank") <= 2)).then(pl.lit("WR") + pl.col("rank").cast(pl.String))
            .when((pl.col("position") == "TE") & (pl.col("rank") == 1)).then(pl.lit("TE1"))
            .otherwise(None))
    return x.with_columns(role=role)


def baselines(x: pl.DataFrame) -> pl.DataFrame:
    """Season-to-date average (last season's for a player's first game) and last-3 average."""
    x = x.sort("gsis_id", "t")
    out = []
    for m, col in STAT_COL.items():
        v = pl.col(col).cast(pl.Float64)
        prev = v.shift(1).over("gsis_id")
        # season-to-date mean of earlier games
        cs = v.cum_sum().over("gsis_id", "season") - v
        n = pl.int_range(pl.len()).over("gsis_id", "season")
        stod = pl.when(n > 0).then(cs / n)
        out += [stod.alias(f"b_std_{m}"),
                pl.mean_horizontal(prev, v.shift(2).over("gsis_id"), v.shift(3).over("gsis_id")).alias(f"b_l3_{m}")]
    x = x.with_columns(out)
    # week-1 fallback: the player's average over his previous season
    prior = (x.group_by("gsis_id", "season").agg([pl.col(c).cast(pl.Float64).mean().alias(f"ps_{m}") for m, c in STAT_COL.items()])
             .with_columns(season=pl.col("season") + 1))
    x = x.join(prior, on=["gsis_id", "season"], how="left")
    return x.with_columns([pl.coalesce(f"b_std_{m}", f"ps_{m}").alias(f"b_std_{m}") for m in STAT_COL]).drop(
        [f"ps_{m}" for m in STAT_COL])


def frames(p: Params) -> pl.DataFrame:
    tf = team_frame(None, p)
    tm = TeamModel.fit(tf)
    tp = tm.predict(tf)
    pf = player_frame(None, p)
    x = project(pf.filter(pl.col("targets").is_not_null()), tp, defense_frame(pf, tf, p), p)
    return assign_roles(x)


def long_form(x: pl.DataFrame) -> pl.DataFrame:
    """One row per tracked player-game-market with projection, baselines and result."""
    rows = []
    for role, markets in ROLE_MARKETS.items():
        r = x.filter(pl.col("role") == role)
        for m in markets:
            rows.append(r.select("season", "week", "game_id", "team", "opp", "gsis_id", "name", "position", "role",
                                 pl.lit(m).alias("market"), pl.col(HAT[m]).alias("proj"),
                                 pl.col(STAT_COL[m]).cast(pl.Float64).alias("actual"),
                                 pl.col(f"b_std_{m}").alias("b_std"), pl.col(f"b_l3_{m}").alias("b_l3")))
    out = pl.concat(rows)
    # QB rushing only matters for QBs who run; judge it where a book would post it
    return out.filter(~((pl.col("market") == "rush_yds") & (pl.col("role") == "QB1") & (pl.col("proj") < 12)))


def score(L: pl.DataFrame, seasons: tuple[int, int]) -> pl.DataFrame:
    d = L.filter(pl.col("season").is_between(*seasons) & pl.col("b_std").is_not_null() & pl.col("b_l3").is_not_null())
    return d.group_by("market").agg(
        n=pl.len(),
        model=(pl.col("proj") - pl.col("actual")).abs().mean(),
        season_avg=(pl.col("b_std") - pl.col("actual")).abs().mean(),
        last3=(pl.col("b_l3") - pl.col("actual")).abs().mean(),
        bias=(pl.col("proj") - pl.col("actual")).mean(),
    ).with_columns(vs_best=pl.col("model") / pl.min_horizontal("season_avg", "last3")).sort("market")


def objective(L: pl.DataFrame) -> float:
    s = score(L, TUNE)
    return float(s["vs_best"].mean())


def tune() -> Params:
    base = Params()
    grid = {"hl_share": [3.0, 5.0, 8.0], "m_share": [15.0, 30.0, 60.0], "hl_eff": [10.0, 20.0, 40.0],
            "m_ypt": [60.0, 120.0], "def_power": [0.5, 1.0]}
    best, best_p = 9e9, base
    # coordinate search: one parameter at a time, two passes
    cur = base.asdict()
    for _ in range(2):
        for k, vals in grid.items():
            for v in vals:
                trial = Params(**{**cur, k: v})
                L = long_form(baselines(frames(trial)))
                obj = objective(L)
                if obj < best:
                    best, best_p = obj, trial
                    cur = trial.asdict()
            print(f"  {k} -> {cur[k]}  (objective {best:.4f})")
    return best_p


# ---------------------------------------------------------------- distributions

def _pit_rows(d: pl.DataFrame, disp: Dispersion, size: int, seed: int) -> pl.DataFrame:
    """Where each result landed in its simulated distribution (randomized at ties,
    so a zero-yard game against a pile of simulated zeros is not all "below")."""
    rng = np.random.default_rng(seed + 1)
    sim = Simulator(disp, seed=seed)
    out = []
    for i in range(0, d.height, 2000):
        chunk = d.slice(i, 2000)
        S = sim.simulate(chunk, size)
        roles = chunk["role"].to_numpy()
        for role, markets in ROLE_MARKETS.items():
            rows = roles == role
            if not rows.any():
                continue
            sub = chunk.filter(pl.col("role") == role)
            for m in markets:
                s = S[m][rows]
                y = sub[STAT_COL[m]].to_numpy().astype(float)
                if m == "receptions":
                    y = y + rng.uniform(-0.5, 0.5, size=y.shape)
                below, tie = (s < y[:, None]).mean(axis=1), (s == y[:, None]).mean(axis=1)
                u = below + rng.uniform(size=y.shape) * tie
                out.append(pl.DataFrame({"market": m, "role": role, "u": u, "median": np.median(s, axis=1),
                                         "actual": y, "season": sub["season"].to_numpy()}))
    P = pl.concat(out)
    return P.filter(~((pl.col("market") == "rush_yds") & (pl.col("role") == "QB1") & (pl.col("median") < 10)))


def pit_table(P: pl.DataFrame) -> pl.DataFrame:
    return P.with_columns(dec=(pl.col("u") * 10).floor().clip(0, 9)).group_by("market").agg(
        n=pl.len(),
        **{f"d{k}": (pl.col("dec") == k).mean() for k in range(10)},
        over_median=(pl.col("actual") > pl.col("median")).mean(),
    ).with_columns(miscal=pl.sum_horizontal([(pl.col(f"d{k}") - 0.1) ** 2 for k in range(10)]) ** 0.5).sort("market")


GROUPS = {"rec": ("receptions", "rec_yds"), "rush": ("rush_yds",), "pass": ("pass_yds",)}
GRIDS = {"rec": {"r_tgt": [4.0, 6.0, 9.0, 14.0], "sd_rec": [0.1, 0.2, 0.3]},
         "rush": {"r_car": [3.0, 5.0, 8.0, 12.0], "sd_rush": [0.1, 0.2, 0.3]},
         "pass": {"r_att": [10.0, 20.0, 40.0], "sd_pass": [0.0, 0.08, 0.16]}}


def calibrate(x: pl.DataFrame, seasons=TUNE, n_rows: int = 6000, size: int = 1000, seed: int = 5) -> Dispersion:
    """Pick count dispersion and efficiency spread per market group, then the
    median shift, to make 2018-2021 forecasts calibrated."""
    d = x.filter(pl.col("season").is_between(*seasons) & pl.col("role").is_not_null())
    d = d.sample(min(n_rows, d.height), seed=seed)
    best = Dispersion(shift={})
    for g, grid in GRIDS.items():
        keys = list(grid)
        scored = []
        for vals in itertools.product(*grid.values()):
            trial = Dispersion(**{**best.as_dict(), **dict(zip(keys, vals)), "shift": {}})
            P = _pit_rows(d, trial, size, seed).filter(pl.col("market").is_in(list(GROUPS[g])))
            # judge the shape after centering, so a median miss does not pick the spread
            P = P.with_columns(u=pl.col("u") - pl.col("u").median().over("market") + 0.5)
            scored.append((float(pit_table(P)["miscal"].mean()), vals))
        vals = min(scored)[1]
        best = Dispersion(**{**best.as_dict(), **dict(zip(keys, vals))})
        print(f"  {g}: {dict(zip(keys, vals))}")
    # median shift: receptions additive, yards multiplicative
    P = _pit_rows(d, best, size, seed)
    shift = {}
    for m in ("receptions", "rec_yds", "rush_yds", "pass_yds"):
        q = P.filter(pl.col("market") == m)
        if m == "receptions":
            shift[m] = float((q["actual"] - q["median"]).median())
        else:
            q = q.filter(pl.col("median") > 1)
            shift[m] = float((q["actual"] / q["median"]).median())
    best.shift = shift
    print("  shift:", {k: round(v, 3) for k, v in shift.items()})
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    if args.no_tune and PARAMS_PATH.exists():
        p = Params(**json.loads(PARAMS_PATH.read_text())["params"])
    else:
        print("tuning on 2018-2021 ...")
        p = tune()
    x = baselines(frames(p))
    L = long_form(x)
    print("tune seasons:\n", score(L, TUNE))
    print("holdout 2022-2025:\n", score(L, HOLD))
    print("calibrating distributions on 2018-2021 ...")
    disp = calibrate(x)
    hold = x.filter(pl.col("season").is_between(*HOLD) & pl.col("role").is_not_null())
    P = _pit_rows(hold, disp, 2000, 11)
    print("PIT deciles, holdout 2022-2025:\n", pit_table(P))
    REPORTS.mkdir(exist_ok=True)
    PARAMS_PATH.write_text(json.dumps({"params": p.asdict(), "dispersion": disp.as_dict(),
                                       "holdout": score(L, HOLD).to_dicts(), "tune": score(L, TUNE).to_dicts(),
                                       "pit": pit_table(P).to_dicts()}, indent=1))
    print("saved", PARAMS_PATH)


def load() -> tuple[Params, Dispersion]:
    saved = json.loads(PARAMS_PATH.read_text())
    return Params(**saved["params"]), Dispersion(**saved["dispersion"])


if __name__ == "__main__":
    main()
