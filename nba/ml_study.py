"""Should the NBA moneyline come from the spread and the model, as the NFL's does? (user, 2026-11-19: "I would like
to see how the model has a say on NBA moneylines as how the nfl has a say")

The NFL prices a moneyline from a win curve on the expected margin (model/probs.WinCurve), with the model moving
that margin by the measured k. The NBA prices it from the moneyline market itself (Shin's no-vig split). This study
asks, on past seasons with real moneyline prices (ESPN's book of record, 2023-24 to 2025-26), walk-forward:

1. the curve alone: P(home wins) from the closing spread (fitted on the seasons before), against the moneyline
   market's own Shin chance: log loss, and how the two disagree;
2. the model's say: the curve on the closing spread moved k of the way to the rebuilt model's margin
   (nba/backtest.py walk-forward margins), k = 0.1, 0.2, 0.3, and a joint fit of the result on the market's chance,
   the curve's and the model's gap;
3. bets: the side with positive EV at the closing (and the opening) moneyline price under each source, by EV bar,
   favorites and underdogs apart, per season.

    python -m nba.ml_study      # on Actions (research.yml job nba-ml): ESPN is not reachable from a cloud session

Prints tallies only (the repository is public); writes reports/nba_ml_study.json.
"""
import json
import math

import numpy as np
import polars as pl

from config import REPORTS
from model.probs import WinCurve, ml_no_vig
from nba import angles
from nba import backtest as BT
from nba.data import load, refresh

OUT = REPORTS / "nba_ml_study.json"
TEST = [2024, 2025, 2026]            # seasons with honest moneyline prices at ESPN
KS = [0.0, 0.1, 0.2, 0.3]
dec = lambda a: None if a is None or (isinstance(a, float) and math.isnan(a)) else (1 + a / 100 if a > 0 else 1 + 100 / -a)


def frame() -> pl.DataFrame:
    refresh()
    o = angles.fetch(range(2022, 2027))
    sched = load("schedule").select(pl.col("game_id").cast(pl.Int64), "season",
                                    (pl.col("home_score").cast(pl.Float64) - pl.col("away_score").cast(pl.Float64)).alias("result"))
    close = BT.closes()                                    # hoopR's consensus closing spread, as the home margin
    model = BT.run(range(2021, 2027)).select("game_id", "model_margin")
    g = (o.select("game_id", "close_ml_home", "close_ml_away", "open_ml_home", "open_ml_away", "open_spread_home")
          .join(sched, on="game_id").join(close, on="game_id", how="left").join(model, on="game_id", how="left"))
    return g.filter(pl.col("result").is_not_null() & (pl.col("result") != 0) & pl.col("close_spread").is_not_null())


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(X.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-X @ beta))
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        beta += np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)
    return beta, np.sqrt(np.diag(cov))


def _roi(r: list[float]) -> dict:
    a = np.array(r)
    return {"n": len(a), "roi": round(float(a.mean()), 4) if len(a) else None,
            "se": round(float(a.std(ddof=1) / math.sqrt(len(a))), 4) if len(a) > 1 else None}


def study() -> dict:
    g = frame()
    out = {"games": {}, "logloss": {}, "joint": {}, "bets": {}, "disagree": {}}
    allrows = []
    for s in TEST:
        tr = load("schedule").select(pl.col("game_id").cast(pl.Int64), "season",
                                     (pl.col("home_score").cast(pl.Float64) - pl.col("away_score").cast(pl.Float64)).alias("result"))
        tr = tr.join(BT.closes(), on="game_id").filter((pl.col("season") < s) & (pl.col("season") >= s - 4) & pl.col("result").is_not_null())
        wc = WinCurve.fit(tr["close_spread"].to_numpy(), tr["result"].to_numpy())
        te = g.filter((pl.col("season") == s) & pl.col("close_ml_home").is_not_null() & pl.col("close_ml_away").is_not_null())
        for r in te.iter_rows(named=True):
            dh, da = dec(r["close_ml_home"]), dec(r["close_ml_away"])
            if not dh or not da or dh <= 1 or da <= 1:
                continue
            row = {"season": s, "y": float(r["result"] > 0), "p_ml": ml_no_vig(dh, da), "dh": dh, "da": da,
                   "close": r["close_spread"], "model": r["model_margin"]}
            for k in KS:
                mu = r["close_spread"] + (k * (r["model_margin"] - r["close_spread"]) if r["model_margin"] is not None else 0.0)
                row[f"p_c{k}"] = wc.home(mu)
            oh, oa = dec(r["open_ml_home"]), dec(r["open_ml_away"])
            if oh and oa and oh > 1 and oa > 1 and r["open_spread_home"] is not None:
                row.update(oh=oh, oa=oa, p_ml_open=ml_no_vig(oh, oa))
                for k in KS:
                    mu = -r["open_spread_home"] + (k * (r["model_margin"] + r["open_spread_home"]) if r["model_margin"] is not None else 0.0)
                    row[f"po_c{k}"] = wc.home(mu)
            allrows.append(row)
    D = pl.DataFrame(allrows, infer_schema_length=None)
    y = D["y"].to_numpy()
    out["games"] = {str(s): int((D["season"] == s).sum()) for s in TEST} | {"all": D.height}
    # 1-2. log loss by source, per season and pooled
    for s in TEST + ["all"]:
        d = D if s == "all" else D.filter(pl.col("season") == s)
        yy = d["y"].to_numpy()
        out["logloss"][str(s)] = {"ml_close": round(_ll(d["p_ml"].to_numpy(), yy), 5),
                                  **{f"curve_k{k}": round(_ll(d[f"p_c{k}"].to_numpy(), yy), 5) for k in KS}}
    m = D.filter(pl.col("model").is_not_null())
    X = np.column_stack([np.ones(m.height), _logit(m["p_ml"].to_numpy()), _logit(m["p_c0.0"].to_numpy()) - _logit(m["p_ml"].to_numpy()),
                         (m["model"] - m["close"]).to_numpy() / 10])
    b, se = _fit(X, m["y"].to_numpy())
    out["joint"] = {"n": m.height, "terms": ["intercept", "market logit", "curve minus market (logit)", "model minus close (per 10 pts)"],
                    "beta": [round(float(v), 3) for v in b], "se": [round(float(v), 3) for v in se]}
    # where the curve and the moneyline disagree by 2+ points of win chance, on the underdog
    dog_home = D["p_ml"] < 0.5
    p_dog_ml = np.where(dog_home, D["p_ml"], 1 - D["p_ml"])
    p_dog_c = np.where(dog_home, D["p_c0.0"], 1 - D["p_c0.0"])
    dog_won = np.where(dog_home, D["y"], 1 - D["y"])
    for lab, sel in (("curve 2+ pts above the moneyline", p_dog_c - p_dog_ml >= 0.02), ("within 2 pts", np.abs(p_dog_c - p_dog_ml) < 0.02),
                     ("curve 2+ pts below", p_dog_c - p_dog_ml <= -0.02)):
        out["disagree"][lab] = {"n": int(sel.sum()), "dog_won": round(float(dog_won[sel].mean()), 4) if sel.sum() else None,
                                "ml_said": round(float(p_dog_ml[sel].mean()), 4) if sel.sum() else None,
                                "curve_said": round(float(p_dog_c[sel].mean()), 4) if sel.sum() else None}
    # 3. bets at the close (and the open) by source and EV bar
    for when, pre, dh_, da_ in (("close", "p_c", "dh", "da"), ("open", "po_c", "oh", "oa")):
        for k in KS:
            col = f"{pre}{k}"
            if col not in D.columns:
                continue
            for bar in (0.0, 0.02, 0.05):
                res = {"all": [], "fav": [], "dog": []}
                by_season = {str(s): [] for s in TEST}
                for r in D.iter_rows(named=True):
                    if r.get(col) is None or r.get(dh_) is None:
                        continue
                    ph = r[col]
                    for p, d_, won in ((ph, r[dh_], r["y"] == 1), (1 - ph, r[da_], r["y"] == 0)):
                        ev = p * d_ - 1
                        if ev >= bar and ev > 0:
                            ret = d_ - 1 if won else -1.0
                            res["all"].append(ret)
                            res["fav" if d_ < 2 else "dog"].append(ret)
                            by_season[str(r["season"])].append(ret)
                out["bets"][f"{when} k{k} ev{int(bar * 100)}%"] = {kk: _roi(v) for kk, v in res.items()} | {"by_season": {kk: _roi(v) for kk, v in by_season.items()}}
    return out


def main() -> None:
    r = study()
    OUT.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print("games", r["games"])
    for s, v in r["logloss"].items():
        print("log loss", s, v)
    print("joint fit", r["joint"])
    for k, v in r["disagree"].items():
        print("underdogs,", k, v)
    for k, v in r["bets"].items():
        print("bets", k, {kk: vv for kk, vv in v.items() if kk != "by_season"}, "| by season", v["by_season"])


if __name__ == "__main__":
    main()
