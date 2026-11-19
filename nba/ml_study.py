"""NBA moneylines: which way of pricing them, with or without the model, predicts best and would have made money?
(user, 2026-11-19: "I would like to see how the model has a say on NBA moneylines as how the nfl has a say", then
"You don't have to do it the same way as the NFL, test different measures to see the best results")

Part A, history: every regular-season game 2021-22 to 2025-26 that ESPN's book of record priced (open and close
moneyline and spread; nba/angles.py), with the rebuilt game model's walk-forward margin (nba/backtest.py) and each
team's rest. Each test season is priced only from what the seasons before it taught (walk-forward). Measures of the
home win chance:

  market         the moneyline's own chance, margin off by Shin's method (the board's rule today)
  market_prop    the same, margin off in proportion (shows why Shin)
  recal          the market's chance recalibrated, a + b * logit (favorite-longshot bias, learned)
  recal2         the same with a tail term, logit * |logit|
  curve          the NFL way: a win curve on the spread's expected margin
  mkt+curve      the market moved by the spread curve's disagreement, weight learned
  model_only     a win curve on the model's margin alone
  blend w        the market and model_only averaged in log-odds, the model's weight w fixed (0.1, 0.2)
  mkt+model      the market moved by the model's gap to the spread (model - spread), weight learned
  mkt+sit        the market plus rest: rest difference, either team on the second night, home underdog
  mkt+model+sit  both
At the open the model is used WITHOUT its lineup part (team ratings and rest only): in the backtest the lineup part
knows who actually played, which the opening line could not, so it leaks. The leaky versions are reported too,
labeled, to show how much of any open-price "edge" was that leak.

Each measure is judged on the same games: log loss against the market's (paired, with its standard error), and
bets at that moment's price on the side with EV >= 0 / 2% / 4%: return, and for bets at the open, EV at the close
(the closing moneyline's Shin chance times the price taken, the board's usual yardstick).

Part B, this season at the board's real lock: every NBA game since 2026-10-20 at its 10:30am ET lock (the ledger's
model margin and Pinnacle's fair numbers then; the archive's DraftKings / FanDuel / Pinnacle moneylines, Actions
cache), each measure with the coefficients history learned, priced against the best DraftKings/FanDuel moneyline
and judged at Pinnacle's close and on results. Also the model as a FILTER: today's +EV line-shopping moneylines
split by whether the model agrees.

Pre-declared pass rule (printed as "passed"): a measure is worth adopting only if its log loss beats the market's by
more than two standard errors out of sample, or its 2%+ bets (at least 50 of them) return (at the open: beat the
close) by two standard errors pooled AND are positive in every test season. With ~40 bet sets, one or two will look
good by chance. (The 50-bet floor was added after the first run, where a 2-bet set went 2-0 and "passed".)

    python -m nba.ml_study      # on Actions (research.yml job nba-ml); prints tallies only, writes reports/nba_ml_study.json
"""
import json
import math
import shutil
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from config import DATA, LEDGER, REPORTS
from model.probs import WinCurve, ml_no_vig
from nba import angles
from nba import backtest as BT
from nba.data import load, refresh

OUT = REPORTS / "nba_ml_study.json"
STORE = DATA / "nba_mlstudy"          # Actions cache of its own: ESPN's odds and the model's walk-forward margins
BARS = (0.0, 0.02, 0.04)
WS = (0.1, 0.2)


def dec(a) -> float | None:
    if a is None or (isinstance(a, float) and math.isnan(a)) or a == 0:
        return None
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def _ok(dh, da) -> bool:
    return bool(dh and da and dh > 1 and da > 1 and 1.0 <= 1 / dh + 1 / da <= 1.15)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _fit(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-X @ beta))
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - lam * beta)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    cov = np.linalg.pinv((X * (p * (1 - p))[:, None]).T @ X)
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None))


def rest_table() -> pl.DataFrame:
    """Days since each team's previous game (capped at 7; a team's first game of a season counts 7)."""
    s = (load("schedule").filter(pl.col("season_type").is_in([2, 3, 5]))
         .select(pl.col("game_id").cast(pl.Int64), "season", pl.col("game_date").cast(pl.Date).alias("date"),
                 pl.col("home_abbreviation").alias("home"), pl.col("away_abbreviation").alias("away")))
    long = pl.concat([s.select("game_id", "season", "date", pl.col("home").alias("team"), pl.lit(True).alias("is_home")),
                      s.select("game_id", "season", "date", pl.col("away").alias("team"), pl.lit(False).alias("is_home"))]).sort("team", "date")
    long = long.with_columns(rest=(pl.col("date") - pl.col("date").shift(1).over(["team", "season"])).dt.total_days().fill_null(7).clip(0, 7))
    h = long.filter(pl.col("is_home")).select("game_id", pl.col("rest").alias("rest_h"))
    a = long.filter(~pl.col("is_home")).select("game_id", pl.col("rest").alias("rest_a"))
    return h.join(a, on="game_id").unique("game_id")


# ---------------------------------------------------------------- the measures

def sit(r: dict) -> list[float]:
    rd = max(-3, min(3, (r["rest_h"] or 7) - (r["rest_a"] or 7))) / 3
    return [rd, float(r["rest_h"] == 1), float(r["rest_a"] == 1)]


def stack_x(name: str, r: dict, m: dict, C: dict) -> list[float] | None:
    """The features of a learned measure, or None when the row lacks them."""
    L = logit(m["p"])
    hd = float(m["p"] < 0.5)
    model = {"mkt+model": r["model"], "mkt+model+sit": r["model"], "mkt+model(leaky)": r["model"],
             "mkt+model0": r["model0"], "mkt+model0+sit": r["model0"]}.get(name)
    if name == "recal":
        return [1.0, L]
    if name == "recal2":
        return [1.0, L, L * abs(L) / 4]
    if name == "mkt+curve":
        return None if m["mu"] is None else [1.0, L, logit(C["curve"].home(m["mu"])) - L]
    if name == "mkt+sit":
        return [1.0, L, *sit(r), hd]
    if name in ("mkt+model", "mkt+model0", "mkt+model(leaky)"):
        return None if model is None or m["mu"] is None else [1.0, L, (model - m["mu"]) / 10]
    if name in ("mkt+model+sit", "mkt+model0+sit"):
        return None if model is None or m["mu"] is None else [1.0, L, (model - m["mu"]) / 10, *sit(r), hd]
    raise KeyError(name)


def fixed_p(name: str, r: dict, m: dict, C: dict) -> float | None:
    if name == "market":
        return m["p"]
    if name == "market_prop":
        return m["pprop"]
    if name == "curve":
        return None if m["mu"] is None else C["curve"].home(m["mu"])
    if name in ("model_only", "model0_only", "model_only(leaky)"):
        mm, cv = (r["model0"], C["m0curve"]) if name == "model0_only" else (r["model"], C["mcurve"])
        return None if mm is None else cv.home(mm)
    if name.startswith("blend"):          # "blend 0.1", "blend0 0.2", "blend 0.2(leaky)"
        w = float(name.split()[1].split("(")[0])
        mo = fixed_p("model0_only" if name.startswith("blend0") else "model_only", r, m, C)
        return None if mo is None else sig((1 - w) * logit(m["p"]) + w * logit(mo))
    raise KeyError(name)


STACKS = ("recal", "recal2", "mkt+curve", "mkt+sit", "mkt+model", "mkt+model+sit", "mkt+model0", "mkt+model0+sit", "mkt+model(leaky)")
MEASURES = {
    "close": ["market", "market_prop", "recal", "recal2", "curve", "mkt+curve", "model_only", "blend 0.1", "blend 0.2",
              "mkt+model", "mkt+sit", "mkt+model+sit"],
    "open": ["market", "market_prop", "recal", "recal2", "curve", "mkt+curve", "model0_only", "blend0 0.1", "blend0 0.2",
             "mkt+model0", "mkt+sit", "mkt+model0+sit", "model_only(leaky)", "blend 0.2(leaky)", "mkt+model(leaky)"],
}


# ---------------------------------------------------------------- data

def history() -> tuple[list[dict], pl.DataFrame]:
    refresh()
    STORE.mkdir(parents=True, exist_ok=True)
    keep = STORE / "espn_odds.parquet"
    if keep.exists() and not angles.OUT.exists():
        shutil.copy(keep, angles.OUT)
    o = angles.fetch(range(2022, 2027))
    shutil.copy(angles.OUT, keep)
    mp = STORE / "model.parquet"
    if mp.exists():
        M = pl.read_parquet(mp)
    else:
        M = BT.run(range(2021, 2027)).select("game_id", "season", "model_margin", "lineup_adj", "close_spread", "result")
        M.write_parquet(mp)
    M = M.filter(pl.col("result").is_not_null() & (pl.col("result") != 0) & pl.col("close_spread").is_not_null()
                 & pl.col("model_margin").is_not_null() & pl.col("lineup_adj").is_not_null())
    rest = rest_table()
    g = o.join(M, on="game_id", how="inner").join(rest, on="game_id", how="left")
    rows = []
    for r in g.iter_rows(named=True):
        row = {"game_id": r["game_id"], "season": r["season"], "y": float(r["result"] > 0),
               "model": r["model_margin"], "model0": None if r["model_margin"] is None else r["model_margin"] - (r["lineup_adj"] or 0.0),
               "rest_h": r["rest_h"], "rest_a": r["rest_a"]}
        for mom in ("close", "open"):
            dh, da = dec(r.get(f"{mom}_ml_home")), dec(r.get(f"{mom}_ml_away"))
            if not _ok(dh, da):
                continue
            sp = r.get(f"{mom}_spread_home")
            mu = -sp if sp is not None and not (isinstance(sp, float) and math.isnan(sp)) else (r["close_spread"] if mom == "close" else None)
            row[mom] = {"dh": dh, "da": da, "p": ml_no_vig(dh, da), "pprop": (1 / dh) / (1 / dh + 1 / da), "mu": mu}
        rows.append(row)
    return rows, M


# ---------------------------------------------------------------- scoring

def _ll(p: float, y: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _mean_se(a) -> tuple[float | None, float | None]:
    a = np.asarray(a, dtype=float)
    if not len(a):
        return None, None
    return float(a.mean()), (float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else None)


def _bets(recs: list[dict], bar: float, clv: bool) -> dict:
    """recs: p (home), y, dh, da, season, pc (the close's Shin home chance, for EV at the close)."""
    out = {"all": [], "fav": [], "dog": [], "clv": [], "by_season": {}}
    for r in recs:
        evs = [(r["p"] * r["dh"] - 1, "home", r["dh"]), ((1 - r["p"]) * r["da"] - 1, "away", r["da"])]
        ev, side, d = max(evs)
        if ev <= 0 or ev < bar:
            continue
        won = (r["y"] == 1) == (side == "home")
        ret = d - 1 if won else -1.0
        out["all"].append(ret)
        out["fav" if d < 2 else "dog"].append(ret)
        out["by_season"].setdefault(str(r["season"]), []).append(ret)
        if clv and r.get("pc") is not None:
            out["clv"].append((r["pc"] if side == "home" else 1 - r["pc"]) * d - 1)
    res = {}
    for k in ("all", "fav", "dog", "clv"):
        m, se = _mean_se(out[k])
        res[k] = {"n": len(out[k]), "mean": None if m is None else round(m, 4), "se": None if se is None else round(se, 4)}
    res["by_season"] = {s: round(float(np.mean(v)), 4) for s, v in out["by_season"].items()}
    res["by_season_n"] = {s: len(v) for s, v in out["by_season"].items()}
    return res


def part_a(rows: list[dict], M: pl.DataFrame) -> tuple[dict, dict]:
    report, pooled_fits = {}, {}
    for mom in ("close", "open"):
        need = [r for r in rows if mom in r and r[mom]["mu"] is not None and r["model"] is not None and r["rest_h"] is not None]
        seasons = sorted({r["season"] for r in need})
        tests = [s for s in seasons if sum(1 for r in need if r["season"] < s) >= 800]
        recs = {m: [] for m in MEASURES[mom]}
        for s in tests:
            hist = M.filter((pl.col("season") < s) & (pl.col("season") >= s - 4))
            C = {"curve": WinCurve.fit(hist["close_spread"].to_numpy(), hist["result"].to_numpy()),
                 "mcurve": WinCurve.fit(hist["model_margin"].to_numpy(), hist["result"].to_numpy()),
                 "m0curve": WinCurve.fit((hist["model_margin"] - hist["lineup_adj"]).to_numpy(), hist["result"].to_numpy())}
            tr = [r for r in need if r["season"] < s]
            te = [r for r in need if r["season"] == s]
            betas = {}
            for name in STACKS:
                if name not in MEASURES[mom]:
                    continue
                X = np.array([stack_x(name, r, r[mom], C) for r in tr])
                betas[name] = _fit(X, np.array([r["y"] for r in tr]))[0]
            for r in te:
                pc = r["close"]["p"] if "close" in r else None
                for name in MEASURES[mom]:
                    p = float(sig(np.dot(betas[name], stack_x(name, r, r[mom], C)))) if name in betas else fixed_p(name, r, r[mom], C)
                    recs[name].append({"p": p, "y": r["y"], "dh": r[mom]["dh"], "da": r[mom]["da"], "season": s, "pc": pc,
                                       "pm": r[mom]["p"]})
        # pooled fits on every season with this moment, for part B and to show the weights
        C_all = {"curve": WinCurve.fit(M["close_spread"].to_numpy(), M["result"].to_numpy())}
        pooled_fits[mom] = {"curve": C_all["curve"]}
        for name in STACKS:
            if name not in MEASURES[mom]:
                continue
            X = np.array([stack_x(name, r, r[mom], C_all) for r in need])
            b, se = _fit(X, np.array([r["y"] for r in need]))
            pooled_fits[mom][name] = b
            report.setdefault("weights", {}).setdefault(mom, {})[name] = {"beta": [round(float(v), 3) for v in b],
                                                                           "se": [round(float(v), 3) for v in se]}
        out = {"games": {str(s): sum(1 for r in need if r["season"] == s) for s in seasons}, "tests": tests, "measures": {}}
        for name, rs in recs.items():
            if not rs:
                continue
            ll = [_ll(x["p"], x["y"]) for x in rs]
            d = [_ll(x["p"], x["y"]) - _ll(x["pm"], x["y"]) for x in rs]
            dm, dse = _mean_se(d)
            entry = {"n": len(rs), "ll": round(float(np.mean(ll)), 5), "dll": round(dm, 5), "dll_se": None if dse is None else round(dse, 5),
                     "by_season": {str(s): round(float(np.mean([_ll(x["p"], x["y"]) for x in rs if x["season"] == s])), 5) for s in tests},
                     "bets": {f"{int(b * 100)}%": _bets(rs, b, mom == "open") for b in BARS}}
            out["measures"][name] = entry
        report[mom] = out
    return report, pooled_fits


# ---------------------------------------------------------------- part B: this season at the 10:30am lock

def part_b(F: dict) -> dict:
    from nba.odds import HIST, fixture_index, lines_at
    lock = {}
    for line in (LEDGER / "nba_picks.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(line) if line.strip() else {}
        if p.get("market") == "spread" and p.get("kind") == "pick" and p.get("game_id") not in lock and "fair_margin" in p:
            lock[p["game_id"]] = p
    s = load("schedule").filter((pl.col("season") == 2027) & pl.col("status_type_completed").fill_null(False))
    finals = {int(r["game_id"]): float(r["home_score"]) - float(r["away_score"]) for r in s.iter_rows(named=True)
              if r["home_score"] is not None and r["away_score"] is not None}
    rest = {r["game_id"]: r for r in rest_table().iter_rows(named=True)}
    rows = []
    for g in fixture_index().iter_rows(named=True):
        lk, res = lock.get(g["game_id"]), finals.get(g["game_id"])
        p = HIST / f"{g['fixture_id']}.parquet"
        if not lk or res is None or res == 0 or not p.exists():
            continue
        h = pl.read_parquet(p).filter(pl.col("ts") < g["tip"])
        t = datetime.fromisoformat(lk["locked_at"])
        at = {b: lines_at(h.filter(pl.col("book") == b), t) for b in ("pinnacle", "draftkings", "fanduel")}
        close = lines_at(h.filter(pl.col("book") == "pinnacle"), g["tip"] - timedelta(minutes=5))
        if "ml_home" not in at["pinnacle"] or "ml_home" not in close:
            continue
        us = [at[b] for b in ("draftkings", "fanduel") if "ml_home" in at[b]]
        if not us:
            continue
        rr = rest.get(g["game_id"], {})
        rows.append({"season": 2027, "y": float(res > 0), "v": lk.get("model_v", 1), "model": lk.get("model_margin"),
                     "p": ml_no_vig(at["pinnacle"]["ml_home"], at["pinnacle"]["ml_away"]), "mu": lk.get("fair_margin"),
                     "dh": max(u["ml_home"] for u in us), "da": max(u["ml_away"] for u in us),
                     "pc": ml_no_vig(close["ml_home"], close["ml_away"]), "rest_h": rr.get("rest_h"), "rest_a": rr.get("rest_a")})
    C = F["close"]
    names = ["market", "recal", "recal2", "mkt+curve", "mkt+sit", "mkt+model", "mkt+model+sit"]
    out = {"games": len(rows), "rebuilt_model_games": sum(1 for r in rows if r["v"] == 2), "measures": {}, "filter": {}}
    for name in names:
        recs = []
        for r in rows:
            if name.startswith("mkt+model") and (r["v"] != 2 or r["model"] is None or r["mu"] is None):
                continue
            m = {"p": r["p"], "mu": r["mu"]}
            p = r["p"] if name == "market" else float(sig(np.dot(C[name], stack_x(name, {**r, "model0": None}, m, C))))
            recs.append({**r, "p": p, "pm": r["p"]})
        ll = [_ll(x["p"], x["y"]) for x in recs]
        d = [_ll(x["p"], x["y"]) - _ll(x["pm"], x["y"]) for x in recs]
        dm, dse = _mean_se(d)
        out["measures"][name] = {"n": len(recs), "ll": round(float(np.mean(ll)), 5) if ll else None,
                                 "dll": None if dm is None else round(dm, 5), "dll_se": None if dse is None else round(dse, 5),
                                 "bets": {f"{int(b * 100)}%": _bets(recs, b, True) for b in BARS}}
    # the model as a filter on today's rule (market, any +EV), rebuilt-model games
    for agree in (True, False):
        recs = []
        for r in rows:
            if r["v"] != 2 or r["model"] is None or r["mu"] is None:
                continue
            evh, eva = r["p"] * r["dh"] - 1, (1 - r["p"]) * r["da"] - 1
            side_home = evh >= eva
            model_home = r["model"] > r["mu"]
            if (side_home == model_home) == agree:
                recs.append({**r, "p": r["p"]})
        out["filter"]["model agrees" if agree else "model disagrees"] = _bets(recs, 0.0, True)
    return out


def passed(A: dict) -> list[str]:
    ok = []
    for mom in ("close", "open"):
        for name, e in A[mom]["measures"].items():
            if name == "market" or "leaky" in name:
                continue
            if e["dll_se"] and e["dll"] < -2 * e["dll_se"]:
                ok.append(f"{mom} {name}: log loss {e['dll']:+.5f} +- {e['dll_se']:.5f}")
            b = e["bets"]["2%"]
            key = "clv" if mom == "open" else "all"
            m, se = b[key]["mean"], b[key]["se"]
            if m is not None and se and b[key]["n"] >= 50 and m > 2 * se and b["by_season"] and all(v > 0 for v in b["by_season"].values()):
                ok.append(f"{mom} {name}: 2%+ bets {'EV at the close' if mom == 'open' else 'return'} {m:+.3f} +- {se:.3f}")
    return ok


def main() -> None:
    rows, M = history()
    A, F = part_a(rows, M)
    try:
        B = part_b(F)
    except Exception as e:          # the archive lives in the build's cache; history stands on its own
        B = {"error": repr(e)}
    rep = {"history": A, "this_season": B, "passed": passed(A)}
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    pm = lambda m, s: "     –" if m is None else f"{100 * m:+5.1f}" + ("" if s is None else f"±{100 * s:.1f}")
    for mom in ("close", "open"):
        a = A[mom]
        print(f"\n=== at the {mom} | games {a['games']} | test seasons {a['tests']}")
        for name, e in a["measures"].items():
            b = e["bets"]["2%"]
            extra = f" EV@close {pm(b['clv']['mean'], b['clv']['se'])}%" if mom == "open" else ""
            print(f"{name:20} LL {e['ll']:.5f} d {e['dll']:+.5f}±{e['dll_se'] or 0:.5f} | 2%+ bets n {b['all']['n']:4} ROI {pm(b['all']['mean'], b['all']['se'])}%{extra}"
                  f" fav {pm(b['fav']['mean'], None)}%({b['fav']['n']}) dog {pm(b['dog']['mean'], None)}%({b['dog']['n']}) | {b['by_season']}")
    print("\nweights", json.dumps(A.get("weights")))
    print("\n=== this season at the 10:30am lock", {k: v for k, v in B.items() if k not in ("measures", "filter")})
    for name, e in B.get("measures", {}).items():
        b0, b2 = e["bets"]["0%"], e["bets"]["2%"]
        print(f"{name:16} n {e['n']} LL {e['ll']} d {e['dll']}±{e['dll_se']} | +EV bets n {b0['all']['n']} ROI {pm(b0['all']['mean'], b0['all']['se'])}%"
              f" EV@close {pm(b0['clv']['mean'], b0['clv']['se'])}% | 2%+ n {b2['all']['n']} ROI {pm(b2['all']['mean'], b2['all']['se'])}% EV@close {pm(b2['clv']['mean'], b2['clv']['se'])}%")
    for k, b in B.get("filter", {}).items():
        print(f"filter, {k}: n {b['all']['n']} ROI {pm(b['all']['mean'], b['all']['se'])}% EV@close {pm(b['clv']['mean'], b['clv']['se'])}%")
    print("\npassed:", rep["passed"] or "none")


if __name__ == "__main__":
    main()
