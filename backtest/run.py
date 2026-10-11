"""Walk-forward backtest: tune on 2018-2021, then judge 2022+ untouched.

    python -m backtest.run --tune     # search parameters (2018-2021 only), save them
    python -m backtest.run            # full walk-forward with saved parameters

Every season S is predicted by a model fit only on seasons before S, using
ratings snapshots taken before each game. Bets are graded at the CLOSING line
and price from nflverse, the hardest available bar.
"""
import argparse
import json
from dataclasses import asdict, replace

import numpy as np
import polars as pl

from config import BACKTEST_FIRST, REPORTS, TUNE_LAST
from model import features, project
from model.blend import Blend
from model.probs import KeyDist, MARGIN_RANGE, TOTAL_RANGE, WinCurve, american_to_decimal, ev, no_vig
from model.ratings import Params, load_inputs

PARAMS_FILE = REPORTS / "params.json"
TRAIN_FIRST = 2017  # 2016 snapshots have no history behind them


def load_params() -> tuple[Params, float]:
    if PARAMS_FILE.exists():
        d = json.loads(PARAMS_FILE.read_text())
        return Params(**d["ratings"]), d["points_alpha"]
    return Params(), 50.0


def walk_forward(inp, games, p: Params, points_alpha: float, seasons, as_of_publish: bool = False) -> pl.DataFrame:
    """Project each season with a model fit on earlier seasons only.

    as_of_publish=True predicts with only Tuesday-of-game-week facts (last
    week's QB, no game-day weather); training rows still use what happened.
    """
    rows, _, _ = features.build(inp, games, p)
    test_rows = features.build(inp, games, p, as_of_publish=True)[0] if as_of_publish else rows
    out = []
    for s in seasons:
        model = project.fit(rows.filter(pl.col("season").is_between(TRAIN_FIRST, s - 1)), points_alpha)
        out.append(project.game_projections(test_rows.filter(pl.col("season") == s), model))
    return pl.concat(out).join(games, on="game_id")


def _score(proj: pl.DataFrame) -> float:
    done = proj.filter(pl.col("result").is_not_null())
    return float((done["model_margin"] - done["result"]).abs().mean()
                 + (done["model_total"] - done["total"]).abs().mean())


def tune(inp, games) -> None:
    """Coordinate search on 2018..TUNE_LAST. Holdout seasons are never scored here."""
    seasons = range(BACKTEST_FIRST, TUNE_LAST + 1)
    grid = {
        "half_life": [4, 6, 8, 12, 16, 24],
        "gap": [0, 4, 8, 12, 20],
        "alpha_play": [100, 200, 400, 800, 1600],
        "alpha_game": [1, 2, 4, 8, 16],
        "qb_half_life": [8, 16, 24, 40, 64, 100, 160, 250],
        "qb_gap": [0, 4, 10, 20],
        "alpha_qb": [50, 100, 250, 500, 1000],
    }
    p, alpha = load_params()  # resume from the last saved search
    best = _score(walk_forward(inp, games, p, alpha, seasons))
    print(f"start {best:.4f}")
    for _ in range(2):  # two passes is enough for a grid this coarse
        for name, values in grid.items():
            for v in values:
                cand = replace(p, **{name: float(v)})
                sc = _score(walk_forward(inp, games, cand, alpha, seasons))
                if sc < best - 1e-4:
                    best, p = sc, cand
            print(f"  {name:<13} -> {getattr(p, name):>7}   score {best:.4f}")
        for a in [5, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
            sc = _score(walk_forward(inp, games, p, float(a), seasons))
            if sc < best - 1e-4:
                best, alpha = sc, float(a)
        print(f"  points_alpha  -> {alpha:>7}   score {best:.4f}")
    PARAMS_FILE.write_text(json.dumps({"ratings": asdict(p), "points_alpha": alpha,
                                       "tuned_on": f"{BACKTEST_FIRST}-{TUNE_LAST}", "score": best}, indent=2))
    print(f"saved {PARAMS_FILE}")


# ---------------------------------------------------------------- grading

def _grade(outcome: float, line: float, side: int) -> float:
    """+1 win, 0 push, -1 loss for a bet that `outcome` beats `line` (side=+1) or not (side=-1)."""
    d = (outcome - line) * side
    return float(np.sign(d))


def _payout(res: float, odds: float, risk: float = 1.0) -> float:
    return risk * (american_to_decimal(odds) - 1) if res > 0 else (0.0 if res == 0 else -risk)


def price_bets(proj: pl.DataFrame, blend: Blend, mdist: KeyDist, tdist: KeyDist, wins: WinCurve) -> pl.DataFrame:
    """Every available bet on every game, with model probability and EV at the closing price."""
    bets = []
    for g in proj.iter_rows(named=True):
        mu_m = blend.margin(g["model_margin"], g["spread_line"])
        mu_t = blend.total(g["model_total"], g["total_line"])
        hc, hp, ac = mdist.side_probs(mu_m, g["spread_line"])
        ov, tp, un = tdist.side_probs(mu_t, g["total_line"])
        hw, tie = wins.home(mu_m), 0.0
        aw = 1 - hw
        mkt_home, _ = no_vig(g["home_moneyline"], g["away_moneyline"])
        common = {"game_id": g["game_id"], "season": g["season"], "week": g["week"],
                  "raw_margin": g["model_margin"], "raw_total": g["model_total"], "mkt_home_ml": mkt_home}
        options = [
            ("spread", "home", hc, hp, g["home_spread_odds"], g["result"], g["spread_line"], +1, mu_m - g["spread_line"]),
            ("spread", "away", ac, hp, g["away_spread_odds"], g["result"], g["spread_line"], -1, g["spread_line"] - mu_m),
            ("total", "over", ov, tp, g["over_odds"], g["total"], g["total_line"], +1, mu_t - g["total_line"]),
            ("total", "under", un, tp, g["under_odds"], g["total"], g["total_line"], -1, g["total_line"] - mu_t),
            ("ml", "home", hw, tie, g["home_moneyline"], g["result"], 0.0, +1, hw - mkt_home),
            ("ml", "away", aw, tie, g["away_moneyline"], g["result"], 0.0, -1, aw - (1 - mkt_home)),
        ]
        for market, side, pw, pp, odds, outcome, line, sgn, edge in options:
            if odds is None or line is None:
                continue
            res = _grade(outcome, line, sgn) if outcome is not None else None
            bets.append({**common, "market": market, "side": side, "p_win": pw, "p_push": pp,
                         "odds": odds, "line": line, "edge": edge, "ev": ev(pw, pp, odds),
                         "res": res, "units": _payout(res, odds) if res is not None else None})
    return pl.DataFrame(bets)


def potw_and_parlays(bets: pl.DataFrame, ev_min: float, k: int = 5):
    """Picks of the Week (top k by EV above ev_min) and the Golden Parlay from them."""
    picks, parlays = [], []
    for (s, w), wk in bets.filter(pl.col("ev") > ev_min).group_by("season", "week"):
        # one bet per game per market; best EV first
        top = wk.sort("ev", descending=True).unique(["game_id", "market"], keep="first", maintain_order=True).head(k)
        picks.append(top)
        legs = top.unique("game_id", keep="first", maintain_order=True).head(3)
        if legs.height >= 2:
            if (legs["res"] < 0).any():
                u = -0.5
            elif (legs["res"] == 0).all():
                u = 0.0
            else:  # pushes drop out of the parlay
                live = legs.filter(pl.col("res") > 0)
                u = 0.5 * (np.prod([american_to_decimal(o) for o in live["odds"]]) - 1)
            parlays.append({"season": s, "week": w, "legs": legs.height, "units": u, "won": u > 0})
    return (pl.concat(picks) if picks else bets.head(0)), pl.DataFrame(parlays)


def _record(df: pl.DataFrame) -> str:
    if df.height == 0:
        return "no bets"
    w, l, p = (df["res"] > 0).sum(), (df["res"] < 0).sum(), (df["res"] == 0).sum()
    u = df["units"].sum()
    return f"{w}-{l}-{p}  ({w / max(w + l, 1):.1%})  {u:+.1f}u  ROI {u / df.height:+.1%}  n={df.height}"


def report(proj: pl.DataFrame, bets: pl.DataFrame, blends: dict, p: Params, alpha: float) -> str:
    done = proj.filter(pl.col("result").is_not_null())
    lines = ["# Backtest report", "",
             f"Parameters tuned on {BACKTEST_FIRST}-{TUNE_LAST}; {TUNE_LAST + 1}+ is holdout.",
             f"Graded at nflverse closing lines and prices. Ratings params: `{asdict(p)}`, points ridge {alpha}.", ""]

    lines += ["## 1. Accuracy vs the closing line", "",
              "| Seasons | Games | Margin MAE model | Margin MAE close | Total MAE model | Total MAE close |",
              "|---|---|---|---|---|---|"]
    for label, lo, hi in (("Tuning", BACKTEST_FIRST, TUNE_LAST), ("Holdout", TUNE_LAST + 1, 2100)):
        d = done.filter(pl.col("season").is_between(lo, hi))
        lines.append(
            f"| {label} {lo}-{min(hi, d['season'].max())} | {d.height} "
            f"| {(d['model_margin'] - d['result']).abs().mean():.3f} | {(d['spread_line'] - d['result']).abs().mean():.3f} "
            f"| {(d['model_total'] - d['total']).abs().mean():.3f} | {(d['total_line'] - d['total']).abs().mean():.3f} |")
    lines += ["", "Per season:", "", "| Season | Margin model | Margin close | Total model | Total close |", "|---|---|---|---|---|"]
    for s in sorted(done["season"].unique()):
        d = done.filter(pl.col("season") == s)
        lines.append(f"| {s} | {(d['model_margin'] - d['result']).abs().mean():.2f} | {(d['spread_line'] - d['result']).abs().mean():.2f} "
                     f"| {(d['model_total'] - d['total']).abs().mean():.2f} | {(d['total_line'] - d['total']).abs().mean():.2f} |")

    lines += ["", "## 2. Raw model vs closing spread, by disagreement (Sam-style)", "",
              "Side = whichever side the raw model margin favors vs the closing spread. Holdout only.", "",
              "| Gap (pts) | Record |", "|---|---|"]
    raw = bets.filter((pl.col("market") == "spread") & (pl.col("season") > TUNE_LAST) & pl.col("res").is_not_null())
    raw = raw.with_columns(gap=(pl.col("raw_margin") - pl.col("line")).abs(),
                           raw_side=pl.when(pl.col("raw_margin") > pl.col("line")).then(pl.lit("home")).otherwise(pl.lit("away")))
    raw = raw.filter(pl.col("side") == pl.col("raw_side"))
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 99)):
        lines.append(f"| {lo}-{hi if hi < 99 else '+'} | {_record(raw.filter(pl.col('gap').is_between(lo, hi, closed='left')))} |")
    lines.append(f"| all | {_record(raw)} |")

    lines += ["", "## 3. How much the model is trusted vs the market (blend weights)", "",
              "Fit on earlier seasons' out-of-sample predictions: `result = a + b*model + c*line`.",
              "b near 0 and c near 1 means the model adds nothing the line doesn't already have.", "",
              "| Season | margin b (model) | margin c (line) | total b (model) | total c (line) | sigma margin | sigma total |",
              "|---|---|---|---|---|---|---|"]
    for s, bl in sorted(blends.items()):
        lines.append(f"| {s} | {bl.m_coef[1]:.2f} | {bl.m_coef[2]:.2f} | {bl.t_coef[1]:.2f} | {bl.t_coef[2]:.2f} | {bl.m_sigma:.1f} | {bl.t_sigma:.1f} |")

    hold = bets.filter((pl.col("season") > TUNE_LAST) & pl.col("res").is_not_null())
    lines += ["", "## 4. Blended bets by EV threshold (holdout, closing prices)", "",
              "| Market | EV > 0% | EV > 2% | EV > 4% | EV > 6% |", "|---|---|---|---|---|"]
    for market in ("spread", "total", "ml"):
        cells = [_record(hold.filter((pl.col("market") == market) & (pl.col("ev") > t))) for t in (0, .02, .04, .06)]
        lines.append(f"| {market} | " + " | ".join(cells) + " |")

    lines += ["", "## 5. Calibration of bets taken (EV > 0, holdout)", "",
              "| Model said | Bets | Actually won (excl. pushes) |", "|---|---|---|"]
    cal = hold.filter((pl.col("ev") > 0) & (pl.col("market") != "ml") & (pl.col("res") != 0))
    for lo in np.arange(0.50, 0.62, 0.02):
        d = cal.filter(pl.col("p_win").is_between(lo, lo + 0.02, closed="left"))
        if d.height:
            lines.append(f"| {lo:.0%}-{lo + 0.02:.0%} | {d.height} | {(d['res'] > 0).mean():.1%} |")

    lines += ["", "## 6. Picks of the Week and Golden Parlay (holdout)", "",
              "| EV floor | Picks of the Week | Golden Parlay (0.5u) |", "|---|---|---|"]
    for t in (0.0, 0.02, 0.04):
        pk, pr = potw_and_parlays(hold, t)
        pr_txt = (f"{int(pr['won'].sum())}-{pr.height - int(pr['won'].sum())}  {pr['units'].sum():+.1f}u  n={pr.height}"
                  if pr.height else "none")
        lines.append(f"| {t:.0%} | {_record(pk)} | {pr_txt} |")

    lines += ["", "## 7. Pre-declared segments (holdout spread bets, EV > 2%)", ""]
    seg = hold.filter((pl.col("market") == "spread") & (pl.col("ev") > 0.02))
    lines.append(f"- Weeks 1-4: {_record(seg.filter(pl.col('week') <= 4))}")
    lines.append(f"- Weeks 5-18: {_record(seg.filter(pl.col('week').is_between(5, 18)))}")
    lines.append(f"- Playoffs: {_record(seg.filter(pl.col('week') > 18))}")
    lines += ["", "Breakeven at -110 is 52.4%. Results within a few points of that over a few hundred bets are noise."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    inp, games = load_inputs()
    if args.tune:
        tune(inp, games)
    p, alpha = load_params()
    seasons = list(range(BACKTEST_FIRST, int(games["season"].max()) + 1))
    proj = walk_forward(inp, games, p, alpha, seasons)
    proj.write_parquet(REPORTS / "projections.parquet")
    bets, blends = price_seasons(proj, games, seasons)
    bets.write_parquet(REPORTS / "bets.parquet")
    text = report(proj, bets, blends, p, alpha)
    (REPORTS / "backtest.md").write_text(text, encoding="utf-8")
    print(text)


def price_seasons(proj: pl.DataFrame, games: pl.DataFrame, seasons: list[int]):
    """Blend, key-number distributions and win curve per season, fit on earlier seasons only."""
    all_bets, blends = [], {}
    hist = games.filter(pl.col("result").is_not_null() & pl.col("spread_line").is_not_null())
    for s in seasons[1:]:  # blend needs at least one earlier out-of-sample season
        prior = proj.filter((pl.col("season") < s) & pl.col("result").is_not_null())
        blend = Blend.fit(prior)
        blends[s] = blend
        h = hist.filter(pl.col("season") < s)
        mdist = KeyDist.fit(h["result"].to_numpy(), h["spread_line"].to_numpy(), MARGIN_RANGE, blend.m_sigma, True)
        tdist = KeyDist.fit(h["total"].to_numpy(), h["total_line"].to_numpy(), TOTAL_RANGE, blend.t_sigma, False)
        wins = WinCurve.fit(h["spread_line"].to_numpy(), h["result"].to_numpy())
        cur = proj.filter((pl.col("season") == s) & pl.col("spread_line").is_not_null())
        all_bets.append(price_bets(cur, blend, mdist, tdist, wins))
    return pl.concat(all_bets), blends


if __name__ == "__main__":
    main()
