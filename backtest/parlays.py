"""Golden Parlay search: which leg count / market mix / selection rule, if any,
has positive ROI?

Legs come from reports/bets.parquet (model probability and EV at the closing
price, every game, every market). One leg per game, because same-game legs are
correlated and books restrict them. Each parlay risks 0.5u. A pushed leg drops
out; any losing leg loses the parlay.

The rule is chosen on 2019-2021 and judged once on 2022-2025.

    python -m backtest.parlays
"""
import itertools

import numpy as np
import polars as pl

from config import PARLAY_RISK, REPORTS, TUNE_LAST
from model.probs import american_to_decimal

MIXES = {
    "any": {"spread", "total", "ml"},
    "spread": {"spread"}, "total": {"total"}, "ml": {"ml"},
    "spread+total": {"spread", "total"},
}
SELECT = {
    "top EV": "ev",           # biggest model edge
    "top p_win": "p_win",     # most likely legs (mostly ML favorites)
}


def week_parlay(wk: pl.DataFrame, k: int | None, mix: set, sort_col: str, ev_min: float):
    """Pick legs for one week. k=None means dynamic: keep adding legs (best EV first)
    while each leg clears ev_min and the parlay's model EV keeps rising, up to 4."""
    pool = wk.filter(pl.col("market").is_in(list(mix)) & (pl.col("ev") > ev_min))
    pool = pool.sort(sort_col, descending=True).unique("game_id", keep="first", maintain_order=True)
    if k is not None:
        legs = pool.head(k)
        return legs if legs.height == k else None
    legs, best = [], 0.0
    for row in pool.head(4).iter_rows(named=True):
        trial = legs + [row]
        model_ev = np.prod([r["p_win"] * american_to_decimal(r["odds"]) for r in trial]) - 1
        if len(trial) >= 2 and model_ev <= best:
            break
        legs, best = trial, model_ev if len(trial) >= 2 else best
    return pl.DataFrame(legs) if len(legs) >= 2 else None


def grade(legs: pl.DataFrame) -> tuple[float, float]:
    """(units won, model EV) for a 0.5u parlay."""
    model_ev = float(np.prod([p * american_to_decimal(o) for p, o in zip(legs["p_win"], legs["odds"])]) - 1)
    if (legs["res"] < 0).any():
        return -PARLAY_RISK, model_ev
    live = legs.filter(pl.col("res") > 0)
    if live.height == 0:
        return 0.0, model_ev
    return PARLAY_RISK * (np.prod([american_to_decimal(o) for o in live["odds"]]) - 1), model_ev


def run_rule(bets: pl.DataFrame, k, mix, sort_col, ev_min) -> dict:
    out = []
    for (s, w), wk in bets.group_by("season", "week"):
        legs = week_parlay(wk, k, MIXES[mix], sort_col, ev_min)
        if legs is None:
            continue
        u, mev = grade(legs)
        out.append({"season": s, "week": w, "legs": legs.height, "units": u, "model_ev": mev})
    if not out:
        return {"n": 0}
    d = pl.DataFrame(out)
    risked = PARLAY_RISK * d.height
    return {"n": d.height, "won": int((d["units"] > 0).sum()), "units": float(d["units"].sum()),
            "roi": float(d["units"].sum() / risked), "avg_legs": float(d["legs"].mean()),
            "model_ev": float(d["model_ev"].mean())}


def main() -> None:
    bets = pl.read_parquet(REPORTS / "bets.parquet").filter(
        pl.col("res").is_not_null() & (pl.col("week") <= 18))  # regular season: playoffs have too few games
    tune = bets.filter(pl.col("season") <= TUNE_LAST)
    hold = bets.filter(pl.col("season").is_between(TUNE_LAST + 1, 2025))

    rules = [(k, mix, sel, ev) for k in (2, 3, 4, 5, None) for mix in MIXES for sel in SELECT for ev in (-1.0, 0.0, 0.02)]
    rows = []
    for k, mix, sel, ev in rules:
        t = run_rule(tune, k, mix, SELECT[sel], ev)
        h = run_rule(hold, k, mix, SELECT[sel], ev)
        if t["n"] < 20 or h["n"] < 20:
            continue
        rows.append({"legs": "dynamic" if k is None else str(k), "mix": mix, "select": sel,
                     "leg_ev_min": "none" if ev < -0.5 else f"{ev:.0%}",
                     "tune_n": t["n"], "tune_roi": t["roi"], "hold_n": h["n"], "hold_won": h["won"],
                     "hold_units": h["units"], "hold_roi": h["roi"], "hold_model_ev": h["model_ev"]})
    R = pl.DataFrame(rows).sort("tune_roi", descending=True)
    chosen = R.row(0, named=True)

    lines = ["# Golden Parlay search", "",
             f"{R.height} rules tested. Legs: one per game, from every spread/total/ML bet, priced at the closing line.",
             f"Each parlay risks {PARLAY_RISK}u. Rule chosen on 2019-{TUNE_LAST}; holdout 2022-2025 judged once.", "",
             "## The rule the tuning seasons picked", "",
             f"- {chosen['legs']} legs, {chosen['mix']}, {chosen['select']}, leg EV floor {chosen['leg_ev_min']}",
             f"- Tuning ROI {chosen['tune_roi']:+.1%} over {chosen['tune_n']} weeks",
             f"- **Holdout: {chosen['hold_won']}-{chosen['hold_n'] - chosen['hold_won']}, "
             f"{chosen['hold_units']:+.1f}u, ROI {chosen['hold_roi']:+.1%}** (model expected {chosen['hold_model_ev']:+.1%})", "",
             "## How many rules were profitable", "",
             f"- Tuning seasons: {(R['tune_roi'] > 0).sum()} of {R.height}",
             f"- Holdout seasons: {(R['hold_roi'] > 0).sum()} of {R.height}",
             f"- Correlation between a rule's tuning ROI and its holdout ROI: {np.corrcoef(R['tune_roi'], R['hold_roi'])[0, 1]:+.2f}",
             "  (near zero means past parlay results say nothing about future ones)", "",
             "## By leg count (best tuning rule for each, holdout result)", "",
             "| Legs | Rule | Tuning ROI | Holdout record | Holdout ROI |", "|---|---|---|---|---|"]
    for legs in (2, 3, 4, 5, "dynamic"):
        sub = R.filter(pl.col("legs").cast(pl.Utf8) == str(legs))
        if sub.height:
            r = sub.row(0, named=True)
            lines.append(f"| {legs} | {r['mix']}, {r['select']}, EV floor {r['leg_ev_min']} | {r['tune_roi']:+.1%} "
                         f"| {r['hold_won']}-{r['hold_n'] - r['hold_won']} ({r['hold_units']:+.1f}u) | {r['hold_roi']:+.1%} |")
    lines += ["", "## By market mix (best tuning rule for each, holdout result)", "",
              "| Mix | Rule | Tuning ROI | Holdout ROI |", "|---|---|---|---|"]
    for mix in MIXES:
        sub = R.filter(pl.col("mix") == mix)
        if sub.height:
            r = sub.row(0, named=True)
            lines.append(f"| {mix} | {r['legs']} legs, {r['select']}, EV floor {r['leg_ev_min']} | {r['tune_roi']:+.1%} | {r['hold_roi']:+.1%} |")
    lines += ["", "## Reference: what a parlay costs with no edge", "",
              "| Legs | ROI at -110 per leg, 50% legs |", "|---|---|"]
    for k in (2, 3, 4, 5):
        lines.append(f"| {k} | {(0.5 * american_to_decimal(-110)) ** k - 1:+.1%} |")
    text = "\n".join(lines)
    (REPORTS / "parlays.md").write_text(text, encoding="utf-8")
    R.write_parquet(REPORTS / "parlay_rules.parquet")
    print(text)


if __name__ == "__main__":
    main()
