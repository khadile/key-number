"""Adaptation-speed test: should each unit (pass/rush, offense/defense, pace,
special teams) update at its own speed, and should defenses carry less of last
season than offenses?

1. Tune the per-unit half-lives and offseason gaps on 2018-2021 only, starting
   from the current saved parameters.
2. Compare current vs per-unit on the 2022+ holdout, once.

    python -m backtest.speeds
"""
import json
from dataclasses import asdict, replace

import numpy as np
import polars as pl

from config import BACKTEST_FIRST, REPORTS, TUNE_LAST
from backtest.run import _record, _score, load_params, price_seasons, walk_forward
from model.ratings import load_inputs

OUT = REPORTS / "params_speeds.json"
HALF_LIVES = [3, 4, 6, 8, 12, 16, 24, 40]
GAPS = [0, 4, 8, 12, 20, 30]
FIELDS = {
    "hl_pass_off": HALF_LIVES, "hl_pass_def": HALF_LIVES,
    "hl_rush_off": HALF_LIVES, "hl_rush_def": HALF_LIVES,
    "hl_tend": HALF_LIVES, "hl_st": HALF_LIVES,
    "gap_off": GAPS, "gap_def": GAPS,
}


def tune(inp, games, base, alpha):
    seasons = range(BACKTEST_FIRST, TUNE_LAST + 1)
    # Start every per-unit value at the shared value so the search begins exactly at the current model.
    p = replace(base, **{f: (base.half_life if f.startswith("hl") else base.gap) for f in FIELDS})
    best = _score(walk_forward(inp, games, p, alpha, seasons))
    print(f"start (current model) {best:.4f}")
    for _ in range(2):
        for name, values in FIELDS.items():
            for v in values:
                cand = replace(p, **{name: float(v)})
                sc = _score(walk_forward(inp, games, cand, alpha, seasons))
                if sc < best - 1e-4:
                    best, p = sc, cand
            print(f"  {name:<12} -> {getattr(p, name):>5}   score {best:.4f}")
    OUT.write_text(json.dumps({"ratings": asdict(p), "points_alpha": alpha,
                               "tuned_on": f"{BACKTEST_FIRST}-{TUNE_LAST}", "score": best}, indent=2))
    return p


def summarize(label, proj, bets):
    hold = proj.filter((pl.col("season") > TUNE_LAST) & (pl.col("season") <= 2025) & pl.col("result").is_not_null())
    tune_ = proj.filter(pl.col("season").is_between(BACKTEST_FIRST, TUNE_LAST) & pl.col("result").is_not_null())
    hb = bets.filter((pl.col("season") > TUNE_LAST) & (pl.col("season") <= 2025) & pl.col("res").is_not_null())
    mae = lambda d, a, b: float((d[a] - d[b]).abs().mean())
    raw = hb.filter(pl.col("market") == "total").with_columns(
        raw_side=pl.when(pl.col("raw_total") > pl.col("line")).then(pl.lit("over")).otherwise(pl.lit("under")))
    raw = raw.filter(pl.col("side") == pl.col("raw_side"))
    raw_ats = hb.filter(pl.col("market") == "spread").with_columns(
        raw_side=pl.when(pl.col("raw_margin") > pl.col("line")).then(pl.lit("home")).otherwise(pl.lit("away")))
    raw_ats = raw_ats.filter(pl.col("side") == pl.col("raw_side"))
    return {
        "label": label,
        "tune_margin": mae(tune_, "model_margin", "result"), "tune_total": mae(tune_, "model_total", "total"),
        "hold_margin": mae(hold, "model_margin", "result"), "hold_total": mae(hold, "model_total", "total"),
        "close_margin": mae(hold, "spread_line", "result"), "close_total": mae(hold, "total_line", "total"),
        "raw_ats": _record(raw_ats), "raw_ou": _record(raw),
        "blend_spread_2": _record(hb.filter((pl.col("market") == "spread") & (pl.col("ev") > .02))),
        "blend_total_0": _record(hb.filter((pl.col("market") == "total") & (pl.col("ev") > 0))),
        "early_ou": _record(raw.filter(pl.col("week") <= 6)),
    }


def main() -> None:
    inp, games = load_inputs()
    base, alpha = load_params()
    speeds = tune(inp, games, base, alpha)
    seasons = list(range(BACKTEST_FIRST, int(games["season"].max()) + 1))
    rows = []
    for label, p in (("current (one speed)", base), ("per-unit speeds", speeds)):
        proj = walk_forward(inp, games, p, alpha, seasons)
        bets, _ = price_seasons(proj, games, seasons)
        rows.append(summarize(label, proj, bets))
        if label.startswith("per-unit"):
            proj.write_parquet(REPORTS / "projections_speeds.parquet")
            bets.write_parquet(REPORTS / "bets_speeds.parquet")

    lines = ["# Adaptation-speed test", "",
             f"Tuned on {BACKTEST_FIRST}-{TUNE_LAST}. Holdout 2022-2025, graded at closing lines.", "",
             "Per-unit parameters chosen (half-life in weeks; gap = offseason length in weeks of decay):", "",
             "| Unit | Current | Per-unit |", "|---|---|---|"]
    for f in FIELDS:
        cur = base.half_life if f.startswith("hl") else base.gap
        lines.append(f"| {f} | {cur:g} | {getattr(speeds, f):g} |")
    lines += ["", "| Metric | " + " | ".join(r["label"] for r in rows) + " |", "|---|---|---|"]
    for key, name in [("tune_margin", "Margin MAE, tuning 2018-21"), ("tune_total", "Total MAE, tuning 2018-21"),
                      ("hold_margin", "Margin MAE, holdout"), ("hold_total", "Total MAE, holdout"),
                      ("close_margin", "  closing line margin MAE"), ("close_total", "  closing line total MAE"),
                      ("raw_ats", "Raw model ATS vs close"), ("raw_ou", "Raw model O/U vs close"),
                      ("early_ou", "Raw model O/U, weeks 1-6"),
                      ("blend_spread_2", "Blended spread bets EV>2%"), ("blend_total_0", "Blended total bets EV>0")]:
        vals = [f"{r[key]:.3f}" if isinstance(r[key], float) else r[key] for r in rows]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    text = "\n".join(lines)
    (REPORTS / "speeds.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
