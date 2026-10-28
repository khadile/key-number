"""How much Pinnacle's closing price overrates overs, by market, measured again every build (user, 2026-10-27).

A report, not a price input. The board prices props at Pinnacle's no-vig number (k near 0) with no over
correction; the lean check (model/checks.py) instead asks overs on receiving yards, rushing yards and receptions
for 5%+ EV, a bar set once from 2025-26 closes that made those overs 2-5 points too likely. The NBA's version of
that correction, fixed from an older season, was wrong by market within a week (assists 6x too small, points 4x
too big), so this keeps the NFL's measured.

Each tracked prop of the season with Pinnacle's close (kickoff - 5 min) and a result, pushes left out, gives
y = over hit against Pinnacle's no-vig chance. Per market, the shift b in logit space with
    P(over) = sigmoid(logit(p_pinnacle) + b)
is fitted MAP: prior N(PRIOR[market], TAU^2), each week weighted by its age (HALF_LIFE weeks). PRIOR is the same
fit on 2025's Pinnacle closes (SportsGameOdds, tracked players, 4,836 props), so it starts where the lean rule's
evidence was. What it means for a pick: an over priced around -110 is worth about 1.91 x (sigmoid(b) - 0.5) less
than its stated EV.

Weeks 1-6 helped set the lean rule; weeks 7 on are its out-of-sample test, shown on their own.
"""
import math

import numpy as np

TAU = 0.10                 # prior sd, logit (same as the NBA's)
HALF_LIFE = 4              # weeks
TEST_WEEK = 7              # first week the lean rule had not seen
DEC = 1.91                 # -110, to turn a probability shift into EV
# 2025 Pinnacle closes, tracked players (reports/props_sgo_lines.parquet; scratch fit 2026-10-27), unweighted MLE:
# pass_yds 472 props +0.5 pts at even money, rush_yds 935 -1.8, rec_yds 1,734 -3.4, receptions 1,695 -5.0.
PRIOR = {"pass_yds": 0.0181, "rush_yds": -0.0705, "rec_yds": -0.1376, "receptions": -0.2010}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def fit(z: np.ndarray, y: np.ndarray, w: np.ndarray, prior: float, tau: float | None = TAU) -> tuple[float, float]:
    """MAP logit shift and its SE: y ~ Bernoulli(sigmoid(z + b)), b ~ N(prior, tau^2) (tau None: no prior)."""
    b = prior
    for _ in range(50):
        p = _sig(z + b)
        g = float(np.sum(w * (y - p))) - ((b - prior) / tau ** 2 if tau else 0.0)
        h = float(np.sum(w * p * (1 - p))) + (1 / tau ** 2 if tau else 0.0)
        step = g / h
        b += step
        if abs(step) < 1e-9:
            break
    return b, 1 / math.sqrt(h)


def rows(close_b: dict, stats: dict, keys, week_of: dict) -> list[dict]:
    """Tracked props with Pinnacle's close and a result, pushes left out: market, week, p (Pinnacle no-vig), y."""
    out = []
    for g, pid, m in keys:
        pc = (close_b.get((g, pid, m)) or {}).get("pinnacle")
        a = (stats.get((g, pid)) or {}).get(m)
        if pc is None or a is None or a == pc["line"] or not 0 < pc["p_over"] < 1:
            continue
        out.append({"market": m, "week": week_of[g], "p": pc["p_over"], "y": float(a > pc["line"])})
    return out


def _side(R: list[dict]) -> dict:
    if not R:
        return {"n": 0}
    said, hit = float(np.mean([r["p"] for r in R])), float(np.mean([r["y"] for r in R]))
    return {"n": len(R), "said": round(said, 4), "hit": round(hit, 4), "gap": round(hit - said, 4),
            "gap_se": round(math.sqrt(max(hit * (1 - hit), 0.01) / len(R)), 4)}


def report(R: list[dict], week: int, lean: dict[str, float], floor: float) -> dict:
    """Per market: this season's raw gap (and weeks TEST_WEEK on alone), the learned shift, and what an over at
    the bar it faces (the lean's 5% or the 2% floor) is worth once the shift is counted."""
    out = {"half_life": HALF_LIFE, "tau": TAU, "test_week": TEST_WEEK, "through": max((r["week"] for r in R), default=None), "markets": {}}
    for m, prior in PRIOR.items():
        Rm = [r for r in R if r["market"] == m]
        z = np.array([math.log(r["p"] / (1 - r["p"])) for r in Rm])
        y = np.array([r["y"] for r in Rm])
        w = np.array([0.5 ** ((week - r["week"]) / HALF_LIFE) for r in Rm])
        b, se = fit(z, y, w, prior) if Rm else (prior, TAU)
        cost = DEC * (float(_sig(b)) - 0.5)
        bar = lean.get(m, floor)
        out["markets"][m] = {**_side(Rm), "test": _side([r for r in Rm if r["week"] >= TEST_WEEK]),
                             "shift": round(b, 4), "se": round(se, 4), "prior": prior,
                             "cost": round(cost, 4), "bar": bar, "lean": m in lean, "worth": round(bar + cost, 4)}
    return out
