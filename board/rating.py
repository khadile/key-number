"""The model rating: a 0-100 scorecard for every board, like a benchmark card for an AI model (user, 2026-11-14).

The user asked for a rubric to judge the models by and improve them against, running from "does not predict
anything, better to pick at random" to "extremely predictive, positive ROI". Each board (NFL game lines, NFL props,
NBA game lines, NBA props, the college tracker) is scored on the picks that carry a stake, in seven categories:

    Beats the close      EV of each pick at the closing price (the sharpest yardstick: it settles at the tip)
    Profit               return per unit staked at 1u
    Calibration          does a pick said to win 55% win 55%? (won minus said, in points)
    Edge ranking         do the bigger stated edges close better than the smaller ones?
    Filter               do the picks the board singles out (Picks of the Day/Week, the bet list) close better?
    Recent form          beats the close over the last three weeks (NFL) or two weeks (NBA, college)
    Evidence             how many graded picks the rest stands on

Each category maps its measurement onto 0-100 through fixed anchors (betting at random, which pays the vig,
lands around 12-19 on "beats the close" and "profit"; breaking even is 45). A measurement only moves away from 45
as far as its evidence allows: fully at two standard errors from zero, proportionally below that (calibration by
sample size). The overall score is a weighted mean (evidence shown, not averaged), and its tier names what the
score means. Random betting at -110 scores about 22 ("No edge").

Computed when the page is rendered (board/render.py) from the payloads every build already writes.
"""
import math
from datetime import date, timedelta

TIERS = [(85, "Strong edge", "Extremely predictive: beats the closing line and makes money, well calibrated, over a large sample."),
         (70, "Solid edge", "Beats the close and the price with good calibration; the evidence is past chance."),
         (55, "Some edge", "Beats the close or the price more often than not; promising, not proven."),
         (40, "Unproven", "No clear edge either way yet: too few picks, or results inside the noise."),
         (20, "No edge", "About what picking at random costs (the vig): not predictive."),
         (0, "Worse than random", "Loses more than the vig: better to pick at random than to use this.")]
# Evidence is shown, not averaged: a big sample of bad picks is not a better model, and small samples are already
# pulled toward 45 category by category. Betting at random at -110 scores about 22 overall ("No edge").
WEIGHTS = {"clv": 35, "roi": 20, "cal": 10, "rank": 10, "filter": 10, "recent": 15, "evidence": 0}
NAMES = {"clv": "Beats the close", "roi": "Profit", "cal": "Calibration", "rank": "Edge ranking", "filter": "Filter",
         "recent": "Recent form", "evidence": "Evidence"}
A_CLV = [(-0.07, 0), (-0.045, 12), (-0.02, 28), (0.0, 45), (0.01, 58), (0.02, 68), (0.035, 80), (0.05, 90), (0.08, 100)]
A_ROI = [(-0.12, 0), (-0.045, 15), (0.0, 45), (0.03, 60), (0.06, 75), (0.10, 88), (0.15, 100)]
A_CAL = [(-0.10, 0), (-0.06, 15), (-0.03, 35), (-0.01, 60), (0.0, 75), (0.02, 85), (0.05, 90)]
A_DIFF = [(-0.04, 0), (-0.015, 20), (0.0, 40), (0.01, 55), (0.02, 68), (0.04, 85), (0.06, 100)]
A_N = [(0, 0), (30, 20), (100, 40), (250, 60), (500, 78), (1000, 92), (2000, 100)]
MID = 45.0


def _interp(x: float, pts) -> float:
    if x <= pts[0][0]:
        return float(pts[0][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(pts[-1][1])


def _mean_se(xs: list[float]) -> tuple[float | None, float | None]:
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, None
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n)


def _pulled(raw: float, conf: float) -> float:
    return MID + (raw - MID) * max(0.0, min(1.0, conf))


def _cat_z(value, se, anchors, n, fmt) -> dict:
    if value is None or n < 5:
        return {"score": None, "value": None, "n": n, "text": "not enough graded picks"}
    z = value / se if se else 0.0
    raw = _interp(value, anchors)
    return {"score": round(_pulled(raw, abs(z) / 2), 1), "raw": round(raw, 1), "value": value, "z": round(z, 2), "n": n, "text": fmt(value, se)}


def _dec(american) -> float | None:
    if american is None:
        return None
    a = float(american)
    return 1 + a / 100 if a > 0 else 1 + 100 / -a if a < 0 else None


def score_board(rows: list[dict], ev_close, recent, filt, label_n: str = "picks") -> dict:
    """rows: graded staked picks (result W/L/P, american, p_win, ev, and whatever ev_close/recent/filt read)."""
    g = [r for r in rows if r.get("result") in ("W", "L", "P") and _dec(r.get("american"))]
    rets = [(_dec(r["american"]) - 1) if r["result"] == "W" else (-1.0 if r["result"] == "L" else 0.0) for r in g]
    evc = [(r, ev_close(r)) for r in g]
    evc = [(r, e) for r, e in evc if e is not None]
    pct = lambda v: f"{100 * v:+.1f}%"
    se_txt = lambda v, se: f"{pct(v)} ± {100 * se:.1f}" if se else pct(v)
    cats = {}
    m, se = _mean_se([e for _, e in evc])
    cats["clv"] = _cat_z(m, se, A_CLV, len(evc), lambda v, s: f"{se_txt(v, s)} EV at the close")
    m, se = _mean_se(rets)
    cats["roi"] = _cat_z(m, se, A_ROI, len(rets), lambda v, s: f"{se_txt(v, s)} per unit")
    dec = [r for r in g if r["result"] in ("W", "L") and r.get("p_win") is not None]
    if len(dec) >= 20:
        said = sum(r["p_win"] for r in dec) / len(dec)
        won = sum(r["result"] == "W" for r in dec) / len(dec)
        gap = won - said
        raw = _interp(gap, A_CAL)
        cats["cal"] = {"score": round(_pulled(raw, len(dec) / 300), 1), "raw": round(raw, 1), "value": gap, "n": len(dec),
                       "text": f"won {100 * won:.1f}% where it said {100 * said:.1f}%"}
    else:
        cats["cal"] = {"score": None, "value": None, "n": len(dec), "text": "not enough graded picks"}
    if len(evc) >= 20:
        srt = sorted(evc, key=lambda t: t[0].get("ev") or 0)
        h = len(srt) // 2
        lo, hi = [e for _, e in srt[:h]], [e for _, e in srt[h:]]
        (ml, sl), (mh, sh) = _mean_se(lo), _mean_se(hi)
        d, s = mh - ml, math.sqrt((sl or 0) ** 2 + (sh or 0) ** 2)
        cats["rank"] = _cat_z(d, s, A_DIFF, len(evc), lambda v, s_: f"bigger edges {pct(v)} better at the close" if v >= 0 else f"bigger edges {pct(-v)} worse at the close")
    else:
        cats["rank"] = {"score": None, "value": None, "n": len(evc), "text": "not enough graded picks"}
    fin = [e for r, e in evc if filt(r)]
    rest = [e for r, e in evc if not filt(r)]
    if len(fin) >= 10 and len(rest) >= 10:
        (mf, sf), (mr, sr) = _mean_se(fin), _mean_se(rest)
        d, s = mf - mr, math.sqrt((sf or 0) ** 2 + (sr or 0) ** 2)
        cats["filter"] = _cat_z(d, s, A_DIFF, len(fin), lambda v, s_: f"singled-out picks {pct(v)} {'better' if v >= 0 else 'worse'} at the close ({len(fin)})")
    else:
        cats["filter"] = {"score": None, "value": None, "n": len(fin), "text": "too few singled-out picks graded"}
    rec = [e for r, e in evc if recent(r)]
    m, se = _mean_se(rec)
    cats["recent"] = _cat_z(m, se, A_CLV, len(rec), lambda v, s: f"{se_txt(v, s)} EV at the close lately")
    cats["evidence"] = {"score": round(_interp(len(evc), A_N), 1), "value": len(evc), "n": len(evc), "text": f"{len(evc)} graded {label_n} with a close"}
    have = {k: c for k, c in cats.items() if c["score"] is not None and WEIGHTS[k]}
    wsum = sum(WEIGHTS[k] for k in have)
    overall = round(sum(WEIGHTS[k] * have[k]["score"] for k in have) / wsum, 1) if wsum else None
    tier = next((t for t in TIERS if overall is not None and overall >= t[0]), None)
    return {"overall": overall, "tier": tier[1] if tier else "No data", "tier_text": tier[2] if tier else "",
            "n": len(g), "categories": [{"key": k, "name": NAMES[k], "weight": WEIGHTS[k], **cats[k]} for k in WEIGHTS]}


def _is_par(r) -> bool:
    return bool(r.get("legs")) or r.get("market") in ("golden", "lottery", "parlay")


def boards(nfl: dict | None, nba: dict | None, ncaab: dict | None, today: date | None = None) -> dict:
    today = today or date.today()
    out = []
    if nfl:
        rows = [r for r in (nfl.get("record") or {}).get("live_rows") or [] if not _is_par(r)]
        wk = max((r.get("week") or 0) for r in rows) if rows else 0
        staked = [r for r in rows if (r.get("risk") is None or r.get("risk") > 0) and (r.get("ev") or 0) > 0 and not r.get("hold") and not r.get("superseded")]
        out.append({"key": "nfl-lines", "sport": "NFL", "name": "NFL game lines", "since": "week 5 (live)",
                    **score_board(staked, lambda r: r.get("ev_close"), lambda r: (r.get("week") or 0) > wk - 3, lambda r: r.get("potw") or r.get("bet"))})
        prow = [r for r in ((nfl.get("props") or {}).get("record") or {}).get("live_rows") or [] if not _is_par(r)]
        pwk = max((r.get("week") or 0) for r in prow) if prow else 0
        out.append({"key": "nfl-props", "sport": "NFL", "name": "NFL player props", "since": "week 5 (live)",
                    **score_board(prow, lambda r: r.get("ev_close"), lambda r: (r.get("week") or 0) > pwk - 3, lambda r: r.get("potw") or r.get("bet"))})
    if nba:
        cut = (today - timedelta(days=14)).isoformat()
        rows = [r for r in (nba.get("record") or {}).get("rows") or [] if not _is_par(r) and r.get("risk", 1) != 0 and (r.get("ev") or 0) > 0]
        out.append({"key": "nba-lines", "sport": "NBA", "name": "NBA game lines", "since": "opening night",
                    **score_board(rows, lambda r: r.get("ev_close"), lambda r: (r.get("day") or "") >= cut, lambda r: r.get("potd") or r.get("bet"))})
        prow = [r for r in ((nba.get("props") or {}).get("record") or {}).get("rows") or [] if not _is_par(r) and r.get("risk", 1) != 0]
        ref_from = ((nba.get("meta") or {}).get("props_ref_from")) or "2026-11-11"
        # before the reference change the close was Pinnacle's with the correction that made the picks; its raw
        # close is the honest judge for those days (CLAUDE.md section 11)
        evc = lambda r: r.get("ev_close") if (r.get("day") or "") >= ref_from else r.get("ev_close_raw")
        out.append({"key": "nba-props", "sport": "NBA", "name": "NBA player props", "since": "opening night",
                    **score_board(prow, evc, lambda r: (r.get("day") or "") >= cut, lambda r: r.get("potd") or r.get("bet"))})
    if ncaab:
        cut = (today - timedelta(days=14)).isoformat()
        rows = [r for r in (ncaab.get("record") or {}).get("rows") or []]
        out.append({"key": "ncaab", "sport": "NCAAB", "name": "College basketball tracker", "since": "Nov 14 (paper, no stake)",
                    **score_board(rows, lambda r: r.get("ev_close"), lambda r: (r.get("day") or "") >= cut, lambda r: r.get("angle") == "fade_cover", "flags")})
    return {"boards": out, "tiers": [{"min": t[0], "name": t[1], "text": t[2]} for t in TIERS], "weights": WEIGHTS, "names": NAMES,
            "anchors": {"clv": A_CLV, "roi": A_ROI, "cal": A_CAL, "diff": A_DIFF, "n": A_N}, "as_of": today.isoformat()}
