"""NBA game lines: Pinnacle's no-vig number, shopped at DraftKings and FanDuel.

The model does not move the number (k = 0). Over six seasons of walk-forward tests (2020-21 to 2025-26) it never
beat the closing line: margin error 11.44 points against the close's 10.51, and its information beyond the
close (beta) was -0.03. Its early-season lean (beta +0.07 over each season's first three weeks) is not told
apart from zero. So a game's fair chances are Pinnacle's own with the vig removed (DraftKings and FanDuel's
average when Pinnacle has no price), and an edge exists only where DraftKings or FanDuel hang a number or a
price that those chances beat: line shopping.

Pinnacle's line and price give the center: the expected margin (or total) at which its no-vig chance of covering
its own line comes out. The chance of covering any other number comes from the shape of past outcomes around
the closing line (model/probs.py KeyDist, fit on hoopR's closes), which also gives the push chance on a whole
number.

Each game gets a spread pick and a total pick: the side and book with the best expected value. When no side is
positive, the pick is the model's side at its best price: a lean, no stake, counted at 1u in the record, so the
first checkup can see whether the early-season lean is worth anything. A moneyline is a pick only at positive EV.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from board.market import american, fmt_line
from model.probs import KeyDist

MARGINS = np.arange(-80, 81)
TOTALS = np.arange(140, 321)
SHOP = ("draftkings", "fanduel")


def _results() -> pl.DataFrame:
    from nba.backtest import closes
    from nba.model import team_games
    tg = team_games(range(2021, 2027)).filter(pl.col("is_home") | (pl.col("home") == 0)).unique("game_id", keep="first")
    return (tg.select("game_id", (pl.col("pts") - pl.col("opp_pts")).alias("margin"), (pl.col("pts") + pl.col("opp_pts")).alias("total"))
            .join(closes(), on="game_id", how="inner").drop_nulls())


def fit() -> tuple[KeyDist, KeyDist, dict]:
    r = _results()
    sm = float((r["margin"] - r["close_spread"]).std())
    st = float((r["total"] - r["close_total"]).std())
    md = KeyDist.fit(r["margin"].to_numpy(), r["close_spread"].to_numpy(), MARGINS, sm, True)
    td = KeyDist.fit(r["total"].to_numpy(), r["close_total"].to_numpy(), TOTALS, st, False)
    return md, td, {"games": r.height, "sigma_margin": round(sm, 2), "sigma_total": round(st, 2)}


def _novig(a: float, b: float) -> float:
    return (1 / a) / (1 / a + 1 / b)


def center(dist: KeyDist, line: float, p_over: float) -> float:
    """The expected outcome at which the chance of beating `line` (pushes aside) is `p_over`."""
    lo, hi = line - 30.0, line + 30.0
    for _ in range(36):
        mid = (lo + hi) / 2
        o, _, u = dist.side_probs(mid, line)
        if o / max(o + u, 1e-12) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class GamePricer:
    mdist: KeyDist
    tdist: KeyDist

    def fair(self, books: dict[str, dict]) -> dict:
        """Fair centers (expected home margin, expected total) and the home win chance, from Pinnacle's no-vig
        prices, else the average of DraftKings' and FanDuel's."""
        out: dict = {}
        for key, dist, name in (("spread", self.mdist, "margin"), ("total", self.tdist, "total")):
            src = ["pinnacle"] if key in books.get("pinnacle", {}) else [b for b in SHOP if key in books.get(b, {})]
            cs = [center(dist, books[b][key], _novig(books[b][f"{key}_price_a"], books[b][f"{key}_price_b"])) for b in src]
            if cs:
                out[name] = float(np.mean(cs))
                out[f"{name}_src"] = "pinnacle" if src == ["pinnacle"] else "dk/fd"
        src = ["pinnacle"] if "ml_home" in books.get("pinnacle", {}) else [b for b in SHOP if "ml_home" in books.get(b, {})]
        ps = [_novig(books[b]["ml_home"], books[b]["ml_away"]) for b in src]
        if ps:
            out["p_home"] = float(np.mean(ps))
        return out

    def options(self, books: dict[str, dict], fair: dict, home: str, away: str) -> dict[str, list[dict]]:
        """Every side at every shop book, with its chance and expected value at the fair number."""
        out: dict[str, list[dict]] = {"spread": [], "total": [], "ml": []}
        for b in SHOP:
            r = books.get(b, {})
            if "spread" in r and "margin" in fair:
                s = r["spread"]
                o, push, u = self.mdist.side_probs(fair["margin"], s)
                for side, team, p, dec, line in (("home", home, o, r["spread_price_a"], -s), ("away", away, u, r["spread_price_b"], s)):
                    out["spread"].append({"market": "spread", "side": side, "team": team, "line": line,
                                          "label": f"{team} {fmt_line(line)}", "book": b, "dec": dec, "p_win": p,
                                          "p_push": push, "ev": p * (dec - 1) - (1 - p - push)})
            if "total" in r and "total" in fair:
                t = r["total"]
                o, push, u = self.tdist.side_probs(fair["total"], t)
                for side, p, dec in (("over", o, r["total_price_a"]), ("under", u, r["total_price_b"])):
                    out["total"].append({"market": "total", "side": side, "team": None, "line": t,
                                         "label": f"{side.title()} {t:g}", "book": b, "dec": dec, "p_win": p,
                                         "p_push": push, "ev": p * (dec - 1) - (1 - p - push)})
            if "ml_home" in r and "p_home" in fair:
                for side, team, p, dec in (("home", home, fair["p_home"], r["ml_home"]), ("away", away, 1 - fair["p_home"], r["ml_away"])):
                    out["ml"].append({"market": "ml", "side": side, "team": team, "line": None, "label": f"{team} ML",
                                      "book": b, "dec": dec, "p_win": p, "p_push": 0.0, "ev": p * (dec - 1) - (1 - p)})
        for v in out.values():
            for o in v:
                o["american"] = american(o["dec"])
        return out

    def picks(self, books: dict[str, dict], fair: dict, home: str, away: str,
              model_margin: float | None, model_total: float | None) -> dict[str, dict]:
        """The spread and total picks (the best positive-EV side, else the model's side as a lean) and the
        moneyline when it has positive EV."""
        opts = self.options(books, fair, home, away)
        out = {}
        for market, model, centre, sides in (("spread", model_margin, fair.get("margin"), ("home", "away")),
                                             ("total", model_total, fair.get("total"), ("over", "under"))):
            o = opts[market]
            if not o:
                continue
            best = max(o, key=lambda x: x["ev"])
            model_side = None if model is None or centre is None else (sides[0] if model > centre else sides[1])
            if best["ev"] <= 0 and model_side:
                best = max((x for x in o if x["side"] == model_side), key=lambda x: x["ev"])
            out[market] = {**best, "model_side": model_side, "with_model": model_side == best["side"] if model_side else None}
        if opts["ml"]:
            best = max(opts["ml"], key=lambda x: x["ev"])
            if best["ev"] > 0:
                out["ml"] = {**best, "model_side": None, "with_model": None}
        return out

    def p_side(self, pick: dict, fair: dict) -> tuple[float, float] | None:
        """(win, push) chance of a locked pick at a later fair number (its closing chance)."""
        if pick["market"] == "spread" and "margin" in fair:
            s = -pick["line"] if pick["side"] == "home" else pick["line"]
            o, push, u = self.mdist.side_probs(fair["margin"], s)
            return (o if pick["side"] == "home" else u), push
        if pick["market"] == "total" and "total" in fair:
            o, push, u = self.tdist.side_probs(fair["total"], pick["line"])
            return (o if pick["side"] == "over" else u), push
        if pick["market"] == "ml" and "p_home" in fair:
            return (fair["p_home"] if pick["side"] == "home" else 1 - fair["p_home"]), 0.0
        return None
