"""The good-pick checks for game-line picks (model/checks.py): Pinnacle's archived prices and the injury report.

Sharp chance: the chance our exact bet wins with the market's number read from Pinnacle's no-vig PRICE, juice
included (the median at which Pinnacle's line is a fair price, on the same key-number distributions the pricer
uses), then moved toward the model by k as the board does. The board's own EV reads Pinnacle's line as a coin
flip whatever its juice: in 2026 weeks 1-5 that made CHI +3 -102 look +3.8% when Pinnacle had CHI +3 at +104
(-4.9% at Pinnacle's price), and moneylines lean on the model's win chance where Pinnacle's own moneyline is
the sharper number. This check prices every pick the corrected way.
"""
from datetime import datetime

import polars as pl

from board.market import BOOK_NAMES, Lines, Pricer, american, fmt_line, reference
from config import RAW
from model import checks as C
from pipeline.odds_history import SHOP_BOOKS


def _novig(a: float, b: float) -> float:
    return (1 / a) / (1 / a + 1 / b)


def _solve_mu(dist, line: float, p: float) -> float:
    """The median outcome at which `line` is beaten (over) with chance p, pushes left out."""
    lo, hi = line - 25.0, line + 25.0
    for _ in range(40):
        mid = (lo + hi) / 2
        over, _, under = dist.side_probs(mid, line)
        if over / max(over + under, 1e-12) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _mu_sharp(pricer: Pricer, market: str, row: dict) -> float | None:
    """The median margin (spread, moneyline) or total at which Pinnacle's line is a fair price."""
    if market == "total":
        if {"total", "total_price_a", "total_price_b"} <= row.keys():
            return _solve_mu(pricer.tdist, row["total"], _novig(row["total_price_a"], row["total_price_b"]))
    elif {"spread", "spread_price_a", "spread_price_b"} <= row.keys():
        return _solve_mu(pricer.mdist, row["spread"], _novig(row["spread_price_a"], row["spread_price_b"]))
    return None


def bet_chance(pricer: Pricer, pk: dict, row: dict, model: float | None = None) -> tuple[float, float] | None:
    """(p_win, p_push) of the pick's exact bet with the market read from Pinnacle's no-vig price in `row`, then,
    given the model's number, moved toward it by k as the board does. None without a Pinnacle price."""
    m, mu = pk["market"], _mu_sharp(pricer, pk["market"], row)
    if m == "ml":
        if not {"ml_home", "ml_away"} <= row.keys():
            return None
        ph = _novig(row["ml_home"], row["ml_away"])
        if mu is not None and model is not None:   # the model's pull, carried through the win curve
            ph += pricer.wins.home(mu + pricer.k_spread * (model - mu)) - pricer.wins.home(mu)
        ph = min(max(ph, 0.001), 0.999)
        return (ph if pk["side"] == "home" else 1 - ph), 0.0
    if mu is None:
        return None
    if model is not None:
        mu += (pricer.k_spread if m == "spread" else pricer.k_total) * (model - mu)
    if m == "spread":
        over, push, under = pricer.mdist.side_probs(mu, -pk["line"] if pk["side"] == "home" else pk["line"])
    else:
        over, push, under = pricer.tdist.side_probs(mu, pk["line"])
    return (over if pk["side"] in ("home", "over") else under), push


def _pin_txt(pk: dict, row: dict, p: float | None) -> str:
    if pk["market"] == "spread" and "spread" in row:
        return fmt_line(-row["spread"] if pk["side"] == "home" else row["spread"])
    if pk["market"] == "total" and "total" in row:
        return f"{row['total']:g}"
    return f"{100 * p:.0f}%" if p is not None else ""


def qb_doubts(season: int, week: int) -> dict[str, list[str]]:
    """Team -> its quarterbacks listed Questionable or Doubtful on that week's injury report."""
    path = RAW / "injuries" / f"{season}.parquet"
    if not path.exists():
        return {}
    i = pl.read_parquet(path).filter((pl.col("season") == season) & (pl.col("week") == week)
                                     & (pl.col("position") == "QB") & pl.col("report_status").is_in(list(C.UNSURE)))
    out: dict[str, list[str]] = {}
    for r in i.iter_rows(named=True):
        out.setdefault(r["team"], []).append(f"{r['team']} QB {r['full_name']} is {r['report_status']}.")
    return out


def now_price(pk: dict, g: dict, L: Lines, pricer: Pricer, when: datetime, model_margin: float,
              model_total: float) -> dict | None:
    """The best DraftKings/FanDuel price on the pick's side at `when`, and its EV by the board's own pricing."""
    books = L.all_at(when, ["pinnacle"] + SHOP_BOOKS)
    ref = reference(books, {"spread": g["spread_line"], "total": g["total_line"]})
    best = None
    for b in SHOP_BOOKS:
        r = books.get(b, {})
        home = pk["side"] == "home"
        if pk["market"] == "spread" and "spread" in r and ref.get("spread") is not None:
            over, push, under = pricer.mdist.side_probs(pricer.expected_margin(model_margin, ref["spread"]), r["spread"])
            p, dec = (over, r["spread_price_a"]) if home else (under, r["spread_price_b"])
            label = f"{pk['team']} {fmt_line(-r['spread'] if home else r['spread'])}"
        elif pk["market"] == "total" and "total" in r and ref.get("total") is not None:
            over, push, under = pricer.tdist.side_probs(pricer.expected_total(model_total, ref["total"]), r["total"])
            p, dec = (over, r["total_price_a"]) if pk["side"] == "over" else (under, r["total_price_b"])
            label = f"{pk['side'].title()} {r['total']:g}"
        elif pk["market"] == "ml" and "ml_home" in r and ref.get("spread") is not None:
            ph = pricer.wins.home(pricer.expected_margin(model_margin, ref["spread"]))
            p, dec, push = (ph, r["ml_home"], 0.0) if home else (1 - ph, r["ml_away"], 0.0)
            label = f"{pk['team']} ML"
        else:
            continue
        ev = p * (dec - 1) - (1 - p - push)
        if best is None or ev > best["ev"]:
            best = {"book": BOOK_NAMES.get(b, b), "label": label, "american": american(dec), "ev": round(ev, 4)}
    return best


def game_checks(pk: dict, g: dict, L: Lines, pricer: Pricer, as_of: datetime, qb: dict[str, list[str]],
                model: dict | None = None) -> dict:
    """Checks, verdict and (before kickoff, given the model's current numbers) the price now."""
    pin_then = L.at("pinnacle", datetime.fromisoformat(pk["locked_at"]))
    pin_now = L.at("pinnacle", as_of)
    bt = bet_chance(pricer, pk, pin_then) if pin_then else None          # the market alone, for its move
    bn = bet_chance(pricer, pk, pin_now) if pin_now else None
    model_then = pk.get("model_total") if pk["market"] == "total" else pk.get("model_margin")
    bk = bet_chance(pricer, pk, pin_then, model_then) if pin_then else None   # the board's method, juice counted
    ev_sharp = None if bk is None else bk[0] * (pk["dec"] - 1) - (1 - bk[0] - bk[1])
    p_then = None if bt is None else bt[0] / max(1 - bt[1], 1e-9)
    p_now = None if bn is None else bn[0] / max(1 - bn[1], 1e-9)
    checks = [C.sharp(ev_sharp), C.size(pk["ev"]),
              C.market(p_then, p_now, _pin_txt(pk, pin_then, p_then) if pin_then else "",
                       _pin_txt(pk, pin_now, p_now) if pin_now else ""),
              C.who(qb.get(g["home_team"], []) + qb.get(g["away_team"], []), "No quarterback on either team is in doubt.")]
    out = {"verdict": C.verdict(checks), "checks": checks}
    if model and as_of < g["kickoff_utc"]:
        out["now"] = now_price(pk, g, L, pricer, as_of, model["model_margin"], model["model_total"])
    return out
