"""Player props against a full season of closing lines (SportsGameOdds archive, 2025 + 2026 so far).

    python -m props.sgo_study

The 64-game sample this season could not tell skill from luck. SportsGameOdds
(pipeline/sgo.py) adds the whole 2025 regular season. What it holds decides
what can be tested (pipeline/sgo_tidy.py):

- 2025: each book's last quote. A book that pulls its props at kickoff leaves
  its close behind (checked on 2026, where SGO stores real closes: a last quote
  before kickoff equals the close 99% of the time). Pinnacle and bet365 do, on
  ~62% of player-markets; DraftKings and FanDuel trade in-game, so their 2025
  closes are gone. bet365 stands in for them: a US retail book with similar
  prop pricing.
- 2026 weeks 1-5: real per-book opens and closes, DraftKings and FanDuel included.
- Nothing usable in 2024 (pregame quotes for ~2% of props).

2025 is holdout for the props model (tuned on 2018-2021), so it is a clean test.
All of it is at closing prices, the hardest test there is: the board locks
earlier (Thursday/Saturday 10am ET), when lines are softer but noisier.

Questions:
1. Is the closing market itself biased (unders, long shots)? Pinnacle no-vig
   vs results; blind overs/unders at bet365 and Pinnacle prices.
2. Does the model know anything the close does not? Log loss of
   location = market + k * (model - market) as k moves from 0 (market) to 1
   (model), fitted on 2025, checked on 2026; plus a logistic check.
3. Would the board's rule have made money at the close? Price each prop at
   the retail book (bet365 in 2025; DraftKings and FanDuel in 2026) with the
   market location from Pinnacle: k = 0 is pure line shopping, k = the fitted
   value adds the model, k = 1 is the model alone. Units at 1u a bet, game-
   clustered bootstrap intervals.

Writes reports/props_sgo_study.json (for the board) and reports/props_sgo_rows.parquet.
"""
import json
from datetime import timedelta

import numpy as np
import polars as pl

from config import REPORTS
from pipeline.sgo import ROOT as SGO
from props.backtest import ROLE_MARKETS, frames, load
from props.model import STAT_COL, Simulator
from props.pricing import shift_to

GRACE = timedelta(minutes=2)       # a quote this soon after the listed start still matches the close 98% of the time
SIZE = 2000                        # simulated games per player-market
RETAIL = {2025: ("bet365",), 2026: ("draftkings", "fanduel")}
OUT = REPORTS / "props_sgo_study.json"
ROWS = REPORTS / "props_sgo_rows.parquet"
KS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
THRESH = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
BOARD_K = 0.005


def dec(am: pl.Expr) -> pl.Expr:
    return pl.when(am > 0).then(1 + am / 100).otherwise(1 + 100 / (-am))


def two_way(Q: pl.DataFrame, when: str) -> pl.DataFrame:
    """One row per (game, player, market, book): the line both sides were priced at, the
    decimal prices and the no-vig chance of the over, at the open or the close."""
    k = ["season", "game_id", "sgo_player", "market", "book"]
    if when == "close":
        last = Q.filter((pl.col("season") < 2026) & (pl.col("last_at") < pl.col("starts_at") + GRACE)).select(
            k + ["side", "line", "odds"])
        real = Q.filter((pl.col("season") >= 2026) & pl.col("close_line").is_not_null()).select(
            k + ["side", pl.col("close_line").alias("line"), pl.col("close_odds").alias("odds")])
        X = pl.concat([last, real])
    else:
        X = Q.filter(pl.col("open_line").is_not_null()).select(
            k + ["side", pl.col("open_line").alias("line"), pl.col("open_odds").alias("odds")])
    X = X.filter(pl.col("odds").is_not_null() & pl.col("line").is_not_null())
    o = X.filter(pl.col("side") == "over").select(k + ["line", pl.col("odds").alias("o_am")])
    u = X.filter(pl.col("side") == "under").select(k + ["line", pl.col("odds").alias("u_am")])
    P = o.join(u, on=k + ["line"]).with_columns(over=dec(pl.col("o_am")), under=dec(pl.col("u_am")))
    P = P.with_columns(p_over=(1 / pl.col("over")) / (1 / pl.col("over") + 1 / pl.col("under")),
                       hold=1 / pl.col("over") + 1 / pl.col("under") - 1)
    # a sane two-way main line: hold between 0 and 15%
    return P.filter(pl.col("hold").is_between(-0.005, 0.15)).unique(k, keep="first").with_columns(when=pl.lit(when))


def samples(keys: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Walk-forward projections for the tracked players in `keys` (game_id, gsis_id), and
    sorted simulated samples per (game_id, gsis_id, market)."""
    p, disp = load()
    x = frames(p).filter(pl.col("role").is_not_null())
    x = x.join(keys.unique(), on=["game_id", "gsis_id"])
    sim = Simulator(disp, seed=17)
    S = {}
    for i in range(0, x.height, 800):
        c = x.slice(i, 800)
        draws = sim.simulate(c, SIZE)
        gid, pid, role = c["game_id"].to_list(), c["gsis_id"].to_list(), c["role"].to_list()
        for j in range(c.height):
            for m in ROLE_MARKETS[role[j]]:
                S[(gid[j], pid[j], m)] = np.sort(draws[m][j]).astype(np.float32)
    # QB rushing only where the model expects a running QB (as in props.backtest.long_form)
    proj = x.select("game_id", "gsis_id", "role", "position", "name", "team", "rush_yds_hat")
    return proj, S


def p_over_at(s: np.ndarray, line: float, shift: float = 0.0) -> tuple[float, float]:
    """(over, under) chances for samples moved by `shift`; whole-number lines can push."""
    n = len(s)
    over = 1 - np.searchsorted(s, line - shift + 1e-9, side="right") / n
    under = np.searchsorted(s, line - shift - 1e-9, side="left") / n
    return float(over), float(under)


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def boot(units: np.ndarray, games: np.ndarray, n: int = 2000, seed: int = 1) -> tuple[float, float]:
    """95% interval for total units, resampling whole games."""
    if len(units) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(games, return_inverse=True)
    per = np.bincount(inv, weights=units)
    tot = per[rng.integers(0, len(ug), size=(n, len(ug)))].sum(axis=1)
    return float(np.quantile(tot, 0.025)), float(np.quantile(tot, 0.975))


def _grade_under(df: pl.DataFrame, price: str = "under") -> pl.DataFrame:
    return df.with_columns(result=pl.when(pl.col("actual") == pl.col("line")).then(pl.lit("P"))
                           .when(pl.col("actual") < pl.col("line")).then(pl.lit("W")).otherwise(pl.lit("L"))).with_columns(
        units=pl.when(pl.col("result") == "P").then(0.0).when(pl.col("result") == "W").then(pl.col(price) - 1).otherwise(-1.0))


def _tally(df: pl.DataFrame) -> dict:
    if df.height == 0:
        return {"n": 0}
    u, g = df["units"].to_numpy(), df["game_id"].to_numpy()
    lo, hi = boot(u, g)
    w, l = int((df["result"] == "W").sum()), int((df["result"] == "L").sum())
    return {"n": df.height, "w": w, "l": l, "p": df.height - w - l, "units": float(u.sum()), "roi": float(u.mean()),
            "ci_lo": lo, "ci_hi": hi, "won": w / max(1, w + l), "implied": float((1 / df["under"]).mean()) if "under" in df.columns else None}


def plus_unders(Q: pl.DataFrame, M: pl.DataFrame) -> dict:
    """Unders priced at even money or longer: the book has made the over the favourite.
    Every mapped player (not just tracked roles), one bet per player-market at the
    better of the user's books (2026: DraftKings/FanDuel; 2025: bet365 stands in)."""
    allL = two_way(Q, "close")
    allO = two_way(Q, "open")
    pg = pl.read_parquet(SGO.parent / "clean" / "props_player_games.parquet").select("game_id", "gsis_id", *STAT_COL.values())

    def attach(X):
        X = X.join(M.select("game_id", "sgo_player", "market", "gsis_id"), on=["game_id", "sgo_player", "market"])
        X = X.join(pg, on=["game_id", "gsis_id"]).with_columns(
            actual=pl.when(pl.col("market") == "pass_yds").then(pl.col("passing_yards"))
            .when(pl.col("market") == "rush_yds").then(pl.col("rushing_yards"))
            .when(pl.col("market") == "rec_yds").then(pl.col("receiving_yards"))
            .otherwise(pl.col("receptions")).cast(pl.Float64)).drop(*STAT_COL.values())
        return _grade_under(X.filter(pl.col("actual").is_not_null()))

    C, O = attach(allL), attach(allO)
    res = {"by_book": [], "user_books": {}, "bands": []}
    for (season, book), g in C.filter(pl.col("under") >= 2.0).group_by("season", "book"):
        if g.height >= 150 and book not in ("prizepicks", "underdog"):   # pick'em sites have no real prices
            res["by_book"].append({"season": season, "book": book, **_tally(g)})
    res["by_book"].sort(key=lambda r: (r["season"], -r["roi"]))
    others = C.filter((pl.col("under") < 2.0) & ~pl.col("book").is_in(["prizepicks", "underdog"]))
    res["other_unders"] = [{"season": s, **_tally(g)} for (s,), g in others.group_by("season")]

    def user(X, season, books):
        b = X.filter((pl.col("season") == season) & pl.col("book").is_in(list(books)) & (pl.col("under") >= 2.0))
        return b.sort("under", descending=True).group_by("game_id", "gsis_id", "market", maintain_order=True).first()

    for tag, X, season, books in (("2025 close (bet365)", C, 2025, ("bet365",)), ("2026 close (DK/FD)", C, 2026, RETAIL[2026]),
                                  ("2026 open (DK/FD)", O, 2026, RETAIL[2026])):
        b = user(X, season, books)
        res["user_books"][tag] = {**_tally(b), "by_market": {m: _tally(g) for (m,), g in b.group_by("market")}}
        bands = b.with_columns(band=pl.when(pl.col("under") < 2.10).then(pl.lit("+100 to +109"))
                               .when(pl.col("under") < 2.25).then(pl.lit("+110 to +124")).otherwise(pl.lit("+125 or longer")))
        for (band,), g in bands.group_by("band"):
            res["bands"].append({"tag": tag, "band": band, **_tally(g)})
    print("plus-money unders:", json.dumps(res["user_books"], default=float, indent=1)[:3000])
    return res


def open_shopping(L: pl.DataFrame, S: dict) -> dict:
    """2026 only: the board's rule (Pinnacle's no-vig location, DK/FD price) at the OPEN, graded,
    with line value: the bet's chance at Pinnacle's close vs its break-even."""
    Lo = L.filter((pl.col("season") == 2026) & (pl.col("when") == "open"))
    Lc = L.filter((pl.col("season") == 2026) & (pl.col("when") == "close") & (pl.col("book") == "pinnacle"))
    pin_o = {(r["game_id"], r["gsis_id"], r["market"]): r for r in Lo.filter(pl.col("book") == "pinnacle").to_dicts()}
    pin_c = {(r["game_id"], r["gsis_id"], r["market"]): r for r in Lc.to_dicts()}
    bets = []
    for r in Lo.filter(pl.col("book").is_in(list(RETAIL[2026]))).to_dicts():
        key = (r["game_id"], r["gsis_id"], r["market"])
        po, pc = pin_o.get(key), pin_c.get(key)
        if po is None or pc is None or key not in S:
            continue
        s = S[key]
        d_open, d_close = shift_to(s, po["line"], po["p_over"]), shift_to(s, pc["line"], pc["p_over"])
        for k in (0.0, BOARD_K):
            ov, un = p_over_at(s, r["line"], (1 - k) * d_open)
            ovc, unc = p_over_at(s, r["line"], d_close)
            for side, p, pcl, price in (("over", ov, ovc, r["over"]), ("under", un, unc, r["under"])):
                push = max(0.0, 1 - ov - un)
                ev = p * (price - 1) - (1 - p - push)
                ev_close = pcl * (price - 1) - (1 - pcl - max(0.0, 1 - ovc - unc))
                won = r["actual"] > r["line"] if side == "over" else r["actual"] < r["line"]
                res = "P" if r["actual"] == r["line"] else ("W" if won else "L")
                bets.append({"k": k, "game_id": r["game_id"], "gsis_id": r["gsis_id"], "market": r["market"], "side": side,
                             "book": r["book"], "ev": ev, "ev_close": ev_close, "result": res,
                             "units": 0.0 if res == "P" else (price - 1 if won else -1.0)})
    if not bets:
        return {}
    B = pl.DataFrame(bets).sort("ev", descending=True).group_by("k", "game_id", "gsis_id", "market", maintain_order=True).first()
    out = []
    for (k,), g in B.group_by("k"):
        for t in (0.0, 0.02, 0.04, 0.06):
            b = g.filter(pl.col("ev") >= t)
            if b.height:
                out.append({"k": k, "min_ev": t, **_tally(b), "ev_at_close": float(b["ev_close"].mean()),
                            "beat_close": float((b["ev_close"] > 0).mean())})
    print("2026 open, board rule:\n", pl.DataFrame(out).sort("k", "min_ev"))
    return {"rows": out}


def lock_checks(S: dict) -> dict:
    """2026 weeks so far, from the OddsPapi timelines: DK/FD main lines at each game's real
    lock time (props.pricing.lock_time), plus-money unders and the board's rule, graded, with
    the same book's close for line value."""
    from props.lines import load as load_lines, main_lines
    from props.pricing import lock_time

    pg = pl.read_parquet(SGO.parent / "clean" / "props_player_games.parquet").filter(pl.col("season") == 2026)
    games = pg["game_id"].unique().to_list()
    P = load_lines(games)
    if P.height == 0:
        return {}
    kick = P.select("game_id", "kickoff").unique()
    kick = kick.with_columns(lock=pl.col("kickoff").map_elements(lock_time, return_dtype=pl.Datetime("us", "UTC")))
    P = P.join(kick.select("game_id", "lock"), on="game_id")
    Lk = main_lines(P, pl.col("lock"))
    Cl = main_lines(P, pl.col("kickoff") - timedelta(minutes=5))
    act = pg.select("game_id", "gsis_id", *STAT_COL.values())
    A = Lk.join(act, on=["game_id", "gsis_id"]).with_columns(
        actual=pl.when(pl.col("market") == "pass_yds").then(pl.col("passing_yards"))
        .when(pl.col("market") == "rush_yds").then(pl.col("rushing_yards"))
        .when(pl.col("market") == "rec_yds").then(pl.col("receiving_yards"))
        .otherwise(pl.col("receptions")).cast(pl.Float64)).drop(*STAT_COL.values()).filter(pl.col("actual").is_not_null())
    close = Cl.select("game_id", "gsis_id", "market", "book", pl.col("line").alias("c_line"), pl.col("under").alias("c_under"),
                      pl.col("p_over").alias("c_p_over"))
    pu = A.filter(pl.col("book").is_in(list(RETAIL[2026])) & (pl.col("under") >= 2.0))
    pu = pu.sort("under", descending=True).group_by("game_id", "gsis_id", "market", maintain_order=True).first()
    pu = _grade_under(pu).join(close, on=["game_id", "gsis_id", "market", "book"], how="left")
    # line value for an under: the line came down, or the same line's under got shorter
    pu = pu.with_columns(clv=pl.when(pl.col("c_line") < pl.col("line")).then(1).when(pl.col("c_line") > pl.col("line")).then(-1)
                         .when(pl.col("c_under") < pl.col("under")).then(1).when(pl.col("c_under") > pl.col("under")).then(-1).otherwise(0))
    res = {"plus_unders": {**_tally(pu), "clv_better": int((pu["clv"] == 1).sum()), "clv_worse": int((pu["clv"] == -1).sum()),
                           "by_market": {m: _tally(g) for (m,), g in pu.group_by("market")}}}
    print("2026 at lock time, plus-money unders (DK/FD):", json.dumps(res["plus_unders"], default=float)[:800])
    # the board's rule at lock, k = 0 and the board's k, tracked players with samples
    pin = {(r["game_id"], r["gsis_id"], r["market"]): r for r in A.filter(pl.col("book") == "pinnacle").to_dicts()}
    pinc = {(r["game_id"], r["gsis_id"], r["market"]): r for r in Cl.filter(pl.col("book") == "pinnacle").to_dicts()}
    bets = []
    for r in A.filter(pl.col("book").is_in(list(RETAIL[2026]))).to_dicts():
        key = (r["game_id"], r["gsis_id"], r["market"])
        if key not in S or key not in pin:
            continue
        s = S[key]
        d = shift_to(s, pin[key]["line"], pin[key]["p_over"])
        pc = pinc.get(key)
        oc = uc = None
        if pc is not None:
            oc, uc = p_over_at(s, r["line"], shift_to(s, pc["line"], pc["p_over"]))
        for k in (0.0, BOARD_K):
            ov, un = p_over_at(s, r["line"], (1 - k) * d)
            push = max(0.0, 1 - ov - un)
            for side, p, price in (("over", ov, r["over"]), ("under", un, r["under"])):
                ev = p * (price - 1) - (1 - p - push)
                ev_close = None
                if oc is not None:
                    q = oc if side == "over" else uc
                    ev_close = q * (price - 1) - (1 - q - max(0.0, 1 - oc - uc))
                won = r["actual"] > r["line"] if side == "over" else r["actual"] < r["line"]
                rr = "P" if r["actual"] == r["line"] else ("W" if won else "L")
                bets.append({"k": k, "game_id": r["game_id"], "gsis_id": r["gsis_id"], "market": r["market"], "side": side,
                             "ev": ev, "ev_close": ev_close, "result": rr, "units": 0.0 if rr == "P" else (price - 1 if won else -1.0)})
    B = pl.DataFrame(bets).sort("ev", descending=True).group_by("k", "game_id", "gsis_id", "market", maintain_order=True).first()
    res["board_rule"] = []
    for (k,), g in B.group_by("k"):
        for t in (0.0, 0.02, 0.04):
            b = g.filter(pl.col("ev") >= t)
            if b.height:
                ec = b["ev_close"].drop_nulls()
                res["board_rule"].append({"k": k, "min_ev": t, **_tally(b),
                                          "beat_close": float((ec > 0).mean()) if ec.len() else None,
                                          "unders": _tally(b.filter(pl.col("side") == "under")),
                                          "overs": _tally(b.filter(pl.col("side") == "over"))})
    print("2026 at lock time, the board's rule:", json.dumps(res["board_rule"], default=float)[:2500])
    return res


def lock_2025() -> dict:
    """2025 at the board's lock time, from Pinnacle's rebuilt line history (props/line_history.py,
    98.9% of lock-time lines right when checked on 2026). Does the model know where Pinnacle's line
    goes between the lock and kickoff (k), and does mixing the model in beat Pinnacle's lock-time
    number on results? Pinnacle only: DraftKings', FanDuel's and bet365's histories do not rebuild."""
    from props.line_history import entries, line_at
    from props.pricing import lock_time

    M = pl.read_parquet(SGO / "props.parquet").filter(pl.col("gsis_id").is_not_null()).select(
        "game_id", "sgo_player", "market", "gsis_id")
    E = entries().filter((pl.col("season") == 2025) & (pl.col("book") == "pinnacle"))
    kick = E.select("game_id", "starts_at").unique().with_columns(
        T=pl.col("starts_at").map_elements(lock_time, return_dtype=pl.Datetime("us", "UTC")))
    lk = line_at(E, kick.select("game_id", "T")).filter(pl.col("price_known")).join(M, on=["game_id", "sgo_player", "market"])
    Q = pl.read_parquet(SGO / "prop_quotes.parquet").filter(pl.col("season") == 2025)
    cl = two_way(Q, "close").filter(pl.col("book") == "pinnacle").join(M, on=["game_id", "sgo_player", "market"])
    J = lk.select("game_id", "gsis_id", "market", pl.col("line").alias("l_line"), pl.col("p_over").alias("l_p"),
                  pl.col("over").alias("l_over"), pl.col("under").alias("l_under")).join(
        cl.select("game_id", "gsis_id", "market", pl.col("line").alias("c_line"), pl.col("p_over").alias("c_p")),
        on=["game_id", "gsis_id", "market"])
    pg = pl.read_parquet(SGO.parent / "clean" / "props_player_games.parquet").select("game_id", "gsis_id", *STAT_COL.values())
    J = J.join(pg, on=["game_id", "gsis_id"]).with_columns(
        actual=pl.when(pl.col("market") == "pass_yds").then(pl.col("passing_yards"))
        .when(pl.col("market") == "rush_yds").then(pl.col("rushing_yards"))
        .when(pl.col("market") == "rec_yds").then(pl.col("receiving_yards"))
        .otherwise(pl.col("receptions")).cast(pl.Float64)).drop(*STAT_COL.values()).filter(pl.col("actual").is_not_null())
    proj, S = samples(J.select("game_id", "gsis_id").unique())
    J = J.join(proj.select("game_id", "gsis_id", "role", "rush_yds_hat"), on=["game_id", "gsis_id"])
    J = J.filter(~((pl.col("market") == "rush_yds") & (pl.col("role") == "QB1") & (pl.col("rush_yds_hat") < 12)))
    rows = []
    for r in J.to_dicts():
        s = S.get((r["game_id"], r["gsis_id"], r["market"]))
        if s is None:
            continue
        sd = max(float(s.std()), 1e-6)
        d_lock, d_close = shift_to(s, r["l_line"], r["l_p"]), shift_to(s, r["c_line"], r["c_p"])
        row = {"game_id": r["game_id"], "market": r["market"], "role": r["role"],
               "gap": -d_lock / sd,                      # model minus market at the lock, in model SDs
               "move": (d_close - d_lock) / sd}          # market's move from lock to close, in model SDs
        if r["actual"] != r["l_line"]:
            row["y"] = float(r["actual"] > r["l_line"])
            row["l_p"] = r["l_p"]
            for k in KS:
                row[f"k{k}"] = p_over_at(s, r["l_line"], (1 - k) * d_lock)[0]
        rows.append(row)
    D = pl.DataFrame(rows)
    g, mv = D["gap"].to_numpy(), D["move"].to_numpy()
    moved = np.abs(mv) > 1e-9
    k = float((g * mv).sum() / (g ** 2).sum())
    # bootstrap k by game
    rng = np.random.default_rng(3)
    gid = D["game_id"].to_numpy()
    ug, inv = np.unique(gid, return_inverse=True)
    gm_gg, gm_gm = np.bincount(inv, weights=g * g), np.bincount(inv, weights=g * mv)
    ks = []
    for _ in range(2000):
        idx = rng.integers(0, len(ug), len(ug))
        ks.append(gm_gm[idx].sum() / gm_gg[idx].sum())
    big = np.abs(g) > 0.25
    res = {"n": int(D.height), "moved": int(moved.sum()),
           "toward": int((np.sign(g[moved]) == np.sign(mv[moved])).sum()),
           "k": k, "k_lo": float(np.quantile(ks, 0.025)), "k_hi": float(np.quantile(ks, 0.975)),
           "big_gap_moved": int((big & moved).sum()), "big_gap_toward": int((big & moved & (np.sign(g) == np.sign(mv))).sum())}
    Y = D.filter(pl.col("y").is_not_null())
    y = Y["y"].to_numpy()
    res["logloss"] = {"n": Y.height, "market_at_lock": logloss(Y["l_p"].to_numpy(), y),
                      **{f"k{kk}": logloss(Y[f"k{kk}"].to_numpy(), y) for kk in KS}}
    res["by_market"] = []
    for (m,), gdf in D.group_by("market"):
        gg, mm = gdf["gap"].to_numpy(), gdf["move"].to_numpy()
        mo = np.abs(mm) > 1e-9
        res["by_market"].append({"market": m, "n": gdf.height, "moved": int(mo.sum()),
                                 "toward": int((np.sign(gg[mo]) == np.sign(mm[mo])).sum()),
                                 "k": float((gg * mm).sum() / (gg ** 2).sum())})
    print("2025 lock -> close at Pinnacle:", json.dumps(res, default=float, indent=1))
    return res


def board_summary(out: dict, best: pl.DataFrame) -> dict:
    """What the board's props "How it works" page shows: the board's rule (k = BOARD_K, any positive
    EV) by when the price was taken, plus-money unders, how overs are priced, and model vs market."""
    def pick(rows, **f):
        return next((r for r in rows if all(abs(r.get(k, -9) - v) < 1e-9 if isinstance(v, float) else r.get(k) == v
                                            for k, v in f.items())), None)

    def slim(r, **extra):
        if not r or not r.get("n"):
            return None
        keep = ("n", "w", "l", "p", "units", "roi", "ci_lo", "ci_hi", "won", "implied")
        return {**{k: r[k] for k in keep if k in r}, **extra}

    b25 = best.filter((pl.col("season") == 2025) & (pl.col("k") == BOARD_K) & (pl.col("ev") >= 0))
    side25 = {s: slim(_tally(g)) for (s,), g in b25.group_by("side")}
    close25 = pick(out["sims"], season=2025, k=BOARD_K, min_ev=0.0)
    close26 = pick(out["sims"], season=2026, k=BOARD_K, min_ev=0.0)
    open26 = pick(out["open_shop"].get("rows", []), k=BOARD_K, min_ev=0.0)
    lock26 = pick(out["lock"].get("board_rule", []), k=BOARD_K, min_ev=0.0)
    shop = [
        {"when": "2026 opens", "books": "DK/FD vs Pinnacle", **(slim(open26, beat_close=(open26 or {}).get("beat_close")) or {})},
        {"when": "2026 at lock", "books": "DK/FD vs Pinnacle",
         **(slim(lock26, beat_close=(lock26 or {}).get("beat_close")) or {})},
        {"when": "2026 closes", "books": "DK/FD vs Pinnacle", **(slim(close26) or {})},
        {"when": "2025 closes", "books": "bet365 vs Pinnacle", **(slim(close25) or {}),
         "overs": side25.get("over"), "unders": side25.get("under")},
    ]
    pu = out["plus_unders"]["user_books"]
    lk = out["lock"].get("plus_unders", {})
    plus = [{"when": "2025 closes", "books": "bet365", **(slim(pu.get("2025 close (bet365)")) or {})},
            {"when": "2026 opens", "books": "DK/FD", **(slim(pu.get("2026 open (DK/FD)")) or {})},
            {"when": "2026 at lock", "books": "DK/FD", **(slim(lk) or {}),
             "clv_better": lk.get("clv_better"), "clv_worse": lk.get("clv_worse")},
            {"when": "2026 closes", "books": "DK/FD", **(slim(pu.get("2026 close (DK/FD)")) or {})}]
    others = {r["season"]: r["roi"] for r in out["plus_unders"].get("other_unders", [])}
    overs = [{"season": r["season"], "market": r["market"], "n": r["n"], "said": r["said_over"], "hit": r["over_hit"]}
             for r in out["card"] if r["book"] == "pinnacle"]
    kc = {r["season"]: r for r in out["k_curve"]}
    lg = {int(k): v for k, v in out["logistic"].items()}  # int keys turn into strings in the saved JSON
    model = [{"season": s, "n": kc[s]["n"], "market": kc[s]["k0.0"], "model": kc[s]["k1.0"],
              "best_k": min(KS, key=lambda k: kc[s][f"k{k}"]), "best": min(kc[s][f"k{k}"] for k in KS),
              "gap_coef": lg[s]["model_gap"], "gap_se": lg[s]["model_gap_se"]} for s in (2025, 2026) if s in kc]
    l25 = out.get("lock_2025") or {}
    lock25 = ({k: l25[k] for k in ("n", "moved", "toward", "k", "k_lo", "k_hi", "big_gap_moved", "big_gap_toward")}
              if l25.get("n") else None)
    return {"shop": shop, "plus_unders": plus, "other_unders_roi": others, "overs": overs, "model": model,
            "lock_2025": lock25, "source": out["source"]}


def main() -> None:
    Q = pl.read_parquet(SGO / "prop_quotes.parquet").filter(pl.col("season") >= 2025)
    M = pl.read_parquet(SGO / "props.parquet").filter((pl.col("season") >= 2025) & pl.col("gsis_id").is_not_null())
    C = two_way(Q, "close")
    O = two_way(Q, "open")
    L = pl.concat([C, O]).join(M.select("game_id", "sgo_player", "market", "gsis_id", "week", "team"),
                               on=["game_id", "sgo_player", "market"])
    pg = pl.read_parquet(SGO.parent / "clean" / "props_player_games.parquet").select(
        "game_id", "gsis_id", *STAT_COL.values())
    L = L.join(pg, on=["game_id", "gsis_id"], how="left").with_columns(
        actual=pl.when(pl.col("market") == "pass_yds").then(pl.col("passing_yards"))
        .when(pl.col("market") == "rush_yds").then(pl.col("rushing_yards"))
        .when(pl.col("market") == "rec_yds").then(pl.col("receiving_yards"))
        .otherwise(pl.col("receptions")).cast(pl.Float64)).drop(*STAT_COL.values())
    L = L.filter(pl.col("actual").is_not_null())
    print(f"{L.height} two-way book lines with results ({L.filter(pl.col('when') == 'close').height} at the close)")
    out = {"source": "SportsGameOdds archive (Pro trial), pulled 2026-10-09", "grace_min": GRACE.total_seconds() / 60}

    # ---------------------------------------------------------------- 1. the closing market itself
    cl = L.filter((pl.col("when") == "close") & (pl.col("actual") != pl.col("line"))).with_columns(
        y=(pl.col("actual") > pl.col("line")).cast(pl.Float64))
    cl = cl.with_columns(roi_over=pl.when(pl.col("y") == 1).then(pl.col("over") - 1).otherwise(-1.0),
                         roi_under=pl.when(pl.col("y") == 0).then(pl.col("under") - 1).otherwise(-1.0))
    card = cl.filter(pl.col("book").is_in(["pinnacle", "bet365", "draftkings", "fanduel"])).group_by("season", "book", "market").agg(
        n=pl.len(), said_over=pl.col("p_over").mean(), over_hit=pl.col("y").mean(), hold=pl.col("hold").mean(),
        roi_over=pl.col("roi_over").mean(), roi_under=pl.col("roi_under").mean()).sort("season", "book", "market")
    print("closing-line report card:\n", card)
    pin = cl.filter(pl.col("book") == "pinnacle").with_columns(
        bucket=(pl.col("p_over") * 10).floor().clip(1, 8) / 10)
    calib = pin.group_by("season", "bucket").agg(n=pl.len(), said=pl.col("p_over").mean(), hit=pl.col("y").mean()).sort("season", "bucket")
    print("Pinnacle no-vig calibration (over chance buckets):\n", calib)
    out["card"] = card.to_dicts()
    out["calibration"] = calib.to_dicts()

    # ---------------------------------------------------------------- model samples
    keys = L.select("game_id", "gsis_id").unique()
    proj, S = samples(keys)
    print(f"{len(S)} simulated player-markets for tracked roles")
    L = L.join(proj, on=["game_id", "gsis_id"], how="inner")
    L = L.filter(~((pl.col("market") == "rush_yds") & (pl.col("role") == "QB1") & (pl.col("rush_yds_hat") < 12)))
    L = L.filter(pl.struct("game_id", "gsis_id", "market").map_elements(
        lambda r: (r["game_id"], r["gsis_id"], r["market"]) in S, return_dtype=pl.Boolean))

    # ---------------------------------------------------------------- 2. model vs the close (Pinnacle)
    P = L.filter((pl.col("when") == "close") & (pl.col("book") == "pinnacle")).to_dicts()
    rows = []
    for r in P:
        s = S[(r["game_id"], r["gsis_id"], r["market"])]
        d = shift_to(s, r["line"], r["p_over"])          # model -> market location
        if r["actual"] == r["line"]:
            continue
        y = float(r["actual"] > r["line"])
        po = [p_over_at(s, r["line"], (1 - k) * d)[0] for k in KS]
        # renormalise over/under so a push-able line still compares like with like
        rows.append({"season": r["season"], "game_id": r["game_id"], "gsis_id": r["gsis_id"], "market": r["market"],
                     "role": r["role"], "week": r["week"], "line": r["line"], "y": y, "p_mkt": r["p_over"],
                     "d_sd": d / max(float(s.std()), 1e-6), "model_med": float(np.median(s)),
                     **{f"k{k}": v for k, v in zip(KS, po)}})
    D = pl.DataFrame(rows)
    D.write_parquet(REPORTS / "props_sgo_pin.parquet")
    L.write_parquet(REPORTS / "props_sgo_lines.parquet")
    fit = []
    for season in (2025, 2026):
        Ds = D.filter(pl.col("season") == season)
        y = Ds["y"].to_numpy()
        fit.append({"season": season, "n": Ds.height, "market_logloss": logloss(Ds["p_mkt"].to_numpy(), y),
                    **{f"k{k}": logloss(Ds[f"k{k}"].to_numpy(), y) for k in KS}})
    F = pl.DataFrame(fit)
    print("log loss at Pinnacle's closing line by k (k=0: the market, k=1: the model):\n", F)
    by_mkt = []
    for (season, m), g in D.group_by("season", "market"):
        y = g["y"].to_numpy()
        by_mkt.append({"season": season, "market": m, "n": g.height, "market": m,
                       **{f"k{k}": logloss(g[f"k{k}"].to_numpy(), y) - logloss(g["k0.0"].to_numpy(), y) for k in KS}})
    print("log loss vs market by market (negative = better than the market):\n", pl.DataFrame(by_mkt).sort("season", "market"))
    k25 = F.filter(pl.col("season") == 2025).row(0, named=True)
    k_fit = min(KS, key=lambda k: k25[f"k{k}"])
    # logistic check: does the model's disagreement predict the result beyond the market?
    import math
    lg = lambda p: math.log(max(min(p, 1 - 1e-4), 1e-4) / (1 - max(min(p, 1 - 1e-4), 1e-4)))
    D2 = D.with_columns(xm=pl.col("p_mkt").map_elements(lg, return_dtype=pl.Float64),
                        xd=pl.col("k1.0").map_elements(lg, return_dtype=pl.Float64) - pl.col("p_mkt").map_elements(lg, return_dtype=pl.Float64))
    logit = {}
    for season in (2025, 2026):
        g = D2.filter(pl.col("season") == season)
        X = np.column_stack([np.ones(g.height), g["xm"].to_numpy(), g["xd"].to_numpy()])
        yv = g["y"].to_numpy()
        b = np.zeros(3)
        for _ in range(50):  # Newton's method
            pr = 1 / (1 + np.exp(-X @ b))
            W = pr * (1 - pr)
            H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(3)
            b += np.linalg.solve(H, X.T @ (yv - pr))
        se = np.sqrt(np.diag(np.linalg.inv(H)))
        logit[season] = {"n": g.height, "intercept": b[0], "market": b[1], "model_gap": b[2], "model_gap_se": se[2]}
    print("logistic: result ~ market logit + model gap:", json.dumps(logit, default=float, indent=1))
    out["k_curve"] = F.to_dicts()
    out["k_by_market"] = by_mkt
    out["k_fit_2025"] = k_fit
    out["logistic"] = logit

    # ---------------------------------------------------------------- 3. the board's rule at closing prices
    Lc = L.filter(pl.col("when") == "close")
    pinmap = {(r["game_id"], r["gsis_id"], r["market"]): r for r in Lc.filter(pl.col("book") == "pinnacle").to_dicts()}
    bets = []
    for r in Lc.filter(pl.col("book").is_in(["bet365", "draftkings", "fanduel"])).to_dicts():
        if r["book"] not in RETAIL[r["season"]]:
            continue
        key = (r["game_id"], r["gsis_id"], r["market"])
        pr = pinmap.get(key)
        if pr is None:
            continue
        s = S[key]
        d = shift_to(s, pr["line"], pr["p_over"])
        for k in (0.0, BOARD_K, k_fit, 1.0):
            ov, un = p_over_at(s, r["line"], (1 - k) * d)
            push = max(0.0, 1 - ov - un)
            for side, p, price in (("over", ov, r["over"]), ("under", un, r["under"])):
                ev = p * (price - 1) - (1 - p - push)
                won = r["actual"] > r["line"] if side == "over" else r["actual"] < r["line"]
                u = 0.0 if r["actual"] == r["line"] else (price - 1 if won else -1.0)
                bets.append({"season": r["season"], "k": k, "game_id": r["game_id"], "gsis_id": r["gsis_id"],
                             "market": r["market"], "role": r["role"], "book": r["book"], "side": side, "line": r["line"],
                             "price": price, "p": p, "ev": ev, "units": u, "result": "P" if u == 0 and r["actual"] == r["line"] else ("W" if won else "L"),
                             "pin_line": pr["line"], "pin_p_over": pr["p_over"]})
    B = pl.DataFrame(bets)
    # one bet per player-market: the best side and book
    best = B.sort("ev", descending=True).group_by("season", "k", "game_id", "gsis_id", "market", maintain_order=True).first()
    sims = []
    for (season, k), g in best.group_by("season", "k"):
        for t in THRESH:
            b = g.filter(pl.col("ev") >= t)
            if b.height == 0:
                continue
            u, gm = b["units"].to_numpy(), b["game_id"].to_numpy()
            lo, hi = boot(u, gm)
            sims.append({"season": season, "k": k, "min_ev": t, "n": b.height, "w": int((b["result"] == "W").sum()),
                         "l": int((b["result"] == "L").sum()), "p": int((b["result"] == "P").sum()),
                         "units": float(u.sum()), "roi": float(u.sum() / b.height), "ci_lo": lo, "ci_hi": hi,
                         "said": float(b["p"].mean()), "won": float((b["result"] == "W").sum() / max(1, (b["result"] != "P").sum())),
                         "avg_price": float(b["price"].mean()), "same_line_share": float((b["line"] == b["pin_line"]).mean())})
    SIM = pl.DataFrame(sims).sort("season", "k", "min_ev")
    print("bets at the retail close (one per player-market, best side and book):\n", SIM)
    out["sims"] = SIM.to_dicts()
    # by market for the board's rule at the main threshold
    bm = best.filter((pl.col("ev") >= 0.04)).group_by("season", "k", "market").agg(
        n=pl.len(), units=pl.col("units").sum(), roi=pl.col("units").mean()).sort("season", "k", "market")
    out["sims_by_market"] = bm.to_dicts()
    print(bm)
    # unders vs overs among the bets
    bs = best.filter((pl.col("ev") >= 0.04)).group_by("season", "k", "side").agg(
        n=pl.len(), units=pl.col("units").sum(), roi=pl.col("units").mean()).sort("season", "k", "side")
    out["sims_by_side"] = bs.to_dicts()
    print(bs)

    # ---------------------------------------------------------------- 4. plus-money unders, every player
    out["plus_unders"] = plus_unders(Q, M)
    # ---------------------------------------------------------------- 5. 2026: line shopping at the open, line value at the close
    out["open_shop"] = open_shopping(L, S)
    # ---------------------------------------------------------------- 6. 2026: both rules at the board's real lock times (OddsPapi)
    out["lock"] = lock_checks(S)
    # ---------------------------------------------------------------- 7. 2025 at the lock time, Pinnacle's rebuilt line history
    out["lock_2025"] = lock_2025()

    out["board"] = board_summary(out, best)
    REPORTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    best.write_parquet(ROWS)
    print("saved", OUT)


if __name__ == "__main__":
    import sys
    if "--lock2025" in sys.argv:   # just the lock-time section, merged into the saved report
        saved = json.loads(OUT.read_text())
        saved["lock_2025"] = lock_2025()
        OUT.write_text(json.dumps(saved, indent=1, default=float))
        print("saved", OUT)
    else:
        main()
