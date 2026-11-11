"""Do the exchanges price NBA props better than FanDuel and DraftKings? (2026-11-11 checkup)

The probe (pipeline/exchange_probe.py) found the exchanges in OddsPapi's free history: Novig carries all eight
tracked prop markets, Polymarket US points, rebounds and assists with the two sides about 1% apart. This asks
the question that matters, on the most recent finished games: at the evening lock (5:45pm ET) and just before
the tip, whose price told more about the result, the exchange's midpoint or the sportsbooks' even split?

An exchange's history is 1 to 10 million price points a game, so each call asks for one book and keeps only the
last price at each cut. Free calls, one per 5 seconds; `--until` stops fetching at a UTC clock time so the run
never overlaps a build's own archive calls. Tallies only in the log, never a price.

    python -m nba.exchange_test --games 40 --until 22:05
"""
import argparse
import gc
import json
import time as _time
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from config import DATA
from nba.odds import OVERROUND, PROP_MARKETS, _markets, fixture_index, load_props, main_lines, player_map
from nba.props import BOX_DIR, COMBOS, STATS
from pipeline.oddspapi import _get

EXCHANGES = ["novig.us", "polymarket.us"]
SNAPS = DATA / "nba_x"      # each game's snapshot per exchange (gitignored; its own Actions cache, so a re-run fetches nothing twice)
SNAPS.mkdir(parents=True, exist_ok=True)
COLS = ["fixture_id", "book", "market", "handicap", "outcome", "player", "cut", "price", "active"]
REFS = ["fanduel", "draftkings", "pinnacle"]
CUTS = {"5:45pm ET": None, "10 min before tip": None}   # filled per game


def _cuts(tip: datetime) -> dict[str, datetime]:
    from zoneinfo import ZoneInfo
    et = tip.astimezone(ZoneInfo("America/New_York"))
    evening = et.replace(hour=17, minute=45, second=0, microsecond=0).astimezone(timezone.utc)
    return {"5:45pm ET": evening, "10 min before tip": tip - timedelta(minutes=10)}


def _snap(h: dict, pmap: dict, cuts: dict[str, datetime], fid: str) -> list[tuple]:
    """Each tracked prop outcome's last price at or before each cut: (fixture, book, market, handicap, outcome,
    player, cut, price, active)."""
    marks = {k: v.strftime("%Y-%m-%dT%H:%M:%S") for k, v in cuts.items()}
    rows = []
    for book, b in h.get("bookmakers", {}).items():
        for m in b.get("markets", {}).values():
            for oid, o in m.get("outcomes", {}).items():
                info = pmap.get(int(oid))
                if not info:
                    continue
                for pid, pts in o.get("players", {}).items():
                    if pid == "0":
                        continue
                    last = {k: None for k in marks}
                    for p in pts:
                        t = p["createdAt"][:19]
                        for k, mk in marks.items():
                            if t <= mk and (last[k] is None or t >= last[k][0]):
                                last[k] = (t, float(p["price"]), bool(p["active"]))
                    for k, v in last.items():
                        if v:
                            rows.append((fid, book, PROP_MARKETS[info[0]], info[1], info[2], int(pid), k, v[1], v[2]))
    return rows


def _fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-(X @ b)))
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(len(b))
        b = b + np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-(X @ b)))
    return b, np.sqrt(np.diag(np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(len(b)))))


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--until", default=None, help="UTC HH:MM: stop fetching then and analyse what is in hand")
    ap.add_argument("--books", default=",".join(EXCHANGES), help="exchanges to ask for, one call each per game")
    args = ap.parse_args()
    EXCHANGES[:] = [b for b in args.books.split(",") if b]
    now = datetime.now(timezone.utc)
    stop = None
    if args.until:
        hh, mm = map(int, args.until.split(":"))
        stop = now.replace(hour=hh, minute=mm, second=0, microsecond=0)

    # results and people from the box scores the builds saved
    stats, people = {}, []
    for f in BOX_DIR.glob("*.json"):
        b = json.loads(f.read_text(encoding="utf-8"))
        for r in b["players"]:
            people.append({"athlete_id": r["athlete_id"], "name": r["name"], "team": r["team"]})
            if r["played"]:
                s = {k: r[k] or 0.0 for k in STATS}
                s.update({c: sum(s[x] for x in parts) for c, parts in COMBOS.items()})
                stats[(b["game_id"], r["athlete_id"])] = s
    pm = player_map(pl.DataFrame(people).unique(["athlete_id", "team"]))
    op_to_ath = dict(zip(pm["op_id"].to_list(), pm["athlete_id"].to_list()))
    done_ids = {g for g, _ in stats}
    idx = (fixture_index().filter((pl.col("tip") < now - timedelta(hours=5)) & pl.col("game_id").is_in(list(done_ids)))
           .sort("tip", descending=True).head(args.games))
    print(f"games asked: {idx.height}, {idx['tip'].min():%m-%d} to {idx['tip'].max():%m-%d}; exchanges {EXCHANGES}")
    omap = _markets(True)
    fetched = {x: 0 for x in EXCHANGES}
    for g in idx.iter_rows(named=True):
        for x in EXCHANGES:
            f = SNAPS / f"{g['fixture_id']}__{x}.parquet"
            if f.exists():
                continue
            if stop and datetime.now(timezone.utc) > stop:
                continue
            h = None
            for i in range(2):
                h = _get("historical-odds", {"fixtureId": g["fixture_id"], "bookmakers": x}, billable=False)
                if "error" not in h:
                    break
                h = None
                _time.sleep(3)
            if h:
                pl.DataFrame(_snap(h, omap, _cuts(g["tip"]), g["fixture_id"]), schema=COLS, orient="row").write_parquet(f)
                fetched[x] += 1
            del h
            gc.collect()
    files = [f for g in idx.iter_rows(named=True) for x in EXCHANGES if (f := SNAPS / f"{g['fixture_id']}__{x}.parquet").exists()]
    print(f"fetched now: {fetched}; snapshots in hand: {len(files)}")
    if not files:
        return
    X = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    analyse(X, load_props(idx["game_id"].to_list(), pm), stats, idx, op_to_ath)


def analyse(X: pl.DataFrame, P: pl.DataFrame, stats: dict, idx: pl.DataFrame, op_to_ath: dict) -> None:
    X = X.filter(pl.col("active")).with_columns(athlete_id=pl.col("player").replace_strict(op_to_ath, default=None, return_dtype=pl.Int64)).drop_nulls("athlete_id")
    X = X.join(idx.select("fixture_id", "game_id"), on="fixture_id").with_columns(pl.col("game_id").cast(P.schema["game_id"]))
    keys = ["game_id", "book", "market", "athlete_id", "cut"]
    o = X.filter(pl.col("outcome") == "Over").select(keys + ["handicap", pl.col("price").alias("over")])
    u = X.filter(pl.col("outcome") == "Under").select(keys + ["handicap", pl.col("price").alias("under")])
    pair = (o.join(u, on=keys + ["handicap"]).with_columns(bk=1 / pl.col("over") + 1 / pl.col("under"),
                                                           gap=(1 / pl.col("over") - 1 / pl.col("under")).abs(),
                                                           p_over=(1 / pl.col("over")) / (1 / pl.col("over") + 1 / pl.col("under"))))
    print("two sides' sum at the exchanges, every paired line:", {x: round(float(pair.filter(pl.col("book") == x)["bk"].median()), 4) for x in EXCHANGES if pair.filter(pl.col("book") == x).height})
    main_x = (pair.filter(pl.col("bk").is_between(*OVERROUND)).sort(["gap", "handicap"]).group_by(keys, maintain_order=True).first()
              .select(keys + [pl.col("handicap").alias("line"), "over", "under", "p_over", "bk"]))

    # the sportsbooks at the same cuts, from the builds' own archive
    parts = []
    for name in CUTS:
        cut = {g["game_id"]: _cuts(g["tip"])[name] for g in idx.iter_rows(named=True)}
        Pc = P.join(pl.DataFrame({"game_id": list(cut), "cut_at": list(cut.values())}).with_columns(pl.col("game_id").cast(P.schema["game_id"])), on="game_id")
        L = main_lines(Pc, pl.col("cut_at")).rename({"gsis_id": "athlete_id"}).with_columns(cut=pl.lit(name), bk=1 / pl.col("over") + 1 / pl.col("under"))
        parts.append(L.select(keys + ["line", "over", "under", "p_over", "bk"]))
    main_r = pl.concat(parts)
    A = pl.concat([main_x, main_r.select(main_x.columns)])
    tips = dict(zip(idx["game_id"].to_list(), idx["tip"].to_list()))

    for name in CUTS:
        a = A.filter(pl.col("cut") == name)
        # a cut after the game's start is no pregame price (afternoon games at 5:45pm)
        ok_games = [g for g, t in tips.items() if _cuts(t)[name] < t - timedelta(minutes=5)]
        a = a.filter(pl.col("game_id").is_in(ok_games))
        wide = a.pivot(on="book", index=["game_id", "market", "athlete_id"], values=["line", "p_over", "over", "under", "bk"])
        act = [stats.get((r["game_id"], r["athlete_id"]), {}).get(r["market"]) for r in wide.iter_rows(named=True)]
        wide = wide.with_columns(actual=pl.Series(act, dtype=pl.Float64)).drop_nulls("actual")
        print(f"\n=== {name}: {wide.height} tracked props with a result, {len(ok_games)} games ===")
        for bk in REFS + EXCHANGES:
            c = f"line_{bk}"
            if c in wide.columns:
                w = wide.drop_nulls(c)
                print(f"  {bk:14} props {w.height:5d}  two sides' sum {w[f'bk_{bk}'].mean():.4f}")
        for x in EXCHANGES:
            if f"line_{x}" not in wide.columns:
                continue
            for ref in ("fanduel", "draftkings", "pinnacle"):
                if f"line_{ref}" not in wide.columns:
                    continue
                w = wide.filter((pl.col(f"line_{x}") == pl.col(f"line_{ref}")) & (pl.col("actual") != pl.col(f"line_{ref}")))
                if w.height < 150:
                    print(f"  {x} vs {ref}: same line on only {w.height}")
                    continue
                y = (w["actual"] > w[f"line_{ref}"]).to_numpy().astype(float)
                px, pr = w[f"p_over_{x}"].to_numpy(), w[f"p_over_{ref}"].to_numpy()
                lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                b, se = _fit(np.column_stack([np.ones(len(y)), lg(pr), lg(px)]), y)
                print(f"  {x} vs {ref}, same line: n {len(y)}  log loss {ref} {_ll(pr, y):.5f}  {x} {_ll(px, y):.5f}  "
                      f"| joint fit: {ref} {b[1]:+.2f}±{se[1]:.2f}  {x} {b[2]:+.2f}±{se[2]:.2f}  intercept {b[0]:+.3f}±{se[0]:.3f}")
                d = px - pr
                for lab, k in (("exchange 2+ points lower on the over", d <= -0.02), ("within 2 points", np.abs(d) < 0.02), ("exchange 2+ points higher", d >= 0.02)):
                    if k.sum() >= 40:
                        print(f"       {lab:38} n {int(k.sum()):4d}  over hit {100 * y[k].mean():.1f}% ± {100 * np.sqrt(.25 / k.sum()):.1f}  {ref} said {100 * pr[k].mean():.1f}%  {x} said {100 * px[k].mean():.1f}%")
            # the board's rule with the exchange as the fair price: the best DraftKings/FanDuel side on the same line
            ev, ret, und = [], [], []
            for r in wide.iter_rows(named=True):
                best = None
                for s in ("draftkings", "fanduel"):
                    if r.get(f"line_{s}") is None or r.get(f"line_{x}") is None or r[f"line_{s}"] != r[f"line_{x}"] or r["actual"] == r[f"line_{s}"]:
                        continue
                    p = r[f"p_over_{x}"]
                    for side, pw, dec in (("over", p, r[f"over_{s}"]), ("under", 1 - p, r[f"under_{s}"])):
                        e = pw * dec - 1
                        if best is None or e > best[0]:
                            won = (r["actual"] > r[f"line_{s}"]) == (side == "over")
                            best = (e, dec - 1 if won else -1.0, side == "under")
                if best:
                    ev.append(best[0]), ret.append(best[1]), und.append(best[2])
            ev, ret, und = np.array(ev), np.array(ret), np.array(und)
            for lo, hi in ((0.0, 0.02), (0.02, 0.04), (0.04, 9)):
                k = (ev >= lo) & (ev < hi)
                if k.sum() >= 20:
                    print(f"  best DK/FD side against {x}'s price, EV {100 * lo:.0f}-{100 * min(hi, 1):.0f}%: n {int(k.sum()):4d}  won {100 * (ret[k] > 0).mean():.1f}%  "
                          f"ROI {100 * ret[k].mean():+.1f}±{100 * ret[k].std(ddof=1) / np.sqrt(k.sum()):.1f}%  unders {100 * und[k].mean():.0f}%")


if __name__ == "__main__":
    main()
