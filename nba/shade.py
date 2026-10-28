"""The books' over shading learned from this season, each prop pick's edge source, and the situational model in
shadow (user, 2026-10-27).

- At every props lock, every tracked prop's Pinnacle price (raw even-split no-vig chance of the over), the best
  DraftKings/FanDuel price on the same line, and the situational model's chance (nba/sides.py, never bet) go to
  data/nba/shade/rows.parquet (odds data: the Actions cache, never the repo). Pinnacle's price near tip-off goes
  to closes.parquet once each game is final.
- `estimate`: per market, the logit shift between Pinnacle's even split and how often the over actually hit,
  starting from zero (PRIOR) with a prior sd of 0.10 and moved by every graded tracked prop of earlier days,
  weighted by a 14-day half-life. Until 2026-10-29 it started from props.OVER_SHADE (five US books, 2022-23; no
  Pinnacle in it), which this season's Pinnacle prices did not bear out: learned from the opening nights
  (2026-10-20 to 22), it priced the overs of 2026-10-23 to 27 worse than no correction at all (log loss 0.69155
  against 0.68983, 1,263 tracked props), hurting points, rebounds and threes; from zero, 0.68984, keeping the
  assists gain. It also put the 2022-23 shifts on every combo with no Pinnacle price (P+R, P+A, R+A), which
  never learn here. On 2026-10-28 that start made 42 of the 98 props picks (threes, rebounds, R+A and P+R unders).
- `edge_of`: a pick is "price" when DraftKings/FanDuel beat Pinnacle's raw price by 1%+ (a real price gap) and
  "correction" when only the over correction makes it +EV. The bet list judges them as separate categories.
- `shadow`: the situational model's picks (5%+ EV at the best DK/FD price on Pinnacle's line), graded and
  valued at Pinnacle's raw close: its record on the props Record page, never a bet.
"""
import json
import math
import subprocess
from datetime import date

import numpy as np
import polars as pl

from config import DATA, ROOT
from nba import props as NP
from nba import sides as S

DIR = DATA / "nba" / "shade"
ROWS, CLOSES = DIR / "rows.parquet", DIR / "closes.parquet"
COEF = ROOT / "nba" / "sides_coef.json"
TAU, HALF_LIFE, SHADOW_EV = 0.10, 14.0, 0.05
PRIOR = 0.0          # the shift every market starts from: the even split taken as right until this season says otherwise
ROW_SCHEMA = {"day": pl.String, "lock": pl.String, "game_id": pl.Int64, "athlete_id": pl.Int64, "name": pl.String,
              "team": pl.String, "opp": pl.String, "home": pl.String, "market": pl.String, "line": pl.Float64,
              "p_raw": pl.Float64, "o_pin": pl.Float64, "pm": pl.Float64, "games_now": pl.Int64, "spread": pl.Float64,
              "total": pl.Float64, "bo_dec": pl.Float64, "bo_book": pl.String, "bu_dec": pl.Float64, "bu_book": pl.String,
              "p_sides": pl.Float64}
CLOSE_SCHEMA = {"game_id": pl.Int64, "athlete_id": pl.Int64, "market": pl.String, "line": pl.Float64, "p_close": pl.Float64}
SHOP = {"draftkings": "DraftKings", "fanduel": "FanDuel"}


def _lg(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sg(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def rows() -> pl.DataFrame:
    return pl.read_parquet(ROWS) if ROWS.exists() else pl.DataFrame(schema=ROW_SCHEMA)


def closes() -> pl.DataFrame:
    return pl.read_parquet(CLOSES) if CLOSES.exists() else pl.DataFrame(schema=CLOSE_SCHEMA)


def save(new: list[dict]) -> int:
    """Append one lock's tracked props (a later lock of the same prop and day replaces it)."""
    if not new:
        return 0
    DIR.mkdir(parents=True, exist_ok=True)
    df = pl.concat([rows(), pl.DataFrame(new, schema=ROW_SCHEMA)], how="vertical_relaxed")
    df.unique(["day", "lock", "game_id", "athlete_id", "market"], keep="last").write_parquet(ROWS)
    return len(new)


def save_closes(new: list[dict]) -> None:
    if new:
        DIR.mkdir(parents=True, exist_ok=True)
        pl.concat([closes(), pl.DataFrame(new, schema=CLOSE_SCHEMA)], how="vertical_relaxed").unique(
            ["game_id", "athlete_id", "market"], keep="last").write_parquet(CLOSES)


# ---------------------------------------------------------------- at a lock

def lock_rows(x: pl.DataFrame, mixes: dict, books: dict, day: date, now, homes: dict, fair: dict) -> list[dict]:
    """Every tracked prop with a Pinnacle price at this lock. `books[(gid, aid, market)][book]` = {line, over, under,
    p_over} (decimal prices, even-split no-vig); `fair[gid]` = {margin (home), total}."""
    out = []
    for r in x.iter_rows(named=True):
        for m in NP.MARKETS:
            bk = books.get((r["game_id"], r["athlete_id"], m)) or {}
            pin = bk.get("pinnacle")
            if not pin:
                continue
            same = {b: bk[b] for b in SHOP if b in bk and bk[b]["line"] == pin["line"]}
            bo = max(same.items(), key=lambda t: t[1]["over"], default=(None, None))
            bu = max(same.items(), key=lambda t: t[1]["under"], default=(None, None))
            mix = mixes.get((r["game_id"], r["athlete_id"], m))
            pm = None
            if mix is not None:
                o, _, u = mix.probs(pin["line"])
                pm = o / max(o + u, 1e-12)
            f = fair.get(r["game_id"]) or {}
            out.append({"day": day.isoformat(), "lock": now.isoformat(), "game_id": r["game_id"], "athlete_id": r["athlete_id"],
                        "name": r["name"], "team": r["team"], "opp": r["opp"], "home": homes.get(r["game_id"]), "market": m,
                        "line": float(pin["line"]), "p_raw": float(pin["p_over"]), "o_pin": float(pin["over"]), "pm": pm,
                        "games_now": int(r.get("games_now") or 0), "spread": f.get("margin"), "total": f.get("total"),
                        "bo_dec": bo[1]["over"] if bo[1] else None, "bo_book": SHOP.get(bo[0]),
                        "bu_dec": bu[1]["under"] if bu[1] else None, "bu_book": SHOP.get(bu[0]), "p_sides": None})
    return out


def with_sides(new: list[dict], hist: pl.DataFrame) -> list[dict]:
    """The situational model's chance of the over for each row it covers (points, rebounds, assists, threes)."""
    coef = {k: np.array(v) for k, v in json.loads(COEF.read_text(encoding="utf-8")).items() if not k.startswith("_")} if COEF.exists() else {}
    todo = [r for r in new if r["market"] in coef]
    if not todo:
        return new
    P = pl.DataFrame([{"i": i, "game_id": r["game_id"], "athlete_id": r["athlete_id"], "market": r["market"], "line": r["line"],
                       "p": r["p_raw"], "o_dec": r["o_pin"], "pm": r["pm"] if r["pm"] is not None else r["p_raw"],
                       "games_now": r["games_now"], "close_spread": r["spread"], "close_total": r["total"]}
                      for i, r in enumerate(new) if r["market"] in coef],
                     schema={"i": pl.Int64, "game_id": pl.Int64, "athlete_id": pl.Int64, "market": pl.String, "line": pl.Float64,
                             "p": pl.Float64, "o_dec": pl.Float64, "pm": pl.Float64, "games_now": pl.Int64,
                             "close_spread": pl.Float64, "close_total": pl.Float64})
    # tonight's games are not in the box scores yet: a placeholder row per tracked player, so his previous games
    # line up as the lags (S.assemble only ever reads games before the one it prices)
    bx = hist
    have = set(bx.filter(pl.col("min") > 0).select("game_id", "athlete_id").iter_rows())
    team = {(r["game_id"], r["athlete_id"]): r["team"] for r in new}
    pos = dict(bx.sort("date").group_by("athlete_id").agg(pl.col("pos").last()).iter_rows())
    season = int(bx["season"].max())
    day = date.fromisoformat(new[0]["day"])
    ph = [{"game_id": g, "season": season, "date": day, "athlete_id": a, "name": "", "team": t, "pos": pos.get(a) or "F",
           "min": 1.0, "starter": False, "did_not_play": False, **{k: 0.0 for k in NP.STATS}}
          for (g, a), t in team.items() if (g, a) not in have]
    if ph:
        bx = pl.concat([bx, pl.DataFrame(ph).select(bx.columns)], how="vertical_relaxed")
    homes = pl.DataFrame({"game_id": [r["game_id"] for r in new], "home": [r["home"] for r in new]}).unique("game_id")
    F = S.assemble(P, bx, homes)
    ps = S.predict(F, coef, S.FEATS)
    out = [dict(r) for r in new]
    for i, p in zip(F["i"].to_list(), ps):
        out[i]["p_sides"] = float(p)
    return out


# ---------------------------------------------------------------- grading and the learned shading

def graded(R: pl.DataFrame, stats: dict) -> pl.DataFrame:
    """The latest lock's row per (day, prop) for games final, with y (over hit); pushes and DNPs dropped."""
    if not R.height:
        return R.with_columns(y=pl.lit(None, pl.Float64))
    R = R.sort("lock").unique(["day", "game_id", "athlete_id", "market"], keep="last")
    act = [(stats.get((g, a)) or {}).get(m) for g, a, m in R.select("game_id", "athlete_id", "market").iter_rows()]
    R = R.with_columns(actual=pl.Series(act, dtype=pl.Float64))
    return (R.filter(pl.col("actual").is_not_null() & (pl.col("actual") != pl.col("line")))
             .with_columns(y=(pl.col("actual") > pl.col("line")).cast(pl.Float64)))


def estimate(G: pl.DataFrame, day: date) -> dict[str, dict]:
    """Per market: the logit shift (MAP, Newton) of the over's true chance from Pinnacle's even split, from graded
    rows of days before `day`; prior PRIOR, sd TAU; recency half-life HALF_LIFE days."""
    out = {}
    G = G.filter(pl.col("day") < day.isoformat()) if G.height else G
    for mk in NP.MARKETS:
        mu0 = PRIOR
        S_ = G.filter(pl.col("market") == mk) if G.height else G
        z = np.array([_lg(p) for p in S_["p_raw"].to_list()]) if S_.height else np.zeros(0)
        y = S_["y"].to_numpy() if S_.height else np.zeros(0)
        age = np.array([(day - date.fromisoformat(d)).days for d in S_["day"].to_list()]) if S_.height else np.zeros(0)
        w = 0.5 ** (age / HALF_LIFE)
        s = mu0
        for _ in range(25):
            q = 1 / (1 + np.exp(-(z + s)))
            g = -(s - mu0) / TAU ** 2 + float(np.sum(w * (y - q)))
            h = 1 / TAU ** 2 + float(np.sum(w * q * (1 - q)))
            s += g / h
            if abs(g / h) < 1e-8:
                break
        out[mk] = {"shift": round(s, 4), "prior": mu0, "se": round(1 / math.sqrt(h), 4), "n": int(S_.height),
                   "said": round(float(np.mean([_sg(v) for v in z])), 4) if len(z) else None,
                   "hit": round(float(np.mean(y)), 4) if len(y) else None}
    return out


def edge_index(R: pl.DataFrame) -> dict:
    """(day, game_id, athlete_id, market) -> (line, p_raw) at the latest lock: for picks locked before `ev_raw`."""
    if not R.height:
        return {}
    R = R.sort("lock").unique(["day", "game_id", "athlete_id", "market"], keep="last")
    return {(d, g, a, m): (l, p) for d, g, a, m, l, p in R.select("day", "game_id", "athlete_id", "market", "line", "p_raw").iter_rows()}


def edge_of(p: dict, idx: dict, ev_min: float) -> str | None:
    """"price" when the pick beats Pinnacle's raw price by ev_min or more, "correction" when only the over
    correction makes it +EV, None when unknown (a line Pinnacle did not have, before this was recorded)."""
    ev = p.get("ev_raw")
    if ev is None:
        hit = idx.get((p["day"], p["game_id"], p["athlete_id"], p["market"]))
        if not hit or hit[0] != p["line"]:
            return None
        pw = hit[1] if p["side"] == "over" else 1 - hit[1]
        ev = pw * p["dec"] - 1
    return "price" if ev >= ev_min else "correction"


# ---------------------------------------------------------------- the situational model, in shadow

def shadow(G: pl.DataFrame, C: pl.DataFrame) -> list[dict]:
    """Its picks: per player and lock day, the side and book with the best EV under its chance, 5%+; graded, with
    EV at Pinnacle's raw close on the same line. Rows shaped like the record's (build._rows)."""
    if not G.height or "p_sides" not in G.columns:
        return []
    G = G.filter(pl.col("p_sides").is_not_null())
    cl = {(g, a, m): (l, p) for g, a, m, l, p in C.select("game_id", "athlete_id", "market", "line", "p_close").iter_rows()} if C.height else {}
    best: dict = {}
    for r in G.iter_rows(named=True):
        for side, dec, book in (("over", r["bo_dec"], r["bo_book"]), ("under", r["bu_dec"], r["bu_book"])):
            if dec is None:
                continue
            pw = r["p_sides"] if side == "over" else 1 - r["p_sides"]
            ev = pw * dec - 1
            k = (r["day"], r["athlete_id"])
            if ev >= SHADOW_EV and (k not in best or ev / (dec - 1) > best[k][0]):
                best[k] = (ev / (dec - 1), r, side, dec, book, ev)
    out = []
    for _, r, side, dec, book, ev in best.values():
        win = (r["y"] == 1.0) == (side == "over")
        c = cl.get((r["game_id"], r["athlete_id"], r["market"]))
        evc = None
        if c and c[0] == r["line"] and c[1] is not None:
            evc = round((c[1] if side == "over" else 1 - c[1]) * dec - 1, 4)
        out.append({"day": r["day"], "market": r["market"], "side": side, "result": "W" if win else "L",
                    "units": round(dec - 1 if win else -1.0, 2), "ev": round(ev, 4), "ev_close": evc, "clv": None,
                    "label": f"{r['name']} {side.title()} {r['line']:g} {NP.MARKET_LABEL[r['market']].lower()}",
                    "book": book, "american": None, "risk": 1.0, "legs": []})
    return out


def close_rows(R: pl.DataFrame, finished: set, have: pl.DataFrame, C: dict) -> list[dict]:
    """Pinnacle's price near tip-off for tracked props of finished games not yet stored. `C`: build._books of the
    closing main lines."""
    if not R.height:
        return []
    done = set(have.select("game_id", "athlete_id", "market").iter_rows()) if have.height else set()
    out = []
    for g, a, m, l in R.select("game_id", "athlete_id", "market", "line").unique().iter_rows():
        if g not in finished or (g, a, m) in done:
            continue
        pin = (C.get((g, a, m)) or {}).get("pinnacle")
        out.append({"game_id": g, "athlete_id": a, "market": m, "line": l,
                    "p_close": float(pin["p_over"]) if pin and pin["line"] == l else None})
    return out


# ---------------------------------------------------------------- the first week, from the committed board

def bootstrap(picks: list[dict], hist: pl.DataFrame) -> int:
    """Rebuild rows for locks before this module existed from the board payload committed at each lock time
    (git history; the workflow fetches full history only when ROWS is missing). Pinnacle's price, the best
    DK/FD on its line, and the props model's chance read off the payload's quantiles."""
    locks = sorted({p["locked_at"][:16] for p in picks if p.get("kind") == "prop" and p.get("locked_at")})
    have = set(rows()["lock"].str.slice(0, 16).to_list()) if ROWS.exists() else set()
    locks = [k for k in locks if k not in have]
    if not locks:
        return 0
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, encoding="utf-8").stdout  # noqa: E731
    dec = lambda a: 1 + (a / 100 if a > 0 else 100 / -a)  # noqa: E731
    qp = [0.1, 0.25, 0.5, 0.75, 0.9]
    new = []
    for h in git("log", "--format=%H", "--", "board/nba_data.json").split():
        try:
            d = json.loads(git("show", f"{h}:board/nba_data.json"))
        except json.JSONDecodeError:
            continue
        k = d["meta"]["generated"][:16]
        if k not in locks:
            continue
        locks.remove(k)
        games = {c["id"]: c for c in d["games"]}
        for p in d["props"]["players"]:
            g = games.get(p["game_id"]) or {}
            if (g.get("tip") or {}).get("iso", "9")[:16] <= k:
                continue
            for m in p["markets"]:
                b = m.get("books") or {}
                q = b.get("Pinnacle")
                if not q or q.get("over") is None or q.get("under") is None or m["market"] not in NP.MARKETS:
                    continue
                io, iu = 1 / dec(q["over"]), 1 / dec(q["under"])
                same = {n: b[n] for n in ("DraftKings", "FanDuel") if b.get(n) and b[n].get("line") == q["line"]
                        and b[n].get("over") is not None and b[n].get("under") is not None}
                bo = max(same.items(), key=lambda t: dec(t[1]["over"]), default=(None, None))
                bu = max(same.items(), key=lambda t: dec(t[1]["under"]), default=(None, None))
                pm = None
                if m.get("q"):
                    xs = np.array(m["q"], float)
                    pm = 0.93 if q["line"] <= xs[0] else 0.07 if q["line"] >= xs[-1] else float(1 - np.interp(q["line"], xs, qp))
                f = g.get("fair") or {}
                new.append({"day": d["meta"]["generated"][:10], "lock": d["meta"]["generated"], "game_id": p["game_id"],
                            "athlete_id": p["id"], "name": p["name"], "team": p["team"], "opp": p["opp"], "home": g.get("home"),
                            "market": m["market"], "line": float(q["line"]), "p_raw": io / (io + iu), "o_pin": dec(q["over"]),
                            "pm": pm, "games_now": int(p.get("games_now") or 0), "spread": f.get("margin"), "total": f.get("total"),
                            "bo_dec": dec(bo[1]["over"]) if bo[1] else None, "bo_book": bo[0],
                            "bu_dec": dec(bu[1]["under"]) if bu[1] else None, "bu_book": bu[0], "p_sides": None})
        if not locks:
            break
    by_lock: dict[str, list] = {}
    for r in new:
        by_lock.setdefault(r["lock"], []).append(r)
    n = 0
    for lk, rs in sorted(by_lock.items()):
        try:
            rs = with_sides(rs, hist.filter(pl.col("date") < date.fromisoformat(rs[0]["day"])))
        except Exception as e:   # noqa: BLE001  (the shadow model never stops the build)
            print(f"shade bootstrap: situational model skipped for {lk}: {type(e).__name__}: {e}")
        n += save(rs)
    return n
