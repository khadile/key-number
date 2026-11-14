"""Early-season college basketball: every simple angle and every pair of them, this season (user, 2026-11-14).

The user: "for this season alone, what edges do you see ... try all types of combinations possible", in a market
less efficient than the NFL's or NBA's. This builds a row per team per game (against the spread) and a row per game
(the total) for every Division I game this season with a closing line, tags each with simple facts known before the
tip, and scores every single tag and every pair of tags.

Searching hundreds of combinations finds some that look strong by chance alone (at |z| >= 2, about 1 in 22). So each
condition is also shown:
- in each half of this season so far (split by date): a real edge should show in both;
- in the same weeks of the last four seasons (the same calendar window, early November on): a guide only, since
  short-lived edges are the point, but it separates an early-season effect from this year's noise.
Nothing here is a pick rule; it is a list of leads to track at the closing line.

    python -m ncaab.explore          # on Actions (ESPN): research.yml job ncaab-explore -> reports/ncaab_explore_2027.json
"""
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import polars as pl

from config import REPORTS
from ncaab.top25 import DIR, CORE, _am, _dig, _json, _num, ranks, schedules

SEASON = 2027
PAST = (2023, 2024, 2025, 2026)
OUT = REPORTS / f"ncaab_explore_{SEASON}.json"
POWER = {2, 4, 7, 8, 23}                 # ACC, Big East, Big Ten, Big 12, SEC
MID = {3, 62, 44, 29, 18, 21}            # A-10, American, Mountain West, WCC, Missouri Valley, Pac-12 (rebuilt)
MIN_N = 40


def _odds(gid: int) -> dict | None:
    d = _json(CORE.format(g=gid))
    if d is None:
        return {"game_id": gid}
    items = sorted(d.get("items", []), key=lambda it: _dig(it, "provider", "priority") or 99)
    if not items:
        return {"game_id": gid}
    it = items[0]
    row = {"game_id": gid, "provider": _dig(it, "provider", "name")}
    for when in ("open", "close"):
        for side in ("home", "away"):
            o = _dig(it, f"{side}TeamOdds", when) or {}
            row[f"{when}_spread_{side}"] = _num(_dig(o, "pointSpread", "american"))
            row[f"{when}_sprice_{side}"] = _am(_dig(o, "spread", "american"))
            row[f"{when}_ml_{side}"] = _am(_dig(o, "moneyLine", "american"))
        t = _dig(it, when) or {}
        row[f"{when}_total"] = _num(_dig(t, "total", "american")) or _num(_dig(t, "total", "alternateDisplayValue"))
        row[f"{when}_over"] = _am(_dig(t, "over", "american"))
        row[f"{when}_under"] = _am(_dig(t, "under", "american"))
    if row.get("close_spread_home") is None and it.get("spread") is not None:
        row["close_spread_home"], row["close_spread_away"] = float(it["spread"]), -float(it["spread"])
    if row.get("close_total") is None and it.get("overUnder") is not None:
        row["close_total"] = float(it["overUnder"])
    return row


def odds_all(gids: list[int]) -> pl.DataFrame:
    f = DIR / "odds_all.parquet"
    have = pl.read_parquet(f) if f.exists() else pl.DataFrame(schema={"game_id": pl.Int64})
    known = set(have["game_id"].to_list()) if have.height else set()
    todo = [g for g in gids if g not in known]
    print(f"odds: {len(todo)} games to fetch, {len(known)} cached", flush=True)
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(_odds, todo)):
            if r:
                rows.append(r)
            if k and k % 1000 == 0:
                print(f"  {k} of {len(todo)}", flush=True)
    if rows:
        new = pl.DataFrame(rows, infer_schema_length=None)
        have = pl.concat([have, new], how="diagonal_relaxed") if have.height else new
        have.write_parquet(f)
    return have.unique("game_id", keep="last")


def tier(c) -> pl.Expr:
    return pl.when(c.is_in(list(POWER))).then(pl.lit("power")).when(c.is_in(list(MID))).then(pl.lit("mid")).when(c.is_null()).then(pl.lit("non-D1")).otherwise(pl.lit("low"))


def frame() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(side rows, game rows) for this season and the same window of past seasons."""
    S = schedules()
    S = S.filter(pl.col("season").is_in([SEASON, *PAST]))
    # the window: this season so far; past seasons from their first game to the same day of the season
    first = S.group_by("season").agg(pl.col("date").min().alias("opener"))
    S = S.join(first, on="season").with_columns(sday=(pl.col("date") - pl.col("opener")).dt.total_days())
    cut = int(S.filter((pl.col("season") == SEASON) & pl.col("done"))["sday"].max())
    S = S.filter(pl.col("sday") <= cut + 3)
    R = ranks(S)
    S = S.join(R.select("game_id", "home_rank", "away_rank"), on="game_id", how="left")
    done = S.filter(pl.col("done") & pl.col("home_score").is_not_null())
    O = odds_all(done.filter(pl.col("home_conf").is_not_null() & pl.col("away_conf").is_not_null())["game_id"].to_list())
    G = done.join(O, on="game_id", how="left")

    # each team's sequence (all its games, lines or not): rest, games played, last result against the line
    long = pl.concat([
        G.select("season", "date", "game_id", pl.col("home_id").alias("tid"), (pl.col("home_score") - pl.col("away_score")).alias("margin"),
                 pl.col("close_spread_home").alias("spread")),
        G.select("season", "date", "game_id", pl.col("away_id").alias("tid"), (pl.col("away_score") - pl.col("home_score")).alias("margin"),
                 pl.col("close_spread_away").alias("spread"))]).sort("tid", "date", "game_id")
    long = long.with_columns(ats=pl.col("margin") + pl.col("spread"))
    by = ["tid", "season"]
    long = long.with_columns(
        n_before=pl.int_range(0, pl.len()).over(by),
        rest=(pl.col("date") - pl.col("date").shift(1).over(by)).dt.total_days(),
        last_ats=pl.col("ats").shift(1).over(by), last_margin=pl.col("margin").shift(1).over(by),
        ats_sum=(pl.col("ats").fill_null(0).cum_sum().over(by) - pl.col("ats").fill_null(0)),
        ats_n=(pl.col("ats").is_not_null().cast(pl.Int32).cum_sum().over(by) - pl.col("ats").is_not_null().cast(pl.Int32)))
    seq = long.select("game_id", "tid", "n_before", "rest", "last_ats", "last_margin", "ats_sum", "ats_n")

    sides = []
    for me, op in (("home", "away"), ("away", "home")):
        x = G.select(
            "game_id", "season", "date", "sday", pl.col("neutral_site").alias("neutral"), pl.col(f"{me}_id").alias("tid"),
            pl.col(f"{op}_id").alias("oid"), pl.lit(me).alias("side"),
            tier(pl.col(f"{me}_conf")).alias("tier"), tier(pl.col(f"{op}_conf")).alias("opp_tier"),
            (pl.col(f"{me}_conf") == pl.col(f"{op}_conf")).alias("conf_game"),
            pl.col(f"{me}_rank").is_between(1, 25).fill_null(False).alias("ranked"),
            pl.col(f"{op}_rank").is_between(1, 25).fill_null(False).alias("opp_ranked"),
            (pl.col(f"{me}_score") - pl.col(f"{op}_score")).alias("margin"),
            pl.col(f"close_spread_{me}").alias("spread"), pl.col(f"open_spread_{me}").alias("open_spread"),
            pl.col(f"close_sprice_{me}").alias("price"), pl.col("close_total").alias("total"))
        sides.append(x)
    D = pl.concat(sides).filter(pl.col("spread").is_not_null())
    D = (D.join(seq, on=["game_id", "tid"], how="left")
          .join(seq.rename({c: f"o_{c}" for c in seq.columns if c not in ("game_id",)}).rename({"o_tid": "oid"}), on=["game_id", "oid"], how="left"))
    D = D.with_columns(ats=pl.col("margin") + pl.col("spread"))
    return D, G


def tags_side(D: pl.DataFrame) -> pl.DataFrame:
    sp, mv = pl.col("spread"), pl.col("spread") - pl.col("open_spread")
    T = {
        "venue": pl.when(pl.col("neutral")).then(pl.lit("neutral")).otherwise(pl.col("side")),
        "role": pl.when(sp > 0).then(pl.lit("dog")).when(sp < 0).then(pl.lit("fav")).otherwise(pl.lit("pick")),
        "size": pl.when(sp.abs() <= 3.5).then(pl.lit("0-3.5")).when(sp.abs() <= 7.5).then(pl.lit("4-7.5")).when(sp.abs() <= 12.5).then(pl.lit("8-12.5"))
                  .when(sp.abs() <= 19.5).then(pl.lit("13-19.5")).otherwise(pl.lit("20+")),
        "tier": pl.col("tier"), "opp_tier": pl.col("opp_tier"),
        "matchup": pl.when(pl.col("tier") == pl.col("opp_tier")).then(pl.lit("same tier")).when(
            (pl.col("tier") == "power") | ((pl.col("tier") == "mid") & (pl.col("opp_tier") == "low"))).then(pl.lit("higher tier")).otherwise(pl.lit("lower tier")),
        "ranked": pl.when(pl.col("ranked")).then(pl.lit("ranked")).when(pl.col("opp_ranked")).then(pl.lit("vs ranked")).otherwise(pl.lit("neither ranked")),
        "move": pl.when(pl.col("open_spread").is_null()).then(pl.lit("no open")).when(mv >= 1.5).then(pl.lit("got 1.5+ more")).when(mv > 0).then(pl.lit("got more"))
                  .when(mv <= -1.5).then(pl.lit("1.5+ fewer")).when(mv < 0).then(pl.lit("fewer")).otherwise(pl.lit("none")),
        "games": pl.when(pl.col("n_before") == 0).then(pl.lit("1st game")).when(pl.col("n_before") <= 2).then(pl.lit("games 2-3")).otherwise(pl.lit("game 4+")),
        "rest": pl.when(pl.col("rest").is_null()).then(pl.lit("opener")).when(pl.col("rest") <= 1).then(pl.lit("1 day")).when(pl.col("rest") <= 3).then(pl.lit("2-3 days"))
                  .otherwise(pl.lit("4+ days")),
        "rest_vs": pl.when(pl.col("rest").is_null() | pl.col("o_rest").is_null()).then(pl.lit("unknown")).when(pl.col("rest") > pl.col("o_rest")).then(pl.lit("more rest"))
                     .when(pl.col("rest") < pl.col("o_rest")).then(pl.lit("less rest")).otherwise(pl.lit("same rest")),
        "last_ats": pl.when(pl.col("last_ats").is_null()).then(pl.lit("none")).when(pl.col("last_ats") >= 10).then(pl.lit("covered by 10+"))
                      .when(pl.col("last_ats") > 0).then(pl.lit("covered")).when(pl.col("last_ats") <= -10).then(pl.lit("missed by 10+")).otherwise(pl.lit("missed")),
        "last_su": pl.when(pl.col("last_margin").is_null()).then(pl.lit("none")).when(pl.col("last_margin") >= 20).then(pl.lit("won by 20+"))
                     .when(pl.col("last_margin") > 0).then(pl.lit("won")).when(pl.col("last_margin") <= -15).then(pl.lit("lost by 15+")).otherwise(pl.lit("lost")),
        "season_ats": pl.when(pl.col("ats_n") < 2).then(pl.lit("<2 lined")).when(pl.col("ats_sum") / pl.col("ats_n") >= 5).then(pl.lit("beating line by 5+"))
                        .when(pl.col("ats_sum") / pl.col("ats_n") <= -5).then(pl.lit("missing line by 5+")).otherwise(pl.lit("near the line")),
        "opp_last_ats": pl.when(pl.col("o_last_ats").is_null()).then(pl.lit("none")).when(pl.col("o_last_ats") >= 10).then(pl.lit("covered by 10+"))
                          .when(pl.col("o_last_ats") > 0).then(pl.lit("covered")).when(pl.col("o_last_ats") <= -10).then(pl.lit("missed by 10+")).otherwise(pl.lit("missed")),
        "total": pl.when(pl.col("total").is_null()).then(pl.lit("no total")).when(pl.col("total") < 135).then(pl.lit("under 135"))
                   .when(pl.col("total") < 150).then(pl.lit("135-149.5")).otherwise(pl.lit("150+")),
        "conf_game": pl.when(pl.col("conf_game")).then(pl.lit("conference game")).otherwise(pl.lit("non-conference")),
        "weekday": pl.when(pl.col("date").dt.weekday() >= 6).then(pl.lit("weekend")).otherwise(pl.lit("weekday")),
    }
    return D.with_columns([e.alias(f"t_{k}") for k, e in T.items()])


def tags_total(G: pl.DataFrame, D: pl.DataFrame) -> pl.DataFrame:
    g = G.filter(pl.col("close_total").is_not_null()).with_columns(
        tot=pl.col("home_score") + pl.col("away_score"), mv=pl.col("close_total") - pl.col("open_total"),
        ht=tier(pl.col("home_conf")), at=tier(pl.col("away_conf")))
    h = D.filter(pl.col("side") == "home").select("game_id", "n_before", "o_n_before", "rest", "o_rest", "spread")
    g = g.join(h, on="game_id", how="left").with_columns(y=pl.col("tot") - pl.col("close_total"))
    T = {
        "size": pl.when(pl.col("close_total") < 135).then(pl.lit("under 135")).when(pl.col("close_total") < 145).then(pl.lit("135-144.5"))
                  .when(pl.col("close_total") < 155).then(pl.lit("145-154.5")).otherwise(pl.lit("155+")),
        "move": pl.when(pl.col("open_total").is_null()).then(pl.lit("no open")).when(pl.col("mv") >= 2).then(pl.lit("up 2+")).when(pl.col("mv") > 0).then(pl.lit("up"))
                  .when(pl.col("mv") <= -2).then(pl.lit("down 2+")).when(pl.col("mv") < 0).then(pl.lit("down")).otherwise(pl.lit("none")),
        "tiers": pl.when((pl.col("ht") == "power") & (pl.col("at") == "power")).then(pl.lit("power v power"))
                   .when((pl.col("ht") == "power") | (pl.col("at") == "power")).then(pl.lit("power v other")).when((pl.col("ht") == "low") & (pl.col("at") == "low")).then(pl.lit("low v low"))
                   .otherwise(pl.lit("mid involved")),
        "spread": pl.when(pl.col("spread").abs() <= 5).then(pl.lit("close game (<=5)")).when(pl.col("spread").abs() <= 12.5).then(pl.lit("5.5-12.5"))
                    .when(pl.col("spread").abs() <= 20).then(pl.lit("13-20")).otherwise(pl.lit("20.5+ blowout")),
        "venue": pl.when(pl.col("neutral_site")).then(pl.lit("neutral")).otherwise(pl.lit("home court")),
        "games": pl.when((pl.col("n_before") == 0) | (pl.col("o_n_before") == 0)).then(pl.lit("a team's 1st game")).when((pl.col("n_before") <= 2) | (pl.col("o_n_before") <= 2))
                   .then(pl.lit("a team in games 2-3")).otherwise(pl.lit("both 4+")),
        "rest": pl.when(pl.col("rest").is_null() | pl.col("o_rest").is_null()).then(pl.lit("opener")).when((pl.col("rest") <= 1) | (pl.col("o_rest") <= 1))
                  .then(pl.lit("a team on 1 day")).otherwise(pl.lit("both 2+ days")),
        "ranked": pl.when(pl.col("home_rank").is_between(1, 25).fill_null(False) | pl.col("away_rank").is_between(1, 25).fill_null(False)).then(pl.lit("ranked team")).otherwise(pl.lit("no ranked team")),
        "weekday": pl.when(pl.col("date").dt.weekday() >= 6).then(pl.lit("weekend")).otherwise(pl.lit("weekday")),
    }
    return g.with_columns([e.alias(f"t_{k}") for k, e in T.items()])


def _score(y: np.ndarray, price: np.ndarray | None = None) -> dict:
    w, l = int((y > 0).sum()), int((y < 0).sum())
    n = w + l
    p = w / n if n else None
    dec = np.full(len(y), 1 + 100 / 110) if price is None else np.where(np.isnan(price), 1 + 100 / 110, np.where(price > 0, 1 + price / 100, 1 + 100 / np.abs(price)))
    ret = np.where(y > 0, dec - 1, np.where(y < 0, -1.0, 0.0))
    return {"n": n, "w": w, "l": l, "rate": p, "z": (p - 0.5) / np.sqrt(0.25 / n) if n else 0.0, "roi": float(ret.mean()) if len(y) else None}


def search(df: pl.DataFrame, ycol: str, tagcols: list[str], key: str, labels: tuple[str, str], price: str | None = None) -> list[dict]:
    """Every single tag and pair of tags. For each, the better side (covering / over, or the other side / under)."""
    halves = sorted(df.filter(pl.col("season") == SEASON)["date"].to_list())
    mid = halves[len(halves) // 2] if halves else None
    cur = df.filter(pl.col("season") == SEASON)
    past = df.filter(pl.col("season").is_in(list(PAST)))
    vals = {c: [v for v in cur[c].unique().to_list() if v is not None] for c in tagcols}
    conds = [((c, v),) for c in tagcols for v in vals[c]]
    conds += [((a, va), (b, vb)) for a, b in itertools.combinations(tagcols, 2) for va in vals[a] for vb in vals[b]]
    out, seen = [], set()
    for cond in conds:
        sel = pl.lit(True)
        for c, v in cond:
            sel = sel & (pl.col(c) == v)
        x = cur.filter(sel)
        if key == "side":   # a game whose two sides both match says nothing (one covers, the other does not)
            dup = x.group_by("game_id").len().filter(pl.col("len") > 1)["game_id"]
            x = x.filter(~pl.col("game_id").is_in(dup.to_list()))
        if x.height < MIN_N:
            continue
        y = x[ycol].to_numpy()
        pr = x[price].cast(pl.Float64).fill_null(float("nan")).to_numpy() if price else None
        s = _score(y, pr)
        flip = s["rate"] is not None and s["rate"] < 0.5
        if flip:
            y = -y
            pr = None   # the other side's own price is not in the row: -110
            s = _score(y, pr)
        # the same bets reached by another description (home & 4-7.5 points is away & 4-7.5 taken the other way;
        # a team's first game is its season opener): kept once
        if key == "side":
            sig = hash(tuple(sorted(zip(x["game_id"].to_list(), [(sd == "home") != flip for sd in x["side"].to_list()]))))
        else:
            sig = hash((flip, tuple(sorted(x["game_id"].to_list()))))
        if sig in seen:
            continue
        seen.add(sig)
        a = x.filter(pl.col("date") < mid)[ycol].to_numpy() * (-1 if flip else 1)
        b = x.filter(pl.col("date") >= mid)[ycol].to_numpy() * (-1 if flip else 1)
        px = past.filter(sel)
        if key == "side":
            dup = px.group_by("game_id").len().filter(pl.col("len") > 1)["game_id"]
            px = px.filter(~pl.col("game_id").is_in(dup.to_list()))
        yp = px[ycol].to_numpy() * (-1 if flip else 1)
        sa, sb, sp = _score(a), _score(b), _score(yp)
        out.append({"cond": " & ".join(f"{c[2:]}={v}" for c, v in cond), "bet": labels[1] if flip else labels[0], "k": len(cond), **{k: s[k] for k in ("n", "w", "l", "rate", "z", "roi")},
                    "h1": [sa["w"], sa["l"]], "h2": [sb["w"], sb["l"]], "past": [sp["w"], sp["l"]], "past_rate": sp["rate"],
                    "both_halves": bool(sa["rate"] is not None and sb["rate"] is not None and sa["rate"] > 0.5 and sb["rate"] > 0.5),
                    # the same window of the last four seasons leans the same way by at least one standard error
                    "past_agrees": bool(sp["rate"] is not None and sp["n"] >= 100 and sp["z"] >= 1.0)})
    return out


def main() -> None:
    D, G = frame()
    S = tags_side(D)
    T = tags_total(G, D)
    sc = [c for c in S.columns if c.startswith("t_")]
    tc = [c for c in T.columns if c.startswith("t_")]
    cur_s, cur_t = S.filter(pl.col("season") == SEASON), T.filter(pl.col("season") == SEASON)
    print(f"this season: {cur_s['game_id'].n_unique()} games with a closing spread, {cur_t.height} with a closing total, through {cur_s['date'].max()}; "
          f"past windows: {S.filter(pl.col('season') != SEASON)['game_id'].n_unique()} games", flush=True)
    base = {"home": _score(cur_s.filter(pl.col("t_venue") == "home")["ats"].to_numpy()), "fav": _score(cur_s.filter(pl.col("t_role") == "fav")["ats"].to_numpy()),
            "over": _score((cur_t["y"]).to_numpy())}
    rs = search(S, "ats", sc, "side", ("this side covers", "the other side covers"), price="price")
    rt = search(T, "y", tc, "total", ("over", "under"))
    tests = len(rs) + len(rt)
    rep = {"through": str(cur_s["date"].max()), "games_spread": cur_s["game_id"].n_unique(), "games_total": cur_t.height, "tests": tests,
           "expected_false_z2": round(tests * 0.0455, 1), "base": base, "spreads": rs, "totals": rt}
    OUT.write_text(json.dumps(rep, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)), encoding="utf-8")
    print(f"{tests} conditions scored (n >= {MIN_N}); at |z| >= 2 about {rep['expected_false_z2']} would show by chance alone")
    print(f"baseline: home sides {base['home']['w']}-{base['home']['l']}, favorites {base['fav']['w']}-{base['fav']['l']}, overs {base['over']['w']}-{base['over']['l']}")
    for name, rows in (("SPREADS", rs), ("TOTALS", rt)):
        good = sorted([r for r in rows if r["z"] >= 2], key=lambda r: -r["z"])
        print(f"\n== {name}: {len(good)} conditions at z >= 2 (of {len(rows)}); with both halves and the past agreeing: {sum(r['both_halves'] and r['past_agrees'] for r in good)}")
        for r in good[:45]:
            pr = "–" if r["past_rate"] is None else f"{100 * r['past_rate']:.1f}% ({r['past'][0]}-{r['past'][1]})"
            print(f"  {r['bet']:22} {r['cond'][:70]:70} {r['w']}-{r['l']} {100 * r['rate']:.1f}% z {r['z']:.2f} ROI {100 * r['roi']:+.1f}% "
                  f"| halves {r['h1'][0]}-{r['h1'][1]}, {r['h2'][0]}-{r['h2'][1]} | past {pr}{'  <== holds' if r['both_halves'] and r['past_agrees'] else ''}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
