"""NBA player props: who is tracked, their minutes and per-minute rates, and simulated stat lines.

Tracked (user, 2026-10-20): each team's five starters and its sixth man. Markets: points, rebounds, assists,
3-pointers, and the combos points+rebounds+assists, points+rebounds, points+assists, rebounds+assists.

- Minutes: a weighted average of recent games he played (newer count more; last season counts `PREV_W` as
  much, so the first weeks lean on it), plus STARTER_BUMP for a starter, with 1.2 x his recent spread (at least
  5.5) as the uncertainty. A rotation player ruled out who played in the last three weeks hands his minutes to
  the available rotation; in a lopsided game the starters sit late (BLOWOUT per point of spread past 7).
- Age is counted in league game days, not calendar days: the summer and the All-Star break do not age a
  player's games. On the calendar, last season's games were six months old on opening night and weighed about
  0.3 minutes against PRIOR_MIN's 150, so every player was priced at his position's average (2026-27 opening
  night: Gilgeous-Alexander 0.49 points a minute against his 0.94), and a starter out since the spring never
  handed his minutes on.
- After an absence he comes in short (BACK_MIN, by league game days missed, fading over BACK_RAMP's four
  games), and so does everyone in the season's first games (SUMMER_MIN: the summer is a short absence of its own).
  His usual minutes keep his ramp-up games as they were: counting them at full minutes instead (and leaving out
  the game he left hurt) made every kind of game worse out of sample in 2023-24 and 2024-25 (first games back
  -0.7 -> -1.4 minutes, games 2-4 back 0.0 -> -1.3, normal games -0.8 -> -1.1). Players back from an absence
  stay below their old minutes for a while, so their latest games are the better guide.
- Rates: each stat per minute over the same games, pulled toward his own record over both seasons (PRIOR_MIN
  minutes of it, itself pulled toward his position's average by as much) so a hot week does not run away, and
  scaled by the game's projected pace.
- Outcomes: 4,000 simulated games: minutes, then each stat as a negative binomial around rate x minutes
  (dispersion DISP, variance over mean given minutes); combos are the sums of the same simulated games.

The model supplies the shape of each player's range; the price comes from the market (props/pricing.py, the
same rule as NFL props: Pinnacle's no-vig number, else the DK/FD average, nudged toward the model by k).
"""
import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from config import DATA
from nba.data import load

STATS = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3": "three_point_field_goals_made"}
COMBOS = {"pra": ("pts", "reb", "ast"), "pr": ("pts", "reb"), "pa": ("pts", "ast"), "ra": ("reb", "ast")}
MARKETS = list(STATS) + list(COMBOS)
MARKET_LABEL = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3": "3-pointers", "pra": "Pts+Reb+Ast",
                "pr": "Pts+Reb", "pa": "Pts+Ast", "ra": "Reb+Ast"}
# Measured on 2025-26 starters (10,672 games, profiles refit every two weeks): variance over mean given minutes was
# 2.49 points, 1.12 rebounds, 1.21 assists, 1.08 threes; starters played 1.6 minutes over their all-games average,
# and minutes missed by 6.8 (sd) against a recent spread of 5.4. With these settings the simulated ranges held 50%
# of outcomes in their middle half and 79% in their 10-90 band, for every stat and for PRA.
DISP = {"pts": 2.5, "reb": 1.15, "ast": 1.2, "fg3": 1.1}
HALF_LIFE, PREV_W, PRIOR_MIN = 21.0, 0.4, 150.0
SD_MIN_SCALE, SD_MIN_FLOOR, SD_MIN_CAP = 1.2, 5.5, 10.0
STARTER_BUMP = 1.5
BLOWOUT = 0.012
# Minutes after an absence, 2025-26 walk-forward (every starter and sixth man with 18+ expected minutes, where a
# normal game ran 1.0 under): the first game back ran 3.8 under after missing 8-15 league game days, 5.3 after
# 15-30 and 8.4 after 30+, and the shortfall faded over about four games (-5.0, -3.7, -2.0, -1.8, then -1.1). The
# season's first four games ran 3.3, 2.8, 2.2 and 1.9 under. Below: the shortfall beyond a normal game.
BACK_MIN = ((30, 7.5), (15, 4.4), (8, 2.9))     # league game days missed -> minutes short in the first game back
SUMMER_MIN = 2.3                                # the season's first game
BACK_RAMP = (1.0, 0.7, 0.4, 0.3)                # its share in games 1-4 back
N_SIM = 4000


def box(seasons) -> pl.DataFrame:
    pb = load("player_box", seasons).filter(pl.col("season_type").is_in([2, 5]))
    return pb.select(pl.col("game_id").cast(pl.Int64), "season", pl.col("game_date").cast(pl.Date).alias("date"),
                     pl.col("athlete_id").cast(pl.Int64), pl.col("athlete_display_name").alias("name"),
                     pl.col("team_abbreviation").alias("team"), pl.col("athlete_position_abbreviation").alias("pos"),
                     pl.col("minutes").fill_null(0.0).alias("min"), "starter", "did_not_play",
                     *[pl.col(c).fill_null(0).cast(pl.Float64).alias(k) for k, c in STATS.items()])


def _pos(p: str | None) -> str:
    p = (p or "F").upper()
    return "C" if p.startswith("C") else "G" if p.startswith(("G", "P", "S")) else "F"


def game_day_age(h: pl.DataFrame) -> pl.DataFrame:
    """Each game's age in league game days: 1 for the last day anyone played, counting back only days with games."""
    days = h.select("date").unique().sort("date", descending=True).with_columns(age=pl.int_range(1, pl.len() + 1).cast(pl.Float64))
    return h.join(days, on="date", how="left")


def _short(missed) -> pl.Expr:
    """BACK_MIN as an expression on league game days missed."""
    e = pl.when(missed >= BACK_MIN[0][0]).then(BACK_MIN[0][1])
    for lo, m in BACK_MIN[1:]:
        e = e.when(missed >= lo).then(m)
    return e.otherwise(0.0)


def profiles(bx: pl.DataFrame, before: date, season: int) -> pl.DataFrame:
    """Per player: expected minutes, minutes spread, per-minute rates (shrunk), starts share, last team;
    `last_age`, the league game days since he last played; his latest return from an absence (`back_short`, the
    minutes it cost his first game back, and `back_n`, his games since) and `season_days`, this season's game
    days so far."""
    h = game_day_age(bx.filter((pl.col("date") < before) & (pl.col("season") >= season - 1) & (pl.col("min") > 0)))
    h = h.with_columns(w=0.5 ** (pl.col("age") / HALF_LIFE) * pl.when(pl.col("season") < season).then(PREV_W).otherwise(1.0),
                       pg=pl.col("pos").map_elements(_pos, return_dtype=pl.String))
    # returns: a game after 8+ league game days without one, or his first of a season (its days missed are the
    # new season's days before it)
    first_age = h.filter(pl.col("season") == season)["age"].max() or 0.0
    g = h.sort("athlete_id", "date").with_columns(prev_age=pl.col("age").shift(1).over("athlete_id"),
                                                  prev_season=pl.col("season").shift(1).over("athlete_id"))
    new = pl.col("prev_season") < pl.col("season")
    g = g.with_columns(short=pl.when(new).then(pl.max_horizontal(_short(first_age - pl.col("age")), pl.lit(SUMMER_MIN)))
                             .otherwise(_short(pl.col("prev_age") - pl.col("age") - 1)))
    ev = (g.filter(pl.col("short") > 0).group_by("athlete_id")
           .agg(back_short=pl.col("short").sort_by("age").first(), back_age=pl.col("age").min()))
    back = (h.join(ev, on="athlete_id").filter(pl.col("age") <= pl.col("back_age"))
             .group_by("athlete_id").agg(back_short=pl.col("back_short").first(), back_n=pl.len()))
    prior = h.filter(pl.col("min") >= 10).group_by("pg").agg(*[(pl.col(k).sum() / pl.col("min").sum()).alias(f"prior_{k}") for k in STATS])
    agg = (h.sort("date").group_by("athlete_id")
             .agg(name=pl.col("name").last(), team=pl.col("team").last(), pg=pl.col("pg").last(),
                  exp_min=(pl.col("min") * pl.col("w")).sum() / pl.col("w").sum(),
                  sd_min=((pl.col("w") * (pl.col("min") - (pl.col("min") * pl.col("w")).sum() / pl.col("w").sum()) ** 2).sum() / pl.col("w").sum()).sqrt(),
                  wmin=(pl.col("min") * pl.col("w")).sum(), amin=pl.col("min").sum(),
                  starts=(pl.col("starter").cast(pl.Float64) * pl.col("w")).sum() / pl.col("w").sum(),
                  games_now=(pl.col("season") == season).sum(), last=pl.col("date").max(), last_age=pl.col("age").min(),
                  *[(pl.col(k) * pl.col("w")).sum().alias(f"s_{k}") for k in STATS],
                  *[pl.col(k).sum().alias(f"a_{k}") for k in STATS])
             .join(prior, on="pg", how="left"))
    # the prior for his recent form: his own record over both seasons, pulled toward his position's average (on
    # 2025-26 walk-forward this cut the points error from 4.77 to 4.64 a game, and from 5.37 to 4.72 in the
    # season's first three weeks, with the ranges still holding 50% / 80%)
    agg = agg.with_columns(*[((pl.col(f"a_{k}") + PRIOR_MIN * pl.col(f"prior_{k}")) / (pl.col("amin") + PRIOR_MIN)).alias(f"base_{k}") for k in STATS])
    return (agg.join(back, on="athlete_id", how="left")
               .with_columns(season_days=pl.lit(float(first_age)), sd_min=pl.col("sd_min").clip(3.5, 9.0),
                             *[((pl.col(f"s_{k}") + PRIOR_MIN * pl.col(f"base_{k}")) / (pl.col("wmin") + PRIOR_MIN)).alias(f"r_{k}") for k in STATS]))


def minutes_short(p: dict) -> float:
    """The minutes he is likely to come in short tonight after an absence (BACK_MIN), the summer included."""
    if not p.get("games_now"):                                      # his first game of the season
        missed = p.get("season_days") or 0.0
        return max(next((m for lo, m in BACK_MIN if missed >= lo), 0.0), SUMMER_MIN)
    missed = (p.get("last_age") or 1.0) - 1                         # league game days since his last game
    if missed >= BACK_MIN[-1][0]:
        return next(m for lo, m in BACK_MIN if missed >= lo)
    k = p.get("back_n")
    return float(p["back_short"]) * BACK_RAMP[k] if p.get("back_short") and k is not None and k < len(BACK_RAMP) else 0.0


def tracked(prof: pl.DataFrame, roster: pl.DataFrame, out_ids: set[int]) -> pl.DataFrame:
    """Each team's five starters and sixth man among players on its roster now and not ruled out: starters by
    how often they started lately (then minutes), the sixth man the most-used of the rest."""
    r = roster.select("athlete_id", pl.col("team").alias("now_team"), pl.col("position").alias("now_pos"), pl.col("name").alias("now_name"))
    p = (r.join(prof, on="athlete_id", how="left")
          .filter(~pl.col("athlete_id").is_in(list(out_ids)))
          .with_columns(exp_min=pl.col("exp_min").fill_null(12.0), starts=pl.col("starts").fill_null(0.0)))
    p = p.sort(["now_team", "starts", "exp_min"], descending=[False, True, True]).with_columns(
        rank=pl.int_range(1, pl.len() + 1).over("now_team"))
    starters = p.filter(pl.col("rank") <= 5).with_columns(role=pl.lit("Starter"))
    bench = (p.filter(pl.col("rank") > 5).sort(["now_team", "exp_min"], descending=[False, True])
               .group_by("now_team", maintain_order=True).head(1).with_columns(role=pl.lit("Sixth man")))
    return pl.concat([starters, bench], how="diagonal_relaxed").drop("rank")


def rotation_minutes(prof: pl.DataFrame, roster: pl.DataFrame, team: str, out_ids: set[int], top: int = 9) -> dict[int, float]:
    """Expected minutes for the team's available rotation, rescaled so the team sums to 240."""
    r = roster.filter(pl.col("team") == team).select("athlete_id").join(prof.select("athlete_id", "exp_min"), on="athlete_id", how="left")
    r = r.filter(~pl.col("athlete_id").is_in(list(out_ids))).with_columns(pl.col("exp_min").fill_null(6.0)).sort("exp_min", descending=True).head(top)
    m = r["exp_min"].to_numpy()
    if m.sum() <= 0:
        return {}
    m = np.minimum(m * 240.0 / m.sum(), 42.0)
    return dict(zip(r["athlete_id"].to_list(), m))


def simulate(exp_min: float, sd_min: float, rates: dict[str, float], pace_factor: float = 1.0,
             rng: np.random.Generator | None = None, n: int = N_SIM) -> dict[str, np.ndarray]:
    """Simulated stat lines: minutes, then each stat around rate x minutes; combos sum the same games."""
    rng = rng or np.random.default_rng(7)
    mins = np.clip(rng.normal(exp_min, sd_min, n), 0, 48)
    out = {}
    for k in STATS:
        mu = np.maximum(rates[k] * mins * pace_factor, 1e-6)
        d = DISP[k]
        if d <= 1.0001:
            out[k] = rng.poisson(mu).astype(float)
        else:
            r_ = mu / (d - 1)               # negative binomial with variance d x mean
            out[k] = rng.negative_binomial(r_, 1 / d).astype(float)
    for c, parts in COMBOS.items():
        out[c] = sum(out[p] for p in parts)
    return {k: np.sort(v) for k, v in out.items()}


# ---------------------------------------------------------------- ESPN box scores (this season, before hoopR has them)

BOX_DIR = DATA / "nba" / "box"
BOX_DIR.mkdir(parents=True, exist_ok=True)


def box_path(game_id: int):
    return BOX_DIR / f"{game_id}.json"


def espn_frame(season: int, positions: dict[int, str]) -> pl.DataFrame:
    """Cached ESPN box scores (nba/build.py saves each final) in box()'s columns."""
    rows = []
    for p in sorted(BOX_DIR.glob("*.json")):
        b = json.loads(p.read_text(encoding="utf-8"))
        day = date.fromisoformat(b["date"])
        for r in b["players"]:
            rows.append({"game_id": b["game_id"], "season": season, "date": day, "athlete_id": r["athlete_id"],
                         "name": r["name"], "team": r["team"], "pos": positions.get(r["athlete_id"]),
                         "min": r["min"] or 0.0, "starter": r["starter"], "did_not_play": not r["played"],
                         **{k: r[k] or 0.0 for k in STATS}})
    schema = {"game_id": pl.Int64, "season": pl.Int32, "date": pl.Date, "athlete_id": pl.Int64, "name": pl.String,
              "team": pl.String, "pos": pl.String, "min": pl.Float64, "starter": pl.Boolean, "did_not_play": pl.Boolean,
              **{k: pl.Float64 for k in STATS}}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def history(season: int, positions: dict[int, str]) -> pl.DataFrame:
    """hoopR player games for last season and this one, plus ESPN's for this season's games hoopR lacks."""
    bx = box(range(season - 1, season + 1))
    es = espn_frame(season, positions)
    if es.height:
        es = es.join(bx.select("game_id").unique(), on="game_id", how="anti")
        bx = pl.concat([bx, es.select(bx.columns)], how="vertical_relaxed")
    return bx


# ---------------------------------------------------------------- tonight's tracked players

def expected_minutes(prof: pl.DataFrame, roster: pl.DataFrame, team: str, out_ids: set[int], before: date) -> dict[int, float]:
    """Each available player's expected minutes: his recent average, plus his share of the minutes of rotation
    players (12+ a game) ruled out who played in the last three weeks of games (an older absence is already in
    everyone's recent averages; the summer does not count, so a starter hurt since the spring hands his minutes
    on at the start of the season), capped at 40; the top ten scaled down if they add up to more than 245."""
    r = (roster.filter(pl.col("team") == team).select("athlete_id")
         .join(prof.select("athlete_id", "exp_min", "last_age"), on="athlete_id", how="left")
         .with_columns(pl.col("exp_min").fill_null(4.0)))
    out = r.filter(pl.col("athlete_id").is_in(list(out_ids)))
    avail = r.filter(~pl.col("athlete_id").is_in(list(out_ids)))
    recent = out.filter((pl.col("exp_min") >= 12) & pl.col("last_age").is_not_null() & (pl.col("last_age") <= 21))
    lost = float(recent["exp_min"].sum())
    m = dict(zip(avail["athlete_id"].to_list(), avail["exp_min"].to_list()))
    rot = {a: v for a, v in m.items() if v >= 12}
    if lost > 0 and rot:
        tot = sum(rot.values())
        for a, v in rot.items():
            m[a] = min(v + lost * v / tot, 40.0)
    top = sorted(m.values(), reverse=True)[:10]
    if sum(top) > 245:
        f = 240.0 / sum(top)
        m = {a: v * f for a, v in m.items()}
    return m


def slate(games: list[dict], prof: pl.DataFrame, roster: pl.DataFrame, injuries: pl.DataFrame, before: date) -> pl.DataFrame:
    """Tracked players for the given games: starters and sixth man, expected minutes, minutes spread, rates.
    `games`: game_id, home, away, spread (expected home margin, market), pace_factor."""
    status = dict(zip(injuries["athlete_id"].to_list(), injuries["status"].to_list())) if injuries.height else {}
    out_ids = {a for a, s in status.items() if s in ("Out", "Doubtful")}
    rows = []
    for g in games:
        for team, opp, sign in ((g["home"], g["away"], 1), (g["away"], g["home"], -1)):
            tr = tracked(prof, roster.filter(pl.col("team") == team), out_ids)
            mins = expected_minutes(prof, roster, team, out_ids, before)
            spread = abs(g.get("spread") or 0.0)
            for p in tr.iter_rows(named=True):
                aid = p["athlete_id"]
                if p.get("r_pts") is None:      # no NBA minutes in the last two seasons: nothing to project from
                    continue
                short = minutes_short(p)
                em = mins.get(aid, p["exp_min"]) + (STARTER_BUMP if p["role"] == "Starter" else 0.0) - short
                if p["role"] == "Starter" and spread > 7:
                    em *= 1 - BLOWOUT * (spread - 7)
                rows.append({"game_id": g["game_id"], "athlete_id": aid, "name": p["now_name"] or p["name"],
                             "team": team, "opp": opp, "home": sign == 1, "pos": p["now_pos"] or p["pg"], "role": p["role"],
                             "status": status.get(aid), "exp_min": float(min(em, 42.0)), "min_short": float(short),
                             "sd_min": float(min(max((p["sd_min"] or 6.0) * SD_MIN_SCALE, SD_MIN_FLOOR), SD_MIN_CAP)),
                             "pace_factor": float(g.get("pace_factor") or 1.0), "games_now": int(p["games_now"] or 0),
                             **{f"r_{k}": float(p[f"r_{k}"]) for k in STATS}})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()   # "status" is null for most players


def samples(x: pl.DataFrame, seed: int = 17) -> dict[tuple[int, int, str], np.ndarray]:
    """Simulated outcomes per (game, player, market), each count spread evenly over its one-unit slot (k - 0.5 to
    k + 0.5). The price anchors the model to the market by shifting it until it gives the market's chance at the
    market's line (props/pricing.py); a whole-number count cannot be shifted to an exact chance, so without the
    spread every 1.5-threes line came out a few points off Pinnacle's own price and looked like an edge (opening
    night 2026: 101 of 245 markets +EV, threes averaging +6%)."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in x.iter_rows(named=True):
        s = simulate(r["exp_min"], r["sd_min"], {k: r[f"r_{k}"] for k in STATS}, r["pace_factor"], rng)
        for m, v in s.items():
            out[(r["game_id"], r["athlete_id"], m)] = np.sort(v + rng.uniform(-0.5, 0.5, len(v)))
    return out


# ---------------------------------------------------------------- pricing: the model's shape at the market's level

N_MIN = 1500


def minute_draws(exp_min: float, sd_min: float, n: int = N_MIN, seed: int = 11) -> np.ndarray:
    """Minutes, normal around the expectation and kept within 0-48 (the same draws for every player: a fixed seed,
    so a locked pick's chances can be recomputed exactly at grading)."""
    return np.clip(np.random.default_rng(seed).normal(exp_min, sd_min, n), 0, 48)


@dataclass
class Mix:
    """A player's count in one market: given his minutes, a negative binomial with mean lam x rate x minutes and
    variance `disp` times the mean (a combo's dispersion is its parts' weighted by their rates), mixed over the
    minutes draws. lam = 1 is the model; the market's level is the lam at which the mix gives the market's chance
    at the market's line. Scaling the rate (rather than shifting the curve) keeps a count a count: at 1.5 threes
    the chance of none grows with a smaller rate the way a count's does."""
    mins: np.ndarray
    rate: float
    disp: float

    @classmethod
    def of(cls, row: dict, market: str) -> "Mix":
        parts = COMBOS.get(market, (market,))
        rates = {k: row[f"r_{k}"] * row.get("pace_factor", 1.0) for k in parts}
        R = sum(rates.values())
        return cls(minute_draws(row["exp_min"], row["sd_min"]), R, sum(DISP[k] * rates[k] for k in parts) / R)

    def params(self) -> dict:
        return {"rate": round(self.rate, 6), "disp": round(self.disp, 4)}

    def _dist(self, lam: float):
        from scipy.stats import nbinom, poisson
        mu = np.maximum(lam * self.rate * self.mins, 1e-9)
        return poisson(mu) if self.disp <= 1.0001 else nbinom(mu / (self.disp - 1), 1 / self.disp)

    def probs(self, line: float, lam: float = 1.0) -> tuple[float, float, float]:
        """(over, push, under) at `line`."""
        d = self._dist(lam)
        if float(line).is_integer():
            over, push = float(d.sf(line).mean()), float(d.pmf(line).mean())
        else:
            over, push = float(d.sf(np.floor(line)).mean()), 0.0
        return over, push, max(1 - over - push, 0.0)

    def solve(self, line: float, p_over: float) -> float:
        """The rate multiplier at which the chance of the over, pushes aside, is `p_over`."""
        lo, hi = np.log(0.15), np.log(6.0)
        for _ in range(30):
            mid = (lo + hi) / 2
            o, _, u = self.probs(line, float(np.exp(mid)))
            if o / max(o + u, 1e-12) < p_over:
                lo = mid
            else:
                hi = mid
        return float(np.exp((lo + hi) / 2))

    def quantiles(self, qs=(0.1, 0.25, 0.5, 0.75, 0.9), lam: float = 1.0) -> list[float]:
        d = self._dist(lam)
        top = int(lam * self.rate * 48 * 2.5) + 25
        grid = np.arange(0, top)
        cdf = np.cumsum(d.pmf(grid[:, None]).mean(axis=1))
        return [float(grid[min(np.searchsorted(cdf, q), len(grid) - 1)]) for q in qs]

    def mean(self, lam: float = 1.0) -> float:
        return float(lam * self.rate * self.mins.mean())


@dataclass
class PropPricerNba:
    """props/pricing.py's rule on counts: the market's level from Pinnacle's no-vig price (else DraftKings and
    FanDuel's), moved toward the model by k (in log rate; 0 for now), priced at DraftKings and FanDuel.

    Pinnacle's NBA props update far less often than DraftKings' and FanDuel's (opening night 2026: 1,408 price
    records in two days against 130,251 at DraftKings; a third of its lines last changed over a day earlier), so
    when DraftKings and FanDuel both sit on the same side of Pinnacle's line, Pinnacle is taken as behind the news
    and the two of them set the level. One book off a line the other two share is the shopping edge."""
    k: float = 0.0

    def anchor(self, mix: Mix, books: dict[str, dict]) -> tuple[float | None, str | None]:
        shop = [b for b in ("draftkings", "fanduel") if b in books]
        if "pinnacle" in books:
            pin = books["pinnacle"]["line"]
            off = [np.sign(books[b]["line"] - pin) for b in shop]
            if not (len(off) == 2 and off[0] == off[1] != 0):
                return mix.solve(pin, books["pinnacle"]["p_over"]), "pinnacle"
        ls = [mix.solve(books[b]["line"], books[b]["p_over"]) for b in shop]
        return (float(np.exp(np.mean(np.log(ls)))), "dk/fd" if "pinnacle" not in books else "dk/fd, pinnacle behind") if ls else (None, None)

    def price(self, mix: Mix, books: dict[str, dict]) -> dict | None:
        lam_m, src = self.anchor(mix, books)
        if lam_m is None:
            return None
        lam = float(np.exp((1 - self.k) * np.log(lam_m)))
        best = None
        for b in ("draftkings", "fanduel"):
            if b not in books:
                continue
            L = books[b]
            over, push, under = mix.probs(L["line"], lam)
            for side, p, dec in (("over", over, L["over"]), ("under", under, L["under"])):
                ev = p * (dec - 1) - (1 - p - push)
                if best is None or ev > best["ev"]:
                    best = {"side": side, "line": L["line"], "book": b, "dec": dec, "p_win": p, "p_push": push, "ev": ev}
        if best is None:
            return None
        best.update(model_median=mix.quantiles((0.5,))[0], market_median=mix.quantiles((0.5,), lam_m)[0],
                    fair_median=mix.quantiles((0.5,), lam)[0], anchor=src, lam=lam_m, model_mean=mix.mean())
        return best
