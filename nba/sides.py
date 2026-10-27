"""Over or under, prop by prop (user, 2026-10-27: "depending on the pick, player, team, situation, would an over or
under be good here", instead of one blanket over correction that made every pick an under).

A logistic model per market on top of the market's own price: the market's no-vig chance of the over (its logit,
with a learned slope), a learned intercept (the books' over shading for that stat, replacing props.OVER_SHADE),
and the situation, every input known before the line opens:

- dmod      the props model's disagreement with the market (logit of its chance minus the market's)
- form5     his last 5 games against this line, scaled by the line's spread
- hit10     share of his last 10 games over this line, minus a half
- mintrend  minutes in his last 3 games minus his last 10
- started   started his last game
- home, b2b (second night of a back-to-back), rest3 (3+ days off)
- margin    his team's expected margin (closing spread; /10, clipped), blowout (beyond 7 points, either way)
- total     the game's total against 225 (/10)
- oppdef    what the opponent allowed per 36 in this stat to his position group this season, against the league
- tmout     rotation minutes his team was missing in its previous game (/48)
- plus      the over at plus money; lowline: a line of 2.5 or less; early: fewer than 10 games this season

Fitted on 2022-23 and tested on 2023-24 (and the reverse), ESPN's prop history at the open (nba/prop_backtest.py).

Result (2026-10-27; research only, NOT used in pricing):
- At the US books' OPEN, 2022-24, it beats the market's own price both ways (log loss 0.68383 -> 0.68241 and
  0.68138 -> 0.68126); 10%+ EV picks +15.2% (+-8.2) and +4.5% (+-3.6). Inputs both seasons agree on: the props
  model's disagreement (z 4.5 to 11.6, every stat), recent hits over the line reversed (the line overreacts:
  points, rebounds, assists), home (+ points, assists, threes), opponent defense (assists, z 6), missing
  rotation teammates (+ rebounds), back-to-backs (- rebounds, assists); over shading real in rebounds and
  threes, ~0 in points and assists.
- This season it does not carry over. At Pinnacle near tip-off (7 nights, 1,406 props) it predicts worse than
  Pinnacle (0.6863 -> 0.6884; the blanket shading alone 0.6867); at the MORNING DraftKings/FanDuel prices its
  picks were worth -5.3% (+-0.5) at Pinnacle's close, no better than a random bet there (~-4.5%). The
  situation is in the price. Re-test with a season of the board's own data before using any of it.

    python -m nba.sides backtest
"""
import argparse
import math
from collections import defaultdict, deque

import numpy as np
import polars as pl

from config import DATA
from nba import props as NP
from nba.data import load

BT = DATA / "nba" / "prop_bt_blind.parquet"
FEATS = ["mkt", "dmod", "form5", "hit10", "mintrend", "started", "home", "b2b", "rest3", "margin", "blowout", "total",
         "oppdef", "tmout", "plus", "lowline", "early"]
LAGS = 10


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- features from box scores (pre-game only)

def player_lags(bx: pl.DataFrame) -> pl.DataFrame:
    """Per player-game he played: his previous LAGS games' stats and minutes (shifted: nothing from the game itself)."""
    g = (bx.filter(pl.col("min") > 0)
           .with_columns(**{m: sum(pl.col(c) for c in parts) for m, parts in NP.COMBOS.items()})
           .sort("athlete_id", "date", "game_id"))
    cols = []
    for m in NP.MARKETS + ["min"]:
        cols += [pl.col(m).shift(i).over("athlete_id").alias(f"{m}_l{i}") for i in range(1, LAGS + 1)]
    cols.append(pl.col("starter").shift(1).over("athlete_id").alias("started"))
    return g.select("game_id", "athlete_id", "team", "pos", "season", "date", *cols)


def team_rest(bx: pl.DataFrame) -> pl.DataFrame:
    t = bx.select("game_id", "team", "date").unique(["game_id", "team"]).sort("team", "date")
    t = t.with_columns(days=(pl.col("date") - pl.col("date").shift(1).over("team")).dt.total_days())
    return t.select("game_id", "team", b2b=(pl.col("days") == 1).cast(pl.Float64),
                    rest3=(pl.col("days").fill_null(9) >= 3).cast(pl.Float64))


def opp_defense(bx: pl.DataFrame, prior_min: float = 300.0) -> pl.DataFrame:
    """Per (game, defending team, position group): what that defense allowed per minute in each market this season
    before the game, shrunk toward the league's rate to date, as a ratio to the league's rate minus one."""
    g = (bx.filter(pl.col("min") > 0)
           .with_columns(**{m: sum(pl.col(c) for c in parts) for m, parts in NP.COMBOS.items()},
                         pg=pl.col("pos").map_elements(NP._pos, return_dtype=pl.String)))
    teams = g.select("game_id", "team").unique()
    opp = teams.join(teams, on="game_id").filter(pl.col("team") != pl.col("team_right")).rename({"team_right": "def"})
    a = (g.join(opp, on=["game_id", "team"]).group_by("season", "date", "game_id", "def", "pg")
          .agg(pl.col("min").sum(), *[pl.col(m).sum() for m in NP.MARKETS]).sort("date", "game_id"))
    a = a.with_columns(pl.col("min").cum_sum().over("season", "def", "pg").shift(1).over("season", "def", "pg").fill_null(0.0).alias("cmin"),
                       *[pl.col(m).cum_sum().over("season", "def", "pg").shift(1).over("season", "def", "pg").fill_null(0.0).alias(f"c_{m}") for m in NP.MARKETS])
    lg = (a.group_by("season", "date", "pg").agg(pl.col("min").sum(), *[pl.col(m).sum() for m in NP.MARKETS]).sort("date")
           .with_columns(lmin=pl.col("min").cum_sum().over("season", "pg") - pl.col("min"),
                         **{f"l_{m}": pl.col(m).cum_sum().over("season", "pg") - pl.col(m) for m in NP.MARKETS})
           .select("season", "date", "pg", "lmin", *[f"l_{m}" for m in NP.MARKETS]))
    a = a.join(lg, on=["season", "date", "pg"], how="left")
    out = a.select("game_id", "def", "pg", *[
        (((pl.col(f"c_{m}") + prior_min * pl.col(f"l_{m}") / pl.col("lmin")) / (pl.col("cmin") + prior_min))
         / (pl.col(f"l_{m}") / pl.col("lmin")) - 1).fill_nan(0.0).fill_null(0.0).alias(f"od_{m}") for m in NP.MARKETS])
    return out


def team_missing(bx: pl.DataFrame, rot_min: float = 20.0) -> pl.DataFrame:
    """Per team-game: rotation minutes (players averaging 20+ over their last five games, seen in the team's last
    three) missing from the team's PREVIOUS game, per 48 (known before the next game's line opens)."""
    played = bx.filter(pl.col("min") > 0).select("game_id", "team", "date", "athlete_id", "min")
    games = played.select("game_id", "team", "date").unique().sort("team", "date", "game_id")
    by_game = {k: dict(zip(v["athlete_id"].to_list(), v["min"].to_list())) for k, v in played.group_by("game_id", "team")}
    rows = []
    for (team,), tg in games.group_by("team"):
        recent: dict[int, deque] = defaultdict(lambda: deque(maxlen=5))
        last_seen: dict[int, int] = {}
        prev_missing = 0.0
        for i, gid in enumerate(tg.sort("date", "game_id")["game_id"].to_list()):
            rows.append({"game_id": gid, "team": team, "tmout": prev_missing / 48.0})
            now = by_game.get((gid, team), {})
            rot = {a: float(np.mean(d)) for a, d in recent.items() if len(d) >= 3 and np.mean(d) >= rot_min and i - last_seen[a] <= 3}
            prev_missing = sum(v for a, v in rot.items() if a not in now)
            for a, m in now.items():
                recent[a].append(m)
                last_seen[a] = i + 1
    return pl.DataFrame(rows)


def assemble(P: pl.DataFrame, bx: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """P: one row per prop with game_id, athlete_id, market, line, p (market no-vig over), o_dec, pm (model over),
    games_now, close_spread (home margin), close_total. Returns P with FEATS."""
    lags = player_lags(bx)
    P = P.join(lags, on=["game_id", "athlete_id"], how="inner")
    P = P.join(team_rest(bx), on=["game_id", "team"], how="left").join(team_missing(bx), on=["game_id", "team"], how="left")
    P = P.join(sched.select("game_id", "home"), on="game_id", how="left")
    teams = bx.select("game_id", "team").unique()
    opp = teams.join(teams, on="game_id").filter(pl.col("team") != pl.col("team_right")).rename({"team_right": "opp"})
    P = P.join(opp, on=["game_id", "team"], how="left").with_columns(pg=pl.col("pos").map_elements(NP._pos, return_dtype=pl.String))
    P = P.join(opp_defense(bx), left_on=["game_id", "opp", "pg"], right_on=["game_id", "def", "pg"], how="left")
    m = P["market"].to_numpy()
    line = P["line"].to_numpy().astype(float)
    # the market's own lags, picked per row
    L = np.full((P.height, LAGS), np.nan)
    for mk in NP.MARKETS:
        idx = np.where(m == mk)[0]
        if not len(idx):
            continue
        for i in range(1, LAGS + 1):
            L[idx, i - 1] = P[f"{mk}_l{i}"].to_numpy()[idx].astype(float)
    Mn = np.column_stack([P[f"min_l{i}"].to_numpy().astype(float) for i in range(1, LAGS + 1)])
    scale = np.sqrt(line + 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        form5 = (np.nanmean(L[:, :5], axis=1) - line) / scale
        hit10 = np.nanmean(np.where(np.isnan(L), np.nan, (L > line[:, None]).astype(float)), axis=1) - 0.5
        mintrend = (np.nanmean(Mn[:, :3], axis=1) - np.nanmean(Mn, axis=1)) / 10.0
    od = np.zeros(P.height)
    for mk in NP.MARKETS:
        idx = np.where(m == mk)[0]
        if len(idx):
            od[idx] = P[f"od_{mk}"].fill_null(0.0).to_numpy()[idx]
    home = (P["team"] == P["home"]).fill_null(False).to_numpy()
    hm = P["close_spread"].fill_null(0.0).to_numpy().astype(float)
    margin = np.where(home, hm, -hm)
    p = P["p"].to_numpy().astype(float)
    pm = P["pm"].to_numpy().astype(float)
    F = {
        "mkt": _logit(p),
        "dmod": np.nan_to_num(_logit(np.where(np.isnan(pm), p, pm)) - _logit(p)),
        "form5": np.nan_to_num(np.clip(form5, -3, 3)),
        "hit10": np.nan_to_num(hit10),
        "mintrend": np.nan_to_num(np.clip(mintrend, -1.5, 1.5)),
        "started": P["started"].fill_null(False).cast(pl.Float64).to_numpy(),
        "home": home.astype(float),
        "b2b": P["b2b"].fill_null(0.0).to_numpy(),
        "rest3": P["rest3"].fill_null(0.0).to_numpy(),
        "margin": np.clip(margin, -15, 15) / 10.0,
        "blowout": np.maximum(np.abs(margin) - 7, 0) / 10.0,
        "total": (P["close_total"].fill_null(225.0).to_numpy().astype(float) - 225.0) / 10.0,
        "oppdef": np.clip(od, -0.5, 0.5),
        "tmout": P["tmout"].fill_null(0.0).to_numpy(),
        "plus": (P["o_dec"].to_numpy().astype(float) > 2.0).astype(float),
        "lowline": (line <= 2.5).astype(float),
        "early": (P["games_now"].fill_null(0).to_numpy() < 10).astype(float),
    }
    keep = [c for c in P.columns if not any(c.startswith(f"{mk}_l") for mk in NP.MARKETS + ["min"]) and not c.startswith("od_")]
    return P.select(keep).with_columns(**{k: pl.Series(v, dtype=pl.Float64) for k, v in F.items()})


# ---------------------------------------------------------------- the model

def fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float = 2.0, prior: np.ndarray | None = None, iters: int = 30) -> np.ndarray:
    """Weighted ridge logistic regression by Newton's method; column 0 is the intercept (not penalised); `prior`
    is the mean the coefficients shrink toward (zero, except the market's slope toward one)."""
    k = X.shape[1]
    b = np.zeros(k) if prior is None else prior.copy()
    pr = np.zeros(k) if prior is None else prior
    R = np.eye(k) * lam
    R[0, 0] = 1e-6
    for _ in range(iters):
        z = X @ b
        q = 1 / (1 + np.exp(-z))
        g = X.T @ (w * (y - q)) - R @ (b - pr)
        H = (X * (w * q * (1 - q))[:, None]).T @ X + R
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-7:
            break
    return b


def design(D: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(D.height)] + [D[f].to_numpy() for f in feats])


def prior_vec(feats: list[str]) -> np.ndarray:
    v = np.zeros(len(feats) + 1)
    v[1 + feats.index("mkt")] = 1.0
    return v


def fit_all(D: pl.DataFrame, feats: list[str], lam: float = 2.0) -> dict[str, np.ndarray]:
    out = {}
    for mk in NP.MARKETS:
        S = D.filter(pl.col("market") == mk)
        if S.height < 500:
            continue
        out[mk] = fit(design(S, feats), S["y"].to_numpy().astype(float), S["w"].to_numpy(), lam, prior_vec(feats))
    return out


def predict(D: pl.DataFrame, coefs: dict[str, np.ndarray], feats: list[str]) -> np.ndarray:
    p = D["p"].to_numpy().astype(float).copy()
    m = D["market"].to_numpy()
    for mk, b in coefs.items():
        idx = np.where(m == mk)[0]
        if len(idx):
            p[idx] = 1 / (1 + np.exp(-(design(D[idx], feats) @ b)))
    return p


# ---------------------------------------------------------------- the back-test

def history() -> pl.DataFrame:
    X = pl.read_parquet(BT).filter(pl.col("season").is_in([2023, 2024]) & pl.col("actual").is_not_null()
                                   & (pl.col("actual") != pl.col("line_open")))
    X = X.select("season", "game_id", "athlete_id", "market", "provider_name", pl.col("line_open").alias("line"),
                 pl.col("p_open").alias("p"), pl.col("o_open").alias("o_dec"), pl.col("u_open").alias("u_dec"),
                 pl.col("pm_open").alias("pm"), "games_now", "close_spread", "close_total", "actual")
    X = X.with_columns(y=(pl.col("actual") > pl.col("line")).cast(pl.Float64),
                       w=1.0 / pl.len().over("game_id", "athlete_id", "market"))
    bx = NP.box([2022, 2023, 2024])
    sched = load("schedule").select(pl.col("id").cast(pl.Int64).alias("game_id"), pl.col("home_abbreviation").alias("home")).unique("game_id")
    return assemble(X, bx, sched)


def report(D: pl.DataFrame, p_hat: np.ndarray, name: str) -> None:
    y = D["y"].to_numpy()
    w = D["w"].to_numpy()
    p0 = D["p"].to_numpy()
    ll = lambda p: -np.sum(w * (y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))) / w.sum()  # noqa: E731
    o, u = D["o_dec"].to_numpy(), D["u_dec"].to_numpy()
    evo, evu = p_hat * o - 1, (1 - p_hat) * u - 1
    print(f"\n== {name}: log loss market {ll(p0):.5f}  model {ll(p_hat):.5f}  (n props {w.sum():.0f})")
    print("   EV at the open   picks  overs  unders   over ROI        under ROI       all ROI (+-2SE)")
    for lo, hi in ((0.02, 0.05), (0.05, 0.10), (0.10, 9)):
        so = (evo >= lo) & (evo < hi) & (evo >= evu)
        su = (evu >= lo) & (evu < hi) & (evu > evo)
        ro = np.where(y == 1, o - 1, -1.0)
        ru = np.where(y == 0, u - 1, -1.0)
        def stat(mask, r):
            if mask.sum() == 0:
                return "      -       "
            ww = w[mask]
            m_ = np.sum(ww * r[mask]) / ww.sum()
            return f"{m_*100:+6.1f}% n{ww.sum():5.0f}"
        allr = np.concatenate([ro[so], ru[su]])
        allw = np.concatenate([w[so], w[su]])
        m_ = np.sum(allw * allr) / max(allw.sum(), 1e-9)
        se = np.sqrt(np.sum(allw * (allr - m_) ** 2) / max(allw.sum(), 1e-9) / max(allw.sum(), 1))
        print(f"   {lo*100:3.0f}-{min(hi,1)*100:3.0f}%       {allw.sum():6.0f} {w[so].sum():6.0f} {w[su].sum():7.0f}   {stat(so, ro)}  {stat(su, ru)}  {m_*100:+5.1f}% ({2*se*100:.1f})")


def backtest() -> None:
    D = history()
    print(f"props with features: {D.height} rows, {D['w'].sum():.0f} props")
    base = ["mkt"]
    for tr, te in ((2023, 2024), (2024, 2023)):
        A, B = D.filter(pl.col("season") == tr), D.filter(pl.col("season") == te)
        print(f"\n######## fit {tr - 1}-{str(tr)[2:]}, test {te - 1}-{str(te)[2:]}")
        report(B, B["p"].to_numpy(), "market as is (even vig split)")
        report(B, predict(B, fit_all(A, base), base), "market + per-stat over shading only (the blanket rule)")
        cf = fit_all(A, FEATS)
        report(B, predict(B, cf, FEATS), "market + shading + situation (sides model)")
        if tr == 2023:
            print("\n   coefficients (fit 2022-23), logit units per unit of each input")
            print("   " + " ".join(f"{h:>8}" for h in ["market", "icpt"] + FEATS))
            for mk, b in cf.items():
                print("   " + f"{mk:>8} " + " ".join(f"{v:+8.3f}" for v in b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["backtest"])
    a = ap.parse_args()
    backtest()
