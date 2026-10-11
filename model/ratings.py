"""Opponent-adjusted team ratings, as of any (season, week).

Each unit rating is a weighted ridge regression over past plays:

    y = mu + offense[team] + defense[opponent] + home * h

Weights decay with age in "football weeks" (the offseason counts as `gap`
extra weeks), plays in garbage time count less, and the ridge penalty pulls
every team toward league average. A snapshot for week N uses only games from
weeks before N, so the backtest can replay the season with no leakage.

Defense effects are "allowed": positive means the defense is worse.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl

from config import CLEAN

SEASON_WEEKS = 22  # regular season + playoffs; the offseason adds `gap`


@dataclass(frozen=True)
class Params:
    half_life: float = 8.0       # weeks for a play's weight to halve
    gap: float = 10.0            # offseason length, in weeks of decay
    alpha_play: float = 600.0    # ridge strength, in pseudo-plays
    alpha_game: float = 4.0      # ridge strength for game-level rows (pace, tendency)
    qb_half_life: float = 24.0   # QBs are more stable than teams
    qb_gap: float = 4.0
    alpha_qb: float = 250.0      # pseudo-dropbacks pulling a QB to his prior
    # Per-unit adaptation speeds. None = use half_life / gap above. Offense
    # and defense come from separate fits so each side can move at its own
    # pace while still being adjusted for the opponents it faced.
    hl_pass_off: float | None = None
    hl_pass_def: float | None = None
    hl_rush_off: float | None = None
    hl_rush_def: float | None = None
    hl_tend: float | None = None   # pace, pass rate over expected, plays
    hl_st: float | None = None
    gap_off: float | None = None   # how much of last season carries, offense
    gap_def: float | None = None   # ... and defense

    def speed(self, unit: str, side: str) -> tuple[float, float]:
        """(half_life, gap) for a unit ('pass', 'rush', 'tend', 'st') and side ('off', 'def')."""
        hl = getattr(self, f"hl_{unit}_{side}", None) if unit in ("pass", "rush") else getattr(self, f"hl_{unit}")
        gap = self.gap_off if side == "off" else self.gap_def
        return (hl if hl is not None else self.half_life), (gap if gap is not None else self.gap)


# Unit ratings fit on plays: name -> (row filter, target column)
PLAY_TARGETS = {
    "pass_epa": ("dropback", "epa"),
    "rush_epa": ("designed_run", "epa"),
    "pass_sr": ("dropback", "success"),
    "rush_sr": ("designed_run", "success"),
}
# Ratings fit on team-game rows (one row per team per game)
GAME_TARGETS = ["neutral_sec_per_play", "neutral_proe", "plays"]


@dataclass
class Rows:
    """Per (game, offense) sums for one target, as numpy arrays."""
    off: np.ndarray
    dfn: np.ndarray
    home: np.ndarray
    season: np.ndarray
    week: np.ndarray
    w: np.ndarray    # sum of play weights
    wy: np.ndarray   # sum of weight * y


class Inputs:
    """Everything a snapshot needs, precomputed once."""

    def __init__(self, plays: pl.DataFrame, team_games: pl.DataFrame, games: pl.DataFrame):
        self.teams = sorted(set(games["home_team"]) | set(games["away_team"]))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        tmap = pl.DataFrame({"team": self.teams, "i": list(range(len(self.teams)))})

        def rows(df: pl.DataFrame) -> Rows:
            df = (
                df.join(tmap.rename({"team": "posteam", "i": "off"}), on="posteam")
                .join(tmap.rename({"team": "defteam", "i": "dfn"}), on="defteam")
            )
            return Rows(*(df[c].to_numpy() for c in ("off", "dfn", "home", "season", "week", "w", "wy")))

        self.play_rows = {}
        for name, (flag, col) in PLAY_TARGETS.items():
            g = (
                plays.filter(pl.col(flag))
                .group_by("game_id", "season", "week", "posteam", "defteam", "home")
                .agg(w=pl.col("g").sum(), wy=(pl.col("g") * pl.col(col)).sum())
            )
            self.play_rows[name] = rows(g)

        sides = games.select("game_id", "season", "week", "home_team", "away_team", "neutral_site")
        tg = team_games.join(sides, on="game_id")
        tg = tg.with_columns(
            posteam=pl.col("team"),
            defteam=pl.when(pl.col("team") == pl.col("home_team")).then("away_team").otherwise("home_team"),
            home=((pl.col("team") == pl.col("home_team")) & (pl.col("neutral_site") == 0)).cast(pl.Int8),
        )
        self.game_rows = {}
        for col in GAME_TARGETS:
            g = tg.filter(pl.col(col).is_not_null()).with_columns(w=pl.lit(1.0), wy=pl.col(col))
            self.game_rows[col] = rows(g)
        st = tg.with_columns(w=pl.lit(1.0), wy=pl.col("st_net_epa"))
        self.st_rows = rows(st)

        # QB rows: per (game, qb) dropback sums, keeping the team and opponent.
        self.qb = (
            plays.filter(pl.col("dropback") & pl.col("qb_id").is_not_null() & pl.col("qb_epa").is_not_null())
            .group_by("game_id", "season", "week", "posteam", "defteam", "home", "qb_id")
            .agg(w=pl.col("g").sum(), wy=(pl.col("g") * pl.col("qb_epa")).sum(), n=pl.len())
        )


def _decay(season: np.ndarray, week: np.ndarray, s: int, wk: int, half_life: float, gap: float) -> np.ndarray:
    age = (s - season) * (SEASON_WEEKS + gap) + (wk - week)
    out = 0.5 ** (age / half_life)
    out[(season > s) | ((season == s) & (week >= wk))] = 0.0  # strictly before the snapshot
    return out


def ridge_off_def(r: Rows, decay: np.ndarray, alpha: float, n: int):
    """Solve the weighted ridge for mu, offense[n], defense[n], home.

    Builds X'WX from bincounts instead of a dense design matrix; that is what
    makes a full grid search over decay parameters cheap.
    """
    W = r.w * decay
    WY = r.wy * decay
    WH = W * r.home
    k = 2 * n + 2
    A = np.zeros((k, k))
    b = np.zeros(k)
    so = np.bincount(r.off, W, n)
    sd = np.bincount(r.dfn, W, n)
    A[0, 0] = W.sum()
    A[0, 1:n + 1] = A[1:n + 1, 0] = so
    A[0, n + 1:2 * n + 1] = A[n + 1:2 * n + 1, 0] = sd
    A[0, -1] = A[-1, 0] = WH.sum()
    A[1:n + 1, 1:n + 1] = np.diag(so)
    A[n + 1:2 * n + 1, n + 1:2 * n + 1] = np.diag(sd)
    od = np.bincount(r.off * n + r.dfn, W, n * n).reshape(n, n)
    A[1:n + 1, n + 1:2 * n + 1] = od
    A[n + 1:2 * n + 1, 1:n + 1] = od.T
    oh = np.bincount(r.off, WH, n)
    dh = np.bincount(r.dfn, WH, n)
    A[1:n + 1, -1] = A[-1, 1:n + 1] = oh
    A[n + 1:2 * n + 1, -1] = A[-1, n + 1:2 * n + 1] = dh
    A[-1, -1] = WH.sum()
    b[0] = WY.sum()
    b[1:n + 1] = np.bincount(r.off, WY, n)
    b[n + 1:2 * n + 1] = np.bincount(r.dfn, WY, n)
    b[-1] = (WY * r.home).sum()
    pen = np.full(k, alpha)
    pen[0] = pen[-1] = 1e-6  # mean and home field are not shrunk
    beta = np.linalg.solve(A + np.diag(pen), b)
    return beta[0], beta[1:n + 1], beta[n + 1:2 * n + 1], beta[-1]


def snapshot(inp: Inputs, season: int, week: int, p: Params) -> dict[str, np.ndarray | float]:
    """Team ratings entering (season, week). Arrays are indexed like inp.teams."""
    n = len(inp.teams)
    out: dict[str, np.ndarray | float] = {}

    def fit(r: Rows, unit: str, alpha: float):
        """Offense (with mu, home) from the offense-speed fit; defense from the defense-speed fit."""
        so, sd = p.speed(unit, "off"), p.speed(unit, "def")
        mu, off, dfn, home = ridge_off_def(r, _decay(r.season, r.week, season, week, *so), alpha, n)
        if sd != so:
            _, _, dfn, _ = ridge_off_def(r, _decay(r.season, r.week, season, week, *sd), alpha, n)
        return mu, off, dfn, home

    for name, r in inp.play_rows.items():
        unit = name.split("_")[0]  # pass_epa -> pass
        out[f"mu_{name}"], out[f"off_{name}"], out[f"def_{name}"], out[f"home_{name}"] = fit(r, unit, p.alpha_play)
    for name, r in inp.game_rows.items():
        out[f"mu_{name}"], out[f"off_{name}"], out[f"def_{name}"], _ = fit(r, "tend", p.alpha_game)
    r = inp.st_rows
    d = _decay(r.season, r.week, season, week, *p.speed("st", "off"))
    W = r.w * d
    out["st_net"] = np.bincount(r.off, r.wy * d, n) / (np.bincount(r.off, W, n) + p.alpha_game)
    return out


def all_snapshots(inp: Inputs, weeks: list[tuple[int, int]], p: Params) -> dict[tuple[int, int], dict]:
    return {(s, w): snapshot(inp, s, w, p) for s, w in weeks}


def snapshot_frame(inp: Inputs, snaps: dict[tuple[int, int], dict]) -> pl.DataFrame:
    """Flatten snapshots into one row per (season, week, team) for inspection and the board."""
    frames = []
    for (s, w), snap in snaps.items():
        cols = {"season": [s] * len(inp.teams), "week": [w] * len(inp.teams), "team": inp.teams}
        for k, v in snap.items():
            cols[k] = v if isinstance(v, np.ndarray) else [float(v)] * len(inp.teams)
        frames.append(pl.DataFrame(cols))
    return pl.concat(frames)


def load_inputs() -> tuple[Inputs, pl.DataFrame]:
    plays = pl.read_parquet(CLEAN / "plays.parquet")
    team_games = pl.read_parquet(CLEAN / "team_games.parquet")
    games = pl.read_parquet(CLEAN / "games.parquet")
    return Inputs(plays, team_games, games), games
