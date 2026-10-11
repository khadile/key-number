"""Player prop projections: team volume x player share x efficiency, as full distributions.

How one player's projection for one game is built (inputs use only games
before it):

1. Team volume. Expected offensive plays and dropback rate, from the team's
   recent pace and pass rate, what the opponent's defense has allowed, and the
   game's spread and total (a regression fitted on 2017-2021). Targets, pass
   attempts and designed runs follow from the team's own ratios.
2. The player's share of team targets and designed runs in the games he
   played, recent games weighted more (exponential decay in football weeks,
   with an offseason gap), shrunk toward a typical share for his position.
   Teammates ruled out hand their shares to the players who remain.
3. Efficiency: catch rate, yards per target, yards per carry, yards per pass
   attempt. Decayed, shrunk hard toward position averages (most of it is
   noise), and adjusted for what this defense has allowed.
4. Distribution: simulated games. Counts are negative binomial around the
   mean; yards add up per-catch, per-carry and per-completion yardage drawn
   from real plays (2016-2021), scaled to the player's efficiency.

The game's spread and total are inputs on purpose: the props market is what
this model is measured against, and a game expected to be a 51-point shootout
really does produce more passing yards. Backtests use nflverse's closing
lines, which a Saturday lock nearly matches; the board uses the lines at lock.
"""
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from props.data import PLAYER_GAMES, TEAM_GAMES, YARD_POOLS

WEEKS = 18
FIRST = 2016
POS = ("QB", "RB", "WR", "TE")
MARKETS = ("pass_yds", "rush_yds", "rec_yds", "receptions")
STAT_COL = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards",
            "receptions": "receptions"}


@dataclass
class Params:
    gap: float = 8.0         # offseason gap, in weeks
    hl_team: float = 8.0     # team volume half-life (weeks)
    hl_share: float = 5.0    # usage share half-life
    hl_eff: float = 16.0     # efficiency half-life
    hl_def: float = 12.0     # defense-allowed half-life
    m_share: float = 30.0    # prior weight, in team targets / designed runs
    m_cr: float = 60.0       # catch-rate prior weight, in targets
    m_ypt: float = 100.0     # yards-per-target prior weight, in targets
    m_ypc: float = 150.0     # yards-per-carry prior weight, in carries
    m_ypa: float = 300.0     # yards-per-attempt prior weight, in attempts
    m_def: float = 250.0     # defense factor prior weight, in targets / carries / attempts
    def_power: float = 1.0   # how much of the defense factor to apply

    def asdict(self) -> dict:
        return asdict(self)


# Position priors: typical share among players who play, and league efficiency.
TSHARE0 = {"WR": 0.14, "TE": 0.11, "RB": 0.08, "QB": 0.0}
RSHARE0 = {"RB": 0.30, "QB": 0.06, "WR": 0.01, "TE": 0.0}
CR0 = {"WR": 0.63, "TE": 0.70, "RB": 0.77, "QB": 0.6}
YPT0 = {"WR": 8.2, "TE": 7.2, "RB": 5.8, "QB": 5.0}
YPC0 = {"RB": 4.3, "QB": 4.6, "WR": 6.5, "TE": 3.0}
YPA0, CMP0, SCR0, YPSCR0 = 7.0, 0.645, 0.045, 7.5


def tindex(season, week, gap: float):
    """Continuous time in football weeks, with an offseason gap between seasons."""
    return (season - FIRST) * (WEEKS + gap) + week


def ew_prior(df: pl.DataFrame, by: str, cols: list[str], hl: float, suffix: str) -> pl.DataFrame:
    """Exponentially weighted sums of `cols` over each group's EARLIER rows, at each row's time.

    sum_{j before k} x_j * 2^-((t_k - t_j)/hl) = 2^(-t_k/hl) * sum_{j<k} x_j 2^(t_j/hl): a running
    sum of rescaled values minus the row's own term. Rows must not share a time within a group."""
    df = df.sort(by, "t")
    f = pl.lit(2.0) ** (pl.col("t") / hl)
    out = []
    for c in cols:
        x = pl.col(c).fill_null(0).cast(pl.Float64) * f
        out.append(((x.cum_sum().over(by) - x) / f).alias(f"{c}{suffix}"))
    return df.with_columns(out)


# ---------------------------------------------------------------- team volume

TEAM_COLS = ["plays", "dropbacks", "targets", "pass_att", "designed_runs", "sacks", "scrambles", "pass_yds", "one"]


def team_frame(upcoming: pl.DataFrame | None, p: Params) -> pl.DataFrame:
    """Team-game rows (history plus any upcoming games, whose stats are null) with
    decayed offense and defense-allowed totals from earlier games."""
    tg = pl.read_parquet(TEAM_GAMES)
    if upcoming is not None and upcoming.height:
        tg = pl.concat([tg, upcoming.select([c for c in upcoming.columns if c in tg.columns])], how="diagonal_relaxed")
    tg = tg.with_columns(t=tindex(pl.col("season"), pl.col("week"), p.gap).cast(pl.Float64),
                         one=pl.when(pl.col("plays").is_not_null()).then(1.0).otherwise(0.0))
    tg = ew_prior(tg, "team", TEAM_COLS, p.hl_team, "_o")
    tg = ew_prior(tg, "opp", TEAM_COLS, p.hl_team, "_d")
    eps = 1e-9
    lg_db, lg_tpd, lg_apd, lg_ypa = 0.60, 0.84, 0.88, 7.0
    return tg.with_columns(
        o_ppg=pl.col("plays_o") / (pl.col("one_o") + eps),
        d_ppg=pl.col("plays_d") / (pl.col("one_d") + eps),
        o_db=(pl.col("dropbacks_o") + 100 * lg_db) / (pl.col("plays_o") + 100),
        d_db=(pl.col("dropbacks_d") + 100 * lg_db) / (pl.col("plays_d") + 100),
        o_tpd=(pl.col("targets_o") + 150 * lg_tpd) / (pl.col("dropbacks_o") + 150),
        o_apd=(pl.col("pass_att_o") + 150 * lg_apd) / (pl.col("dropbacks_o") + 150),
        o_scr=(pl.col("scrambles_o") + 150 * SCR0) / (pl.col("dropbacks_o") + 150),
        o_rpp=(pl.col("designed_runs_o") + 100 * 0.99) / (pl.col("plays_o") - pl.col("dropbacks_o") + 100),
        d_ypa=((pl.col("pass_yds_d") + p.m_def * lg_ypa) / (pl.col("pass_att_d") + p.m_def)) / lg_ypa,
        db_rate=pl.col("dropbacks") / pl.col("plays"),
    )


def _design(df: pl.DataFrame, kind: str) -> np.ndarray:
    if kind == "plays":
        cols = [pl.lit(1.0), pl.col("o_ppg"), pl.col("d_ppg"), pl.col("total_line"), pl.col("spread")]
    else:
        cols = [pl.lit(1.0), pl.col("o_db"), pl.col("d_db"), pl.col("spread"), pl.col("total_line")]
    return df.select([c.cast(pl.Float64).alias(f"x{i}") for i, c in enumerate(cols)]).to_numpy()


@dataclass
class TeamModel:
    b_plays: np.ndarray
    b_db: np.ndarray

    @staticmethod
    def fit(tf: pl.DataFrame, seasons=(2017, 2021)) -> "TeamModel":
        d = tf.filter(pl.col("season").is_between(*seasons) & pl.col("plays").is_not_null()
                      & (pl.col("one_o") > 1.5) & (pl.col("one_d") > 1.5) & pl.col("total_line").is_not_null())
        Xp, Xd = _design(d, "plays"), _design(d, "db")
        bp = np.linalg.lstsq(Xp, d["plays"].to_numpy().astype(float), rcond=None)[0]
        bd = np.linalg.lstsq(Xd, d["db_rate"].to_numpy().astype(float), rcond=None)[0]
        return TeamModel(bp, bd)

    def predict(self, tf: pl.DataFrame) -> pl.DataFrame:
        # Teams with no history yet (a 2016 week 1) fall back to league averages.
        tf = tf.with_columns(
            o_ppg=pl.when(pl.col("one_o") > 0.5).then(pl.col("o_ppg")).otherwise(62.0),
            d_ppg=pl.when(pl.col("one_d") > 0.5).then(pl.col("d_ppg")).otherwise(62.0),
            total_line=pl.col("total_line").fill_null(44.0), spread=pl.col("spread").fill_null(0.0))
        plays = _design(tf, "plays") @ self.b_plays
        db = np.clip(_design(tf, "db") @ self.b_db, 0.4, 0.8)
        tf = tf.with_columns(plays_hat=pl.Series(plays), db_hat=pl.Series(db))
        return tf.with_columns(
            dropbacks_hat=pl.col("plays_hat") * pl.col("db_hat"),
            team_targets_hat=pl.col("plays_hat") * pl.col("db_hat") * pl.col("o_tpd"),
            team_att_hat=pl.col("plays_hat") * pl.col("db_hat") * pl.col("o_apd"),
            team_runs_hat=pl.col("plays_hat") * (1 - pl.col("db_hat")) * pl.col("o_rpp"),
        )


# ---------------------------------------------------------------- players

SHARE_COLS = ["targets", "team_targets", "d_car", "team_runs", "att", "team_att", "scr", "team_dropbacks", "one"]
EFF_COLS = ["receptions", "receiving_yards", "targets", "d_yds", "d_car", "passing_yards", "att", "cmp",
            "scr_yds", "scr"]


def defense_frame(pg: pl.DataFrame, tf: pl.DataFrame, p: Params) -> pl.DataFrame:
    """What each defense has allowed, by position group, decayed: one row per (game_id, opp)."""
    g = pg.filter(pl.col("position").is_in(["WR", "TE", "RB"]) & pl.col("targets").is_not_null())
    agg = g.group_by("game_id", "opp", "position").agg(
        tgt=pl.col("targets").sum(), rec=pl.col("receptions").sum(), yds=pl.col("receiving_yards").sum(),
        car=pl.col("d_car").sum(), ryds=pl.col("d_yds").sum())
    wide = agg.pivot(on="position", index=["game_id", "opp"], values=["tgt", "rec", "yds", "car", "ryds"])
    base = tf.select("game_id", "opp", "t", "plays").join(wide, on=["game_id", "opp"], how="left")
    cols = [c for c in base.columns if c not in ("game_id", "opp", "t", "plays")]
    base = ew_prior(base, "opp", cols, p.hl_def, "_a")
    lg = {"WR": (8.2, 0.63), "TE": (7.2, 0.70), "RB": (5.8, 0.77)}
    ex = []
    for pos, (ypt, cr) in lg.items():
        tg, rc, yd = (pl.col(f"{k}_{pos}_a").fill_null(0) if f"{k}_{pos}_a" in base.columns else pl.lit(0.0)
                      for k in ("tgt", "rec", "yds"))
        ex.append((((yd + p.m_def * ypt) / (tg + p.m_def)) / ypt).alias(f"f_ypt_{pos}"))
        ex.append((((rc + p.m_def * cr) / (tg + p.m_def)) / cr).alias(f"f_cr_{pos}"))
    car = pl.col("car_RB_a").fill_null(0) if "car_RB_a" in base.columns else pl.lit(0.0)
    ryd = pl.col("ryds_RB_a").fill_null(0) if "ryds_RB_a" in base.columns else pl.lit(0.0)
    ex.append((((ryd + p.m_def * 4.3) / (car + p.m_def)) / 4.3).alias("f_ypc"))
    return base.with_columns(ex).select("game_id", "opp", *[e.meta.output_name() for e in ex])


def player_frame(upcoming_players: pl.DataFrame | None, p: Params) -> pl.DataFrame:
    """Player-game rows (history plus upcoming placeholders) with decayed usage and efficiency."""
    pg = pl.read_parquet(PLAYER_GAMES).filter(pl.col("position").is_in(list(POS)) | (pl.col("position") == "FB"))
    pg = pg.with_columns(position=pl.when(pl.col("position") == "FB").then(pl.lit("RB")).otherwise(pl.col("position")))
    if upcoming_players is not None and upcoming_players.height:
        pg = pl.concat([pg, upcoming_players], how="diagonal_relaxed")
    pg = pg.with_columns(t=tindex(pl.col("season"), pl.col("week"), p.gap).cast(pl.Float64),
                         one=pl.when(pl.col("targets").is_not_null()).then(1.0).otherwise(0.0))
    pg = pg.unique(["gsis_id", "t"], keep="first")  # one row per player per week
    pg = ew_prior(pg, "gsis_id", SHARE_COLS, p.hl_share, "_s")
    pg = ew_prior(pg, "gsis_id", EFF_COLS, p.hl_eff, "_e")
    return pg


def _by_pos(d: dict, default: float) -> pl.Expr:
    e = pl.lit(default)
    for k, v in d.items():
        e = pl.when(pl.col("position") == k).then(pl.lit(v)).otherwise(e)
    return e


def project(pf: pl.DataFrame, team: pl.DataFrame, dfn: pl.DataFrame, p: Params) -> pl.DataFrame:
    """Expected stats for every player-game row in `pf` (the rows to project)."""
    x = pf.join(team.select("game_id", "team", "plays_hat", "db_hat", "dropbacks_hat", "team_targets_hat",
                            "team_att_hat", "team_runs_hat", "o_scr", "d_ypa", "spread", "total_line"),
                on=["game_id", "team"], how="left")
    x = x.join(dfn, on=["game_id", "opp"], how="left")
    pos_f = {}
    for k in ("f_ypt", "f_cr"):
        e = pl.lit(1.0)
        for pos in ("WR", "TE", "RB"):
            e = pl.when(pl.col("position") == pos).then(pl.col(f"{k}_{pos}")).otherwise(e)
        pos_f[k] = e.fill_null(1.0) ** p.def_power
    x = x.with_columns(
        tshare=(pl.col("targets_s") + p.m_share * _by_pos(TSHARE0, 0.05)) / (pl.col("team_targets_s") + p.m_share),
        rshare=(pl.col("d_car_s") + p.m_share * _by_pos(RSHARE0, 0.0)) / (pl.col("team_runs_s") + p.m_share),
        att_share=(pl.col("att_s") + 40 * 0.97) / (pl.col("team_att_s") + 40),
        scr_rate=(pl.col("scr_s") + 60 * SCR0) / (pl.col("team_dropbacks_s") + 60),
        cr=(pl.col("receptions_e") + p.m_cr * _by_pos(CR0, 0.6)) / (pl.col("targets_e") + p.m_cr),
        ypt=(pl.col("receiving_yards_e") + p.m_ypt * _by_pos(YPT0, 7.0)) / (pl.col("targets_e") + p.m_ypt),
        ypc=(pl.col("d_yds_e") + p.m_ypc * _by_pos(YPC0, 4.3)) / (pl.col("d_car_e") + p.m_ypc),
        ypa=(pl.col("passing_yards_e") + p.m_ypa * YPA0) / (pl.col("att_e") + p.m_ypa),
        cmpr=(pl.col("cmp_e") + p.m_ypa * CMP0) / (pl.col("att_e") + p.m_ypa),
        ypscr=(pl.col("scr_yds_e") + 40 * YPSCR0) / (pl.col("scr_e") + 40),
        f_ypt=pos_f["f_ypt"], f_cr=pos_f["f_cr"],
        f_ypc=pl.col("f_ypc").fill_null(1.0) ** p.def_power,
        f_ypa=pl.col("d_ypa").fill_null(1.0) ** p.def_power,
    )
    return x.with_columns(
        tgt_hat=pl.col("team_targets_hat") * pl.col("tshare"),
        car_hat=pl.col("team_runs_hat") * pl.col("rshare"),
    ).with_columns(
        rec_hat=pl.col("tgt_hat") * (pl.col("cr") * pl.col("f_cr")).clip(0.2, 0.95),
        rec_yds_hat=pl.col("tgt_hat") * pl.col("ypt") * pl.col("f_ypt"),
        att_hat=pl.when(pl.col("position") == "QB").then(pl.col("team_att_hat") * pl.col("att_share")).otherwise(0.0),
        scr_hat=pl.when(pl.col("position") == "QB").then(pl.col("dropbacks_hat") * pl.col("scr_rate")).otherwise(0.0),
    ).with_columns(
        rush_yds_hat=pl.col("car_hat") * pl.col("ypc") * pl.col("f_ypc") + pl.col("scr_hat") * pl.col("ypscr"),
        pass_yds_hat=pl.col("att_hat") * pl.col("ypa") * pl.col("f_ypa"),
        cmp_hat=pl.col("att_hat") * pl.col("cmpr"),
    )


# ---------------------------------------------------------------- distributions

@dataclass
class Dispersion:
    """Negative binomial size for each count (bigger = closer to Poisson), the
    per-game spread of efficiency (lognormal sigma) that counts alone miss, and a
    median shift per market. All fitted on 2018-2021 so the forecasts are calibrated:
    results land above the simulated median half the time, and in each decile 10%."""
    r_tgt: float = 6.0
    r_car: float = 5.0
    r_att: float = 15.0
    r_scr: float = 2.0
    sd_rec: float = 0.18
    sd_rush: float = 0.18
    sd_pass: float = 0.18
    shift: dict | None = None  # market -> units added to every sample

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class Simulator:
    """Simulated stat lines. Sums of k plays are drawn from banks of real-play sums,
    so a 60-yard catch is as likely as it really is."""

    def __init__(self, disp: Dispersion | None = None, bank_n: int = 4096, seed: int = 7):
        self.disp = disp or Dispersion()
        self.rng = np.random.default_rng(seed)
        pools = pl.read_parquet(YARD_POOLS)
        self.banks, self.means = {}, {}
        sizes = {"catch_WR": 24, "catch_TE": 20, "catch_RB": 16, "run_RB": 40, "run_QB": 20, "run_WR": 8,
                 "scramble": 16, "completion": 50}
        for name, kmax in sizes.items():
            v = pools.filter(pl.col("pool") == name)["yds"].to_numpy()
            self.means[name] = float(v.mean())
            draws = self.rng.choice(v, size=(bank_n, kmax))
            sums = np.concatenate([np.zeros((bank_n, 1)), np.cumsum(draws, axis=1)], axis=1)  # k = 0..kmax
            self.banks[name] = sums.T.copy()  # [k, j]

    def _nb(self, mean: np.ndarray, r: float, size: int) -> np.ndarray:
        mean = np.maximum(mean, 1e-6)[:, None]
        lam = self.rng.gamma(r, mean / r, size=(mean.shape[0], size))
        return self.rng.poisson(lam)

    def _sum(self, pool: str, k: np.ndarray) -> np.ndarray:
        bank = self.banks[pool]
        k = np.minimum(k, bank.shape[0] - 1)
        return bank[k, self.rng.integers(0, bank.shape[1], size=k.shape)]

    def _eff(self, n: int, size: int, s: float) -> np.ndarray:
        return np.exp(self.rng.normal(-s * s / 2, s, size=(n, size)))

    def receiving(self, tgt: np.ndarray, rec: np.ndarray, yds: np.ndarray, pos: np.ndarray, size: int):
        """(receptions, receiving yards) samples, shape [players, size]."""
        T = self._nb(tgt, self.disp.r_tgt, size)
        cr = np.clip(rec / np.maximum(tgt, 1e-6), 0.05, 0.98)[:, None]
        R = self.rng.binomial(T, np.broadcast_to(cr, T.shape))
        Y = np.zeros(R.shape)
        ypr = yds / np.maximum(rec, 1e-6)
        for g in ("WR", "TE", "RB"):
            m = pos == g
            if m.any():
                pool = f"catch_{g}"
                scale = (ypr[m] / self.means[pool])[:, None]
                Y[m] = self._sum(pool, R[m]) * scale * self._eff(int(m.sum()), size, self.disp.sd_rec)
        return R, Y

    def rushing(self, car: np.ndarray, ypc_total: np.ndarray, pos: np.ndarray, scr: np.ndarray,
                scr_yds: np.ndarray, size: int) -> np.ndarray:
        """Rushing yards samples: designed carries plus (for QBs) scrambles."""
        C = self._nb(car, self.disp.r_car, size)
        Y = np.zeros(C.shape)
        ypc = ypc_total / np.maximum(car, 1e-6)
        for g, pool in (("RB", "run_RB"), ("QB", "run_QB"), ("WR", "run_WR"), ("TE", "run_RB")):
            m = pos == g
            if m.any():
                scale = (ypc[m] / self.means[pool])[:, None]
                Y[m] = self._sum(pool, C[m]) * scale * self._eff(int(m.sum()), size, self.disp.sd_rush)
        q = scr > 0
        if q.any():
            S = self._nb(scr[q], self.disp.r_scr, size)
            yps = scr_yds[q] / np.maximum(scr[q], 1e-6)
            Y[q] += self._sum("scramble", S) * (yps / self.means["scramble"])[:, None]
        return Y

    def passing(self, att: np.ndarray, cmp: np.ndarray, yds: np.ndarray, size: int) -> np.ndarray:
        A = self._nb(att, self.disp.r_att, size)
        cr = np.clip(cmp / np.maximum(att, 1e-6), 0.3, 0.85)[:, None]
        C = self.rng.binomial(A, np.broadcast_to(cr, A.shape))
        ypcmp = yds / np.maximum(cmp, 1e-6)
        return (self._sum("completion", C) * (ypcmp / self.means["completion"])[:, None]
                * self._eff(len(att), size, self.disp.sd_pass))

    def simulate(self, proj: pl.DataFrame, size: int = 4000, raw: bool = False) -> dict[str, np.ndarray]:
        """Samples for every market, rows aligned with `proj`. Receptions come back
        jittered (see `jitter`) unless `raw`; every market gets its calibrated shift."""
        pos = proj["position"].to_numpy()
        f = lambda c: proj[c].fill_null(0).to_numpy().astype(float)
        R, Y = self.receiving(f("tgt_hat"), f("rec_hat"), f("rec_yds_hat"), pos, size)
        car_yds = f("car_hat") * f("ypc") * f("f_ypc")
        RU = self.rushing(f("car_hat"), car_yds, pos, f("scr_hat"), f("scr_hat") * f("ypscr"), size)
        PY = self.passing(f("att_hat"), f("cmp_hat"), f("pass_yds_hat"), size)
        out = {"receptions": R.astype(float), "rec_yds": Y, "rush_yds": RU, "pass_yds": PY}
        if not raw:
            out["receptions"] = jitter(out["receptions"], "receptions", self.rng)
            # calibration: receptions move by a fixed amount, yards scale (so zero stays zero)
            for m, d in (self.disp.shift or {}).items():
                out[m] = out[m] + d if m == "receptions" else out[m] * d
        return out


def jitter(samples: np.ndarray, market: str, rng: np.random.Generator) -> np.ndarray:
    """Counts get +-0.5 uniform noise so shifting a distribution and reading
    P(X > 4.5) both work; P(X > 4.5) is unchanged by it."""
    if market == "receptions":
        return samples + rng.uniform(-0.5, 0.5, size=samples.shape)
    return samples
