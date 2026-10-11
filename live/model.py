"""In-game projections: the pregame props model, updated by what has happened so far.

Every update answers: given the score, the clock, and each player's usage and
stats so far, what is the distribution of his final number, and of the final
score? It reuses the pregame model's priors (team volume, shares, efficiency)
and three things fitted on 2018-2021 play-by-play:

- script: how a team's pass rate over the rest of the game moves with the
  score (trailing teams throw more) as the game goes on;
- usage weight: how many team targets (or runs) of pregame belief one
  in-game target is worth when estimating a player's share the rest of the way;
- scoring: how likely the current drive is to end in a touchdown or a field
  goal from where the ball is, and how many points each team scores in the
  time left, scaled to the game's pregame total and spread.

The rest of the game is always played from team strength, not from the
scoreboard: a lead built on return touchdowns does not make a team better.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from config import RAW, REPORTS
from props.model import Simulator

PARAMS = REPORTS / "live_params.json"
CHECKPOINTS = (3150, 2700, 2250, 1800, 1350, 900, 450)  # seconds left


@dataclass
class LiveParams:
    a_lead: float = -0.02      # dropback-rate change per 7 points of lead
    b_lead_late: float = -0.04  # extra per 7 points, scaled by share of the game played
    n0_tgt: float = 20.0       # pregame belief in team targets
    n0_run: float = 15.0       # in team designed runs
    n0_att: float = 30.0       # QB share of attempts
    pts_scale: float = 1.0     # (unused since the curve below; kept so older files still load)
    rem_frac: list | None = None   # [(seconds left, share of the pregame total still to come), ...]
    k_game: float = 0.0        # how far Pinnacle's live win chance moves toward the model in 5 minutes
    td_by_yte: list | None = None   # P(drive ends in TD) by yards to the end zone bucket (0-9, 10-19, ...)
    fg_by_yte: list | None = None

    def save(self) -> None:
        PARAMS.write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load() -> "LiveParams":
        if PARAMS.exists():
            return LiveParams(**json.loads(PARAMS.read_text()))
        return LiveParams()


# ---------------------------------------------------------------- fitting on history

def _pbp(seasons) -> pl.DataFrame:
    cols = ["game_id", "season", "posteam", "defteam", "home_team", "away_team", "game_seconds_remaining", "play_type",
            "qb_dropback", "qb_kneel", "qb_spike", "two_point_attempt", "pass_attempt", "sack", "receiver_player_id",
            "rush_attempt", "qb_scramble", "total_home_score", "total_away_score", "yardline_100", "down",
            "fixed_drive", "fixed_drive_result", "season_type", "home_score", "away_score"]
    frames = [pl.read_parquet(RAW / "pbp" / f"{s}.parquet", columns=cols) for s in seasons]
    p = pl.concat(frames).filter((pl.col("season_type") == "REG") & pl.col("posteam").is_not_null())
    return p


def fit(seasons=(2018, 2019, 2020, 2021)) -> LiveParams:
    from props.backtest import frames
    from props.backtest import load as load_props
    from props.model import TeamModel, team_frame
    p_props, _ = load_props()
    tf = team_frame(None, p_props)
    tp = TeamModel.fit(tf).predict(tf).select("game_id", "team", "plays_hat", "db_hat")
    pbp = _pbp(seasons)
    scrim = pbp.filter(pl.col("play_type").is_in(["pass", "run"]) & (pl.col("qb_kneel").fill_null(0) == 0)
                       & (pl.col("qb_spike").fill_null(0) == 0) & (pl.col("two_point_attempt").fill_null(0) == 0))
    scrim = scrim.with_columns(lead=pl.when(pl.col("posteam") == pl.col("home_team"))
                               .then(pl.col("total_home_score") - pl.col("total_away_score"))
                               .otherwise(pl.col("total_away_score") - pl.col("total_home_score")))
    rows = []
    for cp in CHECKPOINTS:
        after = scrim.filter(pl.col("game_seconds_remaining") <= cp)
        lead = (after.sort("game_seconds_remaining", descending=True).group_by("game_id", "posteam")
                .agg(pl.col("lead").first()))
        rest = after.group_by("game_id", "posteam").agg(rest_plays=pl.len(), rest_db=pl.col("qb_dropback").sum())
        rows.append(rest.join(lead, on=["game_id", "posteam"]).with_columns(e=pl.lit(1 - cp / 3600)))
    R = pl.concat(rows).rename({"posteam": "team"}).join(tp, on=["game_id", "team"]).filter(pl.col("rest_plays") >= 5)
    y = (R["rest_db"] / R["rest_plays"] - R["db_hat"]).to_numpy()
    L = (R["lead"] / 7).clip(-3, 3).to_numpy()
    e = R["e"].to_numpy()
    w = R["rest_plays"].to_numpy()
    X = np.column_stack([np.ones_like(L), L, L * e])
    coef = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    print(f"script: base {coef[0]:+.3f}, per 7-pt lead {coef[1]:+.3f}, x elapsed {coef[2]:+.3f}")

    # usage weight: rest-of-game target share vs (prior * n0 + so far) / (n0 + team so far)
    fr = frames(p_props).filter(pl.col("season").is_between(min(seasons), max(seasons)) & pl.col("role").is_not_null())
    tg = pbp.filter((pl.col("pass_attempt") == 1) & (pl.col("sack").fill_null(0) == 0) & pl.col("receiver_player_id").is_not_null())
    best = {}
    for kind, n0s in (("tgt", (20, 40, 80, 160, 320, 640)),):
        errs = {}
        for cp in (2700, 1800, 900):
            before = tg.filter(pl.col("game_seconds_remaining") > cp)
            after_ = tg.filter(pl.col("game_seconds_remaining") <= cp)
            so = before.group_by("game_id", "posteam", "receiver_player_id").len().rename({"len": "so"})
            team_so = before.group_by("game_id", "posteam").len().rename({"len": "team_so"})
            re_ = after_.group_by("game_id", "posteam", "receiver_player_id").len().rename({"len": "re"})
            team_re = after_.group_by("game_id", "posteam").len().rename({"len": "team_re"})
            pr = fr.filter(pl.col("position").is_in(["WR", "TE", "RB"])).select(
                "game_id", pl.col("team").alias("posteam"), pl.col("gsis_id").alias("receiver_player_id"), "tshare")
            J = (pr.join(team_so, on=["game_id", "posteam"]).join(team_re, on=["game_id", "posteam"])
                 .join(so, on=["game_id", "posteam", "receiver_player_id"], how="left")
                 .join(re_, on=["game_id", "posteam", "receiver_player_id"], how="left")
                 .with_columns(pl.col("so", "re").fill_null(0)))
            for n0 in n0s:
                post = (J["tshare"] * n0 + J["so"]) / (n0 + J["team_so"])
                err = ((J["re"] - post * J["team_re"]) ** 2).sum()
                errs[n0] = errs.get(n0, 0) + float(err)
        best[kind] = min(errs, key=errs.get)
        print(f"usage weight {kind}: n0 = {best[kind]} ({ {k: round(v) for k, v in errs.items()} })")

    # drive outcomes by field position (first down plays as drive states)
    dv = pbp.filter(pl.col("down") == 1).with_columns(b=(pl.col("yardline_100") // 10).clip(0, 9))
    tab = dv.group_by("b").agg(td=(pl.col("fixed_drive_result") == "Touchdown").mean(),
                               fg=(pl.col("fixed_drive_result") == "Field goal").mean()).sort("b")
    # points the rest of the way vs the pregame expectation
    gm = pbp.group_by("game_id").agg(pl.col("home_score").first(), pl.col("away_score").first())
    g = pl.read_parquet(RAW / "schedules.parquet").select("game_id", "spread_line", "total_line")
    curve = [(3600, 1.0)]
    for cp in (3150, 2700, 2250, 1800, 1350, 900, 450, 120, 0):
        sc = (pbp.filter(pl.col("game_seconds_remaining") <= cp).sort("game_seconds_remaining", descending=True)
              .group_by("game_id").agg(pl.col("total_home_score").first().alias("h0"), pl.col("total_away_score").first().alias("a0")))
        j = sc.join(gm, on="game_id").join(g, on="game_id").drop_nulls()
        actual = (j["home_score"] - j["h0"] + j["away_score"] - j["a0"]).sum()
        curve.append((cp, float(actual / j["total_line"].sum())))
    print("share of the pregame total still to come:", [(c, round(f, 3)) for c, f in curve])
    scale = 1.0
    lp = LiveParams(a_lead=float(coef[1]), b_lead_late=float(coef[2]), n0_tgt=float(best["tgt"]),
                    n0_run=float(best["tgt"]) * 0.75, n0_att=30.0, pts_scale=scale, rem_frac=curve,
                    td_by_yte=[round(float(v), 4) for v in tab["td"]], fg_by_yte=[round(float(v), 4) for v in tab["fg"]])
    lp.save()
    return lp


# ---------------------------------------------------------------- projecting the rest of a game

PRIOR_COLS = ["gsis_id", "name", "team", "opp", "position", "role", "tshare", "rshare", "cr", "f_cr", "ypt", "f_ypt",
              "ypc", "f_ypc", "ypa", "f_ypa", "cmpr", "scr_rate", "ypscr", "att_share", "plays_hat", "db_hat",
              "dropbacks_hat", "team_targets_hat", "team_runs_hat", "team_att_hat", "spread", "total_line"]


class LiveModel:
    """Projections for one game from its pregame priors (`priors`: tracked players'
    rows from the props model, with team volume columns)."""

    def __init__(self, priors: pl.DataFrame, lp: LiveParams | None = None, sim: Simulator | None = None, size: int = 3000):
        self.p = priors.select([c for c in PRIOR_COLS if c in priors.columns])
        self.lp = lp or LiveParams.load()
        self.sim = sim or Simulator()
        self.size = size
        self.rng = np.random.default_rng(23)
        t = self.p.group_by("team").agg(pl.col("plays_hat", "db_hat", "dropbacks_hat", "team_targets_hat", "team_runs_hat",
                                               "team_att_hat", "spread", "total_line").first())
        self.team = {r["team"]: r for r in t.iter_rows(named=True)}
        self.offset: dict[tuple[str, str], float] = {}   # (player, market) -> market minus model at kickoff
        self.rem0: dict[tuple[str, str], float] = {}     # expected production at kickoff

    def calibrate(self, close_books: dict[tuple[str, str], dict]) -> None:
        """Start the game agreeing with the market. The pregame model does not beat closing
        prop lines, so its pregame opinion is not an in-game edge: shift each player's
        projection to the closing line's location (Pinnacle, else the DK/FD average), as a
        scale on production still to come. From then on only in-game information moves the
        model."""
        from live.decide import anchor_shift
        teams = list(self.team)
        k0 = self.project({"seconds_left": 3600, "score": {t: 0 for t in teams}, "possession": None,
                           "yards_to_endzone": None}, {}, {}, calibrated=False)
        for pid, mk in k0["players"].items():
            for m, s in mk.items():
                if m.startswith("_"):
                    continue
                books = close_books.get((pid, m))
                if not books:
                    continue
                ref = {"pinnacle": books["pinnacle"]} if "pinnacle" in books else books
                self.offset[(pid, m)] = anchor_shift(np.sort(s), ref)
                self.rem0[(pid, m)] = max(mk["_rem_mean"][m], 1e-6)

    def team_points(self, team: str) -> float:
        r = self.team[team]
        total = r["total_line"] or 44.0
        return (total + (r["spread"] or 0.0)) / 2

    def remaining_volume(self, team: str, opp: str, lead: float, seconds_left: float, plays_so_far: float) -> dict:
        r = self.team[team]
        e = 1 - seconds_left / 3600
        exp_so_far = r["plays_hat"] * e
        pace = float(np.clip((plays_so_far + 8) / (exp_so_far + 8), 0.8, 1.25)) ** 0.5
        plays = r["plays_hat"] * (seconds_left / 3600) * pace
        L = float(np.clip(lead / 7, -3, 3))
        db = float(np.clip(r["db_hat"] + self.lp.a_lead * L + self.lp.b_lead_late * L * e, 0.3, 0.92))
        drop = plays * db
        tpd = r["team_targets_hat"] / max(r["dropbacks_hat"], 1e-6)
        apd = r["team_att_hat"] / max(r["dropbacks_hat"], 1e-6)
        rpp = r["team_runs_hat"] / max(r["plays_hat"] - r["dropbacks_hat"], 1e-6)
        return {"plays": plays, "db": db, "drop": drop, "tgt": drop * tpd, "att": drop * apd,
                "runs": plays * (1 - db) * rpp, "pace": pace}

    def project(self, state: dict, banked: dict[str, dict], team_so_far: dict[str, dict], calibrated: bool = True) -> dict:
        """state: seconds_left, score {team: pts}, possession, yards_to_endzone.
        banked: gsis_id -> {tgt, rec, rec_yds, car, rush_yds, att, cmp, pass_yds} so far.
        team_so_far: team -> {plays, tgt, runs, att}.
        Returns {"players": {gsis_id: {market: final samples}}, "volume": {...}, "game": {...}}."""
        secs = max(float(state["seconds_left"]), 0.0)
        teams = list(self.team)
        vol = {}
        for t in teams:
            o = [x for x in teams if x != t][0]
            lead = state["score"].get(t, 0) - state["score"].get(o, 0)
            vol[t] = self.remaining_volume(t, o, lead, secs, team_so_far.get(t, {}).get("plays", 0))
        rows = []
        for r in self.p.iter_rows(named=True):
            t, v, so = r["team"], vol[r["team"]], team_so_far.get(r["team"], {})
            b = banked.get(r["gsis_id"], {})
            lp = self.lp
            tsh = (r["tshare"] * lp.n0_tgt + b.get("tgt", 0)) / (lp.n0_tgt + so.get("tgt", 0))
            rsh = (r["rshare"] * lp.n0_run + b.get("car_designed", b.get("car", 0))) / (lp.n0_run + so.get("runs", 0))
            ash = ((r.get("att_share") or 0.97) * lp.n0_att + b.get("att", 0)) / (lp.n0_att + so.get("att", 0)) if r["position"] == "QB" else 0.0
            tgt = v["tgt"] * tsh
            att = v["att"] * ash
            rows.append({"position": r["position"], "tgt_hat": tgt, "rec_hat": tgt * float(np.clip(r["cr"] * r["f_cr"], 0.2, 0.95)),
                         "rec_yds_hat": tgt * r["ypt"] * r["f_ypt"], "car_hat": v["runs"] * rsh, "ypc": r["ypc"],
                         "f_ypc": r["f_ypc"], "scr_hat": v["drop"] * (r["scr_rate"] if r["position"] == "QB" else 0.0),
                         "ypscr": r["ypscr"], "att_hat": att, "cmp_hat": att * r["cmpr"],
                         "pass_yds_hat": att * r["ypa"] * r["f_ypa"], "tshare_now": tsh, "rshare_now": rsh})
        R = pl.DataFrame(rows)
        S = self.sim.simulate(R, self.size, raw=True)
        players = {}
        for i, r in enumerate(self.p.iter_rows(named=True)):
            pid = r["gsis_id"]
            b = banked.get(pid, {})
            rem = {"receptions": S["receptions"][i] + self.rng.uniform(-0.5, 0.5, self.size), "rec_yds": S["rec_yds"][i],
                   "rush_yds": S["rush_yds"][i], "pass_yds": S["pass_yds"][i]}
            rem_mean = {m: float(S[m][i].mean()) for m in rem}
            if calibrated:
                # the kickoff offset as a scale on production still to come, so zero stays zero
                # and the offset fades as the game runs out
                for m in rem:
                    if (pid, m) in self.offset:
                        rem[m] = rem[m] * float(np.clip((self.rem0[(pid, m)] + self.offset[(pid, m)]) / self.rem0[(pid, m)], 0.3, 3.0))
            have = {"receptions": b.get("rec", 0), "rec_yds": b.get("rec_yds", 0), "rush_yds": b.get("rush_yds", 0),
                    "pass_yds": b.get("pass_yds", 0)}
            players[pid] = {**{m: have[m] + rem[m] for m in rem},
                            "_share": (rows[i]["tshare_now"], rows[i]["rshare_now"]),
                            "_rem_mean": {m: float(np.median(rem[m])) for m in rem}}  # typical, not average: a few
                                                                                    # big games skew the average
        return {"players": players, "volume": vol, "game": self.game(state)}

    def share_left(self, secs: float) -> float:
        """Share of a game's points still to come with `secs` left (scoring picks up late in halves)."""
        if not self.lp.rem_frac:
            return secs / 3600
        xs, ys = zip(*sorted(self.lp.rem_frac))
        return float(np.interp(secs, xs, ys))

    def game(self, state: dict, size: int = 6000) -> dict:
        """Final score distribution: the current drive (from field position), then
        touchdowns and field goals at each team's pregame rate for the time left."""
        secs = max(float(state["seconds_left"]), 0.0)
        teams = list(self.team)
        rng = self.rng
        final = {}
        lp = self.lp
        for t in teams:
            has_ball = state.get("possession") == t and secs > 0
            fut_secs = max(secs - (120 if has_ball else 0), 0)
            E = self.team_points(t) * self.share_left(fut_secs)
            td = rng.poisson(0.72 * E / 6.96, size)
            fg = rng.poisson(0.28 * E / 3.0, size)
            pts = 6.96 * td + 3 * fg
            if has_ball and state.get("yards_to_endzone") is not None and lp.td_by_yte:
                bkt = int(min(max(state["yards_to_endzone"] // 10, 0), 9))
                u = rng.uniform(size=size)
                p_td, p_fg = lp.td_by_yte[bkt], lp.fg_by_yte[bkt]
                pts = pts + np.where(u < p_td, 6.96, np.where(u < p_td + p_fg, 3.0, 0.0))
            final[t] = state["score"].get(t, 0) + np.round(pts)
        a, b = teams
        margin = final[a] - final[b]
        # regulation ties go to overtime, roughly a coin flip for 3 points
        tie = margin == 0
        margin = np.where(tie, np.where(rng.uniform(size=size) < 0.5, 3, -3), margin)
        return {"teams": teams, "margin": margin, "total": final[a] + final[b]}
