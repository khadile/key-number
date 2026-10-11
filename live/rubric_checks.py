"""Two rubric checks that need the plays, not just the totals (live/decide.py runs them):

- big play: a prop whose numbers lean on one explosive play is not a read on the
  player's game. Skip it when one play of BIG_YDS+ is at least BIG_SHARE of the
  player's yards in that stat, or he had a BIG_YDS+ play in the last RECENT_S of
  game time before the decision.
- still on the field: skip a player the play-by-play says "was injured" who has
  not touched the ball since.

The replay test (python -m live.rubric_checks) adds both to the halftime props
rule on the saved candidates of live/replay_windows.py and prints the change.
"""
import re

BIG_YDS = 20
BIG_SHARE = 0.5
RECENT_S = 120
INJURED = re.compile(r"(?:[A-Z]{2,3}-\d+-)?([A-Z][A-Za-z']*\.\s?[A-Z][A-Za-z'\-]+(?:\s(?:Jr|Sr|II|III|IV)\.?)?) was injured")
SUFFIX = re.compile(r"\s+(?:Jr|Sr|II|III|IV|V)\.?$")
YARD_MARKETS = {"rec_yds", "rush_yds", "pass_yds"}


def short_names(full: str) -> set[str]:
    """The ways a play-by-play line abbreviates a name: 'Chris Godwin Jr.' -> {'c.godwin', 'ch.godwin'}."""
    parts = SUFFIX.sub("", full.replace("'", "'")).split()
    if len(parts) < 2:
        return {full.lower()}
    first, last = parts[0].replace(".", ""), parts[-1]
    return {f"{first[:1]}.{last}".lower(), f"{first[:2]}.{last}".lower()}


def injured_in(text: str) -> list[str]:
    """Abbreviated names a play says were injured, lower-cased: ['l.mcconkey']."""
    return [m.replace(" ", "").lower() for m in INJURED.findall(text or "")]


def big_play(plays: list[tuple[float, float]], banked: float, secs_now: float) -> bool:
    """plays: (game seconds left, yards) for one player in one stat, up to the decision."""
    if not plays:
        return False
    top = max(y for _, y in plays)
    if top >= BIG_YDS and banked > 0 and top / banked >= BIG_SHARE:
        return True
    return any(y >= BIG_YDS and 0 <= s - secs_now <= RECENT_S for s, y in plays)


def flags_from_pbp(gids: list[str], before_secs: float = 1800) -> tuple[set, set]:
    """For a replay: (game, player, stat) with a big play, and (game, player) hurt and not back,
    from the play-by-play up to the decision (halftime by default)."""
    import polars as pl

    from config import RAW
    cols = ["game_id", "play_id", "game_seconds_remaining", "desc", "receiver_player_id", "rusher_player_id",
            "passer_player_id", "receiver_player_name", "rusher_player_name", "passer_player_name",
            "receiving_yards", "rushing_yards", "passing_yards", "complete_pass"]
    p = pl.read_parquet(RAW / "pbp" / "2026.parquet", columns=cols).filter(
        pl.col("game_id").is_in(gids) & (pl.col("game_seconds_remaining") > before_secs)).sort("game_id", "play_id")
    big, hurt = set(), set()
    for gid, g in p.group_by("game_id"):
        gid = gid[0]
        rows = g.to_dicts()
        names = {}
        for r in rows:
            for role in ("receiver", "rusher", "passer"):
                if r[f"{role}_player_id"] and r[f"{role}_player_name"]:
                    names[r[f"{role}_player_name"].replace(" ", "").lower()] = r[f"{role}_player_id"]
        yards: dict = {}
        last_touch: dict = {}
        hurt_at: dict = {}
        for i, r in enumerate(rows):
            s = r["game_seconds_remaining"] or 0
            for pid, m, y in ((r["receiver_player_id"], "rec_yds", r["receiving_yards"] if r["complete_pass"] == 1 else None),
                              (r["rusher_player_id"], "rush_yds", r["rushing_yards"]),
                              (r["passer_player_id"], "pass_yds", r["passing_yards"] if r["complete_pass"] == 1 else None)):
                if pid:
                    last_touch[pid] = i
                    if y is not None:
                        yards.setdefault((pid, m), []).append((s, float(y)))
            for nm in injured_in(r["desc"]):
                if nm in names:
                    hurt_at[names[nm]] = i
        for (pid, m), plays in yards.items():
            if big_play(plays, sum(y for _, y in plays), before_secs):
                big.add((gid, pid, m))
        for pid, i in hurt_at.items():
            if last_touch.get(pid, -1) <= i:      # no touch after the injury
                hurt.add((gid, pid))
    return big, hurt


def apply(C, big: set, hurt: set, event: str = "half"):
    """Drop the candidates the two checks reject at one decision point (props only)."""
    import polars as pl
    if not big and not hurt:
        return C
    B = pl.DataFrame([{"game_id": g, "player": p_, "market": m, "_big": True} for g, p_, m in big],
                     schema={"game_id": pl.String, "player": pl.String, "market": pl.String, "_big": pl.Boolean})
    H = pl.DataFrame([{"game_id": g, "player": p_, "_hurt": True} for g, p_ in hurt],
                     schema={"game_id": pl.String, "player": pl.String, "_hurt": pl.Boolean})
    X = C.join(B, on=["game_id", "player", "market"], how="left").join(H, on=["game_id", "player"], how="left")
    at = (pl.col("kind") == "prop") & (pl.col("event") == event)
    return X.filter(~(at & (pl.col("_big").fill_null(False) | pl.col("_hurt").fill_null(False)))).drop("_big", "_hurt")


def _replay_test() -> None:
    import polars as pl

    import live.replay_windows as rw
    C = pl.read_parquet(rw.CANDS)
    gids = C.filter((pl.col("kind") == "prop") & (pl.col("event") == "half"))["game_id"].unique().to_list()
    big, hurt = flags_from_pbp(gids)
    raw = {w: 1.0 for w in (1, 2, 3, 4)}

    def run(Cx):
        return rw.score(rw.select(Cx, raw, ("half",), "prop", max_gap=0.10, per_window=1, per_game=1))
    print(f"halftime: big play on {len(big)} player-stats, {len(hurt)} players hurt and not back")
    for tag, Cx in (("current rubric", C), ("+ big-play check", apply(C, big, set())),
                    ("+ still-on-field check", apply(C, set(), hurt)), ("+ both checks", apply(C, big, hurt))):
        s = run(Cx)
        print(f"{tag:24s} {s['flags']:3d} flags {s.get('w', 0)}-{s.get('l', 0)} {s.get('units', 0):+.2f}u said {s.get('said')} won {s.get('won')}")


if __name__ == "__main__":
    _replay_test()
