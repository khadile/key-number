"""Who is who in the props data.

OddsPapi keys prop prices by its own player ids, nflverse keys everything by
gsis_id, and PropLine and ESPN use ESPN athlete ids. This maps OddsPapi ids to
gsis_id (and espn_id) from OddsPapi's player list, which is one billable call
cached forever in data/oddspapi/players_s14.json, matched by name within the
player's team. Run again (it only re-reads the cache) after refreshing that file.

    python -m pipeline.prop_players
"""
import json
import re
import unicodedata

import polars as pl

from config import CLEAN, RAW
from pipeline.odds_history import CACHE, TEAMS

OUT = CLEAN / "prop_players.parquet"
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(name: str | None) -> str:
    """Lowercase ASCII letters and single spaces; no punctuation or suffixes."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`’\-]", "", s)
    s = SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _flip(name: str) -> str:
    """OddsPapi writes "Last, First"; everything else writes "First Last"."""
    last, _, first = name.partition(",")
    return f"{first.strip()} {last.strip()}".strip() if first else name.strip()


def oddspapi_players() -> pl.DataFrame:
    raw = json.loads((CACHE / "players_s14.json").read_text(encoding="utf-8"))["participants"]
    fx = json.loads((CACHE / "fixtures_t31.json").read_text(encoding="utf-8"))
    team_of = {}
    for f in fx:
        for side in ("1", "2"):
            abbr = TEAMS.get(f[f"participant{side}Name"])
            if abbr:
                team_of[str(f[f"participant{side}Id"])] = abbr
    rows = [{"op_id": p["playerId"], "op_name": _flip(p["playerName"] or ""), "op_team": team_of[pid]}
            for pid, v in raw.items() if pid in team_of for p in v["players"] if p.get("playerName")]
    return pl.DataFrame(rows).unique("op_id")


def nfl_people(season: int) -> pl.DataFrame:
    """Every rostered player this season and last, with the name spellings worth matching on."""
    frames = []
    for s in (season - 1, season):
        path = RAW / "rosters_weekly" / f"{s}.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path).select(
                "season", "week", "team", "gsis_id", "espn_id", "position", "full_name", "first_name",
                "last_name", "football_name"))
    r = pl.concat(frames).filter(pl.col("gsis_id").is_not_null()).sort("season", "week")
    latest = r.group_by("gsis_id").agg(pl.all().last())
    teams = r.filter(pl.col("season") == season).group_by("gsis_id").agg(pl.col("team").unique().alias("teams"))
    p = pl.read_parquet(RAW / "players.parquet").select("gsis_id", "display_name", "common_first_name")
    return latest.join(teams, on="gsis_id", how="left").join(p, on="gsis_id", how="left")


def build(season: int) -> pl.DataFrame:
    op = oddspapi_players().with_columns(key=pl.col("op_name").map_elements(norm, return_dtype=pl.String))
    people = nfl_people(season)
    # One row per (gsis_id, spelling, team the player was on this season or most recently).
    variants = []
    for r in people.iter_rows(named=True):
        names = {r["full_name"], r["display_name"],
                 f"{r['first_name']} {r['last_name']}", f"{r['football_name']} {r['last_name']}",
                 f"{r['common_first_name']} {r['last_name']}"}
        for t in set(r["teams"] or []) | {r["team"]}:
            for n in names:
                if n and "None" not in n:
                    variants.append({"gsis_id": r["gsis_id"], "key": norm(n), "team": t,
                                     "last": norm(r["last_name"]), "first1": norm(r["first_name"])[:1]})
    V = pl.DataFrame(variants).unique()
    info = people.select("gsis_id", "espn_id", "position", name="full_name", team="team")

    # 1) same spelling on the same team; 2) same spelling anywhere, if only one player has it;
    # 3) same last name and first initial on the same team, if unique.
    m1 = op.join(V.select("gsis_id", "key", "team"), left_on=["key", "op_team"], right_on=["key", "team"])
    m1 = m1.filter(pl.col("op_id").count().over("op_id") == 1).with_columns(method=pl.lit("name+team"))
    rest = op.filter(~pl.col("op_id").is_in(m1["op_id"].implode()))
    uniq = V.group_by("key").agg(pl.col("gsis_id").n_unique().alias("n"), pl.col("gsis_id").first()).filter(pl.col("n") == 1)
    m2 = rest.join(uniq.select("key", "gsis_id"), on="key").with_columns(method=pl.lit("name"))
    rest = rest.filter(~pl.col("op_id").is_in(m2["op_id"].implode()))
    rest = rest.with_columns(last=pl.col("key").str.split(" ").list.last(), first1=pl.col("key").str.slice(0, 1))
    m3 = rest.join(V.select("gsis_id", "team", "last", "first1").unique(), left_on=["op_team", "last", "first1"],
                   right_on=["team", "last", "first1"])
    m3 = m3.filter(pl.col("gsis_id").n_unique().over("op_id") == 1).unique("op_id").with_columns(method=pl.lit("initial+team"))
    cols = ["op_id", "op_name", "op_team", "gsis_id", "method"]
    out = pl.concat([m.select(cols) for m in (m1, m2, m3)]).unique("op_id")
    return out.join(info, on="gsis_id", how="left")


def main() -> None:
    games = pl.read_parquet(CLEAN / "games.parquet")
    season = int(games.filter(pl.col("result").is_null())["season"].min())
    out = build(season)
    out.write_parquet(OUT)
    total = oddspapi_players().height
    print(f"mapped {out.height} of {total} OddsPapi NFL players to nflverse ids -> {OUT}")
    print(out.group_by("method").len())


if __name__ == "__main__":
    main()
