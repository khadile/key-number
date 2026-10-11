"""Per-team pieces of the game breakdown: ratings with league percentiles, and injuries."""
import numpy as np
import polars as pl

from config import RAW

# (label, snapshot key, higher is better?) Defense values are "allowed", so lower is better.
OFFENSE = [("Passing", "off_pass_epa", True), ("Rushing", "off_rush_epa", True),
           ("Pass success", "off_pass_sr", True), ("Rush success", "off_rush_sr", True),
           ("Pass rate", "off_neutral_proe", True), ("Pace", "off_neutral_sec_per_play", False)]
DEFENSE = [("Passing", "def_pass_epa", False), ("Rushing", "def_rush_epa", False),
           ("Pass success", "def_pass_sr", False), ("Rush success", "def_rush_sr", False)]


def _pct(values: np.ndarray, i: int, higher_better: bool) -> int:
    """0-100: share of the other teams this one beats."""
    v = values if higher_better else -values
    return int(round(100 * (v < v[i]).sum() / (len(v) - 1)))


def team_profiles(snap: dict, teams: list[str]) -> dict[str, dict]:
    out = {}
    for i, t in enumerate(teams):
        out[t] = {
            "off": [{"label": lbl, "pct": _pct(snap[k], i, hb), "value": float(snap[k][i])} for lbl, k, hb in OFFENSE],
            "def": [{"label": lbl, "pct": _pct(snap[k], i, hb), "value": float(snap[k][i])} for lbl, k, hb in DEFENSE],
        }
    return out


STATUS_ORDER = {"Out": 0, "Doubtful": 1, "Questionable": 2}
POS_ORDER = {p: i for i, p in enumerate(["QB", "RB", "WR", "TE", "T", "G", "C", "DE", "DT", "LB", "CB", "S", "K", "P"])}


def injuries(season: int, week: int) -> dict[str, list[dict]]:
    path = RAW / "injuries" / f"{season}.parquet"
    if not path.exists():
        return {}
    i = (pl.read_parquet(path)
         .filter((pl.col("week") == week) & pl.col("report_status").is_in(list(STATUS_ORDER)))
         .with_columns(so=pl.col("report_status").replace_strict(STATUS_ORDER, default=9),
                       po=pl.col("position").replace_strict(POS_ORDER, default=20))
         .sort("team", "so", "po"))
    out: dict[str, list[dict]] = {}
    for r in i.iter_rows(named=True):
        out.setdefault(r["team"], []).append({
            "name": r["full_name"], "pos": r["position"], "status": r["report_status"],
            "injury": r["report_primary_injury"]})
    return out
