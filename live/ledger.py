"""The live engine's record: flags, paper fills, your accepts, results. All append-only.

ledger/live_flags.jsonl    every flagged play, with its price, book, chances, EV and reason
ledger/live_events.jsonl   what happened to it: filled / missed (paper), expired, passed, line value
ledger/live_accepts.jsonl  plays you accepted (you placed the bet), at the price you got
ledger/live_results.jsonl  W / L / P / V once the game is final
ledger/live_shadow.jsonl   shadow picks: each game's second-best play of each kind, cut only by the
                           one-a-game cap; graded in live_shadow_results.jsonl, never on the record

Two records come out of it. The model's paper record counts every flag at the
price it was flagged (a flag uses one of the game's slots); the 30-second
re-check only says how often that price was still there. Your record counts
what you accepted. Both risk 1 unit per play. The shadow record answers one
question: would a second play a game earn its keep?
"""
import json
import threading
from datetime import datetime, timezone

import numpy as np

from config import LEDGER
from model.staking import kelly_tally

FLAGS = LEDGER / "live_flags.jsonl"
EVENTS = LEDGER / "live_events.jsonl"
ACCEPTS = LEDGER / "live_accepts.jsonl"
RESULTS = LEDGER / "live_results.jsonl"
SHADOW = LEDGER / "live_shadow.jsonl"
SHADOW_RESULTS = LEDGER / "live_shadow_results.jsonl"
STAKE = 1.0
_LOCK = threading.Lock()


def use(folder) -> None:
    """Point the ledger somewhere else (demo replays use data/live/demo_ledger, never the real record)."""
    global FLAGS, EVENTS, ACCEPTS, RESULTS, SHADOW, SHADOW_RESULTS
    folder.mkdir(parents=True, exist_ok=True)
    FLAGS, EVENTS, ACCEPTS, RESULTS = (folder / "live_flags.jsonl", folder / "live_events.jsonl",
                                       folder / "live_accepts.jsonl", folder / "live_results.jsonl")
    SHADOW, SHADOW_RESULTS = folder / "live_shadow.jsonl", folder / "live_shadow_results.jsonl"


def shadow_record() -> dict:
    """The shadow picks' record, at the price each was seen, and the latest ones."""
    res = {r["id"]: r for r in read(SHADOW_RESULTS)}
    rows = []
    for s in read(SHADOW):
        r = res.get(s["id"], {})
        rows.append({**s, "result": r.get("result"), "actual": r.get("actual"),
                     "units": units(r["result"], s["dec"]) if r.get("result") else None})
    graded = [r for r in rows if r.get("units") is not None]
    return {"record": _tally(graded), "flags": sorted(rows, key=lambda r: r["at"], reverse=True)[:40]}


def read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append(path, row: dict) -> None:
    row = {**row, "logged_at": datetime.now(timezone.utc).isoformat()}
    with _LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def grade(flag: dict, line: float | None, actual: float | None, margin: float | None = None,
          total: float | None = None) -> str:
    """W/L/P/V for a prop (actual = final stat; None = did not play) or a game market."""
    m, side = flag["market"], flag["side"]
    if m in ("ml", "spread", "total"):
        if margin is None:
            return "V"
        if m == "ml":
            return "W" if (margin > 0) == (side == "home") else ("P" if margin == 0 else "L")
        if m == "spread":
            d = (margin if side == "home" else -margin) + line
        else:
            d = total - line if side == "over" else line - total
    else:
        if actual is None:
            return "V"
        d = actual - line if side == "over" else line - actual
    return "W" if d > 0 else ("P" if d == 0 else "L")


def units(res: str, dec: float) -> float:
    return STAKE * (dec - 1) if res == "W" else (-STAKE if res == "L" else 0.0)


def _tally(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("result") in ("W", "L", "P")]
    u = sum(r["units"] for r in rows)
    clv = [r["clv_units"] for r in rows if r.get("clv_units") is not None]
    return {"w": sum(r["result"] == "W" for r in rows), "l": sum(r["result"] == "L" for r in rows),
            "p": sum(r["result"] == "P" for r in rows), "n": len(rows), "units": round(u, 2),
            "roi": round(u / len(rows), 4) if rows else None,
            "clv": round(float(np.mean(clv)), 2) if clv else None, "clv_n": len(clv)}


def records() -> dict:
    """Paper (model, auto-accepted) and your (accepted) records, plus every flag with its status."""
    flags = {f["id"]: f for f in read(FLAGS)}
    ev: dict[str, dict] = {}
    for e in read(EVENTS):
        ev.setdefault(e["id"], {}).update({k: v for k, v in e.items() if k not in ("id", "logged_at")})
    res = {r["id"]: r for r in read(RESULTS)}
    acc = {a["id"]: a for a in read(ACCEPTS)}
    rows = []
    for fid, f in flags.items():
        e = ev.get(fid, {})
        r = res.get(fid, {})
        row = {**f, **e, "result": r.get("result"), "actual": r.get("actual")}
        # every flag counts, at the price it was flagged: it used one of the game's slots
        row["units"] = units(r["result"], f["dec"]) if r.get("result") else None
        a = acc.get(fid)
        if a:
            ra = res.get(fid + ":you", {})
            row["accepted"] = {**a, "result": ra.get("result"),
                               "units": units(ra["result"], a["dec"]) if ra.get("result") else None}
        rows.append(row)
    paper = [r for r in rows if r.get("units") is not None]
    # quarter-Kelly sizing, its own record beside the flat 1u paper record from the first flag (model/staking.py)
    kel = {**kelly_tally([{"result": r.get("result"), "dec": r["dec"], "p_win": r.get("p"), "kelly_u": r.get("kelly_u")}
                          for r in paper]), "flat": _tally(paper)}
    you = [{**r["accepted"], "clv_units": r.get("clv_units")} for r in rows if r.get("accepted") and r["accepted"].get("units") is not None]
    decided_fill = [r for r in rows if r.get("fill") in ("filled", "missed")]
    return {"paper": _tally(paper), "you": _tally(you), "paper_kelly": kel,
            "missed": sum(r.get("fill") == "missed" for r in rows), "fill_checked": len(decided_fill),
            "filled": sum(r.get("fill") == "filled" for r in rows),
            "flags": sorted(rows, key=lambda r: r["at"], reverse=True)}
