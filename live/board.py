"""What the board's Live tab shows.

The live engine runs on the laptop during games (python -m live.engine); the
published board cannot reach it. After games, the board build reads the
engine's ledger and shows the paper record (every flag auto-accepted), your
record (what you accepted), every flag with its reason and result, and the
replay of finished games against the real in-game prices that tested the rules.
"""
import json

from config import DATA, REPORTS
from live import decide, ledger
from live.model import LiveParams
from pipeline import propline

VARIANTS = [
    ("uncal_kickoff", "Model's own chances, flags from kickoff"),
    ("cal_kickoff", "Start each game at the closing lines, flags from kickoff"),
    ("book_vs_book", "Bet whichever of DraftKings / FanDuel is off the other's number"),
    ("", "Start at the closing lines, no flags in the first quarter (the engine's rules)"),
]


STATUS = DATA / "live" / "status.json"   # the engine's heartbeat, rewritten every 20 s while it runs


def _read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def board_data(season: int) -> dict:
    rec = ledger.records()
    flags = [{k: f.get(k) for k in ("id", "game_id", "label", "market", "side", "line", "book", "american", "p", "fair",
                                     "ev", "reason", "at", "fill", "fill_dec", "fill_line", "clv_units", "result",
                                     "actual", "units", "accepted", "expired", "passed", "score", "seconds_left")}
             for f in rec["flags"][:150]]
    variants = []
    for tag, name in VARIANTS:
        r = _read(REPORTS / (f"live_replay_{tag}.json" if tag else "live_replay.json"))
        if r:
            variants.append({"tag": tag or "default", "name": name, **r["summary"]})
    default = _read(REPORTS / "live_replay.json") or {}
    games = _read(REPORTS / "live_replay_games.json") or {}
    sample = []
    fp = REPORTS / "live_replay_flags.parquet"
    if fp.exists():
        import polars as pl
        F = pl.read_parquet(fp).sort("at", descending=True)
        sample = F.select([c for c in ("game_id", "label", "book", "american", "p", "fair", "ev", "reason", "filled",
                                       "fill_dec", "clv_units", "result", "secs_left") if c in F.columns]).head(24).to_dicts()
    lp = LiveParams.load()
    W = _read(REPORTS / "live_windows.json") or {}
    wsample = []
    wf = REPORTS / "live_windows_flags.parquet"
    if wf.exists():
        import polars as pl
        F = pl.read_parquet(wf)
        if "variant" in F.columns:
            F = F.filter(pl.col("variant") == "engine")
        F = F.sort("at", descending=True).head(24)
        wsample = [{**{k: r.get(k) for k in ("game_id", "kind", "event", "label", "book", "p", "fair", "ev_adj", "reason", "result", "secs_left")},
                    "american": decide.american(r["dec"])} for r in F.iter_rows(named=True)]
    return {
        "record": {"paper": rec["paper"], "you": rec["you"], "missed": rec["missed"], "fill_checked": rec["fill_checked"]},
        "shadow": ledger.shadow_record(),
        "flags": flags,
        "replay": {"games": default.get("games"), "variants": variants, "thesis": default.get("thesis"),
                   "k_live": (default.get("summary") or {}).get("k_live"), "game_markets": games, "sample": sample,
                   "rules": default.get("rules")},
        "rules": {"min_ev": decide.MIN_EV, "max_gap": decide.MAX_GAP, "per_game": decide.MAX_PER_GAME,
                  "windows": [decide.WINDOW_LABEL[w] for w in decide.PROP_WINDOWS], "prop_per_game": decide.PROP_PER_GAME,
                  "game_per_game": decide.GAME_PER_GAME, "blowout": decide.BLOWOUT, "price_cap": decide.PRICE_CAP,
                  "persist_s": decide.PERSIST_S, "fresh_s": decide.FRESH_FLAG_S, "ttl": decide.CARD_TTL_S, "stake": decide.STAKE_U},
        "windows": {k: W.get(k) for k in ("games", "weeks", "engine", "props", "game", "reference", "gap_bands", "s_props")}
                   | {"sample": wsample},
        "params": {"n0_tgt": lp.n0_tgt, "k_game": round(lp.k_game, 3), "a_lead": round(lp.a_lead, 4),
                   "b_lead_late": round(lp.b_lead_late, 4)},
        "budget": {"used_today": propline.used_today(), "budget": propline.DAILY_BUDGET, "calls_per_game": 220},
        "engine": _read(STATUS),
    }
