"""Is a +EV pick also a good pick? (user, 2026-10-18)

EV says a price beats our fair number. The user picks from the board rather than taking every pick, so each
+EV pick also gets checks on whether its edge is the kind that has held up, using only what was known at the
last board update before kickoff. A pick that passes every check that applies is a good pick; otherwise it is
+EV only, with the reasons shown.

- sharp: still +EV when the market's number is read from Pinnacle's price, juice included. Line shopping
  DraftKings and FanDuel against the sharp price is the edge that held up (props at lock, 2026: +6.8% on 295
  bets). Props are priced this way already; game lines are not (board/checks.py), so this is where a
  game-line pick's EV gets corrected.
- size: an edge of 2% or more. Props line-shopping bets at 2%+ beat the closing line 81% of the time against
  70% for all of them (SGO, 2026 opens); below that the edge sits inside the error of taking out the vig.
- lean (props): overs on receiving yards, rushing yards and receptions need 5%+. Pinnacle's closing prices made
  those overs 2-5 points likelier than they turned out in 2025 and 2026, worth about 4-9% of EV at these prices
  (the 2026 picks: receiving-yards overs priced to win 56%, won 48.5%).
- market (game lines): Pinnacle has not moved against the pick since it locked on Tuesday. Props lock at the
  last update before their game, so they have no move to check.
- who: nothing about who plays is in doubt. Props: the player is not Questionable or Doubtful, has not just
  changed teams, and was not promoted into the role. Game lines: no QB on either team is Questionable or
  Doubtful (Out is settled news the model already uses).

Not a check: how far the model is from the market. In the 2022-25 backtest the model's side covered about the
same at every size of gap (51%, 52%, 54%, 44%, 51% from under 1 point to over 4), so a big gap is no warning.

The verdict a pick carried at the last update before its game is frozen in ledger/pick_checks.jsonl (append-only;
a line only when the verdict changes), so the good-vs-+EV-only record cannot be rewritten by later changes to
these checks. Picks from before the file existed get the verdict rebuilt from what was known before kickoff.
"""
import json
from datetime import datetime

from config import LEDGER

CHECKS = LEDGER / "pick_checks.jsonl"
SINCE_WEEK = 6   # the checks were chosen on weeks 1-5 (and 2025 data), so week 6 on is their test
MIN_EV = 0.02
OVER_MIN_EV = 0.05
OVER_LEAN = {"rec_yds": "receiving yards", "rush_yds": "rushing yards", "receptions": "receptions"}
MOVE = 0.015                      # a 1.5-point change in the sharp market's chance for our side counts as a move
UNSURE = ("Questionable", "Doubtful")

GOOD, EV_ONLY = "good", "ev"


def _c(key: str, ok: bool | None, text: str) -> dict:
    return {"key": key, "ok": ok, "text": text}


def verdict(checks: list[dict]) -> str:
    return GOOD if all(c["ok"] is not False for c in checks) else EV_ONLY


def failed(checks: list[dict]) -> list[str]:
    return [c["key"] for c in checks if c["ok"] is False]


def pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def sharp(ev_sharp: float | None) -> dict:
    if ev_sharp is None:
        return _c("sharp", False, "No Pinnacle price to check it against: the edge is one US book against the other.")
    if ev_sharp > 0:
        return _c("sharp", True, f"Still +EV at Pinnacle's real price, juice included ({pct(ev_sharp)}).")
    return _c("sharp", False, f"Not +EV once Pinnacle's juice is counted ({pct(ev_sharp)}): the price only looked good "
                              f"against Pinnacle's line read as a coin flip.")


def size(ev: float) -> dict:
    if ev >= MIN_EV:
        return _c("size", True, f"A {pct(ev)} edge, above the 2% that bets need to beat the closing line reliably.")
    return _c("size", False, f"Thin: {pct(ev)}, under 2%, inside the error of taking the vig out of the price.")


def lean(market: str, side: str, ev: float) -> dict:
    if market in OVER_LEAN and side == "over":
        what = OVER_LEAN[market]
        if ev >= OVER_MIN_EV:
            return _c("lean", True, f"Big enough ({pct(ev)}) to clear the premium books put on {what} overs.")
        return _c("lean", False, f"Overs on {what} were priced 2-5 points too likely in 2025-26; "
                                 f"a {pct(ev)} edge may not survive that (needs 5%+).")
    if side == "under" and market in OVER_LEAN:
        return _c("lean", True, f"An under: the premium books put on {OVER_LEAN[market]} overs works for it.")
    return _c("lean", True, "No over premium in this market.")


def market(p_then: float | None, p_now: float | None, then_txt: str = "", now_txt: str = "") -> dict:
    if p_then is None or p_now is None:
        return _c("market", None, "No Pinnacle line at both times to compare.")
    d = p_now - p_then
    moved = f" ({then_txt} to {now_txt})" if then_txt and now_txt and then_txt != now_txt else ""
    if d >= MOVE:
        return _c("market", True, f"Pinnacle moved our way since the pick locked{moved}: the sharp market agrees.")
    if d <= -MOVE:
        return _c("market", False, f"Pinnacle moved against us since the pick locked{moved}: "
                                   f"the sharp market knows something or disagrees.")
    return _c("market", True, "Pinnacle has not really moved since the pick locked.")


def who(problems: list[str], ok_text: str) -> dict:
    return _c("who", False, " ".join(problems)) if problems else _c("who", True, ok_text)


# ------------------------------------------------------------------ the frozen verdicts and their record

def _read() -> list[dict]:
    if not CHECKS.exists():
        return []
    return [json.loads(x) for x in CHECKS.read_text(encoding="utf-8").splitlines() if x.strip()]


def frozen(kickoff_of: dict[str, datetime]) -> dict[str, dict]:
    """Each pick's last verdict written before its game kicked off: {id: {verdict, failed, at}}."""
    out = {}
    for r in _read():   # file order is time order
        k = kickoff_of.get(r["game_id"])
        if k is not None and datetime.fromisoformat(r["at"]) < k:
            out[r["id"]] = r
    return out


def freeze(entries: list[dict], now: datetime) -> int:
    """Append {kind, id, game_id, verdict, failed} for picks whose verdict changed since its last line."""
    last = {r["id"]: (r["verdict"], r["failed"]) for r in _read()}
    new = [{**e, "at": now.isoformat()} for e in entries if last.get(e["id"]) != (e["verdict"], e["failed"])]
    if new:
        with CHECKS.open("a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
    return len(new)


def tally(rows: list[dict], since_week: int | None = None) -> dict:
    """Flat-stake record of graded picks by verdict: {good: {...}, ev: {...}}."""
    out = {}
    for v in (GOOD, EV_ONLY):
        rs = [r for r in rows if r.get("verdict") == v and r.get("result") in ("W", "L", "P")
              and (since_week is None or r["week"] >= since_week)]
        units = sum(r["units"] or 0 for r in rs)
        staked = sum(r.get("risk") or 1.0 for r in rs)
        out[v] = {"w": sum(r["result"] == "W" for r in rs), "l": sum(r["result"] == "L" for r in rs),
                  "p": sum(r["result"] == "P" for r in rs), "n": len(rs), "units": round(units, 2),
                  "roi": round(units / staked, 4) if staked else None}
    return out
