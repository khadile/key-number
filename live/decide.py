"""Which in-game prices to flag, and the one-line reason for each.

The engine watches every play but only bets at decision points (redesign of
2026-10-09, chosen by live/replay_windows.py over 64 games of real in-game
prices; the user wants fewer, more trustworthy plays):

- player props: once, at halftime (PROP_WINDOWS), the single strongest play
  that passes (PROP_PER_GAME). End-of-first-quarter decisions lost every way
  they were cut: there is not enough game yet;
- game markets (moneyline, spread, total): checked after every score, at most
  GAME_PER_GAME a game.

The rubric (gate()), every part required:
1. the model within MAX_GAP (10 points) of the no-vig market: the bigger
   disagreements were where the model was missing news (said 62%, won 37% on
   127 replay flags), the moderate ones held up;
2. expected value of at least MIN_EV (5%) at the book's price, after the vig;
3. the same price passes on two checks PERSIST_S apart (a stale or flickering
   price fails); the quote itself no older than FRESH_FLAG_S;
4. props: not in a blowout (lead over BLOWOUT); game markets: no long shots
   (beyond +/-PRICE_CAP);
5. one play per player, never the same player and market twice; plays that
   ride on the same passing game in the same direction count as one exposure;
6. props: not when one big play carries the player's numbers (a 20+ yard play
   that is half his yards in that stat, or one in the last two minutes);
7. props: not on a player the play-by-play says was hurt, until he touches the
   ball again. (6 and 7 need the plays: live/rubric_checks.py and the engine.)

Every flag counts in the paper record at the price it was flagged, and carries
its reason, built from the numbers that made the edge.
"""
from dataclasses import dataclass

import numpy as np

MIN_EV = 0.05
MAX_GAP = 0.10
PROP_WINDOWS = ("half",)          # where props are decided: "q1", "half", "q3" (live/replay_windows.py)
WINDOW_AT = {"q1": 2700, "half": 1800, "q3": 900}    # game seconds left when each break starts
WINDOW_LABEL = {"q1": "End of the first quarter", "half": "Halftime", "q3": "End of the third quarter"}
PROP_PER_GAME = 1
GAME_PER_GAME = 1
MAX_PER_GAME = PROP_PER_GAME + GAME_PER_GAME
PERSIST_S = 45                    # second price check this long after the first (twice that at halftime)
FRESH_FLAG_S = 60
BLOWOUT = 17
PRICE_CAP = 300
MAX_AGE_S = 90
CARD_TTL_S = 180
EARLIEST_SECS = 1800              # nothing before halftime for props (kept for the board's text)
STAKE_U = 1.0
MARKET_WORD = {"pass_yds": "pass yds", "rush_yds": "rush yds", "rec_yds": "rec yds", "receptions": "receptions",
               "ml": "moneyline", "spread": "spread", "total": "total"}


def no_vig(over: float, under: float) -> float:
    a, b = 1 / over, 1 / under
    return a / (a + b)


def side_p(samples: np.ndarray, line: float) -> tuple[float, float, float]:
    over = float((samples > line + 1e-9).mean())
    under = float((samples < line - 1e-9).mean())
    return over, 1 - over - under, under


@dataclass
class Candidate:
    key: str            # player:market or game:market
    game_id: str
    kind: str           # prop / game
    market: str
    player: str | None
    name: str | None
    team: str | None
    side: str           # over / under / home / away
    line: float | None
    book: str
    dec: float
    p: float
    push: float
    fair: float
    ev: float
    group: str          # correlation group
    reason: str = ""

    def label(self) -> str:
        if self.kind == "prop":
            return f"{self.name} {self.side.title()} {self.line:g} {MARKET_WORD[self.market]}"
        if self.market == "total":
            return f"{self.side.title()} {self.line:g}"
        if self.market == "spread":
            return f"{self.team} {self.line:+g}"
        return f"{self.team} ML"


def american(dec: float) -> int:
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def anchor_shift(s: np.ndarray, books: dict) -> float:
    """Where DraftKings and FanDuel put the median, relative to the model (units): the
    shift that makes the model's own shape give each book's no-vig chance at its line."""
    shifts = []
    for L in books.values():
        p = no_vig(L["over"], L["under"])
        shifts.append(L["line"] - float(np.quantile(s, min(max(1 - p, 0.001), 0.999))))
    return float(np.mean(shifts))


def prop_candidates(game_id: str, proj: dict, lines: dict, info: dict, k: float | None = None) -> list[Candidate]:
    """lines: {(gsis_id, market): {book: {line, over, under}}}; info: gsis_id -> {name, team, position}.

    k=None prices with the model's own chances. A number anchors to the market instead, as
    pregame: the books' consensus location moved k of the way toward the model, so a flag
    means one book's number is off the other's, not that the model disagrees with both."""
    out = []
    for (pid, market), books in lines.items():
        s = proj["players"].get(pid, {}).get(market)
        if s is None:
            continue
        if k is not None:
            s = s + (1 - k) * anchor_shift(s, books)
        same = {}
        for b, L in books.items():
            same.setdefault(L["line"], []).append(no_vig(L["over"], L["under"]))
        for b, L in books.items():
            if b not in ("draftkings", "fanduel"):
                continue
            over, push, under = side_p(s, L["line"])
            fair_over = float(np.mean(same[L["line"]]))
            for side, p, dec, fair in (("over", over, L["over"], fair_over), ("under", under, L["under"], 1 - fair_over)):
                ev = p * (dec - 1) - (1 - p - push)
                pos = info[pid]["position"]
                group = (f"pass:{info[pid]['team']}:{side}" if market in ("pass_yds", "rec_yds", "receptions") and pos != "RB"
                         else f"{pid}")
                out.append(Candidate(f"{pid}:{market}", game_id, "prop", market, pid, info[pid]["name"], info[pid]["team"],
                                     side, L["line"], b, dec, p, push, fair, ev, group))
    return out


def game_candidates(game_id: str, game: dict, lines: dict, home: str, away: str) -> list[Candidate]:
    """lines: {book: {spread (home favored by), spread_price_a (home), spread_price_b, total, total_price_a (over),
    total_price_b, ml_home, ml_away}} in decimal odds."""
    out = []
    a, b = game["teams"]
    margin = game["margin"] if a == home else -game["margin"]   # home minus away
    total = game["total"]
    for bk, L in lines.items():
        if bk not in ("draftkings", "fanduel"):
            continue
        if "ml_home" in L:
            ph = float((margin > 0).mean())
            fair = no_vig(L["ml_home"], L["ml_away"])
            for side, p, dec, f, team in (("home", ph, L["ml_home"], fair, home), ("away", 1 - ph, L["ml_away"], 1 - fair, away)):
                out.append(Candidate(f"game:ml", game_id, "game", "ml", None, None, team, side, None, bk, dec, p, 0.0, f,
                                     p * (dec - 1) - (1 - p), f"game:{game_id}"))
        if "spread" in L:
            s = L["spread"]
            over, push, under = side_p(margin, s)      # home covers when margin > s
            fair = no_vig(L["spread_price_a"], L["spread_price_b"])
            for side, p, dec, f, team, line in (("home", over, L["spread_price_a"], fair, home, -s),
                                               ("away", under, L["spread_price_b"], 1 - fair, away, s)):
                out.append(Candidate(f"game:spread", game_id, "game", "spread", None, None, team, side, line, bk, dec, p, push, f,
                                     p * (dec - 1) - (1 - p - push), f"game:{game_id}"))
        if "total" in L:
            t = L["total"]
            over, push, under = side_p(total, t)
            fair = no_vig(L["total_price_a"], L["total_price_b"])
            for side, p, dec, f in (("over", over, L["total_price_a"], fair), ("under", under, L["total_price_b"], 1 - fair)):
                out.append(Candidate(f"game:total", game_id, "game", "total", None, None, None, side, t, bk, dec, p, push, f,
                                     p * (dec - 1) - (1 - p - push), f"game:{game_id}:total"))
    return out


def gate(c: Candidate, lead: float, quote_age: float | None = None) -> str | None:
    """None if the candidate passes the rubric's price tests, else the first one it fails."""
    if abs(c.p - c.fair) > MAX_GAP:
        return f"model more than {MAX_GAP:.0%} from the market"
    if c.ev < MIN_EV:
        return f"under the {MIN_EV:.0%} bar"
    if c.kind == "prop" and abs(lead) > BLOWOUT:
        return "blowout"
    if c.kind == "game" and not (1 + 100 / PRICE_CAP <= c.dec <= 1 + PRICE_CAP / 100):
        return "long shot"
    if quote_age is not None and quote_age > FRESH_FLAG_S:
        return "quote too old"
    return None


def select(cands: list[Candidate], flagged: list[dict], caps: dict | None = None) -> list[Candidate]:
    """New flags at a decision point: the strongest candidates that passed the rubric, within the
    caps (PROP_PER_GAME props, GAME_PER_GAME game markets, or `caps` as {kind: how many more})
    and the exposure rules."""
    taken_keys = {f["key"] for f in flagged}
    taken_players = {f["player"] for f in flagged if f.get("player")}
    taken_groups = {f["group"] for f in flagged}
    kind = [f.get("kind") or ("prop" if f.get("player") else "game") for f in flagged]   # old replays carry no kind
    left = dict(caps) if caps is not None else {"prop": PROP_PER_GAME - kind.count("prop"),
                                                 "game": GAME_PER_GAME - kind.count("game")}
    out = []
    for c in sorted(cands, key=lambda c: -c.ev):
        if left.get(c.kind, 0) <= 0:
            continue
        if c.ev < MIN_EV or abs(c.p - c.fair) > MAX_GAP:
            continue
        if c.key in taken_keys or (c.player and c.player in taken_players) or c.group in taken_groups:
            continue
        out.append(c)
        taken_keys.add(c.key)
        taken_groups.add(c.group)
        if c.player:
            taken_players.add(c.player)
        left[c.kind] -= 1
    return out


def _mins(secs: float) -> str:
    m = int(secs // 60)
    return f"{m} min" if m >= 1 else f"{int(secs)} sec"


def reason(c: Candidate, ctx: dict) -> str:
    """One line: why the model likes it. ctx: seconds_left, score, banked, rem_mean, share_now, prior_share,
    team_lead, volume, books moved."""
    bits = []
    if c.kind == "prop":
        have = ctx["banked"].get(c.market, 0.0)
        need = (c.line or 0) - have
        rem = ctx["rem_mean"].get(c.market, 0.0)
        def unit(n: float) -> str:
            return ("catch" if round(n) == 1 else "catches") if c.market == "receptions" else "yds"
        if c.side == "over":
            n = float(np.ceil(need)) if need > 0 else 0.0   # 4 catches against 4.5 needs 1 more, not "0"
            bits.append(f"needs {n:.0f} more {unit(n)} in {_mins(ctx['seconds_left'])}; model's typical outcome is {rem:.0f} more")
        else:
            bits.append(f"has {have:.0f} {unit(have)} with the line at {c.line:g}; model's typical outcome is {rem:.0f} more in {_mins(ctx['seconds_left'])}")
        prior, now = ctx.get("prior_share"), ctx.get("share_now")
        if prior and now is not None and c.market != "pass_yds":
            word = "target" if c.market != "rush_yds" else "carry"
            bits.append(f"{word} share {now:.0%} going forward (pregame {prior:.0%})")
        lead = ctx.get("team_lead", 0)
        if abs(lead) >= 7 and c.market in ("pass_yds", "rec_yds", "receptions"):
            bits.append(f"{c.team} {'down' if lead < 0 else 'up'} {abs(lead):.0f}: {'more' if lead < 0 else 'fewer'} passes ahead")
    else:
        bits.append(f"model {c.p:.0%} vs market {c.fair:.0%}")
        if ctx.get("fluke"):
            bits.append(ctx["fluke"])
        bits.append(f"{_mins(ctx['seconds_left'])} left, rest played from team strength")
    return "; ".join(bits[:3])
