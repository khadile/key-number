"""Put the NFL, NBA and college payloads into the board page, with the model rating: board/nfl_board.html.

Each sport's build writes its own payload (board/nfl_data.json from board/build.py, board/nba_data.json from
nba/build.py) and then renders, so the daily NBA build leaves the NFL side as its last build left it, and the
other way round.

    python -m board.render
"""
import json
import re

from config import ROOT

TEMPLATE = ROOT / "board" / "template.html"
OUT = ROOT / "board" / "nfl_board.html"
NFL_DATA = ROOT / "board" / "nfl_data.json"
NBA_DATA = ROOT / "board" / "nba_data.json"
NCAAB_DATA = ROOT / "board" / "ncaab_data.json"


def _embed(text: str) -> str:
    """JSON made safe inside a <script> element."""
    return text.replace("</", "<\\/").replace("<!--", "<\\u0021--")


def _nfl_payload() -> str:
    if NFL_DATA.exists():
        return NFL_DATA.read_text(encoding="utf-8")
    # before the first NFL build that writes its payload: take it from the page the last one rendered
    m = re.search(r'<script id="kn-data" type="application/json">(.*?)</script>', OUT.read_text(encoding="utf-8"), re.S)
    data = json.loads(m.group(1))
    NFL_DATA.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return NFL_DATA.read_text(encoding="utf-8")


def render() -> str:
    from board.rating import boards
    nfl = _nfl_payload()
    nba = NBA_DATA.read_text(encoding="utf-8") if NBA_DATA.exists() else "null"
    ncaab = NCAAB_DATA.read_text(encoding="utf-8") if NCAAB_DATA.exists() else "null"
    # the model rating (board/rating.py) reads every board's record, so it is scored here, where all of them meet
    rating = json.dumps(boards(json.loads(nfl), json.loads(nba), json.loads(ncaab)), separators=(",", ":"))
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__BOARD_DATA__", _embed(nfl))
            .replace("__NBA_DATA__", _embed(nba))
            .replace("__NCAAB_DATA__", _embed(ncaab))
            .replace("__RATING__", _embed(rating)))
    OUT.write_text(html, encoding="utf-8")
    return html


if __name__ == "__main__":
    print(f"wrote {OUT} ({len(render()) / 1024:.0f} KB)")
