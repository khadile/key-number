"""Put the NFL and NBA payloads into the board page: board/nfl_board.html.

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
    nba = NBA_DATA.read_text(encoding="utf-8") if NBA_DATA.exists() else "null"
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__BOARD_DATA__", _embed(_nfl_payload()))
            .replace("__NBA_DATA__", _embed(nba)))
    OUT.write_text(html, encoding="utf-8")
    return html


if __name__ == "__main__":
    print(f"wrote {OUT} ({len(render()) / 1024:.0f} KB)")
