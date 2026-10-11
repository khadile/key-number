"""Turn the built board into a standalone web page for GitHub Pages: site/index.html.

board/nfl_board.html is written for a claude.ai Artifact, which wraps it in a document
skeleton when published. A plain web host needs that skeleton itself (doctype, UTF-8,
viewport), or the page renders in quirks mode with garbled characters.

    python -m board.site
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SITE = ROOT.parent / "site"


def main() -> None:
    body = (ROOT / "nfl_board.html").read_text(encoding="utf-8")
    SITE.mkdir(exist_ok=True)
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>body{margin:0}</style></head><body>')
    (SITE / "index.html").write_text(head + body + "</body></html>", encoding="utf-8")
    (SITE / ".nojekyll").write_text("")
    print(f"wrote {SITE / 'index.html'} ({len(body) // 1024} KB)")


if __name__ == "__main__":
    main()
