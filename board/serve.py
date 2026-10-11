"""Serve board/ locally for a browser check before publishing: python board/serve.py [port]

Python's own http.server sends HTML without a charset, so the built page (which gets its
<meta charset> only when published) shows garbled characters. This one sends UTF-8.
"""
import functools
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8"}

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    print(f"serving {ROOT} at http://localhost:{PORT}/nfl_board.html")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(Handler, directory=str(ROOT))).serve_forever()
