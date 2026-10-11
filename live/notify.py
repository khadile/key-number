"""Phone alerts through ntfy (https://ntfy.sh): free, no account.

Off until you set NTFY_TOPIC in nfl_model/.env to a long random topic name
(anyone who knows the name can read it) and subscribe to it in the ntfy app.
Each alert opens the live page; place bets by hand, after confirming the price.
"""
import ssl
import urllib.request

import certifi

from config import secret

_TLS = ssl.create_default_context(cafile=certifi.where())


def enabled() -> bool:
    return bool(secret("NTFY_TOPIC"))


def send(title: str, body: str, click: str | None = None) -> bool:
    topic = secret("NTFY_TOPIC")
    if not topic:
        return False
    headers = {"Title": title.encode("ascii", "replace").decode(), "Tags": "football", "Priority": "high"}
    if click:
        headers["Click"] = click
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=8, context=_TLS):
            return True
    except Exception:  # noqa: BLE001 - an alert failing must never stop the engine
        return False
