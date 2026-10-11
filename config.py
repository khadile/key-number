"""Shared paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
LINES = DATA / "lines"
SNAPSHOTS = DATA / "snapshots"
REPORTS = ROOT / "reports"
LEDGER = ROOT / "ledger"

# 2016 is the first season with participation data and gives two seasons of
# history before the first backtest season.
FIRST_SEASON = 2016
BACKTEST_FIRST = 2018
TUNE_LAST = 2021  # hyperparameters are tuned on BACKTEST_FIRST..TUNE_LAST only

# Units
STRAIGHT_RISK = 1.0
PARLAY_RISK = 0.5

for p in (RAW, CLEAN, LINES, SNAPSHOTS, REPORTS, LEDGER):
    p.mkdir(parents=True, exist_ok=True)


def secret(name: str) -> str | None:
    """Read a key from the environment, falling back to the project's .env file."""
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip()
    return None
