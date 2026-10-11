"""Does the market move toward our model after we publish? (closing line value)

Uses OddsPapi lines archived by pipeline/odds_history.py for the current
season. For each game, compare the model's number to the line at publish time
(Tuesday 10am ET), then see which way the market moved by kickoff. If the
market keeps moving toward the model, betting at publish time beats the close,
which is the only evidence of edge that stabilizes quickly.

    python -m backtest.clv
"""
import numpy as np
import polars as pl

from backtest.run import load_params, walk_forward
from config import CLEAN, REPORTS
from model.ratings import load_inputs
from pipeline.oddspapi import CACHE


def _rec(r: np.ndarray) -> str:
    return f"{(r > 0).sum()}-{(r < 0).sum()}-{(r == 0).sum()}"


def section(d: pl.DataFrame, kind: str, model: str, start: str, close: str, outcome: str, label: str) -> list[str]:
    x = d.filter(pl.col(start).is_not_null() & pl.col(close).is_not_null())
    if x.height == 0:
        return []
    gap = (x[model] - x[start]).to_numpy()      # + : model above the line we would bet into
    move = (x[close] - x[start]).to_numpy()     # + : line rose by kickoff
    side = np.sign(gap)
    moved = move != 0
    toward = int((np.sign(move[moved]) == side[moved]).sum())
    clv = move * side                            # points gained by taking the model's side at `start`
    big = np.abs(gap) >= 1.5
    res = x[outcome].to_numpy()
    at_start = np.sign((res - x[start].to_numpy()) * side)
    at_close = np.sign((res - x[close].to_numpy()) * side)
    corr = np.corrcoef(gap, move)[0, 1] if moved.sum() > 2 else float("nan")
    return [f"**{kind}, bet at {label}** ({x.height} games)", "",
            f"- Lines that moved by kickoff: {int(moved.sum())}; toward our model **{toward}**, away {int(moved.sum()) - toward}",
            f"- Average CLV on the model's side: **{clv.mean():+.2f} pts**; when the model was 1.5+ pts off: "
            f"{(clv[big].mean() if big.any() else float('nan')):+.2f} over {int(big.sum())} games",
            f"- Correlation of (model - line) with (later move): {corr:+.2f}",
            f"- Model side graded at the {label} line: {_rec(at_start)}; at the closing line: {_rec(at_close)}", ""]


def main() -> None:
    oc = pl.read_parquet(CACHE / "open_close.parquet")
    games_all = pl.read_parquet(CLEAN / "games.parquet")
    games = games_all.select("game_id", "week", "result", "total")
    season = int(games_all.join(oc.select("game_id").unique(), on="game_id")["season"].max())
    inp, g = load_inputs()
    p, alpha = load_params()
    variants = {
        "Tuesday information only (last week's QB, no game-day weather)":
            walk_forward(inp, g, p, alpha, [season], as_of_publish=True),
        "Actual starting QB and game-day weather (slightly optimistic)":
            walk_forward(inp, g, p, alpha, [season]),
    }
    lines = ["# Closing line value test", "",
             f"Lines from OddsPapi, {season} season (history is only kept for the current season).",
             "Publish = Tuesday 10am ET of game week. Model = walk-forward projection, ratings as of each week.", ""]
    for label, proj in variants.items():
        proj = proj.select("game_id", "model_margin", "model_total")
        lines += [f"# Model: {label}", ""]
        for book in ("pinnacle", "draftkings", "fanduel"):
            d = oc.filter(pl.col("book") == book).join(proj, on="game_id").join(games, on="game_id").filter(
                pl.col("result").is_not_null())
            if d.height == 0:
                continue
            lines += [f"## {book}", ""]
            lines += section(d, "Spread", "model_margin", "publish_spread", "close_spread", "result", "publish")
            lines += section(d, "Total", "model_total", "publish_total", "close_total", "total", "publish")
            if book == "pinnacle":
                lines += section(d, "Spread", "model_margin", "open_spread", "close_spread", "result", "open")
                lines += section(d, "Total", "model_total", "open_total", "close_total", "total", "open")
    lines += ["A handful of weeks is a small sample: treat this as a first read, not a verdict.",
              "Re-run weekly as the archive grows (pipeline/odds_history.py, then this)."]
    text = "\n".join(lines)
    (REPORTS / "clv.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
