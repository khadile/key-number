"""How far to trust the model over the market.

Fit on past seasons' out-of-sample projections:

    result = a + b * model_margin + c * spread_line
    total  = a + b * model_total  + c * total_line

If the model knows something the line doesn't, b comes out meaningfully above
zero. If it doesn't, b goes to ~0 and the blended number collapses onto the
line, and the model stops producing edges. That is the intended behavior: it
is the calibration step that keeps stated probabilities honest.
"""
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class Blend:
    m_coef: np.ndarray  # a, b (model), c (line)
    t_coef: np.ndarray
    m_sigma: float
    t_sigma: float

    @classmethod
    def fit(cls, hist: pl.DataFrame) -> "Blend":
        def ols(y, model, line):
            X = np.column_stack([np.ones(len(y)), model, line])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            return coef, float(resid.std(ddof=3))

        h = hist.filter(pl.col("spread_line").is_not_null() & pl.col("total_line").is_not_null())
        mc, ms = ols(h["result"].to_numpy(), h["model_margin"].to_numpy(), h["spread_line"].to_numpy())
        tc, ts = ols(h["total"].to_numpy(), h["model_total"].to_numpy(), h["total_line"].to_numpy())
        return cls(mc, tc, ms, ts)

    def margin(self, model: float, line: float) -> float:
        a, b, c = self.m_coef
        return a + b * model + c * line

    def total(self, model: float, line: float) -> float:
        a, b, c = self.t_coef
        return a + b * model + c * line
