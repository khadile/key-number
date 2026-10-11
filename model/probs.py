"""Turn a projected mean into probabilities that respect key numbers.

NFL margins pile up on 3, 7, 6, 10, 14; totals cluster more weakly. A plain
normal curve gets cover probability wrong exactly where it matters (a line of
-3 vs -3.5). So:

    P(margin = k | mu) ∝ normal_mass(k; mu, sigma) * r(k)

where r(k) is how much more often k happens than a smooth curve predicts,
measured from past games against their closing lines.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

MARGIN_RANGE = np.arange(-70, 71)
TOTAL_RANGE = np.arange(0, 131)


def _normal_mass(values: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return norm.cdf(values + 0.5, mu, sigma) - norm.cdf(values - 0.5, mu, sigma)


@dataclass
class KeyDist:
    values: np.ndarray
    ratio: np.ndarray
    sigma: float
    symmetric: bool

    @classmethod
    def fit(cls, outcomes: np.ndarray, centers: np.ndarray, values: np.ndarray,
            sigma: float, symmetric: bool, smooth: float = 20.0) -> "KeyDist":
        observed = np.zeros(len(values))
        expected = np.zeros(len(values))
        lo = values[0]
        for y, c in zip(outcomes, centers):
            idx = int(round(y)) - lo
            if 0 <= idx < len(values):
                observed[idx] += 1
            expected += _normal_mass(values, c, sigma)
        if symmetric:  # a 3-point home win and a 3-point road win are the same key number
            observed = observed + observed[::-1]
            expected = expected + expected[::-1]
        ratio = (observed + smooth) / (expected + smooth)  # shrink thin cells toward 1
        return cls(values, ratio, sigma, symmetric)

    def _shape(self, loc: float) -> np.ndarray:
        p = _normal_mass(self.values, loc, self.sigma) * self.ratio
        return p / p.sum()

    def _cdf(self, p: np.ndarray, x: float) -> float:
        """CDF with each whole number's probability spread evenly over [k - 0.5, k + 0.5]."""
        k = int(np.floor(x + 0.5))
        i = k - int(self.values[0])
        if i < 0:
            return 0.0
        if i >= len(p):
            return 1.0
        return float(p[:i].sum() + p[i] * (x - (k - 0.5)))

    def pmf(self, mu: float) -> np.ndarray:
        """Outcome distribution whose (smoothed) median is mu.

        Multiplying a normal curve by the key-number weights drags its center
        toward the nearest big key number (a curve centered at 7.5 leans onto
        the 7). Left uncorrected, a pick at the market line looks like a 55%
        underdog cover when such picks win about 50%. So the curve's location
        is solved for: with each whole number's probability spread across its
        one-point slot, mu is the median. A line equal to mu is then a coin
        flip apart from pushes, and moving off it pays off fastest across the
        slots that hold the most probability, the key numbers 3 and 7.
        """
        key = round(float(mu), 3)
        cache = self.__dict__.setdefault("_cache", {})
        if key not in cache:
            lo, hi = key - 8.0, key + 8.0
            for _ in range(45):
                mid = (lo + hi) / 2
                if self._cdf(self._shape(mid), key) > 0.5:
                    lo = mid  # too much probability below mu: move the curve up
                else:
                    hi = mid
            cache[key] = self._shape((lo + hi) / 2)
        return cache[key]

    def side_probs(self, mu: float, line: float) -> tuple[float, float, float]:
        """P(outcome > line), P(outcome == line), P(outcome < line)."""
        p = self.pmf(mu)
        over = p[self.values > line].sum()
        push = p[self.values == line].sum()
        return float(over), float(push), float(1 - over - push)


@dataclass
class WinCurve:
    """P(home wins) as a logistic function of the expected margin.

    Fit directly on past outcomes against closing spreads. The normal-curve
    pmf above is right near the line but overstates how often big underdogs
    win, which is exactly where moneyline value would appear by mistake.
    """
    a: float
    k: float

    @classmethod
    def fit(cls, expected_margin: np.ndarray, result: np.ndarray) -> "WinCurve":
        keep = result != 0  # ties are pushes on a moneyline
        x, y = expected_margin[keep], (result[keep] > 0).astype(float)
        beta = np.zeros(2)
        X = np.column_stack([np.ones(len(x)), x])
        for _ in range(25):  # Newton's method for a 2-parameter logistic regression
            p = 1 / (1 + np.exp(-X @ beta))
            grad = X.T @ (y - p)
            hess = (X * (p * (1 - p))[:, None]).T @ X
            beta += np.linalg.solve(hess, grad)
        return cls(float(beta[0]), float(beta[1]))

    def home(self, mu: float) -> float:
        return float(1 / (1 + np.exp(-(self.a + self.k * mu))))


def american_to_decimal(odds: float) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def no_vig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Market probabilities with the bookmaker's margin removed."""
    ia, ib = 1 / american_to_decimal(odds_a), 1 / american_to_decimal(odds_b)
    return ia / (ia + ib), ib / (ia + ib)


def ev(p_win: float, p_push: float, odds: float) -> float:
    """Expected profit per 1 unit risked."""
    return p_win * (american_to_decimal(odds) - 1) - (1 - p_win - p_push)
