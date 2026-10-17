"""Performance metrics, with the annualisation written down.

``252`` appears nowhere as a literal inside a formula. It is the value of
``BARS_PER_YEAR["daily"]``, and every function that annualises takes
``bars_per_year`` as an argument. A Sharpe ratio whose sampling frequency is
implicit is a number two people will read differently.

Degenerate inputs return ``nan`` rather than a number. A flat equity curve
has no Sharpe ratio; reporting ``0.0`` would be a claim, and reporting
``4.0`` because the denominator underflowed would be worse. The tests in
``tests/test_metrics.py`` pin this down for the all-cash case.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

#: Bars in a trading year, by bar size. Pass the one that matches your data.
BARS_PER_YEAR: dict[str, float] = {
    "daily": 252.0,
    "hourly": 252.0 * 6.5,
    "minute": 252.0 * 390.0,
    "weekly": 52.0,
}

DAILY = BARS_PER_YEAR["daily"]

#: Dispersion at or below this is float noise, not a distribution.
#:
#: ``== 0.0`` was the first version and it was wrong. A series of 500
#: identical 1 bp returns has a sample standard deviation of about 1e-20,
#: because the mean does not come back exactly equal to the input. Not zero,
#: so the guard let it through, and the ratio came out near 1e16 annualised.
#: A test asserting "a constant return series has no Sharpe ratio" is what
#: surfaced it; the relative floor below is the fix.
ZERO_DISPERSION = 1e-12


def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    return r[~np.isnan(r)]


def _degenerate(sd: float, mean: float) -> bool:
    return sd <= ZERO_DISPERSION * max(1.0, abs(mean))


def sharpe(
    returns,
    bars_per_year: float = DAILY,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio of per-bar simple returns.

    ``risk_free`` is a per-bar rate, subtracted before the ratio is taken.
    Returns ``nan`` when there are fewer than two observations or when the
    excess returns have zero dispersion.

    The sample standard deviation (``ddof=1``) is used. With ``ddof=0`` a
    two-point series gets a smaller denominator and a flattering ratio,
    which is the wrong direction for a number people make decisions on.
    """
    r = _clean(returns) - risk_free
    if r.size < 2:
        return float("nan")
    mean = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if _degenerate(sd, mean):
        return float("nan")
    return float(mean / sd * np.sqrt(bars_per_year))


def sortino(returns, bars_per_year: float = DAILY, risk_free: float = 0.0) -> float:
    """Like :func:`sharpe`, but the denominator counts only downside."""
    r = _clean(returns) - risk_free
    if r.size < 2:
        return float("nan")
    downside = r[r < 0.0]
    if downside.size == 0:
        return float("nan")
    mean = float(np.mean(r))
    dd = float(np.sqrt(np.mean(downside**2)))
    if _degenerate(dd, mean):
        return float("nan")
    return float(mean / dd * np.sqrt(bars_per_year))


def equity_curve(returns, start: float = 1.0) -> np.ndarray:
    """Compound per-bar simple returns into an equity curve."""
    r = np.asarray(returns, dtype=float)
    r = np.where(np.isnan(r), 0.0, r)
    return start * np.cumprod(1.0 + r)


def max_drawdown(equity) -> float:
    """Largest peak-to-trough fraction, returned as a non-negative number.

    ``0.0`` means no drawdown, ``0.25`` means a 25 percent fall from a peak.
    The sign convention is stated because both are in use and mixing them up
    flips the sign of Calmar.
    """
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(e)
    dd = (peak - e) / peak
    return float(np.max(dd))


def drawdown_duration(equity) -> int:
    """Longest run of bars spent below a previous peak, in bars."""
    e = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(e)
    under = e < peak
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def cagr(equity, bars_per_year: float = DAILY) -> float:
    """Compound annual growth rate implied by the curve's endpoints."""
    e = np.asarray(equity, dtype=float)
    if e.size < 2 or e[0] <= 0 or e[-1] <= 0:
        return float("nan")
    years = (e.size - 1) / bars_per_year
    if years <= 0:
        return float("nan")
    return float((e[-1] / e[0]) ** (1.0 / years) - 1.0)


def calmar(equity, bars_per_year: float = DAILY) -> float:
    """CAGR divided by max drawdown. ``nan`` when there is no drawdown."""
    mdd = max_drawdown(equity)
    if not np.isfinite(mdd) or mdd <= ZERO_DISPERSION:
        return float("nan")
    return float(cagr(equity, bars_per_year) / mdd)


def win_rate(returns) -> float:
    """Fraction of non-zero bars that were positive.

    Flat bars are excluded from the denominator. Counting them as losses
    would make a strategy that is in cash 90 percent of the time look like it
    loses 90 percent of its bets.
    """
    r = _clean(returns)
    active = r[r != 0.0]
    if active.size == 0:
        return float("nan")
    return float(np.mean(active > 0.0))


def turnover(positions) -> float:
    """Mean absolute change in position per bar.

    A strategy flipping between -1 and +1 every bar has a turnover of 2.0.
    This is the quantity transaction costs are charged on, so it is reported
    next to them rather than buried.
    """
    p = np.asarray(positions, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(np.diff(p, prepend=0.0))))


def exposure(positions) -> float:
    """Fraction of bars with a non-zero position."""
    p = np.asarray(positions, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean(p != 0.0))


@dataclass(frozen=True)
class Report:
    """Everything the backtester reports about one run."""

    bars: int
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    drawdown_bars: int
    calmar: float
    win_rate: float
    turnover: float
    exposure: float
    cost_drag: float
    bars_per_year: float

    def as_dict(self) -> dict:
        return asdict(self)

    def lines(self) -> list[str]:
        """ASCII rows for printing. No unicode: the terminal may be CP950."""
        def num(x: float) -> str:
            return "nan" if not np.isfinite(x) else f"{x:>9.4f}"

        return [
            f"  bars           {self.bars:>9d}",
            f"  total return   {num(self.total_return)}",
            f"  CAGR           {num(self.cagr)}",
            f"  Sharpe         {num(self.sharpe)}",
            f"  Sortino        {num(self.sortino)}",
            f"  max drawdown   {num(self.max_drawdown)}",
            f"  dd duration    {self.drawdown_bars:>9d} bars",
            f"  Calmar         {num(self.calmar)}",
            f"  win rate       {num(self.win_rate)}",
            f"  turnover       {num(self.turnover)}",
            f"  exposure       {num(self.exposure)}",
            f"  cost drag      {num(self.cost_drag)}",
        ]


def summarise(
    net_returns,
    gross_returns,
    positions,
    bars_per_year: float = DAILY,
) -> Report:
    """Build a :class:`Report` from net returns, gross returns and positions.

    Both return streams are required so ``cost_drag`` can be reported as a
    measured difference rather than estimated from an average fill.
    """
    net = np.asarray(net_returns, dtype=float)
    gross = np.asarray(gross_returns, dtype=float)
    eq = equity_curve(net)
    gross_eq = equity_curve(gross)
    return Report(
        bars=int(net.size),
        total_return=float(eq[-1] - 1.0) if eq.size else float("nan"),
        cagr=cagr(eq, bars_per_year),
        sharpe=sharpe(net, bars_per_year),
        sortino=sortino(net, bars_per_year),
        max_drawdown=max_drawdown(eq),
        drawdown_bars=drawdown_duration(eq),
        calmar=calmar(eq, bars_per_year),
        win_rate=win_rate(net),
        turnover=turnover(positions),
        exposure=exposure(positions),
        cost_drag=float(gross_eq[-1] - eq[-1]) if eq.size else float("nan"),
        bars_per_year=float(bars_per_year),
    )
