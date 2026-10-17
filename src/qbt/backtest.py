"""Signal -> position -> equity, with one bar of execution lag.

The lag is the point of this module. A prediction made from the close of bar
``t`` cannot earn the return of bar ``t``; it earns bar ``t + 1``. Written
as one line:

    position[t] = target[t - 1]
    gross[t]    = position[t] * simple_return[t]

Getting that wrong produces a backtest that looks excellent and is a
restatement of the data. ``tests/test_backtest.py`` asserts the shift
directly by feeding in a signal that is the future return itself -- a
perfect oracle, one bar late -- and checking the first bar earns nothing.

Costs default to non-zero. A zero-cost default is the version that gets
left in by accident, and a strategy whose edge is smaller than its spread
is the most common thing a backtest fails to notice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics


@dataclass(frozen=True)
class Costs:
    """Per-side trading frictions, in basis points of notional traded.

    Defaults are deliberately non-zero and deliberately unremarkable: 2 bps
    commission and 5 bps slippage are plausible for a liquid single name and
    optimistic for anything else. They are charged on ``abs`` position
    change, so a flip from -1 to +1 pays twice.
    """

    commission_bps: float = 2.0
    slippage_bps: float = 5.0

    @property
    def per_unit(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4

    @classmethod
    def free(cls) -> "Costs":
        """Zero costs. Exists so a test can name what it is doing."""
        return cls(commission_bps=0.0, slippage_bps=0.0)


FREE = Costs.free()


@dataclass(frozen=True)
class Result:
    """Per-bar arrays plus the summary. Arrays are kept so a caller can plot.

    All arrays have the same length as ``close``, with bar 0 zeroed. Equal
    lengths mean an index means the same thing in every array, which is one
    fewer alignment to get wrong.
    """

    position: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    costs: np.ndarray
    equity: np.ndarray
    report: metrics.Report


def simple_returns(close) -> np.ndarray:
    """One-bar simple returns, with ``0.0`` at bar 0."""
    c = np.asarray(close, dtype=float)
    out = np.zeros(c.size)
    out[1:] = c[1:] / c[:-1] - 1.0
    return out


def positions_from_signal(
    signal,
    threshold: float = 0.0,
    max_position: float = 1.0,
    mode: str = "sign",
) -> np.ndarray:
    """Turn a prediction per bar into a target position per bar.

    ``mode="sign"`` takes long/short/flat from the sign, with a dead band of
    ``threshold`` around zero. ``mode="scaled"`` sizes proportionally to the
    prediction, clipped to ``max_position``.

    ``nan`` becomes a flat position. A model with no prediction for a bar has
    no opinion, and forward-filling the last one would hold a position on the
    strength of a stale number.
    """
    s = np.asarray(signal, dtype=float)
    out = np.zeros(s.size)
    valid = ~np.isnan(s)

    if mode == "sign":
        out[valid & (s > threshold)] = max_position
        out[valid & (s < -threshold)] = -max_position
    elif mode == "scaled":
        scale = np.where(valid, s, 0.0)
        band = np.where(np.abs(scale) > threshold, scale, 0.0)
        denom = np.max(np.abs(band)) or 1.0
        out = np.clip(band / denom, -max_position, max_position)
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'sign' or 'scaled'")
    return out


def run(
    close,
    signal,
    costs: Costs | None = None,
    threshold: float = 0.0,
    mode: str = "sign",
    bars_per_year: float = metrics.DAILY,
) -> Result:
    """Backtest one signal over one price series.

    ``signal[t]`` is what you knew at the close of bar ``t``. The resulting
    position is held over bar ``t + 1``. Costs are charged on the bar the
    position changes, which is the same bar that position starts earning.
    """
    c = np.asarray(close, dtype=float)
    costs = Costs() if costs is None else costs
    target = positions_from_signal(signal, threshold=threshold, mode=mode)
    if target.size != c.size:
        raise ValueError(f"signal has {target.size} bars, close has {c.size}")

    # The whole argument of this module, in one shift.
    position = np.zeros(c.size)
    position[1:] = target[:-1]

    rets = simple_returns(c)
    gross = position * rets
    traded = np.abs(np.diff(position, prepend=0.0))
    cost = traded * costs.per_unit
    net = gross - cost

    return Result(
        position=position,
        gross_returns=gross,
        net_returns=net,
        costs=cost,
        equity=metrics.equity_curve(net),
        report=metrics.summarise(net, gross, position, bars_per_year),
    )


def buy_and_hold(close, costs: Costs | None = None, **kw) -> Result:
    """The benchmark. A strategy that cannot beat it has not earned its fees."""
    c = np.asarray(close, dtype=float)
    return run(c, np.ones(c.size), costs=costs, **kw)


def all_cash(close, costs: Costs | None = None, **kw) -> Result:
    """Hold nothing, ever. The null strategy the metrics are checked against."""
    c = np.asarray(close, dtype=float)
    return run(c, np.zeros(c.size), costs=costs, **kw)
