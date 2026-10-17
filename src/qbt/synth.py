"""Synthetic price series with regimes you know the location of.

Real market data would make this repository need a licence, a network call
and a cache directory, and the tests would stop being run. The bigger
reason is that a synthetic series lets a test *know* where the regime
changed, so a drift detector can be checked against ground truth instead of
against a plot someone eyeballed.

The generator is a regime-switching geometric random walk. Each regime has
its own drift and volatility; the schedule is explicit, not sampled, so
``regime_bounds()`` returns the exact bar indices where behaviour changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Regime:
    """One stretch of bars with constant drift and volatility.

    ``mu`` and ``sigma`` are per-bar, not annualised. Annualisation happens
    once, in :mod:`qbt.metrics`, with the bars-per-year spelled out.
    """

    bars: int
    mu: float
    sigma: float
    label: str = ""


#: Calm -> volatile -> calm-but-trending. The middle regime triples the
#: volatility, which is what the drift tests look for.
DEFAULT_REGIMES: tuple[Regime, ...] = (
    Regime(bars=600, mu=0.0002, sigma=0.008, label="calm"),
    Regime(bars=300, mu=-0.0004, sigma=0.024, label="volatile"),
    Regime(bars=600, mu=0.0006, sigma=0.010, label="trending"),
)


@dataclass(frozen=True)
class Series:
    """A generated price path and the regime labels behind it."""

    close: np.ndarray
    regime: np.ndarray  # int index into ``regimes``
    regimes: tuple[Regime, ...]

    def __len__(self) -> int:
        return int(self.close.size)

    def regime_bounds(self) -> list[int]:
        """Bar indices at which a new regime starts (excluding bar 0)."""
        edges: list[int] = []
        at = 0
        for r in self.regimes[:-1]:
            at += r.bars
            edges.append(at)
        return edges


def generate(
    regimes: tuple[Regime, ...] = DEFAULT_REGIMES,
    seed: int = 20260101,
    start_price: float = 100.0,
    autocorr: float = 0.25,
) -> Series:
    """Generate one price path.

    ``autocorr`` is an AR(1) term on the log returns, and it is the planted
    signal. At ``0.0`` the returns are independent, nothing is predictable,
    and every model configuration scores the same, which makes a training
    test vacuous. At the default ``0.25`` there is something real to find.

    **0.25 is far stronger than any liquid market.** Daily equity return
    autocorrelation is a couple of percent at best. The number is set high
    on purpose: the repository is demonstrating that a harness finds a
    planted signal and prices it honestly, not claiming an edge. The other
    half of that claim is ``tests/test_pipeline.py``, which runs the same
    pipeline at ``autocorr=0.0`` and asserts it does **not** come back with
    a Sharpe ratio worth trading.

    The same ``seed`` gives the same array, bit for bit. That is asserted in
    ``tests/test_reproducibility.py``.
    """
    rng = np.random.default_rng(seed)
    n = sum(r.bars for r in regimes)

    mu = np.empty(n)
    sigma = np.empty(n)
    which = np.empty(n, dtype=np.int64)
    at = 0
    for i, r in enumerate(regimes):
        mu[at : at + r.bars] = r.mu
        sigma[at : at + r.bars] = r.sigma
        which[at : at + r.bars] = i
        at += r.bars

    shocks = rng.standard_normal(n)
    log_ret = np.empty(n)
    prev = 0.0
    for t in range(n):
        innovation = sigma[t] * shocks[t]
        prev = mu[t] + autocorr * prev + innovation
        log_ret[t] = prev

    close = start_price * np.exp(np.cumsum(log_ret))
    return Series(close=close, regime=which, regimes=regimes)
