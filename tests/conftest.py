"""Shared fixtures. Every series here comes from a fixed seed."""

from __future__ import annotations

import numpy as np
import pytest

from qbt import synth

SEED = 20260101


@pytest.fixture(scope="session")
def series() -> synth.Series:
    """The default three-regime path, 1500 bars."""
    return synth.generate(seed=SEED)


@pytest.fixture(scope="session")
def close(series) -> np.ndarray:
    return series.close


@pytest.fixture
def short_close() -> np.ndarray:
    """A 600-bar single-regime path with no planted signal.

    ``autocorr=0.0`` on purpose. The tests that use this fixture are about
    mechanics -- the one-bar lag, cost arithmetic, position bookkeeping --
    and a predictable series would let a real edge leak into assertions that
    are supposed to be measuring plumbing.
    """
    return synth.generate(
        regimes=(synth.Regime(bars=600, mu=0.0003, sigma=0.01, label="calm"),),
        seed=SEED,
        autocorr=0.0,
    ).close
