"""Features written to cheat. Shipped on purpose, as test fixtures.

A checker that is only ever run against correct features proves nothing
about the checker. These four are real mistakes -- each one is a thing that
arrives by a plausible route, not a thing anyone types deliberately -- and
``tests/test_lookahead.py`` asserts that
:func:`qbt.features.causality_report` catches every one of them.

Nothing here is importable from :mod:`qbt.features`, and nothing here
appears in ``FEATURES``. The separation is the point: this module is the
negative control.
"""

from __future__ import annotations

import numpy as np

from .features import zscore_60


def next_bar_return(close: np.ndarray) -> np.ndarray:
    """Shift the wrong way. One bar of pure foresight.

    Arrives from writing ``close[t+1]/close[t]`` for the label and then
    reusing the same line as a feature.
    """
    out = np.full(close.size, np.nan)
    out[:-1] = close[1:] / close[:-1] - 1.0
    return out


def centred_window(close: np.ndarray, window: int = 21) -> np.ndarray:
    """A centred moving average. Half the window is in the future.

    Arrives by way of a smoothing step borrowed from a signal-processing
    notebook, where centring is correct and harmless.
    """
    half = window // 2
    out = np.full(close.size, np.nan)
    for t in range(half, close.size - half):
        out[t] = close[t] / np.mean(close[t - half : t + half + 1]) - 1.0
    return out


def full_sample_zscore(close: np.ndarray) -> np.ndarray:
    """Standardise against the mean and sigma of the entire series.

    The subtlest of the four, because no index in this function reaches
    forward. The scaler does: the mean it subtracts was computed over bars
    that had not happened yet. Fitting a ``StandardScaler`` before splitting
    is the same bug with more machinery.
    """
    return (close - close.mean()) / close.std(ddof=1)


def backfilled_warmup(close: np.ndarray) -> np.ndarray:
    """Back-fill the warm-up ``nan`` from the first valid value.

    A two-character fix for an annoying gap at the left edge of a chart. The
    first sixty bars now hold a number derived from bar sixty. This is the
    one that a single mid-series probe cannot see, which is why
    :func:`qbt.features.default_cuts` includes an early cut.
    """
    out = zscore_60(close)
    first = np.flatnonzero(~np.isnan(out))
    if first.size:
        out[: first[0]] = out[first[0]]
    return out


#: Every leaky feature, by name. Iterated by the test suite and the example.
LEAKY = {
    "next_bar_return": next_bar_return,
    "centred_window": centred_window,
    "full_sample_zscore": full_sample_zscore,
    "backfilled_warmup": backfilled_warmup,
}
