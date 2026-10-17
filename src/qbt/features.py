"""Time-series features, plus a checker that tries to catch them cheating.

Convention, stated once and relied on everywhere: a feature at index ``t``
may read ``close[0..t]`` inclusive. Bar ``t`` has closed, so its close is
known. Turning that number into a trade is the backtester's problem, and
:mod:`qbt.backtest` holds the position one bar before it earns anything.

Every feature returns an array the same length as the input, with ``nan``
during warm-up. Dropping rows instead would silently shift indices, and a
shifted index is the most common way a lookahead bug gets in.

The interesting function here is :func:`causality_report`. It does not read
the source code and it does not trust the docstring. It perturbs the future
and checks whether the past moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

FeatureFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def _rolling(x: np.ndarray, window: int, fn) -> np.ndarray:
    """Trailing rolling reduction. Window ends at ``t``, inclusive.

    Written as an explicit loop rather than a stride trick because the
    index arithmetic is the thing most likely to be wrong, and a loop makes
    it readable. At this size the cost does not matter.
    """
    out = np.full(x.size, np.nan)
    for t in range(window - 1, x.size):
        out[t] = fn(x[t - window + 1 : t + 1])
    return out


def log_returns(close: np.ndarray) -> np.ndarray:
    """One-bar log return. ``nan`` at bar 0: there is no previous bar."""
    out = np.full(close.size, np.nan)
    out[1:] = np.diff(np.log(close))
    return out


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def ret_1(close: np.ndarray) -> np.ndarray:
    return log_returns(close)


def ret_5(close: np.ndarray) -> np.ndarray:
    """Five-bar log return, measured backwards from ``t``."""
    out = np.full(close.size, np.nan)
    out[5:] = np.log(close[5:] / close[:-5])
    return out


def vol_20(close: np.ndarray) -> np.ndarray:
    """Trailing 20-bar standard deviation of one-bar returns."""
    r = log_returns(close)
    out = np.full(close.size, np.nan)
    for t in range(20, close.size):
        out[t] = np.std(r[t - 19 : t + 1], ddof=1)
    return out


def momentum_10(close: np.ndarray) -> np.ndarray:
    """Close over its own trailing 10-bar mean, minus one."""
    ma = _rolling(close, 10, np.mean)
    return close / ma - 1.0


def rsi_14(close: np.ndarray, window: int = 14) -> np.ndarray:
    """Wilder-style RSI on a simple (not exponential) average.

    A simple average is used so the warm-up boundary is exact. An EMA has no
    clean first valid index, which makes "is this value contaminated by
    initialisation" a judgement call rather than a test.
    """
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[0] = loss[0] = np.nan

    out = np.full(close.size, np.nan)
    for t in range(window, close.size):
        avg_gain = np.mean(gain[t - window + 1 : t + 1])
        avg_loss = np.mean(loss[t - window + 1 : t + 1])
        if avg_loss == 0.0:
            out[t] = 100.0 if avg_gain > 0 else 50.0
        else:
            out[t] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def zscore_60(close: np.ndarray, window: int = 60) -> np.ndarray:
    """Where the close sits inside its own trailing distribution."""
    out = np.full(close.size, np.nan)
    for t in range(window - 1, close.size):
        w = close[t - window + 1 : t + 1]
        sd = np.std(w, ddof=1)
        out[t] = 0.0 if sd == 0 else (close[t] - np.mean(w)) / sd
    return out


#: The feature set the model trains on. Ordered, because column order has to
#: be stable between fit and predict and a dict that gets reordered is a
#: silent accuracy loss.
FEATURES: dict[str, FeatureFn] = {
    "ret_1": ret_1,
    "ret_5": ret_5,
    "vol_20": vol_20,
    "momentum_10": momentum_10,
    "rsi_14": rsi_14,
    "zscore_60": zscore_60,
}


# --------------------------------------------------------------------------
# the matrix, and the label
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    """Feature matrix, label, and the rows that survived warm-up.

    ``rows`` maps each row of ``X`` back to its bar index in the original
    series. The backtester needs that mapping to line a prediction up with
    the bar it may act on; carrying it in the same object is what stops the
    two from drifting apart.
    """

    X: np.ndarray
    y: np.ndarray
    rows: np.ndarray
    names: tuple[str, ...]


def forward_return(close: np.ndarray, horizon: int = 1) -> np.ndarray:
    """The label: log return from ``t`` to ``t + horizon``.

    This one *is* a future value, deliberately. It is the thing being
    predicted. The last ``horizon`` entries are ``nan`` and are dropped from
    the design matrix, because a row with an unknown label is not training
    data, and filling it with zero would teach the model that the series
    ends flat.
    """
    out = np.full(close.size, np.nan)
    out[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    return out


def build_design(
    close: np.ndarray,
    features: dict[str, FeatureFn] | None = None,
    horizon: int = 1,
) -> Design:
    """Assemble the feature matrix and drop every row with any ``nan``."""
    features = FEATURES if features is None else features
    names = tuple(features)
    cols = [features[n](close) for n in names]
    mat = np.column_stack(cols)
    y = forward_return(close, horizon)

    keep = ~np.isnan(mat).any(axis=1) & ~np.isnan(y)
    rows = np.flatnonzero(keep)
    return Design(X=mat[keep], y=y[keep], rows=rows, names=names)


# --------------------------------------------------------------------------
# the causality checker
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalityReport:
    """Whether a feature function reads the future.

    ``status`` is ``"causal"`` or ``"leaks"``. ``first_leak`` is the lowest
    index whose value changed when only future bars were edited; it is the
    useful part of a failure, because it tells you how far ahead the
    function is looking.

    ``lookahead`` is ``cut - first_leak + 1``, so it is only meaningful next
    to the ``cut`` that produced it. A feature standardised over the whole
    series leaks at index 0 from every cut, and reading its ``lookahead`` as
    a window length would be wrong; both fields are kept so the distinction
    is visible rather than inferred.
    """

    name: str
    status: str
    first_leak: int | None = None
    lookahead: int | None = None
    cut: int | None = None
    detail: str = ""

    @property
    def causal(self) -> bool:
        return self.status == "causal"


def default_cuts(n: int) -> tuple[int, ...]:
    """Where to split the series when probing for leaks.

    One cut is not enough, and finding that out cost an afternoon. A feature
    that back-fills its warm-up ``nan`` from the first valid value leaks
    information backwards over the first sixty bars only; probed at the
    midpoint it looks perfectly causal, because by then there is nothing left
    to back-fill. An early cut catches it. The late cut catches features that
    only misbehave near the end of the array, which is where off-by-one
    slicing lives.
    """
    return tuple(sorted({20, n // 10, n // 4, n // 2, (3 * n) // 4, n - 5}))


def causality_report(
    fn: FeatureFn,
    close: np.ndarray,
    cut: int | None = None,
    name: str = "",
    seed: int = 7,
) -> CausalityReport:
    """Perturb the future, then look for movement in the past.

    Compute the feature on the full series. Replace every bar after ``cut``
    with a different path. Recompute. Any index at or below ``cut`` whose
    value changed could only have changed by reading a bar it should not
    have seen.

    With ``cut=None`` every cut from :func:`default_cuts` is probed and the
    first leak found is returned.

    The perturbation is a large multiplicative shock rather than noise,
    because a tiny edit can hide inside float rounding and a checker that
    can be fooled by rounding is worse than no checker. It is ``exp`` of a
    normal draw rather than ``1 + normal`` so the perturbed path stays
    strictly positive -- a negative price makes every log-return feature
    return ``nan``, and a checker that reports a leak because it fed the
    feature an impossible series is a checker nobody will trust twice.
    """
    label = name or getattr(fn, "__name__", "?")
    if cut is None:
        for c in default_cuts(close.size):
            report = causality_report(fn, close, cut=c, name=label, seed=seed)
            if not report.causal:
                return report
        return CausalityReport(label, "causal")

    base = np.asarray(fn(close), dtype=float)

    rng = np.random.default_rng(seed)
    altered = close.copy()
    tail = close.size - (cut + 1)
    altered[cut + 1 :] = close[cut + 1 :] * np.exp(0.5 * rng.standard_normal(tail))
    probe = np.asarray(fn(altered), dtype=float)

    head = slice(0, cut + 1)
    a, b = base[head], probe[head]
    # nan appearing or vanishing is also a change, so compare nan-ness too.
    changed = (np.isnan(a) != np.isnan(b)) | (
        ~np.isnan(a) & ~np.isnan(b) & ~np.isclose(a, b, rtol=1e-12, atol=1e-15)
    )
    if not changed.any():
        return CausalityReport(label, "causal")

    first = int(np.flatnonzero(changed)[0])
    return CausalityReport(
        label,
        "leaks",
        first_leak=first,
        lookahead=cut - first + 1,
        cut=cut,
        detail=f"index {first} moved when only bars > {cut} were edited",
    )


def audit(
    close: np.ndarray, features: dict[str, FeatureFn] | None = None
) -> list[CausalityReport]:
    """Run :func:`causality_report` over a whole feature set."""
    features = FEATURES if features is None else features
    return [causality_report(fn, close, name=n) for n, fn in features.items()]
