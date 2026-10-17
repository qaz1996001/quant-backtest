"""Sliding-window drift detection for features and for live performance.

A backtest tells you a strategy worked on a stretch of history. It says
nothing about whether today's input distribution still resembles that
stretch. These are the two checks worth running after deployment:

* ``feature_drift`` -- the inputs moved. The model is being asked about a
  regime it was not fitted on.
* ``performance_drift`` -- the inputs look normal and the output got worse.
  That is the one that means stop trading.

``insufficient_data`` is a first-class verdict, not an error and not an
``ok``. A ten-bar window or a perfectly flat baseline cannot support a
z-score, and returning ``ok`` there is how a monitor reports green for a
week because its baseline table was empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Below this many baseline points a z-score is not reported. Chosen rather
#: than derived: with fewer than 30 points the sample sigma of a fat-tailed
#: return series swings enough to make the threshold meaningless.
MIN_WINDOW = 30

OK = "ok"
DRIFT = "drift"
INSUFFICIENT = "insufficient_data"


@dataclass(frozen=True)
class Verdict:
    """One metric, one window, one answer."""

    name: str
    status: str
    z: float | None = None
    detail: str = ""

    @property
    def drifted(self) -> bool:
        return self.status == DRIFT

    @property
    def actionable(self) -> bool:
        """Whether this verdict supports any decision at all."""
        return self.status in (OK, DRIFT)

    def line(self) -> str:
        z = "    n/a" if self.z is None else f"{self.z:>7.2f}"
        return f"  {self.name:<16} {self.status:<18} z={z}  {self.detail}"


def _zscore(baseline: np.ndarray, current: float) -> tuple[float | None, str]:
    if baseline.size < MIN_WINDOW:
        return None, f"baseline has {baseline.size} points, need {MIN_WINDOW}"
    sigma = float(np.std(baseline, ddof=1))
    if sigma == 0.0:
        return None, "baseline has zero variance"
    return float((current - float(np.mean(baseline))) / sigma), ""


def feature_drift(
    name: str, baseline, current: float, z_threshold: float = 3.0
) -> Verdict:
    """Two-sided: a feature moving either way is out of distribution.

    Direction carries no information for an input. A volatility feature three
    sigma *below* its training range is as unfamiliar to the model as three
    sigma above.
    """
    b = np.asarray(baseline, dtype=float)
    b = b[~np.isnan(b)]
    z, why = _zscore(b, current)
    if z is None:
        return Verdict(name, INSUFFICIENT, detail=why)
    if abs(z) > z_threshold:
        return Verdict(name, DRIFT, z=z, detail=f"|z| over {z_threshold}")
    return Verdict(name, OK, z=z)


def performance_drift(
    name: str,
    baseline,
    current: float,
    z_threshold: float = 2.0,
    higher_is_better: bool = True,
) -> Verdict:
    """One-sided, after normalising direction.

    ``higher_is_better`` flips the sign once, so everything downstream
    compares the same way and a negative z always reads as "worse". Keeping
    two comparison branches would mean two places for the inequality to be
    backwards, and a monitor that is backwards on drawdown stays quiet
    exactly when it should not.
    """
    b = np.asarray(baseline, dtype=float)
    b = b[~np.isnan(b)]
    raw, why = _zscore(b, current)
    if raw is None:
        return Verdict(name, INSUFFICIENT, detail=why)
    z = raw if higher_is_better else -raw
    if z < -z_threshold:
        return Verdict(name, DRIFT, z=z, detail="moved against the metric")
    return Verdict(name, OK, z=z)


def scan_features(
    X,
    names: tuple[str, ...],
    split_at: int,
    z_threshold: float = 3.0,
) -> list[Verdict]:
    """Compare each feature's post-``split_at`` mean against its own history.

    The current value is the mean of the recent block rather than one bar.
    A single bar crosses three sigma often enough to be useless as an alert;
    a block mean crossing it is a distribution change.
    """
    M = np.asarray(X, dtype=float)
    out: list[Verdict] = []
    for j, name in enumerate(names):
        col = M[:, j]
        baseline, recent = col[:split_at], col[split_at:]
        if recent.size == 0:
            out.append(Verdict(name, INSUFFICIENT, detail="no recent block"))
            continue
        out.append(
            feature_drift(name, baseline, float(np.nanmean(recent)), z_threshold)
        )
    return out


def rolling_sharpe(returns, window: int, bars_per_year: float = 252.0) -> np.ndarray:
    """Trailing annualised Sharpe per bar; ``nan`` before the window fills.

    Fed to :func:`performance_drift` as its own baseline, which is how a
    strategy's decay gets caught without waiting for a quarterly review.
    """
    from .metrics import sharpe

    r = np.asarray(returns, dtype=float)
    out = np.full(r.size, np.nan)
    for t in range(window - 1, r.size):
        out[t] = sharpe(r[t - window + 1 : t + 1], bars_per_year)
    return out
