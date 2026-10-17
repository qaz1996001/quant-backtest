"""Drift detection, including the cases where it must refuse to answer."""

from __future__ import annotations

import numpy as np
import pytest

from qbt import backtest, drift, features, synth


# --------------------------------------------------------------------------
# refusing to answer
# --------------------------------------------------------------------------


def test_a_short_window_reports_insufficient_data_not_ok():
    """The failure this is here to prevent: a monitor reporting green.

    An empty or barely-filled baseline table produces a verdict of
    ``insufficient_data``, which is not ``ok`` and is not an exception.
    """
    verdict = drift.feature_drift("vol_20", [0.01] * 5, 0.9)
    assert verdict.status == drift.INSUFFICIENT
    assert verdict.z is None
    assert not verdict.drifted
    assert not verdict.actionable
    assert "need 30" in verdict.detail


def test_a_flat_baseline_reports_insufficient_data_not_an_infinite_z():
    """Zero variance makes the z-score a division by zero.

    Numpy would return ``inf`` and every threshold comparison downstream
    would fire. The constant baseline is also exactly what a stuck sensor or
    a repeated cached value looks like.
    """
    verdict = drift.feature_drift("stuck", [0.5] * 200, 0.9)
    assert verdict.status == drift.INSUFFICIENT
    assert "zero variance" in verdict.detail


@pytest.mark.parametrize("n", [0, 1, 2, 29])
def test_every_window_below_the_minimum_refuses(n):
    rng = np.random.default_rng(3)
    verdict = drift.feature_drift("x", rng.normal(size=n), 1.0)
    assert verdict.status == drift.INSUFFICIENT


def test_one_point_over_the_minimum_does_answer():
    rng = np.random.default_rng(4)
    verdict = drift.feature_drift("x", rng.normal(size=drift.MIN_WINDOW), 0.1)
    assert verdict.status == drift.OK
    assert verdict.z is not None


# --------------------------------------------------------------------------
# actually detecting drift
# --------------------------------------------------------------------------


def test_feature_drift_is_two_sided():
    """An input three sigma low is as unfamiliar as three sigma high."""
    rng = np.random.default_rng(5)
    baseline = rng.normal(0.0, 1.0, 500)
    assert drift.feature_drift("x", baseline, 8.0).drifted
    assert drift.feature_drift("x", baseline, -8.0).drifted
    assert not drift.feature_drift("x", baseline, 0.2).drifted


def test_performance_drift_normalises_direction_once():
    """One comparison for both metric directions; worse is always negative z.

    Sharpe higher-is-better and drawdown lower-is-better go through the same
    inequality. Two branches would be two chances to have the sign backwards,
    and a monitor that is backwards on drawdown stays quiet when it matters.
    """
    rng = np.random.default_rng(6)
    sharpe_hist = rng.normal(1.0, 0.2, 200)
    dd_hist = rng.normal(0.08, 0.01, 200)

    worse_sharpe = drift.performance_drift(
        "sharpe", sharpe_hist, -0.5, higher_is_better=True
    )
    worse_dd = drift.performance_drift(
        "max_dd", dd_hist, 0.25, higher_is_better=False
    )

    assert worse_sharpe.drifted and worse_sharpe.z < 0
    assert worse_dd.drifted and worse_dd.z < 0


def test_an_improvement_is_not_drift():
    """One-sided on purpose. A better Sharpe does not stop a strategy."""
    rng = np.random.default_rng(7)
    hist = rng.normal(1.0, 0.2, 200)
    verdict = drift.performance_drift("sharpe", hist, 3.0, higher_is_better=True)
    assert verdict.status == drift.OK
    assert verdict.z > 0


def test_a_lower_drawdown_is_not_drift():
    rng = np.random.default_rng(8)
    hist = rng.normal(0.08, 0.01, 200)
    verdict = drift.performance_drift("max_dd", hist, 0.01, higher_is_better=False)
    assert verdict.status == drift.OK


# --------------------------------------------------------------------------
# against ground truth
# --------------------------------------------------------------------------


def test_the_volatility_regime_change_is_detected_where_it_happens():
    """The reason the data is synthetic: the regime boundary is known.

    Bar 600 triples volatility by construction. ``vol_20`` scanned across
    that boundary has to come back ``drift``, and the same feature scanned
    inside the calm regime has to come back ``ok``.
    """
    series = synth.generate(seed=99)
    edge = series.regime_bounds()[0]
    assert edge == 600

    vol = features.vol_20(series.close)
    calm = vol[100:edge]
    volatile = vol[edge + 40 : edge + 250]

    crossing = drift.feature_drift("vol_20", calm, float(np.mean(volatile)))
    assert crossing.drifted
    assert crossing.z > 3.0

    within = drift.feature_drift("vol_20", vol[100:400], float(np.mean(vol[400:550])))
    assert within.status == drift.OK


def test_scan_features_flags_volatility_and_not_everything():
    """A scanner that flags every column is as useless as one that flags none."""
    series = synth.generate(seed=101)
    design = features.build_design(series.close)
    edge = int(np.searchsorted(design.rows, series.regime_bounds()[0]))

    verdicts = drift.scan_features(design.X, design.names, split_at=edge)
    names = {v.name: v for v in verdicts}

    assert len(verdicts) == len(design.names)
    assert all(v.actionable for v in verdicts)
    assert names["vol_20"].drifted
    assert not all(v.drifted for v in verdicts)


def test_rolling_sharpe_is_nan_before_the_window_fills(close):
    result = backtest.buy_and_hold(close)
    rs = drift.rolling_sharpe(result.net_returns, window=120)
    assert np.all(np.isnan(rs[:119]))
    assert np.isfinite(rs[119])


def test_performance_drift_catches_a_strategy_that_stops_working(close):
    """Rolling Sharpe as its own baseline, against a deliberately broken tail.

    The tail is the inverse of the signal, so the strategy flips from
    working to losing. The verdict on the recent block has to be ``drift``,
    and the verdict taken before the break has to be ``ok``.
    """
    rets = backtest.simple_returns(close)
    oracle = np.zeros(close.size)
    oracle[:-1] = rets[1:]
    broken = oracle.copy()
    broken[1000:] = -broken[1000:]

    result = backtest.run(close, broken, costs=backtest.FREE)
    rs = drift.rolling_sharpe(result.net_returns, window=100)

    baseline = rs[200:950]
    baseline = baseline[~np.isnan(baseline)]
    after = float(np.nanmean(rs[1150:]))
    before = float(np.nanmean(rs[700:900]))

    assert drift.performance_drift("sharpe", baseline, after).drifted
    assert drift.performance_drift("sharpe", baseline, before).status == drift.OK


def test_verdict_lines_are_ascii():
    rng = np.random.default_rng(9)
    verdicts = [
        drift.feature_drift("x", rng.normal(size=200), 9.0),
        drift.feature_drift("y", [1.0] * 100, 2.0),
    ]
    for v in verdicts:
        v.line().encode("ascii")
