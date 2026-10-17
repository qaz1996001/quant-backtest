"""Metrics, checked against cases where the answer is known in advance.

Most of these are degenerate strategies. A metric function that returns a
plausible number for a strategy that did nothing will return a plausible
number for a strategy that lost money, and then nobody can tell.
"""

from __future__ import annotations

import numpy as np
import pytest

from qbt import backtest, metrics


# --------------------------------------------------------------------------
# the null strategy
# --------------------------------------------------------------------------


def test_all_cash_has_zero_drawdown_and_no_sharpe(close):
    """Hold nothing. Max drawdown is 0; Sharpe is nan, not a number.

    ``0.0`` would be a claim about risk-adjusted return where there were no
    returns to adjust. Returning nan forces whatever reads it to notice.
    """
    result = backtest.all_cash(close)
    r = result.report

    assert r.max_drawdown == 0.0
    assert r.drawdown_bars == 0
    assert np.isnan(r.sharpe)
    assert np.isnan(r.sortino)
    assert np.isnan(r.calmar)  # no drawdown to divide by
    assert np.isnan(r.win_rate)  # no non-flat bars to win or lose
    assert r.total_return == pytest.approx(0.0)
    assert r.turnover == 0.0
    assert r.exposure == 0.0
    assert result.costs.sum() == 0.0  # no trades, so nothing to charge


def test_a_flat_return_series_has_no_sharpe():
    assert np.isnan(metrics.sharpe(np.zeros(500)))


def test_a_constant_non_zero_return_also_has_no_sharpe():
    """Zero dispersion, not zero mean, is what makes Sharpe undefined.

    A series of identical +1 bps bars has infinite risk-adjusted return by
    the formula. Reporting a large finite number instead would be the most
    flattering possible bug.
    """
    assert np.isnan(metrics.sharpe(np.full(500, 0.0001)))


def test_one_observation_has_no_sharpe():
    assert np.isnan(metrics.sharpe([0.01]))
    assert np.isnan(metrics.sharpe([]))


# --------------------------------------------------------------------------
# the strategy that must lose
# --------------------------------------------------------------------------


def test_inverting_a_profitable_signal_inverts_the_sign_of_sharpe(close):
    """A metric that never prints a bad number is not measuring anything.

    The signal is an oracle, so the long version is strongly positive. The
    inverse of it has to be strongly negative, and the two Sharpe ratios
    have to be near mirror images once costs are off.
    """
    rets = backtest.simple_returns(close)
    oracle = np.zeros(close.size)
    oracle[:-1] = rets[1:]

    good = backtest.run(close, oracle, costs=backtest.FREE).report
    bad = backtest.run(close, -oracle, costs=backtest.FREE).report

    assert good.sharpe > 0.0
    assert bad.sharpe < 0.0
    assert bad.sharpe == pytest.approx(-good.sharpe, rel=1e-9)
    assert bad.total_return < 0.0
    assert bad.max_drawdown > 0.5  # wrong on every bar, so it bleeds out


def test_a_losing_strategy_reports_a_negative_calmar(close):
    rets = backtest.simple_returns(close)
    inverse = np.zeros(close.size)
    inverse[:-1] = -rets[1:]
    report = backtest.run(close, inverse, costs=backtest.FREE).report
    assert report.cagr < 0.0
    assert report.calmar < 0.0


# --------------------------------------------------------------------------
# drawdown
# --------------------------------------------------------------------------


def test_max_drawdown_on_a_hand_built_curve():
    equity = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 0.65])
    # Worst fall is 1.3 -> 0.65, a half.
    assert metrics.max_drawdown(equity) == pytest.approx(0.5)


def test_max_drawdown_is_never_negative_and_zero_means_monotone():
    assert metrics.max_drawdown(np.array([1.0, 1.1, 1.2, 1.3])) == 0.0
    assert metrics.max_drawdown(np.array([1.0, 1.0, 1.0])) == 0.0


def test_drawdown_duration_counts_bars_under_the_peak():
    equity = np.array([1.0, 1.2, 1.1, 1.05, 1.15, 1.3, 1.25])
    assert metrics.drawdown_duration(equity) == 3


# --------------------------------------------------------------------------
# annualisation
# --------------------------------------------------------------------------


def test_annualisation_is_a_parameter_not_a_constant():
    """Same returns, two bar sizes, two Sharpe ratios, ratio sqrt(252/52).

    Written as an assertion because "annualised" on its own identifies
    nothing. A number computed at daily frequency and read as weekly is
    wrong by 2.2x.
    """
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 1000)

    daily = metrics.sharpe(r, metrics.BARS_PER_YEAR["daily"])
    weekly = metrics.sharpe(r, metrics.BARS_PER_YEAR["weekly"])
    assert daily / weekly == pytest.approx(np.sqrt(252.0 / 52.0))


def test_the_bar_size_used_is_carried_in_the_report(close):
    report = backtest.buy_and_hold(
        close, bars_per_year=metrics.BARS_PER_YEAR["hourly"]
    ).report
    assert report.bars_per_year == pytest.approx(252.0 * 6.5)


def test_risk_free_rate_lowers_sharpe():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0005, 0.01, 1000)
    assert metrics.sharpe(r, risk_free=0.0002) < metrics.sharpe(r, risk_free=0.0)


def test_sharpe_uses_the_sample_standard_deviation():
    """``ddof=1``. With ddof=0 a short series gets a flattering denominator."""
    r = np.array([0.01, -0.01, 0.02, 0.0, 0.015])
    expected = np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0)
    assert metrics.sharpe(r) == pytest.approx(expected)


# --------------------------------------------------------------------------
# the rest of the report
# --------------------------------------------------------------------------


def test_win_rate_ignores_flat_bars():
    """A strategy in cash most of the time is not losing those bars.

    Three wins, one loss, and a long stretch of nothing gives 0.75, not
    0.75 diluted by the flat bars.
    """
    r = np.array([0.0] * 20 + [0.01, 0.02, -0.01, 0.03])
    assert metrics.win_rate(r) == pytest.approx(0.75)


def test_turnover_and_exposure_on_a_known_position_path():
    pos = np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0])
    # changes: 0, 1, 0, 0, 1, 1, 0 -> 3/7
    assert metrics.turnover(pos) == pytest.approx(3.0 / 7.0)
    assert metrics.exposure(pos) == pytest.approx(5.0 / 7.0)


def test_report_lines_are_pure_ascii(close):
    """The terminal here is CP950. A degree sign in a report crashes a run."""
    report = backtest.buy_and_hold(close).report
    for line in report.lines():
        line.encode("ascii")
    assert "nan" in "".join(backtest.all_cash(close).report.lines())


def test_cost_drag_is_a_measured_difference_not_an_estimate(close):
    flip = np.where(np.arange(close.size) % 2 == 0, 1.0, -1.0)
    result = backtest.run(close, flip, costs=backtest.Costs(3.0, 4.0))
    gross_eq = metrics.equity_curve(result.gross_returns)
    assert result.report.cost_drag == pytest.approx(
        gross_eq[-1] - result.equity[-1]
    )
