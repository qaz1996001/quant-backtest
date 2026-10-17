"""Execution lag and transaction costs, asserted on numbers.

The two questions these answer: does a signal earn the bar it was computed
on (it must not), and do the cost parameters reach the equity curve (they
must).
"""

from __future__ import annotations

import numpy as np
import pytest

from qbt import backtest, metrics


# --------------------------------------------------------------------------
# execution lag
# --------------------------------------------------------------------------


def test_position_is_held_one_bar_after_the_signal(short_close):
    signal = np.zeros(short_close.size)
    signal[10] = 1.0
    result = backtest.run(short_close, signal, costs=backtest.FREE)

    assert result.position[10] == 0.0  # the signal bar earns nothing
    assert result.position[11] == 1.0
    assert result.position[12] == 0.0


def test_bar_zero_is_always_flat(short_close):
    result = backtest.run(short_close, np.ones(short_close.size))
    assert result.position[0] == 0.0
    assert result.gross_returns[0] == 0.0


def test_an_oracle_of_the_next_bar_is_right_on_every_bar(short_close):
    """Signal = next bar's return. With the lag applied it earns ``abs(ret)``.

    Establishes that the shift is wired the way the docstring claims: the
    position held during bar ``t`` is the one decided at ``t - 1``, so an
    oracle that knew ``t`` at ``t - 1`` captures the whole move and never
    takes a losing bar.
    """
    rets = backtest.simple_returns(short_close)
    oracle = np.zeros(short_close.size)
    oracle[:-1] = rets[1:]  # at bar t, know bar t+1

    result = backtest.run(short_close, oracle, costs=backtest.FREE)
    assert result.gross_returns[0] == 0.0
    assert np.all(result.gross_returns[1:] >= 0.0)
    assert np.allclose(result.gross_returns[1:], np.abs(rets[1:]))


def test_a_same_bar_signal_earns_nothing_once_the_lag_is_applied(short_close):
    """The leak, measured. Signal = the bar's own return, which is known.

    Applied without a lag it scores perfectly -- ``sum(abs(ret))``, because
    the position was chosen with the answer in hand. Applied with the one-bar
    lag the harness enforces, the same numbers are worth nothing much. The
    gap between those two totals is what a lookahead bug is worth on paper.

    ``short_close`` has ``autocorr=0.0`` and the reason is worth stating: on
    the default series, which has a planted AR(1) of 0.25, yesterday's
    return genuinely predicts today's, and the lagged version earns a real
    1.0 instead of nothing. That made an earlier version of this test fail
    for the right reason, which is how the fixture ended up signal-free.
    """
    rets = backtest.simple_returns(short_close)

    lagged = backtest.run(short_close, rets, costs=backtest.FREE)
    honest_total = float(np.sum(lagged.gross_returns))
    leaked_total = float(np.sum(np.sign(rets) * rets))  # no lag: the bug

    assert leaked_total == pytest.approx(float(np.sum(np.abs(rets))))
    assert leaked_total > 10.0 * abs(honest_total)
    assert honest_total < 0.0  # on noise, holding yesterday's sign loses


def test_signal_length_mismatch_raises(short_close):
    with pytest.raises(ValueError, match="bars"):
        backtest.run(short_close, np.ones(short_close.size - 3))


def test_nan_signal_means_flat_not_forward_filled(short_close):
    signal = np.full(short_close.size, np.nan)
    signal[5:10] = 1.0
    result = backtest.run(short_close, signal, costs=backtest.FREE)

    assert np.all(result.position[11:] == 0.0)
    assert result.position[6] == 1.0


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------


def test_costs_are_non_zero_by_default():
    """The default has to charge something, or every test here is vacuous."""
    assert backtest.Costs().per_unit > 0.0


def test_zero_cost_and_real_cost_give_different_sharpe(close):
    """The headline cost assertion: the parameter reaches the equity curve.

    A flip-flopping signal is used so turnover is high and the gap is not
    inside float noise.
    """
    flip = np.where(np.arange(close.size) % 2 == 0, 1.0, -1.0)

    free = backtest.run(close, flip, costs=backtest.FREE)
    charged = backtest.run(close, flip, costs=backtest.Costs(2.0, 5.0))

    assert free.report.sharpe != charged.report.sharpe
    assert charged.report.sharpe < free.report.sharpe
    assert free.report.cost_drag == pytest.approx(0.0, abs=1e-12)
    assert charged.report.cost_drag > 0.0
    assert charged.equity[-1] < free.equity[-1]


def test_cost_drag_scales_with_the_cost_parameter(close):
    flip = np.where(np.arange(close.size) % 2 == 0, 1.0, -1.0)
    cheap = backtest.run(close, flip, costs=backtest.Costs(1.0, 1.0))
    dear = backtest.run(close, flip, costs=backtest.Costs(10.0, 10.0))
    assert dear.report.cost_drag > cheap.report.cost_drag


def test_a_flip_pays_twice_a_single_entry(short_close):
    """Turnover is charged on absolute position change, so -1 to +1 is 2.0."""
    costs = backtest.Costs(commission_bps=10.0, slippage_bps=0.0)

    entry = np.zeros(short_close.size)
    entry[10] = 1.0
    flip = np.zeros(short_close.size)
    flip[10] = -1.0
    flip[11] = 1.0

    one_way = backtest.run(short_close, entry, costs=costs).costs.sum()
    two_way = backtest.run(short_close, flip, costs=costs).costs.sum()
    assert two_way == pytest.approx(2.0 * one_way)


def test_turnover_of_a_bar_by_bar_flip_is_two(close):
    flip = np.where(np.arange(close.size) % 2 == 0, 1.0, -1.0)
    result = backtest.run(close, flip)
    assert result.report.turnover == pytest.approx(2.0, abs=0.01)


def test_buy_and_hold_pays_cost_once(close):
    result = backtest.buy_and_hold(close, costs=backtest.Costs(2.0, 5.0))
    assert np.count_nonzero(result.costs) == 1
    assert result.report.turnover == pytest.approx(1.0 / close.size, rel=1e-9)


def test_a_weak_edge_is_eaten_entirely_by_costs(close):
    """The reason the default is not zero: profitable gross, losing net.

    The signal is right on 58 percent of bars, which is a strong edge by the
    standards of anything real. It flips often, so turnover is near 1.0 per
    bar and at 50 bps that is roughly 50 bps of cost against about 13 bps of
    edge. A backtest run with costs off would have reported the first number
    and nothing else.

    A perfect oracle was the first version of this test and it failed: on
    this series the average absolute bar return is about 80 bps, so even a
    50 bps round trip leaves an oracle enormously profitable. The edge has
    to be realistic for the cost to matter, which is itself the lesson.
    """
    rng = np.random.default_rng(11)
    rets = backtest.simple_returns(close)
    truth = np.sign(rets[1:])
    wrong = rng.random(truth.size) > 0.58
    noisy = np.where(wrong, -truth, truth)

    signal = np.zeros(close.size)
    signal[:-1] = noisy

    gross = backtest.run(close, signal, costs=backtest.FREE)
    net = backtest.run(close, signal, costs=backtest.Costs(20.0, 30.0))

    assert gross.report.total_return > 0.0
    assert net.report.total_return < 0.0
    assert net.report.sharpe < 0.0 < gross.report.sharpe
    assert net.report.turnover > 0.7
