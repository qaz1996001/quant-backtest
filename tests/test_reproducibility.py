"""Same seed, same numbers. Otherwise a backtest result is an anecdote."""

from __future__ import annotations

import numpy as np

from qbt import backtest, features, pipeline, synth


def test_the_generator_is_bit_identical_across_calls():
    a = synth.generate(seed=4242)
    b = synth.generate(seed=4242)
    assert np.array_equal(a.close, b.close)
    assert np.array_equal(a.regime, b.regime)


def test_a_different_seed_gives_a_different_path():
    a = synth.generate(seed=4242)
    b = synth.generate(seed=4243)
    assert not np.array_equal(a.close, b.close)


def test_the_regime_schedule_is_exact_not_sampled():
    """Known boundaries are what let the drift tests assert against truth."""
    series = synth.generate()
    assert series.regime_bounds() == [600, 900]
    assert len(series) == 1500
    assert series.regime[599] == 0 and series.regime[600] == 1
    assert series.regime[899] == 1 and series.regime[900] == 2


def test_the_feature_matrix_is_reproducible(close):
    a = features.build_design(close)
    b = features.build_design(close)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.rows, b.rows)


def test_two_identical_walk_forward_runs_give_identical_equity(close):
    """The whole pipeline, twice, compared on the final equity value.

    Model fitting, fold construction and cost accounting all feed this
    number, so a single assertion covers the lot. It is exact equality, not
    approximate: a tolerance here would hide a non-deterministic model.
    """
    _, first = pipeline.run(close, n_folds=4, min_train=300)
    _, second = pipeline.run(close, n_folds=4, min_train=300)

    assert np.array_equal(first.net_returns, second.net_returns)
    assert first.equity[-1] == second.equity[-1]
    assert first.report.as_dict() == second.report.as_dict()


def test_the_report_dict_is_comparable_as_a_whole(close):
    """Report equality is used as a regression fence in the test above, so
    it has to actually distinguish two different runs."""
    free = backtest.buy_and_hold(close, costs=backtest.FREE).report
    charged = backtest.buy_and_hold(close, costs=backtest.Costs(5.0, 5.0)).report
    assert free.as_dict() != charged.as_dict()
