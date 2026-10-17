"""The wiring: out-of-sample coverage, and the in-sample gap it avoids."""

from __future__ import annotations

import numpy as np
import pytest

from qbt import backtest, features, leaky, metrics, pipeline, synth


def test_no_bar_is_predicted_by_a_model_that_trained_on_it(close):
    """The structural claim of the whole repository, asserted on indices.

    For each fold, the bars that received a prediction are checked against
    the bar indices the model trained on. Any intersection is a leak, and it
    is checked in bar space rather than row space because the translation
    between the two is where it would go wrong.
    """
    wf = pipeline.walk_forward_signal(close, n_folds=4, min_train=300)
    rows = wf.design.rows

    for fold in wf.folds:
        trained_on = set(rows[fold.train].tolist())
        predicted = set(rows[fold.test].tolist())
        assert trained_on.isdisjoint(predicted)
        assert max(trained_on) < min(predicted)


def test_bars_before_the_first_test_fold_have_no_signal(close):
    """They must be nan, which the backtester reads as flat.

    Filling them with zero would be the same position, but it would also
    make ``covered`` wrong and hide a fold-construction bug.
    """
    wf = pipeline.walk_forward_signal(close, n_folds=4, min_train=300)
    first_tested = wf.design.rows[wf.folds[0].test[0]]
    assert np.all(np.isnan(wf.signal[:first_tested]))
    assert not np.isnan(wf.signal[first_tested])


def test_every_test_bar_gets_exactly_one_prediction(close):
    wf = pipeline.walk_forward_signal(close, n_folds=5, min_train=300)
    tested = np.concatenate([wf.design.rows[f.test] for f in wf.folds])
    assert tested.size == np.unique(tested).size
    assert wf.covered == tested.size


def test_the_in_sample_sharpe_is_the_flattering_one(close):
    """Same data, same model, two procedures. The gap is the artefact.

    The in-sample model predicts bars it was fitted on, which is the default
    outcome of a notebook that calls ``fit`` once. The assertion is that the
    honest number is lower, which is the point worth having a test for.
    """
    insample = pipeline.in_sample_signal(close)
    wf, oos = pipeline.run(close, n_folds=5, min_train=300)

    in_report = backtest.run(close, insample, costs=backtest.FREE).report
    oos_report = backtest.run(close, wf.signal, costs=backtest.FREE).report

    assert in_report.sharpe > oos_report.sharpe
    assert np.isfinite(oos.report.sharpe)


def test_costs_make_the_out_of_sample_result_worse(close):
    _, free = pipeline.run(close, costs=backtest.FREE, n_folds=4, min_train=300)
    _, charged = pipeline.run(
        close, costs=backtest.Costs(2.0, 5.0), n_folds=4, min_train=300
    )
    assert charged.report.sharpe < free.report.sharpe
    assert charged.report.cost_drag > 0.0


def test_the_report_carries_the_backend_and_the_folds(close):
    wf, _ = pipeline.run(close, n_folds=3, min_train=300)
    assert wf.backend in ("lightgbm", "sklearn-gbr")
    assert len(wf.folds) == 3
    assert len(wf.fold_sharpe) == 3
    assert set(wf.importance) == set(wf.design.names)


def test_a_longer_horizon_purges_more(close):
    """Horizon reaches the fold geometry, not just the label."""
    wf1 = pipeline.walk_forward_signal(close, horizon=1, n_folds=3, min_train=300)
    wf5 = pipeline.walk_forward_signal(close, horizon=5, n_folds=3, min_train=300)

    gap1 = wf1.folds[0].test[0] - wf1.folds[0].train[-1] - 1
    gap5 = wf5.folds[0].test[0] - wf5.folds[0].train[-1] - 1
    assert gap5 == gap1 + 4


def test_a_higher_threshold_trades_less(close):
    """The dead band is wired through to turnover.

    Two runs of the same signal, one with a threshold. Fewer trades and
    lower exposure, which is the only observable difference a threshold
    should make.
    """
    wf = pipeline.walk_forward_signal(close, n_folds=4, min_train=300)
    loose = backtest.run(close, wf.signal, threshold=0.0)
    tight = backtest.run(close, wf.signal, threshold=0.004)

    assert tight.report.turnover < loose.report.turnover
    assert tight.report.exposure < loose.report.exposure


def test_benchmarks_are_comparable_on_the_same_bars(close):
    """Buy-and-hold and all-cash run over the same series and report the same
    bar count, so a Sharpe comparison is not comparing two sample lengths."""
    bh = backtest.buy_and_hold(close).report
    cash = backtest.all_cash(close).report
    _, oos = pipeline.run(close, n_folds=4, min_train=300)

    assert bh.bars == cash.bars == oos.report.bars == close.size
    assert bh.bars_per_year == metrics.DAILY


# --------------------------------------------------------------------------
# does the harness find a signal, and does it refuse to invent one
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_planted_signal_is_found_out_of_sample(seed):
    """AR(1) of 0.30 in the generator comes back as a positive walk-forward
    Sharpe on every seed tried. Without this, the test below would pass for
    a pipeline that is simply broken."""
    close = synth.generate(seed=seed, autocorr=0.30).close
    _, result = pipeline.run(
        close, costs=backtest.FREE, n_folds=5, min_train=300
    )
    assert result.report.sharpe > 1.0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_with_no_signal_planted_the_pipeline_finds_none(seed):
    """AR(1) of 0.0: the returns are independent and nothing is predictable.

    The walk-forward Sharpe has to come back small. A pipeline with a leak
    anywhere in it -- a feature reading ahead, a fold overlapping, a label
    off by one -- produces a confident number here, which is what makes this
    the cheapest regression test in the suite.
    """
    close = synth.generate(seed=seed, autocorr=0.0).close
    _, result = pipeline.run(
        close, costs=backtest.FREE, n_folds=5, min_train=300
    )
    assert abs(result.report.sharpe) < 1.2


def test_a_leaky_feature_set_scores_absurdly_well(close):
    """The positive control for the whole repository.

    Swap one honest feature for ``next_bar_return`` and run the identical
    walk-forward procedure. Correct folds, correct purge, correct embargo,
    correct costs -- and a Sharpe ratio that could not happen. Split
    discipline does not protect you from a feature that reads ahead, which
    is why the causality checker exists as a separate gate.
    """
    honest = dict(features.FEATURES)
    leaking = dict(features.FEATURES)
    leaking["ret_1"] = leaky.LEAKY["next_bar_return"]

    _, clean = pipeline.run(
        close, costs=backtest.FREE, feature_set=honest, n_folds=5, min_train=300
    )
    _, dirty = pipeline.run(
        close, costs=backtest.FREE, feature_set=leaking, n_folds=5, min_train=300
    )

    assert dirty.report.sharpe > 10.0
    assert dirty.report.sharpe > 3.0 * clean.report.sharpe
    assert dirty.report.max_drawdown < clean.report.max_drawdown
