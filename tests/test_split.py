"""Walk-forward folds: no overlap, no backwards training, embargo enforced."""

from __future__ import annotations

import numpy as np
import pytest

from qbt import split


def test_train_and_test_never_share_a_row():
    for fold in split.walk_forward(n=1200, n_folds=5, min_train=300):
        assert np.intersect1d(fold.train, fold.test).size == 0


def test_test_folds_do_not_overlap_each_other():
    folds = split.walk_forward(n=1200, n_folds=5, min_train=300)
    for a, b in zip(folds, folds[1:]):
        assert a.test[-1] < b.test[0]


def test_test_folds_cover_the_tail_without_a_hole():
    """Consecutive test folds are contiguous and the last one reaches the end.

    A gap between test folds means bars that were never evaluated, which
    quietly removes whatever happened in them from the reported Sharpe.
    """
    n = 1200
    folds = split.walk_forward(n=n, n_folds=5, min_train=300)
    for a, b in zip(folds, folds[1:]):
        assert b.test[0] == a.test[-1] + 1
    assert folds[-1].test[-1] == n - 1


def test_training_is_always_strictly_in_the_past():
    for fold in split.walk_forward(n=1200, n_folds=4, min_train=300):
        assert fold.train.max() < fold.test.min()


def test_the_gap_is_exactly_purge_plus_embargo():
    """The measurable consequence of the embargo parameter.

    Last training row to first test row is ``horizon + embargo`` rows. The
    assertion is on the distance, not on the parameter being stored, because
    a parameter that is recorded and not applied is the failure mode.
    """
    horizon, embargo = 3, 15
    for fold in split.walk_forward(
        n=1500, n_folds=4, min_train=400, horizon=horizon, embargo=embargo
    ):
        gap = fold.test[0] - fold.train[-1] - 1
        assert gap == horizon + embargo


@pytest.mark.parametrize("embargo", [0, 5, 25, 60])
def test_a_bigger_embargo_produces_a_bigger_gap(embargo):
    folds = split.walk_forward(
        n=1500, n_folds=3, min_train=400, horizon=2, embargo=embargo
    )
    gap = folds[0].test[0] - folds[0].train[-1] - 1
    assert gap == 2 + embargo


def test_zero_embargo_still_purges_the_label_horizon():
    """Turning the embargo off must not turn the purge off with it.

    They are separate leaks and separate parameters. Tested because the
    tempting implementation computes one gap and uses it for both.
    """
    folds = split.walk_forward(n=1200, n_folds=3, min_train=300, horizon=5, embargo=0)
    for fold in folds:
        assert fold.test[0] - fold.train[-1] - 1 == 5


def test_expanding_folds_grow_and_rolling_folds_do_not():
    expanding = split.walk_forward(n=1500, n_folds=4, min_train=400, expanding=True)
    rolling = split.walk_forward(n=1500, n_folds=4, min_train=400, expanding=False)

    sizes = [f.train.size for f in expanding]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]
    assert all(f.train.size == 400 for f in rolling)
    assert all(f.train[0] == 0 for f in expanding)


def test_impossible_configuration_raises_instead_of_returning_nothing():
    """An empty fold list becomes "the Sharpe was nan" three calls later."""
    with pytest.raises(split.ConfigError) as err:
        split.walk_forward(n=310, n_folds=5, min_train=300, embargo=10)
    assert "folds" in str(err.value)


@pytest.mark.parametrize(
    "kw", [{"n_folds": 0}, {"horizon": 0}, {"embargo": -1}]
)
def test_nonsense_parameters_are_rejected(kw):
    with pytest.raises(split.ConfigError):
        split.walk_forward(n=1200, min_train=300, **kw)


def test_purged_window_removes_both_sides_of_a_holdout():
    """The single-holdout helper: purge before, embargo after.

    Both sides matter here because training data exists on both sides, which
    is the case the walk-forward split does not have.
    """
    n, start, stop = 1000, 400, 500
    train = split.purged_window(n, start, stop, horizon=5, embargo=20)
    assert train.max() >= stop + 20
    assert np.intersect1d(train, np.arange(start, stop)).size == 0
    assert train[train < start].max() == start - 5 - 1
    assert train[train >= stop].min() == stop + 20
