"""The gradient-boosting wrapper: guards at the door, and a named backend."""

from __future__ import annotations

import numpy as np
import pytest

from qbt import features
from qbt.model import Booster, Params, which_backend


def test_the_backend_is_named_not_guessed():
    """A fallback that is not reported is numbers that differ for no reason."""
    assert which_backend() in ("lightgbm", "sklearn-gbr")
    assert Booster().backend == which_backend()


def test_nan_input_is_rejected_with_the_offending_column_named(close):
    """LightGBM would train on this happily. Then the model knows warm-up.

    The error names the column, because "nan in X" on a six-column matrix is
    a message that sends you looking at all six.
    """
    design = features.build_design(close)
    X = design.X.copy()
    X[3, 2] = np.nan

    with pytest.raises(ValueError, match="nan in columns"):
        Booster(names=design.names).fit(X, design.y)


def test_nan_label_is_rejected(close):
    design = features.build_design(close)
    y = design.y.copy()
    y[-1] = np.nan
    with pytest.raises(ValueError, match="horizon was not trimmed"):
        Booster(names=design.names).fit(design.X, y)


def test_column_count_mismatch_is_rejected_at_predict(close):
    design = features.build_design(close)
    booster = Booster(names=design.names).fit(design.X, design.y, design.names)
    with pytest.raises(ValueError, match="columns"):
        booster.predict(design.X[:, :3])


def test_predict_before_fit_raises(close):
    with pytest.raises(RuntimeError, match="before fit"):
        Booster().predict(np.zeros((4, 6)))


def test_importance_is_keyed_by_name_and_sums_to_one(close):
    design = features.build_design(close)
    booster = Booster(names=design.names).fit(design.X, design.y, design.names)
    imp = booster.importance()

    assert set(imp) == set(design.names)
    assert sum(imp.values()) == pytest.approx(1.0)
    assert all(v >= 0.0 for v in imp.values())


def test_the_same_seed_gives_identical_predictions(close):
    design = features.build_design(close)
    a = Booster(Params(seed=7), design.names).fit(design.X, design.y, design.names)
    b = Booster(Params(seed=7), design.names).fit(design.X, design.y, design.names)
    assert np.array_equal(a.predict(design.X), b.predict(design.X))


def test_a_different_seed_gives_different_predictions(close):
    """Subsampling is on, so the seed has to matter. If it does not, the
    reproducibility test above is passing for the wrong reason."""
    design = features.build_design(close)
    a = Booster(Params(seed=7), design.names).fit(design.X, design.y, design.names)
    b = Booster(Params(seed=8), design.names).fit(design.X, design.y, design.names)
    assert not np.array_equal(a.predict(design.X), b.predict(design.X))


def test_the_model_can_learn_a_signal_that_is_actually_there(close):
    """A sanity floor. Train on a label the features determine exactly.

    Without this, every other model test could pass against a wrapper that
    returns a constant. The label here is one of the feature columns, so a
    working booster should fit it almost perfectly.
    """
    design = features.build_design(close)
    target = design.X[:, design.names.index("momentum_10")]
    booster = Booster(names=design.names).fit(design.X, target, design.names)
    pred = booster.predict(design.X)

    corr = float(np.corrcoef(pred, target)[0, 1])
    assert corr > 0.95
