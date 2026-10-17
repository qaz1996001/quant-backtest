"""Gradient-boosting wrapper: LightGBM when present, sklearn when not.

LightGBM is the intended backend, driven through its native
``Dataset`` / ``train`` API rather than ``LGBMRegressor``. That is not a
style preference. ``lightgbm.sklearn`` raises ``LightGBMError`` on import
use if scikit-learn is absent, so the convenient wrapper would have dragged
scikit-learn into the install list of a repository that otherwise needs
numpy and lightgbm. The native API has no such dependency.

The fallback exists so the test suite still runs on a machine where the
wheel will not build, and it is reported rather than hidden --
``which_backend()`` is printed by the example script and asserted in
``tests/test_model.py``. A silent fallback would mean numbers that differ
between machines with no visible reason.

What the wrapper is actually for:

* one place where the column order is fixed, so fit and predict cannot
  disagree about which column is ``rsi_14``;
* ``nan``-free input checked at the door, because LightGBM will happily
  learn a split on a feature that is ``nan`` for the first sixty bars and
  the result is a model that depends on warm-up;
* a fixed seed and single-threaded determinism, so the reproducibility test
  means something.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # pragma: no cover - import path depends on the environment
    import lightgbm as _lgb

    _HAVE_LGB = True
except Exception:  # pragma: no cover
    _lgb = None
    _HAVE_LGB = False


def which_backend() -> str:
    """``"lightgbm"`` or ``"sklearn-gbr"``. Print this in any report."""
    return "lightgbm" if _HAVE_LGB else "sklearn-gbr"


@dataclass
class Params:
    """Hyperparameters, shallow on purpose.

    Financial labels have a signal-to-noise ratio where a deep tree is
    memorising the training window. ``num_leaves=15`` and ``min_child=40``
    are low enough that the model cannot carve out single days, which is
    what makes the walk-forward folds disagree in an informative way rather
    than all overfitting identically.
    """

    num_leaves: int = 15
    max_depth: int = 4
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample: float = 0.8
    reg_lambda: float = 1.0
    seed: int = 20260101


@dataclass
class Booster:
    """Fit/predict over a fixed, named feature set."""

    params: Params = field(default_factory=Params)
    names: tuple[str, ...] = ()
    _model: object | None = field(default=None, repr=False)

    @property
    def backend(self) -> str:
        return which_backend()

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def _check(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if np.isnan(X).any():
            bad = np.unique(np.flatnonzero(np.isnan(X).any(axis=0)))
            cols = [self.names[i] if i < len(self.names) else str(i) for i in bad]
            raise ValueError(
                f"X contains nan in columns {cols}; drop warm-up rows before fitting"
            )
        if self.names and X.shape[1] != len(self.names):
            raise ValueError(
                f"X has {X.shape[1]} columns, model expects {len(self.names)}"
            )
        return X

    def fit(self, X, y, names: tuple[str, ...] | None = None) -> "Booster":
        if names is not None:
            self.names = names
        X = self._check(X)
        y = np.asarray(y, dtype=float)
        if np.isnan(y).any():
            raise ValueError("y contains nan; the label horizon was not trimmed")
        p = self.params

        if _HAVE_LGB:
            names = list(self.names) if self.names else None
            dataset = _lgb.Dataset(X, label=y, feature_name=names, free_raw_data=False)
            self._model = _lgb.train(
                {
                    "objective": "regression",
                    "metric": "l2",
                    "num_leaves": p.num_leaves,
                    "max_depth": p.max_depth,
                    "learning_rate": p.learning_rate,
                    "min_data_in_leaf": p.min_child_samples,
                    "bagging_fraction": p.subsample,
                    "bagging_freq": 1,
                    "feature_fraction": p.colsample,
                    "lambda_l2": p.reg_lambda,
                    "seed": p.seed,
                    "bagging_seed": p.seed,
                    "feature_fraction_seed": p.seed,
                    "num_threads": 1,
                    "deterministic": True,
                    "force_row_wise": True,
                    "verbose": -1,
                },
                dataset,
                num_boost_round=p.n_estimators,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor

            self._model = GradientBoostingRegressor(
                learning_rate=p.learning_rate,
                n_estimators=p.n_estimators,
                max_depth=p.max_depth,
                min_samples_leaf=p.min_child_samples,
                subsample=p.subsample,
                random_state=p.seed,
            ).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict() before fit()")
        return np.asarray(self._model.predict(self._check(X)), dtype=float)

    def importance(self) -> dict[str, float]:
        """Feature importance by name, normalised to sum to 1.

        Returned as a dict rather than an array because an importance list
        whose order you have to remember is how a feature gets credited to
        its neighbour.
        """
        if self._model is None:
            raise RuntimeError("importance() before fit()")
        if hasattr(self._model, "feature_importance"):  # native lightgbm Booster
            raw = self._model.feature_importance(importance_type="gain")
        else:
            raw = getattr(self._model, "feature_importances_", None)
        if raw is None:  # pragma: no cover
            return {}
        raw = np.asarray(raw, dtype=float)
        total = raw.sum() or 1.0
        names = self.names or tuple(f"f{i}" for i in range(raw.size))
        return {n: float(v / total) for n, v in zip(names, raw)}
