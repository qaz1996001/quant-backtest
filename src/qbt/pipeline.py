"""Walk-forward training and the out-of-sample backtest, wired together.

The only function that knows about all five other modules. Kept small so
the wiring is readable, because the wiring is where the leaks are: every
one of the mistakes this repository tests for is a mistake about which rows
went where.

The out-of-sample signal is assembled fold by fold. A bar gets a prediction
only from a model that was fitted without it, and bars before the first
test fold get ``nan``, which the backtester reads as a flat position. The
alternative -- one model fitted on everything, predicting everything --
produces the Sharpe ratio that does not survive contact with a live account.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import backtest, features, metrics, split
from .model import Booster, Params


@dataclass(frozen=True)
class WalkForwardRun:
    """Result of a full walk-forward pass."""

    signal: np.ndarray  # per bar, nan where no out-of-sample model existed
    folds: list[split.Fold]
    fold_sharpe: list[float]
    design: features.Design
    backend: str
    importance: dict[str, float]
    covered: int  # bars that received an out-of-sample prediction


def walk_forward_signal(
    close,
    horizon: int = 1,
    n_folds: int = 5,
    min_train: int = 300,
    embargo: int = 10,
    params: Params | None = None,
    feature_set: dict[str, features.FeatureFn] | None = None,
    bars_per_year: float = metrics.DAILY,
) -> WalkForwardRun:
    """Fit one model per fold and stitch the test-fold predictions together."""
    c = np.asarray(close, dtype=float)
    design = features.build_design(c, feature_set, horizon=horizon)
    folds = split.walk_forward(
        n=design.rows.size,
        n_folds=n_folds,
        min_train=min_train,
        horizon=horizon,
        embargo=embargo,
    )

    signal = np.full(c.size, np.nan)
    fold_sharpe: list[float] = []
    importance: dict[str, float] = {}
    booster = Booster(params=params or Params(), names=design.names)

    for fold in folds:
        booster = Booster(params=params or Params(), names=design.names)
        booster.fit(design.X[fold.train], design.y[fold.train], design.names)
        pred = booster.predict(design.X[fold.test])

        bars = design.rows[fold.test]
        signal[bars] = pred

        # Per-fold score on the label, not on the equity curve. A fold with
        # five trades has no meaningful Sharpe; the label correlation does.
        realised = design.y[fold.test]
        fold_sharpe.append(
            metrics.sharpe(np.sign(pred) * realised, bars_per_year)
        )
        importance = booster.importance()

    return WalkForwardRun(
        signal=signal,
        folds=folds,
        fold_sharpe=fold_sharpe,
        design=design,
        backend=booster.backend,
        importance=importance,
        covered=int(np.sum(~np.isnan(signal))),
    )


def run(
    close,
    costs: backtest.Costs | None = None,
    threshold: float = 0.0,
    bars_per_year: float = metrics.DAILY,
    **kw,
) -> tuple[WalkForwardRun, backtest.Result]:
    """Walk forward, then backtest the stitched signal with costs applied."""
    wf = walk_forward_signal(close, bars_per_year=bars_per_year, **kw)
    result = backtest.run(
        close,
        wf.signal,
        costs=costs,
        threshold=threshold,
        bars_per_year=bars_per_year,
    )
    return wf, result


def in_sample_signal(
    close,
    horizon: int = 1,
    params: Params | None = None,
    feature_set: dict[str, features.FeatureFn] | None = None,
) -> np.ndarray:
    """Fit on everything, predict everything. The wrong way, kept runnable.

    Here so the example script can print both numbers side by side, and so
    ``tests/test_pipeline.py`` can assert the in-sample Sharpe is the higher
    of the two. The gap is the part of a backtest result that is an artefact
    of the procedure.
    """
    c = np.asarray(close, dtype=float)
    design = features.build_design(c, feature_set, horizon=horizon)
    booster = Booster(params=params or Params(), names=design.names)
    booster.fit(design.X, design.y, design.names)
    signal = np.full(c.size, np.nan)
    signal[design.rows] = booster.predict(design.X)
    return signal
