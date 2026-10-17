"""Walk-forward splits with purging and an embargo.

Three separate leaks get conflated under "use time-series cross-validation",
so they are separated here:

1. Training on bars that come after the test window. Fixed by walking
   forward and never looking back.
2. Training on a bar whose label overlaps the test window. A label with
   horizon ``h`` at bar ``t`` is a function of bars up to ``t + h``, so the
   last ``h`` training bars already contain test-period information. Fixed
   by *purging* them.
3. Training on a bar immediately after the test window. Serial correlation
   makes a bar adjacent to the test set nearly a copy of it. Fixed by an
   *embargo*: a dead zone after each test fold.

Only the first of those is addressed by sorting your data by date, which is
why the other two are what the tests in ``tests/test_split.py`` assert.

Indices here are row indices into a :class:`qbt.features.Design`, not bar
indices. The design matrix has already dropped warm-up rows, and splitting
on the wrong index space is its own class of off-by-one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One train/test pair."""

    index: int
    train: np.ndarray
    test: np.ndarray
    purged: int = 0
    embargoed: int = 0

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Fold({self.index}: train {self.train[0]}..{self.train[-1]} "
            f"n={self.train.size}, test {self.test[0]}..{self.test[-1]} "
            f"n={self.test.size}, purged={self.purged}, "
            f"embargoed={self.embargoed})"
        )


class ConfigError(ValueError):
    """The split parameters cannot produce a usable fold.

    Raised rather than returning an empty list. A silently empty fold list
    turns into "the model scored nan" three functions later, and the cause
    is then two hours away.
    """


def walk_forward(
    n: int,
    n_folds: int = 5,
    min_train: int = 200,
    horizon: int = 1,
    embargo: int = 10,
    expanding: bool = True,
) -> list[Fold]:
    """Produce ``n_folds`` chronological folds over ``n`` rows.

    ``expanding`` keeps every past row in the training set (anchored walk
    forward). Setting it to ``False`` gives a rolling window of ``min_train``
    rows, which is the right choice when the relationship is believed to
    decay, and the wrong one when data is scarce. Both are offered because
    the choice is a modelling assumption, not a default.

    The gap between the last training row and the first test row is
    ``horizon + embargo`` rows: ``horizon`` for the purge, which has an exact
    length, and ``embargo`` for the dead zone, which does not. Both are
    reported per fold and asserted in ``tests/test_split.py``.
    """
    if n_folds < 1:
        raise ConfigError("n_folds must be at least 1")
    if horizon < 1:
        raise ConfigError("horizon must be at least 1")
    if embargo < 0:
        raise ConfigError("embargo cannot be negative")

    gap = horizon + embargo
    usable = n - min_train - gap
    if usable < n_folds:
        raise ConfigError(
            f"{n} rows with min_train={min_train}, horizon={horizon}, "
            f"embargo={embargo} leaves {usable} rows for {n_folds} folds"
        )

    test_size = usable // n_folds
    folds: list[Fold] = []
    for k in range(n_folds):
        test_start = min_train + gap + k * test_size
        test_stop = test_start + test_size if k < n_folds - 1 else n
        if test_stop <= test_start:
            break

        train_stop = test_start - gap
        train_start = 0 if expanding else max(0, train_stop - min_train)

        train = np.arange(train_start, train_stop)
        test = np.arange(test_start, test_stop)
        if train.size == 0:
            raise ConfigError(f"fold {k} has an empty training set")

        folds.append(
            Fold(
                index=k, train=train, test=test, purged=horizon, embargoed=embargo
            )
        )
    return folds


def purged_window(
    n: int, test_start: int, test_stop: int, horizon: int = 1, embargo: int = 10
) -> np.ndarray:
    """Training rows for one test window, purged and embargoed on both sides.

    Used for a single hold-out rather than a walk. The embargo applies after
    the test window and the purge before it; the asymmetry is not an
    oversight. A row before the window leaks through its label horizon, which
    is a known length. A row after the window leaks through autocorrelation,
    which has no exact length, so the embargo is a parameter you pick.
    """
    left = np.arange(0, max(0, test_start - horizon))
    right = np.arange(min(n, test_stop + embargo), n)
    return np.concatenate([left, right])


def describe(folds: list[Fold]) -> str:
    """One ASCII line per fold. Printed by the example script."""
    lines = []
    for f in folds:
        lines.append(
            f"  fold {f.index}  train [{f.train[0]:>5}, {f.train[-1]:>5}]  "
            f"test [{f.test[0]:>5}, {f.test[-1]:>5}]  "
            f"gap {f.test[0] - f.train[-1] - 1}"
        )
    return "\n".join(lines)
