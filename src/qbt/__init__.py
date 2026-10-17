"""qbt -- a backtest harness whose tests are the deliverable.

The modules, in the order data moves through them:

    synth     generate a reproducible price path with known regimes
    features  causal time-series features, plus a checker for leaks
    split     walk-forward folds with purging and an embargo
    model     LightGBM wrapper (sklearn fallback, reported not hidden)
    backtest  signal -> position -> equity, with costs and one bar of lag
    metrics   Sharpe, max drawdown, Calmar, win rate, turnover
    drift     sliding-window z-score checks for after deployment
    leaky     four features written to cheat, kept as negative controls
    pipeline  the wiring, kept small because the wiring is where leaks live
"""

from . import (
    backtest,
    drift,
    features,
    leaky,
    metrics,
    model,
    pipeline,
    split,
    synth,
)

__all__ = [
    "backtest",
    "drift",
    "features",
    "leaky",
    "metrics",
    "model",
    "pipeline",
    "split",
    "synth",
]
__version__ = "0.1.0"
