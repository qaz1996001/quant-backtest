"""Run the whole thing and print what actually happened.

    python examples/run_backtest.py           # every section
    python examples/run_backtest.py leak      # just one

Sections
--------
audit     the causality checker over the shipped features, and over four
          deliberately leaky ones
folds     the walk-forward geometry, with the purge and embargo visible
leak      in-sample Sharpe next to walk-forward Sharpe, same data
costs     the same signal at four cost levels
null      the strategies whose metrics must be boring: cash, and an inverse
drift     a regime change found where it was planted

Output is plain ASCII. The terminal this was written on is CP950 and a box
drawing character in a report is a crashed run.
"""

from __future__ import annotations

import sys

import numpy as np

from qbt import backtest, drift, features, metrics, pipeline, split, synth
from qbt.leaky import LEAKY
from qbt.model import which_backend

SEED = 20260101
W = 74


def head(title: str) -> None:
    print(f"\n{'-' * W}\n{title}\n{'-' * W}")


def series():
    return synth.generate(seed=SEED)


# --------------------------------------------------------------------------


def section_audit() -> None:
    head("audit -- perturb the future, see if the past moves")
    close = series().close

    print("  shipped features")
    for report in features.audit(close):
        print(f"    {report.name:<14} {report.status}")

    print("\n  features written to cheat")
    for name, fn in LEAKY.items():
        r = features.causality_report(fn, close, name=name)
        print(
            f"    {name:<20} {r.status:<7} first moved index {r.first_leak:>4}"
            f"   (probe cut {r.cut})"
        )
    print("\n  The moved index is the diagnostic. next_bar_return moves one")
    print("  bar behind the cut, so it is looking exactly one bar ahead. The")
    print("  two standardisation bugs move index 0, a different shape of")
    print("  mistake: the whole column was scaled with numbers from the end.")


def section_folds() -> None:
    head("folds -- walk forward, purged, embargoed")
    close = series().close
    design = features.build_design(close)
    folds = split.walk_forward(
        n=design.rows.size, n_folds=5, min_train=300, horizon=1, embargo=10
    )

    print(f"  {len(close)} bars -> {design.rows.size} usable rows after warm-up")
    print(f"  purge 1 bar (label horizon) + embargo 10 bars = gap of 11\n")
    print(split.describe(folds))
    print("\n  the gap column is the embargo. With embargo=0 it would read 1.")


def section_leak() -> None:
    head("leak -- the same model, scored two ways")
    close = series().close

    in_sample = pipeline.in_sample_signal(close)
    wf, oos = pipeline.run(close, n_folds=5, min_train=300)

    a = backtest.run(close, in_sample, costs=backtest.FREE).report
    b = backtest.run(close, wf.signal, costs=backtest.FREE).report

    print(f"  backend              {which_backend()}")
    print(f"  bars with a forecast {wf.covered} of {len(close)}\n")
    print(f"  fit once on everything   Sharpe {a.sharpe:>7.3f}   <- the artefact")
    print(f"  walk forward, 5 folds    Sharpe {b.sharpe:>7.3f}   <- the number")
    print(f"  per-fold Sharpe          {[round(s, 2) for s in wf.fold_sharpe]}")
    print("\n  feature importance (gain, normalised)")
    for name, value in sorted(wf.importance.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(value * 40))
        print(f"    {name:<14} {value:>6.3f}  {bar}")


def section_costs() -> None:
    head("costs -- the same signal at four cost levels")
    close = series().close
    wf = pipeline.walk_forward_signal(close, n_folds=5, min_train=300)

    levels = [
        ("free (0+0 bps)", backtest.FREE),
        ("default (2+5)", backtest.Costs(2.0, 5.0)),
        ("retail (10+15)", backtest.Costs(10.0, 15.0)),
        ("punishing (30+40)", backtest.Costs(30.0, 40.0)),
    ]
    print(f"  {'costs':<20}{'Sharpe':>9}{'total ret':>11}{'max DD':>9}{'drag':>9}")
    for label, costs in levels:
        r = backtest.run(close, wf.signal, costs=costs).report
        print(
            f"  {label:<20}{r.sharpe:>9.3f}{r.total_return:>11.4f}"
            f"{r.max_drawdown:>9.4f}{r.cost_drag:>9.4f}"
        )
    print("\n  cost is a parameter of the result, not a detail of the fill")


def section_null() -> None:
    head("null -- strategies whose metrics have to be boring")
    close = series().close

    cash = backtest.all_cash(close).report
    print("  all cash, never trades")
    for line in cash.lines():
        print("  " + line)
    print("    Sharpe is nan because there were no returns to adjust for risk.")
    print("    Reporting 0.00 would be a claim about a risk nobody took.")

    rets = backtest.simple_returns(close)
    inverse = np.zeros(close.size)
    inverse[:-1] = -rets[1:]
    bad = backtest.run(close, inverse, costs=backtest.FREE).report
    print("\n  deliberately inverted signal, wrong on every bar")
    print(f"    Sharpe       {bad.sharpe:>9.3f}")
    print(f"    total return {bad.total_return:>9.4f}")
    print(f"    max drawdown {bad.max_drawdown:>9.4f}")
    print("    A metric that never prints a bad number is not measuring.")

    bh = backtest.buy_and_hold(close).report
    print(f"\n  buy and hold benchmark   Sharpe {bh.sharpe:>7.3f}  "
          f"MDD {bh.max_drawdown:>6.3f}")


def section_drift() -> None:
    head("drift -- a regime change found where it was planted")
    s = series()
    edge = s.regime_bounds()[0]
    print(f"  the generator triples volatility at bar {edge}, by construction\n")

    design = features.build_design(s.close)
    at = int(np.searchsorted(design.rows, edge))
    for verdict in drift.scan_features(design.X, design.names, split_at=at):
        print(verdict.line())

    print("\n  the same check with a window too short to support a z-score")
    print(drift.feature_drift("vol_20", [0.01] * 5, 0.9).line())
    print(drift.feature_drift("stuck_sensor", [0.5] * 200, 0.9).line())
    print("\n  insufficient_data is an answer. Reporting ok there is how a")
    print("  monitor shows green for a week on an empty baseline table.")


SECTIONS = {
    "audit": section_audit,
    "folds": section_folds,
    "leak": section_leak,
    "costs": section_costs,
    "null": section_null,
    "drift": section_drift,
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(SECTIONS)
    unknown = [w for w in wanted if w not in SECTIONS]
    if unknown:
        print(f"unknown section(s): {unknown}; pick from {list(SECTIONS)}")
        return 2
    print(f"qbt example run -- seed {SEED}, backend {which_backend()}")
    print(f"annualisation: {metrics.DAILY:.0f} bars per year (daily bars)")
    for name in wanted:
        SECTIONS[name]()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
