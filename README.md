*English ｜ [繁體中文](README.zh-TW.md)*

# qbt — the part of a backtest that goes wrong is the measurement

A runnable harness for one claim: **a backtest result is mostly a statement
about your evaluation procedure, so the procedure is what needs tests.**
Lookahead bias, fold geometry, transaction costs and degenerate metrics are
each pinned down by an assertion that fails if the mechanism is removed.

```bash
uv run --with pytest --with lightgbm --with numpy pytest -q      # 110 passed
uv run --with lightgbm --with numpy python examples/run_backtest.py
```

Offline, synthetic data, fixed seeds. No market data feed, no API key, no
cache directory.

The model is LightGBM. It is the smallest part of the repository, which is
roughly the proportion it deserves.

---

## The number this is about

Same data, same features, same model, two evaluation procedures:

```
fit once on everything   Sharpe   6.744   <- the artefact
walk forward, 5 folds    Sharpe   1.592   <- the number
```

The 6.744 is what a notebook prints when `fit` is called once and
`predict` is called on the same rows. Nothing in that code is buggy. The
procedure is.

And the honest 1.592 is still gross. At a plausible 7 bps round trip it is
**1.208**; at a retail 25 bps it is **0.214**; at 70 bps the strategy
**loses money**. Those four rows come out of `examples/run_backtest.py`.

---

## Four places a backtest leaks, and the test for each

### 1. A feature that reads the future

The checker does not inspect source code and does not trust docstrings. It
computes the feature, replaces every bar after a cut point with a different
path, recomputes, and looks for any index at or below the cut whose value
moved.

```
shipped features
  ret_1          causal
  ret_5          causal
  vol_20         causal
  momentum_10    causal
  rsi_14         causal
  zscore_60      causal

features written to cheat
  next_bar_return      leaks   first moved index   20   (probe cut 20)
  centred_window       leaks   first moved index   11   (probe cut 20)
  full_sample_zscore   leaks   first moved index    0   (probe cut 20)
  backfilled_warmup    leaks   first moved index    0   (probe cut 20)
```

`qbt/leaky.py` ships four features written to cheat, and the suite asserts
all four get caught. A checker only ever run against correct code has never
been tested.

Each of the four has a plausible route in. Nobody types these
deliberately:

| leak | how it gets in |
|---|---|
| `next_bar_return` | the label formula reused as a feature |
| `centred_window` | a smoothing step borrowed from signal processing, where centring is correct |
| `full_sample_zscore` | `StandardScaler` fitted before the split |
| `backfilled_warmup` | back-filling `nan` to fix a gap at the left edge of a chart |

The reported index is the diagnostic. `next_bar_return` moves one bar behind
the cut, so it looks exactly one bar ahead. The two standardisation bugs
move index **0** — a different shape of mistake, where the whole column was
scaled using numbers from the end of the series.

### 2. Folds that touch

```
fold 0  train [    0,   299]  test [  311,   535]  gap 11
fold 1  train [    0,   524]  test [  536,   760]  gap 11
fold 2  train [    0,   749]  test [  761,   985]  gap 11
```

The gap is **11 bars**: 1 for the label horizon, 10 for the embargo.

Three separate leaks get bundled together under "use time-series
cross-validation", so `split.py` separates them:

- Training after the test window. Handled by walking forward.
- Training on a bar whose label overlaps the test window. A label with
  horizon `h` at bar `t` is a function of bars up to `t + h`, so the last
  `h` training bars already contain test-period information. That is the
  **purge**, and its length is exact.
- Training on a bar adjacent to the test window. Serial correlation makes
  the neighbouring bar nearly a copy. That is the **embargo**, and its
  length is a parameter you choose rather than derive.

Only the first is solved by sorting your data by date.

The tests assert the measurable consequence: `gap == horizon + embargo` for
every fold, a larger embargo produces a larger gap, and `embargo=0` still
purges the horizon. A parameter that is stored and never applied is the
failure mode worth testing for.

### 3. Costs that are nominally configurable and actually zero

`Costs()` defaults to 2 bps commission and 5 bps slippage. A zero default is
the one that gets left in by accident.

The assertion that matters: the same signal under zero cost and real cost
has to produce **different** Sharpe ratios, and the measured cost drag has
to be positive. Then one further test shows why it matters — a signal that
is right on 58 percent of bars is **profitable gross and losing net**, with
roughly 50 bps of friction against 13 bps of edge.

### 4. Metrics that always return a flattering number

| strategy | what the metrics must say |
|---|---|
| all cash, never trades | max drawdown **0**, Sharpe **nan**, Calmar **nan**, win rate **nan** |
| inverted signal, wrong every bar | Sharpe **-16.769**, total return **-1.0**, max drawdown **1.0** |
| buy and hold | Sharpe **0.800**, max drawdown **0.584** |

The all-cash row is the one that needs a test. A flat equity curve has no
risk-adjusted return; printing `0.00` would be a claim about a risk nobody
took. `nan` makes whatever reads it notice.

Annualisation is a parameter everywhere. `252` appears only as the value of
`BARS_PER_YEAR["daily"]`, never as a literal inside a formula, and the
`bars_per_year` actually used is carried in the report. A Sharpe ratio
computed on hourly bars and read as daily is wrong by **2.5x**.

---

## Where drift detection comes in

A backtest says a strategy worked on a stretch of history. It says nothing
about whether today's inputs still resemble that stretch. `drift.py` runs a
sliding-window z-score over features and over rolling performance.

The data generator plants a volatility regime change at bar **600**, so the
detector can be checked against ground truth:

```
ret_1            ok                 z=  -0.05
vol_20           drift              z=   5.81  |z| over 3.0
momentum_10      ok                 z=  -0.10
```

Two refusals are tested as carefully as the detection:

```
vol_20           insufficient_data  z=    n/a  baseline has 5 points, need 30
stuck_sensor     insufficient_data  z=    n/a  baseline has zero variance
```

A ten-point window and a perfectly flat baseline both make a z-score
undefined. Reporting `ok` there is how a monitor shows green for a week on
an empty baseline table. A flat baseline is also exactly what a stuck feed
or a repeated cached value looks like.

Feature drift is **two-sided**: an input three sigma below its training
range is as unfamiliar to the model as three sigma above. Performance drift
is **one-sided**, after `higher_is_better` normalises the direction once, so
a negative z always reads as worse for both Sharpe and drawdown. Two
comparison branches would be two chances to have the inequality backwards,
and a monitor that is backwards on drawdown stays quiet exactly when it
should not.

---

## What the tests cover

**110 tests.** The modules total about 1,500 lines and the tests about
1,100, which is the ratio the repository is arguing for.

| file | tests | what it pins down |
|---|---|---|
| `test_lookahead.py` | 17 | every shipped feature is causal; all four leaks are caught; leak distance matches the bug; warm-up rows are dropped rather than filled |
| `test_metrics.py` | 17 | all-cash gives 0 drawdown and nan Sharpe; an inverted signal gives the mirror-image Sharpe; annualisation is a parameter; a constant return series has no Sharpe |
| `test_drift.py` | 16 | short and flat baselines report `insufficient_data`; the planted regime change is found at bar 600; an improvement is not drift |
| `test_split.py` | 16 | folds never share a row; test folds are contiguous and cover the tail; the gap equals purge plus embargo; impossible configurations raise |
| `test_pipeline.py` | 16 | no bar is predicted by a model trained on it; a planted AR(1) is found; **with no signal planted, none is found**; one leaky feature takes Sharpe to 13.0 |
| `test_backtest.py` | 13 | the position is held one bar after the signal; a flip pays cost twice; a weak edge is eaten entirely by 50 bps |
| `test_model.py` | 9 | `nan` input is rejected with the column named; the same seed gives identical predictions and a different seed does not |
| `test_reproducibility.py` | 6 | two identical pipeline runs give byte-identical net returns and an identical report |

Two of those are controls:

**The negative control.** At `autocorr=0.0` the generated returns are
independent and nothing is predictable. The walk-forward Sharpe has to come
back small. A leak anywhere in the pipeline — a feature reading ahead, a
fold overlapping, a label off by one — produces a confident number on noise,
which makes this the cheapest regression test in the suite.

**The positive control.** Swap one honest feature for `next_bar_return` and
run the identical procedure: correct folds, correct purge, correct embargo,
costs charged. Sharpe goes from **1.592 to 13.008** and max drawdown from
**0.201 to 0.002**. Split
discipline does not protect you from a feature that reads ahead. That is
why the causality checker is a separate gate with its own tests.

---

## Running it

```bash
# tests
uv run --with pytest --with lightgbm --with numpy pytest -q

# the whole example, or one section
uv run --with lightgbm --with numpy python examples/run_backtest.py
uv run --with lightgbm --with numpy python examples/run_backtest.py leak costs
```

Sections: `audit`, `folds`, `leak`, `costs`, `null`, `drift`. Output is
plain ASCII, because the terminal this was written on is CP950 and a box
drawing character in a report is a crashed run.

**Backend.** LightGBM is the intended one, driven through its native
`Dataset` / `train` API. `LGBMRegressor` was the first version and it raises
`LightGBMError` when scikit-learn is absent, which would have added
scikit-learn to the install list of a repository that otherwise needs numpy
and lightgbm. The native API has no such dependency.

If no lightgbm wheel is available the wrapper falls back to
`sklearn.ensemble.GradientBoostingRegressor` and says so —
`which_backend()` is printed at the top of every example run and asserted in
the tests. **All 110 tests pass on both backends.** A silent fallback would
mean numbers that differ between two machines with nothing on screen to
explain why.

---

## Things that went wrong while writing this

Kept in because they are the parts that were actually learned here.

**One probe cut is not enough.** The causality checker originally split the
series at the midpoint. Three of the four leaky features were caught.
`backfilled_warmup` passed clean, because by bar 750 there is no warm-up
left to back-fill — it corrupts the first sixty bars only. `default_cuts()`
now probes six cuts including an early one at bar 20.

**`sd == 0.0` is the wrong guard for a degenerate Sharpe ratio.** A test
asserting "a constant return series has no Sharpe" failed. A series of 500
identical 1 bp returns has a sample standard deviation near **1e-20**,
because the computed mean does not come back exactly equal to the input. Not
zero, so the guard let it through, and the ratio came out around **1e16**
annualised. The fix is a relative floor, `ZERO_DISPERSION = 1e-12`.

**A perfect oracle does not demonstrate that costs matter.** The first
version of the cost test used a signal that knew the next bar exactly, and
asserted it would be unprofitable at 50 bps. It failed: the average absolute
bar return on this series is about **80 bps**, so an oracle stays hugely
profitable through any plausible friction. The test now uses a 58 percent
hit rate, which is where the cost actually bites, and the premise correction
is the lesson.

**The perturbation has to keep prices positive.** Perturbing with
`close * (1 + 0.5 * randn)` drove some bars negative, every log-return
feature returned `nan`, and the checker reported leaks that were artefacts
of being fed an impossible series. It is `close * exp(0.5 * randn)` now.

**Raising the planted signal was a judgement call.** The AR(1) coefficient
started at 0.06, which is closer to something real, and the walk-forward
Sharpe came out at **-0.605** — honest, and useless as a demonstration,
since a losing strategy makes the cost ladder and the drift section hard to
read. It is 0.25 now, which is **far stronger than any liquid market**, and
`synth.py` says so in the docstring. The negative control at `autocorr=0.0`
exists to keep that choice from turning into a claim.

---

## Limitations

- **Synthetic data is not a market.** The generator is a regime-switching
  random walk with an AR(1) term. It has no fat tails worth the name, no
  volatility clustering beyond the regime schedule, no jumps, no overnight
  gaps, no earnings dates. A strategy working here implies nothing about one
  working anywhere else.
- **The planted AR(1) of 0.25 is unrealistically strong.** Daily equity
  return autocorrelation is a couple of percent at best. The number is set
  high so the harness has something to find; it is not an estimate of
  anything.
- **One asset, one position, no portfolio.** No cross-sectional ranking, no
  correlation between holdings, no position sizing by volatility target, no
  leverage constraint, no margin.
- **Costs are a linear function of turnover.** Real slippage depends on
  order size against book depth, time of day, and whether you are the one
  moving the price. Market impact is non-linear and is not modelled.
- **No microstructure at all.** No bid-ask spread as a separate quantity, no
  partial fills, no queue position, no exchange fee schedule, no borrow cost
  or availability for the short side. Positions are assumed to execute at
  the close of the following bar.
- **No multiple-testing correction.** The walk-forward procedure gives an
  honest estimate for *one* strategy specification. Trying forty and
  reporting the best needs deflated Sharpe or a reality check, and neither is
  here.
- **The label is a one-bar forward return.** Triple-barrier labelling,
  meta-labelling and sample weighting by uniqueness all belong in this
  problem and none are implemented.
- **Drift detection assumes an approximately normal baseline.** A z-score on
  a fat-tailed feature over-fires. A rank or quantile-based statistic would
  be the right upgrade.

---

## Where this came from

The time-series feature engineering, the sliding-window z-score drift
detection, the evaluation gate and the discipline of testing the measurement
rather than the model all come from production ML work I did on industrial
sensor data and on an agent platform. That code is private.

**No code from any previous employer is used here.** This repository was
written from nothing, against synthetic data, in a domain I do not have
proprietary work in. What carries over is the reasoning, written up in
[docs/design.md](docs/design.md).

Two companion repositories apply the same approach to different problems:
`idemingest` (idempotent replay and failure classification) and
`agent-control-plane` (a deterministic control plane for agent pipelines,
which is where the drift module's shape comes from).

MIT licensed.
