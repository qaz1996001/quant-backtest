"""Lookahead bias: the checker has to catch features that cheat.

A test suite that only verifies the good features pass proves nothing about
the checker. The four in ``qbt.leaky`` are real mistakes, each arriving by a
plausible route, and the checker has to find all four.
"""

from __future__ import annotations

import numpy as np
import pytest

from qbt import features
from qbt.leaky import LEAKY


# --------------------------------------------------------------------------
# the honest features
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(features.FEATURES))
def test_every_shipped_feature_is_causal(close, name):
    report = features.causality_report(features.FEATURES[name], close, name=name)
    assert report.causal, report.detail


def test_audit_passes_the_whole_feature_set(close):
    reports = features.audit(close)
    assert len(reports) == len(features.FEATURES)
    assert [r.name for r in reports if not r.causal] == []


# --------------------------------------------------------------------------
# the leaks, each one the checker must catch
#
# The leaky features live in ``qbt.leaky`` rather than in this file, so the
# example script can print the same four without importing the test suite.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(LEAKY))
def test_checker_catches_every_leak(close, name):
    report = features.causality_report(LEAKY[name], close, name=name)
    assert report.status == "leaks"
    assert report.first_leak is not None
    assert report.detail


def test_leak_distance_is_reported_and_matches_the_bug(close):
    """A one-bar leak reports a one-bar lookahead; a centred window reports ten.

    The number is the diagnostic. Knowing a feature leaks is not enough to
    find it; knowing it leaks exactly ten bars points straight at a 21-bar
    centred window.
    """
    cut = 800
    one = features.causality_report(LEAKY["next_bar_return"], close, cut=cut)
    assert one.lookahead == 1

    centred = features.causality_report(LEAKY["centred_window"], close, cut=cut)
    assert centred.lookahead == 10  # window 21 -> half = 10


def test_full_sample_zscore_leaks_from_the_very_first_bar(close):
    report = features.causality_report(LEAKY["full_sample_zscore"], close, cut=900)
    assert report.first_leak == 0


def test_the_label_is_allowed_to_see_the_future(close):
    """``forward_return`` is the one function that may look ahead.

    Asserted so the distinction is explicit in the suite: the label is a
    future value by construction, and the checker flags it, which is correct.
    """
    report = features.causality_report(features.forward_return, close)
    assert report.status == "leaks"


def test_design_matrix_has_no_nan_and_keeps_its_row_mapping(close):
    design = features.build_design(close)
    assert not np.isnan(design.X).any()
    assert not np.isnan(design.y).any()
    assert design.rows.size == design.X.shape[0]
    assert np.all(np.diff(design.rows) > 0)  # strictly increasing, no reorder
    assert design.names == tuple(features.FEATURES)


def test_warmup_rows_are_dropped_not_filled(close):
    """The first retained bar is at least as late as the longest window.

    ``zscore_60`` needs 60 bars, so no row before index 59 can be valid. If
    this ever passes with a smaller number, something started filling.
    """
    design = features.build_design(close)
    assert design.rows[0] >= 59


def test_label_horizon_is_trimmed_from_the_tail(close):
    """With horizon 5, the last 5 bars cannot be training rows."""
    design = features.build_design(close, horizon=5)
    assert design.rows[-1] <= close.size - 1 - 5
