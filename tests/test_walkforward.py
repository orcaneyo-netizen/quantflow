"""Tests for walk-forward splitting: boundaries, embargo, no overlap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantflow.data import synthetic_bars
from quantflow.strategies import BuyAndHold, SmaCross
from quantflow.walkforward import walk_forward


def test_fold_boundaries_and_no_overlap():
    bars = synthetic_bars(n=120, seed=3, freq="1D")
    result = walk_forward(
        data=bars,
        strategy_factory=lambda: BuyAndHold(weight=0.5),
        n_splits=3,
        train=20,
        test=10,
        embargo=2,
        periods_per_year=252,
    )

    assert len(result.folds) == 3
    for k, fold in enumerate(result.folds):
        train_lo, train_hi = fold.train_slice
        test_lo, test_hi = fold.test_slice

        assert train_lo == k * 10
        assert train_hi - train_lo == 20
        assert test_lo == train_hi + 2  # embargo
        assert test_hi - test_lo == 10
        # no overlap, and at least `embargo` bars in between
        assert test_lo > train_hi - 1
        assert test_lo - (train_hi - 1) - 1 == 2

        # boundaries line up with the bar timestamps
        assert fold.train_start == bars.timestamps[train_lo]
        assert fold.train_end == bars.timestamps[train_hi - 1]
        assert fold.test_start == bars.timestamps[test_lo]
        assert fold.test_end == bars.timestamps[test_hi - 1]
        assert len(fold.equity) == 10

    # the last test window fits inside the data
    assert result.folds[-1].test_slice[1] <= len(bars)


def test_zero_embargo_windows_are_contiguous():
    bars = synthetic_bars(n=90, seed=4, freq="1D")
    result = walk_forward(
        data=bars,
        strategy_factory=lambda: BuyAndHold(weight=0.5),
        n_splits=3,
        train=20,
        test=10,
        embargo=0,
    )
    for prev, nxt in zip(result.folds, result.folds[1:]):
        assert nxt.test_slice[0] == prev.test_slice[1]


def test_oos_curve_is_chained_and_sized():
    bars = synthetic_bars(n=150, seed=5, freq="1D")
    result = walk_forward(
        data=bars,
        strategy_factory=lambda: BuyAndHold(weight=0.5),
        n_splits=4,
        train=25,
        test=15,
        embargo=1,
        initial_cash=50_000.0,
    )

    assert len(result.oos_equity) == 4 * 15
    assert result.oos_equity.index.is_monotonic_increasing
    # each fold starts with the previous fold's ending equity (compounding)
    assert result.oos_equity.iloc[0] == pytest.approx(50_000.0)
    for prev, nxt in zip(result.folds, result.folds[1:]):
        assert nxt.equity.iloc[0] == pytest.approx(prev.equity.iloc[-1])

    assert np.isfinite(result.oos_equity.to_numpy()).all()
    assert (result.oos_equity > 0).all()


def test_to_dataframe_shape():
    bars = synthetic_bars(n=120, seed=6, freq="1D")
    result = walk_forward(
        data=bars,
        strategy_factory=lambda: SmaCross(5, 15),
        n_splits=2,
        train=30,
        test=20,
    )
    df = result.to_dataframe()
    assert len(df) == 2
    assert list(df.columns) == [
        "fold",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "n_trades",
        "total_return",
        "sharpe",
        "max_drawdown",
        "final_equity",
    ]
    assert df["final_equity"].iloc[-1] == pytest.approx(result.oos_equity.iloc[-1])


def test_negative_embargo_rejected():
    bars = synthetic_bars(n=100, seed=7)
    with pytest.raises(ValueError, match="embargo"):
        walk_forward(bars, lambda: BuyAndHold(), n_splits=2, train=20, test=10, embargo=-1)


def test_insufficient_data_rejected():
    bars = synthetic_bars(n=50, seed=8)
    # needs 2*10 + 20 + 2 + 10 = 52 bars
    with pytest.raises(ValueError, match="need at least 52 bars"):
        walk_forward(bars, lambda: BuyAndHold(), n_splits=3, train=20, test=10, embargo=2)


def test_invalid_window_parameters():
    bars = synthetic_bars(n=200, seed=9)
    with pytest.raises(ValueError):
        walk_forward(bars, lambda: BuyAndHold(), n_splits=0)
    with pytest.raises(ValueError):
        walk_forward(bars, lambda: BuyAndHold(), n_splits=2, train=1)
    with pytest.raises(ValueError):
        walk_forward(bars, lambda: BuyAndHold(), n_splits=2, train=20, test=0)
    with pytest.raises(TypeError):
        walk_forward(bars, "not callable", n_splits=2)  # type: ignore[arg-type]


def test_fit_hook_is_called_with_training_window():
    calls = []

    class Fitted:
        def fit(self, train_bars):
            calls.append(len(train_bars))

        def on_bar(self, ctx):
            return []

    bars = synthetic_bars(n=120, seed=10)
    walk_forward(bars, Fitted, n_splits=2, train=30, test=20)
    assert calls == [30, 30]


def test_oos_performance_matches_curve():
    bars = synthetic_bars(n=140, seed=11, freq="1D")
    result = walk_forward(
        bars, lambda: SmaCross(5, 20), n_splits=2, train=40, test=25, periods_per_year=252
    )
    expected = result.oos_equity.iloc[-1] / result.oos_equity.iloc[0] - 1.0
    assert result.oos_performance.total_return == pytest.approx(expected)
    assert result.oos_performance.n_periods == len(result.oos_equity) - 1
