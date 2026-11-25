"""Walk-forward (out-of-sample) evaluation.

Window layout
-------------
With ``train`` training bars, ``test`` test bars, ``embargo`` skipped bars and
a rolling step of ``test`` bars, fold ``k`` (0-based) is

    train = [k * test,              k * test + train)
    gap   = [train_stop,            train_stop + embargo)     (skipped)
    test  = [train_stop + embargo,  train_stop + embargo + test)

so train and test never overlap and at least ``embargo`` bars separate them
(purging the label horizon). The test windows are contiguous, which makes the
concatenated OOS equity curve a genuine "trade only unseen data" record.

Capital is compounded across folds: each fold starts with the equity the
previous fold finished with, so the OOS curve has no artificial jumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from quantflow.broker import BrokerConfig
from quantflow.data import BarSeries
from quantflow.engine import BacktestEngine, Strategy
from quantflow.metrics import Performance, performance

StrategyFactory = Callable[[], Strategy]


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold.

    Attributes:
        fold: Zero-based fold number.
        train_slice: Positional ``(start, stop)`` of the training window.
        test_slice: Positional ``(start, stop)`` of the test window.
        train_start / train_end: Training window boundaries (inclusive).
        test_start / test_end: Test window boundaries (inclusive).
        equity: Out-of-sample equity curve for the fold.
        performance: Metrics of ``equity``.
        n_trades: Number of round-trip trades in the fold.
        trade_pnls: Realized PnL of each round trip.
    """

    fold: int
    train_slice: tuple[int, int]
    test_slice: tuple[int, int]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    equity: pd.Series
    performance: Performance
    n_trades: int
    trade_pnls: tuple[float, ...]


@dataclass(frozen=True)
class WalkForwardResult:
    """Output of :func:`walk_forward`.

    Attributes:
        folds: Per-fold results in chronological order.
        oos_equity: Concatenated, capital-chained out-of-sample equity curve.
        oos_performance: Metrics of the full OOS curve.
        train: Training window length in bars.
        test: Test window length in bars.
        embargo: Bars skipped between train and test.
    """

    folds: list[Fold]
    oos_equity: pd.Series
    oos_performance: Performance
    train: int
    test: int
    embargo: int

    def to_dataframe(self) -> pd.DataFrame:
        """One tidy row per fold (boundaries + headline metrics)."""
        rows = []
        for fold in self.folds:
            perf = fold.performance
            rows.append(
                {
                    "fold": fold.fold,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "n_trades": fold.n_trades,
                    "total_return": perf.total_return,
                    "sharpe": perf.sharpe,
                    "max_drawdown": perf.max_drawdown,
                    "final_equity": float(fold.equity.iloc[-1]),
                }
            )
        return pd.DataFrame(rows)


def _check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def walk_forward(
    data: BarSeries,
    strategy_factory: StrategyFactory,
    n_splits: int = 5,
    train: int = 252,
    test: int = 63,
    embargo: int = 0,
    initial_cash: float = 100_000.0,
    broker_cfg: BrokerConfig | None = None,
    periods_per_year: int = 252,
    rf: float = 0.0,
) -> WalkForwardResult:
    """Run a rolling walk-forward evaluation.

    Args:
        data: Full history to split.
        strategy_factory: Zero-argument callable returning a fresh strategy.
            If the returned object has a ``fit(train_bars)`` method it is
            called with the training window before the test run.
        n_splits: Number of folds (>= 1).
        train: Training window length in bars (>= 2).
        test: Test window length in bars (>= 1); also the rolling step.
        embargo: Bars skipped between train and test (>= 0).
        initial_cash: Starting capital for the first fold.
        broker_cfg: Broker parameters passed to every fold.
        periods_per_year: Annualisation factor for the fold metrics.
        rf: Annual risk-free rate.

    Returns:
        A :class:`WalkForwardResult` with per-fold metrics and the chained
        out-of-sample equity curve.

    Raises:
        ValueError: On invalid window sizes, negative embargo, insufficient
            data, or if a fold loses all of its capital.
    """
    if not isinstance(data, BarSeries):
        raise TypeError("data must be a BarSeries")
    _check_positive_int("n_splits", n_splits)
    _check_positive_int("train", train)
    _check_positive_int("test", test)
    if train < 2:
        raise ValueError(f"train must be at least 2 bars, got {train}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    if not callable(strategy_factory):
        raise TypeError("strategy_factory must be callable")
    if initial_cash <= 0:
        raise ValueError(f"initial_cash must be positive, got {initial_cash}")

    required = (n_splits - 1) * test + train + embargo + test
    if len(data) < required:
        raise ValueError(
            f"need at least {required} bars for {n_splits} folds "
            f"(train={train}, test={test}, embargo={embargo}), got {len(data)}"
        )

    folds: list[Fold] = []
    curves: list[pd.Series] = []
    all_pnls: list[float] = []
    capital = float(initial_cash)

    for k in range(n_splits):
        train_lo = k * test
        train_hi = train_lo + train
        test_lo = train_hi + embargo
        test_hi = test_lo + test
        if test_hi > len(data):
            raise ValueError(f"fold {k} ends at bar {test_hi} beyond data length {len(data)}")

        train_bars = data.slice(train_lo, train_hi)
        test_bars = data.slice(test_lo, test_hi)

        strategy = strategy_factory()
        if hasattr(strategy, "fit") and callable(strategy.fit):
            strategy.fit(train_bars)

        engine = BacktestEngine(
            strategy=strategy,
            data=test_bars,
            broker_cfg=broker_cfg,
            initial_cash=capital,
            periods_per_year=periods_per_year,
            rf=rf,
        )
        result = engine.run()
        fold_equity = result.equity
        if float(fold_equity.iloc[-1]) <= 0:
            raise ValueError(f"fold {k} exhausted its capital ({fold_equity.iloc[-1]:.2f})")

        folds.append(
            Fold(
                fold=k,
                train_slice=(train_lo, train_hi),
                test_slice=(test_lo, test_hi),
                train_start=data.timestamps[train_lo],
                train_end=data.timestamps[train_hi - 1],
                test_start=data.timestamps[test_lo],
                test_end=data.timestamps[test_hi - 1],
                equity=fold_equity,
                performance=result.performance,
                n_trades=len(result.trade_pnls),
                trade_pnls=result.trade_pnls,
            )
        )
        curves.append(fold_equity)
        all_pnls.extend(result.trade_pnls)
        capital = float(fold_equity.iloc[-1])

    oos_equity = pd.concat(curves)
    oos_equity.name = "equity"
    oos_perf = performance(
        oos_equity,
        periods_per_year=periods_per_year,
        rf=rf,
        trade_pnls=all_pnls,
        strict=False,
    )

    return WalkForwardResult(
        folds=folds,
        oos_equity=oos_equity,
        oos_performance=oos_perf,
        train=train,
        test=test,
        embargo=embargo,
    )
