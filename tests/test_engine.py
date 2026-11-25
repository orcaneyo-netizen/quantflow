"""Tests for the engine: event ordering, fills, equity consistency."""

from __future__ import annotations

import numpy as np
import pytest

from quantflow.broker import BrokerConfig
from quantflow.costs import CostConfig, FixedSlippage, PerShareCommission
from quantflow.engine import BAR_COLUMNS, TRADE_COLUMNS, BacktestEngine, Order
from quantflow.events import Direction, FillEvent, MarketEvent, OrderEvent, OrderType
from quantflow.strategies import BuyAndHold, DonchianBreakout, SmaCross


class BuyFixed:
    """Buy a fixed quantity on the first bar, then hold."""

    def __init__(self, qty: float) -> None:
        self.qty = qty

    def on_bar(self, ctx):
        if ctx.index == 0:
            return [ctx.order_market(self.qty, tag="entry")]
        return []


class DoNothing:
    def on_bar(self, ctx):
        return []


def free_broker(max_participation: float = 1.0) -> BrokerConfig:
    return BrokerConfig(
        costs=CostConfig(
            commission=PerShareCommission(0.0, 0.0, 0.0),
            slippage=FixedSlippage(0.0),
        ),
        max_participation=max_participation,
    )


def test_buy_and_hold_monotone_growth(make_bars):
    # open == close so bar 1 opens exactly at 101
    bars = make_bars(np.arange(100.0, 111.0), opens=np.arange(100.0, 111.0))
    engine = BacktestEngine(
        BuyFixed(100), bars, broker_cfg=free_broker(), initial_cash=100_000.0
    )
    result = engine.run()

    # fills on bar 1 at its open (101), marked at the final close (110)
    assert len(result.trades) == 1
    fill = result.trades.iloc[0]
    assert fill["action"] == "BUY"
    assert fill["qty"] == pytest.approx(100.0)
    assert fill["price"] == pytest.approx(101.0)
    assert fill["fee"] == pytest.approx(0.0)

    expected = 100_000.0 + 100 * (110.0 - 101.0)
    assert result.equity.iloc[0] == pytest.approx(100_000.0)
    assert result.equity.iloc[-1] == pytest.approx(expected)
    assert result.final_cash == pytest.approx(100_000.0 - 100 * 101.0)
    assert result.final_position == pytest.approx(100.0)
    assert result.performance.total_return == pytest.approx(expected / 100_000.0 - 1.0)


def test_fees_reduce_equity_exactly(make_bars):
    bars = make_bars(np.arange(100.0, 111.0), opens=np.arange(100.0, 111.0))
    cfg = BrokerConfig(
        costs=CostConfig(
            commission=PerShareCommission(1.0, 0.0, 0.0),  # $1 per share
            slippage=FixedSlippage(0.0),
        )
    )
    result = BacktestEngine(BuyFixed(100), bars, broker_cfg=cfg, initial_cash=100_000.0).run()
    expected = 100_000.0 + 100 * (110.0 - 101.0) - 100.0
    assert result.equity.iloc[-1] == pytest.approx(expected)
    assert result.trades.iloc[0]["fee"] == pytest.approx(100.0)


def test_slippage_is_charged_both_ways(make_bars):
    bars = make_bars(np.arange(100.0, 111.0), opens=np.arange(100.0, 111.0))
    cfg = BrokerConfig(
        costs=CostConfig(
            commission=PerShareCommission(0.0, 0.0, 0.0),
            slippage=FixedSlippage(bps=100.0),  # 1%
        )
    )
    result = BacktestEngine(BuyFixed(10), bars, broker_cfg=cfg, initial_cash=100_000.0).run()
    fill = result.trades.iloc[0]
    assert fill["price"] == pytest.approx(101.0 * 1.01)
    assert fill["slippage"] == pytest.approx(10 * 101.0 * 0.01)


def test_event_log_is_time_ordered(make_bars):
    bars = make_bars(np.arange(100.0, 120.0))
    engine = BacktestEngine(BuyFixed(10), bars, broker_cfg=free_broker())
    result = engine.run()

    timestamps = [e.timestamp for e in result.events]
    assert timestamps == sorted(timestamps)
    seqs = [e.seq for e in result.events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)

    types = {type(e) for e in result.events}
    assert {MarketEvent, OrderEvent, FillEvent} <= types

    df = result.events.to_dataframe()
    assert list(df["type"].unique())[0] == "MarketEvent"
    assert len(df) == len(result.events)


def test_no_lookahead_first_fill_on_second_bar(make_bars):
    bars = make_bars([100.0, 101.0, 102.0, 103.0], opens=[100.0, 101.0, 102.0, 103.0])
    result = BacktestEngine(BuyFixed(1), bars, broker_cfg=free_broker()).run()

    fill = result.trades.iloc[0]
    assert fill["timestamp"] == bars.timestamps[1]
    assert fill["price"] == pytest.approx(101.0)
    assert fill["price"] == pytest.approx(bars.data["open"].iloc[1])
    # bar 0 can never contain a fill
    assert result.equity.iloc[0] == pytest.approx(100_000.0)


def test_results_frames_have_expected_columns(make_bars):
    bars = make_bars(np.arange(100.0, 112.0))
    engine = BacktestEngine(BuyFixed(10), bars, broker_cfg=free_broker())
    result = engine.run()

    assert list(result.trades.columns) == TRADE_COLUMNS
    assert list(result.bars.columns) == BAR_COLUMNS
    assert len(result.bars) == len(bars)
    assert engine.results().equals(result.trades)
    assert engine.equity_curve().equals(result.equity)
    np.testing.assert_allclose(result.bars["equity"].to_numpy(), result.equity.to_numpy())


def test_unfilled_order_at_end_is_reported(make_bars):
    class LateBuyer:
        def on_bar(self, ctx):
            if ctx.index == 2:  # last of three bars
                return [ctx.order_market(1)]
            return []

    bars = make_bars([100.0, 101.0, 102.0])
    result = BacktestEngine(LateBuyer(), bars, broker_cfg=free_broker()).run()
    assert len(result.unfilled_orders) == 1
    assert result.trades.empty


def test_flat_backtest_metrics_are_zero(make_bars):
    bars = make_bars([100.0, 101.0, 100.0, 102.0, 101.0])
    result = BacktestEngine(DoNothing(), bars, broker_cfg=free_broker()).run()

    assert result.trades.empty
    assert (result.equity == 100_000.0).all()
    assert result.performance.total_return == 0.0
    assert result.performance.sharpe == 0.0
    assert result.performance.max_drawdown == 0.0
    assert result.performance.exposure == 0.0


def test_engine_input_validation(make_bars):
    with pytest.raises(ValueError, match="at least 3 bars"):
        BacktestEngine(DoNothing(), make_bars([100.0, 101.0]))
    with pytest.raises(ValueError, match="initial_cash"):
        BacktestEngine(DoNothing(), make_bars([100.0, 101.0, 102.0]), initial_cash=0.0)
    with pytest.raises(TypeError):
        BacktestEngine(DoNothing(), [100.0, 101.0, 102.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        BacktestEngine(object(), make_bars([100.0, 101.0, 102.0]))


def test_strategy_must_return_orders(make_bars):
    class Bad:
        def on_bar(self, ctx):
            return ["not an order"]

    with pytest.raises(TypeError, match="Order objects"):
        BacktestEngine(Bad(), make_bars([100.0, 101.0, 102.0])).run()


def test_order_validation():
    with pytest.raises(ValueError, match="quantity"):
        Order(symbol="X", direction=Direction.BUY, quantity=0)
    with pytest.raises(ValueError, match="limit_price"):
        Order(symbol="X", direction=Direction.BUY, quantity=1, order_type=OrderType.LIMIT)
    with pytest.raises(ValueError, match="stop_price"):
        Order(symbol="X", direction=Direction.BUY, quantity=1, order_type=OrderType.STOP)


def test_context_exposes_history_and_snapshot(make_bars):
    seen = {}

    class Inspector:
        def on_bar(self, ctx):
            seen["len"] = len(ctx.history)
            seen["close"] = ctx.close
            seen["equity"] = ctx.equity
            seen["cash"] = ctx.portfolio["cash"]
            seen["position"] = ctx.position
            return []

    bars = make_bars([100.0, 102.0, 104.0, 106.0])
    BacktestEngine(Inspector(), bars, broker_cfg=free_broker()).run()
    assert seen["len"] == 4
    assert seen["close"] == pytest.approx(106.0)
    assert seen["equity"] == pytest.approx(100_000.0)
    assert seen["cash"] == pytest.approx(100_000.0)
    assert seen["position"] == 0.0


def test_sma_cross_runs_and_trades():
    from quantflow.data import synthetic_bars

    bars = synthetic_bars(n=600, seed=11, freq="1D")
    result = BacktestEngine(SmaCross(10, 30), bars, broker_cfg=free_broker()).run()
    assert len(result.trades) >= 2
    assert result.performance.exposure > 0.0
    assert np.isfinite(result.equity.to_numpy()).all()
    assert (result.equity > 0).all()


def test_donchian_runs_and_trades():
    from quantflow.data import synthetic_bars

    bars = synthetic_bars(n=600, seed=5, freq="1D")
    result = BacktestEngine(
        DonchianBreakout(window=20, atr_window=14, risk_frac=0.05),
        bars,
        broker_cfg=free_broker(),
    ).run()
    assert len(result.trades) >= 2
    assert result.performance.exposure > 0.0


def test_buy_and_hold_strategy(make_bars):
    bars = make_bars(np.arange(100.0, 111.0))
    result = BacktestEngine(BuyAndHold(), bars, broker_cfg=free_broker()).run()
    assert len(result.trades) == 1
    assert result.final_position > 0


def test_strategy_parameter_validation():
    with pytest.raises(ValueError):
        SmaCross(fast=50, slow=20)
    with pytest.raises(ValueError):
        SmaCross(fast=0)
    with pytest.raises(ValueError):
        DonchianBreakout(window=1)
    with pytest.raises(ValueError):
        DonchianBreakout(risk_frac=0.0)
    with pytest.raises(ValueError):
        BuyAndHold(weight=1.5)
