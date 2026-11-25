"""Tests for the simulated broker: no lookahead, order types, caps, rejects."""

from __future__ import annotations

import pandas as pd
import pytest

from quantflow.book import Portfolio
from quantflow.broker import BrokerConfig, OrderStatus, SimulatedBroker
from quantflow.costs import CostConfig, FixedSlippage, PerShareCommission
from quantflow.events import Direction, FillEvent, MarketEvent, OrderEvent, OrderType

START = pd.Timestamp("2020-01-01")


def bar(minute, o, h, l, c, v=1_000_000.0, symbol="X"):
    return MarketEvent(
        timestamp=START + pd.Timedelta(minutes=minute),
        symbol=symbol,
        open_=o,
        high=h,
        low=l,
        close=c,
        volume=v,
    )


def order(direction, qty, order_type=OrderType.MARKET, limit=None, stop=None):
    return OrderEvent(
        timestamp=START,
        symbol="X",
        order_type=order_type,
        direction=direction,
        quantity=qty,
        limit_price=limit,
        stop_price=stop,
    )


def free_costs() -> CostConfig:
    return CostConfig(
        commission=PerShareCommission(0.0, 0.0, 0.0),
        slippage=FixedSlippage(0.0),
    )


def test_no_lookahead_market_fills_next_bar_open():
    broker = SimulatedBroker(BrokerConfig(costs=free_costs()))

    b0 = bar(0, 100.0, 101.0, 99.0, 100.5)
    b1 = bar(1, 105.0, 106.0, 104.0, 105.5)
    assert broker.process_bar(b0) == []

    broker.submit(order(Direction.BUY, 10))
    fills = broker.process_bar(b1)

    assert len(fills) == 1
    assert fills[0].price == pytest.approx(105.0)  # bar 1 open, not bar 0 close
    assert fills[0].timestamp == b1.timestamp
    assert fills[0].quantity == 10
    assert broker.order_status(fills[0].order_seq) is OrderStatus.FILLED


def test_limit_not_filled_when_untouched_then_filled_at_limit():
    broker = SimulatedBroker(BrokerConfig(costs=free_costs()))
    broker.process_bar(bar(0, 100.0, 101.0, 99.0, 100.0))

    broker.submit(order(Direction.BUY, 5, OrderType.LIMIT, limit=90.0))
    assert broker.process_bar(bar(1, 100.0, 101.0, 95.0, 96.0)) == []  # low 95 > 90

    fills = broker.process_bar(bar(2, 100.0, 100.0, 89.0, 92.0))
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(90.0)


def test_sell_limit_needs_high_to_cross():
    broker = SimulatedBroker(BrokerConfig(costs=free_costs()))
    broker.process_bar(bar(0, 100.0, 101.0, 99.0, 100.0))
    broker.submit(order(Direction.SELL, 5, OrderType.LIMIT, limit=110.0))

    assert broker.process_bar(bar(1, 100.0, 108.0, 99.0, 105.0)) == []
    fills = broker.process_bar(bar(2, 100.0, 112.0, 99.0, 111.0))
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(110.0)


def test_stop_triggers_intrabar_at_worse_price():
    broker = SimulatedBroker(BrokerConfig(costs=free_costs()))
    broker.process_bar(bar(0, 100.0, 101.0, 99.0, 100.0))
    broker.submit(order(Direction.SELL, 5, OrderType.STOP, stop=95.0))

    # low 94 breaches the 95 stop; open 100 is better, so fill at the stop
    fills = broker.process_bar(bar(1, 100.0, 105.0, 94.0, 96.0))
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(95.0)

    # a gap through the stop fills at the (worse) open
    broker2 = SimulatedBroker(BrokerConfig(costs=free_costs()))
    broker2.process_bar(bar(0, 100.0, 101.0, 99.0, 100.0))
    broker2.submit(order(Direction.SELL, 5, OrderType.STOP, stop=95.0))
    fills2 = broker2.process_bar(bar(1, 90.0, 92.0, 88.0, 91.0))
    assert fills2[0].price == pytest.approx(90.0)


def test_buy_stop_triggers_above_market():
    broker = SimulatedBroker(BrokerConfig(costs=free_costs()))
    broker.process_bar(bar(0, 100.0, 101.0, 99.0, 100.0))
    broker.submit(order(Direction.BUY, 5, OrderType.STOP, stop=102.0))
    assert broker.process_bar(bar(1, 100.0, 101.0, 99.0, 100.0)) == []
    fills = broker.process_bar(bar(2, 100.0, 103.0, 99.0, 102.5))
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(102.0)


def test_participation_cap_partial_fills():
    cfg = BrokerConfig(costs=free_costs(), max_participation=0.1)
    broker = SimulatedBroker(cfg)
    broker.process_bar(bar(0, 100.0, 100.0, 100.0, 100.0, v=1_000.0))
    order_event = order(Direction.BUY, 500)
    broker.submit(order_event)

    fills = broker.process_bar(bar(1, 100.0, 100.0, 100.0, 100.0, v=1_000.0))
    assert [f.quantity for f in fills] == [100.0]
    assert broker.order_status(order_event.seq) is OrderStatus.PARTIAL

    fills = broker.process_bar(bar(2, 100.0, 100.0, 100.0, 100.0, v=1_000.0))
    assert [f.quantity for f in fills] == [100.0]
    fills = broker.process_bar(bar(3, 100.0, 100.0, 100.0, 100.0, v=10_000.0))
    assert [f.quantity for f in fills] == [300.0]  # remainder, capped at 1000
    assert broker.order_status(order_event.seq) is OrderStatus.FILLED


def test_zero_volume_bar_fills_nothing():
    broker = SimulatedBroker(BrokerConfig(costs=free_costs()))
    broker.process_bar(bar(0, 100.0, 100.0, 100.0, 100.0))
    broker.submit(order(Direction.BUY, 10))
    assert broker.process_bar(bar(1, 100.0, 100.0, 100.0, 100.0, v=0.0)) == []
    assert len(broker.open_orders) == 1
    assert len(broker.process_bar(bar(2, 100.0, 100.0, 100.0, 100.0))) == 1


def test_insufficient_buying_power_rejects():
    broker = SimulatedBroker(BrokerConfig(costs=free_costs()))
    book = Portfolio(initial_cash=1_000.0)
    broker.process_bar(bar(0, 100.0, 100.0, 100.0, 100.0), book)
    broker.submit(order(Direction.BUY, 100))

    assert broker.process_bar(bar(1, 50.0, 50.0, 50.0, 50.0), book) == []
    assert len(broker.rejected) == 1
    assert "buying power" in broker.rejected[0][1]
    assert broker.order_status(broker.rejected[0][0].seq) is OrderStatus.REJECTED


def test_oversell_capped_to_position():
    broker = SimulatedBroker(BrokerConfig(costs=free_costs(), allow_short=False))
    book = Portfolio(initial_cash=10_000.0, allow_short=False)
    broker.process_bar(bar(0, 100.0, 100.0, 100.0, 100.0), book)
    book.apply_fill(
        FillEvent(
            timestamp=START,
            symbol="X",
            order_type=OrderType.MARKET,
            direction=Direction.BUY,
            quantity=5,
            price=100.0,
            commission=0.0,
            slippage=0.0,
            order_seq=0,
        )
    )
    broker.submit(order(Direction.SELL, 10))
    fills = broker.process_bar(bar(1, 100.0, 100.0, 100.0, 100.0), book)
    assert [f.quantity for f in fills] == [5.0]


def test_slippage_applied_against_trader():
    cfg = BrokerConfig(
        costs=CostConfig(
            commission=PerShareCommission(0.0, 0.0, 0.0),
            slippage=FixedSlippage(bps=100.0),  # 1% of price
        )
    )
    broker = SimulatedBroker(cfg)
    broker.process_bar(bar(0, 100.0, 100.0, 100.0, 100.0))
    broker.submit(order(Direction.BUY, 1))
    buy = broker.process_bar(bar(1, 100.0, 100.0, 100.0, 100.0))[0]
    assert buy.price == pytest.approx(101.0)
    assert buy.slippage == pytest.approx(1.0)

    broker.submit(order(Direction.SELL, 1))
    sell = broker.process_bar(bar(2, 100.0, 100.0, 100.0, 100.0))[0]
    assert sell.price == pytest.approx(99.0)


def test_cancel_all_and_bad_config():
    broker = SimulatedBroker(BrokerConfig(costs=free_costs()))
    broker.process_bar(bar(0, 100.0, 100.0, 100.0, 100.0))
    o = order(Direction.BUY, 1)
    broker.submit(o)
    cancelled = broker.cancel_all()
    assert [c.seq for c in cancelled] == [o.seq]
    assert broker.order_status(o.seq) is OrderStatus.CANCELLED
    assert broker.process_bar(bar(1, 100.0, 100.0, 100.0, 100.0)) == []

    with pytest.raises(ValueError):
        BrokerConfig(max_participation=0.0)
    with pytest.raises(ValueError):
        BrokerConfig(max_participation=1.5)
    with pytest.raises(ValueError):
        BrokerConfig(adv_window=0)


def test_order_requires_reference_price():
    with pytest.raises(ValueError, match="limit_price"):
        OrderEvent(
            timestamp=START,
            symbol="X",
            order_type=OrderType.LIMIT,
            direction=Direction.BUY,
            quantity=1,
        )
    with pytest.raises(ValueError, match="stop_price"):
        OrderEvent(
            timestamp=START,
            symbol="X",
            order_type=OrderType.STOP,
            direction=Direction.BUY,
            quantity=1,
        )
