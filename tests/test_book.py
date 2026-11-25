"""Tests for portfolio accounting: exact PnL, sign conventions, rejections."""

from __future__ import annotations

import pandas as pd
import pytest

from quantflow.book import Portfolio
from quantflow.events import Direction, FillEvent, OrderType

TS = pd.Timestamp("2020-01-01")


def fill(direction, qty, price, commission=0.0, slippage=0.0):
    return FillEvent(
        timestamp=TS,
        symbol="X",
        order_type=OrderType.MARKET,
        direction=direction,
        quantity=qty,
        price=price,
        commission=commission,
        slippage=slippage,
        order_seq=1,
    )


def test_long_round_trip_hand_computed():
    book = Portfolio(initial_cash=10_000.0)
    book.apply_fill(fill(Direction.BUY, 10, 100.0, commission=1.0))
    assert book.cash == pytest.approx(10_000.0 - 1000.0 - 1.0)

    record = book.apply_fill(fill(Direction.SELL, 10, 110.0, commission=1.0))
    # gross 100 minus 2 of commissions
    assert record["realized_pnl"] == pytest.approx(98.0)
    assert book.cash == pytest.approx(10_098.0)
    assert book.get_position("X").quantity == 0
    assert book.get_position("X").avg_entry_price == 0.0
    assert book.total_realized_pnl() == pytest.approx(98.0)
    assert book.round_trip_pnls() == [pytest.approx(98.0)]


def test_short_pnl_sign():
    book = Portfolio(initial_cash=10_000.0, allow_short=True)
    book.apply_fill(fill(Direction.SELL, 10, 100.0))
    assert book.cash == pytest.approx(11_000.0)
    assert book.get_position("X").is_short

    # short entry basis is below the raw price (cost-adjusted); no costs here
    assert book.get_position("X").avg_entry_price == pytest.approx(100.0)
    record = book.apply_fill(fill(Direction.BUY, 10, 90.0))
    assert record["realized_pnl"] == pytest.approx(100.0)
    assert book.cash == pytest.approx(10_100.0)
    assert book.get_position("X").is_flat


def test_short_with_costs_keeps_equity_exact():
    book = Portfolio(initial_cash=10_000.0, allow_short=True)
    book.apply_fill(fill(Direction.SELL, 10, 100.0, commission=2.0))
    # cash = 10_000 + 1000 - 2; basis = 100 - 0.2
    assert book.cash == pytest.approx(10_998.0)
    assert book.get_position("X").avg_entry_price == pytest.approx(99.8)
    equity = book.mark_to_market({"X": 100.0}, TS)
    assert equity == pytest.approx(9_998.0)  # exactly the commission paid


def test_oversell_rejected_when_shorts_disallowed():
    book = Portfolio(initial_cash=10_000.0, allow_short=False)
    book.apply_fill(fill(Direction.BUY, 5, 100.0))
    with pytest.raises(ValueError, match="oversell"):
        book.apply_fill(fill(Direction.SELL, 10, 100.0))

    fresh = Portfolio(initial_cash=10_000.0, allow_short=False)
    with pytest.raises(ValueError, match="oversell"):
        fresh.apply_fill(fill(Direction.SELL, 1, 100.0))


def test_buy_beyond_cash_rejected():
    book = Portfolio(initial_cash=1_000.0)
    with pytest.raises(ValueError, match="insufficient cash"):
        book.apply_fill(fill(Direction.BUY, 100, 50.0))


def test_partial_close_and_flip():
    book = Portfolio(initial_cash=10_000.0)
    book.apply_fill(fill(Direction.BUY, 10, 100.0))
    record = book.apply_fill(fill(Direction.SELL, 4, 110.0))
    assert record["realized_pnl"] == pytest.approx(40.0)
    assert book.get_position("X").quantity == pytest.approx(6.0)

    # flip from +6 to -4 at 120: realize 6*20, residual short at 120
    record = book.apply_fill(fill(Direction.SELL, 10, 120.0))
    assert record["realized_pnl"] == pytest.approx(120.0)
    pos = book.get_position("X")
    assert pos.quantity == pytest.approx(-4.0)
    assert pos.avg_entry_price == pytest.approx(120.0)
    assert book.cash == pytest.approx(10_000.0 - 1000.0 + 440.0 + 1200.0)
    # equity is exact: cash + qty*price
    assert book.mark_to_market({"X": 120.0}, TS) == pytest.approx(10_000.0 + 40.0 + 120.0)


def test_mark_to_market_and_unrealized():
    book = Portfolio(initial_cash=10_000.0)
    book.apply_fill(fill(Direction.BUY, 10, 100.0, commission=1.0))
    equity = book.mark_to_market({"X": 105.0}, TS)
    # cash 8999 + 10 * 105 = 10049 -> +49 net of the 1 commission
    assert equity == pytest.approx(10_049.0)
    assert book.total_unrealized_pnl({"X": 105.0}) == pytest.approx(49.0)
    assert len(book.equity_series()) == 1

    with pytest.raises(ValueError, match="missing price"):
        book.mark_to_market({}, TS)


def test_snapshot():
    book = Portfolio(initial_cash=10_000.0)
    book.apply_fill(fill(Direction.BUY, 10, 100.0))
    snap = book.snapshot({"X": 101.0})
    assert snap["cash"] == pytest.approx(9_000.0)
    assert snap["equity"] == pytest.approx(10_010.0)
    assert snap["positions"]["X"]["quantity"] == 10
    assert snap["total_unrealized_pnl"] == pytest.approx(10.0)


def test_initial_cash_must_be_positive():
    with pytest.raises(ValueError):
        Portfolio(initial_cash=0.0)
