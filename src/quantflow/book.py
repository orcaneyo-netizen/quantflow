"""Portfolio accounting for the quantflow backtesting engine.

This module provides the :class:`Position` and :class:`Portfolio` classes that
track cash, positions, realized/unrealized PnL and the equity curve. It applies
:class:`~quantflow.events.FillEvent` objects with sign conventions that keep
``equity = cash + sum(quantity * price)`` exact for both long and short books.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from quantflow.events import Direction, FillEvent


@dataclass
class Position:
    """A net position in a single symbol.

    Attributes:
        symbol: Trading symbol.
        quantity: Signed quantity (positive = long, negative = short).
        avg_entry_price: Cost-adjusted average entry price. For a long it is
            *above* the raw fill price by the per-share trade cost; for a short
            it is *below* the raw fill price. This makes
            ``unrealized_pnl(price)`` already net of entry costs.
        realized_pnl: Cumulative realized PnL from closed portions.
    """

    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        """True when the position is net long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """True when the position is net short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """True when there is no position."""
        return self.quantity == 0

    def update_on_fill(self, fill: FillEvent) -> float:
        """Update the position from a fill and return the realized PnL.

        Accounting convention
        ---------------------
        Cash is the source of truth (see :class:`Portfolio`). A fill has a
        *cost-adjusted execution price*

            basis = price + sign(direction) * (commission + qty*slippage) / qty

        i.e. a buy pays fees on top (higher basis), a sell receives fees net
        (lower basis). Realized PnL on the closed quantity ``q`` is

            long : (price - avg_entry) * q - q * cost_per_share
            short: (avg_entry - price) * q - q * cost_per_share

        When a fill flips the position, the residual quantity is booked at
        ``basis`` so that ``cash + quantity * price`` stays exact.

        Args:
            fill: FillEvent to apply.

        Returns:
            Realized PnL from this fill (0.0 when opening or increasing).
        """
        signed_qty = fill.quantity * fill.direction.sign
        price = fill.price
        total_cost = fill.commission + fill.quantity * fill.slippage
        cost_per_share = total_cost / fill.quantity
        basis = price + fill.direction.sign * cost_per_share

        old_qty = self.quantity
        new_qty = old_qty + signed_qty

        if old_qty == 0.0:
            # Opening a new position; the entry basis includes the trade cost.
            self.quantity = new_qty
            self.avg_entry_price = basis
            return 0.0

        if (old_qty > 0) == (signed_qty > 0):
            # Increasing an existing position: quantity-weighted average basis.
            old_abs, add_abs = abs(old_qty), abs(signed_qty)
            self.avg_entry_price = (
                old_abs * self.avg_entry_price + add_abs * basis
            ) / (old_abs + add_abs)
            self.quantity = new_qty
            return 0.0

        # Reducing, closing or flipping.
        close_qty = min(abs(old_qty), abs(signed_qty))
        if old_qty > 0:
            gross = (price - self.avg_entry_price) * close_qty
        else:
            gross = (self.avg_entry_price - price) * close_qty
        realized = gross - cost_per_share * close_qty
        self.realized_pnl += realized
        self.quantity = new_qty

        if new_qty == 0.0:
            self.avg_entry_price = 0.0
        elif (new_qty > 0) != (old_qty > 0):
            # Flipped through zero: the residual position starts at this fill.
            self.avg_entry_price = basis
        return realized

    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized PnL at the current price.

        Formula:
            long : (price - avg_entry) * quantity
            short: (avg_entry - price) * |quantity|

        Args:
            current_price: Current market price.

        Returns:
            Unrealized PnL (0.0 when flat).
        """
        if self.is_flat:
            return 0.0
        if self.is_long:
            return (current_price - self.avg_entry_price) * self.quantity
        return (self.avg_entry_price - current_price) * (-self.quantity)


@dataclass
class Portfolio:
    """Cash, positions, realized/unrealized PnL and the equity curve.

    Equity is defined as

        equity_t = cash_t + sum_i quantity_i * price_i

    and every fill moves cash by ``-qty*price - fees`` (buy) or
    ``+qty*price - fees`` (sell), where ``fees = commission + qty*slippage``.
    Consequently ``equity`` is exactly the initial cash plus total PnL net of
    costs; no PnL is double counted.

    Args:
        initial_cash: Starting cash balance (must be positive).
        allow_short: Whether net short positions may be opened. When False an
            oversell (sell more than the current long position) raises
            ``ValueError``.
    """

    initial_cash: float
    allow_short: bool = True
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list, init=False)
    trade_log: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        self.cash = float(self.initial_cash)

    def get_position(self, symbol: str) -> Position:
        """Return the position for ``symbol``, creating a flat one if needed."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def apply_fill(self, fill: FillEvent) -> dict[str, Any]:
        """Apply a fill to cash and the position book.

        Args:
            fill: FillEvent to apply.

        Returns:
            A trade record dict (timestamp, symbol, direction, quantity, price,
            commission, slippage, realized_pnl, cash, position).

        Raises:
            ValueError: If a BUY would overdraw cash, or if a SELL would open a
                short while ``allow_short`` is False.
        """
        pos = self.get_position(fill.symbol)
        fees = fill.commission + fill.quantity * fill.slippage

        if fill.direction == Direction.BUY:
            cost = fill.quantity * fill.price + fees
            if cost > self.cash + 1e-9:
                raise ValueError(
                    f"insufficient cash for {fill.symbol}: need {cost:.2f}, "
                    f"have {self.cash:.2f}"
                )
            self.cash -= cost
        else:
            if not self.allow_short and fill.quantity > pos.quantity + 1e-9:
                raise ValueError(
                    f"oversell rejected for {fill.symbol}: sell "
                    f"{fill.quantity:g} > position {pos.quantity:g}"
                )
            self.cash += fill.quantity * fill.price - fees

        realized_pnl = pos.update_on_fill(fill)

        trade_record = {
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "direction": fill.direction.value,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
            "slippage": fill.slippage,
            "realized_pnl": realized_pnl,
            "cash": self.cash,
            "position": pos.quantity,
        }
        self.trade_log.append(trade_record)
        return trade_record

    def mark_to_market(self, prices: dict[str, float], timestamp: datetime) -> float:
        """Mark the book to market and append an equity point.

        Args:
            prices: Mapping of symbol -> current price. Every non-flat position
                must have a price.
            timestamp: Timestamp of the equity point.

        Returns:
            Total equity ``cash + sum(quantity * price)``.

        Raises:
            ValueError: If a held position has no price.
        """
        total_value = self.cash
        for symbol, pos in self.positions.items():
            if not pos.is_flat:
                price = prices.get(symbol)
                if price is None:
                    raise ValueError(f"missing price for {symbol} at mark-to-market")
                total_value += pos.quantity * price

        self.equity_curve.append((timestamp, total_value))
        return total_value

    def equity_series(self) -> pd.Series:
        """Return the recorded equity curve as a pandas Series."""
        if not self.equity_curve:
            return pd.Series(dtype=float, name="equity")
        timestamps, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="equity")

    def total_realized_pnl(self) -> float:
        """Sum of realized PnL across all positions."""
        return sum(pos.realized_pnl for pos in self.positions.values())

    def total_unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Sum of unrealized PnL across all open positions.

        Raises:
            ValueError: If a held position has no price.
        """
        total = 0.0
        for symbol, pos in self.positions.items():
            if not pos.is_flat:
                price = prices.get(symbol)
                if price is None:
                    raise ValueError(f"missing price for {symbol}")
                total += pos.unrealized_pnl(price)
        return total

    def round_trip_pnls(self) -> list[float]:
        """Realized PnL of every closing fill, in chronological order.

        Only fills that reduce/close/flip a position produce realized PnL, so
        this is the per-round-trip series used by the hit-rate statistics.
        """
        return [float(t["realized_pnl"]) for t in self.trade_log if t["realized_pnl"] != 0.0]

    def snapshot(self, prices: dict[str, float]) -> dict[str, Any]:
        """Return a read-only snapshot of the book for strategy contexts.

        Args:
            prices: Current prices used for mark-to-market.

        Returns:
            Dict with ``cash``, ``equity``, ``positions`` (per-symbol detail),
            ``total_realized_pnl`` and ``total_unrealized_pnl``.
        """
        positions_snapshot: dict[str, dict[str, float]] = {}
        for symbol, pos in self.positions.items():
            if not pos.is_flat:
                price = prices.get(symbol, 0.0)
                positions_snapshot[symbol] = {
                    "quantity": pos.quantity,
                    "avg_entry": pos.avg_entry_price,
                    "market_value": pos.quantity * price,
                    "unrealized_pnl": pos.unrealized_pnl(price),
                    "realized_pnl": pos.realized_pnl,
                }

        equity = self.cash + sum(p["market_value"] for p in positions_snapshot.values())
        return {
            "cash": self.cash,
            "equity": equity,
            "positions": positions_snapshot,
            "total_realized_pnl": self.total_realized_pnl(),
            "total_unrealized_pnl": self.total_unrealized_pnl(prices),
        }
