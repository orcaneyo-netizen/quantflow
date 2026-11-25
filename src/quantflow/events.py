"""Event definitions for the quantflow backtesting engine.

This module defines immutable event dataclasses that form the backbone of the
event-driven architecture. All events carry a monotonic sequence ID to ensure
deterministic ordering and auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderType(Enum):
    """Order types supported by the simulated broker."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class Direction(Enum):
    """Trade direction: BUY (long) or SELL (short)."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """Return +1 for BUY, -1 for SELL for PnL calculations."""
        return 1 if self == Direction.BUY else -1


# Global sequence counter for monotonic event IDs
_seq_counter = 0


def _next_seq() -> int:
    """Generate the next monotonic sequence ID."""
    global _seq_counter
    _seq_counter += 1
    return _seq_counter


def reset_seq_counter() -> None:
    """Reset the sequence counter (useful for testing)."""
    global _seq_counter
    _seq_counter = 0


@dataclass(frozen=True, order=True, kw_only=True)
class Event:
    """Base class for all events.

    Events are immutable and ordered by their sequence ID to guarantee
    a total ordering that matches emission order.
    """

    seq: int = field(default_factory=_next_seq, compare=True)
    timestamp: datetime = field(compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError(f"timestamp must be datetime, got {type(self.timestamp)}")


@dataclass(frozen=True, order=True, kw_only=True)
class MarketEvent(Event):
    """Market data event carrying OHLCV bar data.

    Attributes:
        symbol: Trading symbol (e.g., "SPY", "AAPL").
        open_: Open price.
        high: High price.
        low: Low price.
        close: Close price.
        volume: Bar volume.
    """

    symbol: str = field(compare=False)
    open_: float = field(compare=False)
    high: float = field(compare=False)
    low: float = field(compare=False)
    close: float = field(compare=False)
    volume: float = field(compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if self.high < max(self.open_, self.close):
            raise ValueError("high must be >= max(open, close)")
        if self.low > min(self.open_, self.close):
            raise ValueError("low must be <= min(open, close)")
        if any(v <= 0 for v in (self.open_, self.high, self.low, self.close)):
            raise ValueError("OHLC prices must be positive")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")


@dataclass(frozen=True, order=True, kw_only=True)
class SignalEvent(Event):
    """Strategy signal event indicating a desired position change.

    Attributes:
        symbol: Trading symbol.
        direction: Desired trade direction.
        strength: Signal strength in [-1, 1] (1 = max long, -1 = max short).
        metadata: Optional strategy-specific data.
    """

    symbol: str = field(compare=False)
    direction: Direction = field(compare=False)
    strength: float = field(compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [-1, 1], got {self.strength}")


@dataclass(frozen=True, order=True, kw_only=True)
class OrderEvent(Event):
    """Order event sent to the broker for execution.

    Attributes:
        symbol: Trading symbol.
        order_type: Order type (MARKET, LIMIT, STOP).
        direction: Trade direction (BUY/SELL).
        quantity: Number of shares/contracts.
        limit_price: Limit price for LIMIT orders (required for LIMIT).
        stop_price: Stop price for STOP orders (required for STOP).
        tag: Optional user-defined tag for tracking.
    """

    symbol: str = field(compare=False)
    order_type: OrderType = field(compare=False)
    direction: Direction = field(compare=False)
    quantity: float = field(compare=False)
    limit_price: float | None = field(default=None, compare=False)
    stop_price: float | None = field(default=None, compare=False)
    tag: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT order requires limit_price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("STOP order requires stop_price")
        if self.order_type == OrderType.MARKET and (self.limit_price is not None or self.stop_price is not None):
            raise ValueError("MARKET order must not have limit_price or stop_price")


@dataclass(frozen=True, order=True, kw_only=True)
class FillEvent(Event):
    """Fill event representing an executed order.

    Attributes:
        symbol: Trading symbol.
        order_type: Original order type.
        direction: Trade direction.
        quantity: Filled quantity.
        price: Fill price.
        commission: Commission paid.
        slippage: Slippage incurred (in price units).
        order_seq: Sequence ID of the originating OrderEvent.
        tag: Optional user-defined tag from the order.
    """

    symbol: str = field(compare=False)
    order_type: OrderType = field(compare=False)
    direction: Direction = field(compare=False)
    quantity: float = field(compare=False)
    price: float = field(compare=False)
    commission: float = field(compare=False)
    slippage: float = field(compare=False)
    order_seq: int = field(compare=False)
    tag: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.commission < 0:
            raise ValueError("commission must be non-negative")
        if self.slippage < 0:
            raise ValueError("slippage must be non-negative")


@dataclass(frozen=True)
class EventLog:
    """Container for all events emitted during a backtest.

    Provides an auditable, ordered log of every event.
    """

    events: list[Event] = field(default_factory=list)

    def append(self, event: Event) -> None:
        """Append an event to the log."""
        self.events.append(event)

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Event:
        return self.events[idx]

    def filter(self, event_type: type) -> list[Event]:
        """Return all events of a given type."""
        return [e for e in self.events if isinstance(e, event_type)]

    def to_dataframe(self):
        """Convert event log to a pandas DataFrame for analysis."""
        import pandas as pd

        rows = []
        for e in self.events:
            row = {
                "seq": e.seq,
                "timestamp": e.timestamp,
                "type": type(e).__name__,
            }
            if isinstance(e, MarketEvent):
                row.update(
                    symbol=e.symbol,
                    open=e.open_,
                    high=e.high,
                    low=e.low,
                    close=e.close,
                    volume=e.volume,
                )
            elif isinstance(e, SignalEvent):
                row.update(
                    symbol=e.symbol,
                    direction=e.direction.value,
                    strength=e.strength,
                )
            elif isinstance(e, OrderEvent):
                row.update(
                    symbol=e.symbol,
                    order_type=e.order_type.value,
                    direction=e.direction.value,
                    quantity=e.quantity,
                    limit_price=e.limit_price,
                    stop_price=e.stop_price,
                    tag=e.tag,
                )
            elif isinstance(e, FillEvent):
                row.update(
                    symbol=e.symbol,
                    order_type=e.order_type.value,
                    direction=e.direction.value,
                    quantity=e.quantity,
                    price=e.price,
                    commission=e.commission,
                    slippage=e.slippage,
                    order_seq=e.order_seq,
                    tag=e.tag,
                )
            rows.append(row)
        return pd.DataFrame(rows)