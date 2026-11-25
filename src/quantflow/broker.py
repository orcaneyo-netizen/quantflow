"""Simulated broker: turns order intents into fills against the next bar.

No-lookahead rule
-----------------
Orders are *submitted after* a bar has been processed (i.e. after the strategy
has seen that bar's close). :meth:`SimulatedBroker.process_bar` first promotes
every order submitted since the previous call and only then attempts fills
against the bar it is given. Therefore an order created while bar ``t`` is the
current bar can first fill on bar ``t+1`` or later, at a price drawn from bar
``t+1``'s OHLC envelope. The engine enforces the same ordering, so a strategy
can never trade on information from a bar it has not yet been shown.

Execution rules
---------------
* MARKET: fills at the next bar's open, adjusted adversely by slippage.
* LIMIT: buy is eligible when ``low <= limit``, sell when ``high >= limit``.
  The fill price is the limit price itself: we deliberately grant no gap
  price improvement ("limit or open, whichever is worse for the trader").
* STOP: a buy stop triggers when ``high >= stop`` and then executes at
  ``max(open, stop)``; a sell stop triggers when ``low <= stop`` and executes
  at ``min(open, stop)``. Slippage is applied on top, always against the
  trader.
* Partial fills: quantity per bar is capped at ``max_participation * volume``;
  the remainder stays working for later bars. A zero-volume bar fills nothing.
* Buying power: a buy whose notional plus fees exceeds available cash is
  rejected outright (recorded in :attr:`SimulatedBroker.rejected`). With
  ``allow_short=False`` a sell is capped at the held long quantity and rejected
  when there is no long position.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from quantflow.book import Portfolio
from quantflow.costs import (
    CommissionModel,
    CostConfig,
    FixedSlippage,
    PerShareCommission,
    SlippageModel,
)
from quantflow.events import Direction, FillEvent, MarketEvent, OrderEvent, OrderType


class OrderStatus(Enum):
    """Lifecycle state of a working order."""

    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


def _default_costs() -> CostConfig:
    """Default cost stack: $0.005/share + 1 bp, $1 minimum, 1 bp slippage."""
    return CostConfig(
        commission=PerShareCommission(fixed_per_share=0.005, pct_notional=0.0001, min_commission=1.0),
        slippage=FixedSlippage(bps=1.0),
    )


@dataclass(frozen=True)
class BrokerConfig:
    """Broker simulation parameters.

    Args:
        costs: Commission and slippage models.
        max_participation: Maximum fraction of a bar's volume that one order
            may consume per bar, in ``(0, 1]``.
        allow_short: Whether sell orders may open or extend a short position.
        adv_window: Number of bars used for the rolling average volume passed
            to the slippage model (only bars already seen are used).
    """

    costs: CostConfig = field(default_factory=_default_costs)
    max_participation: float = 1.0
    allow_short: bool = False
    adv_window: int = 20

    def __post_init__(self) -> None:
        if not 0.0 < self.max_participation <= 1.0:
            raise ValueError(
                f"max_participation must be in (0, 1], got {self.max_participation}"
            )
        if self.adv_window < 1:
            raise ValueError(f"adv_window must be >= 1, got {self.adv_window}")
        if not isinstance(self.costs.commission, CommissionModel):
            raise TypeError("costs.commission must implement CommissionModel")
        if not isinstance(self.costs.slippage, SlippageModel):
            raise TypeError("costs.slippage must implement SlippageModel")


@dataclass
class WorkingOrder:
    """Broker-side state for an accepted order.

    Attributes:
        order: The originating :class:`OrderEvent`.
        remaining: Quantity still to fill.
        filled: Quantity filled so far.
        status: Current :class:`OrderStatus`.
        bars_active: Number of bars this order has been eligible for.
    """

    order: OrderEvent
    remaining: float
    filled: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    bars_active: int = 0


class SimulatedBroker:
    """Deterministic, next-bar broker simulator.

    Args:
        config: Broker parameters; defaults to :func:`_default_costs`.
    """

    def __init__(self, config: BrokerConfig | None = None) -> None:
        self.config = config if config is not None else BrokerConfig()
        self._pending: list[WorkingOrder] = []
        self._working: list[WorkingOrder] = []
        self._volumes: deque[float] = deque(maxlen=self.config.adv_window)
        self._status: dict[int, OrderStatus] = {}
        self.rejected: list[tuple[OrderEvent, str]] = []
        self.cancelled: list[OrderEvent] = []
        self.fills: list[FillEvent] = []

    # ------------------------------------------------------------------ API
    def submit(self, order: OrderEvent) -> None:
        """Queue an order for execution from the next bar onwards.

        Args:
            order: The order to work.

        Raises:
            ValueError: If an order with the same sequence id was already
                submitted.
        """
        if order.seq in self._status:
            raise ValueError(f"order seq {order.seq} already submitted")
        self._pending.append(WorkingOrder(order=order, remaining=order.quantity))
        self._status[order.seq] = OrderStatus.PENDING

    def process_bar(self, bar: MarketEvent, portfolio: Portfolio | None = None) -> list[FillEvent]:
        """Attempt to fill all working orders against ``bar``.

        Args:
            bar: The bar to trade against.
            portfolio: Optional book used for buying-power and oversell checks.
                When omitted those checks are skipped (useful for unit tests).

        Returns:
            Fill events generated on this bar, in submission order.
        """
        # Promote orders submitted since the previous bar: no same-bar fills.
        self._working.extend(self._pending)
        self._pending.clear()

        available_cash = portfolio.cash if portfolio is not None else float("inf")
        available_pos = float("inf")  # no position cap without a book
        if portfolio is not None and not self.config.allow_short:
            pos = portfolio.positions.get(bar.symbol)
            available_pos = max(pos.quantity, 0.0) if pos is not None else 0.0

        adv = float(sum(self._volumes) / len(self._volumes)) if self._volumes else None
        bar_fills: list[FillEvent] = []
        still_working: list[WorkingOrder] = []

        for wo in self._working:
            order = wo.order
            wo.bars_active += 1

            reference = self._reference_price(order, bar)
            if reference is None:
                still_working.append(wo)
                continue

            cap = self.config.max_participation * bar.volume
            qty = min(wo.remaining, cap)
            if order.direction == Direction.SELL and not self.config.allow_short:
                qty = min(qty, available_pos)
            if qty <= 0.0:
                if order.direction == Direction.SELL and not self.config.allow_short and available_pos <= 0.0:
                    self._reject(wo, f"no long position in {order.symbol} and allow_short=False")
                else:
                    still_working.append(wo)  # zero volume or capped to zero
                continue

            slip = self.config.costs.slippage.calculate(qty, reference, bar.volume, adv)
            price = reference + order.direction.sign * slip
            if price <= 0.0:
                self._reject(wo, f"non-positive fill price {price:.6f}")
                continue
            commission = self.config.costs.commission.calculate(qty, price)
            notional = qty * price + commission

            if order.direction == Direction.BUY and portfolio is not None:
                if notional > available_cash + 1e-9:
                    self._reject(
                        wo,
                        f"insufficient buying power: need {notional:.2f}, have {available_cash:.2f}",
                    )
                    continue
                available_cash -= notional
            elif order.direction == Direction.SELL:
                available_cash += qty * price - commission
                if not self.config.allow_short:
                    available_pos -= qty

            fill = FillEvent(
                timestamp=bar.timestamp,
                symbol=order.symbol,
                order_type=order.order_type,
                direction=order.direction,
                quantity=qty,
                price=price,
                commission=commission,
                slippage=slip,
                order_seq=order.seq,
                tag=order.tag,
            )
            bar_fills.append(fill)
            self.fills.append(fill)

            wo.remaining -= qty
            wo.filled += qty
            if wo.remaining <= 1e-9:
                wo.remaining = 0.0
                wo.status = OrderStatus.FILLED
                self._status[order.seq] = OrderStatus.FILLED
            else:
                wo.status = OrderStatus.PARTIAL
                self._status[order.seq] = OrderStatus.PARTIAL
                still_working.append(wo)

        self._working = still_working
        # Only bars already observed feed the ADV: no lookahead.
        self._volumes.append(bar.volume)
        return bar_fills

    def cancel_all(self) -> list[OrderEvent]:
        """Cancel every working and pending order.

        Returns:
            The orders that were cancelled.
        """
        cancelled = [wo.order for wo in self._working + self._pending]
        for wo in self._working + self._pending:
            wo.status = OrderStatus.CANCELLED
            self._status[wo.order.seq] = OrderStatus.CANCELLED
        self._working.clear()
        self._pending.clear()
        self.cancelled.extend(cancelled)
        return cancelled

    def order_status(self, order_seq: int) -> OrderStatus:
        """Return the current status of an order by its sequence id.

        Raises:
            KeyError: If the order was never submitted.
        """
        if order_seq not in self._status:
            raise KeyError(f"unknown order seq {order_seq}")
        return self._status[order_seq]

    @property
    def open_orders(self) -> list[WorkingOrder]:
        """Working and pending orders, in submission order."""
        return self._working + self._pending

    # -------------------------------------------------------------- helpers
    def _reject(self, wo: WorkingOrder, reason: str) -> None:
        wo.status = OrderStatus.REJECTED
        self._status[wo.order.seq] = OrderStatus.REJECTED
        self.rejected.append((wo.order, reason))

    @staticmethod
    def _reference_price(order: OrderEvent, bar: MarketEvent) -> float | None:
        """Pre-slippage execution price, or None when the order is not eligible."""
        if order.order_type == OrderType.MARKET:
            return bar.open_

        if order.order_type == OrderType.LIMIT:
            assert order.limit_price is not None  # validated by OrderEvent
            if order.direction == Direction.BUY:
                return order.limit_price if bar.low <= order.limit_price else None
            return order.limit_price if bar.high >= order.limit_price else None

        # STOP: trigger, then market at the worse of stop/open.
        assert order.stop_price is not None
        if order.direction == Direction.BUY:
            if bar.high >= order.stop_price:
                return max(bar.open_, order.stop_price)
            return None
        if bar.low <= order.stop_price:
            return min(bar.open_, order.stop_price)
        return None
