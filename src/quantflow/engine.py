"""Event-driven backtest engine.

Execution loop (one iteration per bar ``t``)
-------------------------------------------
1. Append the :class:`MarketEvent` for bar ``t`` to the audit log.
2. Let the broker fill orders that were submitted on or before bar ``t-1``,
   using bar ``t``'s OHLC envelope (the no-lookahead rule).
3. Apply each fill to the portfolio and append the :class:`FillEvent`.
4. Mark the book to market at bar ``t``'s close and record one equity point.
5. Hand the strategy a :class:`Context` whose ``history`` ends *at and
   including* bar ``t``; collect its orders, log them and queue them with the
   broker so they can first fill on bar ``t+1``.

The resulting ``trades`` frame has one row per fill and the ``bars`` frame one
row per bar, so every number in a report can be traced back to the event log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from quantflow.book import Portfolio
from quantflow.broker import BrokerConfig, SimulatedBroker
from quantflow.data import BarSeries
from quantflow.events import (
    Direction,
    EventLog,
    MarketEvent,
    OrderEvent,
    OrderType,
)
from quantflow.metrics import Performance, performance

TRADE_COLUMNS = [
    "timestamp",
    "symbol",
    "action",
    "qty",
    "price",
    "fee",
    "slippage",
    "cash",
    "position",
    "equity",
]
BAR_COLUMNS = ["timestamp", "close", "cash", "position", "equity"]


@dataclass(frozen=True)
class Order:
    """A strategy's intent to trade, before it becomes an :class:`OrderEvent`.

    Quantities are always positive; the side is carried by ``direction``.

    Args:
        symbol: Instrument to trade.
        direction: BUY or SELL.
        quantity: Positive number of shares/contracts.
        order_type: MARKET, LIMIT or STOP.
        limit_price: Required for LIMIT orders.
        stop_price: Required for STOP orders.
        tag: Optional label propagated to the fill.
    """

    symbol: str
    direction: Direction
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError("LIMIT order requires a positive limit_price")
        elif self.limit_price is not None:
            raise ValueError("limit_price is only valid for LIMIT orders")
        if self.order_type == OrderType.STOP:
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError("STOP order requires a positive stop_price")
        elif self.stop_price is not None:
            raise ValueError("stop_price is only valid for STOP orders")


class Context:
    """Read-only view of the world at the close of the current bar.

    Attributes:
        bar: The current :class:`MarketEvent`.
        index: Zero-based bar number.
        history: OHLCV DataFrame from the first bar through the current bar
            (inclusive) - the strategy must not need anything later.
        portfolio: Portfolio snapshot dict (cash, equity, positions, PnL).
    """

    def __init__(
        self,
        bar: MarketEvent,
        index: int,
        history: pd.DataFrame,
        portfolio: Portfolio,
    ) -> None:
        self.bar = bar
        self.index = index
        self.history = history
        self.symbol = bar.symbol
        self._portfolio = portfolio
        self._orders: list[Order] = []
        self.portfolio = portfolio.snapshot({bar.symbol: bar.close})

    # ---------------------------------------------------------- properties
    @property
    def timestamp(self):
        """Timestamp of the current bar."""
        return self.bar.timestamp

    @property
    def close(self) -> float:
        """Close price of the current bar."""
        return self.bar.close

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._portfolio.cash

    @property
    def position(self) -> float:
        """Signed position in the traded symbol."""
        pos = self._portfolio.positions.get(self.symbol)
        return 0.0 if pos is None else pos.quantity

    @property
    def equity(self) -> float:
        """Cash plus the current position marked at the bar close."""
        return self.cash + self.position * self.close

    @property
    def closes(self) -> pd.Series:
        """Close prices up to and including the current bar."""
        return self.history["close"]

    # ------------------------------------------------------------- orders
    def order_market(self, quantity: float, tag: str | None = None) -> Order:
        """Create a market order. ``quantity > 0`` buys, ``< 0`` sells."""
        return self._make(quantity, OrderType.MARKET, tag=tag)

    def order_limit(self, quantity: float, limit_price: float, tag: str | None = None) -> Order:
        """Create a limit order. ``quantity > 0`` buys, ``< 0`` sells."""
        return self._make(quantity, OrderType.LIMIT, limit_price=limit_price, tag=tag)

    def order_stop(self, quantity: float, stop_price: float, tag: str | None = None) -> Order:
        """Create a stop order. ``quantity > 0`` buys, ``< 0`` sells."""
        return self._make(quantity, OrderType.STOP, stop_price=stop_price, tag=tag)

    def target_position(self, target: float, tag: str | None = None) -> Order | None:
        """Order the difference between ``target`` and the current position.

        Returns:
            A market order, or ``None`` when already at the target (within
            ``1e-9`` shares).
        """
        delta = float(target) - self.position
        if abs(delta) < 1e-9:
            return None
        return self.order_market(delta, tag=tag)

    def _make(
        self,
        quantity: float,
        order_type: OrderType,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tag: str | None = None,
    ) -> Order:
        if quantity == 0 or not pd.notna(quantity):
            raise ValueError(f"order quantity must be non-zero, got {quantity}")
        direction = Direction.BUY if quantity > 0 else Direction.SELL
        return Order(
            symbol=self.symbol,
            direction=direction,
            quantity=abs(float(quantity)),
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            tag=tag,
        )

    def submit(self, order: Order | None) -> None:
        """Queue an order (``None`` is ignored) for the current bar."""
        if order is not None:
            self._orders.append(order)

    def take_orders(self) -> list[Order]:
        """Return and clear the orders queued on this context."""
        orders, self._orders = self._orders, []
        return orders


@runtime_checkable
class Strategy(Protocol):
    """Minimal strategy interface."""

    def on_bar(self, ctx: Context) -> list[Order]:
        """Return the orders to queue after seeing ``ctx``."""
        ...


@dataclass(frozen=True)
class BacktestResult:
    """Immutable output of :meth:`BacktestEngine.run`.

    Attributes:
        equity: Per-bar equity curve.
        trades: One row per fill (timestamp, symbol, action, qty, price, fee,
            slippage, cash, position, equity). ``fee`` is commission in
            currency, ``slippage`` is total slippage cost ``qty * per_share``.
        bars: One row per bar (timestamp, close, cash, position, equity).
        events: Full ordered event log.
        performance: Metrics computed from ``equity``.
        final_cash: Cash after the last bar.
        final_position: Position after the last bar.
        trade_pnls: Realized PnL of every closing fill (round trips).
        unfilled_orders: Orders still working when the data ran out.
    """

    equity: pd.Series
    trades: pd.DataFrame
    bars: pd.DataFrame
    events: EventLog
    performance: Performance
    final_cash: float
    final_position: float
    trade_pnls: tuple[float, ...]
    unfilled_orders: tuple[OrderEvent, ...]

    def summary(self) -> str:
        """Metrics summary plus the first rows of the trade blotter."""
        head = self.performance.summary()
        if self.trades.empty:
            return head + "\n\n(no trades)"
        return head + "\n\ntrades:\n" + self.trades.head(10).to_string(index=False)


class BacktestEngine:
    """Runs a strategy over a :class:`BarSeries` with a simulated broker.

    Args:
        strategy: Object with ``on_bar(ctx) -> list[Order]``.
        data: Validated OHLCV bars.
        broker_cfg: Broker parameters (costs, participation cap, shorting).
        initial_cash: Starting equity, must be positive.
        periods_per_year: Annualisation factor for the metrics.
        rf: Annual risk-free rate for Sharpe/Sortino.

    Raises:
        ValueError: If the data has fewer than 3 bars or ``initial_cash`` is
            not positive.
    """

    def __init__(
        self,
        strategy: Strategy,
        data: BarSeries,
        broker_cfg: BrokerConfig | None = None,
        initial_cash: float = 100_000.0,
        periods_per_year: int = 252,
        rf: float = 0.0,
    ) -> None:
        if not isinstance(data, BarSeries):
            raise TypeError("data must be a BarSeries")
        if len(data) < 3:
            raise ValueError(f"backtest needs at least 3 bars, got {len(data)}")
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")
        if not hasattr(strategy, "on_bar"):
            raise TypeError("strategy must define on_bar(ctx)")

        self.strategy = strategy
        self.data = data
        self.broker_cfg = broker_cfg if broker_cfg is not None else BrokerConfig()
        self.initial_cash = float(initial_cash)
        self.periods_per_year = int(periods_per_year)
        self.rf = float(rf)

        self.events = EventLog()
        self.result: BacktestResult | None = None

    # ------------------------------------------------------------------ run
    def run(self) -> BacktestResult:
        """Execute the backtest and return its :class:`BacktestResult`."""
        portfolio = Portfolio(
            initial_cash=self.initial_cash, allow_short=self.broker_cfg.allow_short
        )
        broker = SimulatedBroker(self.broker_cfg)
        self.events = EventLog()

        trade_rows: list[dict] = []
        bar_rows: list[dict] = []
        positions: list[float] = []
        gross_traded = 0.0

        for i, bar in enumerate(self.data.iter_market_events()):
            self.events.append(bar)

            fills = broker.process_bar(bar, portfolio)
            for fill in fills:
                self.events.append(fill)
                portfolio.apply_fill(fill)
                gross_traded += fill.quantity * fill.price
                trade_rows.append(
                    {
                        "timestamp": bar.timestamp,
                        "symbol": fill.symbol,
                        "action": fill.direction.value,
                        "qty": fill.quantity,
                        "price": fill.price,
                        "fee": fill.commission,
                        "slippage": fill.quantity * fill.slippage,
                        "cash": portfolio.cash,
                        "position": 0.0,  # filled in below
                        "equity": 0.0,
                    }
                )

            equity = portfolio.mark_to_market({bar.symbol: bar.close}, bar.timestamp)
            pos = portfolio.positions.get(bar.symbol)
            position = 0.0 if pos is None else pos.quantity
            positions.append(position)

            for row in (trade_rows[-len(fills):] if fills else []):
                row["position"] = position
                row["equity"] = equity

            bar_rows.append(
                {
                    "timestamp": bar.timestamp,
                    "close": bar.close,
                    "cash": portfolio.cash,
                    "position": position,
                    "equity": equity,
                }
            )

            ctx = Context(
                bar=bar,
                index=i,
                history=self.data.data.iloc[: i + 1],
                portfolio=portfolio,
            )
            orders = self.strategy.on_bar(ctx)
            if orders is None:
                orders = []
            orders = list(orders) + ctx.take_orders()
            for order in orders:
                if not isinstance(order, Order):
                    raise TypeError(
                        f"on_bar must return Order objects, got {type(order).__name__}"
                    )
                order_event = OrderEvent(
                    timestamp=bar.timestamp,
                    symbol=order.symbol,
                    order_type=order.order_type,
                    direction=order.direction,
                    quantity=order.quantity,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    tag=order.tag,
                )
                self.events.append(order_event)
                broker.submit(order_event)

        unfilled = tuple(wo.order for wo in broker.open_orders)
        broker.cancel_all()

        equity_series = pd.Series(
            [r["equity"] for r in bar_rows],
            index=pd.DatetimeIndex([r["timestamp"] for r in bar_rows]),
            name="equity",
        )
        trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
        bars = pd.DataFrame(bar_rows, columns=BAR_COLUMNS)

        mean_equity = float(equity_series.mean())
        trade_pnls = tuple(portfolio.round_trip_pnls())
        perf = performance(
            equity_series,
            periods_per_year=self.periods_per_year,
            rf=self.rf,
            trade_pnls=trade_pnls,
            exposure=float(sum(p != 0.0 for p in positions) / len(positions)),
            turnover=(gross_traded / mean_equity) if mean_equity > 0 else 0.0,
            strict=False,
        )

        self.result = BacktestResult(
            equity=equity_series,
            trades=trades,
            bars=bars,
            events=self.events,
            performance=perf,
            final_cash=portfolio.cash,
            final_position=positions[-1],
            trade_pnls=trade_pnls,
            unfilled_orders=unfilled,
        )
        return self.result

    # ------------------------------------------------------------- helpers
    def results(self) -> pd.DataFrame:
        """Trade blotter of the last run (empty frame before :meth:`run`)."""
        if self.result is None:
            return pd.DataFrame(columns=TRADE_COLUMNS)
        return self.result.trades

    def equity_curve(self) -> pd.Series:
        """Equity curve of the last run (empty series before :meth:`run`)."""
        if self.result is None:
            return pd.Series(dtype=float, name="equity")
        return self.result.equity
