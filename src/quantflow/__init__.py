"""quantflow - a small, auditable, event-driven backtesting engine.

Quick example
-------------
>>> from quantflow import BacktestEngine, SmaCross, synthetic_bars
>>> bars = synthetic_bars(n=500, seed=42, freq="1D")
>>> result = BacktestEngine(SmaCross(fast=10, slow=30), bars).run()
>>> round(result.performance.total_return, 4)  # doctest: +SKIP

The package is deliberately dependency-light (numpy, pandas, scipy,
matplotlib), fully offline and deterministic: every random generator takes an
explicit ``seed``.
"""

from __future__ import annotations

from quantflow.book import Portfolio, Position
from quantflow.broker import BrokerConfig, OrderStatus, SimulatedBroker
from quantflow.costs import (
    CommissionModel,
    CostConfig,
    FixedSlippage,
    PerShareCommission,
    SlippageModel,
    SquareRootImpactSlippage,
)
from quantflow.data import BarSeries, load_csv, synthetic_bars
from quantflow.engine import (
    BacktestEngine,
    BacktestResult,
    Context,
    Order,
    Strategy,
)
from quantflow.events import (
    Direction,
    Event,
    EventLog,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    reset_seq_counter,
)
from quantflow.metrics import Performance, drawdown_series, performance
from quantflow.strategies import BuyAndHold, DonchianBreakout, SmaCross
from quantflow.walkforward import Fold, WalkForwardResult, walk_forward

__version__ = "0.1.0"

__all__ = [
    "BarSeries",
    "BacktestEngine",
    "BacktestResult",
    "BrokerConfig",
    "BuyAndHold",
    "CommissionModel",
    "Context",
    "CostConfig",
    "Direction",
    "DonchianBreakout",
    "Event",
    "EventLog",
    "FillEvent",
    "FixedSlippage",
    "Fold",
    "MarketEvent",
    "Order",
    "OrderEvent",
    "OrderStatus",
    "OrderType",
    "Performance",
    "PerShareCommission",
    "Portfolio",
    "Position",
    "SignalEvent",
    "SimulatedBroker",
    "SlippageModel",
    "SmaCross",
    "SquareRootImpactSlippage",
    "Strategy",
    "WalkForwardResult",
    "drawdown_series",
    "load_csv",
    "performance",
    "reset_seq_counter",
    "synthetic_bars",
    "walk_forward",
    "__version__",
]
