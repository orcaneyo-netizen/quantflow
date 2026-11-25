# quantflow — event-driven backtesting engine

Read `..\CONVENTIONS.md` first. Package name: `quantflow`.

## Goal

A realistic, event-driven backtester for a single instrument or a small universe,
with broker simulation, explicit costs, and an auditable event log. It must be
correct enough that a naive strategy's PnL can be verified by hand.

## Modules

`src/quantflow/`

1. `events.py` — immutable `Event` dataclasses: `MarketEvent`, `SignalEvent`,
   `OrderEvent`, `FillEvent`; enums `OrderType` (MARKET, LIMIT, STOP),
   `Direction` (BUY, SELL). Monotonic sequence ids.
2. `data.py` — `BarSeries` dataclass wrapping an OHLCV `pandas.DataFrame`
   (validated index: tz-naive DatetimeIndex, ascending, no NaN in OHLC, high>=low,
   high>=max(open,close), low<=min(open,close)). `load_csv(path)` and
   `synthetic_bars(n=1000, seed=42, start_price=100.0)` (geometric Brownian motion
   with drift/vol defaults; document params). Iterating bars must yield
   `MarketEvent`s.
3. `costs.py` — `CommissionModel` (per-share `fixed_per_share + pct_notional`,
   common minimum), `SlippageModel` (fixed bps of price, plus optional
   square-root market-impact term `k * sqrt(qty / adv)`).
4. `book.py` — `Portfolio` accounting: cash, position, average entry price,
   realized/unrealized PnL, equity curve; applies `FillEvent`s with sign
   conventions for long/short; rejects oversell when `allow_short=False`.
5. `broker.py` — `SimulatedBroker`: turns `OrderEvent`s into `FillEvent`s against
   the next bar (no lookahead — document this rule explicitly), handles LIMIT
   (fill if next bar crosses, at limit or open, whichever is worse for the
   trader), STOP (trigger then market), rejects insufficient buying power, partial
   fills capped by `max_participation` of bar volume.
6. `engine.py` — `BacktestEngine(strategy, data, broker_cfg, initial_cash)`.
   Strategy protocol: `on_bar(ctx) -> list[Order]` where `ctx` exposes history
   up to *and including* the current bar, portfolio snapshot, and helpers
   (`order_market`, `order_limit`, ...). Engine emits ordered events, records
   every event to `self.events` and a tidy `results()` DataFrame
   (timestamp, action, qty, price, fee, slippage, cash, position, equity).
7. `metrics.py` — `performance(equity: pd.Series, periods_per_year=252)` →
   dataclass `Performance` with total_return, CAGR, annualized_vol, Sharpe
   (excess over `rf`), Sortino, Calmar, max_drawdown, drawdown duration,
   hit_rate (per round-trip trade), avg_win/avg_loss, profit_factor, exposure,
   turnover, VaR95/CVaR95 of daily returns. Each formula documented.
8. `walkforward.py` — `walk_forward(data, strategy_factory, n_splits, train, test,
   embargo)` returning per-fold `Performance` and an OOS equity curve; no
   overlap between train and test windows; assert embargo >= 0.
9. `strategies.py` — two reference strategies proving the engine:
   `SmaCross(fast, slow)` and `DonchianBreakout(window, atr_window, risk_frac)`
   (position sized by ATR stop distance).
10. `__main__.py` — `demo` subcommand: synthetic bars → SmaCross → print metrics
    table + first 10 trades; `--plot` saves `equity.png`.

## Tests (`tests/`)

- `test_data.py`: CSV round-trip, OHLC sanity rejection, determinism of
  synthetic generator for a fixed seed.
- `test_costs.py`: commission with minimum, slippage bps math, impact monotonicity.
- `test_book.py`: long round-trip PnL equals hand-computed value; short PnL sign;
  oversell rejected.
- `test_broker.py`: no-lookahead (order at bar t fills at t+1 open), limit not
  filled when untouched, stop triggered intrabar, participation cap.
- `test_metrics.py`: metrics on a hand-built equity curve (e.g. equity
  [100, 110, 99, 120]) — assert exact Sharpe sign / max drawdown = 10%.
- `test_engine.py`: buy-and-hold on a monotone series produces equity growth
  consistent with fees; event log is time-ordered.
- `test_walkforward.py`: fold boundaries correct, no train/test overlap.

## README sections

Quickstart, architecture diagram (ASCII), the no-lookahead rule, cost model,
metric definitions table, limitations.
