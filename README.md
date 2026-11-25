# quantflow

A small, auditable, event-driven backtesting engine for a single instrument or
a small universe. Everything is offline, deterministic and dependency-light:
**numpy, pandas, scipy, matplotlib** only.

- Next-bar execution with an explicit **no-lookahead** rule
- Realistic cost stack: per-share + notional commission with a minimum,
  fixed-bps and square-root market-impact slippage
- Long/short portfolio accounting where `equity = cash + qty * price` is exact
- Full event log (`MarketEvent` -> `OrderEvent` -> `FillEvent`) plus tidy
  DataFrames of fills and bars
- Performance metrics with every formula written down, and walk-forward
  out-of-sample evaluation

## Quickstart

```bash
pip install -e .
python -m quantflow demo            # metrics table + first 10 fills
python -m quantflow demo --plot     # also saves equity.png
python -m quantflow walkforward     # rolling out-of-sample report
```

```python
from quantflow import BacktestEngine, SmaCross, synthetic_bars

bars = synthetic_bars(n=1000, seed=42, freq="1D")   # deterministic GBM
result = BacktestEngine(SmaCross(fast=20, slow=50), bars, initial_cash=100_000).run()

print(result.performance.summary())
print(result.trades.head())          # timestamp, action, qty, price, fee, ...
print(result.equity.tail())          # per-bar equity curve
```

Custom strategy — `on_bar` sees history up to and **including** the current bar
and may only queue orders, which the broker fills on the *next* bar:

```python
from quantflow import Order

class FirstBarBuy:
    def on_bar(self, ctx):
        if ctx.index == 0:
            return [ctx.order_market(100, tag="entry")]
        return []
```

## Architecture

```
                 +-------------------+
                 |   BarSeries       |  validated OHLCV (tz-naive, ascending)
                 +---------+---------+
                           |  MarketEvent (bar t)
                           v
+-----------+     +--------+--------+     +------------------+
| Strategy  |---->| BacktestEngine  |---->| SimulatedBroker  |
|  on_bar   |     |  event log      |     |  next-bar fills  |
+-----------+     |  equity curve   |     |  costs + caps    |
      ^           +--------+--------+     +--------+---------+
      |                    |                       |
      |              +-----v------+         +------v------+
      +--------------| Context    |<--------| FillEvent   |
        snapshot     | (read-only)|  apply  |  (t, px, q) |
                     +------------+         +-------------+
                                                    |
                                            +-------v-------+
                                            | Portfolio     |
                                            | cash/position |
                                            | realized PnL  |
                                            +---------------+
```

Per bar: fill previously queued orders -> apply fills -> mark to market ->
build `Context` -> strategy returns orders -> queue them for the next bar.

## The no-lookahead rule

An order created while bar `t` is the current bar can never fill on bar `t`.
The broker promotes newly submitted orders only at the *start* of the next
`process_bar` call, and fills them from that bar's OHLC envelope. A strategy
therefore only ever sees `history.iloc[:t+1]`, and all of its trades happen at
prices from `t+1` or later. Market orders fill at the next bar's open; limit
and stop orders use the next bar's high/low.

## Cost model

| Component | Formula |
| --- | --- |
| Commission | `max(min_commission, qty * fixed_per_share + qty * price * pct_notional)` |
| Fixed slippage | `price * bps / 10_000` per share |
| Square-root impact | `price * (bps / 10_000 + k * sqrt(qty / ADV))` per share |
| Fill price | buy: `ref + slippage`; sell: `ref - slippage` |
| Participation cap | `qty_filled <= max_participation * bar_volume` per bar |

`ADV` is the mean volume of the bars **already seen** (never the current one),
so impact cannot leak future information. A zero-volume bar fills nothing; the
remainder of a partially filled order stays working.

## Execution rules

| Order | Eligibility on the next bar | Fill price |
| --- | --- | --- |
| MARKET | always | `open` |
| BUY LIMIT | `low <= limit` | `limit` (no gap price improvement) |
| SELL LIMIT | `high >= limit` | `limit` |
| BUY STOP | `high >= stop` | `max(open, stop)` |
| SELL STOP | `low <= stop` | `min(open, stop)` |

Buying power is checked before every fill: a buy whose notional plus fees
exceeds available cash is rejected (recorded in `broker.rejected`). With
`allow_short=False` a sell is capped at the held quantity and rejected when
there is no long position.

## Metric definitions

Let `r_t = E_t / E_{t-1} - 1`, `n` returns, `ppy = periods_per_year`.

| Metric | Definition |
| --- | --- |
| total_return | `E_n / E_0 - 1` |
| CAGR | `(E_n / E_0) ** (ppy / n) - 1` (`-1` if `E_n <= 0`) |
| annualized_vol | `std(r, ddof=1) * sqrt(ppy)` |
| Sharpe | `mean(r - rf/ppy) / std(r, ddof=1) * sqrt(ppy)` |
| Sortino | `mean(r - rf/ppy) / sqrt(mean(min(r - rf/ppy, 0)^2)) * sqrt(ppy)` |
| Calmar | `CAGR / max_drawdown` |
| max_drawdown | `-min(E_t / cummax(E)_t - 1)` |
| drawdown duration | longest run of bars strictly below the running peak |
| hit_rate | share of round-trip trades with PnL > 0 |
| avg_win / avg_loss | mean winning / losing round-trip PnL (loss is negative) |
| profit_factor | `sum(wins) / abs(sum(losses))` |
| exposure | fraction of bars with a non-zero position |
| turnover | traded notional / mean equity |
| VaR95 / CVaR95 | `-q05(r)` and `-mean(r \| r <= q05(r))`, as positive losses |

Degenerate inputs: an empty curve, fewer than 3 points, NaN/inf, a non-positive
start, or an intermediate non-positive value raise `ValueError`. A zero-variance
return series raises `ValueError` from `performance()` because Sharpe/Sortino
are undefined; the engine and walk-forward driver call it with
`strict=False`, which reports those ratios as `0.0` instead.

## Tests

```bash
pytest -q
```

The suite checks hand-computed values: equity `[100, 110, 99, 120]` must give a
10% max drawdown and a positive Sharpe, a long round trip of 10 shares bought
at 100 and sold at 110 with $1 commissions must realize exactly $98, an order
queued on bar `t` must fill at bar `t+1`'s open, and walk-forward train/test
windows must never overlap.

## Limitations

- Single instrument per engine instance (the accounting supports several, but
  the demo and strategies are single-symbol).
- Bar-based fills only: no tick data, order-book depth, queue position, or
  partial fills *inside* a bar. Intrabar path is unknown, so stops and limits
  use the conservative "worse price" assumption and never assume favourable
  gaps.
- Orders are day-less: an unfilled order keeps working until filled or the data
  ends (the engine cancels it and reports it in `unfilled_orders`).
- No borrow costs, financing, dividends, splits, taxes, or margin interest.
- Slippage models are heuristic; `k` must be calibrated per market.
- Metrics are computed on the equity curve you pass in; annualisation uses a
  fixed `periods_per_year` rather than calendar time.

## License

MIT - see [LICENSE](LICENSE).
