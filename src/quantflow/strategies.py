"""Reference strategies used to exercise the engine.

Both strategies are stateless with respect to the engine: they infer their
current exposure from ``ctx.position`` and only trade when the *signal* flips,
so a slow drift in price does not trigger constant rebalancing.

Position sizing
---------------
* :class:`SmaCross` targets ``weight * equity / close`` shares (a constant
  fraction of equity), leaving a small cash buffer for fees.
* :class:`DonchianBreakout` risks ``risk_frac`` of equity per trade against an
  ATR-based stop distance:

      qty = risk_frac * equity / (atr_mult * ATR)

  so a stop-out at ``entry - atr_mult * ATR`` loses roughly ``risk_frac`` of
  equity (before costs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantflow.engine import Context, Order


@dataclass
class BuyAndHold:
    """Buy ``weight * equity / close`` shares on the first eligible bar, hold.

    Args:
        weight: Fraction of equity to invest, in ``(0, 1]``.
        buffer: Cash fraction held back for fees, in ``[0, 1)``.
    """

    weight: float = 1.0
    buffer: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.weight <= 1.0:
            raise ValueError(f"weight must be in (0, 1], got {self.weight}")
        if not 0.0 <= self.buffer < 1.0:
            raise ValueError(f"buffer must be in [0, 1), got {self.buffer}")

    def on_bar(self, ctx: Context) -> list[Order]:
        """Enter once, then do nothing."""
        if ctx.position != 0.0:
            return []
        budget = self.weight * (1.0 - self.buffer) * ctx.cash
        qty = math.floor(budget / ctx.close)
        if qty <= 0:
            return []
        return [ctx.order_market(qty, tag="buy_and_hold")]


@dataclass
class SmaCross:
    """Long/flat (or long/short) moving-average crossover.

    Signal at bar ``t`` (using closes up to and including ``t``):

        fast_ma = mean(close[t-fast+1 : t+1])
        slow_ma = mean(close[t-slow+1 : t+1])
        target  = +1 if fast_ma > slow_ma else (-1 if allow_short else 0)

    A trade is sent only when ``sign(position)`` differs from the target, and
    the target size is ``floor(weight * equity * (1-buffer) / close)``.

    Args:
        fast: Fast SMA window in bars.
        slow: Slow SMA window in bars (must exceed ``fast``).
        weight: Fraction of equity deployed when long/short.
        buffer: Cash held back for commissions and slippage.
        allow_short: Whether a bearish cross may open a short.
    """

    fast: int = 20
    slow: int = 50
    weight: float = 1.0
    buffer: float = 0.01
    allow_short: bool = False

    def __post_init__(self) -> None:
        if self.fast < 1:
            raise ValueError(f"fast must be >= 1, got {self.fast}")
        if self.slow <= self.fast:
            raise ValueError(f"slow ({self.slow}) must exceed fast ({self.fast})")
        if not 0.0 < self.weight <= 1.0:
            raise ValueError(f"weight must be in (0, 1], got {self.weight}")
        if not 0.0 <= self.buffer < 1.0:
            raise ValueError(f"buffer must be in [0, 1), got {self.buffer}")

    def on_bar(self, ctx: Context) -> list[Order]:
        """Return at most one market order."""
        closes = ctx.closes
        if len(closes) < self.slow:
            return []

        fast_ma = float(closes.iloc[-self.fast:].mean())
        slow_ma = float(closes.iloc[-self.slow:].mean())
        if fast_ma > slow_ma:
            target = 1
        elif self.allow_short:
            target = -1
        else:
            target = 0

        current = 1 if ctx.position > 0 else (-1 if ctx.position < 0 else 0)
        if target == current:
            return []

        if target == 0:
            return [ctx.order_market(-ctx.position, tag="sma_exit")]

        budget = self.weight * (1.0 - self.buffer) * ctx.equity
        qty = math.floor(budget / ctx.close)
        if qty <= 0:
            return []
        desired = target * qty
        return [ctx.target_position(desired, tag="sma_entry")]


@dataclass
class DonchianBreakout:
    """Donchian channel breakout with ATR-based position sizing.

    Channel (computed on bars *before* the current one, so the breakout is a
    genuine new high/low):

        upper = max(high[t-window : t])
        lower = min(low[t-window : t])

    Rules: close above ``upper`` -> long; close below ``lower`` -> short (only
    if ``allow_short``); a long exits when close falls below ``lower``, a short
    exits when close rises above ``upper``. Size is

        qty = floor(risk_frac * equity / (atr_mult * ATR))

    capped so a long never spends more than the available cash.

    Args:
        window: Donchian lookback in bars (>= 2).
        atr_window: ATR lookback in bars (>= 1).
        risk_frac: Fraction of equity risked between entry and stop.
        atr_mult: Stop distance in ATRs.
        allow_short: Whether downside breakouts may open shorts.
    """

    window: int = 20
    atr_window: int = 14
    risk_frac: float = 0.01
    atr_mult: float = 2.0
    allow_short: bool = False

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        if self.atr_window < 1:
            raise ValueError(f"atr_window must be >= 1, got {self.atr_window}")
        if not 0.0 < self.risk_frac <= 1.0:
            raise ValueError(f"risk_frac must be in (0, 1], got {self.risk_frac}")
        if self.atr_mult <= 0:
            raise ValueError(f"atr_mult must be positive, got {self.atr_mult}")

    def on_bar(self, ctx: Context) -> list[Order]:
        """Return at most one market order."""
        hist = ctx.history
        need = max(self.window, self.atr_window) + 1
        if len(hist) < need:
            return []

        prior = hist.iloc[:-1]
        upper = float(prior["high"].iloc[-self.window:].max())
        lower = float(prior["low"].iloc[-self.window:].min())
        atr = self._atr(hist)
        close = ctx.close

        if ctx.position > 0 and close < lower:
            return [ctx.order_market(-ctx.position, tag="donchian_exit")]
        if ctx.position < 0 and close > upper:
            return [ctx.order_market(-ctx.position, tag="donchian_exit")]
        if ctx.position != 0:
            return []

        if close > upper:
            target = 1
        elif close < lower and self.allow_short:
            target = -1
        else:
            return []

        if atr <= 0.0:
            return []
        qty = math.floor(self.risk_frac * ctx.equity / (self.atr_mult * atr))
        if target > 0:
            qty = min(qty, math.floor(0.99 * ctx.cash / close))
        if qty <= 0:
            return []
        return [ctx.target_position(target * qty, tag="donchian_entry")]

    def _atr(self, hist) -> float:
        """Average true range over the last ``atr_window`` bars.

        TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
        """
        h = hist["high"].to_numpy(dtype=float)
        l = hist["low"].to_numpy(dtype=float)
        c = hist["close"].to_numpy(dtype=float)
        prev_close = np.concatenate(([c[0]], c[:-1]))
        tr = np.maximum.reduce(
            [h - l, np.abs(h - prev_close), np.abs(l - prev_close)]
        )
        window = tr[-self.atr_window:]
        return float(window.mean())
