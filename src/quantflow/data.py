"""Data handling for the quantflow backtesting engine.

This module provides the BarSeries dataclass for OHLCV data with validation,
CSV loading, and synthetic data generation using geometric Brownian motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarSeries:
    """Container for validated OHLCV bar data.

    The DataFrame index must be a tz-naive DatetimeIndex, ascending, with no NaN
    in OHLC columns. High >= low, high >= max(open, close), low <= min(open, close).
    Volume must be non-negative.

    Attributes:
        data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
              and a tz-naive DatetimeIndex.
        symbol: Trading symbol.
    """

    data: pd.DataFrame
    symbol: str = "SYM"

    def __post_init__(self) -> None:
        df = self.data
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError("data must contain at least one bar")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DatetimeIndex")
        if df.index.tz is not None:
            raise ValueError("Index must be tz-naive")
        if not df.index.is_monotonic_increasing:
            raise ValueError("Index must be strictly ascending")
        if df.index.has_duplicates:
            raise ValueError("Index must not have duplicates")

        required_cols = {"open", "high", "low", "close", "volume"}
        if set(df.columns) != required_cols:
            raise ValueError(f"Columns must be exactly {required_cols}, got {set(df.columns)}")

        ohlc_cols = ["open", "high", "low", "close"]
        if df[ohlc_cols].isna().any().any():
            raise ValueError("OHLC columns must not contain NaN")

        if (df["high"] < df["low"]).any():
            raise ValueError("high must be >= low for all bars")

        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            raise ValueError("high must be >= max(open, close)")

        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            raise ValueError("low must be <= min(open, close)")

        if (df[ohlc_cols] <= 0).any().any():
            raise ValueError("OHLC prices must be positive")

        if (df["volume"] < 0).any():
            raise ValueError("volume must be non-negative")

        object.__setattr__(self, "data", df.copy())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key) -> pd.Series:
        return self.data[key]

    def __iter__(self) -> Iterator[pd.Series]:
        """Iterate over bars as Series with OHLCV data."""
        for idx, row in self.data.iterrows():
            yield row

    def iter_market_events(self):
        """Yield MarketEvent for each bar."""
        from quantflow.events import MarketEvent

        for ts, row in self.data.iterrows():
            yield MarketEvent(
                timestamp=ts.to_pydatetime(),
                symbol=self.symbol,
                open_=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )

    @property
    def close_prices(self) -> pd.Series:
        """Return close price series."""
        return self.data["close"]

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Return the timestamp index."""
        return self.data.index

    def slice(self, start: int, stop: int) -> "BarSeries":
        """Return a positional slice ``[start, stop)`` as a new BarSeries.

        Used by the walk-forward driver to carve train/test windows without
        copying the whole frame twice.

        Args:
            start: First positional index (inclusive).
            stop: Last positional index (exclusive).

        Returns:
            A new validated BarSeries.

        Raises:
            ValueError: If the slice is empty or out of bounds.
        """
        if start < 0 or stop > len(self.data) or start >= stop:
            raise ValueError(
                f"invalid slice [{start}:{stop}) for {len(self.data)} bars"
            )
        return BarSeries(data=self.data.iloc[start:stop].copy(), symbol=self.symbol)


def load_csv(path: str | Path, symbol: str = "SYM") -> BarSeries:
    """Load OHLCV data from a CSV file.

    Expected CSV format:
    - Index column: datetime (parseable by pandas)
    - Columns: open, high, low, close, volume

    Args:
        path: Path to CSV file.
        symbol: Trading symbol.

    Returns:
        Validated BarSeries object.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df = df[required].astype(float)
    return BarSeries(data=df, symbol=symbol)


def synthetic_bars(
    n: int = 1000,
    seed: int = 42,
    start_price: float = 100.0,
    drift: float = 0.0001,
    vol: float = 0.02,
    freq: str = "1h",
    start: str | datetime = "2020-01-01",
) -> BarSeries:
    """Generate synthetic OHLCV bars using geometric Brownian motion.

    The close price follows the exact GBM discretisation

        log S_t = log S_{t-1} + (drift - vol^2 / 2) + vol * Z_t,  Z_t ~ N(0, 1)

    so that ``drift`` and ``vol`` are the **per-bar** arithmetic drift and
    volatility of the log return (annualise with ``drift * periods`` and
    ``vol * sqrt(periods)``). The first close equals ``start_price`` exactly.
    Open is the previous close plus a small gap; the high/low envelope is
    drawn symmetrically around ``[min(open, close), max(open, close)]`` so the
    OHLC invariants always hold. Volume is lognormal with median ~1e6.

    Args:
        n: Number of bars to generate (must be >= 1).
        seed: Random seed for reproducibility (default 42).
        start_price: Starting price level, must be positive.
        drift: Per-bar mean log return.
        vol: Per-bar standard deviation of the log return, must be >= 0.
        freq: Pandas frequency string for the index (e.g. "1h", "1D").
        start: Start timestamp.

    Returns:
        BarSeries with ``n`` synthetic bars.

    Raises:
        ValueError: If ``n < 1``, ``start_price <= 0`` or ``vol < 0``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if start_price <= 0:
        raise ValueError(f"start_price must be positive, got {start_price}")
    if vol < 0:
        raise ValueError(f"vol must be non-negative, got {vol}")

    rng = np.random.default_rng(seed)

    if isinstance(start, str):
        start = pd.Timestamp(start)

    dt = pd.Timedelta(freq).total_seconds() / (24 * 3600)  # in days

    # Log returns for bars 1..n-1; bar 0 is exactly start_price.
    shocks = rng.normal(drift - 0.5 * vol**2, vol, size=max(n - 1, 0))
    log_prices = np.log(start_price) + np.concatenate(([0.0], np.cumsum(shocks)))
    closes = np.exp(log_prices)

    # Generate open as previous close (with small gap noise)
    opens = np.roll(closes, 1)
    opens[0] = start_price
    # Add small gap noise
    gap_noise = rng.normal(0, vol * 0.1, size=n)
    opens = opens * np.exp(gap_noise)

    # Generate high/low from close with intraday range
    # Use Parkinson's estimator idea: range ~ vol * sqrt(dt) * factor
    intraday_vol = vol * np.sqrt(max(dt, 0.0)) * 0.5
    hl_range = np.abs(rng.normal(0, intraday_vol, size=n)) * closes
    highs = np.maximum(opens, closes) + hl_range
    lows = np.minimum(opens, closes) - hl_range

    # Ensure high >= max(open, close) and low <= min(open, close) and low > 0
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    lows = np.maximum(lows, 1e-8)

    # Volume: lognormal with mean ~ 1e6
    volume = rng.lognormal(13.8, 0.5, size=n).astype(np.int64)

    # Create DataFrame
    index = pd.date_range(start=start, periods=n, freq=freq)
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        },
        index=index,
    )

    return BarSeries(data=df, symbol="SYNTH")