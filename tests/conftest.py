"""Shared fixtures for the quantflow test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantflow.data import BarSeries


@pytest.fixture
def make_bars():
    """Factory building a valid BarSeries from close prices.

    Open defaults to the previous close (first open = first close), high/low
    wrap the bar by ``pad`` and volume defaults to 1e6.
    """

    def _make(
        closes,
        opens=None,
        highs=None,
        lows=None,
        volumes=None,
        freq: str = "1D",
        start: str = "2020-01-01",
        symbol: str = "TEST",
    ) -> BarSeries:
        closes = np.asarray(closes, dtype=float)
        n = len(closes)
        if opens is None:
            opens = np.concatenate(([closes[0]], closes[:-1]))
        opens = np.asarray(opens, dtype=float)
        if highs is None:
            highs = np.maximum(opens, closes) + 0.5
        if lows is None:
            lows = np.minimum(opens, closes) - 0.5
        if volumes is None:
            volumes = np.full(n, 1_000_000.0)
        idx = pd.date_range(start=start, periods=n, freq=freq)
        df = pd.DataFrame(
            {
                "open": opens,
                "high": np.asarray(highs, dtype=float),
                "low": np.asarray(lows, dtype=float),
                "close": closes,
                "volume": np.asarray(volumes, dtype=float),
            },
            index=idx,
        )
        return BarSeries(data=df, symbol=symbol)

    return _make
