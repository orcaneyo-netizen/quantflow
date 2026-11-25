"""Tests for bar validation, CSV round-trips and the synthetic generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantflow.data import BarSeries, load_csv, synthetic_bars
from quantflow.events import MarketEvent


def test_csv_roundtrip(tmp_path, make_bars):
    bars = make_bars([100.0, 101.5, 99.25, 103.0, 102.0])
    path = tmp_path / "bars.csv"
    bars.data.to_csv(path)

    loaded = load_csv(path, symbol="ROUND")
    assert loaded.symbol == "ROUND"
    assert list(loaded.data.columns) == ["open", "high", "low", "close", "volume"]
    assert loaded.timestamps.equals(bars.timestamps)
    np.testing.assert_allclose(loaded.data.to_numpy(), bars.data.to_numpy())


def test_ohlc_sanity_rejected(make_bars):
    # high below low
    with pytest.raises(ValueError, match="high must be >= low"):
        make_bars([100.0, 100.0], highs=[101.0, 99.0], lows=[99.5, 100.5])

    # high below the close
    with pytest.raises(ValueError, match="high must be >= max"):
        make_bars([100.0, 100.0], highs=[100.5, 99.0], lows=[99.0, 99.0])

    # low above the open
    with pytest.raises(ValueError, match="low must be <= min"):
        make_bars([100.0, 100.0], highs=[101.0, 101.0], lows=[100.5, 99.0])

    # NaN in OHLC
    with pytest.raises(ValueError, match="NaN"):
        make_bars([100.0, np.nan])

    # non-positive price
    with pytest.raises(ValueError, match="positive"):
        make_bars([0.0, 1.0])


def test_bad_index_rejected(make_bars):
    bars = make_bars([100.0, 101.0, 102.0])
    shuffled = bars.data.iloc[[0, 2, 1]]
    with pytest.raises(ValueError, match="ascending"):
        BarSeries(shuffled)

    tz_aware = bars.data.copy()
    tz_aware.index = tz_aware.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="tz-naive"):
        BarSeries(tz_aware)

    with pytest.raises(ValueError, match="at least one bar"):
        BarSeries(bars.data.iloc[:0])

    with pytest.raises(TypeError):
        BarSeries([1.0, 2.0])  # type: ignore[arg-type]


def test_synthetic_determinism():
    a = synthetic_bars(n=100, seed=42)
    b = synthetic_bars(n=100, seed=42)
    c = synthetic_bars(n=100, seed=43)

    np.testing.assert_array_equal(a.data.to_numpy(), b.data.to_numpy())
    assert not np.array_equal(a.data["close"].to_numpy(), c.data["close"].to_numpy())


def test_synthetic_invariants():
    bars = synthetic_bars(n=250, seed=7, start_price=50.0, freq="1h")
    df = bars.data

    assert len(bars) == 250
    assert df.index.tz is None
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[0] == pytest.approx(50.0)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["low"] > 0).all()
    assert (df["volume"] > 0).all()


def test_synthetic_rejects_bad_parameters():
    with pytest.raises(ValueError, match="n must be"):
        synthetic_bars(n=0)
    with pytest.raises(ValueError, match="start_price"):
        synthetic_bars(n=10, start_price=-1.0)
    with pytest.raises(ValueError, match="vol"):
        synthetic_bars(n=10, vol=-0.01)


def test_iter_market_events():
    bars = synthetic_bars(n=20, seed=1)
    events = list(bars.iter_market_events())

    assert len(events) == 20
    assert all(isinstance(e, MarketEvent) for e in events)
    assert events[0].timestamp == bars.timestamps[0].to_pydatetime()
    np.testing.assert_allclose(
        [e.close for e in events], bars.data["close"].to_numpy()
    )
    # sequence ids are strictly increasing
    assert [e.seq for e in events] == sorted({e.seq for e in events})


def test_slice(make_bars):
    bars = make_bars(np.arange(100.0, 110.0))
    part = bars.slice(2, 5)
    assert len(part) == 3
    assert part.timestamps[0] == bars.timestamps[2]
    assert part.data["close"].iloc[0] == 102.0
    with pytest.raises(ValueError):
        bars.slice(5, 5)
    with pytest.raises(ValueError):
        bars.slice(0, 100)


def test_load_csv_missing_column(tmp_path):
    df = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]},
        index=pd.date_range("2020-01-01", periods=1),
    )
    path = tmp_path / "bad.csv"
    df.to_csv(path)
    with pytest.raises(ValueError, match="Missing required column"):
        load_csv(path)
