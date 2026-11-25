"""Tests for the performance metrics on hand-computable equity curves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantflow.metrics import drawdown_series, performance


def curve(values, freq="1D"):
    idx = pd.date_range("2020-01-01", periods=len(values), freq=freq)
    return pd.Series(np.asarray(values, dtype=float), index=idx, name="equity")


def test_hand_computed_curve():
    # returns: +10%, -10%, +21.2121%
    equity = curve([100.0, 110.0, 99.0, 120.0])
    p = performance(equity, periods_per_year=252)

    assert p.total_return == pytest.approx(0.20)
    # drawdown: 110 -> 99 is exactly 10%
    assert p.max_drawdown == pytest.approx(0.10)
    assert p.max_drawdown_duration == 1
    assert p.n_periods == 3
    assert p.n_trades == 0
    assert p.hit_rate is None

    r = np.array([0.10, -0.10, 120.0 / 99.0 - 1.0])
    assert p.annualized_vol == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    expected_sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert p.sharpe == pytest.approx(expected_sharpe)
    assert p.sharpe > 0
    assert p.sortino > 0
    assert p.calmar == pytest.approx(p.cagr / p.max_drawdown)

    # q05 of [-0.10, 0.10, 0.2121] with linear interpolation = -0.08
    assert p.var_95 == pytest.approx(0.08)
    assert p.cvar_95 == pytest.approx(0.10)  # mean of the <= q05 tail = -0.10
    assert p.best_period == pytest.approx(120.0 / 99.0 - 1.0)
    assert p.worst_period == pytest.approx(-0.10)


def test_declining_curve_negative_sharpe():
    p = performance(curve([100.0, 90.0, 80.0, 70.0]))
    assert p.total_return == pytest.approx(-0.30)
    assert p.sharpe < 0
    assert p.sortino < 0
    assert p.max_drawdown == pytest.approx(0.30)
    assert p.max_drawdown_duration == 3
    r = np.array([-0.10, 80.0 / 90.0 - 1.0, 70.0 / 80.0 - 1.0])
    q05 = np.quantile(r, 0.05)
    assert p.var_95 == pytest.approx(-q05)
    assert p.cvar_95 == pytest.approx(-r[r <= q05].mean())


def test_cagr_matches_compound_growth():
    # exactly doubles over 252 bars (one year) with a little noise so the
    # return variance is non-zero
    n = 252
    base = 100.0 * 2.0 ** (np.arange(n + 1) / n)
    values = base * np.where(np.arange(n + 1) % 2 == 0, 1.0, 1.001)
    p = performance(curve(values), periods_per_year=252)
    assert p.total_return == pytest.approx(1.0, rel=1e-12)
    assert p.cagr == pytest.approx(1.0, rel=1e-9)


def test_trade_statistics():
    p = performance(curve([100.0, 101.0, 100.5, 102.0]), trade_pnls=[10.0, -5.0, 20.0])
    assert p.n_trades == 3
    assert p.hit_rate == pytest.approx(2 / 3)
    assert p.avg_win == pytest.approx(15.0)
    assert p.avg_loss == pytest.approx(-5.0)
    assert p.profit_factor == pytest.approx(6.0)  # 30 / 5


def test_all_winners_profit_factor_is_infinite():
    p = performance(curve([100.0, 101.0, 102.0]), trade_pnls=[1.0, 2.0])
    assert p.hit_rate == pytest.approx(1.0)
    assert np.isinf(p.profit_factor)


def test_exposure_and_turnover_roundtrip():
    p = performance(curve([100.0, 101.0, 102.0]), exposure=0.5, turnover=2.5)
    assert p.exposure == 0.5
    assert p.turnover == 2.5
    with pytest.raises(ValueError):
        performance(curve([100.0, 101.0, 102.0]), exposure=1.5)
    with pytest.raises(ValueError):
        performance(curve([100.0, 101.0, 102.0]), turnover=-1.0)


def test_risk_free_rate_lowers_sharpe():
    equity = curve([100.0, 110.0, 99.0, 120.0])
    no_rf = performance(equity, rf=0.0)
    with_rf = performance(equity, rf=0.10)
    assert with_rf.sharpe < no_rf.sharpe


def test_zero_variance_is_rejected_by_default():
    flat = curve([100.0, 100.0, 100.0, 100.0])
    with pytest.raises(ValueError, match="variance is zero"):
        performance(flat)

    lenient = performance(flat, strict=False)
    assert lenient.sharpe == 0.0
    assert lenient.sortino == 0.0
    assert lenient.annualized_vol == 0.0
    assert lenient.max_drawdown == 0.0
    assert lenient.calmar == 0.0


def test_degenerate_inputs_raise():
    with pytest.raises(ValueError, match="empty"):
        performance(pd.Series(dtype=float))
    with pytest.raises(ValueError, match="at least 3"):
        performance(curve([100.0, 101.0]))
    with pytest.raises(ValueError, match="NaN or infinite"):
        performance(curve([100.0, np.nan, 101.0]))
    with pytest.raises(ValueError, match="initial equity"):
        performance(curve([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="stay positive"):
        performance(curve([100.0, 0.0, 101.0]))
    with pytest.raises(ValueError, match="periods_per_year"):
        performance(curve([100.0, 101.0, 102.0]), periods_per_year=0)
    with pytest.raises(TypeError):
        performance([100.0, 101.0, 102.0])  # type: ignore[arg-type]


def test_total_loss_cagr_is_minus_one():
    p = performance(curve([100.0, 50.0, 0.0]))
    assert p.total_return == pytest.approx(-1.0)
    assert p.cagr == -1.0
    assert p.max_drawdown == pytest.approx(1.0)


def test_drawdown_series():
    dd = drawdown_series(curve([100.0, 110.0, 99.0, 120.0]))
    np.testing.assert_allclose(dd.to_numpy(), [0.0, 0.0, -0.1, 0.0])


def test_summary_and_to_dict():
    p = performance(curve([100.0, 110.0, 99.0, 120.0]))
    text = p.summary()
    assert "max_drawdown" in text and "sharpe" in text
    d = p.to_dict()
    assert d["max_drawdown"] == pytest.approx(0.10)
    assert set(d) == set(p.__dataclass_fields__)
