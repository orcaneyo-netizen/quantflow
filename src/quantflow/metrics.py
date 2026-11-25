"""Performance statistics for an equity curve.

All formulas
------------
Let ``E_0 .. E_n`` be the equity curve and ``r_t = E_t / E_{t-1} - 1`` the
simple per-period return (``n`` returns, ``ppy`` periods per year).

======================  =====================================================
total_return            ``E_n / E_0 - 1``
CAGR                    ``(E_n / E_0) ** (ppy / n) - 1``; ``-1`` if ``E_n <= 0``
annualized_vol          ``std(r, ddof=1) * sqrt(ppy)``
Sharpe                  ``mean(r - rf/ppy) / std(r, ddof=1) * sqrt(ppy)``
Sortino                 ``mean(r - rf/ppy) / dd * sqrt(ppy)`` with
                        ``dd = sqrt(mean(min(r - rf/ppy, 0)^2))`` over all n
Calmar                  ``CAGR / max_drawdown``
max_drawdown            ``-min(E_t / cummax(E)_t - 1)`` (positive fraction)
max_drawdown_duration   longest run of bars strictly below the running peak
hit_rate                share of round-trip trades with PnL > 0
avg_win / avg_loss      mean winning / losing round-trip PnL (loss < 0)
profit_factor           ``sum(wins) / abs(sum(losses))``
exposure                fraction of bars holding a non-zero position
turnover                traded notional / mean equity (whole period)
VaR95 / CVaR95          ``-quantile_5%(r)`` and ``-mean(r | r <= q05)``,
                        reported as positive loss fractions
======================  =====================================================

NaN policy: equity containing NaN/inf, a non-positive starting value, an
intermediate non-positive value, or fewer than three points raises
``ValueError``. A terminal equity of zero or below yields ``CAGR = -1``
(``-100%``) rather than NaN. Ratio metrics are ``inf`` only where the
denominator is genuinely zero while the numerator is positive (e.g. Calmar
with no drawdown); this is documented rather than silently replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Performance:
    """Container for all performance statistics of one equity curve.

    See the module docstring for the formula behind every field. Optional
    trade-based fields are ``None`` when the caller did not supply trades.
    """

    total_return: float
    cagr: float
    annualized_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    best_period: float
    worst_period: float
    hit_rate: float | None
    avg_win: float | None
    avg_loss: float | None
    profit_factor: float | None
    exposure: float | None
    turnover: float | None
    n_periods: int
    n_trades: int
    start: pd.Timestamp
    end: pd.Timestamp

    def to_dict(self) -> dict[str, float | int | None]:
        """Return the metrics as a plain dict (handy for DataFrames/JSON)."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def summary(self) -> str:
        """Human-readable, aligned multi-line summary."""
        pct = {
            "total_return",
            "cagr",
            "annualized_vol",
            "max_drawdown",
            "var_95",
            "cvar_95",
            "best_period",
            "worst_period",
            "hit_rate",
            "exposure",
        }
        rows: list[tuple[str, str]] = []
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in ("start", "end"):
                text = pd.Timestamp(value).isoformat()
            elif value is None:
                text = "n/a"
            elif name in pct:
                text = f"{value:.2%}"
            elif name == "profit_factor":
                text = f"{value:.3f}"
            elif name == "turnover":
                text = f"{value:.3f}x"
            elif isinstance(value, int):
                text = str(value)
            else:
                text = f"{value:.4f}"
            rows.append((name, text))
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return self.summary()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Return the drawdown path ``E_t / cummax(E)_t - 1`` (<= 0)."""
    equity = _validate_equity(equity)
    return equity / equity.cummax() - 1.0


def _validate_equity(equity: pd.Series) -> pd.Series:
    """Validate and normalise an equity curve."""
    if not isinstance(equity, pd.Series):
        raise TypeError(f"equity must be a pandas Series, got {type(equity)}")
    if equity.empty:
        raise ValueError("equity series is empty")
    if len(equity) < 3:
        raise ValueError(
            f"equity series needs at least 3 points (2 returns), got {len(equity)}"
        )
    values = equity.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("equity contains NaN or infinite values")
    if values[0] <= 0:
        raise ValueError(f"initial equity must be positive, got {values[0]}")
    if np.any(values[:-1] <= 0):
        bad = int(np.flatnonzero(values[:-1] <= 0)[0])
        raise ValueError(f"equity must stay positive; found {values[bad]} at position {bad}")
    return equity.astype(float)


def _max_drawdown_duration(equity: pd.Series) -> int:
    """Longest consecutive run of bars strictly below the running peak."""
    below = (equity < equity.cummax()).to_numpy()
    best = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        best = max(best, current)
    return int(best)


def performance(
    equity: pd.Series,
    periods_per_year: int = 252,
    rf: float = 0.0,
    *,
    trade_pnls: Sequence[float] | None = None,
    exposure: float | None = None,
    turnover: float | None = None,
    strict: bool = True,
) -> Performance:
    """Compute the full metric set for an equity curve.

    Args:
        equity: Equity values indexed by timestamp (at least 3 points).
        periods_per_year: Bars per year used to annualise (252 daily, 8760
            hourly, ...). Must be positive.
        rf: Annual risk-free rate, quoted as a decimal (0.02 = 2%).
        trade_pnls: Realized PnL of each closed round trip, in currency units.
            When omitted the trade-based metrics are ``None``.
        exposure: Fraction of bars with a non-zero position, in ``[0, 1]``.
        turnover: Traded notional divided by mean equity (whole period).
        strict: When True (default), a zero-variance return series raises
            ``ValueError`` because Sharpe/Sortino are undefined. When False the
            ratio metrics are reported as 0.0 instead (used for quiet
            walk-forward folds).

    Returns:
        A frozen :class:`Performance` instance.

    Raises:
        ValueError: On empty/short series, non-finite or non-positive equity,
            a non-positive ``periods_per_year``, out-of-range exposure/turnover,
            or (strict only) zero return variance.
    """
    equity = _validate_equity(equity)
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    if exposure is not None and not 0.0 <= exposure <= 1.0:
        raise ValueError(f"exposure must be in [0, 1], got {exposure}")
    if turnover is not None and turnover < 0.0:
        raise ValueError(f"turnover must be non-negative, got {turnover}")

    values = equity.to_numpy(dtype=float)
    returns = pd.Series(values[1:] / values[:-1] - 1.0)
    n = len(returns)

    total_return = float(values[-1] / values[0] - 1.0)
    if values[-1] <= 0:
        cagr = -1.0  # total loss: a real growth rate does not exist
    else:
        cagr = float((values[-1] / values[0]) ** (periods_per_year / n) - 1.0)

    vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    if not np.isfinite(vol) or vol == 0.0:
        if strict:
            raise ValueError(
                "return variance is zero: Sharpe/Sortino are undefined "
                "(pass strict=False to report 0.0)"
            )
        vol = 0.0
        sharpe = 0.0
        sortino = 0.0
    else:
        rf_per_period = rf / periods_per_year
        excess = returns - rf_per_period
        mean_excess = float(excess.mean())
        sharpe = float(mean_excess / returns.std(ddof=1) * np.sqrt(periods_per_year))

        downside = np.minimum(excess.to_numpy(), 0.0)
        downside_dev = float(np.sqrt(np.mean(downside**2)))
        if downside_dev == 0.0:
            sortino = 0.0 if mean_excess == 0.0 else float(np.copysign(np.inf, mean_excess))
        else:
            sortino = float(mean_excess / downside_dev * np.sqrt(periods_per_year))

    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    max_dd = float(-drawdown.min())
    if max_dd == 0.0:
        calmar = 0.0 if cagr <= 0 else float(np.inf)
    else:
        calmar = float(cagr / max_dd)

    q05 = float(np.quantile(returns.to_numpy(), 0.05, method="linear"))
    tail = returns[returns <= q05]
    var_95 = float(-q05)
    cvar_95 = float(-tail.mean()) if len(tail) else var_95

    hit_rate = avg_win = avg_loss = profit_factor = None
    n_trades = 0
    if trade_pnls is not None:
        pnls = np.asarray(list(trade_pnls), dtype=float)
        if pnls.ndim != 1:
            raise ValueError("trade_pnls must be one-dimensional")
        if not np.all(np.isfinite(pnls)):
            raise ValueError("trade_pnls contains NaN or infinite values")
        n_trades = int(len(pnls))
        if n_trades:
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            hit_rate = float(len(wins) / n_trades)
            avg_win = float(wins.mean()) if len(wins) else 0.0
            avg_loss = float(losses.mean()) if len(losses) else 0.0
            gross_profit = float(wins.sum())
            gross_loss = float(-losses.sum())
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = float(np.inf) if gross_profit > 0 else 0.0

    return Performance(
        total_return=total_return,
        cagr=cagr,
        annualized_vol=vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=_max_drawdown_duration(equity),
        var_95=var_95,
        cvar_95=cvar_95,
        best_period=float(returns.max()),
        worst_period=float(returns.min()),
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        exposure=exposure,
        turnover=turnover,
        n_periods=n,
        n_trades=n_trades,
        start=pd.Timestamp(equity.index[0]),
        end=pd.Timestamp(equity.index[-1]),
    )
