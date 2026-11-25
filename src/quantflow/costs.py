"""Cost models for the quantflow backtesting engine.

This module provides CommissionModel and SlippageModel for realistic
transaction cost simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class CommissionModel(Protocol):
    """Protocol for commission calculation."""

    def calculate(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade.

        Args:
            quantity: Number of shares/contracts traded.
            price: Price per share/contract.

        Returns:
            Commission in currency units.
        """
        ...


@runtime_checkable
class SlippageModel(Protocol):
    """Protocol for slippage calculation."""

    def calculate(self, quantity: float, price: float, volume: float, adv: float | None = None) -> float:
        """Calculate slippage for a trade.

        Args:
            quantity: Number of shares/contracts traded.
            price: Reference price (typically mid or close).
            volume: Current bar volume.
            adv: Average daily volume (optional, for market impact).

        Returns:
            Slippage in price units (per share).
        """
        ...


@dataclass(frozen=True)
class PerShareCommission:
    """Per-share commission model with fixed fee + percentage of notional.

    Formula: commission = max(min_commission, quantity * fixed_per_share + notional * pct_notional)

    Args:
        fixed_per_share: Fixed fee per share/contract (e.g., $0.005).
        pct_notional: Percentage of notional value (e.g., 0.0001 for 1 bps).
        min_commission: Minimum commission per trade (e.g., $1.00).
    """

    fixed_per_share: float = 0.005
    pct_notional: float = 0.0001
    min_commission: float = 1.0

    def calculate(self, quantity: float, price: float) -> float:
        """Calculate commission.

        Args:
            quantity: Number of shares.
            price: Price per share.

        Returns:
            Commission amount.
        """
        if quantity <= 0 or price <= 0:
            raise ValueError("quantity and price must be positive")
        notional = quantity * price
        commission = quantity * self.fixed_per_share + notional * self.pct_notional
        return max(self.min_commission, commission)


@dataclass(frozen=True)
class FixedSlippage:
    """Fixed basis points slippage model.

    Slippage = price * bps / 10000

    Args:
        bps: Slippage in basis points (1 bp = 0.01%).
    """

    bps: float = 1.0

    def calculate(self, quantity: float, price: float, volume: float, adv: float | None = None) -> float:
        """Calculate fixed slippage per share.

        Args:
            quantity: Number of shares (unused, kept for interface compatibility).
            price: Reference price.
            volume: Current bar volume (unused).
            adv: Average daily volume (unused).

        Returns:
            Slippage per share in price units.
        """
        if price <= 0:
            raise ValueError("price must be positive")
        return price * self.bps / 10000.0


@dataclass(frozen=True)
class SquareRootImpactSlippage:
    """Slippage model with fixed bps + square-root market impact.

    Formula: slippage = price * (bps / 10000 + k * sqrt(quantity / adv))

    This models the square-root law of market impact where impact scales
    with the square root of participation rate.

    Args:
        bps: Base slippage in basis points.
        k: Market impact coefficient (typical values 0.1-0.3).
        adv_window: Window for computing ADV if not provided externally.
    """

    bps: float = 1.0
    k: float = 0.1
    adv_window: int = 20

    def calculate(self, quantity: float, price: float, volume: float, adv: float | None = None) -> float:
        """Calculate slippage with market impact.

        Args:
            quantity: Number of shares traded.
            price: Reference price.
            volume: Current bar volume (used as ADV fallback).
            adv: Average daily volume (if None, uses current volume).

        Returns:
            Slippage per share in price units.
        """
        if price <= 0:
            raise ValueError("price must be positive")
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        # Use provided ADV or fallback to current volume
        effective_adv = adv if adv is not None and adv > 0 else volume
        if effective_adv <= 0:
            raise ValueError("volume/adv must be positive")

        participation = quantity / effective_adv
        if participation > 1.0:
            # Cap participation at 100% for impact calculation
            participation = 1.0

        impact_bps = self.k * (participation**0.5) * 10000  # Convert to bps
        total_bps = self.bps + impact_bps
        return price * total_bps / 10000.0


@dataclass(frozen=True)
class CostConfig:
    """Combined cost configuration for a backtest.

    Args:
        commission: Commission model instance.
        slippage: Slippage model instance.
    """

    commission: CommissionModel
    slippage: SlippageModel

    def total_cost(self, quantity: float, price: float, volume: float, adv: float | None = None) -> tuple[float, float]:
        """Calculate total commission and slippage for a trade.

        Args:
            quantity: Number of shares.
            price: Execution price.
            volume: Current bar volume.
            adv: Average daily volume (optional).

        Returns:
            Tuple of (commission, slippage_per_share).
        """
        comm = self.commission.calculate(quantity, price)
        slip = self.slippage.calculate(quantity, price, volume, adv)
        return comm, slip