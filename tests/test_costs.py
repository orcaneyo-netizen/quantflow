"""Tests for commission and slippage models."""

from __future__ import annotations

import pytest

from quantflow.costs import (
    CostConfig,
    FixedSlippage,
    PerShareCommission,
    SquareRootImpactSlippage,
)


def test_commission_minimum_binds_for_small_trades():
    model = PerShareCommission(fixed_per_share=0.005, pct_notional=0.0001, min_commission=1.0)
    # 10 * 0.005 + 10*10*0.0001 = 0.05 + 0.01 = 0.06 -> minimum 1.00
    assert model.calculate(quantity=10, price=10.0) == pytest.approx(1.0)


def test_commission_formula_above_minimum():
    model = PerShareCommission(fixed_per_share=0.005, pct_notional=0.0001, min_commission=1.0)
    # 1000 * 0.005 + 1000*100*0.0001 = 5 + 10 = 15
    assert model.calculate(quantity=1000, price=100.0) == pytest.approx(15.0)

    pct_only = PerShareCommission(fixed_per_share=0.0, pct_notional=0.001, min_commission=0.0)
    assert pct_only.calculate(quantity=100, price=50.0) == pytest.approx(5.0)


def test_commission_rejects_bad_inputs():
    model = PerShareCommission()
    with pytest.raises(ValueError):
        model.calculate(quantity=0, price=10.0)
    with pytest.raises(ValueError):
        model.calculate(quantity=10, price=-1.0)


def test_fixed_slippage_bps_math():
    model = FixedSlippage(bps=10.0)
    # 10 bps of 100 = 0.10; 10 bps of 250 = 0.25
    assert model.calculate(quantity=1, price=100.0, volume=1e6) == pytest.approx(0.10)
    assert model.calculate(quantity=999, price=250.0, volume=1e6) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        model.calculate(quantity=1, price=0.0, volume=1e6)


def test_sqrt_impact_formula():
    model = SquareRootImpactSlippage(bps=1.0, k=0.1)
    qty, price, adv = 1_000.0, 100.0, 1_000_000.0
    participation = qty / adv  # 0.001
    expected = price * (1.0 / 10_000 + 0.1 * participation**0.5)
    assert model.calculate(qty, price, volume=adv, adv=adv) == pytest.approx(expected)
    # 0.1 * sqrt(0.001) = 0.316% -> 0.3162 on a 100 price
    assert model.calculate(qty, price, volume=adv, adv=adv) == pytest.approx(0.3262277)


def test_impact_is_monotonic_in_quantity():
    model = SquareRootImpactSlippage(bps=0.0, k=0.2)
    adv = 1_000_000.0
    slips = [model.calculate(q, 100.0, volume=adv, adv=adv) for q in (100, 1_000, 10_000, 100_000)]
    assert slips == sorted(slips)
    assert slips[0] < slips[-1]


def test_impact_participation_capped_at_one():
    model = SquareRootImpactSlippage(bps=0.0, k=0.1)
    adv = 1_000_000.0
    capped = model.calculate(5_000_000.0, 100.0, volume=adv, adv=adv)
    assert capped == pytest.approx(100.0 * 0.1 * 1.0)


def test_impact_rejects_zero_volume():
    model = SquareRootImpactSlippage()
    with pytest.raises(ValueError):
        model.calculate(10.0, 100.0, volume=0.0, adv=None)
    with pytest.raises(ValueError):
        model.calculate(0.0, 100.0, volume=1e6)


def test_cost_config_total_cost():
    cfg = CostConfig(
        commission=PerShareCommission(fixed_per_share=0.01, pct_notional=0.0, min_commission=0.0),
        slippage=FixedSlippage(bps=5.0),
    )
    commission, slippage = cfg.total_cost(quantity=100, price=200.0, volume=1e6)
    assert commission == pytest.approx(1.0)      # 100 * 0.01
    assert slippage == pytest.approx(0.10)       # 200 * 5 / 10000
