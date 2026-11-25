"""Command-line interface: ``python -m quantflow <command>``.

Commands
--------
demo          synthetic bars -> SmaCross -> metrics + first trades
walkforward   rolling out-of-sample evaluation on synthetic bars
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quantflow import (
    BacktestEngine,
    BrokerConfig,
    MarketEvent,
    SmaCross,
    performance,
    synthetic_bars,
    walk_forward,
)
from quantflow.costs import CostConfig, FixedSlippage, PerShareCommission


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantflow",
        description="Event-driven backtesting engine (offline, deterministic).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run a synthetic SmaCross backtest")
    demo.add_argument("--bars", type=int, default=1000, help="number of synthetic bars")
    demo.add_argument("--seed", type=int, default=42, help="random seed")
    demo.add_argument("--fast", type=int, default=20, help="fast SMA window")
    demo.add_argument("--slow", type=int, default=50, help="slow SMA window")
    demo.add_argument("--cash", type=float, default=100_000.0, help="initial cash")
    demo.add_argument("--commission-bps", type=float, default=1.0, help="commission in bps")
    demo.add_argument("--slippage-bps", type=float, default=1.0, help="slippage in bps")
    demo.add_argument("--plot", action="store_true", help="save equity.png instead of plotting")
    demo.add_argument("--out", type=Path, default=Path("equity.png"), help="plot output path")

    wf = sub.add_parser("walkforward", help="rolling walk-forward on synthetic bars")
    wf.add_argument("--bars", type=int, default=1500)
    wf.add_argument("--seed", type=int, default=42)
    wf.add_argument("--splits", type=int, default=4)
    wf.add_argument("--train", type=int, default=252)
    wf.add_argument("--test", type=int, default=63)
    wf.add_argument("--embargo", type=int, default=5)
    return parser


def _cost_config(commission_bps: float, slippage_bps: float) -> CostConfig:
    return CostConfig(
        commission=PerShareCommission(
            fixed_per_share=0.0,
            pct_notional=commission_bps / 10_000.0,
            min_commission=0.0,
        ),
        slippage=FixedSlippage(bps=slippage_bps),
    )


def _cmd_demo(args: argparse.Namespace) -> int:
    bars = synthetic_bars(n=args.bars, seed=args.seed, freq="1D")
    broker_cfg = BrokerConfig(
        costs=_cost_config(args.commission_bps, args.slippage_bps),
        max_participation=1.0,
        allow_short=False,
    )
    strategy = SmaCross(fast=args.fast, slow=args.slow)
    result = BacktestEngine(
        strategy=strategy,
        data=bars,
        broker_cfg=broker_cfg,
        initial_cash=args.cash,
        periods_per_year=252,
    ).run()

    print(f"quantflow {__import__('quantflow').__version__} - synthetic SmaCross demo")
    print(
        f"bars={len(bars)}  {bars.timestamps[0].date()}..{bars.timestamps[-1].date()}  "
        f"seed={args.seed}  SmaCross({args.fast},{args.slow})  "
        f"cash={args.cash:,.0f}"
    )
    print(
        f"costs: commission={args.commission_bps}bps  slippage={args.slippage_bps}bps  "
        f"max_participation=100% of bar volume"
    )
    print()
    print("=== performance ===")
    print(result.performance.summary())
    print()
    print(f"=== first 10 of {len(result.trades)} fills ===")
    if result.trades.empty:
        print("(no trades)")
    else:
        print(result.trades.head(10).to_string(index=False))
    print()
    print(
        f"events: {len(result.events)} total "
        f"({len(result.events.filter(MarketEvent))} market bars, "
        f"{len(result.trades)} fills)"
    )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(bars.close_prices.index, bars.close_prices.to_numpy(), lw=1.0)
        ax1.set_title(f"close - synthetic seed={args.seed}")
        ax1.set_ylabel("price")
        ax2.plot(result.equity.index, result.equity.to_numpy(), lw=1.2, color="C1")
        ax2.set_title("equity")
        ax2.set_ylabel("equity")
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        plt.close(fig)
        print(f"saved plot -> {args.out}")
    return 0


def _cmd_walkforward(args: argparse.Namespace) -> int:
    bars = synthetic_bars(n=args.bars, seed=args.seed, freq="1D")
    result = walk_forward(
        data=bars,
        strategy_factory=lambda: SmaCross(fast=20, slow=50),
        n_splits=args.splits,
        train=args.train,
        test=args.test,
        embargo=args.embargo,
        periods_per_year=252,
    )
    print(f"walk-forward: {args.splits} folds, train={args.train}, test={args.test}, "
          f"embargo={args.embargo}, bars={args.bars}")
    print()
    print(result.to_dataframe().to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print()
    print("=== out-of-sample performance ===")
    print(result.oos_performance.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``quantflow`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "walkforward":
        return _cmd_walkforward(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
