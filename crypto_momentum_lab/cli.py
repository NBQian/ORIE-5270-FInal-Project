"""Command-line interface for crypto_momentum_lab."""
import argparse
from .data_loader import load_universe
from .parallel_grid import parallel_grid_search
from .backtest import parallel_preset_backtest, benchmark_buy_hold
from .visualization import metrics_table


def main():
    p = argparse.ArgumentParser(
        description="Parallel cross-sectional crypto momentum backtester",
    )
    sub = p.add_subparsers(dest="command")

    # Grid search subcommand
    gs = sub.add_parser("grid", help="Run parallel grid search")
    gs.add_argument("--start", default="2020-10-01")
    gs.add_argument("--end", default="2026-04-17")
    gs.add_argument("--interval", default="1h")
    gs.add_argument("--n_workers", type=int, default=4)

    # Preset backtest subcommand
    bt = sub.add_parser("backtest", help="Compare preset strategies")
    bt.add_argument("--start", default="2020-10-01")
    bt.add_argument("--end", default="2026-04-17")
    bt.add_argument("--n_workers", type=int, default=4)

    args = p.parse_args()

    if args.command == "grid":
        prices = load_universe(interval=args.interval, start=args.start,
                               end=args.end)
        print(f"Loaded {prices.shape[0]} bars x {prices.shape[1]} symbols")
        res = parallel_grid_search(
            prices,
            lookbacks=[24, 72, 168, 336],
            holds=[6, 24, 72],
            top_fracs=[0.2, 0.3, 0.4],
            n_workers=args.n_workers,
        )
        print(res.head(10).to_string(index=False))

    elif args.command == "backtest":
        prices = load_universe(start=args.start, end=args.end)
        results = parallel_preset_backtest(prices, n_workers=args.n_workers)
        bench = benchmark_buy_hold(prices)
        print(metrics_table(results, benchmark=bench).to_string())

    else:
        p.print_help()


if __name__ == "__main__":
    main()
