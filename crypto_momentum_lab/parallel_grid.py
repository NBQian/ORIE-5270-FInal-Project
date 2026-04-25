"""Parallel grid search over strategy hyperparameters."""
import itertools
import pandas as pd
from multiprocessing import Pool
from .strategy import backtest_long_short
from .risk import summary_stats


def _eval_single(args):
    """Worker function: run one backtest config and return metrics."""
    prices, lookback, hold, top_frac, cost_bps = args
    bt = backtest_long_short(prices, lookback, hold, top_frac, cost_bps)
    stats = summary_stats(bt["net"].values, bt["equity"].values)
    stats.update({"lookback": lookback, "hold": hold, "top_frac": top_frac})
    return stats


def parallel_grid_search(prices, lookbacks, holds, top_fracs,
                          cost_bps=5, n_workers=4):
    """Search over (lookback, hold, top_frac) in parallel.

    Parameters
    ----------
    prices : pd.DataFrame
        Close prices for the asset universe.
    lookbacks, holds, top_fracs : list
        Parameter values to search over.
    cost_bps : float
        Transaction cost in basis points.
    n_workers : int
        Number of parallel worker processes.

    Returns
    -------
    pd.DataFrame
        One row per config, sorted by Sharpe descending.
    """
    grid = list(itertools.product(lookbacks, holds, top_fracs))
    args = [(prices, lb, h, tf, cost_bps) for lb, h, tf in grid]
    with Pool(n_workers) as pool:
        results = pool.map(_eval_single, args)
    return pd.DataFrame(results).sort_values("sharpe", ascending=False)
