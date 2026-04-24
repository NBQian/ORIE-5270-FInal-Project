"""Parallel multi-strategy backtest comparison."""
import numpy as np
from multiprocessing import Pool
from .strategy import backtest_long_short
from .risk import summary_stats


# Registry of named presets
PRESETS = {
    "fast_mom":    {"lookback": 24,  "hold": 6,  "top_frac": 0.3},
    "med_mom":     {"lookback": 168, "hold": 24, "top_frac": 0.3},
    "slow_mom":    {"lookback": 336, "hold": 72, "top_frac": 0.3},
    "aggressive":  {"lookback": 72,  "hold": 6,  "top_frac": 0.2},
}


def _run_preset(args):
    """Worker: run a single named preset."""
    name, prices, params, cost_bps = args
    bt = backtest_long_short(prices, **params, cost_bps=cost_bps)
    stats = summary_stats(bt["net"].values, bt["equity"].values)
    return name, {"result": bt, "metrics": stats, "params": params}


def parallel_preset_backtest(prices, presets=None, cost_bps=5, n_workers=4):
    """Run multiple named strategy presets in parallel.

    Parameters
    ----------
    prices : pd.DataFrame
        Close prices.
    presets : dict or None
        Dict of {name: {lookback, hold, top_frac}}. Defaults to PRESETS.
    cost_bps : float
        Transaction cost.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    dict
        {name: {"result": DataFrame, "metrics": dict, "params": dict}}
    """
    presets = presets or PRESETS
    args = [(name, prices, params, cost_bps)
            for name, params in presets.items()]
    with Pool(n_workers) as pool:
        return dict(pool.map(_run_preset, args))


def benchmark_buy_hold(prices, asset="BTCUSDT"):
    """Buy-and-hold benchmark for a single asset.

    Returns equity curve and summary stats for comparison.
    """
    rets = prices[asset].pct_change().dropna()
    equity = (1 + rets).cumprod()
    stats = summary_stats(rets.values, equity.values)
    return {"equity": equity, "metrics": stats}
