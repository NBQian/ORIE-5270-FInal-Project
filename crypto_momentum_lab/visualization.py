"""Plotting utilities for strategy analysis."""
import numpy as np
import matplotlib.pyplot as plt


def plot_equity_curves(results, benchmark=None, title="Strategy Equity Curves",
                       figsize=(12, 5)):
    """Plot equity curves for multiple strategies.

    Parameters
    ----------
    results : dict
        {name: {"result": DataFrame with "equity" column, ...}}
    benchmark : dict or None
        {"equity": Series} from benchmark_buy_hold.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, data in results.items():
        eq = data["result"]["equity"]
        ax.plot(eq.index, eq.values, label=name, linewidth=1.2)
    if benchmark is not None:
        ax.plot(benchmark["equity"].index, benchmark["equity"].values,
                label="BTC Buy-Hold", linestyle="--", color="gray", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_drawdowns(results, figsize=(12, 4)):
    """Overlay drawdown fills for each strategy."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, data in results.items():
        eq = data["result"]["equity"].values
        peaks = np.maximum.accumulate(eq)
        dd = (eq - peaks) / peaks
        ax.fill_between(range(len(dd)), dd, alpha=0.35, label=name)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_sharpe_heatmap(grid_df, fix_col="top_frac", fix_val=None,
                         figsize=(8, 5)):
    """Heatmap of Sharpe over (lookback, hold) for one top_frac value.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Output of parallel_grid_search.
    fix_col : str
        Column to hold constant.
    fix_val : float or None
        Value to fix. If None, uses the value from the best row.
    """
    if fix_val is None:
        fix_val = grid_df.iloc[0][fix_col]
    sub = grid_df[grid_df[fix_col] == fix_val]
    pivot = sub.pivot(index="lookback", columns="hold", values="sharpe")
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("hold (bars)")
    ax.set_ylabel("lookback (bars)")
    ax.set_title(f"Sharpe Ratio ({fix_col}={fix_val})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_rolling_sharpe(net_returns, window=720, periods=24*365,
                         figsize=(12, 4)):
    """Rolling annualized Sharpe ratio over time."""
    import pandas as pd
    net = pd.Series(net_returns)
    roll_mean = net.rolling(window).mean() * periods
    roll_vol = net.rolling(window).std() * np.sqrt(periods)
    roll_sharpe = (roll_mean / roll_vol).dropna()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(roll_sharpe.index, roll_sharpe.values)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"Rolling Sharpe ({window}-bar window)")
    ax.set_ylabel("Sharpe")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def metrics_table(results, benchmark=None):
    """Print a formatted metrics comparison table."""
    import pandas as pd
    rows = {}
    for name, data in results.items():
        rows[name] = data["metrics"]
    if benchmark is not None:
        rows["BTC Buy-Hold"] = benchmark["metrics"]
    df = pd.DataFrame(rows).T
    col_order = ["ann_return", "ann_vol", "sharpe", "sortino",
                 "max_dd", "var_5pct", "cvar_5pct"]
    return df[[c for c in col_order if c in df.columns]].round(4)
