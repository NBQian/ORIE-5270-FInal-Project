"""Long-short cross-sectional momentum backtest."""
import numpy as np
import pandas as pd
from .signals import momentum_signal, rank_cross_sectional


def backtest_long_short(prices, lookback=168, hold=24, top_frac=0.3,
                        cost_bps=5):
    """Backtest a cross-sectional momentum long-short portfolio.

    Parameters
    ----------
    prices : pd.DataFrame
        Close prices, columns = assets.
    lookback : int
        Momentum lookback in bars.
    hold : int
        Holding period — rebalance every `hold` bars.
    top_frac : float
        Fraction of universe to go long (same fraction shorted).
    cost_bps : float
        Round-trip transaction cost in basis points.

    Returns
    -------
    pd.DataFrame
        Columns: gross, net, equity, turnover.
    """
    rets = prices.pct_change().fillna(0)
    sig = momentum_signal(prices, lookback=lookback).dropna()
    ranks = rank_cross_sectional(sig)
    n = ranks.shape[1]
    k = max(1, int(top_frac * n))

    # Build target weights at each bar
    weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    for t in ranks.index:
        row = ranks.loc[t]
        weights.loc[t, row.nlargest(k).index] = 1.0 / k
        weights.loc[t, row.nsmallest(k).index] = -1.0 / k

    # Reindex to full return series, forward-fill
    weights = weights.reindex(rets.index).ffill().fillna(0)

    # Hold positions for `hold` bars
    held = weights.copy()
    for i in range(1, len(held)):
        if i % hold != 0:
            held.iloc[i] = held.iloc[i - 1]

    # Lag by 1 bar to prevent look-ahead
    pos = held.shift(1).fillna(0)
    turnover = pos.diff().abs().sum(axis=1).fillna(0)
    gross = (pos * rets).sum(axis=1)
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost
    equity = (1 + net).cumprod()

    return pd.DataFrame({
        "gross": gross, "net": net, "equity": equity, "turnover": turnover,
    })


def equity_curve_metrics(daily_returns, periods_per_year=24 * 365):
    """Quick annualized metrics from a return series."""
    r = np.asarray(daily_returns)
    ann_ret = r.mean() * periods_per_year
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol),
            "sharpe": float(sharpe)}
