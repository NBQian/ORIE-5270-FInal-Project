"""Risk analytics: Sharpe, Sortino, drawdown, VaR/CVaR."""
import numpy as np


def sharpe(returns, periods=24 * 365, rf=0.0):
    """Annualized Sharpe ratio."""
    r = np.asarray(returns)
    vol = r.std() * np.sqrt(periods)
    return float((r.mean() * periods - rf) / vol) if vol > 0 else 0.0


def sortino(returns, periods=24 * 365, rf=0.0):
    """Annualized Sortino ratio (downside vol only)."""
    r = np.asarray(returns)
    downside = r[r < 0].std() * np.sqrt(periods)
    return float((r.mean() * periods - rf) / downside) if downside > 0 else 0.0


def max_drawdown(equity):
    """Maximum peak-to-trough drawdown (negative number)."""
    e = np.asarray(equity)
    peaks = np.maximum.accumulate(e)
    return float(((e - peaks) / peaks).min())


def var_cvar(returns, alpha=0.05):
    """Historical Value-at-Risk and Conditional VaR.

    Parameters
    ----------
    returns : array-like
        Return series.
    alpha : float
        Tail probability (default 5%).

    Returns
    -------
    tuple(float, float)
        (VaR, CVaR) both as positive loss numbers.
    """
    s = np.sort(np.asarray(returns))
    k = max(1, int(alpha * len(s)))
    return float(-s[k - 1]), float(-s[:k].mean())


def summary_stats(net_returns, equity, periods=24 * 365):
    """Comprehensive risk summary dict."""
    var, cvar = var_cvar(net_returns)
    return {
        "sharpe": sharpe(net_returns, periods),
        "sortino": sortino(net_returns, periods),
        "max_dd": max_drawdown(equity),
        "var_5pct": var,
        "cvar_5pct": cvar,
        "ann_return": float(np.asarray(net_returns).mean() * periods),
        "ann_vol": float(np.asarray(net_returns).std() * np.sqrt(periods)),
    }
