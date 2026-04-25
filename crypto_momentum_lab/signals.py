"""Cross-sectional momentum and volatility signals."""
import numpy as np
import pandas as pd


def momentum_signal(prices, lookback=168, skip=1):
    """Past-N-bar return, skipping most recent `skip` bars.

    Parameters
    ----------
    prices : pd.DataFrame
        Close prices, one column per asset.
    lookback : int
        Number of bars to measure return over.
    skip : int
        Bars to skip at the end (avoids microstructure noise).

    Returns
    -------
    pd.DataFrame
        Raw momentum signal for each (time, asset).
    """
    return prices.shift(skip) / prices.shift(skip + lookback) - 1


def volatility_signal(returns, lookback=168):
    """Rolling volatility (standard deviation) of returns.

    Useful as a secondary signal or for vol-weighting.
    """
    return returns.rolling(lookback).std()


def rank_cross_sectional(signal):
    """Rank assets cross-sectionally at each timestamp.

    Returns values in [-1, +1] with zero mean at each row.
    """
    ranks = signal.rank(axis=1, pct=True)
    centered = 2 * ranks - 1
    return centered.sub(centered.mean(axis=1), axis=0)
