"""Binance kline downloader with parquet caching."""
import os
import pandas as pd


DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "TRXUSDT",
]


def fetch_klines(symbol, interval="1h", start="2020-10-01",
                 end="2026-04-17", cache_dir="data"):
    """Download OHLCV klines from Binance, cache as parquet.

    Parameters
    ----------
    symbol : str
        Binance trading pair, e.g. "BTCUSDT".
    interval : str
        Candle interval, e.g. "1h", "4h", "1d".
    start, end : str
        ISO date strings for the query window.
    cache_dir : str
        Directory for parquet cache files.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame indexed by time.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{interval}_{start}_{end}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    from binance.client import Client
    client = Client()
    raw = client.get_historical_klines(symbol, interval, start, end)
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tqav", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df[["time", "open", "high", "low", "close", "volume"]].set_index("time")
    df.to_parquet(path)
    return df


def load_universe(symbols=None, interval="1h", start="2020-10-01",
                  end="2026-04-17", cache_dir="data"):
    """Load close prices for a universe of symbols.

    Returns a DataFrame with one column per symbol, rows = timestamps.
    Drops any row where any symbol has NaN.
    """
    symbols = symbols or DEFAULT_UNIVERSE
    closes = {}
    for s in symbols:
        df = fetch_klines(s, interval, start, end, cache_dir)
        closes[s] = df["close"]
    return pd.DataFrame(closes).dropna(how="any")


def returns_from_prices(prices):
    """Simple percentage returns from a price DataFrame."""
    return prices.pct_change().dropna()
