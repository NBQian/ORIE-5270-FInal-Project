"""Binance kline downloader with parquet caching."""
import os
import logging
import pandas as pd
from datetime import datetime, timezone
from binance.client import Client

DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "TRXUSDT",
]

def fetch_klines(symbol, interval="1h", start="2020-10-01",
                 end=None, cache_dir="data"):
    """Download OHLCV klines from Binance, cache as parquet.

    Parameters
    ----------
    symbol : str
        Binance trading pair, e.g. "BTCUSDT".
    interval : str
        Candle interval, e.g. "1h", "4h", "1d".
    start, end : str
        ISO date strings for the query window. If end is None, uses current UTC date.
    cache_dir : str
        Directory for parquet cache files.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame indexed by time, or empty DataFrame on failure.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Dynamically set end date if not provided
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
    path = os.path.join(cache_dir, f"{symbol}_{interval}_{start}_{end}.parquet")
    
    if os.path.exists(path):
        return pd.read_parquet(path)
        
    client = Client()
    
    # Catch API errors (e.g., rate limits, disconnects)
    try:
        raw = client.get_historical_klines(symbol, interval, start, end)
    except Exception as e:
        logging.error(f"Failed to fetch data for {symbol}: {e}")
        return pd.DataFrame()
        
    if not raw:
        return pd.DataFrame()

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
                  end=None, cache_dir="data"):
    """Load close prices for a universe of symbols.

    Returns a DataFrame with one column per symbol, rows = timestamps.
    Uses forward-fill to handle temporary missing data before dropping initial NaNs.
    """
    symbols = symbols or DEFAULT_UNIVERSE
    closes = {}
    
    for s in symbols:
        df = fetch_klines(s, interval, start, end, cache_dir)
        if not df.empty:
            closes[s] = df["close"]
            
    # ffill() saves temporary missing hours (e.g., exchange maintenance)
    # dropna() removes the empty rows at the beginning before newer coins launched
    return pd.DataFrame(closes).ffill().dropna()

def returns_from_prices(prices):
    """Simple percentage returns from a price DataFrame."""
    # Guard against empty or 1-row dataframes where pct_change results in all NaNs
    if prices.empty or len(prices) < 2:
        return pd.DataFrame(columns=prices.columns) # FIX: Removed the index argument
    return prices.pct_change().dropna()