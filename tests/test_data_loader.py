"""Unit tests for data_loader module."""
import unittest
import os
import tempfile
import numpy as np
import pandas as pd
from crypto_momentum_lab.data_loader import (
    returns_from_prices, fetch_klines, load_universe,
)


class TestReturns(unittest.TestCase):
    def test_simple_returns(self):
        prices = pd.DataFrame({"BTCUSDT": [100.0, 110.0, 121.0]})
        r = returns_from_prices(prices)
        np.testing.assert_allclose(r["BTCUSDT"].values, [0.1, 0.1])

    def test_returns_length(self):
        prices = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        self.assertEqual(len(returns_from_prices(prices)), 4)


class TestCacheHit(unittest.TestCase):
    def test_loads_from_cache(self):
        with tempfile.TemporaryDirectory() as d:
            fname = "BTCUSDT_1h_2020-10-01_2026-04-17.parquet"
            path = os.path.join(d, fname)
            df = pd.DataFrame(
                {"open": [1.0], "high": [2.0], "low": [0.5],
                 "close": [1.5], "volume": [100.0]},
                index=pd.to_datetime(["2023-01-01"]),
            )
            df.index.name = "time"
            df.to_parquet(path)
            result = fetch_klines("BTCUSDT", cache_dir=d)
            self.assertAlmostEqual(result["close"].iloc[0], 1.5)


class TestLoadUniverse(unittest.TestCase):
    def test_drops_nans(self):
        with tempfile.TemporaryDirectory() as d:
            for sym in ["AAUSDT", "BBUSDT"]:
                fname = f"{sym}_1h_2020-10-01_2026-04-17.parquet"
                idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
                close_vals = [1, 2, 3] if sym == "AAUSDT" else [10, None, 30]
                df = pd.DataFrame({
                    "open": close_vals, "high": close_vals,
                    "low": close_vals, "close": close_vals,
                    "volume": [100]*3,
                }, index=idx)
                df.index.name = "time"
                df.to_parquet(os.path.join(d, fname))
            result = load_universe(["AAUSDT", "BBUSDT"], cache_dir=d)
            self.assertEqual(len(result), 2)  # one row dropped


if __name__ == "__main__":
    unittest.main()
