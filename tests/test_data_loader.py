"""Unit tests for data_loader module."""
import unittest
import os
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch
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

    def test_returns_edge_cases(self):
        # Empty DataFrame should return an empty DataFrame safely
        empty_prices = pd.DataFrame()
        self.assertTrue(returns_from_prices(empty_prices).empty)

        # Single row DataFrame should return an empty DataFrame safely
        single_row = pd.DataFrame({"A": [100.0]})
        self.assertTrue(returns_from_prices(single_row).empty)

class TestFetchKlines(unittest.TestCase):
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
            
            # Explicitly pass the end date to match the cache file name
            result = fetch_klines("BTCUSDT", end="2026-04-17", cache_dir=d)
            self.assertAlmostEqual(result["close"].iloc[0], 1.5)

    # FIX: Patch the entire Client class right where it is used in data_loader
    @patch('crypto_momentum_lab.data_loader.Client')
    def test_api_call_and_processing(self, mock_client_class):
        # Setup the fake client instance to return our dummy data
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.get_historical_klines.return_value = [
            [1672531200000, "16500.00", "16550.00", "16450.00", "16520.00", "100.5",
             1672534799999, "1660000.00", 500, "50.0", "825000.00", "0"]
        ]
        
        with tempfile.TemporaryDirectory() as d:
            result = fetch_klines("MOCKCOIN", end="2023-01-01", cache_dir=d)
            
            # Verify the DataFrame shape and data type coercion
            self.assertEqual(len(result), 1)
            self.assertIn("close", result.columns)
            self.assertEqual(result["close"].iloc[0], 16520.0)
            self.assertIsInstance(result.index, pd.DatetimeIndex)

class TestLoadUniverse(unittest.TestCase):
    def test_forward_fills_and_drops(self):
        with tempfile.TemporaryDirectory() as d:
            for sym in ["AAUSDT", "BBUSDT"]:
                fname = f"{sym}_1h_2020-10-01_2026-04-17.parquet"
                idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
                
                # Simulate missing middle value for BBUSDT
                close_vals = [1.0, 2.0, 3.0] if sym == "AAUSDT" else [10.0, None, 30.0]
                
                df = pd.DataFrame({
                    "open": close_vals, "high": close_vals,
                    "low": close_vals, "close": close_vals,
                    "volume": [100.0]*3,
                }, index=idx)
                df.index.name = "time"
                df.to_parquet(os.path.join(d, fname))
                
            result = load_universe(["AAUSDT", "BBUSDT"], end="2026-04-17", cache_dir=d)
            
            # The length should be 3 because ffill() saves the middle row for BBUSDT
            self.assertEqual(len(result), 3)
            
            # The missing value (None) should be forward-filled with the previous price (10.0)
            self.assertEqual(result["BBUSDT"].iloc[1], 10.0)

if __name__ == "__main__":
    unittest.main()