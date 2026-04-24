"""Unit tests for backtest module."""
import unittest
import numpy as np
import pandas as pd
import crypto_momentum_lab.backtest as bt_mod


class FakePool:
    def __init__(self, n): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, args): return [fn(a) for a in args]


class TestPresetBacktest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=600, freq="h")
        self.prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (600, 5)), axis=0)),
            index=idx, columns=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"],
        )
        self._orig = bt_mod.Pool
        bt_mod.Pool = FakePool

    def tearDown(self):
        bt_mod.Pool = self._orig

    def test_returns_all_presets(self):
        res = bt_mod.parallel_preset_backtest(
            self.prices, presets={"fast": {"lookback": 24, "hold": 6, "top_frac": 0.4}},
            n_workers=1,
        )
        self.assertIn("fast", res)
        self.assertIn("metrics", res["fast"])

    def test_metrics_keys(self):
        res = bt_mod.parallel_preset_backtest(
            self.prices,
            presets={"test": {"lookback": 24, "hold": 6, "top_frac": 0.4}},
            n_workers=1,
        )
        for k in ["sharpe", "max_dd", "ann_return"]:
            self.assertIn(k, res["test"]["metrics"])


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=200, freq="h")
        self.prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, (200, 3)), axis=0)),
            index=idx, columns=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        )

    def test_benchmark(self):
        b = bt_mod.benchmark_buy_hold(self.prices, "BTCUSDT")
        self.assertIn("equity", b)
        self.assertIn("metrics", b)
        self.assertGreater(b["equity"].iloc[-1], 0)


if __name__ == "__main__":
    unittest.main()
