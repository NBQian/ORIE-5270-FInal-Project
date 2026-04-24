"""Unit tests for strategy module."""
import unittest
import numpy as np
import pandas as pd
from crypto_momentum_lab.strategy import backtest_long_short, equity_curve_metrics


class TestBacktest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2023-01-01", periods=600, freq="h")
        self.prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, (600, 5)), axis=0)),
            index=idx, columns=list("ABCDE"),
        )

    def test_output_columns(self):
        bt = backtest_long_short(self.prices, lookback=24, hold=6, top_frac=0.4)
        for c in ["gross", "net", "equity", "turnover"]:
            self.assertIn(c, bt.columns)

    def test_equity_positive(self):
        bt = backtest_long_short(self.prices, lookback=24, hold=6, top_frac=0.4)
        self.assertGreater(bt["equity"].iloc[-1], 0)

    def test_costs_reduce_returns(self):
        free = backtest_long_short(self.prices, 24, 6, 0.4, cost_bps=0)
        paid = backtest_long_short(self.prices, 24, 6, 0.4, cost_bps=20)
        self.assertGreaterEqual(free["net"].sum(), paid["net"].sum())

    def test_turnover_nonnegative(self):
        bt = backtest_long_short(self.prices, lookback=24, hold=6, top_frac=0.4)
        self.assertTrue((bt["turnover"] >= 0).all())


class TestMetrics(unittest.TestCase):
    def test_keys(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0001, 0.01, 500)
        m = equity_curve_metrics(r)
        for k in ["ann_return", "ann_vol", "sharpe"]:
            self.assertIn(k, m)


if __name__ == "__main__":
    unittest.main()
