"""Unit tests for signals module."""
import unittest
import numpy as np
import pandas as pd
from crypto_momentum_lab.signals import (
    momentum_signal, volatility_signal, rank_cross_sectional,
)


class TestMomentum(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=500, freq="h")
        self.prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (500, 4)), axis=0)),
            index=idx, columns=list("ABCD"),
        )

    def test_shape_preserved(self):
        s = momentum_signal(self.prices, lookback=24)
        self.assertEqual(s.shape, self.prices.shape)

    def test_nan_at_start(self):
        s = momentum_signal(self.prices, lookback=24, skip=1)
        self.assertTrue(s.iloc[:25].isna().all().all())

    def test_positive_for_uptrend(self):
        p = pd.DataFrame({"A": np.arange(1, 101, dtype=float)})
        s = momentum_signal(p, lookback=10, skip=0)
        self.assertTrue((s.dropna() > 0).all().all())


class TestVolatility(unittest.TestCase):
    def test_positive(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=500, freq="h")
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (500, 4)), axis=0)),
            index=idx, columns=list("ABCD"),
        )
        v = volatility_signal(prices.pct_change().dropna(), lookback=24).dropna()
        self.assertTrue((v >= 0).all().all())


class TestRank(unittest.TestCase):
    def test_bounds(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=500, freq="h")
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (500, 4)), axis=0)),
            index=idx, columns=list("ABCD"),
        )
        s = momentum_signal(prices, lookback=24).dropna()
        r = rank_cross_sectional(s)
        self.assertTrue(((r >= -1) & (r <= 1)).all().all())

    def test_zero_mean(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-01", periods=500, freq="h")
        prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, (500, 4)), axis=0)),
            index=idx, columns=list("ABCD"),
        )
        s = momentum_signal(prices, lookback=24).dropna()
        r = rank_cross_sectional(s)
        np.testing.assert_allclose(r.mean(axis=1).values, 0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
