"""Unit tests for risk module."""
import unittest
import numpy as np
from crypto_momentum_lab.risk import (
    sharpe, sortino, max_drawdown, var_cvar, summary_stats,
)


class TestSharpe(unittest.TestCase):
    def test_positive_drift(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0005, 0.01, 1000)
        self.assertGreater(sharpe(r), 0)

    def test_zero_vol(self):
        r = np.zeros(100)
        self.assertEqual(sharpe(r), 0.0)


class TestSortino(unittest.TestCase):
    def test_positive_drift(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0005, 0.01, 1000)
        self.assertGreater(sortino(r), 0)

    def test_higher_than_sharpe(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0005, 0.01, 1000)
        self.assertGreater(sortino(r), sharpe(r))


class TestDrawdown(unittest.TestCase):
    def test_known(self):
        eq = np.array([1, 1.5, 1.2, 0.9, 1.1])
        self.assertAlmostEqual(max_drawdown(eq), (0.9 - 1.5) / 1.5)

    def test_monotonic_up(self):
        eq = np.array([1, 2, 3, 4, 5])
        self.assertAlmostEqual(max_drawdown(eq), 0.0)


class TestVarCvar(unittest.TestCase):
    def test_cvar_greater_than_var(self):
        rng = np.random.default_rng(0)
        v, c = var_cvar(rng.normal(0, 0.01, 1000))
        self.assertGreater(c, v)

    def test_positive(self):
        rng = np.random.default_rng(0)
        v, c = var_cvar(rng.normal(0, 0.01, 1000))
        self.assertGreater(v, 0)
        self.assertGreater(c, 0)


class TestSummary(unittest.TestCase):
    def test_all_keys(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.0005, 0.01, 1000)
        eq = (1 + r).cumprod()
        s = summary_stats(r, eq)
        for k in ["sharpe", "sortino", "max_dd", "var_5pct",
                   "cvar_5pct", "ann_return", "ann_vol"]:
            self.assertIn(k, s)


if __name__ == "__main__":
    unittest.main()
