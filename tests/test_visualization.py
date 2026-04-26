"""Unit tests for visualization module."""
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from crypto_momentum_lab.visualization import (
    plot_equity_curves, plot_drawdowns, metrics_table,
    plot_rolling_sharpe,
)


def _fake_result():
    idx = pd.date_range("2023-01-01", periods=200, freq="h")
    eq = pd.Series((1 + np.random.default_rng(0).normal(0.0001, 0.005, 200)).cumprod(),
                    index=idx)
    net = eq.pct_change().fillna(0)
    return {"result": pd.DataFrame({"equity": eq, "net": net}),
            "metrics": {"ann_return": 0.1, "ann_vol": 0.3, "sharpe": 0.33,
                        "sortino": 0.5, "max_dd": -0.1, "var_5pct": 0.01,
                        "cvar_5pct": 0.015}}


class TestPlots(unittest.TestCase):
    def test_equity_plot(self):
        fig = plot_equity_curves({"test": _fake_result()})
        self.assertIsNotNone(fig)
        plt_mod = matplotlib.pyplot
        plt_mod.close(fig)

    def test_drawdown_plot(self):
        fig = plot_drawdowns({"test": _fake_result()})
        self.assertIsNotNone(fig)
        matplotlib.pyplot.close(fig)

    def test_rolling_sharpe(self):
        rng = np.random.default_rng(0)
        fig = plot_rolling_sharpe(rng.normal(0, 0.01, 1000), window=100)
        self.assertIsNotNone(fig)
        matplotlib.pyplot.close(fig)


class TestMetricsTable(unittest.TestCase):
    def test_table(self):
        t = metrics_table({"a": _fake_result(), "b": _fake_result()})
        self.assertEqual(len(t), 2)
        self.assertIn("sharpe", t.columns)

    def test_with_benchmark(self):
        bench = {"metrics": {"ann_return": 0.5, "ann_vol": 0.6, "sharpe": 0.8,
                              "sortino": 1.0, "max_dd": -0.3,
                              "var_5pct": 0.02, "cvar_5pct": 0.03}}
        t = metrics_table({"a": _fake_result()}, benchmark=bench)
        self.assertEqual(len(t), 2)


if __name__ == "__main__":
    unittest.main()
