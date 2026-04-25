"""Unit tests for parallel_grid module using FakePool."""
import unittest
import numpy as np
import pandas as pd
import crypto_momentum_lab.parallel_grid as pg


class FakePool:
    """Mock multiprocessing.Pool that runs serially in-process."""
    def __init__(self, n):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, fn, args):
        return [fn(a) for a in args]


class TestParallelGrid(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        idx = pd.date_range("2023-01-01", periods=600, freq="h")
        self.prices = pd.DataFrame(
            100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, (600, 5)), axis=0)),
            index=idx, columns=list("ABCDE"),
        )
        self._orig = pg.Pool
        pg.Pool = FakePool

    def tearDown(self):
        pg.Pool = self._orig

    def test_correct_row_count(self):
        res = pg.parallel_grid_search(
            self.prices,
            lookbacks=[24, 48], holds=[6], top_fracs=[0.4],
            n_workers=2,
        )
        self.assertEqual(len(res), 2)

    def test_has_sharpe_column(self):
        res = pg.parallel_grid_search(
            self.prices,
            lookbacks=[24], holds=[6], top_fracs=[0.3],
            n_workers=1,
        )
        self.assertIn("sharpe", res.columns)

    def test_sorted_descending(self):
        res = pg.parallel_grid_search(
            self.prices,
            lookbacks=[24, 48, 72], holds=[6, 12],
            top_fracs=[0.3, 0.4],
            n_workers=2,
        )
        sharpes = res["sharpe"].values
        self.assertTrue((sharpes[:-1] >= sharpes[1:]).all())

    def test_params_in_output(self):
        res = pg.parallel_grid_search(
            self.prices,
            lookbacks=[24], holds=[6], top_fracs=[0.3],
            n_workers=1,
        )
        for col in ["lookback", "hold", "top_frac"]:
            self.assertIn(col, res.columns)


if __name__ == "__main__":
    unittest.main()
