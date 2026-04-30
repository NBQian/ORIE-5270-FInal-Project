"""crypto_momentum_lab — parallel cross-sectional crypto momentum backtester."""

__version__ = "0.1.0"

from .data_loader import fetch_klines, load_universe, returns_from_prices
from .signals import momentum_signal, volatility_signal, rank_cross_sectional
from .strategy import backtest_long_short, equity_curve_metrics
from .risk import sharpe, sortino, max_drawdown, var_cvar, summary_stats
from .parallel_grid import parallel_grid_search