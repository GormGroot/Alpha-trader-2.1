"""
Backtest-modul: kør strategier mod historisk data.

    from src.backtest import Backtester, StrategyComparison
    bt = Backtester(strategy, symbols=["AAPL"], start="2024-01-01")
    result = bt.run()
"""

from src.backtest.backtester import Backtester, BacktestResult, Trade
from src.backtest.comparison import StrategyComparison, ComparisonReport, BenchmarkResult
from src.backtest.stress_test import (
    StressTester,
    StressTestReport,
    ScenarioResult,
    MonteCarloResult,
    CrisisScenario,
    ScenarioType,
    Vulnerability,
    HISTORICAL_CRISES,
    SYNTHETIC_SCENARIOS,
)

__all__ = [
    "Backtester",
    "BacktestResult",
    "Trade",
    "StrategyComparison",
    "ComparisonReport",
    "BenchmarkResult",
    # stress_test
    "StressTester",
    "StressTestReport",
    "ScenarioResult",
    "MonteCarloResult",
    "CrisisScenario",
    "ScenarioType",
    "Vulnerability",
    "HISTORICAL_CRISES",
    "SYNTHETIC_SCENARIOS",
]
