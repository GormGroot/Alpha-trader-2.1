"""
Tests for backtest-modulet: Trade, BacktestResult, Backtester, Comparison.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtester import Backtester, BacktestResult, Trade
from src.backtest.comparison import (
    BenchmarkResult,
    ComparisonReport,
    StrategyComparison,
)
from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult


# ── Hjælpefunktioner ──────────────────────────────────────────


def _make_ohlcv(n: int = 200, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generér syntetisk OHLCV-data med realistisk prisudvikling."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    returns = rng.normal(0.0005, 0.015, n)
    prices = start_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "Open": prices * (1 - rng.uniform(0, 0.005, n)),
            "High": prices * (1 + rng.uniform(0, 0.01, n)),
            "Low": prices * (1 - rng.uniform(0, 0.01, n)),
            "Close": prices,
            "Volume": rng.randint(1_000_000, 10_000_000, n),
        },
        index=dates,
    )
    return df


class _AlwaysBuyStrategy(BaseStrategy):
    """Teststrategisom altid siger BUY."""

    @property
    def name(self) -> str:
        return "AlwaysBuy"

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        return StrategyResult(Signal.BUY, 80.0, "altid køb")


class _AlwaysHoldStrategy(BaseStrategy):
    """Teststrategisom altid siger HOLD."""

    @property
    def name(self) -> str:
        return "AlwaysHold"

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        return StrategyResult(Signal.HOLD, 50.0, "aldrig gør noget")


class _AlternateBuySellStrategy(BaseStrategy):
    """Teststrategider skifter mellem BUY og SELL."""

    def __init__(self):
        self._call_count = 0

    @property
    def name(self) -> str:
        return "AlternateBuySell"

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        self._call_count += 1
        if self._call_count % 2 == 1:
            return StrategyResult(Signal.BUY, 75.0, "køb signal")
        return StrategyResult(Signal.SELL, 75.0, "sælg signal")


def _mock_market_data(data: dict[str, pd.DataFrame]):
    """Opret en mocket MarketDataFetcher."""
    mock = MagicMock()

    def get_hist(symbol, interval="1d", start=None, end=None, lookback_days=None):
        return data.get(symbol.upper(), pd.DataFrame())

    mock.get_historical.side_effect = get_hist
    return mock


# ══════════════════════════════════════════════════════════════
#  Test Trade
# ══════════════════════════════════════════════════════════════


class TestTrade:
    def test_long_profit(self):
        t = Trade(
            symbol="AAPL", side="long", qty=10,
            entry_price=100.0, exit_price=110.0,
            entry_date="2024-01-01", exit_date="2024-02-01",
            entry_reason="buy", exit_reason="sell",
        )
        assert t.gross_pnl == pytest.approx(100.0)
        assert t.net_pnl == pytest.approx(100.0)
        assert t.return_pct == pytest.approx(10.0)

    def test_long_loss(self):
        t = Trade(
            symbol="AAPL", side="long", qty=10,
            entry_price=100.0, exit_price=90.0,
            entry_date="2024-01-01", exit_date="2024-02-01",
            entry_reason="buy", exit_reason="sell",
        )
        assert t.gross_pnl == pytest.approx(-100.0)

    def test_commission_reduces_pnl(self):
        t = Trade(
            symbol="AAPL", side="long", qty=10,
            entry_price=100.0, exit_price=110.0,
            entry_date="2024-01-01", exit_date="2024-02-01",
            entry_reason="buy", exit_reason="sell",
            commission_cost=5.0,
        )
        assert t.net_pnl == pytest.approx(95.0)
        assert t.return_pct == pytest.approx(9.5)

    def test_short_profit(self):
        t = Trade(
            symbol="TSLA", side="short", qty=5,
            entry_price=200.0, exit_price=180.0,
            entry_date="2024-01-01", exit_date="2024-02-01",
            entry_reason="short", exit_reason="cover",
        )
        assert t.gross_pnl == pytest.approx(100.0)


# ══════════════════════════════════════════════════════════════
#  Test BacktestResult
# ══════════════════════════════════════════════════════════════


class TestBacktestResult:
    def test_empty_result(self):
        r = BacktestResult(
            strategy_name="test", symbols=["AAPL"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=100_000,
        )
        assert r.total_return_pct == 0.0
        assert r.num_trades == 0
        assert r.win_rate == 0.0
        assert r.avg_profit_per_trade == 0.0
        assert r.profit_factor == 0.0

    def test_metrics_with_trades(self):
        trades = [
            Trade("AAPL", "long", 10, 100.0, 120.0, "2024-01", "2024-02", "b", "s"),
            Trade("MSFT", "long", 5, 200.0, 190.0, "2024-02", "2024-03", "b", "s"),
            Trade("GOOGL", "long", 3, 150.0, 165.0, "2024-03", "2024-04", "b", "s"),
        ]
        eq = pd.Series([100_000, 100_200, 100_150, 100_300, 100_250])
        daily_ret = eq.pct_change().dropna()

        r = BacktestResult(
            strategy_name="test", symbols=["AAPL", "MSFT", "GOOGL"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=100_250,
            trades=trades, equity_curve=eq, daily_returns=daily_ret,
        )
        assert r.num_trades == 3
        assert r.win_rate == pytest.approx(200 / 3)  # 2 ud af 3
        assert r.profit_factor > 1.0
        assert r.total_return_pct == pytest.approx(0.25)

    def test_max_drawdown(self):
        eq = pd.Series([100_000, 105_000, 95_000, 102_000])
        r = BacktestResult(
            strategy_name="test", symbols=["X"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=102_000,
            equity_curve=eq,
        )
        # Peak 105k, trough 95k → DD = 10/105 ≈ 9.52%
        assert r.max_drawdown_pct == pytest.approx(10 / 105 * 100, rel=0.01)

    def test_sharpe_ratio_positive_for_gains(self):
        eq = pd.Series([100_000 + i * 100 for i in range(252)])
        daily_ret = eq.pct_change().dropna()
        r = BacktestResult(
            strategy_name="test", symbols=["X"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=eq.iloc[-1],
            equity_curve=eq, daily_returns=daily_ret,
        )
        assert r.sharpe_ratio > 0

    def test_summary_returns_dict(self):
        r = BacktestResult(
            strategy_name="test", symbols=["AAPL"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=110_000,
        )
        s = r.summary()
        assert isinstance(s, dict)
        assert "Strategi" in s
        assert "Totalt afkast" in s

    def test_summary_table_returns_string(self):
        r = BacktestResult(
            strategy_name="test", symbols=["AAPL"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=110_000,
        )
        table = r.summary_table()
        assert isinstance(table, str)
        assert "test" in table

    def test_annualized_return(self):
        eq = pd.Series([100_000] * 126 + [110_000] * 126)  # 252 dage, 10% totalt
        r = BacktestResult(
            strategy_name="test", symbols=["X"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=110_000,
            equity_curve=eq,
        )
        # 10% over 1 år ≈ 10% annualiseret
        assert abs(r.annualized_return_pct - 10.0) < 1.0

    def test_total_commission(self):
        trades = [
            Trade("A", "long", 10, 100, 110, "d1", "d2", "b", "s", commission_cost=5),
            Trade("B", "long", 5, 200, 210, "d3", "d4", "b", "s", commission_cost=3),
        ]
        r = BacktestResult(
            strategy_name="test", symbols=["A", "B"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=100_300,
            trades=trades,
        )
        assert r.total_commission == 8.0


# ══════════════════════════════════════════════════════════════
#  Test Backtester
# ══════════════════════════════════════════════════════════════


class TestBacktester:
    def test_always_hold_no_trades(self):
        df = _make_ohlcv(200)
        mock_md = _mock_market_data({"AAPL": df})
        bt = Backtester(
            strategy=_AlwaysHoldStrategy(),
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            market_data=mock_md,
        )
        result = bt.run()
        assert result.num_trades == 0
        assert result.final_equity == pytest.approx(100_000)

    def test_always_buy_makes_trades(self):
        df = _make_ohlcv(200, seed=1)
        mock_md = _mock_market_data({"AAPL": df})
        bt = Backtester(
            strategy=_AlwaysBuyStrategy(),
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            market_data=mock_md,
        )
        result = bt.run()
        # Skal have mindst 1 trade (backtest_slut lukker position)
        assert result.num_trades >= 1

    def test_alternate_makes_multiple_trades(self):
        df = _make_ohlcv(200, seed=2)
        mock_md = _mock_market_data({"AAPL": df})
        bt = Backtester(
            strategy=_AlternateBuySellStrategy(),
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            market_data=mock_md,
        )
        result = bt.run()
        assert result.num_trades >= 2

    def test_commission_applied(self):
        df = _make_ohlcv(200, seed=3)
        mock_md = _mock_market_data({"AAPL": df})
        bt = Backtester(
            strategy=_AlternateBuySellStrategy(),
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            commission_pct=0.01,
            market_data=mock_md,
        )
        result = bt.run()
        assert result.total_commission > 0

    def test_multiple_symbols(self):
        df_a = _make_ohlcv(200, start_price=150, seed=4)
        df_b = _make_ohlcv(200, start_price=300, seed=5)
        mock_md = _mock_market_data({"AAPL": df_a, "MSFT": df_b})
        bt = Backtester(
            strategy=_AlternateBuySellStrategy(),
            symbols=["AAPL", "MSFT"],
            start="2023-01-01",
            end="2023-12-31",
            market_data=mock_md,
        )
        result = bt.run()
        symbols_traded = {t.symbol for t in result.trades}
        assert len(symbols_traded) >= 1

    def test_equity_curve_length(self):
        df = _make_ohlcv(200, seed=6)
        mock_md = _mock_market_data({"AAPL": df})
        bt = Backtester(
            strategy=_AlwaysHoldStrategy(),
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            market_data=mock_md,
        )
        result = bt.run()
        assert len(result.equity_curve) == 200

    def test_empty_data_returns_initial(self):
        mock_md = _mock_market_data({})
        bt = Backtester(
            strategy=_AlwaysBuyStrategy(),
            symbols=["FAKE"],
            start="2023-01-01",
            end="2023-12-31",
            market_data=mock_md,
        )
        result = bt.run()
        assert result.final_equity == 100_000
        assert result.num_trades == 0

    def test_initial_capital_propagated(self):
        df = _make_ohlcv(200, seed=7)
        mock_md = _mock_market_data({"AAPL": df})
        bt = Backtester(
            strategy=_AlwaysHoldStrategy(),
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            initial_capital=50_000,
            market_data=mock_md,
        )
        result = bt.run()
        assert result.initial_capital == 50_000

    def test_spread_cost_reduces_profit(self):
        df = _make_ohlcv(200, seed=8)
        mock_md_1 = _mock_market_data({"AAPL": df})
        mock_md_2 = _mock_market_data({"AAPL": df})
        bt_no_spread = Backtester(
            strategy=_AlternateBuySellStrategy(),
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            spread_pct=0.0,
            commission_pct=0.0,
            market_data=mock_md_1,
        )
        bt_with_spread = Backtester(
            strategy=_AlternateBuySellStrategy(),
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            spread_pct=0.01,  # 1% spread
            commission_pct=0.0,
            market_data=mock_md_2,
        )
        r1 = bt_no_spread.run()
        r2 = bt_with_spread.run()
        # Spread skal koste noget
        if r1.num_trades > 0 and r2.num_trades > 0:
            assert r2.final_equity < r1.final_equity


# ══════════════════════════════════════════════════════════════
#  Test BenchmarkResult
# ══════════════════════════════════════════════════════════════


class TestBenchmarkResult:
    def test_basic_metrics(self):
        eq = pd.Series([100_000, 105_000, 110_000])
        daily = eq.pct_change().dropna()
        b = BenchmarkResult(
            name="SPY", initial_capital=100_000, final_equity=110_000,
            equity_curve=eq, daily_returns=daily,
        )
        assert b.total_return_pct == pytest.approx(10.0)
        assert b.max_drawdown_pct == 0.0  # monotont stigende
        assert b.sharpe_ratio > 0

    def test_drawdown(self):
        eq = pd.Series([100_000, 110_000, 95_000, 105_000])
        b = BenchmarkResult(
            name="SPY", initial_capital=100_000, final_equity=105_000,
            equity_curve=eq,
        )
        # Peak 110k → 95k = 13.6%
        assert b.max_drawdown_pct == pytest.approx(15_000 / 110_000 * 100, rel=0.01)


# ══════════════════════════════════════════════════════════════
#  Test ComparisonReport
# ══════════════════════════════════════════════════════════════


class TestComparisonReport:
    def _make_results(self):
        r1 = BacktestResult(
            strategy_name="Strat A", symbols=["AAPL"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=115_000,
            daily_returns=pd.Series([0.001] * 252),
        )
        r2 = BacktestResult(
            strategy_name="Strat B", symbols=["AAPL"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=108_000,
            daily_returns=pd.Series([0.0005] * 252),
        )
        bench = BenchmarkResult(
            name="SPY", initial_capital=100_000, final_equity=112_000,
        )
        return [r1, r2], bench

    def test_ranking_by_sharpe(self):
        results, bench = self._make_results()
        report = ComparisonReport(results=results, benchmark=bench)
        ranking = report.ranking()
        assert ranking[0][0] == "Strat A"

    def test_best_strategy(self):
        results, bench = self._make_results()
        report = ComparisonReport(results=results, benchmark=bench)
        best = report.best_strategy()
        assert best.strategy_name == "Strat A"

    def test_comparison_table_string(self):
        results, bench = self._make_results()
        report = ComparisonReport(results=results, benchmark=bench)
        table = report.comparison_table()
        assert "Strat A" in table
        assert "Strat B" in table
        assert "SPY" in table

    def test_detailed_report(self):
        results, bench = self._make_results()
        report = ComparisonReport(results=results, benchmark=bench)
        detail = report.detailed_report()
        assert "alpha" in detail.lower()
        assert "RANGERING" in detail

    def test_no_benchmark(self):
        r1 = BacktestResult(
            strategy_name="Test", symbols=["X"],
            start_date="2024-01-01", end_date="2024-12-31",
            initial_capital=100_000, final_equity=105_000,
        )
        report = ComparisonReport(results=[r1])
        table = report.comparison_table()
        assert "Test" in table


# ══════════════════════════════════════════════════════════════
#  Test StrategyComparison
# ══════════════════════════════════════════════════════════════


class TestStrategyComparison:
    def test_compare_two_strategies(self):
        df = _make_ohlcv(200, seed=10)
        mock_md = _mock_market_data({"AAPL": df, "SPY": df})

        comp = StrategyComparison(
            strategies=[_AlwaysBuyStrategy(), _AlwaysHoldStrategy()],
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            benchmark_symbol="SPY",
            market_data=mock_md,
        )
        report = comp.run()

        assert len(report.results) == 2
        assert report.benchmark is not None
        assert report.results[0].strategy_name == "AlwaysBuy"
        assert report.results[1].strategy_name == "AlwaysHold"

    def test_comparison_without_benchmark(self):
        df = _make_ohlcv(200, seed=11)
        mock_md = _mock_market_data({"AAPL": df})  # Ingen SPY data

        comp = StrategyComparison(
            strategies=[_AlwaysHoldStrategy()],
            symbols=["AAPL"],
            start="2023-01-01",
            end="2023-12-31",
            benchmark_symbol="SPY",
            market_data=mock_md,
        )
        report = comp.run()
        assert report.benchmark is None
        assert len(report.results) == 1
