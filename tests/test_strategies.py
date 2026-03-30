"""
Tests for handelsstrategier.

Bruger konstruerede datasæt med kendte SMA/RSI-mønstre, så
signaler er forudsigelige og deterministiske.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.indicators import add_sma, add_rsi, add_volume_analysis, add_all_indicators
from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult
from src.strategy.sma_crossover import SMACrossoverStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.combined_strategy import CombinedStrategy


# ── Helpers ──────────────────────────────────────────────────

def _make_df(closes: list[float], volume: int = 5_000_000) -> pd.DataFrame:
    """Lav en minimal OHLCV DataFrame fra en liste af lukkekurser."""
    n = len(closes)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": [volume] * n,
        },
        index=dates,
    )


def _make_golden_cross_df(short: int = 20, long: int = 50) -> pd.DataFrame:
    """
    Konstruér data hvor kort SMA krydser over lang SMA mellem [-2] og [-1].

    Strategi: byg priser iterativt og brug bisection på den sidste pris
    så crossover sker præcist på den sidste bar.
    """
    n = long + short + 10
    # Jævnt faldende priser → SMA_short < SMA_long hele vejen
    prices = [100 - i * 0.3 for i in range(n)]

    df = _make_df(prices)
    add_sma(df, short)
    add_sma(df, long)
    add_volume_analysis(df)

    # Bekræft at short < long inden vi justerer
    s_col, l_col = f"SMA_{short}", f"SMA_{long}"
    gap = df[l_col].iloc[-1] - df[s_col].iloc[-1]

    # Justér den sidste pris så SMA_short krydser over SMA_long.
    # Net effekt = bump * (1/short - 1/long). Løs for bump:
    net_per_unit = (1.0 / short) - (1.0 / long)
    needed_bump = (gap / net_per_unit) + 1.0  # lidt ekstra for at sikre kryds
    prices[-1] += needed_bump

    df = _make_df(prices)
    add_sma(df, short)
    add_sma(df, long)
    add_volume_analysis(df)
    return df


def _make_death_cross_df(short: int = 20, long: int = 50) -> pd.DataFrame:
    """Konstruér data hvor kort SMA krydser under lang SMA mellem [-2] og [-1]."""
    n = long + short + 10
    # Jævnt stigende priser → SMA_short > SMA_long
    prices = [100 + i * 0.3 for i in range(n)]

    df = _make_df(prices)
    add_sma(df, short)
    add_sma(df, long)
    add_volume_analysis(df)

    s_col, l_col = f"SMA_{short}", f"SMA_{long}"
    gap = df[s_col].iloc[-1] - df[l_col].iloc[-1]

    # Sænk den sidste pris så SMA_short krydser under SMA_long
    net_per_unit = (1.0 / short) - (1.0 / long)
    needed_drop = (gap / net_per_unit) + 1.0
    prices[-1] -= needed_drop

    df = _make_df(prices)
    add_sma(df, short)
    add_sma(df, long)
    add_volume_analysis(df)
    return df


def _make_oversold_df(period: int = 14) -> pd.DataFrame:
    """Konstruér data med kraftigt fald → RSI < 30."""
    n = period + 50
    prices = [100.0]
    for i in range(1, n):
        if i < n - 15:
            prices.append(prices[-1] + np.random.uniform(-0.3, 0.5))
        else:
            prices.append(prices[-1] - 3.0)  # kraftigt fald

    df = _make_df(prices)
    add_rsi(df, period=period)
    return df


def _make_overbought_df(period: int = 14) -> pd.DataFrame:
    """Konstruér data med kraftig stigning → RSI > 70."""
    n = period + 50
    prices = [100.0]
    for i in range(1, n):
        if i < n - 15:
            prices.append(prices[-1] + np.random.uniform(-0.5, 0.3))
        else:
            prices.append(prices[-1] + 3.0)  # kraftig stigning

    df = _make_df(prices)
    add_rsi(df, period=period)
    return df


def _make_neutral_df() -> pd.DataFrame:
    """Sidelæns data – ingen klar trend. RSI omkring 50."""
    np.random.seed(99)
    n = 200
    prices = 100 + np.cumsum(np.random.randn(n) * 0.3)
    df = _make_df(prices.tolist())
    add_all_indicators(df)
    return df


# ── BaseStrategy tests ──────────────────────────────────────

class TestBaseStrategy:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseStrategy()

    def test_strategy_result_clamps_confidence(self):
        r = StrategyResult(Signal.BUY, 150, "test")
        assert r.confidence == 100

        r2 = StrategyResult(Signal.SELL, -10, "test")
        assert r2.confidence == 0

    def test_position_size_hold_is_zero(self):
        class DummyStrategy(BaseStrategy):
            @property
            def name(self):
                return "dummy"
            def analyze(self, df):
                return StrategyResult(Signal.HOLD, 50, "test")

        s = DummyStrategy()
        result = StrategyResult(Signal.HOLD, 50, "")
        assert s.get_position_size(result, 100_000) == 0.0

    def test_position_size_scales_with_confidence(self):
        class DummyStrategy(BaseStrategy):
            @property
            def name(self):
                return "dummy"
            def analyze(self, df):
                return StrategyResult(Signal.BUY, 50, "test")

        s = DummyStrategy()
        # confidence=50, max_pos=5%, portfolio=100k → 50% af 5000 = 2500
        r = StrategyResult(Signal.BUY, 50, "")
        assert s.get_position_size(r, 100_000, 0.05) == 2500.0

        # confidence=100 → 5000
        r2 = StrategyResult(Signal.BUY, 100, "")
        assert s.get_position_size(r2, 100_000, 0.05) == 5000.0

    def test_validate_data(self):
        class DummyStrategy(BaseStrategy):
            @property
            def name(self):
                return "dummy"
            def analyze(self, df):
                return StrategyResult(Signal.HOLD, 0, "")

        s = DummyStrategy()
        df = _make_df([1, 2, 3])
        assert s.validate_data(df, 3) is True
        assert s.validate_data(df, 4) is False


# ── SMA Crossover tests ─────────────────────────────────────

class TestSMACrossover:

    def test_golden_cross_gives_buy(self):
        df = _make_golden_cross_df(short=20, long=50)
        strategy = SMACrossoverStrategy(short_window=20, long_window=50)
        result = strategy.analyze(df)
        assert result.signal == Signal.BUY
        assert result.confidence > 0
        assert "Golden cross" in result.reason

    def test_death_cross_gives_sell(self):
        df = _make_death_cross_df(short=20, long=50)
        strategy = SMACrossoverStrategy(short_window=20, long_window=50)
        result = strategy.analyze(df)
        assert result.signal == Signal.SELL
        assert result.confidence > 0
        assert "Death cross" in result.reason

    def test_neutral_gives_hold(self):
        df = _make_neutral_df()
        strategy = SMACrossoverStrategy(short_window=20, long_window=50)
        result = strategy.analyze(df)
        assert result.signal == Signal.HOLD

    def test_insufficient_data_gives_hold(self):
        df = _make_df([100, 101, 102])
        strategy = SMACrossoverStrategy(short_window=20, long_window=50)
        result = strategy.analyze(df)
        assert result.signal == Signal.HOLD
        assert "data" in result.reason.lower()

    def test_custom_windows(self):
        strategy = SMACrossoverStrategy(short_window=5, long_window=15)
        assert "5/15" in strategy.name

    def test_confidence_range(self):
        df = _make_golden_cross_df()
        strategy = SMACrossoverStrategy(short_window=20, long_window=50)
        result = strategy.analyze(df)
        assert 0 <= result.confidence <= 100

    def test_adds_sma_if_missing(self):
        """Strategien bør selv tilføje SMA-kolonner hvis de mangler."""
        prices = list(range(50, 130))
        df = _make_df(prices)
        assert "SMA_20" not in df.columns

        strategy = SMACrossoverStrategy(short_window=20, long_window=50)
        result = strategy.analyze(df)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


# ── RSI Strategy tests ──────────────────────────────────────

class TestRSIStrategy:

    def test_oversold_gives_buy(self):
        df = _make_oversold_df()
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        result = strategy.analyze(df)
        assert result.signal == Signal.BUY
        assert result.confidence >= 50
        assert "oversolgt" in result.reason

    def test_overbought_gives_sell(self):
        df = _make_overbought_df()
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        result = strategy.analyze(df)
        assert result.signal == Signal.SELL
        assert result.confidence >= 50
        assert "overkøbt" in result.reason

    def test_neutral_gives_hold(self):
        df = _make_neutral_df()
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)
        result = strategy.analyze(df)
        assert result.signal == Signal.HOLD

    def test_insufficient_data_gives_hold(self):
        df = _make_df([100, 101, 102])
        strategy = RSIStrategy(period=14)
        result = strategy.analyze(df)
        assert result.signal == Signal.HOLD

    def test_custom_thresholds(self):
        strategy = RSIStrategy(period=14, oversold=25, overbought=75)
        assert "25/75" in strategy.name

    def test_extreme_rsi_high_confidence(self):
        """RSI = 10 bør give højere confidence end RSI = 28."""
        df_extreme = _make_oversold_df()
        strategy = RSIStrategy(period=14, oversold=30, overbought=70)

        # Forcér ekstrem RSI via .loc
        df_extreme.loc[df_extreme.index[-1], "RSI"] = 10.0
        df_extreme.loc[df_extreme.index[-2], "RSI"] = 12.0
        r_extreme = strategy.analyze(df_extreme)

        df_mild = df_extreme.copy()
        df_mild.loc[df_mild.index[-1], "RSI"] = 28.0
        df_mild.loc[df_mild.index[-2], "RSI"] = 29.0
        r_mild = strategy.analyze(df_mild)

        assert r_extreme.confidence > r_mild.confidence

    def test_adds_rsi_if_missing(self):
        prices = [100 - i * 0.5 for i in range(60)]
        df = _make_df(prices)
        assert "RSI" not in df.columns

        strategy = RSIStrategy(period=14)
        result = strategy.analyze(df)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)


# ── Combined Strategy tests ──────────────────────────────────

class TestCombinedStrategy:

    def _make_stub(self, signal: Signal, confidence: float = 60) -> BaseStrategy:
        """Lav en stub-strategi der altid returnerer det givne signal."""
        class StubStrategy(BaseStrategy):
            def __init__(self, sig, conf):
                self._sig = sig
                self._conf = conf
            @property
            def name(self):
                return f"Stub({self._sig.value})"
            def analyze(self, df):
                return StrategyResult(self._sig, self._conf, "stub")

        return StubStrategy(signal, confidence)

    def test_consensus_buy(self):
        strategies = [
            (self._make_stub(Signal.BUY, 80), 1.0),
            (self._make_stub(Signal.BUY, 60), 1.0),
            (self._make_stub(Signal.HOLD, 0), 1.0),
        ]
        combined = CombinedStrategy(strategies, min_agreement=2)
        df = _make_neutral_df()
        result = combined.analyze(df)

        assert result.signal == Signal.BUY
        assert result.confidence > 0
        assert "2/3" in result.reason

    def test_consensus_sell(self):
        strategies = [
            (self._make_stub(Signal.SELL, 70), 1.0),
            (self._make_stub(Signal.SELL, 90), 1.0),
            (self._make_stub(Signal.BUY, 50), 1.0),
        ]
        combined = CombinedStrategy(strategies, min_agreement=2)
        result = combined.analyze(_make_neutral_df())

        assert result.signal == Signal.SELL

    def test_no_consensus_gives_hold(self):
        strategies = [
            (self._make_stub(Signal.BUY, 80), 1.0),
            (self._make_stub(Signal.SELL, 80), 1.0),
            (self._make_stub(Signal.HOLD, 0), 1.0),
        ]
        combined = CombinedStrategy(strategies, min_agreement=2)
        result = combined.analyze(_make_neutral_df())

        assert result.signal == Signal.HOLD
        assert "Ingen konsensus" in result.reason

    def test_all_hold_gives_hold(self):
        strategies = [
            (self._make_stub(Signal.HOLD, 0), 1.0),
            (self._make_stub(Signal.HOLD, 0), 1.0),
        ]
        combined = CombinedStrategy(strategies, min_agreement=2)
        result = combined.analyze(_make_neutral_df())

        assert result.signal == Signal.HOLD

    def test_weighted_confidence(self):
        """Høj vægt + høj confidence bør dominere."""
        strategies = [
            (self._make_stub(Signal.BUY, 90), 3.0),  # høj vægt
            (self._make_stub(Signal.BUY, 30), 1.0),  # lav vægt
        ]
        combined = CombinedStrategy(strategies, min_agreement=2)
        result = combined.analyze(_make_neutral_df())

        assert result.signal == Signal.BUY
        # Vægtet: (90*0.75 + 30*0.25) / 1.0 = 75
        assert result.confidence == pytest.approx(75.0, abs=1)

    def test_min_agreement_3_requires_all_3(self):
        strategies = [
            (self._make_stub(Signal.BUY, 80), 1.0),
            (self._make_stub(Signal.BUY, 60), 1.0),
            (self._make_stub(Signal.HOLD, 0), 1.0),
        ]
        combined = CombinedStrategy(strategies, min_agreement=3)
        result = combined.analyze(_make_neutral_df())

        # Kun 2 ud af 3 siger BUY → HOLD
        assert result.signal == Signal.HOLD

    def test_requires_at_least_2_strategies(self):
        with pytest.raises(ValueError, match="mindst 2"):
            CombinedStrategy(
                [(self._make_stub(Signal.BUY), 1.0)],
                min_agreement=1,
            )

    def test_with_real_strategies_neutral_data(self):
        """Integrationstest: ægte strategier med neutral data → HOLD."""
        sma = SMACrossoverStrategy(short_window=20, long_window=50)
        rsi = RSIStrategy(period=14)
        combined = CombinedStrategy(
            [(sma, 1.0), (rsi, 1.0)],
            min_agreement=2,
        )

        df = _make_neutral_df()
        result = combined.analyze(df)
        # Med neutral data bør ingen strategi give klart signal
        assert result.signal == Signal.HOLD
