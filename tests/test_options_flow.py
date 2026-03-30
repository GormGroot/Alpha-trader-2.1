"""
Tests for OptionsFlowTracker – options flow, UOA, P/C ratio, max pain, IV.

Alle yfinance API-kald mockes – ingen netværksforbindelse kræves.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.options_flow import (
    OptionsFlowTracker,
    UnusualOption,
    PutCallRatio,
    MaxPainResult,
    IVAnalysis,
    OptionsFlowSummary,
    UOA_VOLUME_MULTIPLIER,
    UOA_MIN_PREMIUM,
    BLOCK_TRADE_THRESHOLD,
)


# ── Helpers ──────────────────────────────────────────────────

def _tmp_cache_dir() -> str:
    return tempfile.mkdtemp()


def _make_uoa(
    symbol: str = "AAPL",
    strike: float = 150.0,
    option_type: str = "call",
    volume: int = 10_000,
    oi: int = 1_000,
    premium: float = 500_000,
    is_block: bool = False,
    vol_vs_normal: float = 10.0,
) -> UnusualOption:
    return UnusualOption(
        symbol=symbol,
        expiration="2026-04-17",
        strike=strike,
        option_type=option_type,
        volume=volume,
        open_interest=oi,
        implied_volatility=0.35,
        last_price=5.0,
        bid=4.90,
        ask=5.10,
        premium_total=premium,
        volume_oi_ratio=vol_vs_normal,
        volume_vs_normal=vol_vs_normal,
        is_block_trade=is_block,
        in_the_money=False,
    )


def _make_options_df(n: int = 10, base_strike: float = 140.0) -> pd.DataFrame:
    """Opret syntetisk options chain DataFrame."""
    rng = np.random.RandomState(42)
    strikes = base_strike + np.arange(n) * 5.0
    return pd.DataFrame({
        "strike": strikes,
        "volume": rng.randint(100, 5000, n),
        "openInterest": rng.randint(500, 10000, n),
        "impliedVolatility": 0.20 + rng.uniform(0, 0.30, n),
        "lastPrice": np.maximum(0.5, 10 - np.arange(n) * 0.5 + rng.uniform(-1, 1, n)),
        "bid": np.maximum(0.1, 9.5 - np.arange(n) * 0.5),
        "ask": np.maximum(0.3, 10.5 - np.arange(n) * 0.5),
        "inTheMoney": [i < n // 2 for i in range(n)],
        "contractSymbol": [f"AAPL260417C{int(s):05d}000" for s in strikes],
    })


def _make_high_volume_options_df() -> pd.DataFrame:
    """Options chain med én usædvanlig høj volume entry."""
    return pd.DataFrame({
        "strike": [145.0, 150.0, 155.0],
        "volume": [200, 50_000, 300],          # 50k vol at 150
        "openInterest": [1000, 2000, 1500],
        "impliedVolatility": [0.25, 0.30, 0.28],
        "lastPrice": [8.0, 5.0, 2.0],
        "bid": [7.80, 4.90, 1.90],
        "ask": [8.20, 5.10, 2.10],
        "inTheMoney": [True, False, False],
        "contractSymbol": ["AAPL260417C00145000", "AAPL260417C00150000", "AAPL260417C00155000"],
    })


# ── Test UnusualOption ───────────────────────────────────────

class TestUnusualOption:
    def test_bullish_signal(self):
        u = _make_uoa(option_type="call", vol_vs_normal=10.0)
        assert u.signal == "BULLISH"

    def test_bearish_signal(self):
        u = _make_uoa(option_type="put", vol_vs_normal=10.0)
        assert u.signal == "BEARISH"

    def test_neutral_signal_low_volume(self):
        u = _make_uoa(option_type="call", vol_vs_normal=2.0)
        assert u.signal == "NEUTRAL"

    def test_alert_text_call(self):
        u = _make_uoa(option_type="call", strike=150.0)
        text = u.alert_text
        assert "CALL" in text
        assert "AAPL" in text
        assert "150" in text
        assert "🟢" in text

    def test_alert_text_put(self):
        u = _make_uoa(option_type="put")
        assert "🔴" in u.alert_text
        assert "PUT" in u.alert_text

    def test_itm_shown(self):
        u = UnusualOption(
            symbol="AAPL", expiration="2026-04-17", strike=150.0,
            option_type="call", volume=10000, open_interest=1000,
            implied_volatility=0.3, last_price=5.0, bid=4.9, ask=5.1,
            premium_total=500000, volume_oi_ratio=10.0,
            volume_vs_normal=10.0, is_block_trade=False, in_the_money=True,
        )
        assert "(ITM)" in u.alert_text


# ── Test PutCallRatio ────────────────────────────────────────

class TestPutCallRatio:
    def test_high_ratio_bullish_interpretation(self):
        pcr = PutCallRatio(
            symbol="AAPL", put_volume=5000, call_volume=3000,
            ratio=1.67, put_oi=10000, call_oi=8000,
            oi_ratio=1.25, signal="bullish",
        )
        assert "contrarian BULLISH" in pcr.interpretation

    def test_low_ratio_bearish_interpretation(self):
        pcr = PutCallRatio(
            symbol="AAPL", put_volume=1000, call_volume=5000,
            ratio=0.2, put_oi=3000, call_oi=15000,
            oi_ratio=0.2, signal="bearish",
        )
        assert "contrarian" in pcr.interpretation.lower()

    def test_normal_ratio(self):
        pcr = PutCallRatio(
            symbol="AAPL", put_volume=3000, call_volume=3000,
            ratio=1.0, put_oi=8000, call_oi=8000,
            oi_ratio=1.0, signal="neutral",
        )
        assert "neutral" in pcr.interpretation.lower()

    def test_very_low_ratio(self):
        pcr = PutCallRatio(
            symbol="AAPL", put_volume=100, call_volume=10000,
            ratio=0.01, put_oi=500, call_oi=50000,
            oi_ratio=0.01, signal="bearish",
        )
        assert "WARNING" in pcr.interpretation or "contrarian" in pcr.interpretation.lower()


# ── Test MaxPainResult ───────────────────────────────────────

class TestMaxPainResult:
    def test_direction_down(self):
        mp = MaxPainResult(
            symbol="AAPL", expiration="2026-04-17",
            max_pain_price=145.0, current_price=155.0,
            distance_pct=6.9,
        )
        assert "NED" in mp.direction

    def test_direction_up(self):
        mp = MaxPainResult(
            symbol="AAPL", expiration="2026-04-17",
            max_pain_price=160.0, current_price=150.0,
            distance_pct=-6.25,
        )
        assert "OP" in mp.direction

    def test_direction_close(self):
        mp = MaxPainResult(
            symbol="AAPL", expiration="2026-04-17",
            max_pain_price=150.0, current_price=151.0,
            distance_pct=0.67,
        )
        assert "Tæt på" in mp.direction


# ── Test IVAnalysis ──────────────────────────────────────────

class TestIVAnalysis:
    def test_alert_high_iv(self):
        iv = IVAnalysis(
            symbol="AAPL", current_iv=0.50, historical_vol=0.25,
            iv_rank=92.0, iv_percentile=95.0, iv_hv_ratio=2.0,
            iv_high_52w=0.55, iv_low_52w=0.15, is_elevated=True,
        )
        assert iv.alert_text is not None
        assert "92%" in iv.alert_text
        assert "🔥" in iv.alert_text

    def test_alert_moderate_iv(self):
        iv = IVAnalysis(
            symbol="AAPL", current_iv=0.35, historical_vol=0.25,
            iv_rank=75.0, iv_percentile=70.0, iv_hv_ratio=1.4,
            iv_high_52w=0.50, iv_low_52w=0.15, is_elevated=True,
        )
        assert iv.alert_text is not None
        assert "⚡" in iv.alert_text

    def test_no_alert_low_iv(self):
        iv = IVAnalysis(
            symbol="AAPL", current_iv=0.20, historical_vol=0.22,
            iv_rank=30.0, iv_percentile=25.0, iv_hv_ratio=0.91,
            iv_high_52w=0.50, iv_low_52w=0.15, is_elevated=False,
        )
        assert iv.alert_text is None

    def test_interpretation_high(self):
        iv = IVAnalysis(
            symbol="AAPL", current_iv=0.50, historical_vol=0.25,
            iv_rank=85.0, iv_percentile=90.0, iv_hv_ratio=2.0,
            iv_high_52w=0.55, iv_low_52w=0.15, is_elevated=True,
        )
        assert "sælge premium" in iv.interpretation

    def test_interpretation_low(self):
        iv = IVAnalysis(
            symbol="AAPL", current_iv=0.15, historical_vol=0.20,
            iv_rank=25.0, iv_percentile=20.0, iv_hv_ratio=0.75,
            iv_high_52w=0.50, iv_low_52w=0.10, is_elevated=False,
        )
        assert "købe premium" in iv.interpretation


# ── Test OptionsFlowSummary ──────────────────────────────────

class TestOptionsFlowSummary:
    def test_confidence_bullish_uoa(self):
        uoa = [
            _make_uoa(option_type="call", vol_vs_normal=10.0),
            _make_uoa(option_type="call", vol_vs_normal=8.0),
        ]
        summary = OptionsFlowSummary(
            symbol="AAPL",
            unusual_activity=uoa,
            put_call_ratio=None,
            max_pain=None,
            iv_analysis=None,
        )
        assert summary.confidence_adjustment > 0

    def test_confidence_bearish_uoa(self):
        uoa = [
            _make_uoa(option_type="put", vol_vs_normal=10.0),
            _make_uoa(option_type="put", vol_vs_normal=8.0),
            _make_uoa(option_type="put", vol_vs_normal=6.0),
        ]
        summary = OptionsFlowSummary(
            symbol="AAPL",
            unusual_activity=uoa,
            put_call_ratio=None,
            max_pain=None,
            iv_analysis=None,
        )
        assert summary.confidence_adjustment < 0

    def test_confidence_bounded(self):
        uoa = [_make_uoa(option_type="call", vol_vs_normal=10.0) for _ in range(20)]
        summary = OptionsFlowSummary(
            symbol="AAPL",
            unusual_activity=uoa,
            put_call_ratio=None,
            max_pain=None,
            iv_analysis=None,
        )
        assert -10 <= summary.confidence_adjustment <= 10

    def test_pcr_contrarian_boost(self):
        pcr = PutCallRatio(
            symbol="AAPL", put_volume=10000, call_volume=5000,
            ratio=2.0, put_oi=20000, call_oi=10000,
            oi_ratio=2.0, signal="bullish",
        )
        summary = OptionsFlowSummary(
            symbol="AAPL",
            unusual_activity=[],
            put_call_ratio=pcr,
            max_pain=None,
            iv_analysis=None,
        )
        assert summary.confidence_adjustment > 0

    def test_empty_summary_neutral(self):
        summary = OptionsFlowSummary(
            symbol="AAPL",
            unusual_activity=[],
            put_call_ratio=None,
            max_pain=None,
            iv_analysis=None,
        )
        assert summary.confidence_adjustment == 0


# ── Test OptionsFlowTracker Init ─────────────────────────────

class TestOptionsFlowTrackerInit:
    def test_creates_db(self):
        cache_dir = _tmp_cache_dir()
        tracker = OptionsFlowTracker(cache_dir=cache_dir)
        db_path = Path(cache_dir) / "options_flow.db"
        assert db_path.exists()

    def test_db_tables(self):
        cache_dir = _tmp_cache_dir()
        tracker = OptionsFlowTracker(cache_dir=cache_dir)
        db_path = Path(cache_dir) / "options_flow.db"
        conn = sqlite3.connect(db_path)
        tables = {t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "unusual_options" in tables
        assert "put_call_ratio" in tables
        assert "iv_history" in tables
        conn.close()


# ── Test UOA Detection ───────────────────────────────────────

class TestUOADetection:
    def test_detects_high_volume(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        calls = _make_high_volume_options_df()
        puts = _make_options_df(n=3, base_strike=145.0)

        with patch.object(tracker, "_get_options_chain", return_value=(calls, puts, 150.0)):
            uoa = tracker.detect_unusual_activity("AAPL")

        # 50k volume / 2k OI = 25x → bør detekteres
        high_vol = [u for u in uoa if u.strike == 150.0 and u.option_type == "call"]
        assert len(high_vol) > 0

    def test_no_uoa_normal_volume(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        calls = pd.DataFrame({
            "strike": [150.0],
            "volume": [100],
            "openInterest": [5000],
            "impliedVolatility": [0.25],
            "lastPrice": [0.50],
            "bid": [0.45],
            "ask": [0.55],
            "inTheMoney": [False],
            "contractSymbol": ["AAPL260417C00150000"],
        })
        puts = calls.copy()
        puts["contractSymbol"] = ["AAPL260417P00150000"]

        with patch.object(tracker, "_get_options_chain", return_value=(calls, puts, 150.0)):
            uoa = tracker.detect_unusual_activity("AAPL")

        assert len(uoa) == 0

    def test_no_chain_returns_empty(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        with patch.object(tracker, "_get_options_chain", return_value=None):
            uoa = tracker.detect_unusual_activity("AAPL")
        assert uoa == []

    def test_sorted_by_premium(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        calls = pd.DataFrame({
            "strike": [145.0, 150.0],
            "volume": [20_000, 50_000],
            "openInterest": [1000, 2000],
            "impliedVolatility": [0.25, 0.30],
            "lastPrice": [10.0, 5.0],
            "bid": [9.80, 4.90],
            "ask": [10.20, 5.10],
            "inTheMoney": [True, False],
            "contractSymbol": ["AAPL260417C00145000", "AAPL260417C00150000"],
        })
        puts = pd.DataFrame(columns=calls.columns)

        with patch.object(tracker, "_get_options_chain", return_value=(calls, puts, 150.0)):
            uoa = tracker.detect_unusual_activity("AAPL", min_premium=0)

        if len(uoa) >= 2:
            assert uoa[0].premium_total >= uoa[1].premium_total


# ── Test UOA Cache ───────────────────────────────────────────

class TestUOACache:
    def test_write_and_read(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        uoa = [_make_uoa()]
        tracker._write_uoa_cache(uoa)
        cached = tracker.get_recent_uoa("AAPL", hours=24)
        assert len(cached) == 1
        assert cached[0].symbol == "AAPL"
        assert cached[0].strike == 150.0


# ── Test Put/Call Ratio ──────────────────────────────────────

class TestPutCallRatioCalc:
    def test_computes_ratio(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        calls = _make_options_df(n=5, base_strike=140.0)
        puts = _make_options_df(n=5, base_strike=140.0)
        # Sæt puts volume højere
        puts["volume"] = puts["volume"] * 3

        chains = [("2026-04-17", calls, puts)]
        with patch.object(tracker, "_get_all_chains", return_value=chains):
            pcr = tracker.get_put_call_ratio("AAPL")

        assert pcr is not None
        assert pcr.ratio > 1.0  # Puts har højere volume
        assert pcr.symbol == "AAPL"

    def test_no_chains_returns_none(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        with patch.object(tracker, "_get_all_chains", return_value=None):
            pcr = tracker.get_put_call_ratio("AAPL")
        assert pcr is None

    def test_zero_call_volume(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        calls = pd.DataFrame({
            "strike": [150.0], "volume": [0], "openInterest": [1000],
            "impliedVolatility": [0.25], "lastPrice": [5.0],
        })
        puts = pd.DataFrame({
            "strike": [150.0], "volume": [1000], "openInterest": [2000],
            "impliedVolatility": [0.25], "lastPrice": [3.0],
        })
        chains = [("2026-04-17", calls, puts)]
        with patch.object(tracker, "_get_all_chains", return_value=chains):
            pcr = tracker.get_put_call_ratio("AAPL")

        assert pcr is not None
        assert pcr.ratio == 99.0  # Capped inf

    def test_signal_bullish_high_pcr(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        calls = _make_options_df(n=5)
        puts = _make_options_df(n=5)
        puts["volume"] = puts["volume"] * 5

        chains = [("2026-04-17", calls, puts)]
        with patch.object(tracker, "_get_all_chains", return_value=chains):
            pcr = tracker.get_put_call_ratio("AAPL")

        assert pcr.signal == "bullish"  # Contrarian

    def test_market_pcr(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        calls = _make_options_df(n=5)
        puts = _make_options_df(n=5)

        chains = [("2026-04-17", calls, puts)]
        with patch.object(tracker, "_get_all_chains", return_value=chains):
            pcr = tracker.get_market_pcr("SPY")

        assert pcr is not None
        assert pcr.symbol == "SPY"


# ── Test Max Pain ────────────────────────────────────────────

class TestMaxPain:
    def test_calculates_max_pain(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        calls = pd.DataFrame({
            "strike": [140.0, 145.0, 150.0, 155.0, 160.0],
            "openInterest": [500, 1000, 5000, 2000, 500],
            "volume": [100, 200, 500, 200, 100],
            "impliedVolatility": [0.3] * 5,
            "lastPrice": [12, 8, 5, 2, 0.5],
            "bid": [11, 7, 4, 1.5, 0.3],
            "ask": [13, 9, 6, 2.5, 0.7],
            "inTheMoney": [True, True, False, False, False],
            "contractSymbol": [f"C{s}" for s in range(5)],
        })
        puts = pd.DataFrame({
            "strike": [140.0, 145.0, 150.0, 155.0, 160.0],
            "openInterest": [500, 2000, 5000, 1000, 500],
            "volume": [100, 200, 500, 200, 100],
            "impliedVolatility": [0.3] * 5,
            "lastPrice": [0.5, 2, 5, 8, 12],
            "bid": [0.3, 1.5, 4, 7, 11],
            "ask": [0.7, 2.5, 6, 9, 13],
            "inTheMoney": [False, False, False, True, True],
            "contractSymbol": [f"P{s}" for s in range(5)],
        })

        with patch.object(tracker, "_get_options_chain", return_value=(calls, puts, 152.0)):
            mp = tracker.calculate_max_pain("AAPL")

        assert mp is not None
        assert mp.symbol == "AAPL"
        assert mp.max_pain_price > 0
        assert mp.current_price == 152.0
        # Max pain bør være nær 150 (højest OI for både calls og puts)
        assert 140 <= mp.max_pain_price <= 160

    def test_no_chain_returns_none(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        with patch.object(tracker, "_get_options_chain", return_value=None):
            mp = tracker.calculate_max_pain("AAPL")
        assert mp is None

    def test_empty_chain_returns_none(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        empty = pd.DataFrame()
        with patch.object(tracker, "_get_options_chain", return_value=(empty, empty, 150.0)):
            mp = tracker.calculate_max_pain("AAPL")
        assert mp is None


# ── Test IV Analysis ─────────────────────────────────────────

class TestIVCalcCurrentIV:
    def test_near_money_iv(self):
        calls = pd.DataFrame({
            "strike": [140.0, 145.0, 150.0, 155.0, 160.0],
            "impliedVolatility": [0.35, 0.30, 0.28, 0.30, 0.35],
        })
        puts = pd.DataFrame({
            "strike": [140.0, 145.0, 150.0, 155.0, 160.0],
            "impliedVolatility": [0.36, 0.31, 0.29, 0.31, 0.36],
        })
        iv = OptionsFlowTracker._calc_current_iv(calls, puts, current_price=150.0)
        # Alle strikes er ±10% af 150, so avg of all
        assert 0.25 < iv < 0.40

    def test_empty_df(self):
        empty = pd.DataFrame(columns=["strike", "impliedVolatility"])
        iv = OptionsFlowTracker._calc_current_iv(empty, empty, 150.0)
        assert iv == 0.0


# ── Test IV Cache ────────────────────────────────────────────

class TestIVCache:
    def test_write_and_read_history(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        iv = IVAnalysis(
            symbol="AAPL", current_iv=0.30, historical_vol=0.25,
            iv_rank=60.0, iv_percentile=55.0, iv_hv_ratio=1.2,
            iv_high_52w=0.50, iv_low_52w=0.15, is_elevated=True,
        )
        tracker._write_iv_cache(iv)
        history = tracker._get_iv_history("AAPL")
        assert len(history) == 1
        assert history[0] == pytest.approx(0.30)


# ── Test PCR Cache ───────────────────────────────────────────

class TestPCRCache:
    def test_write_pcr(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        pcr = PutCallRatio(
            symbol="AAPL", put_volume=3000, call_volume=5000,
            ratio=0.6, put_oi=8000, call_oi=12000,
            oi_ratio=0.67, signal="neutral",
        )
        tracker._write_pcr_cache(pcr)
        # Verify in DB
        with tracker._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM put_call_ratio WHERE symbol = 'AAPL'"
            ).fetchall()
        assert len(rows) == 1


# ── Test Full Summary ────────────────────────────────────────

class TestFullSummary:
    def test_summary_with_all_data(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())

        mock_uoa = [_make_uoa(option_type="call", vol_vs_normal=10.0)]
        mock_pcr = PutCallRatio(
            symbol="AAPL", put_volume=3000, call_volume=5000,
            ratio=0.6, put_oi=8000, call_oi=12000,
            oi_ratio=0.67, signal="neutral",
        )
        mock_mp = MaxPainResult(
            symbol="AAPL", expiration="2026-04-17",
            max_pain_price=148.0, current_price=152.0,
            distance_pct=2.7,
        )
        mock_iv = IVAnalysis(
            symbol="AAPL", current_iv=0.35, historical_vol=0.25,
            iv_rank=80.0, iv_percentile=75.0, iv_hv_ratio=1.4,
            iv_high_52w=0.50, iv_low_52w=0.15, is_elevated=True,
        )

        with patch.object(tracker, "detect_unusual_activity", return_value=mock_uoa), \
             patch.object(tracker, "get_put_call_ratio", return_value=mock_pcr), \
             patch.object(tracker, "calculate_max_pain", return_value=mock_mp), \
             patch.object(tracker, "analyze_iv", return_value=mock_iv):
            summary = tracker.get_options_flow_summary("AAPL")

        assert summary.symbol == "AAPL"
        assert len(summary.alerts) > 0
        assert summary.overall_signal in ("bullish", "bearish", "neutral")

    def test_summary_handles_errors(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())

        with patch.object(tracker, "detect_unusual_activity", side_effect=Exception("fail")), \
             patch.object(tracker, "get_put_call_ratio", side_effect=Exception("fail")), \
             patch.object(tracker, "calculate_max_pain", side_effect=Exception("fail")), \
             patch.object(tracker, "analyze_iv", side_effect=Exception("fail")):
            summary = tracker.get_options_flow_summary("AAPL")

        assert summary.symbol == "AAPL"
        assert summary.overall_signal == "neutral"
        assert len(summary.unusual_activity) == 0

    def test_confidence_adjustment_method(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        mock_summary = OptionsFlowSummary(
            symbol="AAPL",
            unusual_activity=[_make_uoa(option_type="call", vol_vs_normal=10.0)],
            put_call_ratio=None,
            max_pain=None,
            iv_analysis=None,
        )
        with patch.object(tracker, "get_options_flow_summary", return_value=mock_summary):
            adj = tracker.get_confidence_adjustment("AAPL")
        assert isinstance(adj, int)


# ── Test Explain ─────────────────────────────────────────────

class TestExplain:
    def test_explain_contains_sections(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())

        mock_summary = OptionsFlowSummary(
            symbol="AAPL",
            unusual_activity=[_make_uoa()],
            put_call_ratio=PutCallRatio(
                symbol="AAPL", put_volume=3000, call_volume=5000,
                ratio=0.6, put_oi=8000, call_oi=12000,
                oi_ratio=0.67, signal="neutral",
            ),
            max_pain=MaxPainResult(
                symbol="AAPL", expiration="2026-04-17",
                max_pain_price=148.0, current_price=152.0,
                distance_pct=2.7,
            ),
            iv_analysis=IVAnalysis(
                symbol="AAPL", current_iv=0.30, historical_vol=0.25,
                iv_rank=60.0, iv_percentile=55.0, iv_hv_ratio=1.2,
                iv_high_52w=0.50, iv_low_52w=0.15, is_elevated=True,
            ),
            overall_signal="neutral",
        )

        with patch.object(tracker, "get_options_flow_summary", return_value=mock_summary):
            text = tracker.explain("AAPL")

        assert "OPTIONS FLOW RAPPORT" in text
        assert "USÆDVANLIG OPTIONS AKTIVITET" in text
        assert "PUT/CALL RATIO" in text
        assert "MAX PAIN" in text
        assert "IMPLIED VOLATILITY" in text
        assert "SAMLET VURDERING" in text

    def test_print_report(self, capsys):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())

        mock_summary = OptionsFlowSummary(
            symbol="AAPL",
            unusual_activity=[],
            put_call_ratio=None,
            max_pain=None,
            iv_analysis=None,
            overall_signal="neutral",
        )

        with patch.object(tracker, "get_options_flow_summary", return_value=mock_summary):
            tracker.print_report("AAPL")

        captured = capsys.readouterr()
        assert "OPTIONS FLOW RAPPORT" in captured.out


# ── Test Scan & IV Ranking ───────────────────────────────────

class TestScanAndRanking:
    def test_scan_symbols(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        mock_summary = OptionsFlowSummary(
            symbol="AAPL",
            unusual_activity=[],
            put_call_ratio=None,
            max_pain=None,
            iv_analysis=None,
        )
        with patch.object(tracker, "get_options_flow_summary", return_value=mock_summary):
            results = tracker.scan_symbols(["AAPL", "MSFT"])
        assert len(results) == 2

    def test_scan_handles_errors(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        with patch.object(tracker, "get_options_flow_summary", side_effect=Exception("fail")):
            results = tracker.scan_symbols(["AAPL"])
        assert len(results) == 0

    def test_iv_ranking(self):
        tracker = OptionsFlowTracker(cache_dir=_tmp_cache_dir())
        iv_aapl = IVAnalysis(
            symbol="AAPL", current_iv=0.30, historical_vol=0.25,
            iv_rank=80.0, iv_percentile=75.0, iv_hv_ratio=1.2,
            iv_high_52w=0.50, iv_low_52w=0.15, is_elevated=True,
        )
        iv_msft = IVAnalysis(
            symbol="MSFT", current_iv=0.20, historical_vol=0.18,
            iv_rank=40.0, iv_percentile=35.0, iv_hv_ratio=1.1,
            iv_high_52w=0.35, iv_low_52w=0.12, is_elevated=False,
        )
        with patch.object(tracker, "analyze_iv", side_effect=[iv_aapl, iv_msft]):
            ranking = tracker.get_iv_ranking(["AAPL", "MSFT"])

        assert len(ranking) == 2
        assert ranking[0][0] == "AAPL"  # Højeste IV Rank først
        assert ranking[0][1] > ranking[1][1]
