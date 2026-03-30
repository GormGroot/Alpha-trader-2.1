"""Tests for src.data.onchain – On-chain krypto-analyse."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.onchain import (
    CRYPTO_IDS,
    ActiveAddresses,
    BitcoinDominance,
    DeFiMetrics,
    ExchangeFlowData,
    FearGreedIndex,
    FearGreedLevel,
    HashRateData,
    NVTRatio,
    OnChainReport,
    OnChainSignal,
    OnChainTracker,
    WhaleActivity,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def tracker(tmp_path: Path) -> OnChainTracker:
    """Tracker med tmp cache-dir."""
    return OnChainTracker(cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def _no_throttle(monkeypatch):
    """Fjern rate limiting i tests."""
    monkeypatch.setattr(OnChainTracker, "_throttle", lambda self: None)


# ══════════════════════════════════════════════════════════════
#  Dataclass-tests
# ══════════════════════════════════════════════════════════════


class TestFearGreedIndex:
    def test_score_extreme_fear(self):
        fg = FearGreedIndex(
            value=10, classification="Extreme Fear",
            level=FearGreedLevel.EXTREME_FEAR,
            timestamp="12345", contrarian_signal="buy",
        )
        assert fg.score == 90.0

    def test_score_extreme_greed(self):
        fg = FearGreedIndex(
            value=90, classification="Extreme Greed",
            level=FearGreedLevel.EXTREME_GREED,
            timestamp="12345", contrarian_signal="sell",
        )
        assert fg.score == 10.0

    def test_score_neutral(self):
        fg = FearGreedIndex(
            value=50, classification="Neutral",
            level=FearGreedLevel.NEUTRAL,
            timestamp="12345", contrarian_signal="neutral",
        )
        assert fg.score == 50.0


class TestNVTRatio:
    def test_overvalued(self):
        nvt = NVTRatio(nvt=120.0, signal="overvalued", description="High")
        assert nvt.is_overvalued is True
        assert nvt.is_undervalued is False

    def test_undervalued(self):
        nvt = NVTRatio(nvt=30.0, signal="undervalued", description="Low")
        assert nvt.is_overvalued is False
        assert nvt.is_undervalued is True

    def test_fair(self):
        nvt = NVTRatio(nvt=70.0, signal="fair", description="Fair")
        assert nvt.is_overvalued is False
        assert nvt.is_undervalued is False


class TestEnums:
    def test_fear_greed_levels(self):
        assert FearGreedLevel.EXTREME_FEAR.value == "extreme_fear"
        assert FearGreedLevel.EXTREME_GREED.value == "extreme_greed"

    def test_onchain_signals(self):
        assert OnChainSignal.STRONG_BULLISH.value == "strong_bullish"
        assert OnChainSignal.STRONG_BEARISH.value == "strong_bearish"
        assert OnChainSignal.NEUTRAL.value == "neutral"


# ══════════════════════════════════════════════════════════════
#  Tracker – init & cache
# ══════════════════════════════════════════════════════════════


class TestTrackerInit:
    def test_creates_cache_dir(self, tmp_path: Path):
        cache = tmp_path / "nested" / "cache"
        t = OnChainTracker(cache_dir=str(cache))
        assert cache.exists()
        assert (cache / "onchain_cache.db").exists()

    def test_db_tables_exist(self, tracker: OnChainTracker):
        with tracker._get_conn() as conn:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
        assert "onchain_cache" in tables
        assert "fear_greed_history" in tables


class TestCache:
    def test_write_and_read(self, tracker: OnChainTracker):
        tracker._write_cache("test_key", "hello")
        assert tracker._read_cache("test_key", max_age_hours=1.0) == "hello"

    def test_expired_cache(self, tracker: OnChainTracker):
        # Skriv med gammel timestamp
        with tracker._get_conn() as conn:
            old = (datetime.now() - timedelta(hours=5)).isoformat()
            conn.execute(
                "INSERT INTO onchain_cache (key, value, fetched_at) VALUES (?,?,?)",
                ("old_key", "old_value", old),
            )
        assert tracker._read_cache("old_key", max_age_hours=1.0) is None

    def test_missing_key(self, tracker: OnChainTracker):
        assert tracker._read_cache("nonexistent") is None


# ══════════════════════════════════════════════════════════════
#  API-metoder (alle mocked)
# ══════════════════════════════════════════════════════════════


class TestGetFearGreed:
    @pytest.fixture(autouse=True)
    def _setup(self, _no_throttle):
        pass

    def test_returns_fear_greed(self, tracker: OnChainTracker):
        mock_resp = {
            "data": [{
                "value": "15",
                "value_classification": "Extreme Fear",
                "timestamp": "1710000000",
            }]
        }
        with patch.object(tracker, "_get_json", return_value=mock_resp):
            fg = tracker.get_fear_greed()
        assert fg is not None
        assert fg.value == 15
        assert fg.level == FearGreedLevel.EXTREME_FEAR
        assert fg.contrarian_signal == "buy"

    def test_extreme_greed(self, tracker: OnChainTracker):
        mock_resp = {
            "data": [{"value": "85", "value_classification": "Extreme Greed", "timestamp": "0"}]
        }
        with patch.object(tracker, "_get_json", return_value=mock_resp):
            fg = tracker.get_fear_greed()
        assert fg.level == FearGreedLevel.EXTREME_GREED
        assert fg.contrarian_signal == "sell"

    def test_neutral_zone(self, tracker: OnChainTracker):
        mock_resp = {
            "data": [{"value": "50", "value_classification": "Neutral", "timestamp": "0"}]
        }
        with patch.object(tracker, "_get_json", return_value=mock_resp):
            fg = tracker.get_fear_greed()
        assert fg.level == FearGreedLevel.NEUTRAL
        assert fg.contrarian_signal == "neutral"

    def test_fear_zone(self, tracker: OnChainTracker):
        mock_resp = {
            "data": [{"value": "30", "value_classification": "Fear", "timestamp": "0"}]
        }
        with patch.object(tracker, "_get_json", return_value=mock_resp):
            fg = tracker.get_fear_greed()
        assert fg.level == FearGreedLevel.FEAR
        assert fg.contrarian_signal == "buy"

    def test_greed_zone(self, tracker: OnChainTracker):
        mock_resp = {
            "data": [{"value": "70", "value_classification": "Greed", "timestamp": "0"}]
        }
        with patch.object(tracker, "_get_json", return_value=mock_resp):
            fg = tracker.get_fear_greed()
        assert fg.level == FearGreedLevel.GREED
        assert fg.contrarian_signal == "sell"

    def test_none_on_api_error(self, tracker: OnChainTracker):
        with patch.object(tracker, "_get_json", return_value=None):
            assert tracker.get_fear_greed() is None

    def test_caches_result(self, tracker: OnChainTracker):
        mock_resp = {
            "data": [{"value": "50", "value_classification": "Neutral", "timestamp": "0"}]
        }
        with patch.object(tracker, "_get_json", return_value=mock_resp):
            tracker.get_fear_greed()

        # Anden kald bør ramme cache
        cached = tracker._read_cache("fear_greed", max_age_hours=6.0)
        assert cached is not None
        d = json.loads(cached)
        assert d["value"] == 50


class TestGetBtcDominance:
    @pytest.fixture(autouse=True)
    def _setup(self, _no_throttle):
        pass

    def test_btc_strength(self, tracker: OnChainTracker):
        mock_resp = {
            "data": {
                "market_cap_percentage": {"btc": 60.0, "eth": 18.0},
                "market_cap_change_percentage_24h_usd": -1.5,
            }
        }
        with patch.object(tracker, "_get_json", return_value=mock_resp):
            dom = tracker.get_btc_dominance()
        assert dom is not None
        assert dom.dominance_pct == 60.0
        assert dom.signal == "btc_strength"
        assert dom.alt_season is False

    def test_alt_season(self, tracker: OnChainTracker):
        mock_resp = {
            "data": {
                "market_cap_percentage": {"btc": 35.0},
                "market_cap_change_percentage_24h_usd": 2.0,
            }
        }
        with patch.object(tracker, "_get_json", return_value=mock_resp):
            dom = tracker.get_btc_dominance()
        assert dom.alt_season is True
        assert dom.signal == "alt_season"

    def test_neutral(self, tracker: OnChainTracker):
        mock_resp = {
            "data": {
                "market_cap_percentage": {"btc": 48.0},
                "market_cap_change_percentage_24h_usd": 0.5,
            }
        }
        with patch.object(tracker, "_get_json", return_value=mock_resp):
            dom = tracker.get_btc_dominance()
        assert dom.signal == "neutral"

    def test_none_on_error(self, tracker: OnChainTracker):
        with patch.object(tracker, "_get_json", return_value=None):
            assert tracker.get_btc_dominance() is None


class TestGetExchangeFlow:
    @pytest.fixture(autouse=True)
    def _setup(self, _no_throttle):
        pass

    def _make_stats(self, tx_vol: float = 5e9, price: float = 50000, n_tx: int = 300000):
        return {
            "n_tx": n_tx,
            "hash_rate": 5e20,
            "total_btc_sent": int(1000 * 1e8),
            "estimated_transaction_volume_usd": tx_vol,
            "market_price_usd": price,
        }

    def test_bearish_high_flow(self, tracker: OnChainTracker):
        # flow_ratio = 5e10 / (50000*21M) = ~0.047 < 0.05 so neutral actually
        # Let's make it > 0.05: tx_vol needs to be > 0.05 * 50000*21e6 = 52.5B
        stats = self._make_stats(tx_vol=60e9)
        with patch.object(tracker, "_get_json", return_value=stats):
            ef = tracker.get_exchange_flow()
        assert ef is not None
        assert ef.signal == "bearish"

    def test_bullish_low_flow(self, tracker: OnChainTracker):
        # flow_ratio < 0.01: tx_vol < 0.01 * 1.05T = 10.5B
        stats = self._make_stats(tx_vol=5e9)
        with patch.object(tracker, "_get_json", return_value=stats):
            ef = tracker.get_exchange_flow()
        assert ef.signal == "bullish"

    def test_net_flow_positive(self, tracker: OnChainTracker):
        stats = self._make_stats()
        with patch.object(tracker, "_get_json", return_value=stats):
            ef = tracker.get_exchange_flow()
        # inflow (30%) > outflow (25%) → net positive
        assert ef.net_flow > 0

    def test_none_on_error(self, tracker: OnChainTracker):
        with patch.object(tracker, "_get_json", return_value=None):
            assert tracker.get_exchange_flow() is None


class TestGetActiveAddresses:
    @pytest.fixture(autouse=True)
    def _setup(self, _no_throttle):
        pass

    def test_bullish_increasing(self, tracker: OnChainTracker):
        stats = {"n_tx": 500000, "n_btc_mined": 900}
        chart = {
            "values": [{"y": 800000 + i * 1000} for i in range(30)]
        }
        calls = [stats, chart]
        with patch.object(tracker, "_get_json", side_effect=calls):
            aa = tracker.get_active_addresses()
        assert aa is not None
        assert aa.count > 0

    def test_none_on_error(self, tracker: OnChainTracker):
        with patch.object(tracker, "_get_json", return_value=None):
            assert tracker.get_active_addresses() is None

    def test_with_flat_chart_data(self, tracker: OnChainTracker):
        stats = {"n_tx": 300000, "n_btc_mined": 900}
        chart = {"values": [{"y": 800000} for _ in range(30)]}
        with patch.object(tracker, "_get_json", side_effect=[stats, chart]):
            aa = tracker.get_active_addresses()
        assert aa is not None
        assert aa.signal == "neutral"
        assert aa.change_pct_7d == pytest.approx(0.0, abs=0.1)


class TestGetHashRate:
    @pytest.fixture(autouse=True)
    def _setup(self, _no_throttle):
        pass

    def test_bullish_rising(self, tracker: OnChainTracker):
        stats = {"hash_rate": 5e20}
        # Stigende hash rate over 60 dage
        vals = [{"y": 4e20 + i * 5e18} for i in range(60)]
        chart = {"values": vals}
        with patch.object(tracker, "_get_json", side_effect=[stats, chart]):
            hr = tracker.get_hash_rate()
        assert hr is not None
        assert hr.hash_rate > 0
        assert hr.signal == "bullish"

    def test_none_on_error(self, tracker: OnChainTracker):
        with patch.object(tracker, "_get_json", return_value=None):
            assert tracker.get_hash_rate() is None


class TestGetNVT:
    @pytest.fixture(autouse=True)
    def _setup(self, _no_throttle):
        pass

    def test_overvalued(self, tracker: OnChainTracker):
        # NVT = 50000 * 19.5M / tx_vol. For NVT > 95: tx_vol < 975e9/95 ≈ 10.26B
        stats = {
            "market_price_usd": 50000,
            "estimated_transaction_volume_usd": 5e9,
        }
        with patch.object(tracker, "_get_json", return_value=stats):
            nvt = tracker.get_nvt_ratio()
        assert nvt is not None
        assert nvt.nvt > 95
        assert nvt.signal == "overvalued"

    def test_undervalued(self, tracker: OnChainTracker):
        # NVT < 45: tx_vol > 975e9/45 ≈ 21.67B
        stats = {
            "market_price_usd": 50000,
            "estimated_transaction_volume_usd": 30e9,
        }
        with patch.object(tracker, "_get_json", return_value=stats):
            nvt = tracker.get_nvt_ratio()
        assert nvt.nvt < 45
        assert nvt.signal == "undervalued"

    def test_fair(self, tracker: OnChainTracker):
        # NVT between 45-95: tx_vol between 10.26B-21.67B
        stats = {
            "market_price_usd": 50000,
            "estimated_transaction_volume_usd": 15e9,
        }
        with patch.object(tracker, "_get_json", return_value=stats):
            nvt = tracker.get_nvt_ratio()
        assert 45 <= nvt.nvt <= 95
        assert nvt.signal == "fair"

    def test_zero_volume(self, tracker: OnChainTracker):
        stats = {
            "market_price_usd": 50000,
            "estimated_transaction_volume_usd": 0,
        }
        with patch.object(tracker, "_get_json", return_value=stats):
            nvt = tracker.get_nvt_ratio()
        assert nvt.signal == "unknown"


class TestGetWhaleActivity:
    @pytest.fixture(autouse=True)
    def _setup(self, _no_throttle):
        pass

    def test_accumulating(self, tracker: OnChainTracker):
        # avg_tx > 50000: total_btc * price / n_tx > 50000
        # 10000 BTC * 50000 / 5000 = 100000
        stats = {
            "n_tx": 5000,
            "total_btc_sent": int(10000 * 1e8),
            "market_price_usd": 50000,
        }
        with patch.object(tracker, "_get_json", return_value=stats):
            wa = tracker.get_whale_activity()
        assert wa is not None
        assert wa.whale_sentiment == "accumulating"

    def test_distributing(self, tracker: OnChainTracker):
        # avg_tx < 10000: total_btc * price / n_tx < 10000
        # 100 BTC * 50000 / 500000 = 10
        stats = {
            "n_tx": 500000,
            "total_btc_sent": int(100 * 1e8),
            "market_price_usd": 50000,
        }
        with patch.object(tracker, "_get_json", return_value=stats):
            wa = tracker.get_whale_activity()
        assert wa.whale_sentiment == "distributing"

    def test_neutral(self, tracker: OnChainTracker):
        # avg_tx between 10000-50000
        # 5000 BTC * 50000 / 10000 = 25000
        stats = {
            "n_tx": 10000,
            "total_btc_sent": int(5000 * 1e8),
            "market_price_usd": 50000,
        }
        with patch.object(tracker, "_get_json", return_value=stats):
            wa = tracker.get_whale_activity()
        assert wa.whale_sentiment == "neutral"


class TestGetDefiMetrics:
    @pytest.fixture(autouse=True)
    def _setup(self, _no_throttle):
        pass

    def test_bullish_rising_tvl(self, tracker: OnChainTracker):
        tvl_data = [{"tvl": 90e9 + i * 1e9} for i in range(10)]
        protocols = [
            {"name": "Lido", "tvl": 30e9, "change_1d": 2.0},
            {"name": "Aave", "tvl": 20e9, "change_1d": 1.5},
        ]
        stablecoins = {"peggedAssets": []}

        with patch.object(tracker, "_get_json", side_effect=[tvl_data, protocols, stablecoins]):
            dm = tracker.get_defi_metrics()
        assert dm is not None
        assert dm.total_tvl_usd > 0
        assert dm.tvl_change_7d_pct > 5
        assert dm.signal == "bullish"
        assert len(dm.top_protocols) == 2

    def test_bearish_falling_tvl(self, tracker: OnChainTracker):
        tvl_data = [{"tvl": 100e9 - i * 2e9} for i in range(10)]
        with patch.object(tracker, "_get_json", side_effect=[tvl_data, [], {"peggedAssets": []}]):
            dm = tracker.get_defi_metrics()
        assert dm.tvl_change_7d_pct < -5
        assert dm.signal == "bearish"

    def test_handles_none_responses(self, tracker: OnChainTracker):
        with patch.object(tracker, "_get_json", return_value=None):
            dm = tracker.get_defi_metrics()
        assert dm is not None
        assert dm.total_tvl_usd == 0.0
        assert dm.signal == "neutral"

    def test_stablecoin_mcap(self, tracker: OnChainTracker):
        tvl_data = [{"tvl": 100e9}]
        stablecoins = {
            "peggedAssets": [{
                "chainCirculating": {
                    "Ethereum": {"current": {"peggedUSD": 50e9}},
                    "Tron": {"current": {"peggedUSD": 30e9}},
                }
            }]
        }
        with patch.object(tracker, "_get_json", side_effect=[tvl_data, [], stablecoins]):
            dm = tracker.get_defi_metrics()
        assert dm.stablecoin_mcap == pytest.approx(80e9, rel=0.01)


# ══════════════════════════════════════════════════════════════
#  get_report & aggregation
# ══════════════════════════════════════════════════════════════


class TestGetReport:
    @pytest.fixture(autouse=True)
    def _setup(self, _no_throttle):
        pass

    def test_report_structure(self, tracker: OnChainTracker):
        """Rapport med alt mocked til None."""
        with patch.object(tracker, "_get_json", return_value=None):
            report = tracker.get_report("BTC-USD")
        assert isinstance(report, OnChainReport)
        assert report.symbol == "BTC-USD"
        assert report.timestamp
        assert report.overall_signal == OnChainSignal.NEUTRAL
        # DeFi metrics returnerer tom objekt selv ved None → 1 metric tæller med
        assert report.confidence <= 35

    def test_btc_gets_all_metrics(self, tracker: OnChainTracker):
        """BTC henter alle on-chain metrics."""
        call_args = []
        original = tracker._get_json

        def capture_url(url, **kw):
            call_args.append(url)
            return None

        with patch.object(tracker, "_get_json", side_effect=capture_url):
            tracker.get_report("BTC-USD")

        # Skal kalde Fear&Greed, CoinGecko global, Blockchain stats (flere gange), DeFi Llama
        assert len(call_args) >= 4

    def test_non_btc_skips_onchain(self, tracker: OnChainTracker):
        """Ikke-BTC symbols springer BTC-specifik on-chain over."""
        call_args = []

        def capture_url(url, **kw):
            call_args.append(url)
            return None

        with patch.object(tracker, "_get_json", side_effect=capture_url):
            tracker.get_report("ETH-USD")

        # Skal IKKE kalde blockchain.com stats
        blockchain_calls = [u for u in call_args if "blockchain" in u]
        assert len(blockchain_calls) == 0


class TestAggregateSignals:
    def _make_tracker(self, tmp_path: Path) -> OnChainTracker:
        return OnChainTracker(cache_dir=str(tmp_path / "cache"))

    def test_strong_bullish(self, tmp_path: Path):
        t = self._make_tracker(tmp_path)
        fg = FearGreedIndex(value=10, classification="Extreme Fear",
                           level=FearGreedLevel.EXTREME_FEAR,
                           timestamp="0", contrarian_signal="buy")
        ef = ExchangeFlowData(inflow_btc=100, outflow_btc=200,
                              net_flow=-100, signal="bullish", description="OK")
        aa = ActiveAddresses(count=900000, change_pct_7d=10.0, signal="bullish")
        hr = HashRateData(hash_rate=500, change_pct_30d=10.0, signal="bullish")
        nvt = NVTRatio(nvt=30, signal="undervalued", description="Low")
        whale = WhaleActivity(large_txs_24h=100, whale_sentiment="accumulating",
                             largest_tx_usd=5e6, description="Acc")
        defi = DeFiMetrics(total_tvl_usd=100e9, tvl_change_24h_pct=2,
                          tvl_change_7d_pct=10, top_protocols=[], stablecoin_mcap=80e9,
                          signal="bullish")

        signal, conf, summary = t._aggregate_signals(
            "BTC-USD", fg, ef, aa, hr, nvt, whale, defi, None,
        )
        assert signal == OnChainSignal.STRONG_BULLISH
        assert conf > 50

    def test_strong_bearish(self, tmp_path: Path):
        t = self._make_tracker(tmp_path)
        fg = FearGreedIndex(value=90, classification="Extreme Greed",
                           level=FearGreedLevel.EXTREME_GREED,
                           timestamp="0", contrarian_signal="sell")
        ef = ExchangeFlowData(inflow_btc=200, outflow_btc=100,
                              net_flow=100, signal="bearish", description="Bad")
        nvt = NVTRatio(nvt=120, signal="overvalued", description="High")
        whale = WhaleActivity(large_txs_24h=50, whale_sentiment="distributing",
                             largest_tx_usd=1e6, description="Dist")
        defi = DeFiMetrics(total_tvl_usd=50e9, tvl_change_24h_pct=-3,
                          tvl_change_7d_pct=-10, top_protocols=[], stablecoin_mcap=50e9,
                          signal="bearish")

        signal, conf, summary = t._aggregate_signals(
            "BTC-USD", fg, ef, None, None, nvt, whale, defi, None,
        )
        assert signal in (OnChainSignal.STRONG_BEARISH, OnChainSignal.BEARISH)

    def test_neutral_when_no_data(self, tmp_path: Path):
        t = self._make_tracker(tmp_path)
        signal, conf, summary = t._aggregate_signals(
            "BTC-USD", None, None, None, None, None, None, None, None,
        )
        assert signal == OnChainSignal.NEUTRAL
        assert conf == 30.0

    def test_alt_season_boost_for_alts(self, tmp_path: Path):
        t = self._make_tracker(tmp_path)
        dom = BitcoinDominance(dominance_pct=35, change_7d=-2, alt_season=True,
                               signal="alt_season", description="Alt season")
        signal, conf, summary = t._aggregate_signals(
            "ETH-USD", None, None, None, None, None, None, None, dom,
        )
        # Alt season giver bullish boost for ETH
        assert signal == OnChainSignal.NEUTRAL or signal == OnChainSignal.BULLISH


# ══════════════════════════════════════════════════════════════
#  Strategy integration
# ══════════════════════════════════════════════════════════════


class TestConfidenceAdjustment:
    def test_bullish_positive(self, tracker: OnChainTracker):
        report = OnChainReport(
            symbol="BTC-USD", timestamp="now",
            fear_greed=None, exchange_flow=None, active_addresses=None,
            hash_rate=None, nvt_ratio=None, whale_activity=None,
            defi_metrics=None, btc_dominance=None,
            overall_signal=OnChainSignal.STRONG_BULLISH,
            confidence=80.0, summary="test",
        )
        adj = tracker.get_confidence_adjustment(report)
        assert 0 < adj <= 15

    def test_bearish_negative(self, tracker: OnChainTracker):
        report = OnChainReport(
            symbol="BTC-USD", timestamp="now",
            fear_greed=None, exchange_flow=None, active_addresses=None,
            hash_rate=None, nvt_ratio=None, whale_activity=None,
            defi_metrics=None, btc_dominance=None,
            overall_signal=OnChainSignal.STRONG_BEARISH,
            confidence=80.0, summary="test",
        )
        adj = tracker.get_confidence_adjustment(report)
        assert -15 <= adj < 0

    def test_neutral_zero(self, tracker: OnChainTracker):
        report = OnChainReport(
            symbol="BTC-USD", timestamp="now",
            fear_greed=None, exchange_flow=None, active_addresses=None,
            hash_rate=None, nvt_ratio=None, whale_activity=None,
            defi_metrics=None, btc_dominance=None,
            overall_signal=OnChainSignal.NEUTRAL,
            confidence=50.0, summary="test",
        )
        assert tracker.get_confidence_adjustment(report) == 0

    def test_max_adjustment_cap(self, tracker: OnChainTracker):
        report = OnChainReport(
            symbol="BTC-USD", timestamp="now",
            fear_greed=None, exchange_flow=None, active_addresses=None,
            hash_rate=None, nvt_ratio=None, whale_activity=None,
            defi_metrics=None, btc_dominance=None,
            overall_signal=OnChainSignal.BULLISH,
            confidence=100.0, summary="test",
        )
        adj = tracker.get_confidence_adjustment(report)
        assert adj <= 15


# ══════════════════════════════════════════════════════════════
#  is_crypto & explain
# ══════════════════════════════════════════════════════════════


class TestIsCrypto:
    def test_known_crypto(self, tracker: OnChainTracker):
        assert tracker.is_crypto("BTC-USD") is True
        assert tracker.is_crypto("ETH-USD") is True
        assert tracker.is_crypto("SOL-USD") is True

    def test_unknown_usd_suffix(self, tracker: OnChainTracker):
        assert tracker.is_crypto("FAKE-USD") is True  # ends with -USD

    def test_stocks_not_crypto(self, tracker: OnChainTracker):
        assert tracker.is_crypto("AAPL") is False
        assert tracker.is_crypto("MSFT") is False
        assert tracker.is_crypto("TSLA") is False


class TestExplain:
    def test_explain_full_report(self, tracker: OnChainTracker):
        fg = FearGreedIndex(value=25, classification="Fear",
                           level=FearGreedLevel.FEAR,
                           timestamp="12345", contrarian_signal="buy")
        report = OnChainReport(
            symbol="BTC-USD", timestamp="2026-03-17T12:00:00",
            fear_greed=fg, exchange_flow=None, active_addresses=None,
            hash_rate=None, nvt_ratio=None, whale_activity=None,
            defi_metrics=None, btc_dominance=None,
            overall_signal=OnChainSignal.BULLISH,
            confidence=65.0, summary="BTC: bullish",
        )
        text = tracker.explain(report)
        assert "BTC-USD" in text
        assert "Fear & Greed" in text
        assert "25" in text
        assert "buy" in text

    def test_explain_empty_report(self, tracker: OnChainTracker):
        report = OnChainReport(
            symbol="ETH-USD", timestamp="now",
            fear_greed=None, exchange_flow=None, active_addresses=None,
            hash_rate=None, nvt_ratio=None, whale_activity=None,
            defi_metrics=None, btc_dominance=None,
            overall_signal=OnChainSignal.NEUTRAL,
            confidence=30.0, summary="Ingen data",
        )
        text = tracker.explain(report)
        assert "ETH-USD" in text
        assert "neutral" in text.lower()


class TestCryptoIDs:
    def test_all_ids_present(self):
        assert len(CRYPTO_IDS) >= 18
        assert "BTC-USD" in CRYPTO_IDS
        assert "ETH-USD" in CRYPTO_IDS

    def test_values_are_strings(self):
        for k, v in CRYPTO_IDS.items():
            assert isinstance(v, str)
            assert len(v) > 0


class TestThrottle:
    def test_throttle_delays(self, tracker: OnChainTracker):
        """Throttle sætter _last_request."""
        tracker._last_request = 0.0
        tracker._throttle()
        assert tracker._last_request > 0

    def test_get_json_returns_none_without_requests(self, tracker: OnChainTracker, monkeypatch):
        """Uden requests library returnerer None."""
        import src.data.onchain as onchain_mod
        monkeypatch.setattr(onchain_mod, "_HAS_REQUESTS", False)
        assert tracker._get_json("http://example.com") is None
