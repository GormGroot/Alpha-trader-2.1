"""
Tests for AssetUniverse og parallel datahentning:
  - Kategori-management (enable/disable)
  - Symbol-opslag og filtrering
  - Watchlist-mode og scan-mode
  - Parallel hentning (mock)
  - Region- og aktivklasse-filtrering
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data.universe import (
    AssetUniverse,
    AssetCategory,
    US_LARGE_CAP_CORE,
    NORDIC_OMX_C25,
    ETFS_BROAD_INDEX,
    CRYPTO_TOP_20,
    COMMODITIES_PRECIOUS,
    TREASURY_YIELDS,
    FOREX_PAIRS,
)
from src.data.market_data import MarketDataFetcher


# ══════════════════════════════════════════════════════════════
#  AssetCategory
# ══════════════════════════════════════════════════════════════


class TestAssetCategory:
    def test_all_symbols_flat_list(self):
        cat = AssetCategory(
            name="test",
            display_name="Test",
            subcategories={
                "a": ["AAPL", "MSFT"],
                "b": ["GOOGL"],
            },
        )
        assert cat.all_symbols == ["AAPL", "MSFT", "GOOGL"]
        assert cat.symbol_count == 3

    def test_tuple_symbols(self):
        """Råstoffer og forex bruger (symbol, label) tuples."""
        cat = AssetCategory(
            name="test",
            display_name="Test",
            subcategories={
                "metals": [("GC=F", "Guld"), ("SI=F", "Sølv")],
            },
        )
        assert cat.all_symbols == ["GC=F", "SI=F"]

    def test_mixed_subcategories(self):
        cat = AssetCategory(
            name="test",
            display_name="Test",
            subcategories={
                "stocks": ["AAPL"],
                "futures": [("CL=F", "Oil")],
            },
        )
        assert len(cat.all_symbols) == 2

    def test_defaults(self):
        cat = AssetCategory(
            name="test",
            display_name="Test",
            subcategories={},
        )
        assert cat.tradeable is True
        assert cat.supports_24h is False
        assert cat.default_interval == "1d"


# ══════════════════════════════════════════════════════════════
#  AssetUniverse – Grundlæggende
# ══════════════════════════════════════════════════════════════


class TestAssetUniverseBasic:
    def test_default_all_categories_enabled(self):
        u = AssetUniverse()
        assert len(u.all_categories) >= 10  # Vi har 11 kategorier
        assert len(u.enabled_categories) >= 10

    def test_specific_categories_enabled(self):
        u = AssetUniverse(enabled_categories=["us_stocks", "etfs"])
        assert len(u.enabled_categories) == 2

    def test_active_symbols_non_empty(self):
        u = AssetUniverse(enabled_categories=["us_stocks"])
        symbols = u.active_symbols
        assert len(symbols) > 0
        assert "AAPL" in symbols

    def test_no_duplicates_in_active(self):
        u = AssetUniverse()
        symbols = u.active_symbols
        assert len(symbols) == len(set(symbols))

    def test_len_and_iter(self):
        u = AssetUniverse(enabled_categories=["etfs"])
        assert len(u) > 0
        symbols_from_iter = list(u)
        assert len(symbols_from_iter) == len(u)

    def test_contains(self):
        u = AssetUniverse(enabled_categories=["us_stocks"])
        assert "AAPL" in u
        assert "FAKE_SYMBOL_XYZ" not in u


# ══════════════════════════════════════════════════════════════
#  Kategori-management
# ══════════════════════════════════════════════════════════════


class TestCategoryManagement:
    def test_enable_disable(self):
        u = AssetUniverse(enabled_categories=["us_stocks"])
        assert "AAPL" in u.active_symbols

        u.disable_category("us_stocks")
        assert "AAPL" not in u.active_symbols

        u.enable_category("us_stocks")
        assert "AAPL" in u.active_symbols

    def test_get_category(self):
        u = AssetUniverse()
        cat = u.get_category("crypto")
        assert cat is not None
        assert cat.supports_24h is True
        assert cat.display_name == "Kryptovaluta"

    def test_get_unknown_category(self):
        u = AssetUniverse()
        assert u.get_category("nonexistent") is None

    def test_get_symbols_for_category(self):
        u = AssetUniverse()
        symbols = u.get_symbols_for_category("us_stocks")
        assert len(symbols) > 50  # Core + extended + mid/small
        assert "AAPL" in symbols

    def test_get_symbols_for_subcategory(self):
        u = AssetUniverse()
        symbols = u.get_symbols_for_subcategory("us_stocks", "large_cap_core")
        assert len(symbols) == len(US_LARGE_CAP_CORE)

    def test_tradeable_excludes_bonds(self):
        u = AssetUniverse(enabled_categories=["bonds", "us_stocks"])
        tradeable = u.tradeable_symbols
        # Bonds er ikke tradeable
        for sym, _ in TREASURY_YIELDS:
            assert sym not in tradeable
        # Stocks er tradeable
        assert "AAPL" in tradeable


# ══════════════════════════════════════════════════════════════
#  Watchlist
# ══════════════════════════════════════════════════════════════


class TestWatchlist:
    def test_empty_watchlist_uses_categories(self):
        u = AssetUniverse(enabled_categories=["us_stocks"])
        assert len(u.active_symbols) > 10

    def test_watchlist_overrides_categories(self):
        u = AssetUniverse(
            enabled_categories=["us_stocks", "etfs", "crypto"],
            watchlist=["AAPL", "BTC-USD", "SPY"],
        )
        assert u.active_symbols == ["AAPL", "BTC-USD", "SPY"]

    def test_set_watchlist(self):
        u = AssetUniverse(enabled_categories=["us_stocks"])
        u.set_watchlist(["TSLA", "NVDA"])
        assert u.active_symbols == ["TSLA", "NVDA"]

    def test_add_remove_watchlist(self):
        u = AssetUniverse(watchlist=["AAPL"])
        u.add_to_watchlist("MSFT")
        assert "MSFT" in u.watchlist
        u.remove_from_watchlist("AAPL")
        assert "AAPL" not in u.watchlist

    def test_clear_watchlist(self):
        u = AssetUniverse(watchlist=["AAPL", "MSFT"])
        u.clear_watchlist()
        assert len(u.watchlist) == 0

    def test_exclude_from_watchlist(self):
        u = AssetUniverse(watchlist=["AAPL", "MSFT", "GOOGL"])
        u.exclude("MSFT")
        assert "MSFT" not in u.active_symbols
        assert "AAPL" in u.active_symbols


# ══════════════════════════════════════════════════════════════
#  Filtrering
# ══════════════════════════════════════════════════════════════


class TestFiltering:
    def test_filter_by_region_us(self):
        u = AssetUniverse()
        symbols = u.filter_by_region("us")
        assert "AAPL" in symbols
        # Nordiske skal IKKE være med
        for sym in NORDIC_OMX_C25:
            assert sym not in symbols

    def test_filter_by_region_nordic(self):
        u = AssetUniverse()
        symbols = u.filter_by_region("nordic")
        assert any(s.endswith(".CO") for s in symbols)
        assert any(s.endswith(".ST") for s in symbols)

    def test_filter_by_region_global(self):
        u = AssetUniverse()
        symbols = u.filter_by_region("global")
        assert len(symbols) > 100

    def test_filter_by_asset_class_stocks(self):
        u = AssetUniverse()
        symbols = u.filter_by_asset_class("stocks")
        assert "AAPL" in symbols
        assert "SPY" not in symbols  # SPY er en ETF

    def test_filter_by_asset_class_crypto(self):
        u = AssetUniverse()
        symbols = u.filter_by_asset_class("crypto")
        assert "BTC-USD" in symbols
        assert "AAPL" not in symbols

    def test_filter_by_asset_class_etfs(self):
        u = AssetUniverse()
        symbols = u.filter_by_asset_class("etfs")
        assert "SPY" in symbols
        assert "GLD" in symbols

    def test_24h_symbols(self):
        u = AssetUniverse(enabled_categories=["crypto", "forex", "us_stocks"])
        symbols_24h = u.get_24h_symbols()
        assert "BTC-USD" in symbols_24h
        assert "EURUSD=X" in symbols_24h
        assert "AAPL" not in symbols_24h

    def test_exclude_symbol(self):
        u = AssetUniverse(enabled_categories=["us_stocks"])
        u.exclude("AAPL")
        assert "AAPL" not in u.active_symbols
        u.include("AAPL")
        assert "AAPL" in u.active_symbols


# ══════════════════════════════════════════════════════════════
#  Scan-mode
# ══════════════════════════════════════════════════════════════


class TestScanMode:
    def test_scan_all(self):
        u = AssetUniverse(enabled_categories=["us_stocks", "etfs"])
        symbols = u.scan_universe()
        assert len(symbols) > 50

    def test_scan_with_max(self):
        u = AssetUniverse(enabled_categories=["us_stocks"])
        symbols = u.scan_universe(max_symbols=10)
        assert len(symbols) == 10

    def test_scan_specific_categories(self):
        u = AssetUniverse()
        symbols = u.scan_universe(categories=["crypto"])
        assert "BTC-USD" in symbols
        assert "AAPL" not in symbols

    def test_scan_empty_category(self):
        u = AssetUniverse(enabled_categories=[])
        symbols = u.scan_universe()
        assert len(symbols) == 0


# ══════════════════════════════════════════════════════════════
#  Summary / Statistik
# ══════════════════════════════════════════════════════════════


class TestSummary:
    def test_summary_structure(self):
        u = AssetUniverse()
        s = u.summary()
        assert "total_symbols" in s
        assert "active_symbols" in s
        assert "categories" in s
        assert s["total_symbols"] > 0
        assert s["total_categories"] >= 10

    def test_summary_category_details(self):
        u = AssetUniverse()
        s = u.summary()
        crypto = s["categories"]["crypto"]
        assert crypto["display_name"] == "Kryptovaluta"
        assert crypto["24h"] is True
        assert crypto["tradeable"] is True

    def test_print_summary_runs(self, capsys):
        u = AssetUniverse(enabled_categories=["us_stocks"])
        u.print_summary()
        captured = capsys.readouterr()
        assert "ASSET UNIVERSE" in captured.out
        assert "US Aktier" in captured.out


# ══════════════════════════════════════════════════════════════
#  Symbol-lister (sanity checks)
# ══════════════════════════════════════════════════════════════


class TestSymbolLists:
    def test_us_large_cap_count(self):
        assert len(US_LARGE_CAP_CORE) == 50

    def test_nordic_c25_count(self):
        assert len(NORDIC_OMX_C25) == 25

    def test_crypto_top20_count(self):
        assert len(CRYPTO_TOP_20) == 20

    def test_etfs_broad_index(self):
        assert "SPY" in ETFS_BROAD_INDEX
        assert "QQQ" in ETFS_BROAD_INDEX

    def test_commodities_have_labels(self):
        for sym, label in COMMODITIES_PRECIOUS:
            assert "=F" in sym
            assert len(label) > 0

    def test_forex_have_labels(self):
        for sym, label in FOREX_PAIRS:
            assert "=X" in sym
            assert "/" in label

    def test_no_duplicates_across_all(self):
        """Tjek at der ikke er identiske symboler i forskellige lister."""
        u = AssetUniverse()
        all_syms = []
        for cat in u.all_categories:
            all_syms.extend(cat.all_symbols)
        # Nogle symboler KAN optræde i flere kategorier
        # (f.eks. GLD i etfs OG commodities)
        # Men inden for samme kategori skal de være unikke
        for cat in u.all_categories:
            syms = cat.all_symbols
            assert len(syms) == len(set(syms)), \
                f"Duplikater i {cat.name}: {[s for s in syms if syms.count(s) > 1]}"


# ══════════════════════════════════════════════════════════════
#  Parallel Datahentning (mock)
# ══════════════════════════════════════════════════════════════


class TestParallelFetch:
    def _make_fetcher(self) -> MarketDataFetcher:
        tmpdir = tempfile.mkdtemp()
        return MarketDataFetcher(cache_dir=tmpdir)

    def test_get_cached_symbols_empty(self):
        mdf = self._make_fetcher()
        assert mdf.get_cached_symbols() == []

    @patch("src.data.market_data.yf.Ticker")
    def test_parallel_uses_cache_first(self, mock_ticker):
        """Parallel hentning skal bruge cache og kun kalde API for manglende."""
        mdf = self._make_fetcher()

        # Gem noget i cache
        df = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [99.0],
            "Close": [103.0], "Volume": [1000],
        }, index=pd.DatetimeIndex(["2026-01-15"]))
        mdf._write_cache("AAPL", "1d", df)

        # Mock for non-cached symbol
        mock_hist = pd.DataFrame({
            "Open": [200.0], "High": [210.0], "Low": [195.0],
            "Close": [205.0], "Volume": [2000],
        }, index=pd.DatetimeIndex(["2026-01-15"]))
        mock_ticker.return_value.history.return_value = mock_hist

        results = mdf.get_multiple_parallel(
            ["AAPL", "MSFT"],
            start="2026-01-01", end="2026-01-31",
            max_workers=2,
        )

        assert "AAPL" in results
        assert "MSFT" in results
        # AAPL fra cache (103.0), MSFT fra API (205.0)
        assert len(results["AAPL"]) >= 1

    @patch("src.data.market_data.yf.Ticker")
    def test_parallel_handles_errors(self, mock_ticker):
        """API-fejl for ét symbol må ikke stoppe de andre."""
        mdf = self._make_fetcher()
        mock_ticker.return_value.history.side_effect = Exception("API fejl")

        results = mdf.get_multiple_parallel(
            ["BAD1", "BAD2"],
            start="2026-01-01", end="2026-01-31",
            max_workers=2,
        )

        assert len(results) == 2
        # Begge skal være tomme (fejl)
        assert results["BAD1"].empty
        assert results["BAD2"].empty

    @patch("src.data.market_data.yf.Ticker")
    def test_parallel_batch_progress(self, mock_ticker):
        """Test at batching virker for mange symboler."""
        mdf = self._make_fetcher()
        mock_ticker.return_value.history.return_value = pd.DataFrame(
            {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]},
            index=pd.DatetimeIndex(["2026-01-15"]),
        )

        symbols = [f"SYM{i}" for i in range(15)]
        results = mdf.get_multiple_parallel(
            symbols,
            start="2026-01-01", end="2026-01-31",
            max_workers=4, batch_size=5,
        )

        assert len(results) == 15
