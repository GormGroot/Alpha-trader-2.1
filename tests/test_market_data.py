"""
Tests for market data fetcher and technical indicators.

Bruger syntetisk data så tests kører hurtigt og uden netværk.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import MarketDataFetcher, MarketDataError
from src.data.indicators import (
    add_sma,
    add_ema,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_volume_analysis,
    add_all_indicators,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generer 100 dage syntetisk OHLCV-data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(100) * 0.5,
            "High": close + abs(np.random.randn(100)) * 1.5,
            "Low": close - abs(np.random.randn(100)) * 1.5,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, size=100),
        },
        index=dates,
    )


@pytest.fixture
def fetcher(tmp_path):
    """MarketDataFetcher med midlertidig cache-mappe."""
    return MarketDataFetcher(cache_dir=str(tmp_path / "cache"))


# ── MarketDataFetcher tests ──────────────────────────────────

class TestMarketDataFetcher:

    def test_init_creates_db(self, fetcher, tmp_path):
        db_path = tmp_path / "cache" / "market_data.db"
        assert db_path.exists()

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_invalid_interval_raises(self, fetcher):
        with pytest.raises(MarketDataError, match="Ugyldigt interval"):
            fetcher.get_historical("AAPL", interval="3d")

    @patch("src.data.market_data.yf.Ticker")
    def test_get_historical_returns_dataframe(self, mock_ticker_cls, fetcher, sample_df):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_df
        mock_ticker_cls.return_value = mock_ticker

        result = fetcher.get_historical("AAPL", interval="1d", lookback_days=100)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        mock_ticker.history.assert_called_once()

    @patch("src.data.market_data.yf.Ticker")
    def test_cache_stores_and_retrieves(self, mock_ticker_cls, fetcher, sample_df):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_df
        mock_ticker_cls.return_value = mock_ticker

        # Første kald – henter fra API
        fetcher.get_historical("AAPL", interval="1d", start="2024-01-01", end="2024-06-01")
        assert mock_ticker.history.call_count == 1

        # Andet kald – bør bruge cache
        fetcher.get_historical("AAPL", interval="1d", start="2024-01-01", end="2024-06-01")
        assert mock_ticker.history.call_count == 1  # stadig kun 1 API-kald

    @patch("src.data.market_data.yf.Ticker")
    def test_get_historical_empty_response(self, mock_ticker_cls, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = fetcher.get_historical("FAKE", interval="1d", lookback_days=30)
        assert result.empty

    @patch("src.data.market_data.yf.Ticker")
    def test_get_historical_api_error(self, mock_ticker_cls, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_ticker_cls.return_value = mock_ticker

        with pytest.raises(MarketDataError, match="Kunne ikke hente data"):
            fetcher.get_historical("AAPL", interval="1d", lookback_days=30)

    @patch("src.data.market_data.yf.Ticker")
    def test_get_multiple_returns_dict(self, mock_ticker_cls, fetcher, sample_df):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_df
        mock_ticker_cls.return_value = mock_ticker

        results = fetcher.get_multiple(
            symbols=["AAPL", "MSFT"], interval="1d", lookback_days=100,
        )

        assert isinstance(results, dict)
        assert "AAPL" in results
        assert "MSFT" in results
        assert len(results["AAPL"]) == 100

    @patch("src.data.market_data.yf.Ticker")
    def test_get_multiple_handles_partial_failure(self, mock_ticker_cls, fetcher, sample_df):
        mock_ticker = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return sample_df
            raise Exception("API down")

        mock_ticker.history.side_effect = side_effect
        mock_ticker_cls.return_value = mock_ticker

        results = fetcher.get_multiple(
            symbols=["AAPL", "BAD"], interval="1d", lookback_days=100,
        )
        assert len(results["AAPL"]) == 100
        assert results["BAD"].empty

    @patch("src.data.market_data.yf.Ticker")
    def test_get_latest_price(self, mock_ticker_cls, fetcher):
        mock_ticker = MagicMock()
        mock_ticker.fast_info = {"lastPrice": 175.50}
        mock_ticker_cls.return_value = mock_ticker

        price = fetcher.get_latest_price("AAPL")
        assert price == 175.50

    def test_clear_cache(self, fetcher):
        # Indsæt dummy data i cache
        with fetcher._get_conn() as conn:
            conn.execute(
                "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("AAPL", "1d", "2024-01-01", 100, 105, 99, 103, 5000000, "now"),
            )

        fetcher.clear_cache("AAPL")

        with fetcher._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        assert count == 0


# ── Indicator tests ──────────────────────────────────────────

class TestSMA:

    def test_sma_column_added(self, sample_df):
        result = add_sma(sample_df, window=20)
        assert "SMA_20" in result.columns

    def test_sma_first_values_are_nan(self, sample_df):
        result = add_sma(sample_df, window=20)
        assert result["SMA_20"].isna().sum() == 19  # window-1 NaN

    def test_sma_value_is_correct(self, sample_df):
        result = add_sma(sample_df, window=5)
        expected = sample_df["Close"].iloc[:5].mean()
        assert abs(result["SMA_5"].iloc[4] - expected) < 1e-10


class TestEMA:

    def test_ema_column_added(self, sample_df):
        result = add_ema(sample_df, window=12)
        assert "EMA_12" in result.columns

    def test_ema_no_nans_after_first(self, sample_df):
        result = add_ema(sample_df, window=12)
        # EMA har ingen NaN (ewm beregner fra start)
        assert result["EMA_12"].isna().sum() == 0


class TestRSI:

    def test_rsi_column_added(self, sample_df):
        result = add_rsi(sample_df)
        assert "RSI" in result.columns

    def test_rsi_range(self, sample_df):
        result = add_rsi(sample_df)
        valid = result["RSI"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_all_gains(self):
        """RSI bør nærme sig 100 hvis prisen kun stiger."""
        df = pd.DataFrame({"Close": np.arange(1, 51, dtype=float)})
        result = add_rsi(df, period=14)
        assert result["RSI"].iloc[-1] > 95


class TestMACD:

    def test_macd_columns_added(self, sample_df):
        result = add_macd(sample_df)
        assert "MACD" in result.columns
        assert "MACD_Signal" in result.columns
        assert "MACD_Hist" in result.columns

    def test_macd_hist_equals_diff(self, sample_df):
        result = add_macd(sample_df)
        diff = result["MACD"] - result["MACD_Signal"]
        pd.testing.assert_series_equal(result["MACD_Hist"], diff, check_names=False)


class TestBollingerBands:

    def test_bb_columns_added(self, sample_df):
        result = add_bollinger_bands(sample_df)
        for col in ["BB_Upper", "BB_Middle", "BB_Lower", "BB_Width"]:
            assert col in result.columns

    def test_upper_above_lower(self, sample_df):
        result = add_bollinger_bands(sample_df)
        valid = result.dropna()
        assert (valid["BB_Upper"] >= valid["BB_Lower"]).all()

    def test_middle_equals_sma(self, sample_df):
        result = add_bollinger_bands(sample_df, window=20)
        sma = sample_df["Close"].rolling(20).mean()
        pd.testing.assert_series_equal(result["BB_Middle"], sma, check_names=False)


class TestVolumeAnalysis:

    def test_volume_columns_added(self, sample_df):
        result = add_volume_analysis(sample_df)
        assert "Volume_SMA" in result.columns
        assert "Volume_Ratio" in result.columns
        assert "OBV" in result.columns

    def test_volume_ratio_around_one(self, sample_df):
        result = add_volume_analysis(sample_df, window=20)
        valid_ratio = result["Volume_Ratio"].dropna()
        mean_ratio = valid_ratio.mean()
        assert 0.5 < mean_ratio < 2.0  # bør svinge omkring 1


class TestAddAllIndicators:

    def test_all_indicators_added(self, sample_df):
        result = add_all_indicators(sample_df)
        expected = [
            "SMA_20", "SMA_50", "SMA_200",
            "EMA_12", "EMA_26",
            "RSI",
            "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width",
            "Volume_SMA", "Volume_Ratio", "OBV",
        ]
        for col in expected:
            assert col in result.columns, f"Mangler kolonne: {col}"

    def test_original_columns_preserved(self, sample_df):
        result = add_all_indicators(sample_df)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns
