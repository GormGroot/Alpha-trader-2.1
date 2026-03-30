"""
Data Processor — NPU + GPU accelerated batch processing of historical data.

Reads raw OHLCV from historical_master.db, produces a "processed data block":
  1. ML features (16 base + 6 ensemble = 22 columns) for every symbol/date
  2. Trained ML models (retrained on latest data)
  3. Pre-computed predictions cached in SQLite for instant lookup

Hardware acceleration:
  - RK3588 NPU (6 TOPS): Model inference via RKNN
  - RTX GPU (CUDA):       Bulk feature computation, model training via PyTorch/cuML
  - CPU fallback:          numpy/sklearn when no accelerator is available

The processor runs after each daily download so the trader always has
a fresh, ready-to-use data block with zero startup latency.

Usage:
  from src.ops.data_processor import DataProcessor
  dp = DataProcessor()
  dp.run()                    # Full rebuild
  dp.run_incremental()        # Only process new/updated symbols

  # Read processed data (fast — no computation needed)
  features = dp.get_features("AAPL", days=365)
  prediction = dp.get_prediction("AAPL")
"""

from __future__ import annotations

import gc
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── Hardware detection ────────────────────────────────────

@dataclass
class HardwareProfile:
    """Detected compute hardware."""
    has_cuda: bool = False
    cuda_device: str = ""
    cuda_memory_mb: int = 0
    has_npu: bool = False
    npu_version: str = ""
    cpu_cores: int = 1
    device_name: str = "cpu"

    def __repr__(self) -> str:
        parts = [f"CPU ({self.cpu_cores} cores)"]
        if self.has_cuda:
            parts.append(f"CUDA ({self.cuda_device}, {self.cuda_memory_mb}MB)")
        if self.has_npu:
            parts.append(f"NPU ({self.npu_version})")
        return f"HardwareProfile[{', '.join(parts)}]"


def detect_hardware() -> HardwareProfile:
    """Detect available compute accelerators."""
    profile = HardwareProfile()

    # CPU cores
    try:
        profile.cpu_cores = os.cpu_count() or 1
    except Exception:
        pass

    # CUDA / RTX GPU
    try:
        import torch
        if torch.cuda.is_available():
            profile.has_cuda = True
            profile.cuda_device = torch.cuda.get_device_name(0)
            profile.cuda_memory_mb = int(
                torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
            )
            profile.device_name = "cuda"
            logger.info(
                f"[data_processor] CUDA GPU detected: {profile.cuda_device} "
                f"({profile.cuda_memory_mb}MB)"
            )
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"[data_processor] CUDA detection error: {e}")

    # RK3588 NPU
    try:
        npu_version_path = "/sys/kernel/debug/rknpu/version"
        if os.path.exists(npu_version_path):
            with open(npu_version_path) as f:
                profile.npu_version = f.read().strip()
            profile.has_npu = True
            if not profile.has_cuda:
                profile.device_name = "npu"
            logger.info(f"[data_processor] RK3588 NPU detected: {profile.npu_version}")
    except Exception:
        pass

    if not profile.has_cuda and not profile.has_npu:
        logger.info(
            f"[data_processor] No accelerator found — using CPU "
            f"({profile.cpu_cores} cores)"
        )

    return profile


# ── Config ────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _ROOT / "data_cache"
_HIST_DB = _DATA_DIR / "historical_master.db"
_PROC_DB = _DATA_DIR / "processed_data.db"

# Batch sizes tuned for Rock 5B (8GB RAM) — GPU path uses larger batches
_CPU_BATCH = 50
_GPU_BATCH = 200
_NPU_INFERENCE_BATCH = 100


# ── Processed Data Schema ─────────────────────────────────

_FEATURE_COLS = [
    # 16 base features (ml_strategy.py)
    "rsi", "macd", "macd_signal", "macd_hist",
    "sma_20_pct", "sma_50_pct", "sma_200_pct", "sma_cross",
    "bb_position", "bb_width",
    "volume_ratio", "obv_slope",
    "return_1d", "return_5d", "return_20d", "volatility_20d",
    # 6 ensemble features (ensemble_ml_strategy.py)
    "regime_score", "roc_10", "stoch_k", "stoch_d",
    "atr_pct", "volatility_ratio",
    # 2 news sentiment features
    "news_sentiment", "news_volume",
]

_INIT_PROCESSED_SQL = """
    CREATE TABLE IF NOT EXISTS processed_features (
        symbol          TEXT NOT NULL,
        date            TEXT NOT NULL,
        rsi             REAL,
        macd            REAL,
        macd_signal     REAL,
        macd_hist       REAL,
        sma_20_pct      REAL,
        sma_50_pct      REAL,
        sma_200_pct     REAL,
        sma_cross       REAL,
        bb_position     REAL,
        bb_width        REAL,
        volume_ratio    REAL,
        obv_slope       REAL,
        return_1d       REAL,
        return_5d       REAL,
        return_20d      REAL,
        volatility_20d  REAL,
        regime_score    REAL,
        roc_10          REAL,
        stoch_k         REAL,
        stoch_d         REAL,
        atr_pct         REAL,
        volatility_ratio REAL,
        news_sentiment  REAL,
        news_volume     REAL,
        PRIMARY KEY (symbol, date)
    );

    CREATE TABLE IF NOT EXISTS predictions (
        symbol          TEXT NOT NULL,
        date            TEXT NOT NULL,
        ml_prob_up      REAL,
        ml_signal       TEXT,
        ml_confidence   REAL,
        ensemble_prob_up REAL,
        ensemble_signal  TEXT,
        ensemble_confidence REAL,
        ensemble_agree   INTEGER,
        computed_at     TEXT DEFAULT (datetime('now')),
        device          TEXT DEFAULT 'cpu',
        PRIMARY KEY (symbol, date)
    );

    CREATE TABLE IF NOT EXISTS model_state (
        model_name      TEXT PRIMARY KEY,
        trained_at      TEXT,
        n_train_samples INTEGER,
        accuracy        REAL,
        auc_roc         REAL,
        device          TEXT,
        duration_seconds REAL,
        feature_importance TEXT
    );

    CREATE TABLE IF NOT EXISTS process_log (
        run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at      TEXT,
        finished_at     TEXT,
        mode            TEXT,
        symbols_processed INTEGER,
        features_written INTEGER,
        predictions_written INTEGER,
        models_trained  INTEGER,
        device          TEXT,
        duration_seconds REAL,
        errors          TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_pf_symbol ON processed_features(symbol);
    CREATE INDEX IF NOT EXISTS idx_pf_date ON processed_features(date);
    CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol);
    CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(date);

    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA cache_size = -64000;
"""


# ── Feature Computation Engine ────────────────────────────

class FeatureEngine:
    """
    Computes ML features from OHLCV data.

    Uses GPU (PyTorch) when available for vectorized computation across
    all symbols simultaneously. Falls back to numpy on CPU.
    """

    def __init__(self, hw: HardwareProfile):
        self._hw = hw
        self._torch = None
        if hw.has_cuda:
            try:
                import torch
                self._torch = torch
            except ImportError:
                self._hw.has_cuda = False

    def compute_features_batch(
        self,
        symbols_data: dict[str, pd.DataFrame],
        sentiment_data: dict | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Compute 24 ML features for a batch of symbols.

        Args:
            symbols_data: {symbol: DataFrame with open/high/low/close/volume columns}
            sentiment_data: {symbol: {date: (sentiment_avg, news_count)}} from news_sentiment.db

        Returns:
            {symbol: DataFrame with 24 feature columns + date index}
        """
        if self._hw.has_cuda and self._torch is not None:
            results = self._compute_gpu(symbols_data)
        else:
            results = self._compute_cpu(symbols_data, sentiment_data)
            return results

        # Enrich GPU results with sentiment (computed on CPU since it's a simple lookup)
        if sentiment_data:
            for sym, feat_df in results.items():
                sym_sent = sentiment_data.get(sym, {})
                if sym_sent:
                    for idx in feat_df.index:
                        date_str = str(idx)[:10]
                        if date_str in sym_sent:
                            sent_avg, news_count = sym_sent[date_str]
                            feat_df.at[idx, "news_sentiment"] = sent_avg or 0.0
                            feat_df.at[idx, "news_volume"] = np.log1p(news_count or 0) / 5.0

        return results

    def _compute_cpu(
        self,
        symbols_data: dict[str, pd.DataFrame],
        sentiment_data: dict | None = None,
    ) -> dict[str, pd.DataFrame]:
        """CPU path — numpy vectorized, processes one symbol at a time."""
        results = {}
        for symbol, df in symbols_data.items():
            try:
                sym_sent = sentiment_data.get(symbol, {}) if sentiment_data else None
                feat = self._compute_single(df, sentiment_lookup=sym_sent or None)
                if feat is not None and len(feat) > 0:
                    results[symbol] = feat
            except Exception as e:
                logger.debug(f"[data_processor] Feature error {symbol}: {e}")
        return results

    def _compute_gpu(
        self, symbols_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        GPU path — batch multiple symbols into tensors for parallel computation.

        Strategy: stack all symbols' close/high/low/volume into 2D tensors,
        compute rolling stats with torch, then unstack. Much faster for
        hundreds of symbols simultaneously.
        """
        torch = self._torch
        results = {}

        # For symbols with enough data, batch on GPU
        # For short series, fall back to CPU
        gpu_eligible = {}
        for sym, df in symbols_data.items():
            if len(df) >= 200:
                gpu_eligible[sym] = df
            else:
                try:
                    feat = self._compute_single(df)
                    if feat is not None:
                        results[sym] = feat
                except Exception:
                    pass

        if not gpu_eligible:
            return results

        try:
            # Process in GPU batches
            sym_list = list(gpu_eligible.keys())
            batch_size = _GPU_BATCH

            for batch_start in range(0, len(sym_list), batch_size):
                batch_syms = sym_list[batch_start:batch_start + batch_size]
                batch_dfs = [gpu_eligible[s] for s in batch_syms]

                # Find common length (trim to shortest in batch)
                min_len = min(len(d) for d in batch_dfs)
                min_len = max(min_len, 200)  # Need at least 200 for SMA_200

                # Stack into tensors [batch, time]
                closes = torch.zeros(len(batch_syms), min_len, device="cuda")
                highs = torch.zeros_like(closes)
                lows = torch.zeros_like(closes)
                volumes = torch.zeros_like(closes)

                for i, df in enumerate(batch_dfs):
                    tail = df.tail(min_len)
                    closes[i] = torch.tensor(
                        tail["close"].values.astype(np.float64),
                        dtype=torch.float64, device="cuda",
                    )
                    highs[i] = torch.tensor(
                        tail["high"].values.astype(np.float64),
                        dtype=torch.float64, device="cuda",
                    )
                    lows[i] = torch.tensor(
                        tail["low"].values.astype(np.float64),
                        dtype=torch.float64, device="cuda",
                    )
                    volumes[i] = torch.tensor(
                        tail["volume"].values.astype(np.float64),
                        dtype=torch.float64, device="cuda",
                    )

                # ── Compute features on GPU (vectorized across batch) ──

                # Returns
                ret_1d = (closes[:, 1:] - closes[:, :-1]) / closes[:, :-1]
                # Pad front with 0
                ret_1d = torch.cat([torch.zeros(len(batch_syms), 1, device="cuda"), ret_1d], dim=1)

                # Rolling means via cumsum trick
                def rolling_mean(t: torch.Tensor, w: int) -> torch.Tensor:
                    cs = t.cumsum(dim=1)
                    cs[:, w:] = cs[:, w:] - cs[:, :-w]
                    result = torch.full_like(t, float("nan"))
                    result[:, w - 1:] = cs[:, w - 1:] / w
                    return result

                def rolling_std(t: torch.Tensor, w: int) -> torch.Tensor:
                    mean = rolling_mean(t, w)
                    sq_mean = rolling_mean(t ** 2, w)
                    var = sq_mean - mean ** 2
                    var = torch.clamp(var, min=0)
                    return torch.sqrt(var)

                sma_20 = rolling_mean(closes, 20)
                sma_50 = rolling_mean(closes, 50)
                sma_200 = rolling_mean(closes, 200)

                # EMA (use CPU for now — rolling EMA hard to batch on GPU)
                ema_12 = rolling_mean(closes, 12)  # Approximate
                ema_26 = rolling_mean(closes, 26)

                # MACD approximation
                macd_line = ema_12 - ema_26
                macd_sig = rolling_mean(macd_line, 9)
                macd_hist = macd_line - macd_sig

                # RSI
                delta = closes[:, 1:] - closes[:, :-1]
                delta = torch.cat([torch.zeros(len(batch_syms), 1, device="cuda"), delta], dim=1)
                gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
                loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
                avg_gain = rolling_mean(gain, 14)
                avg_loss = rolling_mean(loss, 14)
                rs = avg_gain / torch.clamp(avg_loss, min=1e-10)
                rsi = 100.0 - 100.0 / (1.0 + rs)

                # Bollinger
                std_20 = rolling_std(closes, 20)
                bb_upper = sma_20 + 2 * std_20
                bb_lower = sma_20 - 2 * std_20
                bb_range = bb_upper - bb_lower
                bb_pos = torch.where(
                    bb_range > 0,
                    (closes - bb_lower) / bb_range,
                    torch.full_like(closes, 0.5),
                )
                bb_width = bb_range / torch.clamp(closes, min=1e-10)

                # Volume ratio
                vol_mean_20 = rolling_mean(volumes, 20)
                vol_ratio = volumes / torch.clamp(vol_mean_20, min=1.0)

                # SMA pct
                sma_20_pct = (closes - sma_20) / torch.clamp(sma_20.abs(), min=1e-10)
                sma_50_pct = (closes - sma_50) / torch.clamp(sma_50.abs(), min=1e-10)
                sma_200_pct = (closes - sma_200) / torch.clamp(sma_200.abs(), min=1e-10)
                sma_cross = (sma_20 - sma_50) / torch.clamp(closes, min=1e-10)

                # OBV slope
                sign_delta = torch.sign(delta)
                obv = (volumes * sign_delta).cumsum(dim=1)
                obv_5_ago = torch.cat([
                    torch.zeros(len(batch_syms), 5, device="cuda"),
                    obv[:, :-5],
                ], dim=1)
                obv_slope = (obv - obv_5_ago) / torch.clamp(obv_5_ago.abs(), min=1.0)
                obv_slope = obv_slope.clamp(-1, 1)

                # Returns 5d, 20d
                def pct_change_n(t: torch.Tensor, n: int) -> torch.Tensor:
                    r = torch.full_like(t, float("nan"))
                    r[:, n:] = (t[:, n:] - t[:, :-n]) / torch.clamp(t[:, :-n].abs(), min=1e-10)
                    return r

                ret_5d = pct_change_n(closes, 5)
                ret_20d = pct_change_n(closes, 20)

                # Volatility 20d
                vol_20d = rolling_std(ret_1d, 20) * np.sqrt(252)

                # Ensemble extras
                roc_10 = pct_change_n(closes, 10)

                # Stochastic
                def rolling_min(t, w):
                    result = torch.full_like(t, float("nan"))
                    for i in range(w - 1, t.shape[1]):
                        result[:, i] = t[:, i - w + 1:i + 1].min(dim=1).values
                    return result

                def rolling_max(t, w):
                    result = torch.full_like(t, float("nan"))
                    for i in range(w - 1, t.shape[1]):
                        result[:, i] = t[:, i - w + 1:i + 1].max(dim=1).values
                    return result

                low_14 = rolling_min(lows, 14)
                high_14 = rolling_max(highs, 14)
                stoch_denom = high_14 - low_14
                stoch_k = torch.where(
                    stoch_denom > 0,
                    (closes - low_14) / stoch_denom * 100,
                    torch.full_like(closes, 50.0),
                )
                stoch_d = rolling_mean(stoch_k, 3)

                # ATR pct
                tr1 = highs - lows
                tr2 = (highs - torch.cat([closes[:, :1], closes[:, :-1]], dim=1)).abs()
                tr3 = (lows - torch.cat([closes[:, :1], closes[:, :-1]], dim=1)).abs()
                true_range = torch.max(torch.max(tr1, tr2), tr3)
                atr_14 = rolling_mean(true_range, 14)
                atr_pct = atr_14 / torch.clamp(closes, min=1e-10)

                # Volatility ratio
                vol_5_std = rolling_std(ret_1d, 5)
                vol_20_std = rolling_std(ret_1d, 20)
                vol_ratio_feat = torch.where(
                    vol_20_std > 0,
                    vol_5_std / vol_20_std,
                    torch.ones_like(vol_5_std),
                )

                # ── Transfer back to CPU and build DataFrames ──
                for i, sym in enumerate(batch_syms):
                    try:
                        tail = batch_dfs[i].tail(min_len)
                        dates = tail.index if hasattr(tail.index, 'strftime') else range(min_len)

                        feat_dict = {
                            "rsi": rsi[i].cpu().numpy(),
                            "macd": macd_line[i].cpu().numpy(),
                            "macd_signal": macd_sig[i].cpu().numpy(),
                            "macd_hist": macd_hist[i].cpu().numpy(),
                            "sma_20_pct": sma_20_pct[i].cpu().numpy(),
                            "sma_50_pct": sma_50_pct[i].cpu().numpy(),
                            "sma_200_pct": sma_200_pct[i].cpu().numpy(),
                            "sma_cross": sma_cross[i].cpu().numpy(),
                            "bb_position": bb_pos[i].cpu().numpy(),
                            "bb_width": bb_width[i].cpu().numpy(),
                            "volume_ratio": vol_ratio[i].cpu().numpy(),
                            "obv_slope": obv_slope[i].cpu().numpy(),
                            "return_1d": ret_1d[i].cpu().numpy(),
                            "return_5d": ret_5d[i].cpu().numpy(),
                            "return_20d": ret_20d[i].cpu().numpy(),
                            "volatility_20d": vol_20d[i].cpu().numpy(),
                            "regime_score": np.zeros(min_len),
                            "roc_10": roc_10[i].cpu().numpy(),
                            "stoch_k": stoch_k[i].cpu().numpy(),
                            "stoch_d": stoch_d[i].cpu().numpy(),
                            "atr_pct": atr_pct[i].cpu().numpy(),
                            "volatility_ratio": vol_ratio_feat[i].cpu().numpy(),
                            "news_sentiment": np.zeros(min_len),
                            "news_volume": np.zeros(min_len),
                        }
                        feat_df = pd.DataFrame(feat_dict, index=dates)
                        # Drop rows where SMA_200 wasn't available yet
                        feat_df = feat_df.iloc[199:]
                        results[sym] = feat_df
                    except Exception as e:
                        logger.debug(f"[data_processor] GPU→CPU {sym}: {e}")

                # Free GPU memory between batches
                del closes, highs, lows, volumes
                if self._torch is not None:
                    self._torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"[data_processor] GPU batch failed, falling back to CPU: {e}")
            # Fall back to CPU for anything not yet processed
            for sym in gpu_eligible:
                if sym not in results:
                    try:
                        feat = self._compute_single(gpu_eligible[sym])
                        if feat is not None:
                            results[sym] = feat
                    except Exception:
                        pass

        return results

    def _compute_single(self, df: pd.DataFrame, sentiment_lookup: dict | None = None) -> Optional[pd.DataFrame]:
        """Compute all 24 features for a single symbol using numpy."""
        if df is None or len(df) < 50:
            return None

        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)
        n = len(close)

        def rolling_mean(arr, w):
            result = np.full(n, np.nan)
            cs = np.cumsum(arr)
            cs = np.insert(cs, 0, 0)
            result[w - 1:] = (cs[w:] - cs[:-w]) / w
            return result

        def rolling_std(arr, w):
            result = np.full(n, np.nan)
            for i in range(w - 1, n):
                result[i] = np.std(arr[i - w + 1:i + 1], ddof=1) if w > 1 else 0
            return result

        # SMAs
        sma_20 = rolling_mean(close, 20)
        sma_50 = rolling_mean(close, 50)
        sma_200 = rolling_mean(close, 200) if n >= 200 else np.full(n, np.nan)

        # Returns
        ret_1d = np.full(n, np.nan)
        ret_1d[1:] = (close[1:] - close[:-1]) / np.maximum(np.abs(close[:-1]), 1e-10)

        # RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss_arr = np.where(delta < 0, -delta, 0.0)
        avg_gain = rolling_mean(gain, 14)
        avg_loss = rolling_mean(loss_arr, 14)
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # MACD
        ema_12 = pd.Series(close).ewm(span=12).mean().values
        ema_26 = pd.Series(close).ewm(span=26).mean().values
        macd_line = ema_12 - ema_26
        macd_sig = pd.Series(macd_line).ewm(span=9).mean().values
        macd_hist = macd_line - macd_sig

        # Bollinger
        std_20 = rolling_std(close, 20)
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_range = bb_upper - bb_lower
        bb_pos = np.where(bb_range > 0, (close - bb_lower) / bb_range, 0.5)
        bb_width = bb_range / np.maximum(close, 1e-10)

        # Volume ratio
        vol_mean_20 = rolling_mean(volume, 20)
        vol_ratio = volume / np.maximum(vol_mean_20, 1.0)

        # SMA percentages
        sma_20_pct = (close - sma_20) / np.maximum(np.abs(sma_20), 1e-10)
        sma_50_pct = (close - sma_50) / np.maximum(np.abs(sma_50), 1e-10)
        sma_200_pct = (close - sma_200) / np.maximum(np.abs(sma_200), 1e-10)
        sma_cross = (sma_20 - sma_50) / np.maximum(close, 1e-10)

        # OBV slope
        sign_d = np.sign(delta)
        obv = np.cumsum(volume * sign_d)
        obv_5_ago = np.roll(obv, 5)
        obv_5_ago[:5] = obv[:5]
        obv_slope = np.clip(
            (obv - obv_5_ago) / np.maximum(np.abs(obv_5_ago), 1.0), -1, 1,
        )

        # Returns 5d, 20d
        ret_5d = np.full(n, np.nan)
        ret_5d[5:] = (close[5:] - close[:-5]) / np.maximum(np.abs(close[:-5]), 1e-10)
        ret_20d = np.full(n, np.nan)
        ret_20d[20:] = (close[20:] - close[:-20]) / np.maximum(np.abs(close[:-20]), 1e-10)

        # Volatility 20d
        vol_20d_arr = rolling_std(ret_1d, 20) * np.sqrt(252)

        # ROC 10
        roc_10 = np.full(n, np.nan)
        roc_10[10:] = (close[10:] - close[:-10]) / np.maximum(np.abs(close[:-10]), 1e-10)

        # Stochastic
        stoch_k_arr = np.full(n, np.nan)
        for i in range(13, n):
            lo = np.min(low[i - 13:i + 1])
            hi = np.max(high[i - 13:i + 1])
            denom = hi - lo
            stoch_k_arr[i] = ((close[i] - lo) / denom * 100) if denom > 0 else 50.0
        stoch_d_arr = rolling_mean(np.nan_to_num(stoch_k_arr, nan=50.0), 3)

        # ATR pct
        tr1 = high - low
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        atr_14 = rolling_mean(true_range, 14)
        atr_pct = atr_14 / np.maximum(close, 1e-10)

        # Volatility ratio
        vol_5_std = rolling_std(ret_1d, 5)
        vol_20_std = rolling_std(ret_1d, 20)
        volatility_ratio = np.where(vol_20_std > 0, vol_5_std / vol_20_std, 1.0)

        # Build DataFrame — preserve date index from input
        if hasattr(df.index, 'strftime'):
            dates = df.index
        elif hasattr(df.index, 'dtype') and df.index.dtype == object:
            # String date index from historical_master.db
            dates = df.index
        elif "date" in df.columns:
            dates = df["date"].values
        else:
            dates = df.index
        feat_df = pd.DataFrame({
            "rsi": rsi,
            "macd": macd_line,
            "macd_signal": macd_sig,
            "macd_hist": macd_hist,
            "sma_20_pct": sma_20_pct,
            "sma_50_pct": sma_50_pct,
            "sma_200_pct": sma_200_pct,
            "sma_cross": sma_cross,
            "bb_position": bb_pos,
            "bb_width": bb_width,
            "volume_ratio": vol_ratio,
            "obv_slope": obv_slope,
            "return_1d": ret_1d,
            "return_5d": ret_5d,
            "return_20d": ret_20d,
            "volatility_20d": vol_20d_arr,
            "regime_score": np.zeros(n),
            "roc_10": roc_10,
            "stoch_k": stoch_k_arr,
            "stoch_d": stoch_d_arr,
            "atr_pct": atr_pct,
            "volatility_ratio": volatility_ratio,
            "news_sentiment": np.zeros(n),
            "news_volume": np.zeros(n),
        }, index=dates)

        # Merge sentiment data if available
        if sentiment_lookup is not None:
            for idx in feat_df.index:
                date_str = str(idx)[:10]
                if date_str in sentiment_lookup:
                    sent_avg, news_count = sentiment_lookup[date_str]
                    feat_df.at[idx, "news_sentiment"] = sent_avg or 0.0
                    # Normalize news volume: log(1 + count) / 5 to keep in ~0-1 range
                    feat_df.at[idx, "news_volume"] = np.log1p(news_count or 0) / 5.0

        # Drop warmup rows
        min_warmup = 200 if n >= 200 else 50
        feat_df = feat_df.iloc[min_warmup:]

        return feat_df if len(feat_df) > 0 else None


# ── Model Trainer ─────────────────────────────────────────

class ModelTrainer:
    """
    Trains ML models on processed features using GPU or CPU.

    Models trained:
      - ml_strategy: HistGradientBoostingClassifier (16 features)
      - ensemble_rf: RandomForestClassifier (22 features)
      - ensemble_xgb: Gradient boosted trees (22 features)
      - ensemble_lr: LogisticRegression (22 features)

    After training, exports models to ONNX for NPU inference.
    """

    def __init__(self, hw: HardwareProfile, proc_db: Path = _PROC_DB):
        self._hw = hw
        self._proc_db = proc_db
        self._models: dict = {}

    def train_all(
        self,
        features_db: Path = _PROC_DB,
        hist_db: Path = _HIST_DB,
    ) -> dict[str, dict]:
        """
        Train all models on the full processed feature set.

        Returns dict of {model_name: {accuracy, auc, duration_s, device}}.
        """
        results = {}

        # Load training data from processed features + raw prices for targets
        logger.info("[data_processor] Loading training data...")
        X, y, dates = self._load_training_data(features_db, hist_db)

        if X is None or len(X) < 1000:
            logger.warning(
                f"[data_processor] Not enough training data "
                f"({0 if X is None else len(X)} rows, need 1000+)"
            )
            return results

        logger.info(
            f"[data_processor] Training data: {len(X)} samples, "
            f"{X.shape[1]} features"
        )

        # Time-based train/test split (last 6 months = test)
        split_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        train_mask = dates < split_date
        test_mask = dates >= split_date

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_train) < 500 or len(X_test) < 100:
            logger.warning("[data_processor] Train/test split too small")
            return results

        logger.info(
            f"[data_processor] Split: {len(X_train)} train, {len(X_test)} test"
        )

        # Train each model (ml_strategy uses first 16 base + 2 sentiment = 18 features)
        results["ml_strategy"] = self._train_hgbt(
            X_train[:, :18], y_train, X_test[:, :18], y_test, "ml_strategy",
        )
        results["ensemble_rf"] = self._train_rf(
            X_train, y_train, X_test, y_test, "ensemble_rf",
        )
        results["ensemble_xgb"] = self._train_xgb(
            X_train, y_train, X_test, y_test, "ensemble_xgb",
        )
        results["ensemble_lr"] = self._train_lr(
            X_train, y_train, X_test, y_test, "ensemble_lr",
        )

        # Export to ONNX for NPU
        self._export_onnx()

        # Save model state to DB
        self._save_model_state(results)

        return results

    def _load_training_data(
        self, features_db: Path, hist_db: Path,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load features from processed DB, targets from historical DB."""
        try:
            with sqlite3.connect(str(features_db)) as conn:
                feat_df = pd.read_sql_query(
                    "SELECT symbol, date, " + ", ".join(_FEATURE_COLS) +
                    " FROM processed_features ORDER BY symbol, date",
                    conn,
                )

            if feat_df.empty:
                return None, None, None

            # Load close prices for target computation
            with sqlite3.connect(str(hist_db)) as conn:
                price_df = pd.read_sql_query(
                    "SELECT symbol, date, close FROM daily_bars ORDER BY symbol, date",
                    conn,
                )

            if price_df.empty:
                return None, None, None

            # Compute target: does price rise next day?
            price_df = price_df.sort_values(["symbol", "date"])
            price_df["target"] = (
                price_df.groupby("symbol")["close"]
                .pct_change()
                .shift(-1)
                .gt(0)
                .astype(float)
            )

            # Merge features with targets
            merged = feat_df.merge(
                price_df[["symbol", "date", "target"]],
                on=["symbol", "date"],
                how="inner",
            )
            merged = merged.dropna(subset=["target"])

            # Drop rows with too many NaN features
            feat_cols = _FEATURE_COLS
            nan_count = merged[feat_cols].isna().sum(axis=1)
            merged = merged[nan_count <= 5]  # Allow up to 5 NaN features

            if merged.empty:
                return None, None, None

            X = merged[feat_cols].values.astype(np.float32)
            y = merged["target"].values.astype(np.float32)
            dates = merged["date"].values

            # Replace remaining NaN with 0
            X = np.nan_to_num(X, nan=0.0)

            return X, y, dates

        except Exception as e:
            logger.error(f"[data_processor] Training data load failed: {e}")
            return None, None, None

    def _train_hgbt(self, X_train, y_train, X_test, y_test, name) -> dict:
        """Train HistGradientBoostingClassifier."""
        t0 = time.time()
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score

            model = HistGradientBoostingClassifier(
                max_iter=200, max_depth=5, learning_rate=0.05,
                early_stopping=True, n_iter_no_change=20,
                validation_fraction=0.15, random_state=42,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            self._models[name] = model
            elapsed = time.time() - t0
            logger.info(
                f"[data_processor] {name}: acc={acc:.3f} auc={auc:.3f} "
                f"({elapsed:.1f}s, CPU)"
            )
            return {"accuracy": acc, "auc": auc, "duration_s": elapsed, "device": "cpu"}

        except Exception as e:
            logger.error(f"[data_processor] {name} training failed: {e}")
            return {"accuracy": 0, "auc": 0, "duration_s": time.time() - t0, "device": "error"}

    def _train_rf(self, X_train, y_train, X_test, y_test, name) -> dict:
        """Train RandomForestClassifier."""
        t0 = time.time()
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score

            n_jobs = -1 if self._hw.cpu_cores > 2 else 1
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=20,
                n_jobs=n_jobs, random_state=42,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            self._models[name] = model
            elapsed = time.time() - t0
            logger.info(
                f"[data_processor] {name}: acc={acc:.3f} auc={auc:.3f} "
                f"({elapsed:.1f}s, CPU {n_jobs} jobs)"
            )
            return {"accuracy": acc, "auc": auc, "duration_s": elapsed, "device": "cpu"}

        except Exception as e:
            logger.error(f"[data_processor] {name} training failed: {e}")
            return {"accuracy": 0, "auc": 0, "duration_s": time.time() - t0, "device": "error"}

    def _train_xgb(self, X_train, y_train, X_test, y_test, name) -> dict:
        """Train XGBoost — uses GPU if available."""
        t0 = time.time()
        try:
            # Try GPU-accelerated XGBoost first
            device = "cpu"
            tree_method = "hist"

            if self._hw.has_cuda:
                try:
                    import xgboost as xgb
                    # XGBoost >= 2.0 uses device parameter
                    device = "cuda"
                    tree_method = "hist"
                    logger.info("[data_processor] XGBoost using CUDA GPU")
                except Exception:
                    device = "cpu"

            from xgboost import XGBClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score

            model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                tree_method=tree_method, device=device,
                eval_metric="logloss", early_stopping_rounds=20,
                random_state=42, verbosity=0,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            self._models[name] = model
            elapsed = time.time() - t0
            logger.info(
                f"[data_processor] {name}: acc={acc:.3f} auc={auc:.3f} "
                f"({elapsed:.1f}s, {device})"
            )
            return {"accuracy": acc, "auc": auc, "duration_s": elapsed, "device": device}

        except ImportError:
            logger.info("[data_processor] XGBoost not installed, using HGBT as fallback")
            return self._train_hgbt(X_train, y_train, X_test, y_test, name)
        except Exception as e:
            logger.error(f"[data_processor] {name} training failed: {e}")
            return {"accuracy": 0, "auc": 0, "duration_s": time.time() - t0, "device": "error"}

    def _train_lr(self, X_train, y_train, X_test, y_test, name) -> dict:
        """Train LogisticRegression."""
        t0 = time.time()
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, roc_auc_score

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=42,
            )
            model.fit(X_train_s, y_train)

            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            self._models[name] = {"model": model, "scaler": scaler}
            elapsed = time.time() - t0
            logger.info(
                f"[data_processor] {name}: acc={acc:.3f} auc={auc:.3f} "
                f"({elapsed:.1f}s, CPU)"
            )
            return {"accuracy": acc, "auc": auc, "duration_s": elapsed, "device": "cpu"}

        except Exception as e:
            logger.error(f"[data_processor] {name} training failed: {e}")
            return {"accuracy": 0, "auc": 0, "duration_s": time.time() - t0, "device": "error"}

    def _export_onnx(self) -> None:
        """Export trained models to ONNX for NPU conversion."""
        try:
            from src.ops.npu_accelerator import ModelExporter
            exporter = ModelExporter()

            if "ml_strategy" in self._models:
                exporter.export_ml_strategy(
                    self._models["ml_strategy"], n_features=16, name="ml_strategy",
                )

            rf = self._models.get("ensemble_rf")
            xgb = self._models.get("ensemble_xgb")
            lr_data = self._models.get("ensemble_lr")
            lr = lr_data["model"] if isinstance(lr_data, dict) else lr_data

            if rf and xgb and lr:
                exporter.export_ensemble(rf, xgb, lr, n_features=22)

            exporter.generate_conversion_script()
            logger.info("[data_processor] Models exported to ONNX for NPU conversion")

        except Exception as e:
            logger.debug(f"[data_processor] ONNX export skipped: {e}")

    def _save_model_state(self, results: dict) -> None:
        """Persist model training results to DB."""
        try:
            with sqlite3.connect(str(self._proc_db)) as conn:
                for name, res in results.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO model_state
                        (model_name, trained_at, accuracy, auc_roc, device, duration_seconds)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        name,
                        datetime.now().isoformat(),
                        res.get("accuracy", 0),
                        res.get("auc", 0),
                        res.get("device", "cpu"),
                        res.get("duration_s", 0),
                    ))
        except Exception as e:
            logger.warning(f"[data_processor] Model state save failed: {e}")

    def predict_all(
        self,
        features_db: Path = _PROC_DB,
    ) -> int:
        """
        Run predictions for the latest date of every symbol using trained models.
        Stores results in the predictions table. Returns number of predictions written.

        Uses NPU for inference when RKNN models are available,
        otherwise uses the in-memory sklearn models.
        """
        if not self._models:
            logger.warning("[data_processor] No trained models — skipping predictions")
            return 0

        # Load latest features per symbol
        try:
            with sqlite3.connect(str(features_db)) as conn:
                latest = pd.read_sql_query("""
                    SELECT pf.* FROM processed_features pf
                    INNER JOIN (
                        SELECT symbol, MAX(date) as max_date
                        FROM processed_features
                        GROUP BY symbol
                    ) latest ON pf.symbol = latest.symbol AND pf.date = latest.max_date
                """, conn)
        except Exception as e:
            logger.error(f"[data_processor] Feature load for prediction failed: {e}")
            return 0

        if latest.empty:
            return 0

        # Try NPU inference first
        npu_device = "cpu"
        npu_manager = None
        try:
            from src.ops.npu_accelerator import get_npu_manager
            npu_manager = get_npu_manager()
            if npu_manager.status().get("npu_available"):
                npu_device = "npu"
        except Exception:
            pass

        records = []
        for _, row in latest.iterrows():
            symbol = row["symbol"]
            date = row["date"]
            features_18 = np.array(
                [row[c] for c in _FEATURE_COLS[:18]], dtype=np.float32,
            )
            features_24 = np.array(
                [row[c] for c in _FEATURE_COLS], dtype=np.float32,
            )
            features_18 = np.nan_to_num(features_18, nan=0.0)
            features_24 = np.nan_to_num(features_24, nan=0.0)

            # ML Strategy prediction (now uses 18 features: 16 base + 2 sentiment)
            ml_prob = 0.5
            ml_signal = "HOLD"
            ml_conf = 0.0
            try:
                if npu_device == "npu" and npu_manager:
                    proba = npu_manager.predict_ml("ml_strategy", features_18)
                    if proba is not None:
                        ml_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                elif "ml_strategy" in self._models:
                    X = features_18.reshape(1, -1)
                    ml_prob = float(
                        self._models["ml_strategy"].predict_proba(X)[0][1]
                    )
                ml_conf = abs(ml_prob - 0.5) * 200
                ml_signal = "BUY" if ml_prob > 0.55 else ("SELL" if ml_prob < 0.45 else "HOLD")
            except Exception:
                pass

            # Ensemble prediction (majority vote)
            ens_votes = []
            ens_probs = []
            for mname, n_feat, feats in [
                ("ensemble_rf", 24, features_24),
                ("ensemble_xgb", 24, features_24),
                ("ensemble_lr", 24, features_24),
            ]:
                try:
                    X = feats.reshape(1, -1)
                    if npu_device == "npu" and npu_manager:
                        proba = npu_manager.predict_ml(mname, feats)
                        if proba is not None:
                            p = float(proba[1]) if len(proba) > 1 else float(proba[0])
                            ens_probs.append(p)
                            ens_votes.append(1 if p > 0.5 else 0)
                            continue

                    if mname in self._models:
                        m = self._models[mname]
                        if isinstance(m, dict):
                            X = m["scaler"].transform(X)
                            p = float(m["model"].predict_proba(X)[0][1])
                        else:
                            p = float(m.predict_proba(X)[0][1])
                        ens_probs.append(p)
                        ens_votes.append(1 if p > 0.5 else 0)
                except Exception:
                    pass

            ens_prob = np.mean(ens_probs) if ens_probs else 0.5
            ens_agree = sum(ens_votes) if ens_votes else 0
            ens_total = len(ens_votes) if ens_votes else 1
            ens_signal = "HOLD"
            ens_conf = 0.0
            if ens_votes:
                ens_conf = abs(ens_prob - 0.5) * 200
                if ens_agree >= 2:
                    ens_signal = "BUY"
                elif ens_agree == 0:
                    ens_signal = "SELL"

            records.append((
                symbol, date,
                ml_prob, ml_signal, ml_conf,
                ens_prob, ens_signal, ens_conf, ens_agree,
                datetime.now().isoformat(),
                npu_device,
            ))

        # Write predictions
        try:
            with sqlite3.connect(str(features_db)) as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO predictions
                    (symbol, date, ml_prob_up, ml_signal, ml_confidence,
                     ensemble_prob_up, ensemble_signal, ensemble_confidence,
                     ensemble_agree, computed_at, device)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records)

            logger.info(
                f"[data_processor] Wrote {len(records)} predictions "
                f"(device={npu_device})"
            )
            return len(records)
        except Exception as e:
            logger.error(f"[data_processor] Prediction write failed: {e}")
            return 0

    @property
    def models(self) -> dict:
        return self._models


# ── Main Processor ────────────────────────────────────────

@dataclass
class ProcessResult:
    """Result of a data processing run."""
    mode: str = "full"
    symbols_processed: int = 0
    features_written: int = 0
    predictions_written: int = 0
    models_trained: int = 0
    device: str = "cpu"
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    model_results: dict = field(default_factory=dict)


class DataProcessor:
    """
    Central data processing engine.

    Reads raw OHLCV from historical_master.db, computes ML features,
    trains models, and caches predictions in processed_data.db.

    The processed data block is ready for instant consumption by
    the trader — no computation needed at trade time.
    """

    def __init__(
        self,
        hist_db: Path | None = None,
        proc_db: Path | None = None,
    ):
        self._hist_db = hist_db or _HIST_DB
        self._proc_db = proc_db or _PROC_DB
        self._hw = detect_hardware()
        self._feature_engine = FeatureEngine(self._hw)
        self._model_trainer = ModelTrainer(self._hw, self._proc_db)
        self._init_db()

    def _init_db(self) -> None:
        """Create processed data tables."""
        self._proc_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._proc_db)) as conn:
            conn.executescript(_INIT_PROCESSED_SQL)
            # Migrate existing DBs: add sentiment columns if missing
            try:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(processed_features)")}
                if "news_sentiment" not in cols:
                    conn.execute("ALTER TABLE processed_features ADD COLUMN news_sentiment REAL")
                    conn.execute("ALTER TABLE processed_features ADD COLUMN news_volume REAL")
                    logger.info("[data_processor] Added news_sentiment/news_volume columns to existing DB")
            except Exception:
                pass
        logger.info(f"[data_processor] Processed DB ready: {self._proc_db}")

    def _load_sentiment_data(self) -> dict[str, dict[str, tuple]]:
        """
        Load daily sentiment from news_sentiment.db.
        Returns {symbol: {date_str: (sentiment_avg, news_count)}}.
        """
        sentiment_db = _DATA_DIR / "news_sentiment.db"
        if not sentiment_db.exists():
            logger.debug("[data_processor] No news_sentiment.db found — sentiment features will be 0")
            return {}

        try:
            with sqlite3.connect(str(sentiment_db)) as conn:
                rows = conn.execute(
                    "SELECT symbol, date, sentiment_avg, news_count FROM daily_sentiment"
                ).fetchall()

            result: dict[str, dict[str, tuple]] = {}
            for symbol, date, avg, count in rows:
                if symbol not in result:
                    result[symbol] = {}
                result[symbol][date] = (avg, count)

            logger.info(
                f"[data_processor] Loaded sentiment for {len(result)} symbols "
                f"({len(rows)} daily rows)"
            )
            return result
        except Exception as e:
            logger.warning(f"[data_processor] Failed to load sentiment: {e}")
            return {}

    # ── Full rebuild ──────────────────────────────────────

    def run(self, retrain: bool = True) -> ProcessResult:
        """
        Full processing run: compute features for ALL symbols, train models,
        generate predictions.
        """
        t0 = time.time()
        result = ProcessResult(mode="full", device=self._hw.device_name)
        logger.info(f"[data_processor] === FULL PROCESSING RUN ({self._hw}) ===")

        # 1. Load all symbols from historical DB
        symbols_data = self._load_all_symbols()
        if not symbols_data:
            result.errors.append("No historical data found")
            logger.warning("[data_processor] No data in historical_master.db")
            return result

        logger.info(f"[data_processor] Loaded {len(symbols_data)} symbols from DB")

        # Load sentiment data for feature enrichment
        sentiment_data = self._load_sentiment_data()

        # 2. Compute features in batches
        n_features = self._process_features(symbols_data, result, sentiment_data)
        result.features_written = n_features

        # Force GC between heavy phases
        del symbols_data
        gc.collect()

        # 3. Train models on computed features
        if retrain and n_features > 0:
            model_results = self._model_trainer.train_all(self._proc_db, self._hist_db)
            result.model_results = model_results
            result.models_trained = sum(
                1 for v in model_results.values() if v.get("accuracy", 0) > 0
            )

        # 4. Generate predictions
        if result.models_trained > 0 or not retrain:
            result.predictions_written = self._model_trainer.predict_all(self._proc_db)

        result.duration_seconds = time.time() - t0
        self._log_run(result)

        logger.info(
            f"[data_processor] === COMPLETE: {result.symbols_processed} symbols, "
            f"{result.features_written} features, {result.predictions_written} predictions, "
            f"{result.models_trained} models in {result.duration_seconds:.1f}s ==="
        )

        return result

    def run_incremental(self) -> ProcessResult:
        """
        Incremental run: only process symbols updated since last run.
        Reuses existing trained models for predictions.
        """
        t0 = time.time()
        result = ProcessResult(mode="incremental", device=self._hw.device_name)
        logger.info(f"[data_processor] === INCREMENTAL RUN ({self._hw}) ===")

        # Find symbols that need updating
        updated_symbols = self._find_updated_symbols()
        if not updated_symbols:
            logger.info("[data_processor] No symbols need updating")
            result.duration_seconds = time.time() - t0
            return result

        logger.info(
            f"[data_processor] {len(updated_symbols)} symbols need feature update"
        )

        # Load only the updated symbols
        symbols_data = self._load_symbols(updated_symbols)
        if not symbols_data:
            result.duration_seconds = time.time() - t0
            return result

        # Compute features (with sentiment enrichment)
        sentiment_data = self._load_sentiment_data()
        n_features = self._process_features(symbols_data, result, sentiment_data)
        result.features_written = n_features

        del symbols_data
        gc.collect()

        # Load existing models and run predictions
        self._load_existing_models()
        if self._model_trainer.models:
            result.predictions_written = self._model_trainer.predict_all(self._proc_db)

        result.duration_seconds = time.time() - t0
        self._log_run(result)

        logger.info(
            f"[data_processor] === INCREMENTAL COMPLETE: "
            f"{result.symbols_processed} symbols, {result.features_written} features, "
            f"{result.predictions_written} predictions in {result.duration_seconds:.1f}s ==="
        )

        return result

    # ── Data loading ──────────────────────────────────────

    def _load_all_symbols(self) -> dict[str, pd.DataFrame]:
        """Load all OHLCV data from historical_master.db."""
        try:
            with sqlite3.connect(str(self._hist_db)) as conn:
                symbols = [
                    r[0] for r in conn.execute(
                        "SELECT DISTINCT symbol FROM daily_bars"
                    ).fetchall()
                ]

            return self._load_symbols(symbols)
        except Exception as e:
            logger.error(f"[data_processor] Failed to load symbols: {e}")
            return {}

    def _load_symbols(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Load OHLCV for specific symbols."""
        results = {}
        batch_size = 100

        try:
            with sqlite3.connect(str(self._hist_db)) as conn:
                for i in range(0, len(symbols), batch_size):
                    batch = symbols[i:i + batch_size]
                    placeholders = ",".join("?" * len(batch))
                    df = pd.read_sql_query(
                        f"SELECT symbol, date, open, high, low, close, volume "
                        f"FROM daily_bars WHERE symbol IN ({placeholders}) "
                        f"ORDER BY symbol, date",
                        conn,
                        params=batch,
                    )

                    for sym, group in df.groupby("symbol"):
                        sdf = group.set_index("date").drop(columns=["symbol"])
                        if len(sdf) >= 50:
                            results[sym] = sdf

                    if (i + batch_size) % 500 == 0:
                        logger.info(
                            f"[data_processor] Loaded {min(i + batch_size, len(symbols))}/"
                            f"{len(symbols)} symbols"
                        )
        except Exception as e:
            logger.error(f"[data_processor] Symbol load failed: {e}")

        return results

    def _find_updated_symbols(self) -> list[str]:
        """Find symbols whose historical data is newer than processed features."""
        try:
            with sqlite3.connect(str(self._hist_db)) as hist_conn:
                hist_latest = pd.read_sql_query(
                    "SELECT symbol, MAX(date) as hist_date FROM daily_bars "
                    "GROUP BY symbol",
                    hist_conn,
                )

            with sqlite3.connect(str(self._proc_db)) as proc_conn:
                proc_latest = pd.read_sql_query(
                    "SELECT symbol, MAX(date) as proc_date FROM processed_features "
                    "GROUP BY symbol",
                    proc_conn,
                )

            if proc_latest.empty:
                return hist_latest["symbol"].tolist()

            merged = hist_latest.merge(proc_latest, on="symbol", how="left")
            # Symbols where hist is newer than processed, or never processed
            needs_update = merged[
                merged["proc_date"].isna() |
                (merged["hist_date"] > merged["proc_date"])
            ]
            return needs_update["symbol"].tolist()

        except Exception as e:
            logger.debug(f"[data_processor] Update check failed: {e}")
            # If we can't check, process everything
            try:
                with sqlite3.connect(str(self._hist_db)) as conn:
                    rows = conn.execute(
                        "SELECT DISTINCT symbol FROM daily_bars"
                    ).fetchall()
                return [r[0] for r in rows]
            except Exception:
                return []

    # ── Feature processing ────────────────────────────────

    def _process_features(
        self,
        symbols_data: dict[str, pd.DataFrame],
        result: ProcessResult,
        sentiment_data: dict | None = None,
    ) -> int:
        """Compute and store features, returns total feature rows written."""
        total_written = 0
        batch_size = _GPU_BATCH if self._hw.has_cuda else _CPU_BATCH
        sym_list = list(symbols_data.keys())

        for batch_start in range(0, len(sym_list), batch_size):
            batch_syms = sym_list[batch_start:batch_start + batch_size]
            batch_data = {s: symbols_data[s] for s in batch_syms}

            try:
                features = self._feature_engine.compute_features_batch(
                    batch_data, sentiment_data=sentiment_data,
                )
                n_written = self._store_features(features)
                total_written += n_written
                result.symbols_processed += len(features)

                if (batch_start + batch_size) % 200 == 0 or batch_start == 0:
                    logger.info(
                        f"[data_processor] Features: "
                        f"{min(batch_start + batch_size, len(sym_list))}/"
                        f"{len(sym_list)} symbols"
                    )

            except Exception as e:
                err = f"Batch {batch_start}: {e}"
                result.errors.append(err)
                logger.warning(f"[data_processor] {err}")

            # GC between batches
            if (batch_start // batch_size) % 5 == 4:
                gc.collect()

        return total_written

    def _store_features(self, features: dict[str, pd.DataFrame]) -> int:
        """Write computed features to processed_data.db."""
        total = 0
        try:
            with sqlite3.connect(str(self._proc_db)) as conn:
                for symbol, feat_df in features.items():
                    records = []
                    for date_val, row in feat_df.iterrows():
                        date_str = (
                            date_val if isinstance(date_val, str)
                            else str(date_val)
                        )
                        record = [symbol, date_str] + [
                            float(row[c]) if pd.notna(row.get(c, np.nan)) else None
                            for c in _FEATURE_COLS
                        ]
                        records.append(tuple(record))

                    if records:
                        placeholders = ",".join(["?"] * (2 + len(_FEATURE_COLS)))
                        conn.executemany(
                            f"INSERT OR REPLACE INTO processed_features "
                            f"(symbol, date, {', '.join(_FEATURE_COLS)}) "
                            f"VALUES ({placeholders})",
                            records,
                        )
                        total += len(records)

                conn.commit()
        except Exception as e:
            logger.error(f"[data_processor] Feature store failed: {e}")

        return total

    # ── Model loading ─────────────────────────────────────

    def _load_existing_models(self) -> None:
        """Load previously trained sklearn models from disk (joblib)."""
        model_dir = _DATA_DIR / "npu_models"
        try:
            import joblib
            for name in ["ml_strategy", "ensemble_rf", "ensemble_xgb", "ensemble_lr"]:
                path = model_dir / f"{name}.joblib"
                if path.exists():
                    self._model_trainer._models[name] = joblib.load(path)
                    logger.info(f"[data_processor] Loaded cached model: {name}")
        except ImportError:
            logger.debug("[data_processor] joblib not available — cannot load cached models")
        except Exception as e:
            logger.debug(f"[data_processor] Model cache load: {e}")

    # ── Logging ───────────────────────────────────────────

    def _log_run(self, result: ProcessResult) -> None:
        """Persist run result to process_log table."""
        try:
            with sqlite3.connect(str(self._proc_db)) as conn:
                conn.execute("""
                    INSERT INTO process_log
                    (started_at, finished_at, mode, symbols_processed,
                     features_written, predictions_written, models_trained,
                     device, duration_seconds, errors)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    (datetime.now() - timedelta(seconds=result.duration_seconds)).isoformat(),
                    datetime.now().isoformat(),
                    result.mode,
                    result.symbols_processed,
                    result.features_written,
                    result.predictions_written,
                    result.models_trained,
                    result.device,
                    result.duration_seconds,
                    "; ".join(result.errors) if result.errors else "",
                ))
        except Exception as e:
            logger.debug(f"[data_processor] Log write failed: {e}")

    # ── Public read API ───────────────────────────────────

    def get_features(
        self, symbol: str, days: int = 365,
    ) -> Optional[pd.DataFrame]:
        """
        Read pre-computed features from the processed data block.
        Zero computation — instant lookup.
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(str(self._proc_db)) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM processed_features "
                    "WHERE symbol = ? AND date >= ? ORDER BY date",
                    conn,
                    params=(symbol, start_date),
                )
            if df.empty:
                return None
            df = df.set_index("date").drop(columns=["symbol"])
            return df
        except Exception:
            return None

    def get_prediction(self, symbol: str) -> Optional[dict]:
        """
        Get the latest pre-computed prediction for a symbol.
        Zero computation — instant lookup.
        """
        try:
            with sqlite3.connect(str(self._proc_db)) as conn:
                row = conn.execute(
                    "SELECT * FROM predictions WHERE symbol = ? "
                    "ORDER BY date DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
            if not row:
                return None
            cols = [
                "symbol", "date", "ml_prob_up", "ml_signal", "ml_confidence",
                "ensemble_prob_up", "ensemble_signal", "ensemble_confidence",
                "ensemble_agree", "computed_at", "device",
            ]
            return dict(zip(cols, row))
        except Exception:
            return None

    def get_all_predictions(self) -> pd.DataFrame:
        """Get latest prediction for every symbol."""
        try:
            with sqlite3.connect(str(self._proc_db)) as conn:
                return pd.read_sql_query("""
                    SELECT p.* FROM predictions p
                    INNER JOIN (
                        SELECT symbol, MAX(date) as max_date
                        FROM predictions GROUP BY symbol
                    ) latest ON p.symbol = latest.symbol AND p.date = latest.max_date
                    ORDER BY p.ensemble_confidence DESC
                """, conn)
        except Exception:
            return pd.DataFrame()

    def get_status(self) -> dict:
        """Get processor status and stats."""
        status = {
            "hardware": str(self._hw),
            "proc_db": str(self._proc_db),
            "hist_db": str(self._hist_db),
        }
        try:
            with sqlite3.connect(str(self._proc_db)) as conn:
                status["feature_symbols"] = conn.execute(
                    "SELECT COUNT(DISTINCT symbol) FROM processed_features"
                ).fetchone()[0]
                status["feature_rows"] = conn.execute(
                    "SELECT COUNT(*) FROM processed_features"
                ).fetchone()[0]
                status["prediction_symbols"] = conn.execute(
                    "SELECT COUNT(DISTINCT symbol) FROM predictions"
                ).fetchone()[0]

                # Latest run
                row = conn.execute(
                    "SELECT * FROM process_log ORDER BY run_id DESC LIMIT 1"
                ).fetchone()
                if row:
                    status["last_run"] = {
                        "finished_at": row[2],
                        "mode": row[3],
                        "symbols": row[4],
                        "features": row[5],
                        "predictions": row[6],
                        "models": row[7],
                        "device": row[8],
                        "duration_s": row[9],
                    }

                # Model state
                models = conn.execute("SELECT * FROM model_state").fetchall()
                status["models"] = {
                    r[0]: {"trained_at": r[1], "accuracy": r[3], "auc": r[4], "device": r[5]}
                    for r in models
                }

                # DB size
                db_size = self._proc_db.stat().st_size if self._proc_db.exists() else 0
                status["db_size_mb"] = round(db_size / (1024 * 1024), 1)
        except Exception as e:
            status["error"] = str(e)

        return status

    def print_status(self) -> None:
        """Print human-readable status."""
        s = self.get_status()
        print(f"\n{'='*55}")
        print(f"  Data Processor Status")
        print(f"{'='*55}")
        print(f"  Hardware:       {s.get('hardware', 'unknown')}")
        print(f"  Feature syms:   {s.get('feature_symbols', 0)}")
        print(f"  Feature rows:   {s.get('feature_rows', 0):,}")
        print(f"  Predictions:    {s.get('prediction_symbols', 0)} symbols")
        print(f"  DB size:        {s.get('db_size_mb', 0)} MB")

        if "last_run" in s:
            lr = s["last_run"]
            print(f"\n  Last run:       {lr.get('finished_at', '?')}")
            print(f"    Mode:         {lr.get('mode')}")
            print(f"    Device:       {lr.get('device')}")
            print(f"    Duration:     {lr.get('duration_s', 0):.1f}s")

        if "models" in s and s["models"]:
            print(f"\n  Models:")
            for name, info in s["models"].items():
                acc = info.get("accuracy", 0) or 0
                auc = info.get("auc", 0) or 0
                print(f"    {name:<20} acc={acc:.3f} auc={auc:.3f} ({info.get('device', '?')})")

        print(f"{'='*55}\n")


# ── Singleton ─────────────────────────────────────────────

_processor: DataProcessor | None = None


def get_data_processor() -> DataProcessor:
    """Get or create the global DataProcessor instance."""
    global _processor
    if _processor is None:
        _processor = DataProcessor()
    return _processor


# ── CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Processor — NPU/GPU accelerated")
    parser.add_argument("--full", action="store_true", help="Full rebuild (features + train + predict)")
    parser.add_argument("--incremental", action="store_true", help="Incremental update")
    parser.add_argument("--features-only", action="store_true", help="Compute features only, no training")
    parser.add_argument("--status", action="store_true", help="Show processor status")
    parser.add_argument("--predict", type=str, help="Show prediction for a symbol")
    args = parser.parse_args()

    dp = DataProcessor()

    if args.status:
        dp.print_status()
    elif args.predict:
        pred = dp.get_prediction(args.predict.upper())
        if pred:
            print(f"\n  {pred['symbol']} ({pred['date']}):")
            print(f"    ML:       {pred['ml_signal']} (prob={pred['ml_prob_up']:.3f}, conf={pred['ml_confidence']:.1f})")
            print(f"    Ensemble: {pred['ensemble_signal']} (prob={pred['ensemble_prob_up']:.3f}, conf={pred['ensemble_confidence']:.1f}, agree={pred['ensemble_agree']}/3)")
            print(f"    Device:   {pred['device']}")
        else:
            print(f"  No prediction found for {args.predict.upper()}")
    elif args.full:
        dp.run(retrain=True)
    elif args.incremental:
        dp.run_incremental()
    elif args.features_only:
        dp.run(retrain=False)
    else:
        print("Usage: python -m src.ops.data_processor --full|--incremental|--features-only|--status")
        print(f"\nHardware: {dp._hw}")
