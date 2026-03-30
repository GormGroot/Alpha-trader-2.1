"""
Pattern Strategy — wraps PatternScanner as a BaseStrategy
so pattern analysis (chart patterns, candlesticks, S/R, divergences,
seasonality, multi-timeframe) feeds into the trading signal pipeline.

Runs a background thread that continuously scans all symbols and caches
results, so when enabled the data is immediately available without
delaying the main scan cycle.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from loguru import logger

from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult
from src.strategy.patterns import PatternScanner, PatternScanResult


class PatternStrategy(BaseStrategy):
    """
    Strategy that runs the full PatternScanner and converts
    the aggregated result into a BUY/SELL/HOLD signal.

    Background scanning runs continuously regardless of whether the
    strategy is active in the trading pipeline. When activated, cached
    results are returned instantly. When deactivated, scanning still
    runs so data is warm when re-enabled.
    """

    def __init__(
        self,
        swing_order: int = 5,
        price_tolerance: float = 0.02,
        sr_min_touches: int = 2,
        candlestick_lookback: int = 5,
        divergence_lookback: int = 50,
    ):
        self._scanner = PatternScanner(
            swing_order=swing_order,
            price_tolerance=price_tolerance,
            sr_min_touches=sr_min_touches,
            candlestick_lookback=candlestick_lookback,
            divergence_lookback=divergence_lookback,
        )
        # Cache: symbol -> (timestamp, PatternScanResult)
        self._cache: Dict[str, tuple[float, PatternScanResult]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._lock = threading.Lock()

        # Background scanner
        self._bg_running = False
        self._bg_thread: Optional[threading.Thread] = None
        self._bg_data: Dict[str, pd.DataFrame] = {}  # symbol -> latest df
        self._bg_interval = 60  # scan every 60 seconds
        self._bg_max_symbols = 100  # max symbols to keep in memory

    @property
    def name(self) -> str:
        return "pattern_analysis"

    # ── Background scanner ──────────────────────────────────

    def start_background(self) -> None:
        """Start background pattern scanning thread."""
        if self._bg_running:
            return
        self._bg_running = True
        self._bg_thread = threading.Thread(
            target=self._bg_loop,
            daemon=True,
            name="PatternBgScanner",
        )
        self._bg_thread.start()
        logger.info("[pattern] Background scanner started")

    def stop_background(self) -> None:
        """Stop background scanning."""
        self._bg_running = False
        if self._bg_thread:
            self._bg_thread.join(timeout=5)
        logger.info("[pattern] Background scanner stopped")

    def update_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Feed latest market data for background scanning."""
        self._bg_data[symbol] = df

    def update_all_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Feed all symbol data at once. Replaces old data to cap memory."""
        # Replace entirely instead of accumulating — only keep current scan set
        self._bg_data = dict(data)

    def _bg_loop(self) -> None:
        """Background loop: scan all symbols with cached data."""
        while self._bg_running:
            # Evict stale cache entries (older than 2x TTL)
            stale_cutoff = time.time() - self._cache_ttl * 2
            with self._lock:
                stale = [s for s, (ts, _) in self._cache.items() if ts < stale_cutoff]
                for s in stale:
                    del self._cache[s]
            symbols = list(self._bg_data.keys())
            if symbols:
                scanned = 0
                for symbol in symbols:
                    if not self._bg_running:
                        return
                    df = self._bg_data.get(symbol)
                    if df is None or len(df) < 50:
                        continue
                    try:
                        result = self._scanner.scan(
                            df, symbol=symbol,
                            include_seasonal=True,
                            include_mtf=True,
                        )
                        with self._lock:
                            self._cache[symbol] = (time.time(), result)
                        scanned += 1
                    except Exception as e:
                        logger.debug(f"[pattern] bg scan {symbol}: {e}")

                if scanned > 0:
                    logger.debug(
                        f"[pattern] Background scan: {scanned}/{len(symbols)} symbols updated"
                    )

            # Wait for next cycle
            for _ in range(self._bg_interval):
                if not self._bg_running:
                    return
                time.sleep(1)

    def get_cached_result(self, symbol: str) -> Optional[PatternScanResult]:
        """Get cached scan result for a symbol, if fresh enough."""
        with self._lock:
            entry = self._cache.get(symbol)
        if entry is None:
            return None
        ts, result = entry
        if time.time() - ts > self._cache_ttl:
            return None
        return result

    # ── BaseStrategy interface ──────────────────────────────

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        if not self.validate_data(df, min_rows=50):
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data for pattern analysis")

        # Try to identify symbol from df name attribute or use cache lookup
        symbol = getattr(df, "name", None) or ""

        # Check cache first (from background scanner)
        cached = self.get_cached_result(symbol) if symbol else None

        try:
            if cached is not None:
                result = cached
            else:
                # Run scan inline (fallback if no cache)
                result = self._scanner.scan(
                    df, symbol=symbol,
                    include_seasonal=True,
                    include_mtf=True,
                )
                # Store in cache
                if symbol:
                    with self._lock:
                        self._cache[symbol] = (time.time(), result)

            signal = result.overall_signal
            confidence = result.overall_confidence

            # If nothing was detected, return HOLD with 0 confidence
            # so SignalEngine skips this strategy instead of diluting
            if confidence <= 0 and signal == Signal.HOLD:
                return StrategyResult(Signal.HOLD, 0.0, "No patterns detected")

            # Build reason from detected patterns
            reasons = []
            if result.chart_patterns:
                names = [p.pattern_type.value for p in result.chart_patterns[:3]]
                reasons.append(f"chart: {', '.join(names)}")
            if result.candlestick_patterns:
                names = [p.pattern_type.value for p in result.candlestick_patterns[:2]]
                reasons.append(f"candle: {', '.join(names)}")
            if result.breakouts:
                reasons.append(f"{len(result.breakouts)} breakout(s)")
            if result.divergences:
                names = [d.divergence_type for d in result.divergences[:2]]
                reasons.append(f"div: {', '.join(names)}")
            if result.multi_timeframe:
                consensus = result.multi_timeframe.consensus
                c_str = consensus.value if hasattr(consensus, "value") else str(consensus)
                reasons.append(f"MTF: {c_str}")

            reason = " | ".join(reasons) if reasons else result.summary

            return StrategyResult(signal, confidence, reason)

        except Exception as e:
            logger.debug(f"PatternStrategy error: {e}")
            return StrategyResult(Signal.HOLD, 0.0, f"Pattern scan error: {e}")
