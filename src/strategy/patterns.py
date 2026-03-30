"""
Mønstergenkendelse til teknisk aktieanalyse.

Indhold:
  1. Chart Patterns: Head & Shoulders, Double Top/Bottom, Cup & Handle,
     Triangle, Flag, Wedge
  2. Candlestick Patterns: Doji, Hammer, Engulfing, Morning/Evening Star,
     Three White Soldiers / Three Black Crows
  3. Support & Resistance: pivot points, volume-baseret, breakout detection
  4. Seasonality: månedlige, ugentlige, Santa Claus Rally, January Effect
  5. Divergens: RSI/MACD/MFI/OBV divergens-detektion
  6. Multi-timeframe analyse: daglig/ugentlig/månedlig konsensus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.data.indicators import (
    find_swing_points,
    SwingPoint,
    add_rsi,
    add_macd,
    add_mfi,
    add_volume_analysis,
    add_all_indicators,
    add_advanced_indicators,
)
from src.strategy.base_strategy import Signal


# ══════════════════════════════════════════════════════════════
#  Fælles dataclasses og enums
# ══════════════════════════════════════════════════════════════

class PatternDirection(Enum):
    """Retningen et mønster forudsiger."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternType(Enum):
    """Type af mønster."""
    # Chart patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    CUP_AND_HANDLE = "cup_and_handle"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    # Candlestick
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    # Divergens
    BULLISH_DIVERGENCE = "bullish_divergence"
    BEARISH_DIVERGENCE = "bearish_divergence"


@dataclass
class DetectedPattern:
    """Et detekteret mønster."""
    pattern_type: PatternType
    direction: PatternDirection
    confidence: float             # 0–100
    start_idx: int                # startposition i DataFrame
    end_idx: int                  # slutposition
    description: str              # menneskelig forklaring
    price_target: float | None = None   # evt. kursmål
    volume_confirmed: bool = False       # volumen bekræftet?
    indicator: str = ""                  # hvilken indikator (for divergens)


@dataclass
class SupportResistanceLevel:
    """Et support- eller resistance-niveau."""
    price: float
    level_type: str        # "support" eller "resistance"
    strength: int          # antal berøringer
    volume_weight: float   # volumen-styrke (0–1)
    first_touch: object    # dato
    last_touch: object     # dato

    @property
    def is_strong(self) -> bool:
        return self.strength >= 3


@dataclass
class BreakoutSignal:
    """Breakout over resistance eller under support."""
    level: SupportResistanceLevel
    breakout_price: float
    volume_ratio: float        # volumen vs. gennemsnit
    direction: str             # "up" eller "down"
    description: str


@dataclass
class SeasonalPattern:
    """Sæsonmønster for en aktie eller marked."""
    period: str                        # "monthly", "weekly", "daily"
    data: dict[str, float]             # nøgle → gennemsnitligt afkast
    best_period: str
    worst_period: str
    sell_in_may_effect: float | None   # gennemsnitligt afkast maj-okt vs. nov-apr
    santa_rally_avg: float | None      # gennemsnitligt december-afkast
    january_effect: float | None       # gennemsnitligt januar-afkast


@dataclass
class DivergenceSignal:
    """Divergens mellem pris og indikator."""
    divergence_type: str      # "bullish" eller "bearish"
    indicator: str            # "RSI", "MACD", "MFI", "OBV"
    price_direction: str      # "higher_highs" eller "lower_lows"
    indicator_direction: str  # "lower_highs" eller "higher_lows"
    confidence: float
    start_idx: int
    end_idx: int
    description: str


@dataclass
class TimeframeSignal:
    """Signal for en enkelt timeframe."""
    timeframe: str           # "daily", "weekly", "monthly"
    signal: Signal
    confidence: float
    reason: str


@dataclass
class MultiTimeframeResult:
    """Samlet multi-timeframe analyse."""
    signals: list[TimeframeSignal]
    consensus: Signal
    consensus_confidence: float
    aligned: bool             # alle timeframes enige?
    description: str


@dataclass
class PatternScanResult:
    """Samlet resultat af mønster-scanning."""
    symbol: str
    chart_patterns: list[DetectedPattern]
    candlestick_patterns: list[DetectedPattern]
    support_resistance: list[SupportResistanceLevel]
    breakouts: list[BreakoutSignal]
    divergences: list[DivergenceSignal]
    seasonal: SeasonalPattern | None
    multi_timeframe: MultiTimeframeResult | None
    overall_signal: Signal
    overall_confidence: float
    summary: str


# ══════════════════════════════════════════════════════════════
#  1. CHART PATTERNS
# ══════════════════════════════════════════════════════════════

class ChartPatternDetector:
    """Detekterer chart patterns via peak/trough analyse."""

    def __init__(self, order: int = 5, tolerance: float = 0.02):
        """
        Args:
            order: swing-punkt orden.
            tolerance: pris-tolerance for at matche niveauer (2%).
        """
        self.order = order
        self.tolerance = tolerance

    def detect_all(self, df: pd.DataFrame) -> list[DetectedPattern]:
        """Detektér alle chart patterns i DataFrame."""
        patterns: list[DetectedPattern] = []
        points = find_swing_points(df, order=self.order)

        if len(points) < 3:
            return patterns

        highs = [p for p in points if p.swing_type == "high"]
        lows = [p for p in points if p.swing_type == "low"]

        # Head and Shoulders / Inverse
        patterns.extend(self._detect_head_and_shoulders(highs, lows, df))

        # Double Top / Double Bottom
        patterns.extend(self._detect_double_top_bottom(highs, lows, df))

        # Triangle
        patterns.extend(self._detect_triangles(highs, lows, df))

        # Flag
        patterns.extend(self._detect_flags(highs, lows, df))

        # Wedge
        patterns.extend(self._detect_wedges(highs, lows, df))

        # Cup and Handle
        patterns.extend(self._detect_cup_and_handle(highs, lows, df))

        return patterns

    def _prices_match(self, p1: float, p2: float) -> bool:
        """Tjek om to priser er tæt nok (inden for tolerance)."""
        if p1 == 0:
            return False
        return abs(p1 - p2) / p1 < self.tolerance

    def _volume_confirms(self, df: pd.DataFrame, idx: int, direction: str) -> bool:
        """Tjek om volumen bekræfter mønsteret."""
        # H-17: Copy to avoid mutating caller's DataFrame (thread-safety)
        if "Volume_Ratio" not in df.columns:
            if "Volume" in df.columns and "Volume_SMA" not in df.columns:
                df = df.copy()
                add_volume_analysis(df)
            else:
                return False
        if idx >= len(df) or "Volume_Ratio" not in df.columns:
            return False
        ratio = df["Volume_Ratio"].iloc[min(idx, len(df) - 1)]
        return not pd.isna(ratio) and ratio > 1.2

    def _detect_head_and_shoulders(
        self,
        highs: list[SwingPoint],
        lows: list[SwingPoint],
        df: pd.DataFrame,
    ) -> list[DetectedPattern]:
        """Detektér Head and Shoulders / Inverse Head and Shoulders."""
        patterns = []

        # Head and Shoulders (top) – 3 swing highs, midterste højest
        for i in range(len(highs) - 2):
            left, head, right = highs[i], highs[i + 1], highs[i + 2]

            if head.price > left.price and head.price > right.price:
                # Skuldrene skal matche nogenlunde
                if self._prices_match(left.price, right.price):
                    conf = 50.0
                    shoulder_diff = abs(left.price - right.price) / left.price
                    if shoulder_diff < 0.01:
                        conf += 15
                    head_prominence = (head.price - max(left.price, right.price)) / head.price
                    if head_prominence > 0.03:
                        conf += 10
                    vol_ok = self._volume_confirms(df, right.index, "bearish")
                    if vol_ok:
                        conf += 10

                    neckline = min(left.price, right.price) * 0.98
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.HEAD_AND_SHOULDERS,
                        direction=PatternDirection.BEARISH,
                        confidence=min(conf, 90),
                        start_idx=left.index,
                        end_idx=right.index,
                        description=f"Head & Shoulders: hoved {head.price:.1f}, "
                                    f"skuldre {left.price:.1f}/{right.price:.1f}",
                        price_target=neckline - (head.price - neckline),
                        volume_confirmed=vol_ok,
                    ))

        # Inverse Head and Shoulders (bund)
        for i in range(len(lows) - 2):
            left, head, right = lows[i], lows[i + 1], lows[i + 2]

            if head.price < left.price and head.price < right.price:
                if self._prices_match(left.price, right.price):
                    conf = 50.0
                    shoulder_diff = abs(left.price - right.price) / left.price
                    if shoulder_diff < 0.01:
                        conf += 15
                    vol_ok = self._volume_confirms(df, right.index, "bullish")
                    if vol_ok:
                        conf += 10

                    neckline = max(left.price, right.price) * 1.02
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                        direction=PatternDirection.BULLISH,
                        confidence=min(conf, 90),
                        start_idx=left.index,
                        end_idx=right.index,
                        description=f"Inverse H&S: hoved {head.price:.1f}, "
                                    f"skuldre {left.price:.1f}/{right.price:.1f}",
                        price_target=neckline + (neckline - head.price),
                        volume_confirmed=vol_ok,
                    ))

        return patterns

    def _detect_double_top_bottom(
        self,
        highs: list[SwingPoint],
        lows: list[SwingPoint],
        df: pd.DataFrame,
    ) -> list[DetectedPattern]:
        """Detektér Double Top / Double Bottom."""
        patterns = []

        # Double Top – to næsten ens highs
        for i in range(len(highs) - 1):
            h1, h2 = highs[i], highs[i + 1]
            if self._prices_match(h1.price, h2.price) and h2.index - h1.index >= 10:
                conf = 55.0
                diff_pct = abs(h1.price - h2.price) / h1.price
                if diff_pct < 0.005:
                    conf += 15
                vol_ok = self._volume_confirms(df, h2.index, "bearish")
                if vol_ok:
                    conf += 10
                # Find trough imellem
                troughs_between = [l for l in lows if h1.index < l.index < h2.index]
                trough_price = min(l.price for l in troughs_between) if troughs_between else h1.price * 0.95
                target = trough_price - (h1.price - trough_price)

                patterns.append(DetectedPattern(
                    pattern_type=PatternType.DOUBLE_TOP,
                    direction=PatternDirection.BEARISH,
                    confidence=min(conf, 90),
                    start_idx=h1.index, end_idx=h2.index,
                    description=f"Double Top: {h1.price:.1f} og {h2.price:.1f}",
                    price_target=target,
                    volume_confirmed=vol_ok,
                ))

        # Double Bottom
        for i in range(len(lows) - 1):
            l1, l2 = lows[i], lows[i + 1]
            if self._prices_match(l1.price, l2.price) and l2.index - l1.index >= 10:
                conf = 55.0
                diff_pct = abs(l1.price - l2.price) / l1.price
                if diff_pct < 0.005:
                    conf += 15
                vol_ok = self._volume_confirms(df, l2.index, "bullish")
                if vol_ok:
                    conf += 10
                peaks_between = [h for h in highs if l1.index < h.index < l2.index]
                peak_price = max(h.price for h in peaks_between) if peaks_between else l1.price * 1.05
                target = peak_price + (peak_price - l1.price)

                patterns.append(DetectedPattern(
                    pattern_type=PatternType.DOUBLE_BOTTOM,
                    direction=PatternDirection.BULLISH,
                    confidence=min(conf, 90),
                    start_idx=l1.index, end_idx=l2.index,
                    description=f"Double Bottom: {l1.price:.1f} og {l2.price:.1f}",
                    price_target=target,
                    volume_confirmed=vol_ok,
                ))

        return patterns

    def _detect_triangles(
        self,
        highs: list[SwingPoint],
        lows: list[SwingPoint],
        df: pd.DataFrame,
    ) -> list[DetectedPattern]:
        """Detektér Ascending/Descending Triangle."""
        patterns = []
        if len(highs) < 2 or len(lows) < 2:
            return patterns

        # Brug de seneste 4-6 swing points
        recent_highs = highs[-3:] if len(highs) >= 3 else highs[-2:]
        recent_lows = lows[-3:] if len(lows) >= 3 else lows[-2:]

        # Ascending Triangle: flat highs + rising lows
        highs_flat = all(
            self._prices_match(recent_highs[0].price, h.price)
            for h in recent_highs[1:]
        )
        lows_rising = all(
            recent_lows[i + 1].price > recent_lows[i].price
            for i in range(len(recent_lows) - 1)
        )

        if highs_flat and lows_rising and len(recent_highs) >= 2:
            conf = 50.0 + len(recent_highs) * 5
            vol_ok = self._volume_confirms(df, recent_highs[-1].index, "bullish")
            if vol_ok:
                conf += 10
            resistance = np.mean([h.price for h in recent_highs])
            patterns.append(DetectedPattern(
                pattern_type=PatternType.ASCENDING_TRIANGLE,
                direction=PatternDirection.BULLISH,
                confidence=min(conf, 85),
                start_idx=min(recent_highs[0].index, recent_lows[0].index),
                end_idx=max(recent_highs[-1].index, recent_lows[-1].index),
                description=f"Ascending Triangle: modstand ~{resistance:.1f}, "
                            f"stigende støtte",
                price_target=resistance + (resistance - recent_lows[0].price),
                volume_confirmed=vol_ok,
            ))

        # Descending Triangle: flat lows + falling highs
        lows_flat = all(
            self._prices_match(recent_lows[0].price, l.price)
            for l in recent_lows[1:]
        )
        highs_falling = all(
            recent_highs[i + 1].price < recent_highs[i].price
            for i in range(len(recent_highs) - 1)
        )

        if lows_flat and highs_falling and len(recent_lows) >= 2:
            conf = 50.0 + len(recent_lows) * 5
            vol_ok = self._volume_confirms(df, recent_lows[-1].index, "bearish")
            if vol_ok:
                conf += 10
            support = np.mean([l.price for l in recent_lows])
            patterns.append(DetectedPattern(
                pattern_type=PatternType.DESCENDING_TRIANGLE,
                direction=PatternDirection.BEARISH,
                confidence=min(conf, 85),
                start_idx=min(recent_highs[0].index, recent_lows[0].index),
                end_idx=max(recent_highs[-1].index, recent_lows[-1].index),
                description=f"Descending Triangle: støtte ~{support:.1f}, "
                            f"faldende modstand",
                price_target=support - (recent_highs[0].price - support),
                volume_confirmed=vol_ok,
            ))

        return patterns

    def _detect_flags(
        self,
        highs: list[SwingPoint],
        lows: list[SwingPoint],
        df: pd.DataFrame,
    ) -> list[DetectedPattern]:
        """Detektér Bull/Bear Flag (kort konsolidering efter stærk move)."""
        patterns = []
        if len(df) < 30 or len(highs) < 2 or len(lows) < 2:
            return patterns

        close = df["Close"].values
        n = len(close)

        # Find stærke bevægelser (>5% over 10 bars)
        lookback = min(50, n)
        for i in range(max(15, n - lookback), n - 5):
            move = (close[i] - close[max(0, i - 10)]) / close[max(0, i - 10)]

            if abs(move) < 0.05:
                continue

            # Tjek om der er konsolidering efter (flag-fasen)
            flag_range = close[i:min(i + 15, n)]
            if len(flag_range) < 5:
                continue

            flag_vol = np.std(flag_range) / np.mean(flag_range) if np.mean(flag_range) > 0 else 1
            if flag_vol > 0.03:
                continue  # for volatil – ikke en flag

            if move > 0.05:
                # Bull Flag
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.BULL_FLAG,
                    direction=PatternDirection.BULLISH,
                    confidence=55.0,
                    start_idx=max(0, i - 10),
                    end_idx=min(i + 15, n - 1),
                    description=f"Bull Flag: +{move * 100:.1f}% stigning fulgt af konsolidering",
                    price_target=close[i] + (close[i] - close[max(0, i - 10)]),
                ))
                break  # kun det seneste
            elif move < -0.05:
                # Bear Flag
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.BEAR_FLAG,
                    direction=PatternDirection.BEARISH,
                    confidence=55.0,
                    start_idx=max(0, i - 10),
                    end_idx=min(i + 15, n - 1),
                    description=f"Bear Flag: {move * 100:.1f}% fald fulgt af konsolidering",
                    price_target=close[i] - (close[max(0, i - 10)] - close[i]),
                ))
                break

        return patterns

    def _detect_wedges(
        self,
        highs: list[SwingPoint],
        lows: list[SwingPoint],
        df: pd.DataFrame,
    ) -> list[DetectedPattern]:
        """Detektér Rising/Falling Wedge."""
        patterns = []
        if len(highs) < 3 or len(lows) < 3:
            return patterns

        rh = highs[-3:]
        rl = lows[-3:]

        # Rising Wedge: stigende highs + stigende lows, men highs konvergerer mod lows
        highs_rising = all(rh[i + 1].price > rh[i].price for i in range(len(rh) - 1))
        lows_rising = all(rl[i + 1].price > rl[i].price for i in range(len(rl) - 1))

        if highs_rising and lows_rising:
            high_slope = (rh[-1].price - rh[0].price) / max(1, rh[-1].index - rh[0].index)
            low_slope = (rl[-1].price - rl[0].price) / max(1, rl[-1].index - rl[0].index)

            if 0 < high_slope < low_slope * 1.5 and low_slope > high_slope * 0.5:
                # Converging = wedge
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.RISING_WEDGE,
                    direction=PatternDirection.BEARISH,
                    confidence=50.0,
                    start_idx=min(rh[0].index, rl[0].index),
                    end_idx=max(rh[-1].index, rl[-1].index),
                    description="Rising Wedge: konvergerende stigende trendlinjer (bearish)",
                ))

        # Falling Wedge
        highs_falling = all(rh[i + 1].price < rh[i].price for i in range(len(rh) - 1))
        lows_falling = all(rl[i + 1].price < rl[i].price for i in range(len(rl) - 1))

        if highs_falling and lows_falling:
            high_slope = (rh[-1].price - rh[0].price) / max(1, rh[-1].index - rh[0].index)
            low_slope = (rl[-1].price - rl[0].price) / max(1, rl[-1].index - rl[0].index)

            if high_slope < 0 and low_slope < 0 and abs(high_slope) < abs(low_slope) * 1.5:
                patterns.append(DetectedPattern(
                    pattern_type=PatternType.FALLING_WEDGE,
                    direction=PatternDirection.BULLISH,
                    confidence=50.0,
                    start_idx=min(rh[0].index, rl[0].index),
                    end_idx=max(rh[-1].index, rl[-1].index),
                    description="Falling Wedge: konvergerende faldende trendlinjer (bullish)",
                ))

        return patterns

    def _detect_cup_and_handle(
        self,
        highs: list[SwingPoint],
        lows: list[SwingPoint],
        df: pd.DataFrame,
    ) -> list[DetectedPattern]:
        """Detektér Cup and Handle mønster."""
        patterns = []
        if len(highs) < 3 or len(lows) < 2:
            return patterns

        # Kig efter to high-points der er tæt i pris (rim af koppen)
        # med et lavt punkt imellem (bunden af koppen)
        for i in range(len(highs) - 1):
            h1, h2 = highs[i], highs[i + 1]

            # Rim skal matche
            if not self._prices_match(h1.price, h2.price):
                continue

            # Bunden: find det laveste punkt imellem
            lows_between = [l for l in lows if h1.index < l.index < h2.index]
            if not lows_between:
                continue

            cup_bottom = min(lows_between, key=lambda l: l.price)
            depth = (h1.price - cup_bottom.price) / h1.price

            # Koppen skal være 10-40% dyb
            if not (0.10 <= depth <= 0.40):
                continue

            # Handle: lille dip efter h2
            handle_lows = [l for l in lows if l.index > h2.index and
                          l.price > cup_bottom.price and
                          l.index - h2.index < 20]

            conf = 50.0
            if depth > 0.15:
                conf += 10
            if handle_lows:
                conf += 10
            vol_ok = self._volume_confirms(df, h2.index, "bullish")
            if vol_ok:
                conf += 10

            target = h1.price + (h1.price - cup_bottom.price)

            patterns.append(DetectedPattern(
                pattern_type=PatternType.CUP_AND_HANDLE,
                direction=PatternDirection.BULLISH,
                confidence=min(conf, 85),
                start_idx=h1.index,
                end_idx=h2.index + (handle_lows[-1].index - h2.index if handle_lows else 0),
                description=f"Cup & Handle: rim ~{h1.price:.1f}, bund {cup_bottom.price:.1f}, "
                            f"dybde {depth * 100:.0f}%",
                price_target=target,
                volume_confirmed=vol_ok,
            ))
            break  # kun den bedste

        return patterns


# ══════════════════════════════════════════════════════════════
#  2. CANDLESTICK PATTERNS
# ══════════════════════════════════════════════════════════════

class CandlestickDetector:
    """Detekterer candlestick patterns i OHLC data."""

    def __init__(self, body_threshold: float = 0.001):
        """
        Args:
            body_threshold: minimum body-størrelse relativt til pris
                            for at tælle som ikke-doji.
        """
        self.body_threshold = body_threshold

    def detect_all(self, df: pd.DataFrame, lookback: int = 5) -> list[DetectedPattern]:
        """Detektér candlestick patterns i de seneste `lookback` bars."""
        patterns: list[DetectedPattern] = []
        n = len(df)
        if n < 3:
            return patterns

        start = max(0, n - lookback)

        for i in range(start, n):
            patterns.extend(self._check_single(df, i))
            if i >= 1:
                patterns.extend(self._check_two(df, i))
            if i >= 2:
                patterns.extend(self._check_three(df, i))

        return patterns

    def _body(self, row) -> float:
        return abs(row["Close"] - row["Open"])

    def _range(self, row) -> float:
        return row["High"] - row["Low"]

    def _is_bullish(self, row) -> bool:
        return row["Close"] > row["Open"]

    def _is_bearish(self, row) -> bool:
        return row["Close"] < row["Open"]

    def _upper_shadow(self, row) -> float:
        return row["High"] - max(row["Open"], row["Close"])

    def _lower_shadow(self, row) -> float:
        return min(row["Open"], row["Close"]) - row["Low"]

    def _check_single(self, df: pd.DataFrame, i: int) -> list[DetectedPattern]:
        """Tjek enkelt-bar mønstre."""
        patterns = []
        row = df.iloc[i]
        body = self._body(row)
        rng = self._range(row)
        if rng == 0:
            return patterns

        body_ratio = body / rng
        upper = self._upper_shadow(row)
        lower = self._lower_shadow(row)

        # Doji: lille krop, lange skygger
        if body_ratio < 0.1:
            patterns.append(DetectedPattern(
                pattern_type=PatternType.DOJI,
                direction=PatternDirection.NEUTRAL,
                confidence=45.0 + (1 - body_ratio) * 20,
                start_idx=i, end_idx=i,
                description=f"Doji: krop {body:.2f}, range {rng:.2f} – ubeslutsomhed",
            ))

        # Hammer (i downtrend): lille krop øverst, lang nedre skygge
        if body_ratio < 0.3 and lower > body * 2 and upper < body * 0.5:
            # Tjek om der er en forudgående downtrend
            if i >= 5:
                prev_close = df["Close"].iloc[max(0, i - 5):i]
                in_downtrend = prev_close.iloc[-1] < prev_close.iloc[0]
                if in_downtrend:
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.HAMMER,
                        direction=PatternDirection.BULLISH,
                        confidence=55.0,
                        start_idx=i, end_idx=i,
                        description=f"Hammer: lang nedre skygge ({lower:.2f}), lille krop – potentiel reversal",
                    ))
                else:
                    patterns.append(DetectedPattern(
                        pattern_type=PatternType.HANGING_MAN,
                        direction=PatternDirection.BEARISH,
                        confidence=50.0,
                        start_idx=i, end_idx=i,
                        description=f"Hanging Man: lang nedre skygge i uptrend – potentiel reversal",
                    ))

        return patterns

    def _check_two(self, df: pd.DataFrame, i: int) -> list[DetectedPattern]:
        """Tjek to-bar mønstre."""
        patterns = []
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        curr_body = self._body(curr)
        prev_body = self._body(prev)

        # Bullish Engulfing
        if (self._is_bearish(prev) and self._is_bullish(curr) and
                curr["Open"] <= prev["Close"] and curr["Close"] >= prev["Open"] and
                curr_body > prev_body):
            patterns.append(DetectedPattern(
                pattern_type=PatternType.BULLISH_ENGULFING,
                direction=PatternDirection.BULLISH,
                confidence=60.0,
                start_idx=i - 1, end_idx=i,
                description="Bullish Engulfing: grøn candle opsluger rød – bullish reversal",
            ))

        # Bearish Engulfing
        if (self._is_bullish(prev) and self._is_bearish(curr) and
                curr["Open"] >= prev["Close"] and curr["Close"] <= prev["Open"] and
                curr_body > prev_body):
            patterns.append(DetectedPattern(
                pattern_type=PatternType.BEARISH_ENGULFING,
                direction=PatternDirection.BEARISH,
                confidence=60.0,
                start_idx=i - 1, end_idx=i,
                description="Bearish Engulfing: rød candle opsluger grøn – bearish reversal",
            ))

        return patterns

    def _check_three(self, df: pd.DataFrame, i: int) -> list[DetectedPattern]:
        """Tjek tre-bar mønstre."""
        patterns = []
        c0 = df.iloc[i - 2]
        c1 = df.iloc[i - 1]
        c2 = df.iloc[i]

        # Morning Star
        if (self._is_bearish(c0) and
                self._body(c1) < self._body(c0) * 0.3 and
                self._is_bullish(c2) and
                c2["Close"] > (c0["Open"] + c0["Close"]) / 2):
            patterns.append(DetectedPattern(
                pattern_type=PatternType.MORNING_STAR,
                direction=PatternDirection.BULLISH,
                confidence=65.0,
                start_idx=i - 2, end_idx=i,
                description="Morning Star: 3-bar bullish reversal mønster",
            ))

        # Evening Star
        if (self._is_bullish(c0) and
                self._body(c1) < self._body(c0) * 0.3 and
                self._is_bearish(c2) and
                c2["Close"] < (c0["Open"] + c0["Close"]) / 2):
            patterns.append(DetectedPattern(
                pattern_type=PatternType.EVENING_STAR,
                direction=PatternDirection.BEARISH,
                confidence=65.0,
                start_idx=i - 2, end_idx=i,
                description="Evening Star: 3-bar bearish reversal mønster",
            ))

        # Three White Soldiers
        if (self._is_bullish(c0) and self._is_bullish(c1) and self._is_bullish(c2) and
                c1["Close"] > c0["Close"] and c2["Close"] > c1["Close"] and
                self._body(c1) > self._range(c1) * 0.5 and
                self._body(c2) > self._range(c2) * 0.5):
            patterns.append(DetectedPattern(
                pattern_type=PatternType.THREE_WHITE_SOLDIERS,
                direction=PatternDirection.BULLISH,
                confidence=65.0,
                start_idx=i - 2, end_idx=i,
                description="Three White Soldiers: 3 stigende grønne candles – stærkt bullish",
            ))

        # Three Black Crows
        if (self._is_bearish(c0) and self._is_bearish(c1) and self._is_bearish(c2) and
                c1["Close"] < c0["Close"] and c2["Close"] < c1["Close"] and
                self._body(c1) > self._range(c1) * 0.5 and
                self._body(c2) > self._range(c2) * 0.5):
            patterns.append(DetectedPattern(
                pattern_type=PatternType.THREE_BLACK_CROWS,
                direction=PatternDirection.BEARISH,
                confidence=65.0,
                start_idx=i - 2, end_idx=i,
                description="Three Black Crows: 3 faldende røde candles – stærkt bearish",
            ))

        return patterns


# ══════════════════════════════════════════════════════════════
#  3. SUPPORT & RESISTANCE
# ══════════════════════════════════════════════════════════════

class SupportResistanceDetector:
    """Detekterer support- og resistance-niveauer."""

    def __init__(self, tolerance: float = 0.015, min_touches: int = 2):
        self.tolerance = tolerance
        self.min_touches = min_touches

    def detect_levels(
        self,
        df: pd.DataFrame,
        order: int = 5,
        max_levels: int = 10,
    ) -> list[SupportResistanceLevel]:
        """Find support- og resistance-niveauer fra pivot points."""
        points = find_swing_points(df, order=order)

        if not points:
            return []

        # Gruppér priser der er tæt nok
        levels: list[SupportResistanceLevel] = []
        used = set()

        for i, p in enumerate(points):
            if i in used:
                continue

            cluster = [p]
            used.add(i)

            for j, q in enumerate(points):
                if j in used:
                    continue
                if abs(p.price - q.price) / p.price < self.tolerance:
                    cluster.append(q)
                    used.add(j)

            if len(cluster) < self.min_touches:
                continue

            avg_price = np.mean([c.price for c in cluster])
            # Type: over current pris = resistance, under = support
            current_price = df["Close"].iloc[-1]
            level_type = "resistance" if avg_price > current_price else "support"

            # Volumen-vægt
            vol_weight = 0.5
            if "Volume" in df.columns:
                indices = [min(c.index, len(df) - 1) for c in cluster]
                level_vols = [df["Volume"].iloc[idx] for idx in indices]
                avg_vol = df["Volume"].mean()
                vol_weight = min(1.0, np.mean(level_vols) / avg_vol) if avg_vol > 0 else 0.5

            levels.append(SupportResistanceLevel(
                price=float(avg_price),
                level_type=level_type,
                strength=len(cluster),
                volume_weight=vol_weight,
                first_touch=cluster[0].date,
                last_touch=cluster[-1].date,
            ))

        # Sortér: stærkeste først
        levels.sort(key=lambda x: x.strength, reverse=True)
        return levels[:max_levels]

    def detect_breakouts(
        self,
        df: pd.DataFrame,
        levels: list[SupportResistanceLevel],
        lookback: int = 5,
    ) -> list[BreakoutSignal]:
        """Detektér breakouts over resistance / under support."""
        breakouts = []
        if len(df) < lookback or not levels:
            return breakouts

        current = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-lookback]

        # Volumen-ratio
        vol_ratio = 1.0
        if "Volume_Ratio" in df.columns:
            vol_ratio = df["Volume_Ratio"].iloc[-1]
            if pd.isna(vol_ratio):
                vol_ratio = 1.0

        for level in levels:
            # Breakout op over resistance
            if level.level_type == "resistance":
                if prev_close < level.price and current > level.price:
                    breakouts.append(BreakoutSignal(
                        level=level,
                        breakout_price=current,
                        volume_ratio=vol_ratio,
                        direction="up",
                        description=f"Breakout over modstand {level.price:.1f} "
                                    f"({level.strength} berøringer) med "
                                    f"{vol_ratio:.1f}x volumen",
                    ))

            # Breakout ned under support
            elif level.level_type == "support":
                if prev_close > level.price and current < level.price:
                    breakouts.append(BreakoutSignal(
                        level=level,
                        breakout_price=current,
                        volume_ratio=vol_ratio,
                        direction="down",
                        description=f"Breakout under støtte {level.price:.1f} "
                                    f"({level.strength} berøringer) med "
                                    f"{vol_ratio:.1f}x volumen",
                    ))

        return breakouts


# ══════════════════════════════════════════════════════════════
#  4. SÆSONMØNSTRE (SEASONALITY)
# ══════════════════════════════════════════════════════════════

class SeasonalityAnalyzer:
    """Analyserer sæsonmønstre i aktiekurser."""

    MONTH_NAMES = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Maj", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dec",
    }

    def analyze(self, df: pd.DataFrame, min_years: int = 2) -> SeasonalPattern | None:
        """
        Analysér sæsonmønstre baseret på historisk data.

        Kræver mindst `min_years` års data.
        """
        if len(df) < 252 * min_years:
            return None

        daily_returns = df["Close"].pct_change().dropna()

        # Månedlige afkast
        monthly = {}
        for month in range(1, 13):
            mask = daily_returns.index.month == month
            month_ret = daily_returns[mask]
            if len(month_ret) > 0:
                monthly[self.MONTH_NAMES[month]] = float(month_ret.mean() * 21 * 100)

        if not monthly:
            return None

        best = max(monthly, key=monthly.get)
        worst = min(monthly, key=monthly.get)

        # Sell in May: sammenlign maj-okt vs. nov-apr
        summer = daily_returns[daily_returns.index.month.isin([5, 6, 7, 8, 9, 10])]
        winter = daily_returns[daily_returns.index.month.isin([11, 12, 1, 2, 3, 4])]

        sell_in_may = None
        if len(summer) > 0 and len(winter) > 0:
            sell_in_may = float((winter.mean() - summer.mean()) * 252 * 100)

        # Santa Claus Rally (december)
        dec = daily_returns[daily_returns.index.month == 12]
        santa = float(dec.mean() * 21 * 100) if len(dec) > 0 else None

        # January Effect
        jan = daily_returns[daily_returns.index.month == 1]
        jan_effect = float(jan.mean() * 21 * 100) if len(jan) > 0 else None

        return SeasonalPattern(
            period="monthly",
            data=monthly,
            best_period=best,
            worst_period=worst,
            sell_in_may_effect=sell_in_may,
            santa_rally_avg=santa,
            january_effect=jan_effect,
        )


# ══════════════════════════════════════════════════════════════
#  5. DIVERGENS-DETEKTION
# ══════════════════════════════════════════════════════════════

class DivergenceDetector:
    """Detekterer pris/indikator-divergenser."""

    def detect_all(
        self,
        df: pd.DataFrame,
        lookback: int = 50,
        order: int = 5,
    ) -> list[DivergenceSignal]:
        """Detektér divergenser på RSI, MACD, MFI og OBV."""
        # H-17: Copy to avoid mutating caller's DataFrame (thread-safety)
        df = df.copy()
        divergences = []

        # Sørg for at indikatorerne er beregnet
        if "RSI" not in df.columns:
            add_rsi(df)
        if "MACD" not in df.columns:
            add_macd(df)
        if "MFI" not in df.columns and "Volume" in df.columns:
            add_mfi(df)
        if "OBV" not in df.columns and "Volume" in df.columns:
            add_volume_analysis(df)

        indicators = {
            "RSI": "RSI",
            "MACD": "MACD",
        }
        if "MFI" in df.columns:
            indicators["MFI"] = "MFI"
        if "OBV" in df.columns:
            indicators["OBV"] = "OBV"

        # Brug kun lookback-perioden
        df_window = df.tail(lookback) if len(df) > lookback else df
        price_points = find_swing_points(df_window, order=order, column_high="Close", column_low="Close")
        price_highs = [p for p in price_points if p.swing_type == "high"]
        price_lows = [p for p in price_points if p.swing_type == "low"]

        for ind_name, ind_col in indicators.items():
            if ind_col not in df_window.columns:
                continue

            # Lav en temp DataFrame med indikatoren som High/Low for swing detection
            ind_series = df_window[ind_col].values
            n = len(df_window)

            # Find swing highs/lows i indikatoren manuelt
            ind_highs = []
            ind_lows = []
            for i in range(order, n - order):
                # Swing high
                is_high = True
                for j in range(1, order + 1):
                    if ind_series[i] <= ind_series[i - j] or ind_series[i] <= ind_series[i + j]:
                        is_high = False
                        break
                if is_high:
                    ind_highs.append((i, float(ind_series[i])))

                # Swing low
                is_low = True
                for j in range(1, order + 1):
                    if ind_series[i] >= ind_series[i - j] or ind_series[i] >= ind_series[i + j]:
                        is_low = False
                        break
                if is_low:
                    ind_lows.append((i, float(ind_series[i])))

            # Bearish Divergence: pris nye highs + indikator lavere highs
            if len(price_highs) >= 2 and len(ind_highs) >= 2:
                ph1, ph2 = price_highs[-2], price_highs[-1]
                # Find nærmeste indikator-highs
                ih_near = [h for h in ind_highs if abs(h[0] - ph2.index) < order * 3]
                ih_prev = [h for h in ind_highs if abs(h[0] - ph1.index) < order * 3]

                if ih_near and ih_prev:
                    if ph2.price > ph1.price and ih_near[-1][1] < ih_prev[-1][1]:
                        divergences.append(DivergenceSignal(
                            divergence_type="bearish",
                            indicator=ind_name,
                            price_direction="higher_highs",
                            indicator_direction="lower_highs",
                            confidence=60.0,
                            start_idx=ph1.index,
                            end_idx=ph2.index,
                            description=f"Bearish divergens: pris nye highs, men {ind_name} lavere highs",
                        ))

            # Bullish Divergence: pris nye lows + indikator højere lows
            if len(price_lows) >= 2 and len(ind_lows) >= 2:
                pl1, pl2 = price_lows[-2], price_lows[-1]
                il_near = [l for l in ind_lows if abs(l[0] - pl2.index) < order * 3]
                il_prev = [l for l in ind_lows if abs(l[0] - pl1.index) < order * 3]

                if il_near and il_prev:
                    if pl2.price < pl1.price and il_near[-1][1] > il_prev[-1][1]:
                        divergences.append(DivergenceSignal(
                            divergence_type="bullish",
                            indicator=ind_name,
                            price_direction="lower_lows",
                            indicator_direction="higher_lows",
                            confidence=60.0,
                            start_idx=pl1.index,
                            end_idx=pl2.index,
                            description=f"Bullish divergens: pris nye lows, men {ind_name} højere lows",
                        ))

        return divergences


# ══════════════════════════════════════════════════════════════
#  6. MULTI-TIMEFRAME ANALYSE
# ══════════════════════════════════════════════════════════════

class MultiTimeframeAnalyzer:
    """Analyserer signaler på tværs af daglig, ugentlig og månedlig timeframe."""

    def analyze(self, df: pd.DataFrame) -> MultiTimeframeResult:
        """
        Kør analyse på daglig, ugentlig og månedlig data.

        Signal er stærkest når alle timeframes er enige.
        """
        signals = []

        # Daglig (brug df som den er)
        daily_signal = self._analyze_timeframe(df, "daily")
        signals.append(daily_signal)

        # Ugentlig (resample)
        if len(df) >= 20:
            df_weekly = self._resample(df, "W")
            if len(df_weekly) >= 10:
                weekly_signal = self._analyze_timeframe(df_weekly, "weekly")
                signals.append(weekly_signal)

        # Månedlig (resample)
        if len(df) >= 60:
            df_monthly = self._resample(df, "ME")
            if len(df_monthly) >= 5:
                monthly_signal = self._analyze_timeframe(df_monthly, "monthly")
                signals.append(monthly_signal)

        # Konsensus
        buys = sum(1 for s in signals if s.signal == Signal.BUY)
        sells = sum(1 for s in signals if s.signal == Signal.SELL)
        total = len(signals)

        if buys > sells and buys > total / 2:
            consensus = Signal.BUY
        elif sells > buys and sells > total / 2:
            consensus = Signal.SELL
        else:
            consensus = Signal.HOLD

        aligned = all(s.signal == consensus for s in signals) and consensus != Signal.HOLD
        avg_conf = np.mean([s.confidence for s in signals if s.signal == consensus]) if any(
            s.signal == consensus for s in signals) else 30.0

        # Boost confidence when aligned
        if aligned:
            avg_conf = min(95, avg_conf + 15)

        labels = {s.timeframe: s.signal.value for s in signals}
        desc_parts = [f"{tf}={sig}" for tf, sig in labels.items()]
        desc = f"Multi-TF: {', '.join(desc_parts)}"
        if aligned:
            desc += f" → Stærkt {consensus.value}"

        return MultiTimeframeResult(
            signals=signals,
            consensus=consensus,
            consensus_confidence=float(avg_conf),
            aligned=aligned,
            description=desc,
        )

    @staticmethod
    def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLCV til en lavere frekvens."""
        resampled = pd.DataFrame()
        resampled["Open"] = df["Open"].resample(freq).first()
        resampled["High"] = df["High"].resample(freq).max()
        resampled["Low"] = df["Low"].resample(freq).min()
        resampled["Close"] = df["Close"].resample(freq).last()
        resampled["Volume"] = df["Volume"].resample(freq).sum()
        return resampled.dropna()

    @staticmethod
    def _analyze_timeframe(df: pd.DataFrame, timeframe: str) -> TimeframeSignal:
        """Enkel trendbaseret analyse for en timeframe."""
        if len(df) < 5:
            return TimeframeSignal(timeframe, Signal.HOLD, 30.0, "For lidt data")

        close = df["Close"]
        n = len(close)

        # SMA trend
        sma_short = close.rolling(min(10, n // 2)).mean()
        sma_long = close.rolling(min(30, n)).mean()

        last_close = close.iloc[-1]
        last_sma_short = sma_short.iloc[-1]
        last_sma_long = sma_long.iloc[-1]

        bullish_points = 0
        bearish_points = 0
        reasons = []

        # Kurs vs SMA
        if not pd.isna(last_sma_short):
            if last_close > last_sma_short:
                bullish_points += 1
                reasons.append("over kort SMA")
            else:
                bearish_points += 1
                reasons.append("under kort SMA")

        if not pd.isna(last_sma_long):
            if last_close > last_sma_long:
                bullish_points += 1
                reasons.append("over lang SMA")
            else:
                bearish_points += 1
                reasons.append("under lang SMA")

        # Momentum (5-bar return)
        if n >= 6:
            ret_5 = (close.iloc[-1] / close.iloc[-6] - 1)
            if ret_5 > 0.02:
                bullish_points += 1
                reasons.append(f"momentum +{ret_5 * 100:.1f}%")
            elif ret_5 < -0.02:
                bearish_points += 1
                reasons.append(f"momentum {ret_5 * 100:.1f}%")

        if bullish_points > bearish_points:
            signal = Signal.BUY
            conf = 40 + bullish_points * 10
        elif bearish_points > bullish_points:
            signal = Signal.SELL
            conf = 40 + bearish_points * 10
        else:
            signal = Signal.HOLD
            conf = 35

        return TimeframeSignal(
            timeframe=timeframe,
            signal=signal,
            confidence=min(conf, 80),
            reason="; ".join(reasons) if reasons else "neutral",
        )


# ══════════════════════════════════════════════════════════════
#  PATTERN SCANNER (samler alt)
# ══════════════════════════════════════════════════════════════

class PatternScanner:
    """
    Samlet mønstergenkendelse – kører alle detektorer og
    returnerer et samlet PatternScanResult.
    """

    def __init__(
        self,
        swing_order: int = 5,
        price_tolerance: float = 0.02,
        sr_min_touches: int = 2,
        candlestick_lookback: int = 5,
        divergence_lookback: int = 50,
    ):
        self.chart_detector = ChartPatternDetector(order=swing_order, tolerance=price_tolerance)
        self.candle_detector = CandlestickDetector()
        self.sr_detector = SupportResistanceDetector(tolerance=price_tolerance, min_touches=sr_min_touches)
        self.seasonal_analyzer = SeasonalityAnalyzer()
        self.divergence_detector = DivergenceDetector()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.candlestick_lookback = candlestick_lookback
        self.divergence_lookback = divergence_lookback

    def scan(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        include_seasonal: bool = True,
        include_mtf: bool = True,
    ) -> PatternScanResult:
        """
        Kør komplet mønster-scanning.

        Args:
            df: OHLCV DataFrame (med indikatorer).
            symbol: aktie-symbol.
            include_seasonal: inkludér sæsonanalyse.
            include_mtf: inkludér multi-timeframe analyse.
        """
        # Sørg for basis-indikatorer
        if "RSI" not in df.columns:
            add_rsi(df)
        if "MACD" not in df.columns:
            add_macd(df)
        if "Volume_Ratio" not in df.columns and "Volume" in df.columns:
            add_volume_analysis(df)

        # 1. Chart patterns
        chart_patterns = self.chart_detector.detect_all(df)

        # 2. Candlestick patterns
        candle_patterns = self.candle_detector.detect_all(df, lookback=self.candlestick_lookback)

        # 3. Support & Resistance + Breakouts
        sr_levels = self.sr_detector.detect_levels(df)
        breakouts = self.sr_detector.detect_breakouts(df, sr_levels)

        # 4. Divergenser
        divergences = self.divergence_detector.detect_all(df, lookback=self.divergence_lookback)

        # 5. Sæsonmønstre
        seasonal = self.seasonal_analyzer.analyze(df) if include_seasonal else None

        # 6. Multi-timeframe
        mtf = self.mtf_analyzer.analyze(df) if include_mtf else None

        # Samlet signal
        overall_signal, overall_confidence, summary = self._aggregate(
            chart_patterns, candle_patterns, breakouts, divergences, mtf, symbol,
        )

        return PatternScanResult(
            symbol=symbol,
            chart_patterns=chart_patterns,
            candlestick_patterns=candle_patterns,
            support_resistance=sr_levels,
            breakouts=breakouts,
            divergences=divergences,
            seasonal=seasonal,
            multi_timeframe=mtf,
            overall_signal=overall_signal,
            overall_confidence=overall_confidence,
            summary=summary,
        )

    def _aggregate(
        self,
        chart: list[DetectedPattern],
        candles: list[DetectedPattern],
        breakouts: list[BreakoutSignal],
        divergences: list[DivergenceSignal],
        mtf: MultiTimeframeResult | None,
        symbol: str,
    ) -> tuple[Signal, float, str]:
        """Aggregér alle signaler til ét samlet signal."""
        bullish_score = 0.0
        bearish_score = 0.0
        reasons = []

        # Chart patterns (vægt 30%)
        for p in chart:
            weight = p.confidence / 100 * 0.30
            if p.volume_confirmed:
                weight *= 1.3
            if p.direction == PatternDirection.BULLISH:
                bullish_score += weight
                reasons.append(f"Chart: {p.pattern_type.value} (bullish)")
            elif p.direction == PatternDirection.BEARISH:
                bearish_score += weight
                reasons.append(f"Chart: {p.pattern_type.value} (bearish)")

        # Candlestick patterns (vægt 15%)
        for p in candles:
            weight = p.confidence / 100 * 0.15
            if p.direction == PatternDirection.BULLISH:
                bullish_score += weight
            elif p.direction == PatternDirection.BEARISH:
                bearish_score += weight

        # Breakouts (vægt 25%)
        for b in breakouts:
            weight = 0.25 * min(b.volume_ratio / 2, 1.0)
            if b.direction == "up":
                bullish_score += weight
                reasons.append(f"Breakout op: {b.description}")
            else:
                bearish_score += weight
                reasons.append(f"Breakout ned: {b.description}")

        # Divergenser (vægt 20%)
        for d in divergences:
            weight = d.confidence / 100 * 0.20
            if d.divergence_type == "bullish":
                bullish_score += weight
                reasons.append(f"Bullish divergens ({d.indicator})")
            else:
                bearish_score += weight
                reasons.append(f"Bearish divergens ({d.indicator})")

        # Multi-timeframe (vægt 10%)
        if mtf:
            weight = mtf.consensus_confidence / 100 * 0.10
            if mtf.aligned:
                weight *= 1.5
            if mtf.consensus == Signal.BUY:
                bullish_score += weight
            elif mtf.consensus == Signal.SELL:
                bearish_score += weight

        # Bestem signal
        total = bullish_score + bearish_score
        if total < 0.05:
            signal = Signal.HOLD
            confidence = 30.0
        elif bullish_score > bearish_score * 1.3:
            signal = Signal.BUY
            confidence = min(85, 40 + bullish_score * 100)
        elif bearish_score > bullish_score * 1.3:
            signal = Signal.SELL
            confidence = min(85, 40 + bearish_score * 100)
        else:
            signal = Signal.HOLD
            confidence = 35.0

        summary_str = f"{symbol}: {signal.value} ({confidence:.0f}%)"
        if reasons:
            summary_str += " – " + "; ".join(reasons[:3])

        return signal, confidence, summary_str

    def get_confidence_adjustment(self, result: PatternScanResult) -> int:
        """
        Returnér confidence-justering til strategier (±15 points).

        Bruges til at booste/reducere strategi-confidence baseret på
        mønstergenkendelse.
        """
        if result.overall_signal == Signal.BUY:
            return min(15, int(result.overall_confidence / 6))
        elif result.overall_signal == Signal.SELL:
            return max(-15, -int(result.overall_confidence / 6))
        return 0

    def explain(self, result: PatternScanResult) -> str:
        """Generér human-readable forklaring af alle detekterede mønstre."""
        lines = [
            f"=== Mønsteranalyse: {result.symbol} ===",
            f"Samlet signal: {result.overall_signal.value} "
            f"(confidence: {result.overall_confidence:.0f}%)",
            "",
        ]

        if result.chart_patterns:
            lines.append("── Chart Patterns ──")
            for p in result.chart_patterns:
                vol = " ✓ vol" if p.volume_confirmed else ""
                target = f" → mål {p.price_target:.1f}" if p.price_target else ""
                lines.append(f"  {p.pattern_type.value}: {p.description}{vol}{target} "
                           f"[{p.confidence:.0f}%]")
            lines.append("")

        if result.candlestick_patterns:
            lines.append("── Candlestick Patterns ──")
            for p in result.candlestick_patterns:
                lines.append(f"  {p.pattern_type.value}: {p.description} [{p.confidence:.0f}%]")
            lines.append("")

        if result.support_resistance:
            lines.append("── Support & Resistance ──")
            for sr in result.support_resistance:
                strength = "★" * min(sr.strength, 5)
                lines.append(f"  {sr.level_type.upper()} {sr.price:.1f} "
                           f"({sr.strength} berøringer) {strength}")
            lines.append("")

        if result.breakouts:
            lines.append("── Breakouts ──")
            for b in result.breakouts:
                lines.append(f"  {b.description}")
            lines.append("")

        if result.divergences:
            lines.append("── Divergenser ──")
            for d in result.divergences:
                lines.append(f"  {d.description} [{d.confidence:.0f}%]")
            lines.append("")

        if result.seasonal:
            s = result.seasonal
            lines.append("── Sæsonmønstre ──")
            lines.append(f"  Bedste måned: {s.best_period} ({s.data.get(s.best_period, 0):+.2f}%)")
            lines.append(f"  Værste måned: {s.worst_period} ({s.data.get(s.worst_period, 0):+.2f}%)")
            if s.sell_in_may_effect is not None:
                lines.append(f"  Sell in May effekt: {s.sell_in_may_effect:+.2f}% "
                           f"(vinter vs. sommer)")
            if s.santa_rally_avg is not None:
                lines.append(f"  Santa Claus Rally (dec): {s.santa_rally_avg:+.2f}%")
            if s.january_effect is not None:
                lines.append(f"  January Effect: {s.january_effect:+.2f}%")
            lines.append("")

        if result.multi_timeframe:
            mtf = result.multi_timeframe
            lines.append("── Multi-Timeframe ──")
            for s in mtf.signals:
                lines.append(f"  {s.timeframe}: {s.signal.value} ({s.confidence:.0f}%) – {s.reason}")
            aligned = "✓ ALIGNED" if mtf.aligned else "✗ ikke aligned"
            lines.append(f"  Konsensus: {mtf.consensus.value} ({mtf.consensus_confidence:.0f}%) {aligned}")

        return "\n".join(lines)

    def print_report(self, result: PatternScanResult) -> None:
        """Print rapport til konsollen."""
        print(self.explain(result))
