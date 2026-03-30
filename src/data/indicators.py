"""
Tekniske indikatorer til aktieanalyse.

Alle funktioner tager en pandas DataFrame med mindst en "Close"-kolonne
(og "High"/"Low"/"Volume" hvor relevant) og returnerer DataFrame'en
med nye kolonner tilføjet.

Indhold:
  Basis: SMA, EMA, RSI, MACD, Bollinger Bands, Volume-analyse
  Fibonacci: Retracement, Extensions, swing-detection
  Ichimoku: Tenkan, Kijun, Senkou A/B, Chikou, signaler
  Elliott Wave: impulsbølger (5-wave), korrektive (ABC)
  Volume Profile: VAP, POC, Value Area
  Momentum: Stochastic RSI, Williams %R, MFI, CCI, ADX
  Volatilitet: ATR, Keltner Channels, Donchian Channels, Historical Vol
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════
#  BASIS-INDIKATORER (eksisterende)
# ══════════════════════════════════════════════════════════════


# ── SMA (Simple Moving Average) ─────────────────────────────

def add_sma(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Tilføj Simple Moving Average.

    Beregner gennemsnittet af de seneste `window` lukkekurser.
    Bruges til at identificere trendretning og støtte/modstand.
    """
    df[f"SMA_{window}"] = df[column].rolling(window=window).mean()
    return df


# ── EMA (Exponential Moving Average) ────────────────────────

def add_ema(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Tilføj Exponential Moving Average.

    Vægter nyere priser højere end ældre – reagerer hurtigere end SMA.
    Bruges til at fange momentum og hurtigere trend-ændringer.
    """
    df[f"EMA_{window}"] = df[column].ewm(span=window, adjust=False).mean()
    return df


# ── RSI (Relative Strength Index) ───────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.DataFrame:
    """
    Tilføj RSI (0–100).

    Måler styrken af prisbevægelser. Over 70 = overkøbt, under 30 = oversolgt.
    Bruges til at finde vendepunkter i kursen.
    """
    delta = df[column].diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Håndtér division med nul: kun gains → RSI=100, kun losses → RSI=0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss > 0, 100.0)
    rsi = rsi.where(avg_gain > 0, 0.0)
    df["RSI"] = rsi
    return df


# ── MACD (Moving Average Convergence Divergence) ────────────

def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Tilføj MACD-linje, signallinje og histogram.

    MACD = EMA(fast) − EMA(slow). Når MACD krydser over signallinjen,
    er det et købssignal – og omvendt. Bruges til at spotte trendskift.
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


# ── Bollinger Bands ──────────────────────────────────────────

def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Tilføj Bollinger Bands (øvre, midter, nedre).

    Måler volatilitet. Prisen holder sig typisk inden for båndene.
    Når prisen rammer det nedre bånd, kan den være oversolgt – og omvendt.
    """
    sma = df[column].rolling(window=window).mean()
    std = df[column].rolling(window=window).std()

    df["BB_Upper"] = sma + (num_std * std)
    df["BB_Middle"] = sma
    df["BB_Lower"] = sma - (num_std * std)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    return df


# ── Volume-analyse ───────────────────────────────────────────

def add_volume_analysis(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Tilføj volume-baserede indikatorer.

    - Volume SMA: gennemsnitlig volumen over `window` dage.
    - Volume Ratio: dagens volumen ÷ gennemsnit (>1 = over normalt).
    - OBV (On-Balance Volume): kumulativt mål for købs-/salgspres.
    """
    df["Volume_SMA"] = df["Volume"].rolling(window=window).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

    # OBV – On-Balance Volume
    direction = np.sign(df["Close"].diff())
    df["OBV"] = (direction * df["Volume"]).cumsum()

    return df


# ══════════════════════════════════════════════════════════════
#  FIBONACCI ANALYSE
# ══════════════════════════════════════════════════════════════

# Standard Fibonacci niveauer
FIB_RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXTENSION_LEVELS = [0.0, 0.618, 1.0, 1.272, 1.618, 2.0, 2.618]


@dataclass
class SwingPoint:
    """Et swing high eller swing low punkt."""
    index: int          # position i DataFrame
    date: object        # dato
    price: float        # pris
    swing_type: str     # "high" eller "low"


@dataclass
class FibonacciLevels:
    """Fibonacci retracement/extension niveauer beregnet fra swing points."""
    swing_high: SwingPoint
    swing_low: SwingPoint
    retracement_levels: dict[float, float]   # ratio → pris
    extension_levels: dict[float, float]     # ratio → pris
    trend: str                                # "uptrend" eller "downtrend"

    @property
    def current_retracement(self) -> float | None:
        """Returnér den nærmeste retracement-ratio baseret på seneste pris."""
        return None  # sættes udefra


def find_swing_points(
    df: pd.DataFrame,
    order: int = 5,
    column_high: str = "High",
    column_low: str = "Low",
) -> list[SwingPoint]:
    """
    Find swing highs og swing lows i prisdata.

    Et swing high er et punkt der er højere end de `order` foregående
    og efterfølgende bars. Et swing low er tilsvarende lavere.

    Args:
        df: OHLCV DataFrame.
        order: antal bars på hver side til at bekræfte swing (default 5).
    """
    points: list[SwingPoint] = []
    highs = df[column_high].values
    lows = df[column_low].values
    n = len(df)

    for i in range(order, n - order):
        # Swing High: højere end alle naboer
        is_swing_high = True
        for j in range(1, order + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing_high = False
                break
        if is_swing_high:
            points.append(SwingPoint(
                index=i,
                date=df.index[i],
                price=float(highs[i]),
                swing_type="high",
            ))

        # Swing Low: lavere end alle naboer
        is_swing_low = True
        for j in range(1, order + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing_low = False
                break
        if is_swing_low:
            points.append(SwingPoint(
                index=i,
                date=df.index[i],
                price=float(lows[i]),
                swing_type="low",
            ))

    # Sortér efter dato
    points.sort(key=lambda p: p.index)
    return points


def calc_fibonacci_levels(
    swing_high: SwingPoint,
    swing_low: SwingPoint,
) -> FibonacciLevels:
    """
    Beregn Fibonacci retracement og extension niveauer.

    I en uptrend: niveauer beregnes fra low → high.
    I en downtrend: niveauer beregnes fra high → low.
    """
    high_price = swing_high.price
    low_price = swing_low.price
    diff = high_price - low_price

    # Bestem trend: hvis high kom EFTER low, er det en uptrend
    trend = "uptrend" if swing_high.index > swing_low.index else "downtrend"

    # Retracements
    retracements: dict[float, float] = {}
    for level in FIB_RETRACEMENT_LEVELS:
        if trend == "uptrend":
            # Retracement ned fra toppen
            retracements[level] = high_price - diff * level
        else:
            # Retracement op fra bunden
            retracements[level] = low_price + diff * level

    # Extensions
    extensions: dict[float, float] = {}
    for level in FIB_EXTENSION_LEVELS:
        if trend == "uptrend":
            extensions[level] = high_price + diff * level
        else:
            extensions[level] = low_price - diff * level

    return FibonacciLevels(
        swing_high=swing_high,
        swing_low=swing_low,
        retracement_levels=retracements,
        extension_levels=extensions,
        trend=trend,
    )


def add_fibonacci(
    df: pd.DataFrame,
    order: int = 5,
    lookback: int = 100,
) -> pd.DataFrame:
    """
    Tilføj Fibonacci retracement-niveauer til DataFrame.

    Finder automatisk de mest recente swing high/low og beregner niveauer.
    Tilføjer kolonner: Fib_0, Fib_236, Fib_382, Fib_500, Fib_618, Fib_786, Fib_1.

    Signal: Kurs rammer 61.8% retracement + RSI oversold = stærkt køb.
    """
    df_window = df.tail(lookback) if len(df) > lookback else df
    points = find_swing_points(df_window, order=order)

    # Find seneste swing high og swing low
    swing_highs = [p for p in points if p.swing_type == "high"]
    swing_lows = [p for p in points if p.swing_type == "low"]

    if not swing_highs or not swing_lows:
        # Ikke nok data – brug periodisk high/low
        high_idx = df_window["High"].idxmax()
        low_idx = df_window["Low"].idxmin()
        high_pos = df_window.index.get_loc(high_idx)
        low_pos = df_window.index.get_loc(low_idx)
        swing_highs = [SwingPoint(high_pos, high_idx, float(df_window["High"].loc[high_idx]), "high")]
        swing_lows = [SwingPoint(low_pos, low_idx, float(df_window["Low"].loc[low_idx]), "low")]

    sh = swing_highs[-1]
    sl = swing_lows[-1]
    fib = calc_fibonacci_levels(sh, sl)

    # Tilføj som konstante niveauer
    level_names = {0.0: "Fib_0", 0.236: "Fib_236", 0.382: "Fib_382",
                   0.5: "Fib_500", 0.618: "Fib_618", 0.786: "Fib_786", 1.0: "Fib_1"}
    for ratio, col_name in level_names.items():
        df[col_name] = fib.retracement_levels.get(ratio, np.nan)

    # Extension targets
    ext_names = {1.272: "Fib_Ext_1272", 1.618: "Fib_Ext_1618", 2.618: "Fib_Ext_2618"}
    for ratio, col_name in ext_names.items():
        df[col_name] = fib.extension_levels.get(ratio, np.nan)

    # Fibonacci signal: tæt på 61.8% retracement?
    close = df["Close"].values
    fib_618 = fib.retracement_levels.get(0.618, np.nan)
    if not np.isnan(fib_618) and fib_618 > 0:
        tolerance = abs(fib.swing_high.price - fib.swing_low.price) * 0.02
        df["Fib_Near_618"] = np.abs(close - fib_618) < tolerance
    else:
        df["Fib_Near_618"] = False

    return df


# ══════════════════════════════════════════════════════════════
#  ICHIMOKU CLOUD
# ══════════════════════════════════════════════════════════════

@dataclass
class IchimokuSignal:
    """Sammenfatning af Ichimoku-signaler."""
    price_vs_cloud: str    # "above", "below", "inside"
    tk_cross: str          # "bullish", "bearish", "none"
    cloud_twist: bool      # True hvis Senkou A og B krydser snart
    overall: str           # "bullish", "bearish", "neutral"


def add_ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    Tilføj alle 5 Ichimoku Cloud linjer.

    - Tenkan-sen (conversion): midpoint af highest high + lowest low over 9 perioder
    - Kijun-sen (base): midpoint over 26 perioder
    - Senkou Span A: (Tenkan + Kijun) / 2, forskudt 26 perioder frem
    - Senkou Span B: midpoint over 52 perioder, forskudt 26 perioder frem
    - Chikou Span: Close forskudt 26 perioder tilbage

    Signaler:
    - Kurs over cloud = bullish
    - TK cross (Tenkan krydser Kijun) = trendændring
    - Cloud twist = muligt regime-skift
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan).max()
    tenkan_low = low.rolling(window=tenkan).min()
    df["Ichimoku_Tenkan"] = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun).max()
    kijun_low = low.rolling(window=kijun).min()
    df["Ichimoku_Kijun"] = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A) – forskudt frem
    span_a = (df["Ichimoku_Tenkan"] + df["Ichimoku_Kijun"]) / 2
    df["Ichimoku_SpanA"] = span_a.shift(displacement)

    # Senkou Span B (Leading Span B) – forskudt frem
    span_b_high = high.rolling(window=senkou_b).max()
    span_b_low = low.rolling(window=senkou_b).min()
    span_b = (span_b_high + span_b_low) / 2
    df["Ichimoku_SpanB"] = span_b.shift(displacement)

    # Chikou Span (Lagging Span) – forskudt tilbage
    df["Ichimoku_Chikou"] = close.shift(-displacement)

    # Cloud top og bund for nemhed
    df["Ichimoku_Cloud_Top"] = df[["Ichimoku_SpanA", "Ichimoku_SpanB"]].max(axis=1)
    df["Ichimoku_Cloud_Bottom"] = df[["Ichimoku_SpanA", "Ichimoku_SpanB"]].min(axis=1)

    return df


def get_ichimoku_signal(df: pd.DataFrame) -> IchimokuSignal:
    """
    Analysér Ichimoku-signaler baseret på seneste bar.

    Returns:
        IchimokuSignal med price_vs_cloud, tk_cross, cloud_twist, overall.
    """
    if len(df) < 2:
        return IchimokuSignal("inside", "none", False, "neutral")

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Kurs vs cloud
    close = last["Close"]
    cloud_top = last.get("Ichimoku_Cloud_Top", np.nan)
    cloud_bottom = last.get("Ichimoku_Cloud_Bottom", np.nan)

    if pd.isna(cloud_top) or pd.isna(cloud_bottom):
        price_vs_cloud = "inside"
    elif close > cloud_top:
        price_vs_cloud = "above"
    elif close < cloud_bottom:
        price_vs_cloud = "below"
    else:
        price_vs_cloud = "inside"

    # TK cross
    tenkan_curr = last.get("Ichimoku_Tenkan", np.nan)
    kijun_curr = last.get("Ichimoku_Kijun", np.nan)
    tenkan_prev = prev.get("Ichimoku_Tenkan", np.nan)
    kijun_prev = prev.get("Ichimoku_Kijun", np.nan)

    tk_cross = "none"
    if not any(pd.isna(v) for v in [tenkan_curr, kijun_curr, tenkan_prev, kijun_prev]):
        if tenkan_prev <= kijun_prev and tenkan_curr > kijun_curr:
            tk_cross = "bullish"
        elif tenkan_prev >= kijun_prev and tenkan_curr < kijun_curr:
            tk_cross = "bearish"

    # Cloud twist: Span A og B krydser
    span_a_curr = last.get("Ichimoku_SpanA", np.nan)
    span_b_curr = last.get("Ichimoku_SpanB", np.nan)
    span_a_prev = prev.get("Ichimoku_SpanA", np.nan)
    span_b_prev = prev.get("Ichimoku_SpanB", np.nan)

    cloud_twist = False
    if not any(pd.isna(v) for v in [span_a_curr, span_b_curr, span_a_prev, span_b_prev]):
        if (span_a_prev <= span_b_prev and span_a_curr > span_b_curr) or \
           (span_a_prev >= span_b_prev and span_a_curr < span_b_curr):
            cloud_twist = True

    # Overall signal
    bullish_points = 0
    bearish_points = 0
    if price_vs_cloud == "above":
        bullish_points += 2
    elif price_vs_cloud == "below":
        bearish_points += 2

    if tk_cross == "bullish":
        bullish_points += 1
    elif tk_cross == "bearish":
        bearish_points += 1

    if bullish_points > bearish_points:
        overall = "bullish"
    elif bearish_points > bullish_points:
        overall = "bearish"
    else:
        overall = "neutral"

    return IchimokuSignal(price_vs_cloud, tk_cross, cloud_twist, overall)


# ══════════════════════════════════════════════════════════════
#  ELLIOTT WAVE (simpel implementation)
# ══════════════════════════════════════════════════════════════

class WaveType(Enum):
    """Type af Elliott Wave."""
    IMPULSE = "impulse"       # 5-bølge mønster (trend-retning)
    CORRECTIVE = "corrective" # 3-bølge mønster (mod trend)
    UNKNOWN = "unknown"


@dataclass
class ElliottWave:
    """En identificeret Elliott Wave."""
    wave_number: int | str   # 1-5 for impulse, A/B/C for corrective
    start_price: float
    end_price: float
    start_idx: int
    end_idx: int
    wave_type: WaveType


@dataclass
class ElliottWaveAnalysis:
    """Samlet Elliott Wave analyse."""
    waves: list[ElliottWave]
    wave_type: WaveType
    current_wave: int | str     # hvilken bølge vi er i nu
    expected_direction: str     # "up" eller "down"
    confidence: float           # 0-100
    description: str


def analyze_elliott_waves(
    df: pd.DataFrame,
    order: int = 5,
    min_wave_pct: float = 0.03,
) -> ElliottWaveAnalysis:
    """
    Simpel Elliott Wave identifikation.

    Finder swing-points og forsøger at matche 5-bølge (impulse) eller
    3-bølge (corrective) mønstre. Elliott Wave er subjektivt –
    brug som supplement, ikke primært signal.

    Args:
        df: OHLCV DataFrame.
        order: swing-punkt orden.
        min_wave_pct: minimum procentvis bevægelse for at tælle som bølge.
    """
    points = find_swing_points(df, order=order)

    if len(points) < 4:
        return ElliottWaveAnalysis(
            waves=[], wave_type=WaveType.UNKNOWN,
            current_wave=0, expected_direction="unknown",
            confidence=0.0, description="Ikke nok datapunkter til wave-analyse",
        )

    # Brug de seneste swing points til at identificere waves
    recent = points[-7:] if len(points) >= 7 else points

    waves: list[ElliottWave] = []
    for i in range(len(recent) - 1):
        p1 = recent[i]
        p2 = recent[i + 1]
        move_pct = abs(p2.price - p1.price) / p1.price
        if move_pct < min_wave_pct:
            continue
        waves.append(ElliottWave(
            wave_number=len(waves) + 1,
            start_price=p1.price,
            end_price=p2.price,
            start_idx=p1.index,
            end_idx=p2.index,
            wave_type=WaveType.UNKNOWN,
        ))

    if not waves:
        return ElliottWaveAnalysis(
            waves=[], wave_type=WaveType.UNKNOWN,
            current_wave=0, expected_direction="unknown",
            confidence=0.0, description="Ingen signifikante bølger fundet",
        )

    # Prøv at matche impulsbølge (5 waves)
    wave_type = WaveType.UNKNOWN
    confidence = 20.0

    if len(waves) >= 5:
        # Tjek om de opfylder Elliott reglerne for impulsbølge
        w = waves[-5:]
        up_waves = [w[0], w[2], w[4]]  # 1, 3, 5
        down_waves = [w[1], w[3]]       # 2, 4

        all_up_positive = all(wv.end_price > wv.start_price for wv in up_waves)
        all_down_negative = all(wv.end_price < wv.start_price for wv in down_waves)
        all_up_negative = all(wv.end_price < wv.start_price for wv in up_waves)
        all_down_positive = all(wv.end_price > wv.start_price for wv in down_waves)

        if (all_up_positive and all_down_negative) or \
           (all_up_negative and all_down_positive):
            wave_type = WaveType.IMPULSE
            confidence = 55.0

            # Regel: Wave 3 er typisk den længste
            wave3_size = abs(w[2].end_price - w[2].start_price)
            wave1_size = abs(w[0].end_price - w[0].start_price)
            wave5_size = abs(w[4].end_price - w[4].start_price)
            if wave3_size >= wave1_size and wave3_size >= wave5_size:
                confidence += 15.0

            # Regel: Wave 2 retraces ikke mere end 100% af Wave 1
            w2_retrace = abs(w[1].end_price - w[1].start_price) / wave1_size if wave1_size > 0 else 999
            if w2_retrace < 1.0:
                confidence += 10.0

            # Re-label waves
            for idx, wv in enumerate(w):
                wv.wave_number = idx + 1
                wv.wave_type = WaveType.IMPULSE

    elif len(waves) >= 3:
        # Tjek korrektiv ABC
        w = waves[-3:]
        # A og C bevæger sig i samme retning, B i modsat
        a_dir = np.sign(w[0].end_price - w[0].start_price)
        b_dir = np.sign(w[1].end_price - w[1].start_price)
        c_dir = np.sign(w[2].end_price - w[2].start_price)

        if a_dir == c_dir and a_dir != b_dir:
            wave_type = WaveType.CORRECTIVE
            confidence = 45.0
            labels = ["A", "B", "C"]
            for idx, wv in enumerate(w):
                wv.wave_number = labels[idx]
                wv.wave_type = WaveType.CORRECTIVE

    # Bestem forventet retning
    if wave_type == WaveType.IMPULSE and len(waves) >= 5:
        # Efter 5 bølger forventes korrektion
        last_dir = "up" if waves[-1].end_price > waves[-1].start_price else "down"
        expected_direction = "down" if last_dir == "up" else "up"
        current_wave = 5
        description = (
            f"Muligt 5-bølge impulsmønster identificeret. "
            f"Wave 3 er {'den længste (klassisk)' if confidence >= 70 else 'ikke den længste'}. "
            f"Forventet korrektion {'ned' if expected_direction == 'down' else 'op'}."
        )
    elif wave_type == WaveType.CORRECTIVE and len(waves) >= 3:
        c_dir = "up" if waves[-1].end_price > waves[-1].start_price else "down"
        expected_direction = "up" if c_dir == "down" else "down"
        current_wave = "C"
        description = (
            f"Muligt korrektivt ABC-mønster. "
            f"Forventet ny impulsbølge {'op' if expected_direction == 'up' else 'ned'}."
        )
    else:
        expected_direction = "unknown"
        current_wave = len(waves)
        description = f"{len(waves)} bølger fundet, men mønsteret er uklart. OBS: Elliott Wave er subjektivt."

    confidence = min(confidence, 80.0)  # cap – aldrig for sikker

    return ElliottWaveAnalysis(
        waves=waves,
        wave_type=wave_type,
        current_wave=current_wave,
        expected_direction=expected_direction,
        confidence=confidence,
        description=description,
    )


# ══════════════════════════════════════════════════════════════
#  VOLUME PROFILE
# ══════════════════════════════════════════════════════════════

@dataclass
class VolumeProfile:
    """Volume at Price profil."""
    price_bins: np.ndarray        # pris-niveauer (midtpunkter)
    volume_at_price: np.ndarray   # volumen ved hvert niveau
    poc: float                     # Point of Control – pris med mest volumen
    value_area_high: float         # øverste grænse for Value Area (70%)
    value_area_low: float          # nederste grænse for Value Area (70%)
    total_volume: float


def calc_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    """
    Beregn Volume at Price profil.

    - POC (Point of Control): prisen med mest akkumuleret volumen.
    - Value Area: priszone med 70% af total volumen.

    Args:
        df: OHLCV DataFrame.
        num_bins: antal pris-niveauer at dele op i.
        value_area_pct: andel af volumen i value area (default 70%).
    """
    price_range = (df["Low"].min(), df["High"].max())
    bins = np.linspace(price_range[0], price_range[1], num_bins + 1)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2

    # Fordel volumen til bins baseret på OHLC
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    volume_at_price = np.zeros(num_bins)

    for i in range(len(df)):
        tp = typical_price.iloc[i]
        vol = df["Volume"].iloc[i]
        bin_idx = np.searchsorted(bins[1:], tp, side="left")
        bin_idx = min(bin_idx, num_bins - 1)
        volume_at_price[bin_idx] += vol

    # POC
    poc_idx = np.argmax(volume_at_price)
    poc = float(bin_midpoints[poc_idx])

    # Value Area (70%)
    total_volume = float(volume_at_price.sum())
    target_volume = total_volume * value_area_pct

    # Start fra POC og udvid symmetrisk
    va_indices = {poc_idx}
    accumulated = volume_at_price[poc_idx]
    low_idx = poc_idx - 1
    high_idx = poc_idx + 1

    while accumulated < target_volume:
        add_low = volume_at_price[low_idx] if low_idx >= 0 else 0
        add_high = volume_at_price[high_idx] if high_idx < num_bins else 0

        if add_low == 0 and add_high == 0:
            break

        if add_high >= add_low:
            if high_idx < num_bins:
                va_indices.add(high_idx)
                accumulated += add_high
                high_idx += 1
            if accumulated >= target_volume:
                break
            if low_idx >= 0:
                va_indices.add(low_idx)
                accumulated += add_low
                low_idx -= 1
        else:
            if low_idx >= 0:
                va_indices.add(low_idx)
                accumulated += add_low
                low_idx -= 1
            if accumulated >= target_volume:
                break
            if high_idx < num_bins:
                va_indices.add(high_idx)
                accumulated += add_high
                high_idx += 1

    va_high = float(bin_midpoints[max(va_indices)])
    va_low = float(bin_midpoints[min(va_indices)])

    return VolumeProfile(
        price_bins=bin_midpoints,
        volume_at_price=volume_at_price,
        poc=poc,
        value_area_high=va_high,
        value_area_low=va_low,
        total_volume=total_volume,
    )


def add_volume_profile(
    df: pd.DataFrame,
    lookback: int = 60,
    num_bins: int = 50,
) -> pd.DataFrame:
    """
    Tilføj Volume Profile niveauer til DataFrame.

    Tilføjer kolonner: VP_POC, VP_VA_High, VP_VA_Low.
    """
    df_window = df.tail(lookback) if len(df) > lookback else df
    vp = calc_volume_profile(df_window, num_bins=num_bins)

    df["VP_POC"] = vp.poc
    df["VP_VA_High"] = vp.value_area_high
    df["VP_VA_Low"] = vp.value_area_low

    return df


# ══════════════════════════════════════════════════════════════
#  AVANCEREDE MOMENTUM-INDIKATORER
# ══════════════════════════════════════════════════════════════

def add_stochastic_rsi(
    df: pd.DataFrame,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> pd.DataFrame:
    """
    Tilføj Stochastic RSI (kombination af Stochastic og RSI).

    StochRSI = (RSI - RSI_Low) / (RSI_High - RSI_Low)
    Mere følsom end standard RSI. Område: 0–100.
    Over 80 = overkøbt, under 20 = oversolgt.
    """
    # Beregn RSI først hvis den ikke findes
    if "RSI" not in df.columns:
        add_rsi(df, period=rsi_period)

    rsi = df["RSI"]
    rsi_low = rsi.rolling(window=stoch_period).min()
    rsi_high = rsi.rolling(window=stoch_period).max()

    rsi_range = rsi_high - rsi_low
    stoch_rsi = ((rsi - rsi_low) / rsi_range).where(rsi_range > 0, 0.5) * 100

    df["StochRSI_K"] = stoch_rsi.rolling(window=k_smooth).mean()
    df["StochRSI_D"] = df["StochRSI_K"].rolling(window=d_smooth).mean()

    return df


def add_williams_r(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Tilføj Williams %R momentum-oscillator.

    Williams %R = (Highest High - Close) / (Highest High - Lowest Low) × (-100)
    Område: -100 til 0. Under -80 = oversolgt, over -20 = overkøbt.
    """
    highest = df["High"].rolling(window=period).max()
    lowest = df["Low"].rolling(window=period).min()

    hl_range = highest - lowest
    df["Williams_R"] = ((highest - df["Close"]) / hl_range).where(hl_range > 0, 0.5) * (-100)

    return df


def add_mfi(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Tilføj Money Flow Index (MFI) – "volume-weighted RSI".

    Bruger pris OG volumen til at måle købs-/salgspres.
    Område: 0–100. Over 80 = overkøbt, under 20 = oversolgt.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]

    positive_flow = money_flow.where(typical_price.diff() > 0, 0.0)
    negative_flow = money_flow.where(typical_price.diff() < 0, 0.0)

    pos_sum = positive_flow.rolling(window=period).sum()
    neg_sum = negative_flow.rolling(window=period).sum()

    mfr = pos_sum / neg_sum.where(neg_sum > 0, 1.0)
    df["MFI"] = 100 - (100 / (1 + mfr))

    return df


def add_cci(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    """
    Tilføj Commodity Channel Index (CCI).

    CCI = (Typical Price - SMA(TP)) / (0.015 × Mean Deviation)
    Over +100 = stærk optrend, under -100 = stærk nedtrend.
    Bruges til at finde overkøbte/oversolgte niveauer og trendstyrke.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )

    df["CCI"] = (typical_price - sma_tp) / (0.015 * mean_dev)

    return df


def add_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Tilføj Average Directional Index (ADX) – trendstyrke.

    ADX > 25 = stærk trend, ADX < 20 = svag/ingen trend.
    +DI > -DI = bullish trend, +DI < -DI = bearish trend.

    Tilføjer: ADX, Plus_DI, Minus_DI.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # Smooth med Wilder's EMA (alpha = 1/period)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    df["Plus_DI"] = (plus_di_smooth / atr * 100).where(atr > 0, 0)
    df["Minus_DI"] = (minus_di_smooth / atr * 100).where(atr > 0, 0)

    dx = (abs(df["Plus_DI"] - df["Minus_DI"]) /
          (df["Plus_DI"] + df["Minus_DI"]).where(
              (df["Plus_DI"] + df["Minus_DI"]) > 0, 1)) * 100

    df["ADX"] = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return df


# ══════════════════════════════════════════════════════════════
#  VOLATILITETS-INDIKATORER
# ══════════════════════════════════════════════════════════════

def add_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Tilføj Average True Range (ATR) – daglig volatilitet.

    ATR bruges til at sætte stop-loss og position sizing.
    Høj ATR = høj volatilitet, lav ATR = lav volatilitet.
    """
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return df


def add_keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Tilføj Keltner Channels (ATR-baserede bands).

    Bruger EMA som midterlinje og ATR til at beregne båndene.
    Mere stabile end Bollinger Bands da de bruger ATR i stedet for std.
    """
    if f"EMA_{ema_period}" not in df.columns:
        add_ema(df, window=ema_period)

    if "ATR" not in df.columns:
        add_atr(df, period=atr_period)

    ema = df[f"EMA_{ema_period}"]
    atr = df["ATR"]

    df["Keltner_Upper"] = ema + atr_multiplier * atr
    df["Keltner_Middle"] = ema
    df["Keltner_Lower"] = ema - atr_multiplier * atr

    return df


def add_donchian_channels(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    """
    Tilføj Donchian Channels (breakout system).

    - Upper: højeste high over N perioder
    - Lower: laveste low over N perioder
    - Middle: gennemsnit

    Breakout over upper = køb, breakout under lower = salg.
    """
    df["Donchian_Upper"] = df["High"].rolling(window=period).max()
    df["Donchian_Lower"] = df["Low"].rolling(window=period).min()
    df["Donchian_Middle"] = (df["Donchian_Upper"] + df["Donchian_Lower"]) / 2

    return df


def add_historical_volatility(
    df: pd.DataFrame,
    period: int = 20,
    annualize: bool = True,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Tilføj Historical Volatility (HV).

    HV = standardafvigelse af log-returns, annualiseret (×√252).
    Bruges til at sammenligne med Implied Volatility.
    """
    log_returns = np.log(df[column] / df[column].shift(1))
    hv = log_returns.rolling(window=period).std()

    if annualize:
        hv *= np.sqrt(252)

    df[f"HV_{period}"] = hv

    return df


# ══════════════════════════════════════════════════════════════
#  CONVENIENCE: tilføj alle indikatorer
# ══════════════════════════════════════════════════════════════

def add_all_indicators(
    df: pd.DataFrame,
    sma_windows: list[int] | None = None,
    ema_windows: list[int] | None = None,
    rsi_period: int = 14,
    macd_params: tuple[int, int, int] = (12, 26, 9),
    bb_window: int = 20,
    bb_std: float = 2.0,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Tilføj alle tekniske indikatorer til en OHLCV DataFrame."""
    sma_windows = sma_windows or [20, 50, 200]
    ema_windows = ema_windows or [12, 26]

    for w in sma_windows:
        add_sma(df, window=w)
    for w in ema_windows:
        add_ema(df, window=w)

    add_rsi(df, period=rsi_period)
    add_macd(df, fast=macd_params[0], slow=macd_params[1], signal=macd_params[2])
    add_bollinger_bands(df, window=bb_window, num_std=bb_std)
    add_volume_analysis(df, window=vol_window)

    return df


def add_advanced_indicators(
    df: pd.DataFrame,
    fibonacci: bool = True,
    ichimoku: bool = True,
    volume_profile: bool = True,
    momentum: bool = True,
    volatility: bool = True,
) -> pd.DataFrame:
    """
    Tilføj alle avancerede indikatorer til en OHLCV DataFrame.

    Hver kategori kan slås til/fra med boolean parametre.
    """
    if fibonacci:
        add_fibonacci(df)

    if ichimoku:
        add_ichimoku(df)

    if volume_profile:
        add_volume_profile(df)

    if momentum:
        add_stochastic_rsi(df)
        add_williams_r(df)
        add_mfi(df)
        add_cci(df)
        add_adx(df)

    if volatility:
        add_atr(df)
        add_keltner_channels(df)
        add_donchian_channels(df)
        add_historical_volatility(df)

    return df
