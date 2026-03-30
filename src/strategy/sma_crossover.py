"""
SMA Crossover strategi.

KØB  når kort SMA krydser over lang SMA  (golden cross).
SÆLG når kort SMA krydser under lang SMA (death cross).

Confidence baseres på:
  - Afstanden mellem de to SMA'er (jo større gap, jo stærkere signal)
  - Volumen-bekræftelse (høj volumen ved kryds = stærkere)
"""

from __future__ import annotations

import pandas as pd

from config.settings import settings
from src.data.indicators import add_sma, add_volume_analysis
from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult


class SMACrossoverStrategy(BaseStrategy):

    def __init__(
        self,
        short_window: int | None = None,
        long_window: int | None = None,
    ) -> None:
        self.short_window = short_window or settings.strategy.sma_short_window
        self.long_window = long_window or settings.strategy.sma_long_window

    @property
    def name(self) -> str:
        return f"SMA_Crossover({self.short_window}/{self.long_window})"

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        if not self.validate_data(df, self.long_window + 2):
            return StrategyResult(Signal.HOLD, 0, "Ikke nok data")

        df = df.copy()

        # Tilføj SMA-kolonner hvis de mangler
        short_col = f"SMA_{self.short_window}"
        long_col = f"SMA_{self.long_window}"
        if short_col not in df.columns:
            add_sma(df, self.short_window)
        if long_col not in df.columns:
            add_sma(df, self.long_window)

        # Tilføj volumen-ratio hvis den mangler
        if "Volume_Ratio" not in df.columns:
            add_volume_analysis(df)

        # Brug de to seneste rækker til at detektere kryds
        current = df.iloc[-1]
        previous = df.iloc[-2]

        sma_short_now = current[short_col]
        sma_long_now = current[long_col]
        sma_short_prev = previous[short_col]
        sma_long_prev = previous[long_col]

        # Tjek for NaN
        if pd.isna(sma_short_now) or pd.isna(sma_long_now):
            return StrategyResult(Signal.HOLD, 0, "SMA endnu ikke beregnet")

        # Beregn afstanden som pct af pris
        price = current["Close"]
        gap_pct = abs(sma_short_now - sma_long_now) / price * 100

        # Volumen-bekræftelse
        vol_ratio = current.get("Volume_Ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0

        # Golden cross: kort SMA krydser op over lang SMA
        if sma_short_prev <= sma_long_prev and sma_short_now > sma_long_now:
            confidence = self._calc_confidence(gap_pct, vol_ratio)
            return StrategyResult(
                Signal.BUY, confidence,
                f"Golden cross: SMA{self.short_window} krydser over SMA{self.long_window} "
                f"(gap={gap_pct:.2f}%, vol={vol_ratio:.1f}x)",
            )

        # Death cross: kort SMA krydser ned under lang SMA
        if sma_short_prev >= sma_long_prev and sma_short_now < sma_long_now:
            confidence = self._calc_confidence(gap_pct, vol_ratio)
            return StrategyResult(
                Signal.SELL, confidence,
                f"Death cross: SMA{self.short_window} krydser under SMA{self.long_window} "
                f"(gap={gap_pct:.2f}%, vol={vol_ratio:.1f}x)",
            )

        # Ingen krydsning – HOLD men rapportér trend
        if sma_short_now > sma_long_now:
            return StrategyResult(Signal.HOLD, 0, "Opadgående trend – intet nyt kryds")
        else:
            return StrategyResult(Signal.HOLD, 0, "Nedadgående trend – intet nyt kryds")

    def _calc_confidence(self, gap_pct: float, vol_ratio: float) -> float:
        """
        Confidence = base fra gap-størrelse + bonus for høj volumen.
        Gap 0% → 40, gap 2%+ → 80. Volumen-bonus op til +20.
        """
        # Gap-komponent: 40–80
        gap_score = min(80, 40 + gap_pct * 20)

        # Volumen-bonus: 0–20
        vol_bonus = min(20, max(0, (vol_ratio - 1.0) * 20))

        return min(100, gap_score + vol_bonus)
