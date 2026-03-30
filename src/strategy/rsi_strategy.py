"""
RSI (Relative Strength Index) strategi.

KØB  når RSI falder under oversold-tærsklen (default 30).
SÆLG når RSI stiger over overbought-tærsklen (default 70).

Confidence baseres på:
  - Hvor langt RSI er fra neutralzonen (50)
  - Retning af RSI-bevægelse (faldende RSI i oversold = stærkere)
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.indicators import add_rsi
from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult


class RSIStrategy(BaseStrategy):

    def __init__(
        self,
        period: int | None = None,
        oversold: float | None = None,
        overbought: float | None = None,
    ) -> None:
        self.period = period or settings.strategy.rsi_period
        self.oversold = oversold if oversold is not None else settings.strategy.rsi_oversold
        self.overbought = overbought if overbought is not None else settings.strategy.rsi_overbought

    @property
    def name(self) -> str:
        return f"RSI({self.period}, {self.oversold}/{self.overbought})"

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        min_rows = self.period + 5
        if not self.validate_data(df, min_rows):
            return StrategyResult(Signal.HOLD, 0, "Ikke nok data")

        df = df.copy()

        # Tilføj RSI hvis den mangler
        if "RSI" not in df.columns:
            add_rsi(df, period=self.period)

        rsi_now = df["RSI"].iloc[-1]
        rsi_prev = df["RSI"].iloc[-2]

        if pd.isna(rsi_now):
            return StrategyResult(Signal.HOLD, 0, "RSI endnu ikke beregnet")

        # Oversolgt → KØB
        if rsi_now < self.oversold:
            confidence = self._calc_confidence_buy(rsi_now, rsi_prev)
            return StrategyResult(
                Signal.BUY, confidence,
                f"RSI={rsi_now:.1f} < {self.oversold} (oversolgt)",
            )

        # Overkøbt → SÆLG
        if rsi_now > self.overbought:
            confidence = self._calc_confidence_sell(rsi_now, rsi_prev)
            return StrategyResult(
                Signal.SELL, confidence,
                f"RSI={rsi_now:.1f} > {self.overbought} (overkøbt)",
            )

        # Neutral zone
        return StrategyResult(
            Signal.HOLD, 0,
            f"RSI={rsi_now:.1f} – neutral zone ({self.oversold}–{self.overbought})",
        )

    def _calc_confidence_buy(self, rsi: float, rsi_prev: float) -> float:
        """
        Lavere RSI → højere confidence.
        RSI 30 → ~50, RSI 10 → ~85. Bonus hvis RSI vender opad.
        """
        # Distance fra oversold: 0 (ved grænsen) til 30 (RSI=0)
        distance = max(0, self.oversold - rsi)
        base = 50 + (distance / self.oversold) * 35

        # Bonus for RSI-vending (begynder at stige)
        reversal_bonus = 10 if rsi > rsi_prev else 0

        return min(100, base + reversal_bonus)

    def _calc_confidence_sell(self, rsi: float, rsi_prev: float) -> float:
        """
        Højere RSI → højere confidence.
        RSI 70 → ~50, RSI 90 → ~85. Bonus hvis RSI vender nedad.
        """
        distance = max(0, rsi - self.overbought)
        max_distance = 100 - self.overbought
        base = 50 + (distance / max_distance) * 35

        reversal_bonus = 10 if rsi < rsi_prev else 0

        return min(100, base + reversal_bonus)
