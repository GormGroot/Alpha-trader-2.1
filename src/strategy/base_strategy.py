"""
Abstrakt base-klasse for handelsstrategier.

Alle strategier arver fra BaseStrategy og implementerer:
  - analyze()        → BUY / SELL / HOLD
  - get_confidence() → 0–100
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StrategyResult:
    """Container for resultatet af en strategi-analyse."""

    __slots__ = ("signal", "confidence", "reason")

    def __init__(self, signal: Signal, confidence: float, reason: str = "") -> None:
        self.signal = signal
        self.confidence = max(0.0, min(100.0, confidence))
        self.reason = reason

    def __repr__(self) -> str:
        return f"StrategyResult({self.signal.value}, conf={self.confidence:.0f}, {self.reason!r})"


class BaseStrategy(ABC):
    """
    Abstrakt base-klasse for alle handelsstrategier.

    Subklasser implementerer:
      - name           (property)
      - analyze(df)    → StrategyResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unikt navn for strategien."""

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        """
        Analysér markedsdata og returnér et signal.

        Args:
            df: DataFrame med OHLCV + indikatorer (index = dato).

        Returns:
            StrategyResult med signal, confidence og begrundelse.
        """

    def get_position_size(
        self,
        result: StrategyResult,
        portfolio_value: float,
        max_position_pct: float = 0.05,
    ) -> float:
        """
        Beregn position size baseret på signal og confidence.

        Skalerer lineært: confidence 50 → halvt af max, 100 → fuldt max.
        HOLD → 0.

        Args:
            result: Signal + confidence fra analyze().
            portfolio_value: Samlet porteføljeværdi i USD.
            max_position_pct: Maks andel af portefølje per position.

        Returns:
            Dollar-beløb der skal handles.
        """
        if result.signal == Signal.HOLD:
            return 0.0

        # Skalér lineært med confidence (0–100 → 0–max)
        fraction = result.confidence / 100.0
        return round(portfolio_value * max_position_pct * fraction, 2)

    def validate_data(self, df: pd.DataFrame, min_rows: int) -> bool:
        """Tjek at DataFrame har nok rækker til analyse."""
        return len(df) >= min_rows
