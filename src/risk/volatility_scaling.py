"""
VolatilityScaler – volatilitetsbaseret positionsstoerrelsesberegning.

Features:
  - ATR-baseret position sizing (hoej vol = mindre position)
  - Konstant risiko per handel (dollar-risk = fast)
  - Risk parity: hver position bidrager lige meget risiko
  - Volatilitets-targeting: juster samlet eksponering til target-vol
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class PositionSize:
    """Beregnet positionsstorrelse."""
    symbol: str
    shares: int
    dollar_amount: float
    weight_pct: float              # Andel af portefoelje
    atr: float                     # Average True Range
    volatility_pct: float          # Annualiseret volatilitet
    risk_per_share: float          # Dollar-risiko per aktie
    method: str                    # "atr", "risk_parity", "vol_target"


@dataclass
class RiskParityAllocation:
    """Risk parity allokering for hele portefoeljen."""
    allocations: dict[str, float]       # symbol -> vaegt (0-1)
    risk_contributions: dict[str, float] # symbol -> risiko-bidrag
    total_volatility: float
    target_volatility: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class VolatilityScaler:
    """
    Beregn optimale positionsstoerrelser baseret paa volatilitet.

    Princip: Hoej volatilitet = mindre position, saa risiko per
    handel holdes konstant uanset aktieus volatilitet.

    Brug:
        scaler = VolatilityScaler(equity=100_000)
        size = scaler.calculate_position_size(df, "AAPL", price=175.0)
        allocation = scaler.risk_parity({sym: df for ...})
    """

    def __init__(
        self,
        equity: float = 100_000,
        risk_per_trade_pct: float = 0.01,    # 1% risiko per handel
        atr_period: int = 14,
        atr_multiplier: float = 2.0,         # Stop-loss = 2x ATR
        target_volatility: float = 0.15,      # 15% aarlig vol-target
        max_position_pct: float = 0.10,       # Max 10% per position
        min_position_pct: float = 0.005,      # Min 0.5% per position
    ) -> None:
        self._equity = equity
        self._risk_pct = risk_per_trade_pct
        self._atr_period = atr_period
        self._atr_mult = atr_multiplier
        self._target_vol = target_volatility
        self._max_pos_pct = max_position_pct
        self._min_pos_pct = min_position_pct

    @property
    def equity(self) -> float:
        return self._equity

    @equity.setter
    def equity(self, value: float) -> None:
        self._equity = max(0, value)

    # ── ATR-baseret Position Sizing ──────────────────────────

    def calculate_atr(self, df: pd.DataFrame, period: int | None = None) -> float:
        """
        Beregn Average True Range (ATR).

        ATR = gennemsnit af True Range over N perioder.
        True Range = max(H-L, |H-Cprev|, |L-Cprev|)
        """
        if df is None or df.empty or len(df) < 2:
            return 0.0

        p = period or self._atr_period

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = float(tr.rolling(min(p, len(tr))).mean().iloc[-1])
        return atr if not np.isnan(atr) else 0.0

    def calculate_position_size(
        self,
        df: pd.DataFrame,
        symbol: str,
        price: float,
    ) -> PositionSize:
        """
        Beregn optimal positionsstorrelse baseret paa ATR.

        Logik:
          1. Beregn ATR (volatilitetsmaal)
          2. Risk per aktie = ATR * multiplier (= stop-loss afstand)
          3. Dollar-risk = equity * risk_per_trade_pct
          4. Antal aktier = dollar_risk / risk_per_aktie
          5. Cap til max_position_pct af portefoelje

        Args:
            df: OHLCV DataFrame.
            symbol: Aktiesymbol.
            price: Nuvaerende pris.

        Returns:
            PositionSize med antal aktier og beloeb.
        """
        atr = self.calculate_atr(df)

        if atr <= 0 or price <= 0:
            return PositionSize(
                symbol=symbol, shares=0, dollar_amount=0,
                weight_pct=0, atr=0, volatility_pct=0,
                risk_per_share=0, method="atr",
            )

        risk_per_share = atr * self._atr_mult
        dollar_risk = self._equity * self._risk_pct
        shares = int(dollar_risk / risk_per_share)

        # Cap
        max_shares = int(self._equity * self._max_pos_pct / price)
        min_shares = max(1, int(self._equity * self._min_pos_pct / price))
        shares = max(min_shares, min(shares, max_shares))

        dollar_amount = shares * price
        weight_pct = dollar_amount / self._equity if self._equity > 0 else 0

        # Annualiseret vol
        returns = df["Close"].pct_change().dropna()
        vol_pct = float(returns.std() * np.sqrt(252)) if len(returns) > 5 else 0.0

        return PositionSize(
            symbol=symbol,
            shares=shares,
            dollar_amount=dollar_amount,
            weight_pct=weight_pct,
            atr=atr,
            volatility_pct=vol_pct,
            risk_per_share=risk_per_share,
            method="atr",
        )

    # ── Volatilitets-skaleret Position ───────────────────────

    def volatility_adjusted_weight(
        self, df: pd.DataFrame, vol_window: int = 60,
    ) -> float:
        """
        Beregn vaegt baseret paa inverse volatilitet.

        Lav vol -> hoejere vaegt, hoej vol -> lavere vaegt.
        Normaliseret saa target_volatility opnaas.

        Returns:
            Vaegt (0.0 til 1.0+) – kan vaere > 1.0 ved meget lav vol.
        """
        if df is None or df.empty or len(df) < vol_window:
            return 0.0

        returns = df["Close"].pct_change().dropna()
        if len(returns) < 10:
            return 0.0

        recent_vol = float(returns.iloc[-vol_window:].std()) * np.sqrt(252)
        if recent_vol <= 0:
            return 0.0

        # Weight = target_vol / realized_vol
        weight = self._target_vol / recent_vol
        # Cap til rimelig range
        return max(0.1, min(2.0, weight))

    # ── Risk Parity ──────────────────────────────────────────

    def risk_parity(
        self,
        price_data: dict[str, pd.DataFrame],
        vol_window: int = 60,
    ) -> RiskParityAllocation:
        """
        Beregn risk parity allokering: hver position bidrager lige meget risiko.

        Princip: Vaegt_i = (1/vol_i) / sum(1/vol_j for alle j)

        Args:
            price_data: Dict af symbol -> OHLCV DataFrame.
            vol_window: Vindue til volatilitetsberegning.

        Returns:
            RiskParityAllocation med vaegter og risiko-bidrag.
        """
        vols: dict[str, float] = {}

        for symbol, df in price_data.items():
            if df is None or df.empty or len(df) < vol_window:
                continue
            returns = df["Close"].pct_change().dropna()
            if len(returns) < 10:
                continue
            vol = float(returns.iloc[-vol_window:].std()) * np.sqrt(252)
            if vol > 0:
                vols[symbol] = vol

        if not vols:
            return RiskParityAllocation(
                allocations={}, risk_contributions={},
                total_volatility=0, target_volatility=self._target_vol,
            )

        # Inverse volatilitet
        inv_vols = {s: 1.0 / v for s, v in vols.items()}
        total_inv = sum(inv_vols.values())

        # Normalisér vaegter
        allocations = {s: iv / total_inv for s, iv in inv_vols.items()}

        # Cap individuelle vaegter
        for s in allocations:
            allocations[s] = min(allocations[s], self._max_pos_pct * 5)

        # Re-normalisér
        alloc_sum = sum(allocations.values())
        if alloc_sum > 0:
            allocations = {s: w / alloc_sum for s, w in allocations.items()}

        # Risiko-bidrag
        risk_contributions = {s: allocations[s] * vols[s] for s in allocations}

        # Portfolio-volatilitet (simpel approx – ignorerer korrelation)
        total_vol = sum(
            allocations[s] * vols[s] for s in allocations
        )

        return RiskParityAllocation(
            allocations=allocations,
            risk_contributions=risk_contributions,
            total_volatility=total_vol,
            target_volatility=self._target_vol,
        )

    # ── Vol-targeting ────────────────────────────────────────

    def vol_target_leverage(
        self,
        df: pd.DataFrame,
        vol_window: int = 60,
    ) -> float:
        """
        Beregn leverage-faktor for at naa target-volatilitet.

        Hvis realiseret vol = 10% og target = 15%, leverage = 1.5x.
        Hvis realiseret vol = 30% og target = 15%, leverage = 0.5x.

        Returns:
            Leverage-faktor (capped til 0.1-2.0).
        """
        if df is None or df.empty:
            return 1.0

        returns = df["Close"].pct_change().dropna()
        if len(returns) < vol_window:
            return 1.0

        vol = float(returns.iloc[-vol_window:].std()) * np.sqrt(252)
        if vol <= 0:
            return 1.0

        leverage = self._target_vol / vol
        return max(0.1, min(2.0, leverage))

    # ── Batch Position Sizing ────────────────────────────────

    def size_all_positions(
        self,
        price_data: dict[str, pd.DataFrame],
        prices: dict[str, float],
    ) -> dict[str, PositionSize]:
        """
        Beregn positionsstoerrelser for alle symboler.

        Args:
            price_data: Dict af symbol -> OHLCV DataFrame.
            prices: Dict af symbol -> nuvaerende pris.

        Returns:
            Dict af symbol -> PositionSize.
        """
        results = {}
        for symbol, df in price_data.items():
            price = prices.get(symbol, 0)
            if price > 0:
                results[symbol] = self.calculate_position_size(df, symbol, price)
        return results
