"""
AnomalyDetector – detektér unormale events i trading-systemet.

Detekterer:
  - Usædvanligt store tab/gevinster på én handel
  - Unormalt mange handler på kort tid (burst)
  - Data-anomalier (gaps, duplikater, manglende værdier)
  - Pludselig ændring i strategi-adfærd
  - Uventet korrelationsændring

Brug:
    from src.monitoring.anomaly_detector import AnomalyDetector, Anomaly
    detector = AnomalyDetector()
    detector.check_trade(symbol="AAPL", pnl=-5000, avg_pnl=200)
    alerts = detector.get_active_anomalies()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
from loguru import logger


class AnomalyType(Enum):
    """Type af anomali."""
    LARGE_LOSS = "large_loss"               # Usædvanligt stort tab
    LARGE_WIN = "large_win"                 # Usædvanligt stor gevinst
    TRADE_BURST = "trade_burst"             # For mange handler hurtigt
    DATA_GAP = "data_gap"                   # Manglende data
    DATA_DUPLICATE = "data_duplicate"       # Duplikerede data
    DATA_SPIKE = "data_spike"              # Pludselig prisspring
    STRATEGY_SHIFT = "strategy_shift"       # Ændret adfærd
    CORRELATION_BREAK = "correlation_break" # Uventet korrelation
    VOLUME_ANOMALY = "volume_anomaly"       # Unormal volumen


class AnomalySeverity(Enum):
    """Alvorlighed."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Én detekteret anomali."""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    title: str
    description: str
    details: dict = field(default_factory=dict)
    timestamp: str = ""
    resolved: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def severity_icon(self) -> str:
        return {
            AnomalySeverity.LOW: "🟢",
            AnomalySeverity.MEDIUM: "🟡",
            AnomalySeverity.HIGH: "🟠",
            AnomalySeverity.CRITICAL: "🔴",
        }.get(self.severity, "⚪")


@dataclass
class AnomalyReport:
    """Samlet anomali-rapport."""
    timestamp: str
    total_anomalies: int
    active_anomalies: int
    by_type: dict[str, int]
    by_severity: dict[str, int]
    anomalies: list[Anomaly]


class AnomalyDetector:
    """
    Detektér anomalier i trading-systemet.

    Args:
        loss_threshold_std: Antal std.afv. for at markere en handel som anomali.
        trade_burst_window_minutes: Tidsvindue for burst-detektion.
        trade_burst_max: Max handler i vinduet.
        data_gap_max_minutes: Max tilladte gap i data.
        price_spike_pct: Max tilladt prisændring per bar.
    """

    def __init__(
        self,
        loss_threshold_std: float = 3.0,
        trade_burst_window_minutes: int = 10,
        trade_burst_max: int = 20,
        data_gap_max_minutes: int = 60,
        price_spike_pct: float = 10.0,
    ) -> None:
        self._loss_std = loss_threshold_std
        self._burst_window = timedelta(minutes=trade_burst_window_minutes)
        self._burst_max = trade_burst_max
        self._gap_max = timedelta(minutes=data_gap_max_minutes)
        self._spike_pct = price_spike_pct

        self._anomalies: list[Anomaly] = []
        self._trade_timestamps: list[str] = []
        self._trade_pnls: list[float] = []
        self._strategy_signal_counts: dict[str, list[str]] = {}
        self._max_history = 1000

    # ── Trade-anomalier ─────────────────────────────────────────

    def check_trade(
        self,
        symbol: str,
        pnl: float,
        avg_pnl: float = 0.0,
        std_pnl: float = 0.0,
    ) -> Anomaly | None:
        """
        Tjek om en handel er anomal.

        Args:
            symbol: Aktiesymbol.
            pnl: Profit/loss for handlen.
            avg_pnl: Gennemsnitlig P&L (beregnes automatisk hvis 0).
            std_pnl: Std.afv. af P&L (beregnes automatisk hvis 0).

        Returns:
            Anomaly hvis detekteret, ellers None.
        """
        self._trade_pnls.append(pnl)
        self._trade_timestamps.append(datetime.now().isoformat())
        if len(self._trade_pnls) > self._max_history:
            self._trade_pnls = self._trade_pnls[-self._max_history:]
            self._trade_timestamps = self._trade_timestamps[-self._max_history:]
        if len(self._anomalies) > self._max_history * 2:
            self._anomalies = self._anomalies[-self._max_history:]

        # Beregn stats hvis ikke angivet
        if len(self._trade_pnls) >= 10 and std_pnl == 0:
            arr = np.array(self._trade_pnls)
            avg_pnl = float(arr.mean())
            std_pnl = float(arr.std())

        if std_pnl <= 0:
            return None

        z_score = (pnl - avg_pnl) / std_pnl

        # Stort tab
        if z_score < -self._loss_std:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.LARGE_LOSS,
                severity=(AnomalySeverity.CRITICAL if z_score < -4
                          else AnomalySeverity.HIGH),
                title=f"Usædvanligt stort tab: {symbol}",
                description=(
                    f"Tab ${pnl:,.2f} er {abs(z_score):.1f} std.afv. "
                    f"fra gennemsnittet (${avg_pnl:,.2f})"
                ),
                details={"symbol": symbol, "pnl": pnl, "z_score": z_score,
                         "avg_pnl": avg_pnl, "std_pnl": std_pnl},
            )
            self._anomalies.append(anomaly)
            logger.warning(f"Anomali: {anomaly.title}")
            return anomaly

        # Stor gevinst (også usædvanligt – kan indikere datafejl)
        if z_score > self._loss_std:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.LARGE_WIN,
                severity=AnomalySeverity.LOW,
                title=f"Usædvanligt stor gevinst: {symbol}",
                description=(
                    f"Gevinst ${pnl:,.2f} er {z_score:.1f} std.afv. "
                    f"over gennemsnittet"
                ),
                details={"symbol": symbol, "pnl": pnl, "z_score": z_score},
            )
            self._anomalies.append(anomaly)
            return anomaly

        return None

    def check_trade_burst(self) -> Anomaly | None:
        """Tjek om der er for mange handler på kort tid."""
        now = datetime.now()
        cutoff = now - self._burst_window

        recent = [
            ts for ts in self._trade_timestamps
            if ts >= cutoff.isoformat()
        ]

        if len(recent) > self._burst_max:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.TRADE_BURST,
                severity=AnomalySeverity.HIGH,
                title=f"Handels-burst: {len(recent)} handler i {self._burst_window}",
                description=(
                    f"{len(recent)} handler inden for "
                    f"{self._burst_window.seconds // 60} minutter "
                    f"(max tilladt: {self._burst_max})"
                ),
                details={"count": len(recent), "max": self._burst_max},
            )
            self._anomalies.append(anomaly)
            logger.warning(f"Anomali: {anomaly.title}")
            return anomaly

        return None

    # ── Data-anomalier ──────────────────────────────────────────

    def check_data_quality(
        self,
        symbol: str,
        timestamps: list,
        prices: list[float],
        volumes: list[float] | None = None,
    ) -> list[Anomaly]:
        """
        Tjek datakvalitet for ét symbol.

        Detekterer:
          - Gaps (manglende tidsperioder)
          - Duplikater
          - Prisspikes (>X% ændring per bar)
          - Nul-volumener

        Args:
            symbol: Aktiesymbol.
            timestamps: Liste af timestamps.
            prices: Liste af priser.
            volumes: Liste af volumener (valgfri).
        """
        results: list[Anomaly] = []

        if not timestamps or not prices:
            return results

        if len(timestamps) != len(prices):
            results.append(Anomaly(
                anomaly_type=AnomalyType.DATA_GAP,
                severity=AnomalySeverity.HIGH,
                title=f"Data mismatch: {symbol}",
                description=f"Timestamps ({len(timestamps)}) != priser ({len(prices)})",
                details={"symbol": symbol},
            ))
            self._anomalies.extend(results)
            return results

        # Duplikater
        unique_ts = set(str(t) for t in timestamps)
        if len(unique_ts) < len(timestamps):
            dup_count = len(timestamps) - len(unique_ts)
            anomaly = Anomaly(
                anomaly_type=AnomalyType.DATA_DUPLICATE,
                severity=AnomalySeverity.MEDIUM,
                title=f"Duplikerede data: {symbol}",
                description=f"{dup_count} duplikerede timestamps fundet",
                details={"symbol": symbol, "duplicate_count": dup_count},
            )
            results.append(anomaly)

        # Prisspikes
        for i in range(1, len(prices)):
            if prices[i - 1] <= 0:
                continue
            change_pct = abs(prices[i] - prices[i - 1]) / prices[i - 1] * 100
            if change_pct > self._spike_pct:
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.DATA_SPIKE,
                    severity=(AnomalySeverity.HIGH if change_pct > 20
                              else AnomalySeverity.MEDIUM),
                    title=f"Prisspring: {symbol} ({change_pct:.1f}%)",
                    description=(
                        f"{symbol}: pris ændret {change_pct:.1f}% "
                        f"(${prices[i-1]:.2f} → ${prices[i]:.2f})"
                    ),
                    details={"symbol": symbol, "change_pct": change_pct,
                             "from_price": prices[i-1], "to_price": prices[i],
                             "index": i},
                )
                results.append(anomaly)

        # Volume-anomalier
        if volumes and len(volumes) == len(prices):
            zero_vol = sum(1 for v in volumes if v == 0)
            if zero_vol > len(volumes) * 0.1:
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.VOLUME_ANOMALY,
                    severity=AnomalySeverity.MEDIUM,
                    title=f"Nul-volumen: {symbol}",
                    description=f"{zero_vol}/{len(volumes)} bars med nul volumen",
                    details={"symbol": symbol, "zero_vol_count": zero_vol},
                )
                results.append(anomaly)

        self._anomalies.extend(results)
        return results

    # ── Strategi-anomalier ──────────────────────────────────────

    def check_strategy_shift(
        self,
        strategy: str,
        signal: str,
        expected_buy_rate: float = 0.3,
    ) -> Anomaly | None:
        """
        Tjek om en strategi pludselig opfører sig anderledes.

        Sporer signal-fordeling over tid. Hvis buy-rate afviger
        markant fra forventet, er der muligvis en fejl.
        """
        if strategy not in self._strategy_signal_counts:
            self._strategy_signal_counts[strategy] = []
        self._strategy_signal_counts[strategy].append(signal)

        signals = self._strategy_signal_counts[strategy]
        if len(signals) < 20:
            return None

        # Seneste 20 signaler
        recent = signals[-20:]
        buy_rate = sum(1 for s in recent if s.upper() in ("BUY", "LONG")) / len(recent)

        # Sammenlign med forventet
        deviation = abs(buy_rate - expected_buy_rate)
        if deviation > 0.4:
            anomaly = Anomaly(
                anomaly_type=AnomalyType.STRATEGY_SHIFT,
                severity=AnomalySeverity.HIGH,
                title=f"Strategi-skift: {strategy}",
                description=(
                    f"{strategy} buy-rate: {buy_rate:.0%} "
                    f"(forventet ~{expected_buy_rate:.0%}). "
                    f"Mulig fejl eller ændret markedsforhold."
                ),
                details={"strategy": strategy, "buy_rate": buy_rate,
                         "expected": expected_buy_rate},
            )
            self._anomalies.append(anomaly)
            return anomaly

        return None

    # ── Hent anomalier ──────────────────────────────────────────

    def get_active_anomalies(self) -> list[Anomaly]:
        """Hent alle uløste anomalier."""
        return [a for a in self._anomalies if not a.resolved]

    def get_all_anomalies(self) -> list[Anomaly]:
        """Hent alle anomalier (inkl. løste)."""
        return list(self._anomalies)

    def resolve(self, index: int) -> bool:
        """Markér en anomali som løst."""
        if 0 <= index < len(self._anomalies):
            self._anomalies[index].resolved = True
            return True
        return False

    def resolve_all(self) -> int:
        """Markér alle aktive anomalier som løst."""
        count = 0
        for a in self._anomalies:
            if not a.resolved:
                a.resolved = True
                count += 1
        return count

    # ── Rapport ─────────────────────────────────────────────────

    def report(self) -> AnomalyReport:
        """Generér anomali-rapport."""
        active = self.get_active_anomalies()

        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for a in active:
            t = a.anomaly_type.value
            s = a.severity.value
            by_type[t] = by_type.get(t, 0) + 1
            by_severity[s] = by_severity.get(s, 0) + 1

        return AnomalyReport(
            timestamp=datetime.now().isoformat(),
            total_anomalies=len(self._anomalies),
            active_anomalies=len(active),
            by_type=by_type,
            by_severity=by_severity,
            anomalies=active,
        )

    @property
    def anomaly_count(self) -> int:
        return len(self._anomalies)

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._anomalies if not a.resolved)
