"""
HealthMonitor – overvåg at alle dele af systemet kører korrekt.

Tjekker hvert minut:
  - Data-pipeline: leverer friske data?
  - Strategier: producerer signaler?
  - Broker: forbindelse aktiv?
  - Database: tilgængelig?
  - Risiko: circuit breakers aktive?

Hvis noget fejler → STOP handler + send alert.

Brug:
    from src.monitoring.health_monitor import HealthMonitor, ComponentStatus
    monitor = HealthMonitor()
    health = monitor.check_all()
    print(health.overall_status)
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from loguru import logger


class HealthStatus(Enum):
    """Sundhedsstatus for en komponent."""
    HEALTHY = "healthy"         # Grøn – alt ok
    DEGRADED = "degraded"       # Gul – virker men langsomt/delvist
    UNHEALTHY = "unhealthy"     # Rød – fejl
    UNKNOWN = "unknown"         # Grå – ikke tjekket endnu


@dataclass
class ComponentStatus:
    """Status for én systemkomponent."""
    name: str
    status: HealthStatus
    message: str
    last_check: str
    response_time_ms: float = 0.0
    details: dict = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    @property
    def status_icon(self) -> str:
        return {
            HealthStatus.HEALTHY: "🟢",
            HealthStatus.DEGRADED: "🟡",
            HealthStatus.UNHEALTHY: "🔴",
            HealthStatus.UNKNOWN: "⚪",
        }.get(self.status, "⚪")


@dataclass
class SystemHealth:
    """Samlet systemsundhed."""
    timestamp: str
    components: list[ComponentStatus]
    uptime_seconds: float
    total_checks: int
    failed_checks: int

    @property
    def overall_status(self) -> HealthStatus:
        """Samlet status = den dårligste komponents status."""
        if not self.components:
            return HealthStatus.UNKNOWN
        statuses = [c.status for c in self.components]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.components if c.status == HealthStatus.HEALTHY)

    @property
    def unhealthy_components(self) -> list[ComponentStatus]:
        return [c for c in self.components if c.status == HealthStatus.UNHEALTHY]

    def summary_table(self) -> str:
        lines = [
            f"{'═' * 65}",
            f"  SYSTEM HEALTH – {self.timestamp}",
            f"  Status: {self.overall_status.value.upper()} | "
            f"Uptime: {self.uptime_seconds / 3600:.1f}t",
            f"{'═' * 65}",
        ]
        for c in self.components:
            lines.append(
                f"  {c.status_icon} {c.name:<25} {c.status.value:<12} "
                f"{c.response_time_ms:>6.0f}ms  {c.message}"
            )
        lines.append(f"{'═' * 65}")
        return "\n".join(lines)


@dataclass
class HealthEvent:
    """Registreret health-event (for historik)."""
    timestamp: str
    component: str
    status: str
    message: str


class HealthMonitor:
    """
    Overvåg systemets sundhed.

    Kører health checks og vedligeholder historik.

    Args:
        data_freshness_minutes: Max alder for data (default 15 min).
        signal_freshness_minutes: Max alder for signaler (default 30 min).
        db_path: Sti til health-historik database.
    """

    def __init__(
        self,
        data_freshness_minutes: int = 15,
        signal_freshness_minutes: int = 30,
        db_path: str = "data_cache/health.db",
    ) -> None:
        self._data_freshness = timedelta(minutes=data_freshness_minutes)
        self._signal_freshness = timedelta(minutes=signal_freshness_minutes)
        self._start_time = time.time()
        self._check_count = 0
        self._fail_count = 0
        self._history: list[HealthEvent] = []
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Externe afhængigheder (injicer via set_*)
        self._data_pipeline = None
        self._signal_engine = None
        self._broker = None
        self._portfolio = None
        self._risk_manager = None

        # Seneste data-timestamps (sættes udefra)
        self._last_data_time: datetime | None = None
        self._last_signal_time: datetime | None = None

    # ── Dependency injection ────────────────────────────────────

    def set_data_pipeline(self, pipeline) -> None:
        self._data_pipeline = pipeline

    def set_signal_engine(self, engine) -> None:
        self._signal_engine = engine

    def set_broker(self, broker) -> None:
        self._broker = broker

    def set_portfolio(self, portfolio) -> None:
        self._portfolio = portfolio

    def set_risk_manager(self, risk_manager) -> None:
        self._risk_manager = risk_manager

    def report_data_received(self) -> None:
        """Kald dette når data modtages fra pipeline."""
        self._last_data_time = datetime.now()

    def report_signal_generated(self) -> None:
        """Kald dette når signaler genereres."""
        self._last_signal_time = datetime.now()

    # ── Individuelle checks ─────────────────────────────────────

    def check_data_pipeline(self) -> ComponentStatus:
        """Tjek om data-pipeline leverer friske data."""
        start = time.time()
        try:
            if self._data_pipeline is None:
                return ComponentStatus(
                    name="Data Pipeline",
                    status=HealthStatus.UNKNOWN,
                    message="Ikke konfigureret",
                    last_check=datetime.now().isoformat(),
                )

            # Tjek om der er modtaget data for nylig
            if self._last_data_time:
                age = datetime.now() - self._last_data_time
                elapsed = (time.time() - start) * 1000
                if age <= self._data_freshness:
                    return ComponentStatus(
                        name="Data Pipeline",
                        status=HealthStatus.HEALTHY,
                        message=f"Frisk data ({age.seconds}s siden)",
                        last_check=datetime.now().isoformat(),
                        response_time_ms=elapsed,
                    )
                elif age <= self._data_freshness * 3:
                    return ComponentStatus(
                        name="Data Pipeline",
                        status=HealthStatus.DEGRADED,
                        message=f"Gammel data ({age.seconds}s)",
                        last_check=datetime.now().isoformat(),
                        response_time_ms=elapsed,
                    )
                else:
                    return ComponentStatus(
                        name="Data Pipeline",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Ingen data i {age.seconds}s!",
                        last_check=datetime.now().isoformat(),
                        response_time_ms=elapsed,
                    )

            # Prøv at hente data
            try:
                data = self._data_pipeline.get_all_latest()
                elapsed = (time.time() - start) * 1000
                if data and len(data) > 0:
                    self._last_data_time = datetime.now()
                    return ComponentStatus(
                        name="Data Pipeline",
                        status=HealthStatus.HEALTHY,
                        message=f"{len(data)} symboler aktive",
                        last_check=datetime.now().isoformat(),
                        response_time_ms=elapsed,
                    )
                return ComponentStatus(
                    name="Data Pipeline",
                    status=HealthStatus.DEGRADED,
                    message="Ingen data i cache",
                    last_check=datetime.now().isoformat(),
                    response_time_ms=elapsed,
                )
            except Exception as exc:
                elapsed = (time.time() - start) * 1000
                return ComponentStatus(
                    name="Data Pipeline",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Fejl: {exc}",
                    last_check=datetime.now().isoformat(),
                    response_time_ms=elapsed,
                )

        except Exception as exc:
            return ComponentStatus(
                name="Data Pipeline",
                status=HealthStatus.UNHEALTHY,
                message=f"Check fejlede: {exc}",
                last_check=datetime.now().isoformat(),
            )

    def check_signal_engine(self) -> ComponentStatus:
        """Tjek om strategi-motoren producerer signaler."""
        start = time.time()
        if self._signal_engine is None:
            return ComponentStatus(
                name="Signal Engine",
                status=HealthStatus.UNKNOWN,
                message="Ikke konfigureret",
                last_check=datetime.now().isoformat(),
            )

        if self._last_signal_time:
            age = datetime.now() - self._last_signal_time
            elapsed = (time.time() - start) * 1000
            if age <= self._signal_freshness:
                return ComponentStatus(
                    name="Signal Engine",
                    status=HealthStatus.HEALTHY,
                    message=f"Signal {age.seconds}s siden",
                    last_check=datetime.now().isoformat(),
                    response_time_ms=elapsed,
                )
            elif age <= self._signal_freshness * 3:
                return ComponentStatus(
                    name="Signal Engine",
                    status=HealthStatus.DEGRADED,
                    message=f"Gammel signal ({age.seconds}s)",
                    last_check=datetime.now().isoformat(),
                    response_time_ms=elapsed,
                )

        elapsed = (time.time() - start) * 1000
        return ComponentStatus(
            name="Signal Engine",
            status=HealthStatus.DEGRADED,
            message="Ingen signaler endnu",
            last_check=datetime.now().isoformat(),
            response_time_ms=elapsed,
        )

    def check_broker(self) -> ComponentStatus:
        """Tjek broker-forbindelse."""
        start = time.time()
        if self._broker is None:
            return ComponentStatus(
                name="Broker",
                status=HealthStatus.UNKNOWN,
                message="Ikke konfigureret",
                last_check=datetime.now().isoformat(),
            )

        try:
            account = self._broker.get_account()
            elapsed = (time.time() - start) * 1000
            return ComponentStatus(
                name="Broker",
                status=HealthStatus.HEALTHY,
                message=f"{self._broker.name}: ${account.equity:,.0f}",
                last_check=datetime.now().isoformat(),
                response_time_ms=elapsed,
                details={"equity": account.equity, "cash": account.cash},
            )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return ComponentStatus(
                name="Broker",
                status=HealthStatus.UNHEALTHY,
                message=f"Fejl: {exc}",
                last_check=datetime.now().isoformat(),
                response_time_ms=elapsed,
            )

    def check_database(self) -> ComponentStatus:
        """Tjek database-tilgængelighed."""
        start = time.time()
        try:
            db_path = self._db_path.parent / "indicators.db"
            if not db_path.exists():
                # Prøv at oprette en test-forbindelse
                db_path = self._db_path

            with sqlite3.connect(str(db_path), timeout=5) as conn:
                conn.execute("SELECT 1")
                elapsed = (time.time() - start) * 1000

                if elapsed > 500:
                    return ComponentStatus(
                        name="Database",
                        status=HealthStatus.DEGRADED,
                        message=f"Langsom ({elapsed:.0f}ms)",
                        last_check=datetime.now().isoformat(),
                        response_time_ms=elapsed,
                    )
                return ComponentStatus(
                    name="Database",
                    status=HealthStatus.HEALTHY,
                    message=f"OK ({elapsed:.0f}ms)",
                    last_check=datetime.now().isoformat(),
                    response_time_ms=elapsed,
                )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return ComponentStatus(
                name="Database",
                status=HealthStatus.UNHEALTHY,
                message=f"Fejl: {exc}",
                last_check=datetime.now().isoformat(),
                response_time_ms=elapsed,
            )

    def check_risk_manager(self) -> ComponentStatus:
        """Tjek risikostyring og circuit breakers."""
        start = time.time()
        if self._risk_manager is None:
            return ComponentStatus(
                name="Risk Manager",
                status=HealthStatus.UNKNOWN,
                message="Ikke konfigureret",
                last_check=datetime.now().isoformat(),
            )

        try:
            elapsed = (time.time() - start) * 1000
            if self._risk_manager.is_trading_halted:
                return ComponentStatus(
                    name="Risk Manager",
                    status=HealthStatus.DEGRADED,
                    message="Trading stoppet!",
                    last_check=datetime.now().isoformat(),
                    response_time_ms=elapsed,
                    details={"trading_halted": True},
                )
            return ComponentStatus(
                name="Risk Manager",
                status=HealthStatus.HEALTHY,
                message="Aktiv, ingen restriktioner",
                last_check=datetime.now().isoformat(),
                response_time_ms=elapsed,
                details={"trading_halted": False},
            )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return ComponentStatus(
                name="Risk Manager",
                status=HealthStatus.UNHEALTHY,
                message=f"Fejl: {exc}",
                last_check=datetime.now().isoformat(),
                response_time_ms=elapsed,
            )

    def check_portfolio(self) -> ComponentStatus:
        """Tjek portfolio tracker."""
        start = time.time()
        if self._portfolio is None:
            return ComponentStatus(
                name="Portfolio",
                status=HealthStatus.UNKNOWN,
                message="Ikke konfigureret",
                last_check=datetime.now().isoformat(),
            )

        try:
            summary = self._portfolio.summary()
            elapsed = (time.time() - start) * 1000
            return ComponentStatus(
                name="Portfolio",
                status=HealthStatus.HEALTHY,
                message=f"${summary['total_equity']:,.0f}, "
                        f"{summary['positions']} positioner",
                last_check=datetime.now().isoformat(),
                response_time_ms=elapsed,
                details=summary,
            )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return ComponentStatus(
                name="Portfolio",
                status=HealthStatus.UNHEALTHY,
                message=f"Fejl: {exc}",
                last_check=datetime.now().isoformat(),
                response_time_ms=elapsed,
            )

    # ── Samlet check ────────────────────────────────────────────

    def check_all(self) -> SystemHealth:
        """Kør alle health checks og returnér samlet status."""
        self._check_count += 1

        components = [
            self.check_data_pipeline(),
            self.check_signal_engine(),
            self.check_broker(),
            self.check_database(),
            self.check_risk_manager(),
            self.check_portfolio(),
        ]

        failed = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
        self._fail_count += 1 if failed > 0 else 0

        health = SystemHealth(
            timestamp=datetime.now().isoformat(),
            components=components,
            uptime_seconds=time.time() - self._start_time,
            total_checks=self._check_count,
            failed_checks=self._fail_count,
        )

        # Log til historik
        for c in components:
            self._history.append(HealthEvent(
                timestamp=c.last_check,
                component=c.name,
                status=c.status.value,
                message=c.message,
            ))

        # Trim historik
        if len(self._history) > 10000:
            self._history = self._history[-5000:]

        if health.overall_status == HealthStatus.UNHEALTHY:
            logger.warning(
                f"System UNHEALTHY: {', '.join(c.name for c in health.unhealthy_components)}"
            )

        return health

    # ── Historik ────────────────────────────────────────────────

    def get_history(
        self,
        component: str | None = None,
        limit: int = 100,
    ) -> list[HealthEvent]:
        """Hent health-historik."""
        hist = self._history
        if component:
            hist = [h for h in hist if h.component == component]
        return list(reversed(hist[-limit:]))

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    @property
    def check_count(self) -> int:
        return self._check_count

    @property
    def fail_count(self) -> int:
        return self._fail_count
