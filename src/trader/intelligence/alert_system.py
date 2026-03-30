"""
Alert System — real-time advarsler og notifikationer.

Alert-typer:
  - PRICE: Pris over/under threshold
  - VOLUME: Usædvanlig volumen
  - SENTIMENT: Pludselig sentiment-skift
  - EVENT: Corporate event (earnings, M&A, etc.)
  - ALPHA_SCORE: Score ændring over threshold
  - CROSS_IMPACT: Nyhed påvirker din portefølje
  - RISK: Stop-loss, drawdown, overexposure

Severity:
  - LOW: Informativ — ingen handling nødvendig
  - MEDIUM: Bør kigges på inden for timer
  - HIGH: Kræver hurtig handling
  - CRITICAL: Kræver ØJEBLIKKELIG handling
"""

from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger


# ── Enums ────────────────────────────────────────────────────

class AlertType(Enum):
    PRICE = "price"
    VOLUME = "volume"
    SENTIMENT = "sentiment"
    EVENT = "event"
    ALPHA_SCORE = "alpha_score"
    CROSS_IMPACT = "cross_impact"
    RISK = "risk"
    SYSTEM = "system"


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class Alert:
    """En enkelt alert."""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    symbol: str
    title: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def age_minutes(self) -> float:
        return (datetime.now() - self.created_at).total_seconds() / 60

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "title": self.title,
            "message": self.message,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "age_minutes": round(self.age_minutes, 1),
        }


@dataclass
class AlertRule:
    """Regel der trigger en alert."""
    name: str
    alert_type: AlertType
    symbol: str                    # "*" for alle symboler
    condition: Callable[..., bool]
    severity: AlertSeverity = AlertSeverity.MEDIUM
    cooldown_minutes: int = 30     # Minimum tid mellem samme alert
    enabled: bool = True
    message_template: str = ""


# ── Alert System ─────────────────────────────────────────────

class AlertSystem:
    """
    Real-time alert system med persistence og cooldown.

    Brug:
        alerts = AlertSystem()

        # Tilføj regler
        alerts.add_price_alert("AAPL", above=200.0, severity="high")
        alerts.add_volume_alert("AAPL", ratio_threshold=3.0)
        alerts.add_alpha_score_alert("*", drop_threshold=15)

        # Check (kald periodisk)
        new_alerts = alerts.check_all(market_data=data)

        # Hent aktive alerts
        active = alerts.get_active()
    """

    def __init__(self, db_path: str = "data_cache/alerts.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        self._rules: list[AlertRule] = []
        self._last_triggered: dict[str, datetime] = {}  # rule_name → last trigger
        self._callbacks: list[Callable[[Alert], None]] = []
        self._alert_counter = 0

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    symbol TEXT,
                    title TEXT NOT NULL,
                    message TEXT,
                    data TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    acknowledged_at TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol)"
            )

    # ── Alert Rules ──────────────────────────────────────────

    def add_price_alert(
        self,
        symbol: str,
        above: float | None = None,
        below: float | None = None,
        severity: str = "medium",
    ) -> None:
        """Tilføj pris-alert."""
        sev = AlertSeverity(severity)

        if above is not None:
            self._rules.append(AlertRule(
                name=f"price_above_{symbol}_{above}",
                alert_type=AlertType.PRICE,
                symbol=symbol,
                condition=lambda data, s=symbol, t=above: (
                    data.get(s, {}).get("close", 0) > t
                ),
                severity=sev,
                message_template=f"{symbol} over {above:.2f}",
            ))

        if below is not None:
            self._rules.append(AlertRule(
                name=f"price_below_{symbol}_{below}",
                alert_type=AlertType.PRICE,
                symbol=symbol,
                condition=lambda data, s=symbol, t=below: (
                    data.get(s, {}).get("close", 0) < t and
                    data.get(s, {}).get("close", 0) > 0
                ),
                severity=sev,
                message_template=f"{symbol} under {below:.2f}",
            ))

    def add_volume_alert(
        self,
        symbol: str,
        ratio_threshold: float = 3.0,
        severity: str = "medium",
    ) -> None:
        """Alert ved usædvanlig volumen."""
        self._rules.append(AlertRule(
            name=f"volume_{symbol}_{ratio_threshold}",
            alert_type=AlertType.VOLUME,
            symbol=symbol,
            condition=lambda data, s=symbol, t=ratio_threshold: (
                data.get(s, {}).get("volume_ratio", 0) > t
            ),
            severity=AlertSeverity(severity),
            message_template=f"{symbol}: volumen over {ratio_threshold}x normalt",
        ))

    def add_alpha_score_alert(
        self,
        symbol: str = "*",
        drop_threshold: float = 15,
        rise_threshold: float = 15,
        severity: str = "high",
    ) -> None:
        """Alert ved stor Alpha Score ændring."""
        self._rules.append(AlertRule(
            name=f"alpha_drop_{symbol}_{drop_threshold}",
            alert_type=AlertType.ALPHA_SCORE,
            symbol=symbol,
            condition=lambda data, s=symbol, dt=drop_threshold: any(
                score.get("change", 0) < -dt
                for sym, score in data.get("alpha_scores", {}).items()
                if s == "*" or sym == s
            ),
            severity=AlertSeverity(severity),
            message_template=f"Alpha Score faldet med >{drop_threshold} point",
            cooldown_minutes=120,
        ))

    def add_sentiment_alert(
        self,
        symbol: str,
        threshold: float = -0.5,
        severity: str = "high",
    ) -> None:
        """Alert ved stærkt negativt sentiment."""
        self._rules.append(AlertRule(
            name=f"sentiment_{symbol}_{threshold}",
            alert_type=AlertType.SENTIMENT,
            symbol=symbol,
            condition=lambda data, s=symbol, t=threshold: (
                data.get("sentiments", {}).get(s, 0) < t
            ),
            severity=AlertSeverity(severity),
            message_template=f"{symbol}: stærkt negativt sentiment",
        ))

    # ── Alert Triggers ───────────────────────────────────────

    def trigger(
        self,
        alert_type: AlertType,
        symbol: str,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        data: dict | None = None,
        ttl_hours: float = 24,
    ) -> Alert:
        """Trigger en alert manuelt."""
        self._alert_counter += 1
        alert_id = f"ALT-{self._alert_counter:06d}"

        expires = datetime.now() + timedelta(hours=ttl_hours) if ttl_hours > 0 else None

        alert = Alert(
            id=alert_id,
            alert_type=alert_type,
            severity=severity,
            symbol=symbol,
            title=title,
            message=message,
            data=data or {},
            expires_at=expires,
        )

        # Persist
        self._save_alert(alert)

        # Callbacks
        for cb in self._callbacks:
            try:
                cb(alert)
            except Exception as exc:
                logger.warning(f"[alerts] Callback fejl: {exc}")

        logger.info(
            f"[alerts] {severity.value.upper()}: [{alert_type.value}] "
            f"{symbol} — {title}"
        )

        return alert

    def check_all(self, market_data: dict[str, Any]) -> list[Alert]:
        """
        Kør alle regler mod aktuelle data.

        Args:
            market_data: Dict med markedsdata per symbol.

        Returns:
            Liste af nye alerts der blev triggered.
        """
        new_alerts: list[Alert] = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            # Cooldown check
            last = self._last_triggered.get(rule.name)
            if last:
                elapsed = (datetime.now() - last).total_seconds() / 60
                if elapsed < rule.cooldown_minutes:
                    continue

            # Evaluer condition
            try:
                if rule.condition(market_data):
                    alert = self.trigger(
                        alert_type=rule.alert_type,
                        symbol=rule.symbol,
                        title=rule.message_template or rule.name,
                        message=rule.message_template,
                        severity=rule.severity,
                    )
                    new_alerts.append(alert)
                    self._last_triggered[rule.name] = datetime.now()
            except Exception as exc:
                logger.debug(f"[alerts] Rule '{rule.name}' fejl: {exc}")

        return new_alerts

    # ── Alert Management ─────────────────────────────────────

    def get_active(self, symbol: str | None = None) -> list[Alert]:
        """Hent aktive alerts."""
        with sqlite3.connect(self._db_path) as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE status = 'active' AND symbol = ? "
                    "ORDER BY created_at DESC",
                    (symbol,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE status = 'active' "
                    "ORDER BY created_at DESC"
                ).fetchall()

        alerts = [self._row_to_alert(r) for r in rows]

        # Filtrer expired
        active = []
        for a in alerts:
            if a.is_expired:
                self._update_status(a.id, AlertStatus.EXPIRED)
            else:
                active.append(a)

        return active

    def acknowledge(self, alert_id: str) -> None:
        """Markér alert som set."""
        self._update_status(alert_id, AlertStatus.ACKNOWLEDGED)

    def resolve(self, alert_id: str) -> None:
        """Markér alert som løst."""
        self._update_status(alert_id, AlertStatus.RESOLVED)

    def on_alert(self, callback: Callable[[Alert], None]) -> None:
        """Registrér callback for nye alerts."""
        self._callbacks.append(callback)

    def get_summary(self) -> dict[str, int]:
        """Oversigt over alerts per severity."""
        active = self.get_active()
        summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "total": len(active)}
        for a in active:
            summary[a.severity.value] = summary.get(a.severity.value, 0) + 1
        return summary

    # ── Persistence ──────────────────────────────────────────

    def _save_alert(self, alert: Alert) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO alerts
                   (id, alert_type, severity, symbol, title, message, data,
                    status, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (alert.id, alert.alert_type.value, alert.severity.value,
                 alert.symbol, alert.title, alert.message,
                 json.dumps(alert.data, default=str),
                 alert.status.value, alert.created_at.isoformat(),
                 alert.expires_at.isoformat() if alert.expires_at else None),
            )

    def _update_status(self, alert_id: str, status: AlertStatus) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE alerts SET status = ? WHERE id = ?",
                (status.value, alert_id),
            )

    def _row_to_alert(self, row: tuple) -> Alert:
        return Alert(
            id=row[0],
            alert_type=AlertType(row[1]),
            severity=AlertSeverity(row[2]),
            symbol=row[3] or "",
            title=row[4],
            message=row[5] or "",
            data=json.loads(row[6]) if row[6] else {},
            status=AlertStatus(row[7]),
            created_at=datetime.fromisoformat(row[8]),
            expires_at=datetime.fromisoformat(row[9]) if row[9] else None,
        )
