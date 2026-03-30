"""
AuditLog – ubrydelig, append-only log af alle systemhandlinger.

Alt logges:
  - Alle handler (køb/sælg) med fuld begrundelse
  - Regime-skift
  - Risiko-overrides og circuit breaker aktiveringer
  - System-fejl og advarsler
  - Config-ændringer

Gem i append-only SQLite (ingen UPDATE/DELETE).
Vigtigt for skat, compliance og eventuel revision.

Brug:
    from src.monitoring.audit_log import AuditLog, AuditEntry, AuditCategory
    log = AuditLog()
    log.record(AuditCategory.TRADE, "BUY AAPL", details={...})
    entries = log.query(category=AuditCategory.TRADE, limit=100)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from loguru import logger


class AuditCategory(Enum):
    """Kategori for audit-logposter."""
    TRADE = "trade"                 # Køb/sælg-handler
    SIGNAL = "signal"               # Strategi-signaler
    REGIME = "regime"               # Regime-skift
    RISK = "risk"                   # Risiko-beslutninger
    CIRCUIT_BREAKER = "circuit_breaker"
    SYSTEM = "system"               # Start/stop, fejl
    CONFIG = "config"               # Konfigurationsændringer
    ALERT = "alert"                 # Advarsler sendt
    DATA = "data"                   # Data-pipeline events


class AuditSeverity(Enum):
    """Alvorlighed af audit-event."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    """Én post i audit-loggen."""
    id: int
    timestamp: str
    category: str
    severity: str
    action: str
    details: dict
    checksum: str                   # SHA-256 for integritet
    session_id: str = ""

    @property
    def category_enum(self) -> AuditCategory:
        try:
            return AuditCategory(self.category)
        except ValueError:
            return AuditCategory.SYSTEM

    @property
    def severity_enum(self) -> AuditSeverity:
        try:
            return AuditSeverity(self.severity)
        except ValueError:
            return AuditSeverity.INFO


@dataclass
class AuditSummary:
    """Sammenfatning af audit-log."""
    total_entries: int
    entries_by_category: dict[str, int]
    entries_by_severity: dict[str, int]
    first_entry: str                # Timestamp
    last_entry: str
    trade_count: int
    error_count: int
    integrity_ok: bool


class AuditLog:
    """
    Append-only audit log med SHA-256 integritetskontrol.

    Alle poster er ubrydelige – ingen UPDATE eller DELETE.
    Hver post linkes til forrige via checksum-kæde.

    Args:
        db_path: Sti til SQLite-database.
        session_id: Unik identifikator for denne kørsel.
    """

    def __init__(
        self,
        db_path: str = "data_cache/audit_log.db",
        session_id: str | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._lock = threading.Lock()
        self._last_checksum = "GENESIS"
        self._init_db()

    def _init_db(self) -> None:
        """Opret tabeller hvis de ikke eksisterer."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    category    TEXT NOT NULL,
                    severity    TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    details     TEXT NOT NULL,
                    checksum    TEXT NOT NULL,
                    session_id  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_category
                ON audit_log (category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_log (timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_severity
                ON audit_log (severity)
            """)

            # Hent sidste checksum for kæden
            row = conn.execute(
                "SELECT checksum FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                self._last_checksum = row[0]

    def _compute_checksum(
        self,
        timestamp: str,
        category: str,
        action: str,
        details_json: str,
        prev_checksum: str,
    ) -> str:
        """Beregn SHA-256 checksum der linker til forrige post."""
        payload = f"{prev_checksum}|{timestamp}|{category}|{action}|{details_json}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    # ── Record ──────────────────────────────────────────────────

    def record(
        self,
        category: AuditCategory,
        action: str,
        details: dict | None = None,
        severity: AuditSeverity = AuditSeverity.INFO,
    ) -> int:
        """
        Tilføj en post til audit-loggen.

        Args:
            category: Type af event.
            action: Kort beskrivelse (f.eks. "BUY AAPL 50 shares @ $175").
            details: Fuld detaljer som dict (JSON-serialiserbar).
            severity: Alvorlighed.

        Returns:
            ID for den nye post.
        """
        now = datetime.now().isoformat()
        details = details or {}
        details_json = json.dumps(details, default=str, ensure_ascii=False)

        with self._lock:
            checksum = self._compute_checksum(
                now, category.value, action, details_json, self._last_checksum,
            )

            with sqlite3.connect(str(self._db_path)) as conn:
                cursor = conn.execute(
                    """INSERT INTO audit_log
                       (timestamp, category, severity, action, details, checksum, session_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (now, category.value, severity.value, action,
                     details_json, checksum, self._session_id),
                )
                entry_id = cursor.lastrowid

            self._last_checksum = checksum

        logger.debug(f"Audit [{category.value}] {action}")
        return entry_id

    # ── Convenience-metoder ─────────────────────────────────────

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reason: str,
        **extra,
    ) -> int:
        """Log en handel."""
        return self.record(
            AuditCategory.TRADE,
            f"{side.upper()} {symbol} {qty:.0f} @ ${price:.2f}",
            details={"symbol": symbol, "side": side, "qty": qty,
                     "price": price, "reason": reason, **extra},
        )

    def log_regime_shift(
        self, from_regime: str, to_regime: str, confidence: float,
    ) -> int:
        """Log et regime-skift."""
        return self.record(
            AuditCategory.REGIME,
            f"Regime: {from_regime} → {to_regime} ({confidence:.0f}%)",
            details={"from": from_regime, "to": to_regime,
                     "confidence": confidence},
            severity=AuditSeverity.WARNING,
        )

    def log_risk_override(self, action: str, details: dict) -> int:
        """Log en risiko-override."""
        return self.record(
            AuditCategory.RISK, action, details,
            severity=AuditSeverity.WARNING,
        )

    def log_circuit_breaker(self, level: str, reason: str) -> int:
        """Log circuit breaker aktivering."""
        return self.record(
            AuditCategory.CIRCUIT_BREAKER,
            f"Circuit Breaker: {level} – {reason}",
            details={"level": level, "reason": reason},
            severity=AuditSeverity.CRITICAL,
        )

    def log_error(self, error: str, details: dict | None = None) -> int:
        """Log en systemfejl."""
        return self.record(
            AuditCategory.SYSTEM, f"ERROR: {error}", details,
            severity=AuditSeverity.ERROR,
        )

    def log_system_event(self, event: str, details: dict | None = None) -> int:
        """Log et system-event (start, stop, osv.)."""
        return self.record(AuditCategory.SYSTEM, event, details)

    # ── Query ───────────────────────────────────────────────────

    def query(
        self,
        category: AuditCategory | None = None,
        severity: AuditSeverity | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """
        Hent audit-poster med filtrering.

        Args:
            category: Filter på kategori.
            severity: Filter på alvorlighed.
            since: Kun poster efter dette tidspunkt (ISO).
            until: Kun poster før dette tidspunkt.
            limit: Max antal poster.
            offset: Spring over N poster.

        Returns:
            Liste af AuditEntry.
        """
        conditions: list[str] = []
        params: list = []

        if category:
            conditions.append("category = ?")
            params.append(category.value)
        if severity:
            conditions.append("severity = ?")
            params.append(severity.value)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                f"SELECT id, timestamp, category, severity, action, "
                f"details, checksum, session_id "
                f"FROM audit_log {where} "
                f"ORDER BY id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()

        return [
            AuditEntry(
                id=r[0], timestamp=r[1], category=r[2], severity=r[3],
                action=r[4], details=json.loads(r[5]),
                checksum=r[6], session_id=r[7],
            )
            for r in rows
        ]

    def get_recent(self, limit: int = 20) -> list[AuditEntry]:
        """Hent seneste poster."""
        return self.query(limit=limit)

    def count(self, category: AuditCategory | None = None) -> int:
        """Tæl poster."""
        with sqlite3.connect(str(self._db_path)) as conn:
            if category:
                row = conn.execute(
                    "SELECT COUNT(*) FROM audit_log WHERE category = ?",
                    (category.value,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()
        return row[0] if row else 0

    # ── Integritetskontrol ──────────────────────────────────────

    def verify_integrity(self, limit: int = 0) -> bool:
        """
        Verificér at checksum-kæden er intakt.

        Genberegner checksums og sammenligner med gemte.
        Returnerer False hvis nogen post er ændret/slettet.

        Args:
            limit: Max antal poster at verificere (0 = alle).
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            query = "SELECT id, timestamp, category, action, details, checksum FROM audit_log ORDER BY id ASC"
            if limit > 0:
                query += f" LIMIT {limit}"
            rows = conn.execute(query).fetchall()

        if not rows:
            return True

        prev_checksum = "GENESIS"
        for row in rows:
            _id, ts, cat, action, details, stored_checksum = row
            expected = self._compute_checksum(ts, cat, action, details, prev_checksum)
            if expected != stored_checksum:
                logger.error(
                    f"Audit-log integritetsfejl ved id={_id}: "
                    f"forventet={expected}, gemt={stored_checksum}"
                )
                return False
            prev_checksum = stored_checksum

        return True

    # ── Sammenfatning ───────────────────────────────────────────

    def summary(self) -> AuditSummary:
        """Generér sammenfatning af loggen."""
        with sqlite3.connect(str(self._db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

            cats = conn.execute(
                "SELECT category, COUNT(*) FROM audit_log GROUP BY category"
            ).fetchall()
            by_category = {r[0]: r[1] for r in cats}

            sevs = conn.execute(
                "SELECT severity, COUNT(*) FROM audit_log GROUP BY severity"
            ).fetchall()
            by_severity = {r[0]: r[1] for r in sevs}

            first = conn.execute(
                "SELECT timestamp FROM audit_log ORDER BY id ASC LIMIT 1"
            ).fetchone()
            last = conn.execute(
                "SELECT timestamp FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()

        return AuditSummary(
            total_entries=total,
            entries_by_category=by_category,
            entries_by_severity=by_severity,
            first_entry=first[0] if first else "",
            last_entry=last[0] if last else "",
            trade_count=by_category.get("trade", 0),
            error_count=by_severity.get("error", 0) + by_severity.get("critical", 0),
            integrity_ok=self.verify_integrity(limit=100),
        )

    # ── Export ──────────────────────────────────────────────────

    def export_csv(self, path: str, category: AuditCategory | None = None) -> int:
        """Eksportér audit-log til CSV (for revisor)."""
        import csv

        entries = self.query(category=category, limit=100_000)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "timestamp", "category", "severity",
                "action", "details", "checksum",
            ])
            for e in reversed(entries):  # Ældste først
                writer.writerow([
                    e.id, e.timestamp, e.category, e.severity,
                    e.action, json.dumps(e.details, default=str), e.checksum,
                ])
        return len(entries)
