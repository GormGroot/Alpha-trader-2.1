"""
Dividend Tracker — udbytteregistrering og kildeskat-beregning.

Features:
  - Log alle udbytter: dato, symbol, brutto, kildeskat, netto
  - Udenlandsk kildeskat-kredit beregning (DBO-aftaler)
  - Reclaimable withholding tax tracking
  - Årsrapport til revisor

DBO-satser (Double Taxation Agreement):
  - USA: 15% (DTAA DK-US)
  - Tyskland: 26.375% (inkl. Solidaritätszuschlag)
  - UK: 0%
  - Sverige: 15% (nordisk DBO)
  - Norge: 15%
  - Finland: 15%
  - Frankrig: 12.8%
  - Holland: 15%
  - Schweiz: 15% (35% brutto, 20% refunderes)

For selskaber: Udbytter beskattes med 22% selskabsskat.
Kildeskat betalt i udlandet krediteres mod dansk skat (max DBO-sats).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# ── DBO Rates ───────────────────────────────────────────────

# Land → max kildeskat-sats under DBO med Danmark
DBO_RATES: dict[str, float] = {
    "US": 0.15,
    "DE": 0.15,       # DBO sats (reelt tilbageholdes 26.375%, 11.375% refunderbar)
    "GB": 0.00,
    "UK": 0.00,
    "SE": 0.15,
    "NO": 0.15,
    "FI": 0.15,
    "FR": 0.128,
    "NL": 0.15,
    "CH": 0.15,       # 35% brutto, 20% refunderes via DBO
    "IE": 0.25,
    "DK": 0.22,       # Dansk udbytte → 22% selskabsskat
    "IT": 0.15,
    "ES": 0.15,
    "BE": 0.15,
    "AT": 0.15,
    "LU": 0.15,
    "JP": 0.15,
    "AU": 0.15,
    "CA": 0.15,
    "HK": 0.00,
    "SG": 0.00,
}

# Lande med refunderbar kildeskat over DBO-sats
RECLAIMABLE_EXCESS: dict[str, float] = {
    "DE": 0.11375,    # 26.375% - 15% DBO = 11.375% refunderbar
    "CH": 0.20,       # 35% - 15% DBO = 20% refunderbar
    "IT": 0.11,       # 26% - 15% = 11% refunderbar
    "FR": 0.172,      # 30% - 12.8% = 17.2% refunderbar
}


# ── Dataklasser ─────────────────────────────────────────────

@dataclass
class DividendRecord:
    """Én udbyttebetaling."""
    id: int = 0
    symbol: str = ""
    pay_date: str = ""
    ex_date: str = ""
    gross_amount: float = 0.0      # Brutto i original valuta
    currency: str = "USD"
    fx_rate: float = 1.0           # Kurs til DKK
    gross_dkk: float = 0.0        # Brutto i DKK
    withholding_pct: float = 0.0   # Faktisk kildeskat-procent
    withholding_amount: float = 0.0  # Kildeskat betalt (original valuta)
    withholding_dkk: float = 0.0   # Kildeskat i DKK
    net_dkk: float = 0.0          # Netto i DKK
    country: str = ""
    dbo_rate: float = 0.0          # DBO-sats
    creditable_dkk: float = 0.0    # Kredit mod dansk skat
    reclaimable_dkk: float = 0.0   # Refunderbar kildeskat
    broker: str = ""
    notes: str = ""


@dataclass
class DividendSummary:
    """Årsoversigt over udbytter."""
    year: int
    total_gross_dkk: float = 0.0
    total_withholding_dkk: float = 0.0
    total_net_dkk: float = 0.0
    total_creditable_dkk: float = 0.0
    total_reclaimable_dkk: float = 0.0
    by_country: dict[str, dict] = None
    by_symbol: dict[str, dict] = None
    dividend_count: int = 0

    def __post_init__(self) -> None:
        if self.by_country is None:
            self.by_country = {}
        if self.by_symbol is None:
            self.by_symbol = {}


# ── Dividend Tracker ────────────────────────────────────────

class DividendTracker:
    """
    Udbytte-tracking med kildeskat-kredit beregning.

    Brug:
        tracker = DividendTracker()

        # Registrér udbytte
        tracker.record_dividend(
            symbol="AAPL",
            pay_date="2025-05-15",
            gross_amount=250.0,
            currency="USD",
            fx_rate=6.85,
            country="US",
            broker="alpaca",
        )

        # Årsoversigt
        summary = tracker.get_annual_summary(2025)

        # Reclaimable kildeskat
        reclaimable = tracker.get_reclaimable(2025)
    """

    def __init__(self, db_path: str = "data_cache/dividends.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dividends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    pay_date TEXT NOT NULL,
                    ex_date TEXT DEFAULT '',
                    gross_amount REAL NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    fx_rate REAL DEFAULT 1.0,
                    gross_dkk REAL NOT NULL,
                    withholding_pct REAL DEFAULT 0,
                    withholding_amount REAL DEFAULT 0,
                    withholding_dkk REAL DEFAULT 0,
                    net_dkk REAL NOT NULL,
                    country TEXT DEFAULT '',
                    dbo_rate REAL DEFAULT 0,
                    creditable_dkk REAL DEFAULT 0,
                    reclaimable_dkk REAL DEFAULT 0,
                    broker TEXT DEFAULT '',
                    notes TEXT DEFAULT ''
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_div_date ON dividends(pay_date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_div_symbol ON dividends(symbol)"
            )

    # ── Record Dividend ─────────────────────────────────────

    def record_dividend(
        self,
        symbol: str,
        pay_date: str,
        gross_amount: float,
        currency: str = "USD",
        fx_rate: float = 1.0,
        country: str = "",
        withholding_pct: float | None = None,
        withholding_amount: float | None = None,
        broker: str = "",
        ex_date: str = "",
        notes: str = "",
    ) -> DividendRecord:
        """
        Registrér en udbyttebetaling.

        Beregner automatisk:
          - Kildeskat (hvis ikke angivet)
          - DBO-kredit
          - Refunderbar kildeskat
        """
        country = country.upper()[:2]

        # Bestem kildeskat
        dbo_rate = DBO_RATES.get(country, 0.15)  # Default 15%
        if withholding_pct is None:
            withholding_pct = dbo_rate
        if withholding_amount is None:
            withholding_amount = gross_amount * withholding_pct

        # DKK beregninger
        gross_dkk = gross_amount * fx_rate
        withholding_dkk = withholding_amount * fx_rate
        net_dkk = gross_dkk - withholding_dkk

        # Kredit: max DBO-sats × brutto
        max_creditable = gross_dkk * dbo_rate
        creditable_dkk = min(withholding_dkk, max_creditable)

        # Refunderbar: faktisk betalt - DBO-sats (for lande med højere sats)
        reclaimable_dkk = 0.0
        excess_rate = RECLAIMABLE_EXCESS.get(country, 0)
        if excess_rate > 0:
            reclaimable_dkk = gross_dkk * excess_rate

        record = DividendRecord(
            symbol=symbol.upper(),
            pay_date=pay_date,
            ex_date=ex_date,
            gross_amount=gross_amount,
            currency=currency,
            fx_rate=fx_rate,
            gross_dkk=round(gross_dkk, 2),
            withholding_pct=withholding_pct,
            withholding_amount=withholding_amount,
            withholding_dkk=round(withholding_dkk, 2),
            net_dkk=round(net_dkk, 2),
            country=country,
            dbo_rate=dbo_rate,
            creditable_dkk=round(creditable_dkk, 2),
            reclaimable_dkk=round(reclaimable_dkk, 2),
            broker=broker,
            notes=notes,
        )

        # Gem i database
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO dividends "
                "(symbol, pay_date, ex_date, gross_amount, currency, fx_rate, "
                "gross_dkk, withholding_pct, withholding_amount, withholding_dkk, "
                "net_dkk, country, dbo_rate, creditable_dkk, reclaimable_dkk, "
                "broker, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.symbol, record.pay_date, record.ex_date,
                    record.gross_amount, record.currency, record.fx_rate,
                    record.gross_dkk, record.withholding_pct,
                    record.withholding_amount, record.withholding_dkk,
                    record.net_dkk, record.country, record.dbo_rate,
                    record.creditable_dkk, record.reclaimable_dkk,
                    record.broker, record.notes,
                ),
            )
            record.id = cursor.lastrowid

        logger.info(
            f"[dividend] {symbol}: {gross_amount:.2f} {currency} "
            f"({country}, kildeskat {withholding_pct*100:.1f}%) → "
            f"netto {net_dkk:,.0f} DKK"
        )

        return record

    # ── Queries ─────────────────────────────────────────────

    def get_dividends(
        self,
        year: int | None = None,
        symbol: str | None = None,
        limit: int = 500,
    ) -> list[DividendRecord]:
        """Hent udbytter med filtrering."""
        query = "SELECT * FROM dividends WHERE 1=1"
        params: list[Any] = []

        if year:
            query += " AND pay_date LIKE ?"
            params.append(f"{year}%")
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())

        query += " ORDER BY pay_date DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_record(r) for r in rows]

    def get_annual_summary(self, year: int) -> DividendSummary:
        """Årsoversigt over udbytter."""
        dividends = self.get_dividends(year=year)

        summary = DividendSummary(year=year, dividend_count=len(dividends))

        for d in dividends:
            summary.total_gross_dkk += d.gross_dkk
            summary.total_withholding_dkk += d.withholding_dkk
            summary.total_net_dkk += d.net_dkk
            summary.total_creditable_dkk += d.creditable_dkk
            summary.total_reclaimable_dkk += d.reclaimable_dkk

            # By country
            if d.country not in summary.by_country:
                summary.by_country[d.country] = {
                    "gross_dkk": 0, "withholding_dkk": 0,
                    "net_dkk": 0, "count": 0,
                }
            c = summary.by_country[d.country]
            c["gross_dkk"] += d.gross_dkk
            c["withholding_dkk"] += d.withholding_dkk
            c["net_dkk"] += d.net_dkk
            c["count"] += 1

            # By symbol
            if d.symbol not in summary.by_symbol:
                summary.by_symbol[d.symbol] = {
                    "gross_dkk": 0, "net_dkk": 0, "count": 0,
                }
            s = summary.by_symbol[d.symbol]
            s["gross_dkk"] += d.gross_dkk
            s["net_dkk"] += d.net_dkk
            s["count"] += 1

        return summary

    def get_reclaimable(self, year: int) -> list[dict]:
        """Refunderbar kildeskat per land for et år."""
        dividends = self.get_dividends(year=year)
        by_country: dict[str, float] = {}

        for d in dividends:
            if d.reclaimable_dkk > 0:
                by_country[d.country] = (
                    by_country.get(d.country, 0) + d.reclaimable_dkk
                )

        return [
            {"country": country, "reclaimable_dkk": round(amount, 2)}
            for country, amount in sorted(
                by_country.items(), key=lambda x: -x[1]
            )
        ]

    # ── Helpers ─────────────────────────────────────────────

    def _row_to_record(self, row: tuple) -> DividendRecord:
        return DividendRecord(
            id=row[0],
            symbol=row[1],
            pay_date=row[2],
            ex_date=row[3] or "",
            gross_amount=row[4],
            currency=row[5],
            fx_rate=row[6],
            gross_dkk=row[7],
            withholding_pct=row[8],
            withholding_amount=row[9],
            withholding_dkk=row[10],
            net_dkk=row[11],
            country=row[12],
            dbo_rate=row[13],
            creditable_dkk=row[14],
            reclaimable_dkk=row[15],
            broker=row[16],
            notes=row[17] or "",
        )
