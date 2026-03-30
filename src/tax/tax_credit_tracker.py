"""
Skattetilgodehavende Tracker.

Tracker selskabets skattetilgodehavende (fremført underskud) over tid:
  - Start-balance (manuelt input)
  - Tilgang: tab-år tilføjer til tilgodehavende
  - Forbrug: gevinst-år modregner
  - Aktuel balance
  - Projektioner: "Hvad bliver tilgodehavendet ved årsskifte?"
  - Alerts: "Tilgodehavende næsten opbrugt"

Persistence: SQLite database.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger


# ── Dataklasser ─────────────────────────────────────────────

@dataclass
class CreditMovement:
    """En bevægelse i skattetilgodehavende."""
    date: str
    movement_type: str       # "initial", "loss_year", "gain_offset", "adjustment"
    amount_dkk: float        # Positiv = tilgang, negativ = forbrug
    balance_after: float
    description: str = ""
    year: int = 0


@dataclass
class CreditProjection:
    """Projektion af tilgodehavende."""
    current_balance: float
    projected_tax: float            # Estimeret skat ved årsskifte
    projected_offset: float         # Forventet modregning
    projected_balance: float        # Balance efter modregning
    fully_utilized: bool = False    # Bruges tilgodehavendet helt op?


# ── Tax Credit Tracker ──────────────────────────────────────

class TaxCreditTracker:
    """
    Skattetilgodehavende tracking og projektion.

    Brug:
        tracker = TaxCreditTracker(initial_credit=500_000)

        # Registrér bevægelser
        tracker.add_loss_year(2023, 200_000)    # Tab → tilgodehavende stiger
        tracker.offset_gain(2024, 150_000)       # Gevinst → tilgodehavende falder

        # Status
        print(f"Balance: {tracker.balance:,.0f} DKK")
        print(tracker.get_history())

        # Projektion
        projection = tracker.project(estimated_tax=80_000)
    """

    def __init__(
        self,
        initial_credit: float = 0.0,
        db_path: str = "data_cache/tax_credit.db",
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._balance = initial_credit
        self._alert_callbacks: list[Callable[[str], None]] = []
        self._init_db()
        self._load_balance()

        # Hvis ingen data i DB, gem initial balance
        if initial_credit > 0 and self._get_movement_count() == 0:
            self._record_movement(
                "initial", initial_credit, "Initial skattetilgodehavende"
            )

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS credit_movements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    movement_type TEXT NOT NULL,
                    amount_dkk REAL NOT NULL,
                    balance_after REAL NOT NULL,
                    description TEXT DEFAULT '',
                    year INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS credit_balance (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    balance REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def _load_balance(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT balance FROM credit_balance WHERE id = 1"
            ).fetchone()
            if row:
                self._balance = row[0]

    def _save_balance(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO credit_balance (id, balance, updated_at) "
                "VALUES (1, ?, ?)",
                (self._balance, datetime.now().isoformat()),
            )

    def _get_movement_count(self) -> int:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM credit_movements"
            ).fetchone()
            return row[0] if row else 0

    def _record_movement(
        self,
        movement_type: str,
        amount: float,
        description: str = "",
        year: int = 0,
    ) -> None:
        self._balance += amount
        self._save_balance()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO credit_movements "
                "(date, movement_type, amount_dkk, balance_after, description, year) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    movement_type,
                    amount,
                    self._balance,
                    description,
                    year or datetime.now().year,
                ),
            )

        logger.info(
            f"[tax-credit] {movement_type}: {amount:+,.0f} DKK → "
            f"balance: {self._balance:,.0f} DKK"
        )

    # ── Properties ──────────────────────────────────────────

    @property
    def balance(self) -> float:
        """Aktuelt skattetilgodehavende i DKK."""
        return self._balance

    # ── Bevægelser ──────────────────────────────────────────

    def add_loss_year(self, year: int, taxable_loss: float) -> float:
        """
        Registrér et tab-år → tilgodehavende stiger.

        Args:
            year: Regnskabsår.
            taxable_loss: Skattemæssigt underskud (positivt tal).

        Returns:
            Ny balance.
        """
        credit = abs(taxable_loss) * 0.22  # 22% af underskuddet
        self._record_movement(
            "loss_year",
            credit,
            f"Underskud {year}: {taxable_loss:,.0f} DKK → "
            f"kredit {credit:,.0f} DKK",
            year=year,
        )
        return self._balance

    def offset_gain(self, year: int, tax_amount: float) -> float:
        """
        Modregn tilgodehavende mod skat.

        Args:
            year: Regnskabsår.
            tax_amount: Skat der skal betales (positivt tal).

        Returns:
            Beløb faktisk modregnet.
        """
        offset = min(abs(tax_amount), self._balance)
        if offset > 0:
            self._record_movement(
                "gain_offset",
                -offset,
                f"Modregning {year}: -{offset:,.0f} DKK mod skat "
                f"på {tax_amount:,.0f} DKK",
                year=year,
            )

            # Check alerts
            self._check_alerts()

        return offset

    def manual_adjustment(self, amount: float, description: str = "") -> float:
        """Manuel justering (revisor-korrektion etc.)."""
        self._record_movement(
            "adjustment",
            amount,
            description or f"Manuel justering: {amount:+,.0f} DKK",
        )
        return self._balance

    # ── History ─────────────────────────────────────────────

    def get_history(self, limit: int = 100) -> list[CreditMovement]:
        """Hent bevægelseshistorik."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT date, movement_type, amount_dkk, balance_after, "
                "description, year "
                "FROM credit_movements "
                "ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            CreditMovement(
                date=r[0],
                movement_type=r[1],
                amount_dkk=r[2],
                balance_after=r[3],
                description=r[4],
                year=r[5],
            )
            for r in rows
        ]

    def get_yearly_summary(self) -> dict[int, dict]:
        """Summary per år."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT year, movement_type, SUM(amount_dkk) "
                "FROM credit_movements "
                "WHERE year > 0 "
                "GROUP BY year, movement_type "
                "ORDER BY year"
            ).fetchall()

        summary: dict[int, dict] = {}
        for year, mtype, total in rows:
            if year not in summary:
                summary[year] = {"additions": 0, "offsets": 0, "net": 0}
            if mtype in ("loss_year", "initial", "adjustment"):
                summary[year]["additions"] += total
            elif mtype == "gain_offset":
                summary[year]["offsets"] += total
            summary[year]["net"] += total

        return summary

    # ── Projektion ──────────────────────────────────────────

    def project(
        self,
        estimated_tax: float = 0.0,
        estimated_loss_credit: float = 0.0,
    ) -> CreditProjection:
        """
        Projicér tilgodehavende ved årsskifte.

        Args:
            estimated_tax: Estimeret skat at betale (positiv).
            estimated_loss_credit: Estimeret ny kredit fra tab (positiv).

        Returns:
            CreditProjection med forventede tal.
        """
        projected_offset = min(estimated_tax, self._balance)
        projected_balance = self._balance - projected_offset + estimated_loss_credit

        return CreditProjection(
            current_balance=round(self._balance, 2),
            projected_tax=round(estimated_tax, 2),
            projected_offset=round(projected_offset, 2),
            projected_balance=round(projected_balance, 2),
            fully_utilized=projected_balance <= 0,
        )

    # ── Alerts ──────────────────────────────────────────────

    def on_alert(self, callback: Callable[[str], None]) -> None:
        """Registrér alert callback."""
        self._alert_callbacks.append(callback)

    def _check_alerts(self) -> None:
        """Check om tilgodehavende kræver alerts."""
        if self._balance <= 0:
            self._notify("Skattetilgodehavende er OPBRUGT (0 DKK)")
        elif self._balance < 50_000:
            self._notify(
                f"Skattetilgodehavende er lavt: {self._balance:,.0f} DKK"
            )

    def _notify(self, message: str) -> None:
        logger.warning(f"[tax-credit] {message}")
        for cb in self._alert_callbacks:
            try:
                cb(message)
            except Exception:
                pass

    # ── Dashboard Data ──────────────────────────────────────

    def dashboard_data(self) -> dict[str, Any]:
        """Data til dashboard widget."""
        history = self.get_history(limit=10)
        return {
            "balance_dkk": round(self._balance, 2),
            "recent_movements": [
                {
                    "date": m.date[:10],
                    "type": m.movement_type,
                    "amount": round(m.amount_dkk, 2),
                    "balance": round(m.balance_after, 2),
                }
                for m in history[:5]
            ],
            "yearly_summary": self.get_yearly_summary(),
        }
