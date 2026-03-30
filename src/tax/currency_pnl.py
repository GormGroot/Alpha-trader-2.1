"""
Currency P&L Tracker — valutakursgevinster og -tab.

Valutakursgevinster er skattemæssigt relevante for selskaber (Kursgevinstloven).
Denne modul tracker:
  - FX-kurser ved køb og salg af positioner i fremmed valuta
  - Beregner valutakursgevinst/-tab ved realisering
  - Aggregerer per valuta per år
  - Genererer rapport til revisor
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# ── Dataklasser ─────────────────────────────────────────────

@dataclass
class FXTransaction:
    """En valutatransaktion (implicit fra aktiehandel)."""
    id: int = 0
    date: str = ""
    symbol: str = ""               # Aktiesymbol der blev handlet
    trade_side: str = ""           # "buy" eller "sell"
    currency: str = "USD"          # Fremmed valuta
    amount_foreign: float = 0.0    # Beløb i fremmed valuta
    fx_rate_at_trade: float = 0.0  # Kurs DKK/foreign ved handlen
    amount_dkk: float = 0.0       # Beløb i DKK
    broker: str = ""
    order_id: str = ""


@dataclass
class FXGainLoss:
    """Valutakursgevinst/-tab beregning."""
    currency: str
    acquired_amount: float          # Fremmed valuta købt
    acquired_rate: float            # FIFO-vægtet anskaffelseskurs
    disposed_amount: float          # Fremmed valuta solgt
    disposed_rate: float            # Kurs ved salg
    fx_gain_dkk: float             # Gevinst/tab i DKK
    trade_date: str = ""
    symbol: str = ""


@dataclass
class CurrencyPnLSummary:
    """Årsoversigt valutakursgevinster."""
    year: int
    total_fx_gain_dkk: float = 0.0
    total_fx_loss_dkk: float = 0.0
    net_fx_pnl_dkk: float = 0.0
    by_currency: dict[str, dict] = None
    transaction_count: int = 0

    def __post_init__(self) -> None:
        if self.by_currency is None:
            self.by_currency = {}


# ── Currency P&L Tracker ───────────────────────────────────

class CurrencyPnLTracker:
    """
    Valutakurs P&L tracker.

    Brug:
        tracker = CurrencyPnLTracker()

        # Registrér valuta-eksponering ved køb
        tracker.record_fx_buy(
            date="2025-03-15",
            symbol="AAPL",
            currency="USD",
            amount_foreign=5000.0,
            fx_rate=6.85,
            broker="alpaca",
        )

        # Ved salg: beregn FX-gevinst
        fx_gain = tracker.record_fx_sell(
            date="2025-09-20",
            symbol="AAPL",
            currency="USD",
            amount_foreign=5500.0,
            fx_rate=7.10,
            broker="alpaca",
        )

        # Årsoversigt
        summary = tracker.get_annual_summary(2025)
    """

    def __init__(self, db_path: str = "data_cache/currency_pnl.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            # FX lot tracking (FIFO)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fx_lots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    currency TEXT NOT NULL,
                    amount REAL NOT NULL,
                    remaining REAL NOT NULL,
                    rate REAL NOT NULL,
                    date TEXT NOT NULL,
                    symbol TEXT DEFAULT '',
                    broker TEXT DEFAULT ''
                )
            """)
            # FX realized gains
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fx_gains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    symbol TEXT DEFAULT '',
                    amount_foreign REAL NOT NULL,
                    acquired_rate REAL NOT NULL,
                    disposed_rate REAL NOT NULL,
                    fx_gain_dkk REAL NOT NULL,
                    broker TEXT DEFAULT ''
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fxg_date ON fx_gains(date)"
            )

    # ── Record FX Transactions ──────────────────────────────

    def record_fx_buy(
        self,
        date: str,
        symbol: str,
        currency: str,
        amount_foreign: float,
        fx_rate: float,
        broker: str = "",
    ) -> None:
        """
        Registrér valutakøb (implicit ved aktiekøb).

        Opretter en FIFO FX-lot.
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO fx_lots "
                "(currency, amount, remaining, rate, date, symbol, broker) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (currency.upper(), amount_foreign, amount_foreign,
                 fx_rate, date, symbol.upper(), broker),
            )

        logger.debug(
            f"[fx-pnl] Køb {amount_foreign:.0f} {currency} @ {fx_rate:.4f} "
            f"({symbol})"
        )

    def record_fx_sell(
        self,
        date: str,
        symbol: str,
        currency: str,
        amount_foreign: float,
        fx_rate: float,
        broker: str = "",
    ) -> FXGainLoss:
        """
        Registrér valutasalg (implicit ved aktiesalg).

        Beregner FX-gevinst via FIFO lots.

        Returns:
            FXGainLoss med beregnet valutakursgevinst.
        """
        currency = currency.upper()
        total_fx_gain = 0.0
        total_consumed = 0.0
        weighted_rate = 0.0
        remaining = amount_foreign

        with sqlite3.connect(self._db_path) as conn:
            # FIFO: ældste lots først
            lots = conn.execute(
                "SELECT id, remaining, rate FROM fx_lots "
                "WHERE currency = ? AND remaining > 0 "
                "ORDER BY date ASC, id ASC",
                (currency,),
            ).fetchall()

            for lot_id, lot_remaining, lot_rate in lots:
                if remaining <= 0:
                    break

                consume = min(remaining, lot_remaining)
                # FX-gevinst = (salg-kurs - køb-kurs) × beløb
                fx_gain = (fx_rate - lot_rate) * consume
                total_fx_gain += fx_gain
                total_consumed += consume
                weighted_rate += lot_rate * consume

                conn.execute(
                    "UPDATE fx_lots SET remaining = ? WHERE id = ?",
                    (lot_remaining - consume, lot_id),
                )
                remaining -= consume

            # Vægtet gennemsnitskurs
            avg_rate = weighted_rate / total_consumed if total_consumed > 0 else fx_rate

            # Gem realized gain
            conn.execute(
                "INSERT INTO fx_gains "
                "(date, currency, symbol, amount_foreign, acquired_rate, "
                "disposed_rate, fx_gain_dkk, broker) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (date, currency, symbol.upper(), amount_foreign,
                 avg_rate, fx_rate, round(total_fx_gain, 2), broker),
            )

        result = FXGainLoss(
            currency=currency,
            acquired_amount=amount_foreign,
            acquired_rate=round(avg_rate, 4),
            disposed_amount=amount_foreign,
            disposed_rate=fx_rate,
            fx_gain_dkk=round(total_fx_gain, 2),
            trade_date=date,
            symbol=symbol,
        )

        if abs(total_fx_gain) > 100:
            logger.info(
                f"[fx-pnl] {symbol}: FX {'gevinst' if total_fx_gain > 0 else 'tab'} "
                f"{total_fx_gain:,.0f} DKK "
                f"({avg_rate:.4f} → {fx_rate:.4f} {currency}/DKK)"
            )

        return result

    # ── Queries ─────────────────────────────────────────────

    def get_annual_summary(self, year: int) -> CurrencyPnLSummary:
        """Årsoversigt over valutakursgevinster."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT currency, symbol, amount_foreign, acquired_rate, "
                "disposed_rate, fx_gain_dkk, date "
                "FROM fx_gains WHERE date LIKE ? "
                "ORDER BY date",
                (f"{year}%",),
            ).fetchall()

        summary = CurrencyPnLSummary(
            year=year, transaction_count=len(rows)
        )

        for r in rows:
            ccy, symbol, amount, acq_rate, disp_rate, gain, date = r

            if gain > 0:
                summary.total_fx_gain_dkk += gain
            else:
                summary.total_fx_loss_dkk += gain  # Negativt tal

            if ccy not in summary.by_currency:
                summary.by_currency[ccy] = {
                    "gains_dkk": 0, "losses_dkk": 0,
                    "net_dkk": 0, "transactions": 0,
                }
            c = summary.by_currency[ccy]
            if gain > 0:
                c["gains_dkk"] += gain
            else:
                c["losses_dkk"] += gain
            c["net_dkk"] += gain
            c["transactions"] += 1

        summary.net_fx_pnl_dkk = round(
            summary.total_fx_gain_dkk + summary.total_fx_loss_dkk, 2
        )
        summary.total_fx_gain_dkk = round(summary.total_fx_gain_dkk, 2)
        summary.total_fx_loss_dkk = round(summary.total_fx_loss_dkk, 2)

        return summary

    def get_open_fx_positions(self) -> dict[str, dict]:
        """Åbne valutapositioner (urealiseret FX-eksponering)."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT currency, SUM(remaining), "
                "SUM(remaining * rate) / SUM(remaining) "
                "FROM fx_lots WHERE remaining > 0 "
                "GROUP BY currency"
            ).fetchall()

        return {
            r[0]: {
                "amount": round(r[1], 2),
                "avg_rate": round(r[2], 4),
            }
            for r in rows if r[1] > 0
        }
