"""
Transaktionslog – komplet dokumentation af alle handler.

Logger hver handel med dato, aktie, antal, kurs i USD, valutakurs og
DKK-beløb. Kan eksporteres som CSV til dokumentation overfor SKAT.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.risk.portfolio_tracker import ClosedTrade
from src.tax.currency import CurrencyConverter


@dataclass
class TransactionRecord:
    """Én transaktion med fuld dokumentation."""

    # Identifikation
    transaction_id: int
    trade_date: str          # ISO dato for afslutning
    entry_date: str          # ISO dato for åbning

    # Handel
    symbol: str
    side: str                # "long" / "short"
    qty: float
    entry_price_usd: float
    exit_price_usd: float
    exit_reason: str

    # Valuta
    entry_fx_rate: float     # USD/DKK på åbningsdatoen
    exit_fx_rate: float      # USD/DKK på lukningstidspunktet

    # DKK-beregninger
    entry_value_dkk: float   # Anskaffelsessum i DKK
    exit_value_dkk: float    # Afståelsessum i DKK
    realized_pnl_usd: float
    realized_pnl_dkk: float

    # Kumulativ
    cumulative_pnl_dkk: float


class TransactionLog:
    """
    Logger alle handler med fuld valuta- og DKK-dokumentation.

    Data gemmes i SQLite og kan eksporteres som CSV.
    """

    def __init__(
        self,
        currency: CurrencyConverter | None = None,
        cache_dir: str = "data_cache",
    ) -> None:
        self._currency = currency or CurrencyConverter(cache_dir=cache_dir)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._cache_dir / "transaction_log.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    entry_price_usd REAL NOT NULL,
                    exit_price_usd REAL NOT NULL,
                    exit_reason TEXT,
                    entry_fx_rate REAL NOT NULL,
                    exit_fx_rate REAL NOT NULL,
                    entry_value_dkk REAL NOT NULL,
                    exit_value_dkk REAL NOT NULL,
                    realized_pnl_usd REAL NOT NULL,
                    realized_pnl_dkk REAL NOT NULL,
                    cumulative_pnl_dkk REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tx_date
                ON transactions (trade_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tx_symbol
                ON transactions (symbol)
            """)

    # ── Log handler ───────────────────────────────────────────

    def log_trade(self, trade: ClosedTrade) -> TransactionRecord:
        """
        Log en lukket handel med fuld DKK-beregning.

        Args:
            trade: ClosedTrade fra PortfolioTracker.

        Returns:
            TransactionRecord med alle felter udfyldt.
        """
        # Udtræk datoer (YYYY-MM-DD)
        entry_date = self._extract_date(trade.entry_time)
        exit_date = self._extract_date(trade.exit_time)

        # Hent valutakurser
        entry_fx = self._currency.get_rate(entry_date)
        exit_fx = self._currency.get_rate(exit_date)

        # DKK-beregninger
        entry_value_dkk = trade.entry_price * trade.qty * entry_fx
        exit_value_dkk = trade.exit_price * trade.qty * exit_fx
        pnl_usd = trade.realized_pnl
        pnl_dkk = exit_value_dkk - entry_value_dkk

        # Kumulativ P&L
        cum_pnl = self._get_cumulative_pnl() + pnl_dkk

        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO transactions (
                    trade_date, entry_date, symbol, side, qty,
                    entry_price_usd, exit_price_usd, exit_reason,
                    entry_fx_rate, exit_fx_rate,
                    entry_value_dkk, exit_value_dkk,
                    realized_pnl_usd, realized_pnl_dkk,
                    cumulative_pnl_dkk
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    exit_date, entry_date, trade.symbol, trade.side,
                    trade.qty, trade.entry_price, trade.exit_price,
                    trade.exit_reason, entry_fx, exit_fx,
                    entry_value_dkk, exit_value_dkk,
                    pnl_usd, pnl_dkk, cum_pnl,
                ),
            )
            tx_id = cursor.lastrowid

        record = TransactionRecord(
            transaction_id=tx_id,
            trade_date=exit_date,
            entry_date=entry_date,
            symbol=trade.symbol,
            side=trade.side,
            qty=trade.qty,
            entry_price_usd=trade.entry_price,
            exit_price_usd=trade.exit_price,
            exit_reason=trade.exit_reason,
            entry_fx_rate=entry_fx,
            exit_fx_rate=exit_fx,
            entry_value_dkk=entry_value_dkk,
            exit_value_dkk=exit_value_dkk,
            realized_pnl_usd=pnl_usd,
            realized_pnl_dkk=pnl_dkk,
            cumulative_pnl_dkk=cum_pnl,
        )

        logger.info(
            f"[skat] Logget: {trade.symbol} {trade.side} {trade.qty}x "
            f"P&L={pnl_dkk:+,.2f} DKK (kurs {exit_fx:.4f})"
        )
        return record

    def log_trades(self, trades: list[ClosedTrade]) -> list[TransactionRecord]:
        """Log flere handler på én gang."""
        return [self.log_trade(t) for t in trades]

    # ── Hent data ─────────────────────────────────────────────

    def get_transactions(
        self,
        year: int | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """
        Hent transaktioner som DataFrame.

        Args:
            year: Filtrer på skatteår.
            symbol: Filtrer på aktiesymbol.
        """
        query = "SELECT * FROM transactions WHERE 1=1"
        params: list = []

        if year:
            query += " AND trade_date LIKE ?"
            params.append(f"{year}-%")
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())

        query += " ORDER BY trade_date ASC"

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_yearly_summary(self, year: int) -> dict:
        """Hent opsummering for et skatteår."""
        df = self.get_transactions(year=year)
        if df.empty:
            return {
                "year": year,
                "num_trades": 0,
                "total_pnl_usd": 0.0,
                "total_pnl_dkk": 0.0,
                "gains_dkk": 0.0,
                "losses_dkk": 0.0,
            }

        gains = df[df["realized_pnl_dkk"] > 0]["realized_pnl_dkk"].sum()
        losses = df[df["realized_pnl_dkk"] < 0]["realized_pnl_dkk"].sum()

        return {
            "year": year,
            "num_trades": len(df),
            "total_pnl_usd": df["realized_pnl_usd"].sum(),
            "total_pnl_dkk": df["realized_pnl_dkk"].sum(),
            "gains_dkk": gains,
            "losses_dkk": losses,
        }

    # ── Eksport ───────────────────────────────────────────────

    def export_csv(self, filepath: str, year: int | None = None) -> str:
        """
        Eksportér transaktionslog som CSV (til revisor/SKAT).

        Returns:
            Stien til den oprettede CSV-fil.
        """
        df = self.get_transactions(year=year)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"[skat] Eksporteret {len(df)} transaktioner til {path}")
        return str(path)

    # ── Interne ───────────────────────────────────────────────

    def _get_cumulative_pnl(self) -> float:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT cumulative_pnl_dkk FROM transactions "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else 0.0

    @staticmethod
    def _extract_date(timestamp: str) -> str:
        """Udtræk YYYY-MM-DD fra en ISO-timestamp."""
        if not timestamp:
            return datetime.now().strftime("%Y-%m-%d")
        return timestamp[:10]
