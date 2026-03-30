"""
Watchlist Manager — intelligent watchlist med Alpha Score ranking.

Features:
  - Watchlist CRUD (tilføj, fjern, kategoriser)
  - Automatisk Alpha Score beregning for alle symboler
  - Ranking: top aktier sorteret efter Alpha Score
  - Kategorier: "core", "opportunistic", "monitoring"
  - Score-historik og trend-tracking
  - Smart forslag: symboler der bør tilføjes/fjernes
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.trader.intelligence.alpha_score import AlphaScoreEngine, AlphaScore


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class WatchlistEntry:
    """En aktie på watchlisten."""
    symbol: str
    category: str = "monitoring"    # "core", "opportunistic", "monitoring"
    added_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    target_price: float | None = None
    stop_price: float | None = None
    alpha_score: float | None = None
    alpha_signal: str = ""
    last_scored: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "category": self.category,
            "alpha_score": self.alpha_score,
            "alpha_signal": self.alpha_signal,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "notes": self.notes,
            "added_at": self.added_at.isoformat(),
        }


# ── Watchlist Manager ────────────────────────────────────────

class WatchlistManager:
    """
    Intelligent watchlist med Alpha Score integration.

    Brug:
        wl = WatchlistManager(alpha_engine=engine)

        # CRUD
        wl.add("AAPL", category="core", notes="Tech bellwether")
        wl.add("NOVO-B.CO", category="core", notes="GLP-1 leader")
        wl.remove("TSLA")

        # Score alle
        wl.refresh_scores()

        # Hent ranked
        ranked = wl.get_ranked()
        for entry in ranked:
            print(f"{entry.symbol}: {entry.alpha_score:.0f} ({entry.alpha_signal})")
    """

    def __init__(
        self,
        alpha_engine: AlphaScoreEngine | None = None,
        db_path: str = "data_cache/watchlist.db",
    ) -> None:
        self._engine = alpha_engine
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    symbol TEXT PRIMARY KEY,
                    category TEXT NOT NULL DEFAULT 'monitoring',
                    notes TEXT DEFAULT '',
                    target_price REAL,
                    stop_price REAL,
                    alpha_score REAL,
                    alpha_signal TEXT DEFAULT '',
                    added_at TEXT NOT NULL,
                    last_scored TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    symbol TEXT NOT NULL,
                    score REAL NOT NULL,
                    signal TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (symbol) REFERENCES watchlist(symbol)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_score_hist ON score_history(symbol, timestamp)"
            )

    # ── CRUD ─────────────────────────────────────────────────

    def add(
        self,
        symbol: str,
        category: str = "monitoring",
        notes: str = "",
        target_price: float | None = None,
        stop_price: float | None = None,
    ) -> WatchlistEntry:
        """Tilføj symbol til watchlist."""
        symbol = symbol.upper()
        entry = WatchlistEntry(
            symbol=symbol,
            category=category,
            notes=notes,
            target_price=target_price,
            stop_price=stop_price,
        )

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO watchlist
                   (symbol, category, notes, target_price, stop_price, added_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (symbol, category, notes, target_price, stop_price,
                 entry.added_at.isoformat()),
            )

        logger.info(f"[watchlist] Tilføjet {symbol} ({category})")
        return entry

    def remove(self, symbol: str) -> bool:
        """Fjern symbol fra watchlist."""
        symbol = symbol.upper()
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM watchlist WHERE symbol = ?", (symbol,)
            )
            return cursor.rowcount > 0

    def update(self, symbol: str, **kwargs) -> None:
        """Opdatér watchlist-entry."""
        symbol = symbol.upper()
        allowed = {"category", "notes", "target_price", "stop_price"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if not updates:
            return

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [symbol]

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                f"UPDATE watchlist SET {set_clause} WHERE symbol = ?",
                values,
            )

    # ── Queries ──────────────────────────────────────────────

    def get_all(self) -> list[WatchlistEntry]:
        """Hent alle watchlist entries."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM watchlist ORDER BY alpha_score DESC NULLS LAST"
            ).fetchall()

        return [self._row_to_entry(r) for r in rows]

    def get_by_category(self, category: str) -> list[WatchlistEntry]:
        """Hent entries for en kategori."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM watchlist WHERE category = ? "
                "ORDER BY alpha_score DESC NULLS LAST",
                (category,),
            ).fetchall()

        return [self._row_to_entry(r) for r in rows]

    def get_ranked(self, limit: int = 20) -> list[WatchlistEntry]:
        """Hent top entries ranked by Alpha Score."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM watchlist WHERE alpha_score IS NOT NULL "
                "ORDER BY alpha_score DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [self._row_to_entry(r) for r in rows]

    def get_symbols(self) -> list[str]:
        """Hent alle symboler."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT symbol FROM watchlist").fetchall()
        return [r[0] for r in rows]

    # ── Alpha Score Integration ──────────────────────────────

    def refresh_scores(self) -> list[WatchlistEntry]:
        """Genberegn Alpha Scores for alle symboler."""
        if not self._engine:
            logger.warning("[watchlist] Ingen AlphaScoreEngine — kan ikke score")
            return self.get_all()

        symbols = self.get_symbols()
        if not symbols:
            return []

        logger.info(f"[watchlist] Scorer {len(symbols)} symboler...")

        scores = self._engine.calculate_batch(symbols)

        now = datetime.now().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            for score in scores:
                conn.execute(
                    """UPDATE watchlist SET alpha_score = ?, alpha_signal = ?,
                       last_scored = ? WHERE symbol = ?""",
                    (score.total, score.signal, now, score.symbol),
                )
                # Gem historik
                conn.execute(
                    """INSERT INTO score_history (symbol, score, signal, timestamp)
                       VALUES (?, ?, ?, ?)""",
                    (score.symbol, score.total, score.signal, now),
                )

        logger.info(f"[watchlist] {len(scores)} scores opdateret")
        return self.get_all()

    def get_score_history(
        self, symbol: str, days: int = 30
    ) -> list[dict]:
        """Hent score-historik for et symbol."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                """SELECT score, signal, timestamp FROM score_history
                   WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?""",
                (symbol.upper(), days * 4),  # ~4 scores per dag
            ).fetchall()

        return [
            {"score": r[0], "signal": r[1], "timestamp": r[2]}
            for r in rows
        ]

    # ── Smart Suggestions ────────────────────────────────────

    def get_buy_candidates(self, min_score: float = 70) -> list[WatchlistEntry]:
        """Aktier med Alpha Score over threshold."""
        return [
            e for e in self.get_all()
            if e.alpha_score is not None and e.alpha_score >= min_score
        ]

    def get_sell_candidates(self, max_score: float = 30) -> list[WatchlistEntry]:
        """Aktier med Alpha Score under threshold."""
        return [
            e for e in self.get_all()
            if e.alpha_score is not None and e.alpha_score <= max_score
        ]

    def get_deteriorating(self, threshold: float = -10) -> list[dict]:
        """Aktier med faldende Alpha Score."""
        results = []
        for symbol in self.get_symbols():
            history = self.get_score_history(symbol, days=7)
            if len(history) >= 2:
                latest = history[0]["score"]
                oldest = history[-1]["score"]
                change = latest - oldest
                if change < threshold:
                    results.append({
                        "symbol": symbol,
                        "current_score": latest,
                        "change": change,
                        "signal": history[0]["signal"],
                    })

        return sorted(results, key=lambda x: x["change"])

    # ── Hjælper ──────────────────────────────────────────────

    def _row_to_entry(self, row: tuple) -> WatchlistEntry:
        return WatchlistEntry(
            symbol=row[0],
            category=row[1],
            notes=row[2] or "",
            target_price=row[3],
            stop_price=row[4],
            alpha_score=row[5],
            alpha_signal=row[6] or "",
            added_at=datetime.fromisoformat(row[7]) if row[7] else datetime.now(),
            last_scored=datetime.fromisoformat(row[8]) if row[8] else None,
        )

    def summary(self) -> dict:
        """Kort oversigt."""
        entries = self.get_all()
        return {
            "total": len(entries),
            "core": sum(1 for e in entries if e.category == "core"),
            "opportunistic": sum(1 for e in entries if e.category == "opportunistic"),
            "monitoring": sum(1 for e in entries if e.category == "monitoring"),
            "scored": sum(1 for e in entries if e.alpha_score is not None),
            "buy_signals": sum(1 for e in entries
                              if e.alpha_signal in ("BUY", "STRONG_BUY")),
            "sell_signals": sum(1 for e in entries
                              if e.alpha_signal in ("SELL", "STRONG_SELL")),
        }
