"""
DataPipeline – scheduler der henter markedsdata og beregner indikatorer.

Kører som et loop der:
  1. Tjekker om markedet er åbent (US Eastern)
  2. Henter friske OHLCV-data for alle konfigurerede aktier
  3. Beregner alle tekniske indikatorer
  4. Gemmer resultatet i SQLite (via MarketDataFetcher-cache)
  5. Venter til næste interval og gentager

Robust mod nedbrud: al data caches i SQLite, så pipeline kan
genstartes uden tab.
"""

from __future__ import annotations

import io
import pickle
import signal
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from config.settings import settings
from src.data.market_data import MarketDataFetcher
from src.data.indicators import add_all_indicators


# ── Markedskalender (US) ─────────────────────────────────────

# Faste helligdage (måned, dag). Flydende helligdage (MLK, Presidents Day
# osv.) kræver en hel kalender – vi springer dem over for enkelhedens skyld
# og tjekker i stedet om yfinance returnerer data.
_US_FIXED_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (7, 4),   # Independence Day
    (12, 25), # Christmas
}


class MarketCalendar:
    """Hjælper til at afgøre om det amerikanske aktiemarked er åbent."""

    def __init__(self, timezone: str | None = None) -> None:
        self.tz = ZoneInfo(timezone or settings.trading.timezone)

        h, m = settings.trading.market_open.split(":")
        self._open_hour, self._open_min = int(h), int(m)

        h, m = settings.trading.market_close.split(":")
        self._close_hour, self._close_min = int(h), int(m)

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Returnér True hvis markedet er åbent lige nu."""
        dt = (dt or self.now()).astimezone(self.tz)

        # Weekend
        if dt.weekday() >= 5:
            return False

        # Kendte helligdage
        if (dt.month, dt.day) in _US_FIXED_HOLIDAYS:
            return False

        # Inden for åbningstid
        market_open = dt.replace(
            hour=self._open_hour, minute=self._open_min, second=0, microsecond=0,
        )
        market_close = dt.replace(
            hour=self._close_hour, minute=self._close_min, second=0, microsecond=0,
        )
        return market_open <= dt <= market_close

    def seconds_until_open(self, dt: datetime | None = None) -> float:
        """Sekunder til næste markedsåbning. 0 hvis allerede åbent."""
        dt = (dt or self.now()).astimezone(self.tz)

        if self.is_market_open(dt):
            return 0.0

        # Find næste hverdag
        next_open = dt.replace(
            hour=self._open_hour, minute=self._open_min, second=0, microsecond=0,
        )
        if next_open <= dt:
            next_open += timedelta(days=1)

        while next_open.weekday() >= 5 or (next_open.month, next_open.day) in _US_FIXED_HOLIDAYS:
            next_open += timedelta(days=1)

        return (next_open - dt).total_seconds()

    def seconds_until_close(self, dt: datetime | None = None) -> float:
        """Sekunder til markedet lukker. 0 hvis allerede lukket."""
        dt = (dt or self.now()).astimezone(self.tz)

        if not self.is_market_open(dt):
            return 0.0

        market_close = dt.replace(
            hour=self._close_hour, minute=self._close_min, second=0, microsecond=0,
        )
        return (market_close - dt).total_seconds()


# ── Indikator-cache (SQLite) ─────────────────────────────────

class IndicatorStore:
    """Gemmer DataFrames med indikatorer i SQLite som parquet-blobs."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS indicator_snapshots (
                    symbol     TEXT NOT NULL,
                    interval   TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data       BLOB NOT NULL,
                    PRIMARY KEY (symbol, interval)
                )
            """)

    def save(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        blob = pickle.dumps(df)
        now = datetime.now(tz=None).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO indicator_snapshots
                    (symbol, interval, updated_at, data)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, interval, now, blob),
            )

    def load(self, symbol: str, interval: str) -> pd.DataFrame | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT data FROM indicator_snapshots WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            ).fetchone()

        if row is None:
            return None
        return pickle.loads(row[0])

    def load_all(self, interval: str = "1d") -> dict[str, pd.DataFrame]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT symbol, data FROM indicator_snapshots WHERE interval = ?",
                (interval,),
            ).fetchall()

        return {
            symbol: pickle.loads(blob)
            for symbol, blob in rows
        }

    def cleanup(self, max_age_days: int = 7) -> None:
        """Remove indicator snapshots older than max_age_days."""
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "DELETE FROM indicator_snapshots WHERE updated_at < ?",
                (cutoff,),
            )


# ── DataPipeline ─────────────────────────────────────────────

class DataPipeline:
    """
    Kører en data-pipeline der periodisk:
      1. Henter OHLCV-data (parallel for store univers)
      2. Beregner tekniske indikatorer
      3. Gemmer resultater

    Kan køres som blocking loop (run) eller single-shot (run_once).
    Understøtter AssetUniverse for dynamisk symbol-liste.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        interval: str | None = None,
        check_interval: int | None = None,
        cache_dir: str | None = None,
        universe=None,
    ) -> None:
        self._universe = universe
        self.symbols = symbols or settings.trading.symbols
        self.interval = interval or settings.market_data.interval
        self.check_interval = check_interval or settings.trading.check_interval_seconds

        cache_path = Path(cache_dir or settings.market_data.cache_dir)
        self.fetcher = MarketDataFetcher(cache_dir=str(cache_path))
        self.store = IndicatorStore(cache_path / "indicators.db")
        self.calendar = MarketCalendar()

        self._stop_event = threading.Event()
        self._run_count = 0

    def _get_symbols(self) -> list[str]:
        """Hent aktive symboler fra universe eller config."""
        if self._universe is not None:
            return self._universe.active_symbols
        return self.symbols

    # ── Single-shot ──────────────────────────────────────────

    def run_once(self) -> dict[str, pd.DataFrame]:
        """
        Kør pipelinen én gang: hent data → beregn indikatorer → gem.

        For > 10 symboler bruges parallel hentning automatisk.

        Returns:
            Dict med symbol → DataFrame (OHLCV + alle indikatorer).
        """
        self._run_count += 1
        symbols = self._get_symbols()
        now = self.calendar.now()
        logger.info(
            f"Pipeline run #{self._run_count} startet "
            f"({now.strftime('%Y-%m-%d %H:%M %Z')}) – "
            f"{len(symbols)} symboler"
        )

        results: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            try:
                df = self._process_symbol(symbol)
                results[symbol] = df
            except Exception as exc:
                logger.error(f"Pipeline fejl for {symbol}: {exc}")
                # Forsøg at loade seneste gemte data
                cached = self.store.load(symbol, self.interval)
                if cached is not None:
                    logger.info(f"Bruger cached indikatorer for {symbol}")
                    results[symbol] = cached
                else:
                    results[symbol] = pd.DataFrame()

        success = sum(1 for df in results.values() if not df.empty)
        logger.info(
            f"Pipeline run #{self._run_count} færdig: "
            f"{success}/{len(symbols)} symboler OK"
        )

        return results

    def _process_symbol(self, symbol: str) -> pd.DataFrame:
        """Hent data og beregn indikatorer for ét symbol."""
        # Hent nok data til at SMA_200 kan beregnes
        lookback = max(settings.market_data.lookback_days, 250)

        df = self.fetcher.get_historical(
            symbol,
            interval=self.interval,
            lookback_days=lookback,
        )

        if df.empty:
            logger.warning(f"Ingen data for {symbol} – springer over")
            return df

        # Standardisér kolonnenavne (cache returnerer lowercase)
        col_map = {c.lower(): c.title() for c in ["open", "high", "low", "close", "volume"]}
        df.rename(columns=col_map, inplace=True)

        # Beregn alle indikatorer
        df = add_all_indicators(df)

        # Gem i indicator store
        self.store.save(symbol, self.interval, df)
        logger.debug(
            f"{symbol}: {len(df)} rækker, "
            f"{len(df.columns)} kolonner, "
            f"seneste: {df.index[-1]}"
        )

        return df

    # ── Continuous loop ──────────────────────────────────────

    def run(self) -> None:
        """
        Kør pipelinen i et kontinuerligt loop.

        - I markedets åbningstid: kør hvert `check_interval` sekund.
        - Uden for åbningstid: sov til næste markedsåbning.
        - Stop med Ctrl+C eller stop().
        """
        symbols = self._get_symbols()
        logger.info(
            f"Pipeline startet – {len(symbols)} symboler, "
            f"interval={self.interval}, "
            f"check hvert {self.check_interval}s"
        )

        # Graceful shutdown via SIGINT/SIGTERM
        def _handle_signal(sig, frame):
            logger.info(f"Signal {sig} modtaget – stopper pipeline")
            self.stop()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # Kør altid én gang ved opstart (hent historisk data)
        logger.info("Indledende datahentning...")
        self.run_once()

        while not self._stop_event.is_set():
            if self.calendar.is_market_open():
                wait = self.check_interval
                self.run_once()
            else:
                secs = self.calendar.seconds_until_open()
                if secs > 0:
                    hours = secs / 3600
                    logger.info(
                        f"Markedet er lukket. Næste åbning om {hours:.1f} timer. "
                        f"Sover..."
                    )
                    # Sov i bidder så vi kan stoppes hurtigt
                    wait = min(secs, 300)  # tjek mindst hvert 5. minut
                else:
                    wait = 60

            # Vent – men kan afbrydes af stop()
            self._stop_event.wait(timeout=wait)

        logger.info("Pipeline stoppet.")

    def stop(self) -> None:
        """Stop det kontinuerlige loop."""
        self._stop_event.set()

    # ── Utility ──────────────────────────────────────────────

    def get_latest(self, symbol: str) -> pd.DataFrame | None:
        """Hent senest beregnede indikatorer fra store."""
        return self.store.load(symbol, self.interval)

    def get_all_latest(self) -> dict[str, pd.DataFrame]:
        """Hent seneste indikatorer for alle symboler."""
        return self.store.load_all(self.interval)
