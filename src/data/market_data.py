"""
MarketDataFetcher – henter og cacher aktiekurser via yfinance + SQLite.

Understøtter:
  - Sekventiel hentning med rate limiting (standard)
  - Parallel batch-hentning for store univers (get_multiple_parallel)
  - Smart caching med SQLite og batch-write
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
import yfinance.shared as _yf_shared
from loguru import logger

from config.settings import settings

# ── Konstanter ───────────────────────────────────────────────
_VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}
_MIN_REQUEST_GAP = 0.35  # sekunder mellem API-kald (rate limiting)


class MarketDataError(Exception):
    """Fejl ved hentning af markedsdata."""


class MarketDataFetcher:
    """Henter aktiekurser via yfinance med SQLite-cache."""

    def __init__(self, cache_dir: str | None = None) -> None:
        self._last_request_time: float = 0.0
        self._session = None  # reusable HTTP session
        self._session_call_count: int = 0

        cache_path = Path(cache_dir or settings.market_data.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_path / "market_data.db"
        self._init_db()

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Create Ticker with shared session to reduce memory allocations."""
        # Reset session every 200 calls to prevent connection/memory accumulation
        self._session_call_count += 1
        if self._session is not None and self._session_call_count >= 200:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
            self._session_call_count = 0

        t = yf.Ticker(symbol)
        if self._session is None:
            self._session = t.session
        else:
            t.session = self._session
        return t

    # ── Database ─────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol    TEXT    NOT NULL,
                    interval  TEXT    NOT NULL,
                    date      TEXT    NOT NULL,
                    open      REAL,
                    high      REAL,
                    low       REAL,
                    close     REAL,
                    volume    INTEGER,
                    fetched_at TEXT   NOT NULL,
                    PRIMARY KEY (symbol, interval, date)
                )
            """)

    # ── Rate limiting ────────────────────────────────────────

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_GAP:
            time.sleep(_MIN_REQUEST_GAP - elapsed)
        self._last_request_time = time.monotonic()

    # ── Cache helpers ────────────────────────────────────────

    def _read_cache(
        self, symbol: str, interval: str, start: str, end: str,
    ) -> pd.DataFrame | None:
        if not settings.market_data.cache_enabled:
            return None

        query = """
            SELECT date, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND interval = ? AND date >= ? AND date <= ?
            ORDER BY date
        """
        with self._get_conn() as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, interval, start, end))

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"], utc=True)
        df.set_index("date", inplace=True)
        df.index = df.index.tz_localize(None)
        # SQLite-kolonner er lowercase – omdøb til standard OHLCV
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        }, inplace=True)
        return df

    def _write_cache(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        if not settings.market_data.cache_enabled or df.empty:
            return

        now = datetime.now(tz=None).isoformat()
        rows = [
            (
                symbol,
                interval,
                idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"]),
                now,
            )
            for idx, row in df.iterrows()
        ]

        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO ohlcv
                    (symbol, interval, date, open, high, low, close, volume, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        logger.debug(f"Cached {len(rows)} rows for {symbol} ({interval})")

    # ── Public API ───────────────────────────────────────────

    def get_historical(
        self,
        symbol: str,
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        """
        Hent historisk OHLCV-data for ét symbol.

        Args:
            symbol: Ticker (f.eks. "AAPL").
            interval: Tidsinterval – "1d", "1h", "5m" osv.
            start: Startdato "YYYY-MM-DD" (valgfrit).
            end: Slutdato "YYYY-MM-DD" (valgfrit).
            lookback_days: Alternativ til start – antal dage tilbage.

        Returns:
            DataFrame med kolonner: Open, High, Low, Close, Volume.
        """
        if interval not in _VALID_INTERVALS:
            raise MarketDataError(f"Ugyldigt interval '{interval}'. Brug: {_VALID_INTERVALS}")

        if start is None:
            days = lookback_days or settings.market_data.lookback_days
            start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # Forsøg cache — for intraday data (1m, 5m, 15m) spring cache over
        # da vi har brug for live/frisk data
        is_intraday = interval in ("1m", "2m", "5m", "15m", "30m")
        if not is_intraday:
            cached = self._read_cache(symbol, interval, start, end)
            if cached is not None and len(cached) > 0:
                logger.debug(f"Cache hit: {symbol} ({interval}) – {len(cached)} rows")
                return cached

        # Hent fra yfinance
        logger.info(f"Henter {symbol} ({interval}) fra yfinance: {start} → {end}")
        self._throttle()

        try:
            ticker = self._get_ticker(symbol)
            df = ticker.history(interval=interval, start=start, end=end)
        except Exception as exc:
            raise MarketDataError(
                f"Kunne ikke hente data for {symbol}: {exc}"
            ) from exc

        if df.empty:
            logger.warning(f"Ingen data returneret for {symbol} ({interval})")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        # Behold kun OHLCV
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Clear yfinance global cache to prevent memory leak
        _yf_shared._DFS.clear()
        _yf_shared._ERRORS.clear()

        # Cache resultatet (kun for daglige+ intervaller, ikke intraday)
        if not is_intraday:
            self._write_cache(symbol, interval, df)

        return df

    def get_multiple(
        self,
        symbols: list[str] | None = None,
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
        lookback_days: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Hent historisk data for flere symboler.

        Returns:
            Dict med symbol → DataFrame.
        """
        symbols = symbols or settings.trading.symbols
        results: dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            try:
                results[symbol] = self.get_historical(
                    symbol, interval=interval, start=start, end=end,
                    lookback_days=lookback_days,
                )
            except MarketDataError as exc:
                logger.error(f"Fejl for {symbol}: {exc}")
                results[symbol] = pd.DataFrame(
                    columns=["Open", "High", "Low", "Close", "Volume"]
                )

        return results

    def get_latest_price(self, symbol: str) -> float:
        """Hent seneste lukkekurs for ét symbol."""
        self._throttle()
        try:
            ticker = self._get_ticker(symbol)
            fast = ticker.fast_info
            price = fast.get("lastPrice") or fast.get("last_price")
            if price is None:
                hist = ticker.history(period="1d")
                if hist.empty:
                    raise MarketDataError(f"Ingen pris fundet for {symbol}")
                price = float(hist["Close"].iloc[-1])
            logger.debug(f"{symbol} seneste pris: ${price:.2f}")
            return float(price)
        except MarketDataError:
            raise
        except Exception as exc:
            raise MarketDataError(f"Kunne ikke hente pris for {symbol}: {exc}") from exc

    def get_multiple_parallel(
        self,
        symbols: list[str],
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
        lookback_days: int | None = None,
        max_workers: int = 8,
        batch_size: int = 50,
    ) -> dict[str, pd.DataFrame]:
        """
        Hent data for mange symboler parallelt med batching.

        Optimeret til store univers (100+ symboler):
          - Tjekker cache først (batch)
          - Henter kun manglende data fra API
          - Parallel API-kald med ThreadPoolExecutor
          - Progress-logging

        Args:
            symbols: Liste af symboler.
            interval: Tidsinterval.
            start: Startdato.
            end: Slutdato.
            lookback_days: Antal dage tilbage.
            max_workers: Max parallelle tråde.
            batch_size: Symboler per batch (for progress-logging).

        Returns:
            Dict med symbol → DataFrame.
        """
        if start is None:
            days = lookback_days or settings.market_data.lookback_days
            start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        total = len(symbols)
        results: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []

        # ── Fase 1: Tjek cache ──
        for symbol in symbols:
            cached = self._read_cache(symbol, interval, start, end)
            if cached is not None and len(cached) > 0:
                results[symbol] = cached
            else:
                to_fetch.append(symbol)

        cache_hits = total - len(to_fetch)
        logger.info(
            f"[parallel] {total} symboler: {cache_hits} cache hits, "
            f"{len(to_fetch)} at hente"
        )

        if not to_fetch:
            return results

        # ── Fase 2: Parallel API-hentning ──
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        fetched = 0
        errors = 0

        def _fetch_one(sym: str) -> tuple[str, pd.DataFrame]:
            try:
                ticker = self._get_ticker(sym)
                df = ticker.history(interval=interval, start=start, end=end)
                if df.empty:
                    return sym, empty_df.copy()
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                self._write_cache(sym, interval, df)
                return sym, df
            except Exception as exc:
                logger.debug(f"[parallel] Fejl for {sym}: {exc}")
                return sym, empty_df.copy()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process i batches for progress-logging
            for batch_start in range(0, len(to_fetch), batch_size):
                batch = to_fetch[batch_start:batch_start + batch_size]
                futures = {
                    executor.submit(_fetch_one, sym): sym
                    for sym in batch
                }

                for future in as_completed(futures):
                    sym, df = future.result()
                    results[sym] = df
                    if not df.empty:
                        fetched += 1
                    else:
                        errors += 1

                done = min(batch_start + batch_size, len(to_fetch))
                logger.info(
                    f"[parallel] Progress: {done}/{len(to_fetch)} "
                    f"({fetched} OK, {errors} fejl)"
                )

        # Clear yfinance global cache to prevent memory leak
        _yf_shared._DFS.clear()
        _yf_shared._ERRORS.clear()

        logger.info(
            f"[parallel] Færdig: {total} symboler, "
            f"{cache_hits} cached + {fetched} hentet + {errors} fejl"
        )
        return results

    def get_cached_symbols(self, interval: str = "1d") -> list[str]:
        """Returnér alle symboler i cachen for et givet interval."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM ohlcv WHERE interval = ?",
                (interval,),
            ).fetchall()
        return [r[0] for r in rows]

    def clear_cache(self, symbol: str | None = None) -> None:
        """Ryd cache – enten for ét symbol eller alt."""
        with self._get_conn() as conn:
            if symbol:
                conn.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
                logger.info(f"Cache ryddet for {symbol}")
            else:
                conn.execute("DELETE FROM ohlcv")
                logger.info("Hele cachen ryddet")
