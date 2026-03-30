"""
NewsFetcher – henter nyheder, earnings-kalender og økonomiske events.

Kilder:
  - Finnhub API (firmanyheder + sentiment)
  - Alpha Vantage News Sentiment API
  - RSS feeds (Reuters, CNBC, Yahoo Finance, MarketWatch)
  - Finnhub earnings-kalender
  - Finnhub economic calendar

Rate limiting og caching er built-in for alle kilder.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import feedparser
from loguru import logger


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class NewsArticle:
    """En enkelt nyhedsartikel."""
    title: str
    summary: str
    url: str
    source: str
    published: str          # ISO-format
    symbols: list[str] = field(default_factory=list)
    sentiment_api: float | None = None   # Sentiment fra API (-1 til 1)
    category: str = "general"            # company, market, economy, earnings
    relevance: float = 1.0               # 0-1, hvor relevant for symbolet

    @property
    def age_hours(self) -> float:
        """Timer siden publikation."""
        try:
            pub = datetime.fromisoformat(self.published.replace("Z", "+00:00"))
            now = datetime.now(pub.tzinfo) if pub.tzinfo else datetime.now()
            return max(0, (now - pub).total_seconds() / 3600)
        except (ValueError, TypeError):
            return 999.0

    @property
    def id(self) -> str:
        """Unikt hash-ID baseret på URL."""
        return hashlib.md5(self.url.encode()).hexdigest()[:12]


@dataclass
class EarningsEvent:
    """Earnings release event."""
    symbol: str
    date: str               # YYYY-MM-DD
    hour: str                # "bmo" (before market open), "amc" (after market close)
    eps_estimate: float | None = None
    eps_actual: float | None = None
    revenue_estimate: float | None = None
    revenue_actual: float | None = None

    @property
    def has_reported(self) -> bool:
        return self.eps_actual is not None

    @property
    def eps_surprise_pct(self) -> float | None:
        if self.eps_actual is None or self.eps_estimate is None or self.eps_estimate == 0:
            return None
        return (self.eps_actual - self.eps_estimate) / abs(self.eps_estimate)


@dataclass
class EconomicEvent:
    """Makroøkonomisk event (rentemøde, jobtal osv.)."""
    name: str
    country: str
    date: str
    time: str
    impact: str              # "low", "medium", "high"
    actual: str = ""
    estimate: str = ""
    previous: str = ""
    category: str = ""       # "interest_rate", "employment", "inflation", etc.


# ── RSS Feed konfiguration ───────────────────────────────────

RSS_FEEDS: dict[str, str] = {
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Investing.com": "https://www.investing.com/rss/news.rss",
}

# Kilde-troværdighed (0-1) – bruges til vægtning af sentiment
SOURCE_CREDIBILITY: dict[str, float] = {
    "Reuters": 0.95,
    "Reuters Business": 0.95,
    "Bloomberg": 0.95,
    "CNBC": 0.80,
    "CNBC Top News": 0.80,
    "Yahoo Finance": 0.70,
    "MarketWatch": 0.75,
    "Investing.com": 0.65,
    "Finnhub": 0.80,
    "Alpha Vantage": 0.75,
    "Reddit": 0.30,
    "Unknown": 0.40,
}


def get_source_credibility(source: str) -> float:
    """Returnér troværdighedsscore for en kilde."""
    for key, score in SOURCE_CREDIBILITY.items():
        if key.lower() in source.lower():
            return score
    return SOURCE_CREDIBILITY["Unknown"]


# ── Rate Limiter ─────────────────────────────────────────────

class RateLimiter:
    """Simpel token-bucket rate limiter."""

    def __init__(self, calls_per_minute: int = 30) -> None:
        self._interval = 60.0 / calls_per_minute
        self._last_call: dict[str, float] = {}

    def wait(self, key: str = "default") -> None:
        now = time.time()
        last = self._last_call.get(key, 0)
        elapsed = now - last
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call[key] = time.time()


# ── News Cache ───────────────────────────────────────────────

class NewsCache:
    """SQLite-baseret cache for nyheder og events."""

    def __init__(self, cache_dir: str = "data_cache") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "news_cache.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_cache (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    source TEXT NOT NULL,
                    symbol TEXT,
                    fetched_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_cache(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_expires ON news_cache(expires_at)")

    def get_cached_response(self, key: str) -> str | None:
        """Hent cached API-response hvis den ikke er udløbet."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT data FROM api_cache WHERE cache_key = ? AND expires_at > ?",
                (key, datetime.now().isoformat()),
            ).fetchone()
            return row[0] if row else None

    def set_cached_response(self, key: str, data: str, ttl_minutes: int = 15) -> None:
        """Gem API-response i cache."""
        expires = (datetime.now() + timedelta(minutes=ttl_minutes)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO api_cache (cache_key, data, fetched_at, expires_at)
                   VALUES (?, ?, ?, ?)""",
                (key, data, datetime.now().isoformat(), expires),
            )

    def cleanup(self) -> int:
        """Slet udløbne cache-poster. Returnér antal slettede."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM api_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),),
            )
            cursor2 = conn.execute(
                "DELETE FROM news_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),),
            )
            return (cursor.rowcount or 0) + (cursor2.rowcount or 0)


# ── NewsFetcher ──────────────────────────────────────────────

class NewsFetcher:
    """
    Henter nyheder fra flere kilder med rate limiting og caching.

    Kilder:
      - Finnhub API (kræver FINNHUB_API_KEY)
      - Alpha Vantage News (kræver ALPHA_VANTAGE_KEY)
      - RSS feeds (Reuters, CNBC, Yahoo Finance, MarketWatch)

    Brug:
        fetcher = NewsFetcher(finnhub_key="...", av_key="...")
        articles = fetcher.fetch_company_news("AAPL")
        earnings = fetcher.fetch_earnings_calendar("AAPL")
    """

    def __init__(
        self,
        finnhub_key: str = "",
        av_key: str = "",
        cache_dir: str = "data_cache",
        rate_limit: int = 30,
    ) -> None:
        self._finnhub_key = finnhub_key
        self._av_key = av_key
        self._cache = NewsCache(cache_dir)
        self._rate_limiter = RateLimiter(rate_limit)
        self._finnhub_client = None

        if finnhub_key:
            try:
                import finnhub
                self._finnhub_client = finnhub.Client(api_key=finnhub_key)
            except ImportError:
                logger.warning("[news] finnhub-python ikke installeret")

    @property
    def has_finnhub(self) -> bool:
        return self._finnhub_client is not None

    @property
    def has_alpha_vantage(self) -> bool:
        return bool(self._av_key)

    # ── Finnhub ───────────────────────────────────────────────

    def fetch_company_news_finnhub(
        self,
        symbol: str,
        days_back: int = 7,
    ) -> list[NewsArticle]:
        """Hent firmanyheder fra Finnhub API."""
        if not self._finnhub_client:
            return []

        cache_key = f"finnhub_news_{symbol}_{days_back}"
        cached = self._cache.get_cached_response(cache_key)
        if cached:
            import json
            return [NewsArticle(**a) for a in json.loads(cached)]

        try:
            self._rate_limiter.wait("finnhub")
            end = datetime.now()
            start = end - timedelta(days=days_back)

            raw = self._finnhub_client.company_news(
                symbol,
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
            )

            articles = []
            for item in (raw or []):
                pub = datetime.fromtimestamp(item.get("datetime", 0)).isoformat()
                articles.append(NewsArticle(
                    title=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    url=item.get("url", ""),
                    source=item.get("source", "Finnhub"),
                    published=pub,
                    symbols=[symbol],
                    category="company",
                    relevance=1.0,
                ))

            # Cache
            import json
            self._cache.set_cached_response(
                cache_key,
                json.dumps([{
                    "title": a.title, "summary": a.summary, "url": a.url,
                    "source": a.source, "published": a.published,
                    "symbols": a.symbols, "category": a.category,
                    "relevance": a.relevance,
                } for a in articles]),
                ttl_minutes=30,
            )

            logger.info(f"[news] Finnhub: {len(articles)} nyheder for {symbol}")
            return articles

        except Exception as exc:
            logger.error(f"[news] Finnhub fejl for {symbol}: {exc}")
            return []

    # ── Alpha Vantage ─────────────────────────────────────────

    def fetch_news_alpha_vantage(
        self,
        symbol: str | None = None,
        topics: str = "",
        limit: int = 50,
    ) -> list[NewsArticle]:
        """Hent nyheder fra Alpha Vantage News Sentiment API."""
        if not self._av_key:
            return []

        cache_key = f"av_news_{symbol or 'market'}_{topics}"
        cached = self._cache.get_cached_response(cache_key)
        if cached:
            import json
            return [NewsArticle(**a) for a in json.loads(cached)]

        try:
            import urllib.request
            import json

            self._rate_limiter.wait("alpha_vantage")

            params = f"function=NEWS_SENTIMENT&apikey={self._av_key}&limit={limit}"
            if symbol:
                params += f"&tickers={symbol}"
            if topics:
                params += f"&topics={topics}"

            url = f"https://www.alphavantage.co/query?{params}"
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            articles = []
            for item in data.get("feed", []):
                # Beregn relevans for specifikt symbol
                relevance = 1.0
                symbols = []
                for ticker_info in item.get("ticker_sentiment", []):
                    ticker = ticker_info.get("ticker", "")
                    symbols.append(ticker)
                    if ticker == symbol:
                        relevance = float(ticker_info.get("relevance_score", 0.5))

                # API sentiment score
                api_sentiment = float(item.get("overall_sentiment_score", 0))

                pub = item.get("time_published", "")
                if pub and len(pub) >= 8:
                    # Format: 20260315T120000
                    try:
                        pub = datetime.strptime(pub[:15], "%Y%m%dT%H%M%S").isoformat()
                    except ValueError:
                        pub = datetime.now().isoformat()

                articles.append(NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("summary", ""),
                    url=item.get("url", ""),
                    source=item.get("source", "Alpha Vantage"),
                    published=pub,
                    symbols=symbols[:10],
                    sentiment_api=api_sentiment,
                    category="company" if symbol else "market",
                    relevance=relevance,
                ))

            # Cache
            self._cache.set_cached_response(
                cache_key,
                json.dumps([{
                    "title": a.title, "summary": a.summary, "url": a.url,
                    "source": a.source, "published": a.published,
                    "symbols": a.symbols, "sentiment_api": a.sentiment_api,
                    "category": a.category, "relevance": a.relevance,
                } for a in articles]),
                ttl_minutes=30,
            )

            logger.info(f"[news] Alpha Vantage: {len(articles)} nyheder")
            return articles

        except Exception as exc:
            logger.error(f"[news] Alpha Vantage fejl: {exc}")
            return []

    # ── RSS Feeds ─────────────────────────────────────────────

    def fetch_rss_news(
        self,
        symbol: str | None = None,
        feeds: dict[str, str] | None = None,
        max_per_feed: int = 20,
    ) -> list[NewsArticle]:
        """
        Hent nyheder fra RSS feeds.

        Filtrerer på symbol i titel/summary hvis angivet.
        """
        feeds = feeds or RSS_FEEDS
        articles = []

        for feed_name, feed_url in feeds.items():
            cache_key = f"rss_{feed_name}"
            cached = self._cache.get_cached_response(cache_key)

            if cached:
                import json
                feed_articles = [NewsArticle(**a) for a in json.loads(cached)]
            else:
                try:
                    self._rate_limiter.wait("rss")
                    parsed = feedparser.parse(feed_url)

                    feed_articles = []
                    for entry in (parsed.entries or [])[:max_per_feed]:
                        pub = ""
                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                            pub = datetime(*entry.published_parsed[:6]).isoformat()
                        elif hasattr(entry, "published"):
                            pub = entry.published

                        feed_articles.append(NewsArticle(
                            title=entry.get("title", ""),
                            summary=entry.get("summary", "")[:500],
                            url=entry.get("link", ""),
                            source=feed_name,
                            published=pub or datetime.now().isoformat(),
                            category="market",
                        ))

                    # Cache RSS i 15 min
                    import json
                    self._cache.set_cached_response(
                        cache_key,
                        json.dumps([{
                            "title": a.title, "summary": a.summary, "url": a.url,
                            "source": a.source, "published": a.published,
                            "category": a.category,
                        } for a in feed_articles]),
                        ttl_minutes=15,
                    )

                except Exception as exc:
                    logger.warning(f"[news] RSS fejl for {feed_name}: {exc}")
                    feed_articles = []

            # Filtrér på symbol hvis angivet
            if symbol:
                symbol_upper = symbol.upper()
                feed_articles = [
                    a for a in feed_articles
                    if symbol_upper in a.title.upper() or symbol_upper in a.summary.upper()
                ]
                for a in feed_articles:
                    a.symbols = [symbol]
                    a.category = "company"

            articles.extend(feed_articles)

        logger.debug(f"[news] RSS: {len(articles)} artikler" +
                     (f" for {symbol}" if symbol else ""))
        return articles

    # ── Samlet hentning ───────────────────────────────────────

    def fetch_company_news(
        self,
        symbol: str,
        days_back: int = 7,
        include_rss: bool = True,
    ) -> list[NewsArticle]:
        """
        Hent alle nyheder for et symbol fra alle tilgængelige kilder.

        Returns:
            Liste af NewsArticle, sorteret efter dato (nyeste først).
        """
        articles: list[NewsArticle] = []

        # Finnhub
        articles.extend(self.fetch_company_news_finnhub(symbol, days_back))

        # Alpha Vantage
        articles.extend(self.fetch_news_alpha_vantage(symbol))

        # RSS
        if include_rss:
            articles.extend(self.fetch_rss_news(symbol))

        # Dedupliker baseret på URL-hash
        seen: set[str] = set()
        unique = []
        for a in articles:
            if a.id not in seen:
                seen.add(a.id)
                unique.append(a)

        # Sortér efter dato (nyeste først)
        unique.sort(key=lambda a: a.published, reverse=True)
        return unique

    def fetch_market_news(
        self,
        topics: str = "",
        include_rss: bool = True,
    ) -> list[NewsArticle]:
        """Hent generelle markedsnyheder."""
        articles: list[NewsArticle] = []
        articles.extend(self.fetch_news_alpha_vantage(topics=topics))
        if include_rss:
            articles.extend(self.fetch_rss_news())

        seen: set[str] = set()
        unique = []
        for a in articles:
            if a.id not in seen:
                seen.add(a.id)
                unique.append(a)
        unique.sort(key=lambda a: a.published, reverse=True)
        return unique

    # ── Earnings Calendar ─────────────────────────────────────

    def fetch_earnings_calendar(
        self,
        symbol: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[EarningsEvent]:
        """
        Hent earnings-kalender fra Finnhub.

        Args:
            symbol: Specifikt symbol (None = alle).
            from_date: Start-dato (YYYY-MM-DD).
            to_date: Slut-dato (YYYY-MM-DD).

        Returns:
            Liste af EarningsEvent.
        """
        if not self._finnhub_client:
            return []

        cache_key = f"earnings_{symbol or 'all'}_{from_date}_{to_date}"
        cached = self._cache.get_cached_response(cache_key)
        if cached:
            import json
            return [EarningsEvent(**e) for e in json.loads(cached)]

        try:
            self._rate_limiter.wait("finnhub")

            if not from_date:
                from_date = datetime.now().strftime("%Y-%m-%d")
            if not to_date:
                to_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

            raw = self._finnhub_client.earnings_calendar(
                _from=from_date, to=to_date, symbol=symbol or "",
            )

            events = []
            for item in (raw or {}).get("earningsCalendar", []):
                events.append(EarningsEvent(
                    symbol=item.get("symbol", ""),
                    date=item.get("date", ""),
                    hour=item.get("hour", ""),
                    eps_estimate=item.get("epsEstimate"),
                    eps_actual=item.get("epsActual"),
                    revenue_estimate=item.get("revenueEstimate"),
                    revenue_actual=item.get("revenueActual"),
                ))

            if symbol:
                events = [e for e in events if e.symbol == symbol]

            # Cache i 60 min
            import json
            self._cache.set_cached_response(
                cache_key,
                json.dumps([{
                    "symbol": e.symbol, "date": e.date, "hour": e.hour,
                    "eps_estimate": e.eps_estimate, "eps_actual": e.eps_actual,
                    "revenue_estimate": e.revenue_estimate,
                    "revenue_actual": e.revenue_actual,
                } for e in events]),
                ttl_minutes=60,
            )

            logger.info(f"[news] Earnings: {len(events)} events")
            return events

        except Exception as exc:
            logger.error(f"[news] Earnings kalender fejl: {exc}")
            return []

    # ── Economic Calendar ─────────────────────────────────────

    def fetch_economic_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[EconomicEvent]:
        """
        Hent økonomisk kalender fra Finnhub.

        Inkluderer: rentemøder, jobtal, inflation, PMI, osv.
        """
        if not self._finnhub_client:
            return []

        cache_key = f"economic_{from_date}_{to_date}"
        cached = self._cache.get_cached_response(cache_key)
        if cached:
            import json
            return [EconomicEvent(**e) for e in json.loads(cached)]

        try:
            self._rate_limiter.wait("finnhub")

            if not from_date:
                from_date = datetime.now().strftime("%Y-%m-%d")
            if not to_date:
                to_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

            raw = self._finnhub_client.economic_calendar(
                _from=from_date, to=to_date,
            )

            events = []
            for item in (raw or {}).get("economicCalendar", []):
                impact = item.get("impact", "low")
                # Klassificér kategori
                name = item.get("event", "").lower()
                category = _categorize_economic_event(name)

                events.append(EconomicEvent(
                    name=item.get("event", ""),
                    country=item.get("country", ""),
                    date=item.get("date", ""),
                    time=item.get("time", ""),
                    impact=impact,
                    actual=str(item.get("actual", "")),
                    estimate=str(item.get("estimate", "")),
                    previous=str(item.get("prev", "")),
                    category=category,
                ))

            # Cache i 60 min
            import json
            self._cache.set_cached_response(
                cache_key,
                json.dumps([{
                    "name": e.name, "country": e.country, "date": e.date,
                    "time": e.time, "impact": e.impact, "actual": e.actual,
                    "estimate": e.estimate, "previous": e.previous,
                    "category": e.category,
                } for e in events]),
                ttl_minutes=60,
            )

            logger.info(f"[news] Economic calendar: {len(events)} events")
            return events

        except Exception as exc:
            logger.error(f"[news] Economic calendar fejl: {exc}")
            return []

    def is_near_earnings(
        self,
        symbol: str,
        hours_before: int = 1,
        hours_after: int = 1,
    ) -> bool:
        """
        Tjek om et symbol er tæt på earnings release.

        Bruges til at pause handel omkring earnings.
        """
        earnings = self.fetch_earnings_calendar(symbol)
        now = datetime.now()

        for e in earnings:
            try:
                event_date = datetime.strptime(e.date, "%Y-%m-%d")
                # Sæt tid baseret på bmo/amc
                if e.hour == "bmo":
                    event_time = event_date.replace(hour=9, minute=30)
                elif e.hour == "amc":
                    event_time = event_date.replace(hour=16, minute=0)
                else:
                    event_time = event_date.replace(hour=12, minute=0)

                before = event_time - timedelta(hours=hours_before)
                after = event_time + timedelta(hours=hours_after)

                if before <= now <= after:
                    return True
            except (ValueError, TypeError):
                continue

        return False


# ── Hjælpefunktioner ─────────────────────────────────────────

def _categorize_economic_event(name: str) -> str:
    """Kategorisér en økonomisk event baseret på navn."""
    name = name.lower()
    if any(w in name for w in ["interest rate", "fed", "fomc", "ecb", "boe"]):
        return "interest_rate"
    if any(w in name for w in ["nonfarm", "payroll", "employment", "unemployment", "jobless"]):
        return "employment"
    if any(w in name for w in ["cpi", "inflation", "pce", "price index"]):
        return "inflation"
    if any(w in name for w in ["gdp", "growth"]):
        return "gdp"
    if any(w in name for w in ["pmi", "manufacturing", "ism"]):
        return "manufacturing"
    if any(w in name for w in ["retail sales", "consumer"]):
        return "consumer"
    if any(w in name for w in ["housing", "home"]):
        return "housing"
    return "other"
