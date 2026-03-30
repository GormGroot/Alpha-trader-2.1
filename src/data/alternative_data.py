"""
Alternative Data Module – utraditionelle datapunkter for aktieanalyse.

Funktionalitet:
  - Google Trends: søgeinteresse for firmaer/produkter (pytrends)
  - Web-trafik estimater: website trafik-trends
  - Jobopslag-analyse: ansættelsesaktivitet som vækstindikator
  - Patent-tracking: USPTO patentansøgninger
  - App Store rankings: app download-trends
  - GitHub aktivitet: open source aktivitet for tech-firmaer
  - Alt Data Score: aggregeret score 0–100

Gratis datakilder – ingen API-nøgler kræves.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings

# ── Optional imports ─────────────────────────────────────────

try:
    from pytrends.request import TrendReq
    _HAS_PYTRENDS = True
except ImportError:
    _HAS_PYTRENDS = False
    logger.warning("pytrends ikke installeret – Google Trends utilgængelig")

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ── Konstanter ───────────────────────────────────────────────

_MIN_REQUEST_GAP = 0.5   # Rate limiting (sekunder)
GITHUB_API_BASE = "https://api.github.com"
USPTO_API_BASE = "https://developer.uspto.gov/ibd-api/v1/application/publications"

# Mapping: ticker → firma/søgeterm
DEFAULT_SEARCH_TERMS: dict[str, list[str]] = {
    "AAPL": ["Apple", "iPhone", "buy iPhone"],
    "MSFT": ["Microsoft", "Azure", "Microsoft Office"],
    "GOOGL": ["Google", "Google Cloud", "Android"],
    "AMZN": ["Amazon", "Amazon Prime", "AWS"],
    "TSLA": ["Tesla", "Tesla Model", "buy Tesla"],
    "META": ["Meta", "Facebook", "Instagram"],
    "NFLX": ["Netflix", "Netflix stock"],
    "NVDA": ["Nvidia", "Nvidia GPU", "CUDA"],
    "AMD": ["AMD", "AMD Ryzen", "AMD GPU"],
    "CRM": ["Salesforce", "Salesforce CRM"],
    "SHOP": ["Shopify", "Shopify store"],
    "UBER": ["Uber", "Uber ride"],
    "SNAP": ["Snapchat", "Snap Inc"],
    "RBLX": ["Roblox", "Roblox game"],
    "DASH": ["DoorDash", "DoorDash delivery"],
}

# Mapping: ticker → firma website
DEFAULT_WEBSITES: dict[str, str] = {
    "AAPL": "apple.com",
    "MSFT": "microsoft.com",
    "GOOGL": "google.com",
    "AMZN": "amazon.com",
    "TSLA": "tesla.com",
    "META": "facebook.com",
    "NFLX": "netflix.com",
    "NVDA": "nvidia.com",
    "SHOP": "shopify.com",
    "UBER": "uber.com",
}

# Mapping: ticker → GitHub org
DEFAULT_GITHUB_ORGS: dict[str, str] = {
    "MSFT": "microsoft",
    "GOOGL": "google",
    "META": "facebook",
    "AMZN": "aws",
    "AAPL": "apple",
    "TSLA": "teslamotors",
    "NVDA": "NVIDIA",
    "AMD": "AMD",
    "CRM": "salesforce",
    "SHOP": "Shopify",
    "UBER": "uber",
}

# Mapping: ticker → USPTO assignee
DEFAULT_PATENT_ASSIGNEES: dict[str, str] = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "META": "Meta Platforms",
    "NVDA": "NVIDIA",
    "AMD": "Advanced Micro Devices",
    "IBM": "International Business Machines",
    "INTC": "Intel",
}


# ── Enums ────────────────────────────────────────────────────

class TrendDirection(Enum):
    """Retning for en trend."""
    RISING = "rising"
    STABLE = "stable"
    FALLING = "falling"
    SPIKE = "spike"      # Pludselig stor stigning


class AltDataSignal(Enum):
    """Signal fra alternativ data."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ── Dataclasses ──────────────────────────────────────────────

@dataclass
class GoogleTrendsResult:
    """Google Trends søgeinteresse for et firma."""
    symbol: str
    search_terms: list[str]
    current_interest: float          # 0–100 (seneste uge)
    avg_interest_30d: float          # Gennemsnit seneste 30 dage
    avg_interest_90d: float          # Gennemsnit seneste 90 dage
    trend_direction: TrendDirection
    change_pct_30d: float            # % ændring over 30 dage
    spike_detected: bool             # Pludselig stor stigning
    related_rising: list[str]        # Stigende relaterede søgninger
    interest_over_time: pd.DataFrame | None = None
    signal: AltDataSignal = AltDataSignal.NEUTRAL
    date: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        """Score 0–100 baseret på trends."""
        base = 50.0

        # Trend retning
        if self.trend_direction == TrendDirection.RISING:
            base += 15
        elif self.trend_direction == TrendDirection.FALLING:
            base -= 15
        elif self.trend_direction == TrendDirection.SPIKE:
            base += 10  # Spikes kan være negative, men ofte attention = bullish

        # Ændring over 30 dage
        if self.change_pct_30d > 50:
            base += 10
        elif self.change_pct_30d > 20:
            base += 5
        elif self.change_pct_30d < -30:
            base -= 10
        elif self.change_pct_30d < -10:
            base -= 5

        return max(0, min(100, base))


@dataclass
class WebTrafficResult:
    """Web-trafik estimat for et firmas website."""
    symbol: str
    website: str
    estimated_visits: int            # Estimeret månedlige besøg
    trend_direction: TrendDirection
    change_pct: float                # % ændring fra forrige periode
    signal: AltDataSignal = AltDataSignal.NEUTRAL
    date: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        """Score 0–100 baseret på trafik."""
        base = 50.0
        if self.change_pct > 20:
            base += 20
        elif self.change_pct > 5:
            base += 10
        elif self.change_pct < -20:
            base -= 20
        elif self.change_pct < -5:
            base -= 10
        return max(0, min(100, base))


@dataclass
class JobPostingsResult:
    """Jobopslag-analyse for et firma."""
    symbol: str
    company_name: str
    active_postings: int             # Antal aktive jobopslag
    change_pct_30d: float            # % ændring over 30 dage
    trend_direction: TrendDirection
    top_categories: list[str]        # Top job-kategorier
    hiring_signal: AltDataSignal     # bullish=ansætter, bearish=skærer ned
    date: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        """Score 0–100. Flere ansættelser = mere bullish."""
        base = 50.0
        if self.change_pct_30d > 30:
            base += 20
        elif self.change_pct_30d > 10:
            base += 10
        elif self.change_pct_30d < -20:
            base -= 20
        elif self.change_pct_30d < -10:
            base -= 10
        return max(0, min(100, base))


@dataclass
class PatentResult:
    """Patent-tracking resultat for et firma."""
    symbol: str
    assignee: str
    total_patents_ytd: int           # Patenter i år
    total_patents_prev_year: int     # Patenter forrige år
    change_pct: float                # % ændring
    recent_patents: list[str]        # Titler på seneste patenter
    ai_related_count: int = 0        # AI-relaterede patenter
    signal: AltDataSignal = AltDataSignal.NEUTRAL
    date: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        """Score 0–100. Flere patenter = mere innovativ."""
        base = 50.0
        if self.change_pct > 20:
            base += 15
        elif self.change_pct > 5:
            base += 8
        elif self.change_pct < -20:
            base -= 10

        # AI bonus
        if self.ai_related_count > 10:
            base += 10
        elif self.ai_related_count > 3:
            base += 5

        return max(0, min(100, base))


@dataclass
class GitHubActivityResult:
    """GitHub open source aktivitet for et firma."""
    symbol: str
    org_name: str
    public_repos: int
    total_stars: int
    total_forks: int
    recent_commits_30d: int          # Commits seneste 30 dage
    trend_direction: TrendDirection
    top_repos: list[dict]            # [{name, stars, language}]
    signal: AltDataSignal = AltDataSignal.NEUTRAL
    date: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        """Score 0–100. Mere aktivitet = sundere tech-firma."""
        base = 50.0
        if self.total_stars > 100_000:
            base += 15
        elif self.total_stars > 10_000:
            base += 8
        if self.trend_direction == TrendDirection.RISING:
            base += 10
        elif self.trend_direction == TrendDirection.FALLING:
            base -= 10
        return max(0, min(100, base))


@dataclass
class AppRankingResult:
    """App Store ranking resultat."""
    symbol: str
    app_name: str
    category_rank: int | None        # Rank i kategori
    overall_rank: int | None         # Overall rank
    estimated_downloads: int = 0     # Estimerede downloads
    trend_direction: TrendDirection = TrendDirection.STABLE
    signal: AltDataSignal = AltDataSignal.NEUTRAL
    date: datetime = field(default_factory=datetime.now)

    @property
    def score(self) -> float:
        """Score 0–100. Lavere rank = bedre."""
        base = 50.0
        if self.category_rank and self.category_rank <= 5:
            base += 20
        elif self.category_rank and self.category_rank <= 20:
            base += 10
        elif self.category_rank and self.category_rank <= 50:
            base += 5
        if self.trend_direction == TrendDirection.RISING:
            base += 10
        elif self.trend_direction == TrendDirection.FALLING:
            base -= 10
        return max(0, min(100, base))


@dataclass
class AltDataScore:
    """Aggregeret alternativ data score for et symbol."""
    symbol: str
    overall_score: float             # 0–100
    signal: AltDataSignal
    google_trends: GoogleTrendsResult | None = None
    web_traffic: WebTrafficResult | None = None
    job_postings: JobPostingsResult | None = None
    patents: PatentResult | None = None
    github_activity: GitHubActivityResult | None = None
    app_ranking: AppRankingResult | None = None
    components: dict[str, float] = field(default_factory=dict)
    confidence_adjustment: int = 0   # -10 til +10
    alerts: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


# ── Alternative Data Tracker ─────────────────────────────────

class AlternativeDataTracker:
    """
    Samler og analyserer alternative datapunkter.

    Gratis datakilder:
      - Google Trends (pytrends)
      - GitHub API (uautentificeret: 60 req/time)
      - USPTO patent API (gratis)
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        search_terms: dict[str, list[str]] | None = None,
        github_orgs: dict[str, str] | None = None,
        patent_assignees: dict[str, str] | None = None,
    ) -> None:
        self._last_request_time: float = 0.0
        self._search_terms = search_terms or DEFAULT_SEARCH_TERMS
        self._github_orgs = github_orgs or DEFAULT_GITHUB_ORGS
        self._patent_assignees = patent_assignees or DEFAULT_PATENT_ASSIGNEES

        cache_path = Path(cache_dir or settings.market_data.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_path / "alternative_data.db"
        self._init_db()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_GAP:
            time.sleep(_MIN_REQUEST_GAP - elapsed)
        self._last_request_time = time.monotonic()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS google_trends (
                    symbol          TEXT NOT NULL,
                    search_term     TEXT NOT NULL,
                    current_interest REAL,
                    avg_30d         REAL,
                    avg_90d         REAL,
                    change_pct_30d  REAL,
                    trend_direction TEXT,
                    spike_detected  INTEGER DEFAULT 0,
                    date            TEXT NOT NULL,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (symbol, search_term, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS github_activity (
                    symbol          TEXT NOT NULL,
                    org_name        TEXT NOT NULL,
                    public_repos    INTEGER,
                    total_stars     INTEGER,
                    total_forks     INTEGER,
                    recent_commits  INTEGER,
                    trend_direction TEXT,
                    date            TEXT NOT NULL,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patent_data (
                    symbol          TEXT NOT NULL,
                    assignee        TEXT NOT NULL,
                    patents_ytd     INTEGER,
                    patents_prev    INTEGER,
                    change_pct      REAL,
                    ai_related      INTEGER DEFAULT 0,
                    date            TEXT NOT NULL,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alt_data_scores (
                    symbol          TEXT NOT NULL,
                    overall_score   REAL,
                    signal          TEXT,
                    components      TEXT,
                    date            TEXT NOT NULL,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (symbol, date)
                )
            """)

    # ── Google Trends ─────────────────────────────────────────

    def get_google_trends(
        self,
        symbol: str,
        terms: list[str] | None = None,
        timeframe: str = "today 3-m",
        use_cache: bool = True,
    ) -> GoogleTrendsResult:
        """
        Hent Google Trends søgeinteresse for et symbol.

        Args:
            symbol: Ticker symbol.
            terms: Søgetermer (default fra DEFAULT_SEARCH_TERMS).
            timeframe: Tidsramme – "today 3-m", "today 12-m", etc.
            use_cache: Brug SQLite cache.

        Returns:
            GoogleTrendsResult med interesse-data og trend.
        """
        symbol = symbol.upper()
        terms = terms or self._search_terms.get(symbol, [symbol])

        # Cache check
        if use_cache:
            cached = self._read_trends_cache(symbol)
            if cached:
                return cached

        if not _HAS_PYTRENDS:
            logger.warning("[altdata] pytrends ikke tilgængelig")
            return self._empty_trends_result(symbol, terms)

        try:
            self._throttle()
            pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

            # Brug første term som primær
            primary_term = terms[0]
            kw_list = [primary_term]

            pytrends.build_payload(kw_list, timeframe=timeframe, geo="")

            # Interest over time
            interest_df = pytrends.interest_over_time()

            if interest_df.empty:
                return self._empty_trends_result(symbol, terms)

            # Beregn statistikker
            values = interest_df[primary_term].values
            current = float(values[-1]) if len(values) > 0 else 0
            avg_30d = float(np.mean(values[-4:])) if len(values) >= 4 else current
            avg_90d = float(np.mean(values)) if len(values) > 0 else current

            # Trend direction
            if len(values) >= 8:
                recent = np.mean(values[-4:])
                older = np.mean(values[-8:-4])
                if recent > older * 1.5:
                    direction = TrendDirection.SPIKE
                elif recent > older * 1.1:
                    direction = TrendDirection.RISING
                elif recent < older * 0.9:
                    direction = TrendDirection.FALLING
                else:
                    direction = TrendDirection.STABLE
            else:
                direction = TrendDirection.STABLE

            # Change pct
            if avg_90d > 0:
                change_pct = (avg_30d - avg_90d) / avg_90d * 100
            else:
                change_pct = 0.0

            # Spike detection (>2 std dev over mean)
            if len(values) > 4:
                mean_val = np.mean(values[:-1])
                std_val = np.std(values[:-1])
                spike = bool(current > mean_val + 2 * std_val) if std_val > 0 else False
            else:
                spike = False

            # Related queries
            related_rising: list[str] = []
            try:
                related = pytrends.related_queries()
                if primary_term in related and related[primary_term].get("rising") is not None:
                    rising_df = related[primary_term]["rising"]
                    if not rising_df.empty:
                        related_rising = rising_df["query"].head(5).tolist()
            except Exception:
                pass

            # Signal
            if direction in (TrendDirection.RISING, TrendDirection.SPIKE) and change_pct > 10:
                signal = AltDataSignal.BULLISH
            elif direction == TrendDirection.FALLING and change_pct < -15:
                signal = AltDataSignal.BEARISH
            else:
                signal = AltDataSignal.NEUTRAL

            result = GoogleTrendsResult(
                symbol=symbol,
                search_terms=terms,
                current_interest=current,
                avg_interest_30d=round(avg_30d, 1),
                avg_interest_90d=round(avg_90d, 1),
                trend_direction=direction,
                change_pct_30d=round(change_pct, 1),
                spike_detected=spike,
                related_rising=related_rising,
                interest_over_time=interest_df,
                signal=signal,
            )

            # Cache
            self._write_trends_cache(result)
            return result

        except Exception as exc:
            logger.error(f"[altdata] Google Trends fejl for {symbol}: {exc}")
            return self._empty_trends_result(symbol, terms)

    def _empty_trends_result(self, symbol: str, terms: list[str]) -> GoogleTrendsResult:
        return GoogleTrendsResult(
            symbol=symbol,
            search_terms=terms,
            current_interest=0,
            avg_interest_30d=0,
            avg_interest_90d=0,
            trend_direction=TrendDirection.STABLE,
            change_pct_30d=0,
            spike_detected=False,
            related_rising=[],
        )

    def _read_trends_cache(self, symbol: str) -> GoogleTrendsResult | None:
        cache_cutoff = (datetime.now() - timedelta(hours=12)).isoformat()
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT search_term, current_interest, avg_30d, avg_90d,
                          change_pct_30d, trend_direction, spike_detected
                   FROM google_trends
                   WHERE symbol = ? AND fetched_at >= ?
                   ORDER BY date DESC LIMIT 1""",
                (symbol, cache_cutoff),
            ).fetchone()

        if not row:
            return None

        direction = TrendDirection(row[5]) if row[5] else TrendDirection.STABLE
        return GoogleTrendsResult(
            symbol=symbol,
            search_terms=[row[0]],
            current_interest=row[1] or 0,
            avg_interest_30d=row[2] or 0,
            avg_interest_90d=row[3] or 0,
            trend_direction=direction,
            change_pct_30d=row[4] or 0,
            spike_detected=bool(row[6]),
            related_rising=[],
        )

    def _write_trends_cache(self, result: GoogleTrendsResult) -> None:
        now = datetime.now().isoformat()
        today = datetime.now().strftime("%Y-%m-%d")
        term = result.search_terms[0] if result.search_terms else result.symbol
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO google_trends
                   (symbol, search_term, current_interest, avg_30d, avg_90d,
                    change_pct_30d, trend_direction, spike_detected, date, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.symbol, term, result.current_interest,
                    result.avg_interest_30d, result.avg_interest_90d,
                    result.change_pct_30d, result.trend_direction.value,
                    int(result.spike_detected), today, now,
                ),
            )

    # ── Web-trafik ────────────────────────────────────────────

    def estimate_web_traffic(
        self,
        symbol: str,
        website: str | None = None,
    ) -> WebTrafficResult:
        """
        Estimér web-trafik baseret på Google Trends som proxy.

        Rigtig SimilarWeb data kræver betalt API.
        Her bruger vi Google Trends for website-søgning som proxy.
        """
        symbol = symbol.upper()
        website = website or DEFAULT_WEBSITES.get(symbol, "")

        if not website:
            return WebTrafficResult(
                symbol=symbol, website="unknown",
                estimated_visits=0,
                trend_direction=TrendDirection.STABLE,
                change_pct=0.0,
            )

        # Brug Google Trends med website som søgeterm
        trends = self.get_google_trends(symbol, terms=[website])

        # Estimér visits fra interest (rough proxy)
        estimated_visits = int(trends.current_interest * 1_000_000)

        signal = AltDataSignal.NEUTRAL
        if trends.change_pct_30d > 15:
            signal = AltDataSignal.BULLISH
        elif trends.change_pct_30d < -15:
            signal = AltDataSignal.BEARISH

        return WebTrafficResult(
            symbol=symbol,
            website=website,
            estimated_visits=estimated_visits,
            trend_direction=trends.trend_direction,
            change_pct=trends.change_pct_30d,
            signal=signal,
        )

    # ── Jobopslag-analyse ─────────────────────────────────────

    def analyze_job_postings(
        self,
        symbol: str,
        company_name: str | None = None,
    ) -> JobPostingsResult:
        """
        Analysér jobopslag-trends via Google Trends som proxy.

        Rigtige job-API'er (LinkedIn, Indeed) kræver betalt adgang.
        Her bruger vi "[firma] careers" + "[firma] hiring" som proxy.
        """
        symbol = symbol.upper()
        company = company_name or self._search_terms.get(symbol, [symbol])[0]

        # Brug Google Trends for job-relaterede søgninger
        job_terms = [f"{company} careers", f"{company} hiring", f"{company} jobs"]
        trends = self.get_google_trends(symbol, terms=job_terms)

        # Bestem hiring signal
        if trends.change_pct_30d > 20:
            hiring_signal = AltDataSignal.BULLISH
            direction = TrendDirection.RISING
        elif trends.change_pct_30d < -20:
            hiring_signal = AltDataSignal.BEARISH
            direction = TrendDirection.FALLING
        else:
            hiring_signal = AltDataSignal.NEUTRAL
            direction = TrendDirection.STABLE

        # Estimér antal baseret på interest
        estimated_postings = int(trends.current_interest * 100)

        return JobPostingsResult(
            symbol=symbol,
            company_name=company,
            active_postings=estimated_postings,
            change_pct_30d=trends.change_pct_30d,
            trend_direction=direction,
            top_categories=["Engineering", "Sales", "Marketing"],
            hiring_signal=hiring_signal,
        )

    # ── Patent-tracking ───────────────────────────────────────

    def get_patent_activity(
        self,
        symbol: str,
        assignee: str | None = None,
        use_cache: bool = True,
    ) -> PatentResult:
        """
        Hent patent-aktivitet fra USPTO API.

        Args:
            symbol: Ticker symbol.
            assignee: Patent-ejer (firma-navn).
            use_cache: Brug cache.

        Returns:
            PatentResult med patent-statistikker.
        """
        symbol = symbol.upper()
        assignee = assignee or self._patent_assignees.get(symbol, "")

        if not assignee:
            return PatentResult(
                symbol=symbol, assignee="unknown",
                total_patents_ytd=0, total_patents_prev_year=0,
                change_pct=0, recent_patents=[],
            )

        # Cache check
        if use_cache:
            cached = self._read_patent_cache(symbol)
            if cached:
                return cached

        if not _HAS_REQUESTS:
            return PatentResult(
                symbol=symbol, assignee=assignee,
                total_patents_ytd=0, total_patents_prev_year=0,
                change_pct=0, recent_patents=[],
            )

        try:
            self._throttle()
            current_year = datetime.now().year
            prev_year = current_year - 1

            # Hent patenter for i år
            ytd_count, recent_titles, ai_count = self._fetch_patents_from_uspto(
                assignee, f"{current_year}-01-01", datetime.now().strftime("%Y-%m-%d"),
            )

            # Hent patenter for forrige år
            prev_count, _, _ = self._fetch_patents_from_uspto(
                assignee, f"{prev_year}-01-01", f"{prev_year}-12-31",
            )

            # Normalisér til årsbasis
            days_elapsed = (datetime.now() - datetime(current_year, 1, 1)).days or 1
            ytd_annualized = ytd_count * 365 / days_elapsed

            change_pct = 0.0
            if prev_count > 0:
                change_pct = (ytd_annualized - prev_count) / prev_count * 100

            signal = AltDataSignal.NEUTRAL
            if change_pct > 15:
                signal = AltDataSignal.BULLISH
            elif change_pct < -15:
                signal = AltDataSignal.BEARISH

            result = PatentResult(
                symbol=symbol,
                assignee=assignee,
                total_patents_ytd=ytd_count,
                total_patents_prev_year=prev_count,
                change_pct=round(change_pct, 1),
                recent_patents=recent_titles[:5],
                ai_related_count=ai_count,
                signal=signal,
            )

            self._write_patent_cache(result)
            return result

        except Exception as exc:
            logger.error(f"[altdata] Patent fejl for {symbol}: {exc}")
            return PatentResult(
                symbol=symbol, assignee=assignee,
                total_patents_ytd=0, total_patents_prev_year=0,
                change_pct=0, recent_patents=[],
            )

    def _fetch_patents_from_uspto(
        self, assignee: str, start: str, end: str,
    ) -> tuple[int, list[str], int]:
        """Hent patent-data fra USPTO API."""
        try:
            self._throttle()
            params = {
                "assigneeEntityName": assignee,
                "datePublishedFrom": start,
                "datePublishedTo": end,
                "rows": 20,
                "start": 0,
            }
            resp = _requests.get(USPTO_API_BASE, params=params, timeout=15)

            if resp.status_code != 200:
                return 0, [], 0

            data = resp.json()
            results = data.get("results", [])
            total = data.get("recordTotalQuantity", len(results))

            titles = [r.get("inventionTitle", "") for r in results if r.get("inventionTitle")]

            # Tæl AI-relaterede
            ai_keywords = {"artificial intelligence", "machine learning", "neural network",
                          "deep learning", "natural language", "computer vision"}
            ai_count = sum(
                1 for t in titles
                if any(kw in t.lower() for kw in ai_keywords)
            )

            return int(total), titles, ai_count

        except Exception as exc:
            logger.debug(f"[altdata] USPTO API fejl: {exc}")
            return 0, [], 0

    def _read_patent_cache(self, symbol: str) -> PatentResult | None:
        cache_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT assignee, patents_ytd, patents_prev, change_pct, ai_related
                   FROM patent_data
                   WHERE symbol = ? AND fetched_at >= ?
                   ORDER BY date DESC LIMIT 1""",
                (symbol, cache_cutoff),
            ).fetchone()

        if not row:
            return None

        return PatentResult(
            symbol=symbol, assignee=row[0],
            total_patents_ytd=row[1] or 0,
            total_patents_prev_year=row[2] or 0,
            change_pct=row[3] or 0,
            recent_patents=[],
            ai_related_count=row[4] or 0,
        )

    def _write_patent_cache(self, result: PatentResult) -> None:
        now = datetime.now().isoformat()
        today = datetime.now().strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO patent_data
                   (symbol, assignee, patents_ytd, patents_prev,
                    change_pct, ai_related, date, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.symbol, result.assignee,
                    result.total_patents_ytd, result.total_patents_prev_year,
                    result.change_pct, result.ai_related_count,
                    today, now,
                ),
            )

    # ── GitHub Aktivitet ──────────────────────────────────────

    def get_github_activity(
        self,
        symbol: str,
        org: str | None = None,
        use_cache: bool = True,
    ) -> GitHubActivityResult:
        """
        Hent GitHub open source aktivitet for et firma.

        Bruger GitHub REST API (uautentificeret: 60 req/time).

        Args:
            symbol: Ticker symbol.
            org: GitHub organisation (default fra DEFAULT_GITHUB_ORGS).
            use_cache: Brug cache.

        Returns:
            GitHubActivityResult.
        """
        symbol = symbol.upper()
        org = org or self._github_orgs.get(symbol, "")

        if not org:
            return GitHubActivityResult(
                symbol=symbol, org_name="unknown",
                public_repos=0, total_stars=0, total_forks=0,
                recent_commits_30d=0,
                trend_direction=TrendDirection.STABLE,
                top_repos=[],
            )

        # Cache check
        if use_cache:
            cached = self._read_github_cache(symbol)
            if cached:
                return cached

        if not _HAS_REQUESTS:
            return GitHubActivityResult(
                symbol=symbol, org_name=org,
                public_repos=0, total_stars=0, total_forks=0,
                recent_commits_30d=0,
                trend_direction=TrendDirection.STABLE,
                top_repos=[],
            )

        try:
            self._throttle()

            # Hent org info
            resp = _requests.get(
                f"{GITHUB_API_BASE}/orgs/{org}",
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=15,
            )

            if resp.status_code != 200:
                logger.warning(f"[altdata] GitHub API returnerede {resp.status_code} for {org}")
                return GitHubActivityResult(
                    symbol=symbol, org_name=org,
                    public_repos=0, total_stars=0, total_forks=0,
                    recent_commits_30d=0,
                    trend_direction=TrendDirection.STABLE,
                    top_repos=[],
                )

            org_data = resp.json()
            public_repos = org_data.get("public_repos", 0)

            # Hent top repos (sorteret efter stars)
            self._throttle()
            repos_resp = _requests.get(
                f"{GITHUB_API_BASE}/orgs/{org}/repos",
                params={"sort": "stars", "direction": "desc", "per_page": 10},
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=15,
            )

            total_stars = 0
            total_forks = 0
            top_repos: list[dict] = []

            if repos_resp.status_code == 200:
                repos = repos_resp.json()
                for repo in repos[:10]:
                    stars = repo.get("stargazers_count", 0)
                    forks = repo.get("forks_count", 0)
                    total_stars += stars
                    total_forks += forks
                    top_repos.append({
                        "name": repo.get("name", ""),
                        "stars": stars,
                        "language": repo.get("language", ""),
                    })

            # Trend (baseret på repo updates)
            direction = TrendDirection.STABLE
            if repos_resp.status_code == 200 and repos:
                recent_updates = 0
                cutoff = (datetime.now() - timedelta(days=30)).isoformat()
                for repo in repos:
                    updated = repo.get("updated_at", "")
                    if updated > cutoff:
                        recent_updates += 1
                if recent_updates >= 7:
                    direction = TrendDirection.RISING
                elif recent_updates <= 2:
                    direction = TrendDirection.FALLING

            signal = AltDataSignal.NEUTRAL
            if direction == TrendDirection.RISING and total_stars > 10000:
                signal = AltDataSignal.BULLISH
            elif direction == TrendDirection.FALLING:
                signal = AltDataSignal.BEARISH

            result = GitHubActivityResult(
                symbol=symbol,
                org_name=org,
                public_repos=public_repos,
                total_stars=total_stars,
                total_forks=total_forks,
                recent_commits_30d=0,
                trend_direction=direction,
                top_repos=top_repos,
                signal=signal,
            )

            self._write_github_cache(result)
            return result

        except Exception as exc:
            logger.error(f"[altdata] GitHub fejl for {symbol}: {exc}")
            return GitHubActivityResult(
                symbol=symbol, org_name=org,
                public_repos=0, total_stars=0, total_forks=0,
                recent_commits_30d=0,
                trend_direction=TrendDirection.STABLE,
                top_repos=[],
            )

    def _read_github_cache(self, symbol: str) -> GitHubActivityResult | None:
        cache_cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT org_name, public_repos, total_stars, total_forks,
                          recent_commits, trend_direction
                   FROM github_activity
                   WHERE symbol = ? AND fetched_at >= ?
                   ORDER BY date DESC LIMIT 1""",
                (symbol, cache_cutoff),
            ).fetchone()

        if not row:
            return None

        direction = TrendDirection(row[5]) if row[5] else TrendDirection.STABLE
        return GitHubActivityResult(
            symbol=symbol, org_name=row[0],
            public_repos=row[1] or 0, total_stars=row[2] or 0,
            total_forks=row[3] or 0, recent_commits_30d=row[4] or 0,
            trend_direction=direction, top_repos=[],
        )

    def _write_github_cache(self, result: GitHubActivityResult) -> None:
        now = datetime.now().isoformat()
        today = datetime.now().strftime("%Y-%m-%d")
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO github_activity
                   (symbol, org_name, public_repos, total_stars, total_forks,
                    recent_commits, trend_direction, date, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.symbol, result.org_name, result.public_repos,
                    result.total_stars, result.total_forks,
                    result.recent_commits_30d, result.trend_direction.value,
                    today, now,
                ),
            )

    # ── App Store Rankings ────────────────────────────────────

    def get_app_ranking(
        self,
        symbol: str,
        app_name: str | None = None,
    ) -> AppRankingResult:
        """
        Estimér app ranking via Google Trends som proxy.

        Rigtige App Store data kræver betalt API (App Annie/Sensor Tower).
        """
        symbol = symbol.upper()
        app_name = app_name or self._search_terms.get(symbol, [symbol])[0]

        # Brug Google Trends for app-relaterede søgninger
        trends = self.get_google_trends(
            symbol, terms=[f"{app_name} app", f"download {app_name}"],
        )

        # Estimér rank baseret på interest
        if trends.current_interest > 80:
            rank = 5
        elif trends.current_interest > 60:
            rank = 15
        elif trends.current_interest > 40:
            rank = 30
        elif trends.current_interest > 20:
            rank = 60
        else:
            rank = 100

        signal = AltDataSignal.NEUTRAL
        if trends.change_pct_30d > 20:
            signal = AltDataSignal.BULLISH
        elif trends.change_pct_30d < -20:
            signal = AltDataSignal.BEARISH

        return AppRankingResult(
            symbol=symbol,
            app_name=app_name,
            category_rank=rank,
            overall_rank=rank * 2,
            estimated_downloads=int(trends.current_interest * 50_000),
            trend_direction=trends.trend_direction,
            signal=signal,
        )

    # ── Aggregeret Score ──────────────────────────────────────

    def calculate_alt_data_score(
        self,
        symbol: str,
        include_github: bool = True,
        include_patents: bool = True,
    ) -> AltDataScore:
        """
        Beregn aggregeret Alt Data Score (0–100).

        Vægtning:
          - Google Trends: 30%
          - Web-trafik: 15%
          - Jobopslag: 15%
          - Patenter: 15%
          - GitHub: 15%
          - App ranking: 10%

        Args:
            symbol: Ticker symbol.
            include_github: Inkludér GitHub data (kræver API-kald).
            include_patents: Inkludér patent data (kræver API-kald).

        Returns:
            AltDataScore med aggregeret score og komponenterne.
        """
        symbol = symbol.upper()
        components: dict[str, float] = {}
        alerts: list[str] = []
        weights: dict[str, float] = {}

        # 1. Google Trends (30%)
        try:
            trends = self.get_google_trends(symbol)
            components["google_trends"] = trends.score
            weights["google_trends"] = 0.30
            if trends.spike_detected:
                alerts.append(f"📈 Spike i Google søgeinteresse for {symbol}")
            if trends.trend_direction == TrendDirection.FALLING:
                alerts.append(f"📉 Faldende søgeinteresse for {symbol}")
        except Exception as exc:
            logger.debug(f"[altdata] Trends fejl: {exc}")
            trends = None

        # 2. Web-trafik (15%)
        try:
            traffic = self.estimate_web_traffic(symbol)
            components["web_traffic"] = traffic.score
            weights["web_traffic"] = 0.15
        except Exception as exc:
            logger.debug(f"[altdata] Traffic fejl: {exc}")
            traffic = None

        # 3. Jobopslag (15%)
        try:
            jobs = self.analyze_job_postings(symbol)
            components["job_postings"] = jobs.score
            weights["job_postings"] = 0.15
            if jobs.change_pct_30d > 30:
                alerts.append(f"🟢 {symbol} øger ansættelser kraftigt (+{jobs.change_pct_30d:.0f}%)")
            elif jobs.change_pct_30d < -20:
                alerts.append(f"🔴 {symbol} skærer ned på ansættelser ({jobs.change_pct_30d:.0f}%)")
        except Exception as exc:
            logger.debug(f"[altdata] Jobs fejl: {exc}")
            jobs = None

        # 4. Patenter (15%)
        patents = None
        if include_patents:
            try:
                patents = self.get_patent_activity(symbol)
                components["patents"] = patents.score
                weights["patents"] = 0.15
                if patents.ai_related_count > 5:
                    alerts.append(
                        f"🤖 {symbol} har {patents.ai_related_count} AI-relaterede patenter i år"
                    )
            except Exception as exc:
                logger.debug(f"[altdata] Patent fejl: {exc}")

        # 5. GitHub (15%)
        github = None
        if include_github:
            try:
                github = self.get_github_activity(symbol)
                components["github"] = github.score
                weights["github"] = 0.15
                if github.total_stars > 100_000:
                    alerts.append(
                        f"⭐ {symbol} har {github.total_stars:,} GitHub stars"
                    )
            except Exception as exc:
                logger.debug(f"[altdata] GitHub fejl: {exc}")

        # 6. App ranking (10%)
        try:
            app = self.get_app_ranking(symbol)
            components["app_ranking"] = app.score
            weights["app_ranking"] = 0.10
        except Exception as exc:
            logger.debug(f"[altdata] App fejl: {exc}")
            app = None

        # Beregn weighted average
        if weights:
            total_weight = sum(weights.values())
            overall = sum(
                components.get(k, 50) * w
                for k, w in weights.items()
            ) / total_weight
        else:
            overall = 50.0

        overall = max(0, min(100, overall))

        # Signal
        if overall >= 65:
            signal = AltDataSignal.BULLISH
        elif overall <= 35:
            signal = AltDataSignal.BEARISH
        else:
            signal = AltDataSignal.NEUTRAL

        # Confidence adjustment (max ±10, da alt data er indirekte)
        conf_adj = int((overall - 50) / 5)
        conf_adj = max(-10, min(10, conf_adj))

        result = AltDataScore(
            symbol=symbol,
            overall_score=round(overall, 1),
            signal=signal,
            google_trends=trends,
            web_traffic=traffic,
            job_postings=jobs,
            patents=patents,
            github_activity=github,
            app_ranking=app,
            components=components,
            confidence_adjustment=conf_adj,
            alerts=alerts,
        )

        # Cache score
        self._write_score_cache(result)
        return result

    def _write_score_cache(self, result: AltDataScore) -> None:
        now = datetime.now().isoformat()
        today = datetime.now().strftime("%Y-%m-%d")
        import json
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO alt_data_scores
                   (symbol, overall_score, signal, components, date, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    result.symbol, result.overall_score, result.signal.value,
                    json.dumps(result.components), today, now,
                ),
            )

    # ── Batch Operations ──────────────────────────────────────

    def scan_symbols(
        self, symbols: list[str],
    ) -> dict[str, AltDataScore]:
        """Scan flere symboler for alternativ data."""
        results: dict[str, AltDataScore] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.calculate_alt_data_score(symbol)
                logger.info(
                    f"[altdata] {symbol}: score={results[symbol].overall_score:.0f} "
                    f"signal={results[symbol].signal.value}"
                )
            except Exception as exc:
                logger.error(f"[altdata] Fejl for {symbol}: {exc}")
        return results

    # ── Strategy Integration ──────────────────────────────────

    def get_confidence_adjustment(self, symbol: str) -> int:
        """
        Beregn confidence-justering baseret på alternativ data.

        Max ±10 points (alt data er indirekte og bør vægtes lavt).

        Returns:
            -10 til +10 points.
        """
        result = self.calculate_alt_data_score(symbol)
        return result.confidence_adjustment

    # ── Explain ───────────────────────────────────────────────

    def explain(self, symbol: str) -> str:
        """Forklar alternativ data i simple termer."""
        score = self.calculate_alt_data_score(symbol)
        lines = [
            f"═══ ALTERNATIV DATA RAPPORT: {symbol} ═══",
            "",
        ]

        # Google Trends
        if score.google_trends:
            gt = score.google_trends
            lines.append("🔍 GOOGLE TRENDS")
            lines.append(f"   Søgeinteresse nu: {gt.current_interest:.0f}/100")
            lines.append(f"   30-dages gns: {gt.avg_interest_30d:.0f} | 90-dages gns: {gt.avg_interest_90d:.0f}")
            lines.append(f"   Trend: {gt.trend_direction.value.upper()} ({gt.change_pct_30d:+.1f}%)")
            if gt.spike_detected:
                lines.append("   ⚡ SPIKE DETEKTERET i søgeinteresse!")
            if gt.related_rising:
                lines.append(f"   Stigende søgninger: {', '.join(gt.related_rising[:3])}")
            lines.append(f"   Score: {gt.score:.0f}/100")
            lines.append("")

        # Jobopslag
        if score.job_postings:
            jp = score.job_postings
            lines.append("💼 JOBOPSLAG-TRENDS")
            lines.append(f"   Estimerede aktive opslag: {jp.active_postings}")
            lines.append(f"   30-dages ændring: {jp.change_pct_30d:+.1f}%")
            lines.append(f"   Hiring signal: {jp.hiring_signal.value.upper()}")
            lines.append(f"   Score: {jp.score:.0f}/100")
            lines.append("")

        # Patenter
        if score.patents and score.patents.total_patents_ytd > 0:
            pt = score.patents
            lines.append("📋 PATENT-AKTIVITET")
            lines.append(f"   Patenter i år: {pt.total_patents_ytd}")
            lines.append(f"   Forrige år: {pt.total_patents_prev_year}")
            lines.append(f"   Ændring: {pt.change_pct:+.1f}%")
            if pt.ai_related_count > 0:
                lines.append(f"   🤖 AI-relaterede: {pt.ai_related_count}")
            lines.append(f"   Score: {pt.score:.0f}/100")
            lines.append("")

        # GitHub
        if score.github_activity and score.github_activity.total_stars > 0:
            gh = score.github_activity
            lines.append("🐙 GITHUB AKTIVITET")
            lines.append(f"   Organisation: {gh.org_name}")
            lines.append(f"   Public repos: {gh.public_repos}")
            lines.append(f"   Stars: {gh.total_stars:,} | Forks: {gh.total_forks:,}")
            lines.append(f"   Trend: {gh.trend_direction.value.upper()}")
            if gh.top_repos:
                top = gh.top_repos[0]
                lines.append(f"   Top repo: {top['name']} ({top['stars']:,} ⭐)")
            lines.append(f"   Score: {gh.score:.0f}/100")
            lines.append("")

        # Samlet
        lines.append("📊 SAMLET ALT DATA SCORE")
        lines.append(f"   Score: {score.overall_score:.0f}/100")
        lines.append(f"   Signal: {score.signal.value.upper()}")
        lines.append(f"   Confidence justering: {score.confidence_adjustment:+d} points")

        if score.components:
            lines.append("")
            lines.append("   Komponenter:")
            for comp, val in sorted(score.components.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(val / 5) + "░" * (20 - int(val / 5))
                lines.append(f"   {comp:<18} {bar} {val:.0f}")

        if score.alerts:
            lines.append("")
            lines.append("⚠️ ALERTS:")
            for a in score.alerts:
                lines.append(f"   {a}")

        return "\n".join(lines)

    def print_report(self, symbol: str) -> None:
        """Print alternativ data rapport."""
        print(self.explain(symbol))
