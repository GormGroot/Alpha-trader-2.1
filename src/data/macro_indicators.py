"""
Makro-indikatorer Module – ikke-standard økonomiske indikatorer.

Funktionalitet:
  - Shipping & logistik: Baltic Dry Index, container rater
  - Ejendomsmarked: housing starts, mortgage rates, Case-Shiller
  - Energiforbrug: elektricitet, gaspriser
  - Forbrugertillid: Consumer Confidence, Michigan Sentiment
  - Arbejdsmarked: initial claims, JOLTS, quit rate
  - FRED API integration: hundredvis af tidsserier
  - Recession-sandsynlighed: aggregeret indikator
  - Economic Surprise Index: data vs. forventninger

Primær datakilde: FRED (Federal Reserve Economic Data) – gratis API.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings

try:
    from fredapi import Fred
    _HAS_FRED = True
except ImportError:
    _HAS_FRED = False
    logger.warning("fredapi ikke installeret – FRED data utilgængelig")

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


# ── Konstanter ───────────────────────────────────────────────

_MIN_REQUEST_GAP = 0.25  # FRED rate limiting

# FRED Series IDs – alle gratis tilgængelige
FRED_SERIES: dict[str, dict] = {
    # Shipping & Logistik
    "baltic_dry_index": {
        "id": "DBDI",          # Discontinued – fallback til yfinance ^BDI
        "name": "Baltic Dry Index",
        "category": "shipping",
        "description": "Global tørlast shipping pris – ledende indikator for verdenshandel",
        "frequency": "daily",
        "higher_is": "bullish",
    },
    # Ejendomsmarked
    "housing_starts": {
        "id": "HOUST",
        "name": "Housing Starts",
        "category": "housing",
        "description": "Nye boligbyggerier påbegyndt (tusinder, sæsonkorrigeret)",
        "frequency": "monthly",
        "higher_is": "bullish",
    },
    "building_permits": {
        "id": "PERMIT",
        "name": "Building Permits",
        "category": "housing",
        "description": "Udstedte byggetilladelser (tusinder, sæsonkorrigeret)",
        "frequency": "monthly",
        "higher_is": "bullish",
    },
    "mortgage_rate_30y": {
        "id": "MORTGAGE30US",
        "name": "30-Year Mortgage Rate",
        "category": "housing",
        "description": "Gennemsnitlig 30-årig fast realkreditrente",
        "frequency": "weekly",
        "higher_is": "bearish",
    },
    "case_shiller": {
        "id": "CSUSHPINSA",
        "name": "Case-Shiller Home Price Index",
        "category": "housing",
        "description": "S&P/Case-Shiller US National Home Price Index",
        "frequency": "monthly",
        "higher_is": "neutral",
    },
    # Energi
    "natural_gas": {
        "id": "DHHNGSP",
        "name": "Natural Gas Price",
        "category": "energy",
        "description": "Henry Hub naturgas spotpris ($/MMBtu)",
        "frequency": "daily",
        "higher_is": "bearish",
    },
    "crude_oil_wti": {
        "id": "DCOILWTICO",
        "name": "WTI Crude Oil Price",
        "category": "energy",
        "description": "West Texas Intermediate råoliepris ($/barrel)",
        "frequency": "daily",
        "higher_is": "mixed",
    },
    # Forbrugertillid
    "consumer_confidence": {
        "id": "CSCICP03USM665S",
        "name": "Consumer Confidence Index",
        "category": "consumer",
        "description": "OECD Consumer Confidence Index for USA",
        "frequency": "monthly",
        "higher_is": "bullish",
    },
    "michigan_sentiment": {
        "id": "UMCSENT",
        "name": "U. of Michigan Consumer Sentiment",
        "category": "consumer",
        "description": "University of Michigan Consumer Sentiment Index",
        "frequency": "monthly",
        "higher_is": "bullish",
    },
    "retail_sales": {
        "id": "RSXFS",
        "name": "Retail Sales ex Food Services",
        "category": "consumer",
        "description": "Detailsalg ekskl. fødevarer (millioner USD, sæsonkorrigeret)",
        "frequency": "monthly",
        "higher_is": "bullish",
    },
    # Arbejdsmarked
    "initial_claims": {
        "id": "ICSA",
        "name": "Initial Jobless Claims",
        "category": "labor",
        "description": "Ugentlige nye ledighedsansøgninger – hurtigste arbejdsmarkedsindikator",
        "frequency": "weekly",
        "higher_is": "bearish",
    },
    "continuing_claims": {
        "id": "CCSA",
        "name": "Continuing Jobless Claims",
        "category": "labor",
        "description": "Fortsat ledighedsunderstøttelse",
        "frequency": "weekly",
        "higher_is": "bearish",
    },
    "jolts_openings": {
        "id": "JTSJOL",
        "name": "JOLTS Job Openings",
        "category": "labor",
        "description": "Antal ledige stillinger (tusinder)",
        "frequency": "monthly",
        "higher_is": "bullish",
    },
    "jolts_quits": {
        "id": "JTSQUR",
        "name": "JOLTS Quit Rate",
        "category": "labor",
        "description": "Frivillig afgangsrate – høj = folk er trygge ved at skifte job",
        "frequency": "monthly",
        "higher_is": "bullish",
    },
    # Recession-indikatorer
    "yield_spread_10y2y": {
        "id": "T10Y2Y",
        "name": "10Y-2Y Treasury Spread",
        "category": "recession",
        "description": "Forskel mellem 10-årig og 2-årig statsrente – negativ = recession-advarsel",
        "frequency": "daily",
        "higher_is": "bullish",
    },
    "leading_index": {
        "id": "USSLIND",
        "name": "Leading Economic Index",
        "category": "recession",
        "description": "Conference Board Leading Economic Index",
        "frequency": "monthly",
        "higher_is": "bullish",
    },
    "sahm_rule": {
        "id": "SAHMREALTIME",
        "name": "Sahm Rule Recession Indicator",
        "category": "recession",
        "description": "Sahm Rule: >0.5 = recession sandsynlig",
        "frequency": "monthly",
        "higher_is": "bearish",
    },
}

# Kategorier
CATEGORIES = ["shipping", "housing", "energy", "consumer", "labor", "recession"]


# ── Enums ────────────────────────────────────────────────────

class IndicatorTrend(Enum):
    """Trend for en indikator."""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"


class EconomicSignal(Enum):
    """Samlet økonomisk signal."""
    EXPANSION = "expansion"
    STABLE = "stable"
    SLOWDOWN = "slowdown"
    RECESSION_WARNING = "recession_warning"


# ── Dataclasses ──────────────────────────────────────────────

@dataclass
class MacroIndicator:
    """Én makro-indikator med seneste værdi og trend."""
    key: str                         # Internt nøgle (f.eks. "housing_starts")
    name: str                        # Menneskelæseligt navn
    category: str                    # "housing", "labor", etc.
    current_value: float
    previous_value: float
    change_pct: float                # % ændring
    trend: IndicatorTrend
    description: str = ""
    higher_is: str = "bullish"       # "bullish", "bearish", "neutral", "mixed"
    frequency: str = "monthly"
    last_updated: datetime | None = None
    history: pd.Series | None = None

    @property
    def signal(self) -> str:
        """Simpelt signal baseret på trend og retning."""
        if self.higher_is == "bullish":
            if self.trend == IndicatorTrend.IMPROVING:
                return "bullish"
            elif self.trend == IndicatorTrend.DETERIORATING:
                return "bearish"
        elif self.higher_is == "bearish":
            if self.trend == IndicatorTrend.IMPROVING:
                return "bearish"  # Stigende er bearish (f.eks. claims)
            elif self.trend == IndicatorTrend.DETERIORATING:
                return "bullish"  # Faldende er bullish
        return "neutral"

    @property
    def trend_arrow(self) -> str:
        """Trend-pil for UI."""
        if self.trend == IndicatorTrend.IMPROVING:
            return "↑" if self.higher_is != "bearish" else "↓"
        elif self.trend == IndicatorTrend.DETERIORATING:
            return "↓" if self.higher_is != "bearish" else "↑"
        return "→"

    @property
    def color(self) -> str:
        """Farve baseret på signal."""
        s = self.signal
        if s == "bullish":
            return "green"
        elif s == "bearish":
            return "red"
        return "gray"


@dataclass
class CategorySummary:
    """Sammenfatning af en kategori af indikatorer."""
    category: str
    name: str
    indicators: list[MacroIndicator]
    overall_signal: str = "neutral"  # "bullish", "bearish", "neutral"
    score: float = 50.0              # 0–100

    @property
    def bullish_count(self) -> int:
        return sum(1 for i in self.indicators if i.signal == "bullish")

    @property
    def bearish_count(self) -> int:
        return sum(1 for i in self.indicators if i.signal == "bearish")


@dataclass
class RecessionProbability:
    """Recession-sandsynlighed baseret på alle indikatorer."""
    probability: float               # 0–100%
    level: str                       # "low", "moderate", "elevated", "high"
    key_warnings: list[str]
    key_positives: list[str]
    contributing_factors: dict[str, float]  # indikator → bidrag

    @property
    def color(self) -> str:
        if self.probability >= 60:
            return "red"
        elif self.probability >= 35:
            return "orange"
        return "green"


@dataclass
class EconomicSurpriseIndex:
    """Måler om data generelt er bedre eller dårligere end forventet."""
    value: float                     # −100 til +100
    interpretation: str              # "Data overgår forventninger" etc.
    beats: int                       # Antal indikatorer der forbedres
    misses: int                      # Antal der forværres
    total: int


@dataclass
class MacroReport:
    """Samlet makro-rapport med alle indikatorer."""
    indicators: dict[str, MacroIndicator]
    categories: dict[str, CategorySummary]
    recession_probability: RecessionProbability
    surprise_index: EconomicSurpriseIndex
    overall_signal: EconomicSignal
    confidence_adjustment: int       # −15 til +15
    alerts: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


# ── Macro Indicator Tracker ──────────────────────────────────

class MacroIndicatorTracker:
    """
    Tracker for makroøkonomiske indikatorer via FRED API.

    FRED (Federal Reserve Economic Data) giver gratis adgang
    til hundredvis af økonomiske tidsserier.
    """

    def __init__(
        self,
        fred_api_key: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self._last_request_time: float = 0.0
        self._fred: Fred | None = None
        self._api_key = fred_api_key or ""

        if _HAS_FRED and self._api_key:
            try:
                self._fred = Fred(api_key=self._api_key)
            except Exception as exc:
                logger.warning(f"[macro] FRED init fejl: {exc}")

        cache_path = Path(cache_dir or settings.market_data.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_path / "macro_indicators.db"
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
                CREATE TABLE IF NOT EXISTS macro_data (
                    series_key      TEXT NOT NULL,
                    series_id       TEXT NOT NULL,
                    value           REAL,
                    date            TEXT NOT NULL,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (series_key, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_reports (
                    date            TEXT PRIMARY KEY,
                    recession_prob  REAL,
                    surprise_index  REAL,
                    overall_signal  TEXT,
                    fetched_at      TEXT NOT NULL
                )
            """)

    # ── FRED Data Hentning ────────────────────────────────────

    def _fetch_fred_series(
        self,
        series_id: str,
        lookback_years: int = 3,
    ) -> pd.Series | None:
        """Hent én tidsserie fra FRED API."""
        if not self._fred:
            return None

        try:
            self._throttle()
            start = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
            data = self._fred.get_series(series_id, observation_start=start)

            if data is None or data.empty:
                return None

            return data.dropna()

        except Exception as exc:
            logger.debug(f"[macro] FRED fejl for {series_id}: {exc}")
            return None

    def _fetch_yfinance_series(
        self,
        ticker: str,
        lookback_years: int = 3,
    ) -> pd.Series | None:
        """Hent data via yfinance som fallback."""
        if not _HAS_YF:
            return None

        try:
            self._throttle()
            t = yf.Ticker(ticker)
            hist = t.history(period=f"{lookback_years}y")
            if hist.empty:
                return None
            return hist["Close"]
        except Exception as exc:
            logger.debug(f"[macro] yfinance fejl for {ticker}: {exc}")
            return None

    # ── Indikator Hentning ────────────────────────────────────

    def get_indicator(
        self,
        key: str,
        use_cache: bool = True,
    ) -> MacroIndicator | None:
        """
        Hent én makro-indikator.

        Args:
            key: Nøgle fra FRED_SERIES (f.eks. "housing_starts").
            use_cache: Brug SQLite cache.

        Returns:
            MacroIndicator eller None.
        """
        if key not in FRED_SERIES:
            logger.warning(f"[macro] Ukendt indikator: {key}")
            return None

        meta = FRED_SERIES[key]

        # Cache check
        if use_cache:
            cached = self._read_indicator_cache(key)
            if cached is not None:
                return self._build_indicator(key, cached, meta)

        # Hent fra FRED
        series = self._fetch_fred_series(meta["id"])

        # Fallback til yfinance for alle indikatorer
        _yf_fallbacks = {
            "baltic_dry_index": "BDRY",        # Breakwave Dry Bulk Shipping ETF
            "crude_oil_wti": "CL=F",
            "natural_gas": "NG=F",
            "mortgage_rate_30y": "^TYX",       # 30Y Treasury yield (proxy)
            "yield_spread_10y2y": "^TNX",      # 10Y yield (proxy)
            "case_shiller": "ITB",             # iShares Home Construction ETF (housing proxy)
            "housing_starts": "XHB",           # SPDR S&P Homebuilders ETF (housing proxy)
            "building_permits": "ITB",         # iShares Home Construction ETF (housing proxy)
            "consumer_confidence": "XLY",      # Consumer Discretionary ETF (consumer proxy)
            "michigan_sentiment": "XRT",       # SPDR Retail ETF (consumer sentiment proxy)
            "retail_sales": "XRT",             # SPDR Retail ETF (retail proxy)
            "initial_claims": "XLI",           # Industrial Select ETF (inverse labor proxy)
            "continuing_claims": "XLI",        # Industrial Select ETF (inverse labor proxy)
            "jolts_openings": "IZRL",          # ARK Israel Innovation ETF (labor market proxy)
            "jolts_quits": "XLK",              # Technology Select ETF (job mobility proxy)
            "leading_index": "^GSPC",          # S&P 500 (leading indicator proxy)
            "sahm_rule": "^VIX",              # VIX (recession fear proxy, inverted)
        }
        if series is None and key in _yf_fallbacks:
            series = self._fetch_yfinance_series(_yf_fallbacks[key])

        if series is None or series.empty:
            return None

        # Cache
        self._write_indicator_cache(key, meta["id"], series)

        return self._build_indicator(key, series, meta)

    def _build_indicator(
        self, key: str, series: pd.Series, meta: dict,
    ) -> MacroIndicator:
        """Byg MacroIndicator fra en pandas Series."""
        current = float(series.iloc[-1])

        # Find previous (afhænger af frekvens)
        freq = meta.get("frequency", "monthly")
        if freq == "daily" and len(series) > 5:
            previous = float(series.iloc[-5])
        elif freq == "weekly" and len(series) > 4:
            previous = float(series.iloc[-4])
        elif len(series) > 1:
            previous = float(series.iloc[-2])
        else:
            previous = current

        # Change %
        change_pct = ((current - previous) / abs(previous) * 100) if previous != 0 else 0.0

        # Trend
        trend = self._determine_trend(series, meta.get("higher_is", "bullish"))

        return MacroIndicator(
            key=key,
            name=meta["name"],
            category=meta["category"],
            current_value=round(current, 2),
            previous_value=round(previous, 2),
            change_pct=round(change_pct, 2),
            trend=trend,
            description=meta.get("description", ""),
            higher_is=meta.get("higher_is", "bullish"),
            frequency=freq,
            last_updated=series.index[-1].to_pydatetime() if hasattr(series.index[-1], 'to_pydatetime') else datetime.now(),
            history=series,
        )

    @staticmethod
    def _determine_trend(series: pd.Series, higher_is: str) -> IndicatorTrend:
        """Bestem trend baseret på seneste datapunkter."""
        if len(series) < 3:
            return IndicatorTrend.STABLE

        recent = series.iloc[-3:].values
        older = series.iloc[-6:-3].values if len(series) >= 6 else series.iloc[:3].values

        recent_avg = float(np.mean(recent))
        older_avg = float(np.mean(older))

        if older_avg == 0:
            return IndicatorTrend.STABLE

        change = (recent_avg - older_avg) / abs(older_avg)

        if higher_is == "bearish":
            # For bearish indicators: stigning = deteriorating
            if change > 0.02:
                return IndicatorTrend.DETERIORATING
            elif change < -0.02:
                return IndicatorTrend.IMPROVING
        else:
            # For bullish indicators: stigning = improving
            if change > 0.02:
                return IndicatorTrend.IMPROVING
            elif change < -0.02:
                return IndicatorTrend.DETERIORATING

        return IndicatorTrend.STABLE

    # ── Cache ─────────────────────────────────────────────────

    def _read_indicator_cache(self, key: str) -> pd.Series | None:
        """Læs indikator fra cache."""
        meta = FRED_SERIES.get(key, {})
        freq = meta.get("frequency", "monthly")

        # TTL baseret på frekvens
        if freq == "daily":
            ttl_hours = 6
        elif freq == "weekly":
            ttl_hours = 24
        else:
            ttl_hours = 72

        cache_cutoff = (datetime.now() - timedelta(hours=ttl_hours)).isoformat()

        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT date, value FROM macro_data
                   WHERE series_key = ? AND fetched_at >= ?
                   ORDER BY date""",
                (key, cache_cutoff),
            ).fetchall()

        if not rows:
            return None

        dates = [pd.Timestamp(r[0]) for r in rows]
        values = [r[1] for r in rows]
        return pd.Series(values, index=dates, name=key)

    def _write_indicator_cache(
        self, key: str, series_id: str, series: pd.Series,
    ) -> None:
        """Skriv indikator til cache."""
        now = datetime.now().isoformat()
        rows = [
            (key, series_id, float(val),
             idx.isoformat() if hasattr(idx, 'isoformat') else str(idx), now)
            for idx, val in series.items()
            if pd.notna(val)
        ]
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO macro_data
                   (series_key, series_id, value, date, fetched_at)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
        logger.debug(f"[macro] Cached {len(rows)} rows for {key}")

    # ── Kategori-analyse ──────────────────────────────────────

    def get_category(self, category: str) -> CategorySummary:
        """
        Hent alle indikatorer i en kategori.

        Args:
            category: "housing", "labor", "consumer", "energy", "shipping", "recession"

        Returns:
            CategorySummary med alle indikatorer.
        """
        indicators: list[MacroIndicator] = []

        for key, meta in FRED_SERIES.items():
            if meta["category"] != category:
                continue
            ind = self.get_indicator(key)
            if ind:
                indicators.append(ind)

        # Bestem overall signal
        bullish = sum(1 for i in indicators if i.signal == "bullish")
        bearish = sum(1 for i in indicators if i.signal == "bearish")

        if bullish > bearish:
            overall = "bullish"
        elif bearish > bullish:
            overall = "bearish"
        else:
            overall = "neutral"

        # Score 0–100
        total = len(indicators) or 1
        score = 50 + (bullish - bearish) / total * 25

        category_names = {
            "shipping": "Shipping & Logistik",
            "housing": "Ejendomsmarked",
            "energy": "Energi",
            "consumer": "Forbrugertillid",
            "labor": "Arbejdsmarked",
            "recession": "Recession-indikatorer",
        }

        return CategorySummary(
            category=category,
            name=category_names.get(category, category.title()),
            indicators=indicators,
            overall_signal=overall,
            score=round(max(0, min(100, score)), 1),
        )

    # ── Recession-sandsynlighed ───────────────────────────────

    def calculate_recession_probability(self) -> RecessionProbability:
        """
        Beregn recession-sandsynlighed baseret på alle indikatorer.

        Nøgle-indikatorer:
          - Yield curve inversion (10Y-2Y < 0): +30%
          - Sahm Rule > 0.5: +25%
          - Stigende initial claims: +15%
          - Faldende housing starts: +10%
          - Faldende consumer sentiment: +10%
          - Faldende leading index: +10%
        """
        probability = 10.0  # Base (der er altid noget risiko)
        warnings: list[str] = []
        positives: list[str] = []
        factors: dict[str, float] = {}

        # 1. Yield curve
        spread = self.get_indicator("yield_spread_10y2y")
        if spread:
            if spread.current_value < 0:
                probability += 30
                warnings.append(f"Yield curve er inverteret ({spread.current_value:.2f}%) – stærk recession-indikator")
                factors["yield_curve"] = 30
            elif spread.current_value < 0.5:
                probability += 10
                warnings.append(f"Yield curve er flad ({spread.current_value:.2f}%)")
                factors["yield_curve"] = 10
            else:
                positives.append(f"Positiv yield curve ({spread.current_value:.2f}%)")
                factors["yield_curve"] = -5
                probability -= 5

        # 2. Sahm Rule
        sahm = self.get_indicator("sahm_rule")
        if sahm:
            if sahm.current_value >= 0.5:
                probability += 25
                warnings.append(f"Sahm Rule trigget ({sahm.current_value:.2f}) – recession-signal")
                factors["sahm_rule"] = 25
            elif sahm.current_value >= 0.3:
                probability += 10
                warnings.append(f"Sahm Rule nærmer sig grænsen ({sahm.current_value:.2f})")
                factors["sahm_rule"] = 10
            else:
                positives.append(f"Sahm Rule lav ({sahm.current_value:.2f})")
                factors["sahm_rule"] = -3

        # 3. Initial claims
        claims = self.get_indicator("initial_claims")
        if claims:
            if claims.trend == IndicatorTrend.DETERIORATING:
                probability += 15
                warnings.append(f"Stigende ledighedsansøgninger ({claims.change_pct:+.1f}%)")
                factors["initial_claims"] = 15
            elif claims.trend == IndicatorTrend.IMPROVING:
                positives.append(f"Faldende ledighedsansøgninger ({claims.change_pct:+.1f}%)")
                factors["initial_claims"] = -5
                probability -= 5

        # 4. Housing starts
        housing = self.get_indicator("housing_starts")
        if housing:
            if housing.trend == IndicatorTrend.DETERIORATING:
                probability += 10
                warnings.append(f"Faldende housing starts ({housing.change_pct:+.1f}%)")
                factors["housing_starts"] = 10
            elif housing.trend == IndicatorTrend.IMPROVING:
                positives.append(f"Stigende housing starts ({housing.change_pct:+.1f}%)")
                factors["housing_starts"] = -3

        # 5. Consumer sentiment
        sentiment = self.get_indicator("michigan_sentiment")
        if sentiment:
            if sentiment.trend == IndicatorTrend.DETERIORATING:
                probability += 10
                warnings.append(f"Faldende forbrugertillid ({sentiment.change_pct:+.1f}%)")
                factors["consumer_sentiment"] = 10
            elif sentiment.trend == IndicatorTrend.IMPROVING:
                positives.append(f"Stigende forbrugertillid ({sentiment.change_pct:+.1f}%)")
                factors["consumer_sentiment"] = -5

        # 6. Leading index
        leading = self.get_indicator("leading_index")
        if leading:
            if leading.trend == IndicatorTrend.DETERIORATING:
                probability += 10
                warnings.append(f"Leading Economic Index falder ({leading.change_pct:+.1f}%)")
                factors["leading_index"] = 10

        probability = max(0, min(100, probability))

        # Level
        if probability >= 60:
            level = "high"
        elif probability >= 35:
            level = "elevated"
        elif probability >= 20:
            level = "moderate"
        else:
            level = "low"

        return RecessionProbability(
            probability=round(probability, 1),
            level=level,
            key_warnings=warnings,
            key_positives=positives,
            contributing_factors=factors,
        )

    # ── Economic Surprise Index ───────────────────────────────

    def calculate_surprise_index(self) -> EconomicSurpriseIndex:
        """
        Beregn Economic Surprise Index.

        Positiv = data er generelt bedre end trend.
        Negativ = data er generelt dårligere end trend.
        """
        beats = 0
        misses = 0
        total = 0

        for key in FRED_SERIES:
            ind = self.get_indicator(key)
            if not ind:
                continue
            total += 1
            if ind.signal == "bullish":
                beats += 1
            elif ind.signal == "bearish":
                misses += 1

        if total == 0:
            return EconomicSurpriseIndex(
                value=0, interpretation="Ingen data tilgængelig",
                beats=0, misses=0, total=0,
            )

        value = (beats - misses) / total * 100

        if value > 30:
            interp = "Data overgår markant forventninger – stærk økonomi"
        elif value > 10:
            interp = "Data er generelt bedre end forventet"
        elif value > -10:
            interp = "Data er blandet – i linje med forventninger"
        elif value > -30:
            interp = "Data er generelt dårligere end forventet"
        else:
            interp = "Data skuffer markant – økonomien svækkes"

        return EconomicSurpriseIndex(
            value=round(value, 1),
            interpretation=interp,
            beats=beats,
            misses=misses,
            total=total,
        )

    # ── Full Report ───────────────────────────────────────────

    def get_macro_report(self) -> MacroReport:
        """
        Generér komplet makro-rapport med alle indikatorer.

        Returns:
            MacroReport med indikatorer, kategorier, recession-prob og surprise index.
        """
        alerts: list[str] = []

        # Hent alle indikatorer
        indicators: dict[str, MacroIndicator] = {}
        for key in FRED_SERIES:
            try:
                ind = self.get_indicator(key)
                if ind:
                    indicators[key] = ind
            except Exception as exc:
                logger.debug(f"[macro] Fejl for {key}: {exc}")

        # Kategorier
        categories: dict[str, CategorySummary] = {}
        for cat in CATEGORIES:
            try:
                categories[cat] = self.get_category(cat)
            except Exception as exc:
                logger.debug(f"[macro] Kategori fejl for {cat}: {exc}")

        # Recession probability
        try:
            recession = self.calculate_recession_probability()
            if recession.probability >= 50:
                alerts.append(
                    f"🚨 Recession-sandsynlighed: {recession.probability:.0f}% ({recession.level})"
                )
            elif recession.probability >= 30:
                alerts.append(
                    f"⚠️ Forhøjet recession-risiko: {recession.probability:.0f}%"
                )
        except Exception:
            recession = RecessionProbability(
                probability=0, level="unknown",
                key_warnings=[], key_positives=[],
                contributing_factors={},
            )

        # Surprise index
        try:
            surprise = self.calculate_surprise_index()
            if abs(surprise.value) > 30:
                emoji = "🟢" if surprise.value > 0 else "🔴"
                alerts.append(f"{emoji} Economic Surprise: {surprise.value:+.0f} – {surprise.interpretation}")
        except Exception:
            surprise = EconomicSurpriseIndex(
                value=0, interpretation="Ikke tilgængelig",
                beats=0, misses=0, total=0,
            )

        # Nøgle-advarsler fra indikatorer
        for key, ind in indicators.items():
            if key == "yield_spread_10y2y" and ind.current_value < 0:
                alerts.append("📉 Inverteret yield curve – historisk recession-indikator")
            if key == "initial_claims" and ind.trend == IndicatorTrend.DETERIORATING:
                alerts.append(f"📊 Stigende jobless claims: {ind.current_value:,.0f}")
            if key == "mortgage_rate_30y" and ind.current_value > 7:
                alerts.append(f"🏠 Høj boligrente: {ind.current_value:.2f}%")

        # Overall signal
        bullish_cats = sum(1 for c in categories.values() if c.overall_signal == "bullish")
        bearish_cats = sum(1 for c in categories.values() if c.overall_signal == "bearish")

        if recession.probability >= 50:
            overall = EconomicSignal.RECESSION_WARNING
        elif bearish_cats > bullish_cats + 1:
            overall = EconomicSignal.SLOWDOWN
        elif bullish_cats > bearish_cats:
            overall = EconomicSignal.EXPANSION
        else:
            overall = EconomicSignal.STABLE

        # Confidence adjustment
        if overall == EconomicSignal.EXPANSION:
            conf_adj = 10
        elif overall == EconomicSignal.STABLE:
            conf_adj = 0
        elif overall == EconomicSignal.SLOWDOWN:
            conf_adj = -8
        else:  # RECESSION_WARNING
            conf_adj = -15

        return MacroReport(
            indicators=indicators,
            categories=categories,
            recession_probability=recession,
            surprise_index=surprise,
            overall_signal=overall,
            confidence_adjustment=conf_adj,
            alerts=alerts,
        )

    # ── Strategy Integration ──────────────────────────────────

    def get_confidence_adjustment(self) -> int:
        """
        Beregn confidence-justering baseret på makro-data.

        Returns:
            −15 til +10 points.
        """
        report = self.get_macro_report()
        return report.confidence_adjustment

    # ── Explain ───────────────────────────────────────────────

    def explain(self) -> str:
        """Forklar makro-data i simple termer."""
        report = self.get_macro_report()
        lines = [
            "═══ MAKROØKONOMISK RAPPORT ═══",
            "",
        ]

        # Kategorier
        category_names = {
            "shipping": "🚢 SHIPPING & LOGISTIK",
            "housing": "🏠 EJENDOMSMARKED",
            "energy": "⚡ ENERGI",
            "consumer": "🛒 FORBRUGERTILLID",
            "labor": "💼 ARBEJDSMARKED",
            "recession": "📉 RECESSION-INDIKATORER",
        }

        for cat_key in CATEGORIES:
            cat = report.categories.get(cat_key)
            if not cat or not cat.indicators:
                continue

            lines.append(category_names.get(cat_key, cat_key.upper()))
            for ind in cat.indicators:
                arrow = ind.trend_arrow
                signal_emoji = {"bullish": "🟢", "bearish": "🔴"}.get(ind.signal, "⚪")
                lines.append(
                    f"   {signal_emoji} {ind.name}: {ind.current_value:,.2f} "
                    f"{arrow} ({ind.change_pct:+.1f}%)"
                )
            lines.append(f"   Signal: {cat.overall_signal.upper()} (score: {cat.score:.0f}/100)")
            lines.append("")

        # Recession
        lines.append("🎯 RECESSION-SANDSYNLIGHED")
        lines.append(f"   Sandsynlighed: {report.recession_probability.probability:.0f}%")
        lines.append(f"   Niveau: {report.recession_probability.level.upper()}")
        if report.recession_probability.key_warnings:
            lines.append("   Advarsler:")
            for w in report.recession_probability.key_warnings[:3]:
                lines.append(f"     ⚠️ {w}")
        if report.recession_probability.key_positives:
            lines.append("   Positive:")
            for p in report.recession_probability.key_positives[:3]:
                lines.append(f"     ✅ {p}")
        lines.append("")

        # Surprise index
        lines.append("📊 ECONOMIC SURPRISE INDEX")
        si = report.surprise_index
        lines.append(f"   Værdi: {si.value:+.0f}")
        lines.append(f"   {si.interpretation}")
        lines.append(f"   Beats: {si.beats} | Misses: {si.misses} | Total: {si.total}")
        lines.append("")

        # Samlet
        signal_map = {
            EconomicSignal.EXPANSION: "EKSPANSION 🟢",
            EconomicSignal.STABLE: "STABIL ⚪",
            EconomicSignal.SLOWDOWN: "OPBREMSNING ⚠️",
            EconomicSignal.RECESSION_WARNING: "RECESSION-ADVARSEL 🔴",
        }
        lines.append("📋 SAMLET VURDERING")
        lines.append(f"   Signal: {signal_map.get(report.overall_signal, 'UKENDT')}")
        lines.append(f"   Confidence justering: {report.confidence_adjustment:+d} points")

        if report.alerts:
            lines.append("")
            lines.append("⚠️ ALERTS:")
            for a in report.alerts:
                lines.append(f"   {a}")

        return "\n".join(lines)

    def print_report(self) -> None:
        """Print makro-rapport."""
        print(self.explain())
