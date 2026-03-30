"""
News Intelligence Pipeline — aggregér, analysér og prioritér nyheder.

Funktionalitet:
  - Hent nyheder fra alle kilder (Finnhub, Alpha Vantage, RSS)
  - Sentiment scoring per artikel (FinBERT + keyword)
  - Event detection (earnings, M&A, FDA, etc.)
  - Cross-impact detection: nyhed om olie → påvirker EQNR, BP, XOM
  - Prioritering: vigtigste nyheder først
  - Market mood: samlet bull/bear gauge

Bygger på:
  - src/sentiment/news_fetcher.py (eksisterer)
  - src/sentiment/sentiment_analyzer.py (eksisterer)
  - src/sentiment/event_detector.py (eksisterer)
  - src/strategy/regime.py (eksisterer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from src.sentiment.news_fetcher import (
    NewsFetcher,
    NewsArticle,
    EarningsEvent,
    EconomicEvent,
)
from src.sentiment.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentScore,
    AggregatedSentiment,
)
from src.sentiment.event_detector import (
    EventDetector,
    EventSentiment,
    EventImpact,
)


# ── Cross-impact regler ─────────────────────────────────────

CROSS_IMPACT_RULES: dict[str, dict[str, Any]] = {
    "oil_price": {
        "keywords": ["oil price", "opec", "crude", "petroleum", "brent", "wti",
                      "olie", "olieprisen", "opec+"],
        "affected": ["XOM", "CVX", "COP", "EQNR.OL", "BP.L", "TTE.PA",
                      "ENI.MI", "SHEL.L", "SLB", "HAL"],
        "sectors": ["energy"],
    },
    "interest_rate": {
        "keywords": ["interest rate", "fed rate", "rate hike", "rate cut",
                      "fomc", "federal reserve", "ecb rate", "rente",
                      "renteforhøjelse", "rentesænkning"],
        "affected": ["JPM", "BAC", "GS", "MS", "C", "WFC",
                      "DANSKE.CO", "JYSK.CO"],
        "sectors": ["financials", "reits", "utilities"],
    },
    "semiconductor": {
        "keywords": ["chip", "semiconductor", "chips act", "tsmc", "fab",
                      "wafer", "lithography", "halvleder"],
        "affected": ["ASML.AS", "NVDA", "AMD", "TSM", "INTC", "QCOM",
                      "AVGO", "MU", "LRCX", "KLAC"],
        "sectors": ["semiconductors"],
    },
    "ai_spending": {
        "keywords": ["artificial intelligence", "ai spending", "gpu demand",
                      "data center", "ai investment", "large language",
                      "machine learning", "kunstig intelligens"],
        "affected": ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD",
                      "CRM", "SNOW", "PLTR", "AI"],
        "sectors": ["ai_infrastructure"],
    },
    "glp1_obesity": {
        "keywords": ["glp-1", "glp1", "ozempic", "wegovy", "mounjaro",
                      "obesity drug", "weight loss drug", "semaglutide",
                      "tirzepatide"],
        "affected": ["NOVO-B.CO", "NVO", "LLY", "AMGN", "VKTX", "ALT"],
        "sectors": ["pharma"],
    },
    "china_policy": {
        "keywords": ["china stimulus", "china tariff", "beijing", "ccp",
                      "chinese economy", "china gdp", "pmi china",
                      "kina", "handelsaftale"],
        "affected": ["BABA", "JD", "PDD", "NIO", "XPEV", "LI",
                      "FXI", "KWEB"],
        "sectors": ["china_tech"],
    },
    "usd_strength": {
        "keywords": ["dollar strength", "dxy", "dollar index", "usd rally",
                      "dollar surge", "greenback", "dollar svaghed"],
        "affected": ["GLD", "SLV", "EEM", "FXE", "UUP"],
        "sectors": ["commodities", "emerging_markets"],
    },
    "ecb_policy": {
        "keywords": ["ecb", "european central bank", "lagarde",
                      "eurozone", "euro area", "ecb decision"],
        "affected": ["DANSKE.CO", "SAN.MC", "BNP.PA", "DBK.DE",
                      "ING.AS", "FXE"],
        "sectors": ["european_banks"],
    },
    "energy_transition": {
        "keywords": ["renewable", "solar", "wind energy", "ev mandate",
                      "green energy", "carbon tax", "climate policy",
                      "grøn omstilling", "vedvarende energi"],
        "affected": ["ENPH", "SEDG", "FSLR", "NEE", "ORSTED.CO",
                      "VESTAS.CO", "PLUG", "RUN"],
        "sectors": ["clean_energy"],
    },
    "defense_spending": {
        "keywords": ["defense spending", "military", "nato", "arms deal",
                      "defense budget", "forsvarsbudget", "oprustning"],
        "affected": ["LMT", "RTX", "NOC", "GD", "BA", "HII",
                      "SAAB-B.ST", "CHEMM.CO"],
        "sectors": ["defense"],
    },
}


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class ScoredArticle:
    """En artikel med sentiment score."""
    article: NewsArticle
    sentiment: SentimentScore
    events: list[Any] = field(default_factory=list)
    impact_score: float = 0.0  # 0-1: samlet vigtighed


@dataclass
class CrossImpact:
    """En cross-impact relation."""
    source_article: NewsArticle
    rule_name: str
    affected_symbols: list[str]
    sentiment: str  # "bullish", "bearish", "neutral"
    confidence: float


@dataclass
class SymbolIntelligence:
    """Samlet intelligence for ét symbol."""
    symbol: str
    sentiment_score: float           # -1 til 1
    sentiment_label: str             # "bullish", "bearish", "neutral"
    article_count: int
    top_articles: list[ScoredArticle]
    events: list[Any]
    cross_impacts: list[CrossImpact]
    confidence: float


@dataclass
class MarketMood:
    """Samlet markedsstemning."""
    score: float           # -1 til 1
    label: str             # "risk_on", "risk_off", "neutral"
    fear_greed: float      # 0-100 (0=extreme fear, 100=extreme greed)
    top_bullish: str       # Vigtigste bullish nyhed
    top_bearish: str       # Vigtigste bearish nyhed
    trending_themes: list[str]


@dataclass
class IntelligenceReport:
    """Fuldt intelligence report output."""
    timestamp: datetime
    top_events: list[ScoredArticle]
    per_symbol: dict[str, SymbolIntelligence]
    cross_impacts: list[CrossImpact]
    market_mood: MarketMood
    regime: str
    total_articles: int
    processing_time_ms: float


# ── News Pipeline ────────────────────────────────────────────

class NewsPipeline:
    """
    Intelligence pipeline der aggregér, analysér og prioritér nyheder.

    Brug:
        pipeline = NewsPipeline(finnhub_key="...", av_key="...")
        report = pipeline.run(symbols=["AAPL", "MSFT", "NOVO-B.CO"])
        print(report.market_mood)
        for event in report.top_events[:5]:
            print(f"  {event.article.title} ({event.sentiment.label})")
    """

    def __init__(
        self,
        finnhub_key: str = "",
        av_key: str = "",
        cache_dir: str = "data_cache",
    ) -> None:
        self._fetcher = NewsFetcher(
            finnhub_key=finnhub_key,
            av_key=av_key,
            cache_dir=cache_dir,
        )
        self._analyzer = SentimentAnalyzer()
        self._detector = EventDetector()

    def run(
        self,
        symbols: list[str],
        include_market_news: bool = True,
        days_back: int = 3,
        max_articles_per_symbol: int = 30,
    ) -> IntelligenceReport:
        """
        Kør fuld intelligence pipeline.

        Args:
            symbols: Liste af symboler at analysere.
            include_market_news: Inkludér generelle markedsnyheder.
            days_back: Antal dage tilbage at hente nyheder.
            max_articles_per_symbol: Max artikler per symbol.

        Returns:
            IntelligenceReport med alt analyseret data.
        """
        import time
        start = time.time()

        logger.info(f"[news_pipeline] Kører pipeline for {len(symbols)} symboler...")

        # 1. Hent nyheder
        all_articles: list[NewsArticle] = []
        symbol_articles: dict[str, list[NewsArticle]] = {}

        for symbol in symbols:
            try:
                articles = self._fetcher.fetch_company_news(
                    symbol, days_back=days_back
                )[:max_articles_per_symbol]
                symbol_articles[symbol] = articles
                all_articles.extend(articles)
            except Exception as exc:
                logger.warning(f"[news_pipeline] Fejl for {symbol}: {exc}")
                symbol_articles[symbol] = []

        # Generelle markedsnyheder
        if include_market_news:
            try:
                market_news = self._fetcher.fetch_market_news()[:50]
                all_articles.extend(market_news)
            except Exception as exc:
                logger.warning(f"[news_pipeline] Markedsnyheder fejl: {exc}")

        # Dedupliker
        seen: set[str] = set()
        unique_articles: list[NewsArticle] = []
        for a in all_articles:
            if a.id not in seen:
                seen.add(a.id)
                unique_articles.append(a)

        logger.info(f"[news_pipeline] {len(unique_articles)} unikke artikler hentet")

        # 2. Sentiment score per artikel
        scored: list[ScoredArticle] = []
        for article in unique_articles:
            text = f"{article.title}. {article.summary}"
            sentiment = self._analyzer.analyze_text(text)
            events = self._detector.detect_from_articles([article])

            # Impact score: kombination af sentiment styrke og kilde-vigtighed
            from src.sentiment.news_fetcher import get_source_credibility
            cred = get_source_credibility(article.source)
            age_factor = max(0.1, 1.0 - article.age_hours / 72)  # Decay over 3 dage
            impact = abs(sentiment.score) * cred * age_factor

            if events:
                # Events har ekstra impact
                for ev in events:
                    if ev.impact == EventImpact.HIGH:
                        impact *= 1.5
                    elif ev.impact == EventImpact.MEDIUM:
                        impact *= 1.2

            scored.append(ScoredArticle(
                article=article,
                sentiment=sentiment,
                events=events,
                impact_score=min(1.0, impact),
            ))

        # 3. Cross-impact detection
        cross_impacts = self._detect_cross_impacts(scored)

        # 4. Aggregér per symbol
        per_symbol: dict[str, SymbolIntelligence] = {}
        for symbol in symbols:
            sym_articles = symbol_articles.get(symbol, [])
            sym_scored = [
                s for s in scored if symbol in (s.article.symbols or [])
                or s.article in sym_articles
            ]

            # Aggregér sentiment
            if sym_scored:
                agg = self._analyzer.aggregate_sentiment(
                    symbol, [s.article for s in sym_scored]
                )
                sentiment_score = agg.score
                sentiment_label = agg.label
                confidence = agg.confidence
            else:
                sentiment_score = 0.0
                sentiment_label = "neutral"
                confidence = 0.0

            # Find cross-impacts der påvirker dette symbol
            sym_cross = [
                ci for ci in cross_impacts if symbol in ci.affected_symbols
            ]

            per_symbol[symbol] = SymbolIntelligence(
                symbol=symbol,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                article_count=len(sym_scored),
                top_articles=sorted(sym_scored, key=lambda s: s.impact_score,
                                    reverse=True)[:5],
                events=[e for s in sym_scored for e in s.events],
                cross_impacts=sym_cross,
                confidence=confidence,
            )

        # 5. Prioritér top events
        top_events = sorted(scored, key=lambda s: s.impact_score, reverse=True)[:20]

        # 6. Market mood
        market_mood = self._calculate_market_mood(scored, cross_impacts)

        # 7. Regime (actual detection via RegimeDetector + SPY data)
        regime = "unknown"
        try:
            from src.strategy.regime import RegimeDetector
            import yfinance as yf
            detector = RegimeDetector()
            spy_df = yf.download("SPY", period="6mo", interval="1d", progress=False)
            if spy_df is not None and len(spy_df) >= 50:
                result = detector.detect(spy_df)
                regime = result.regime.value  # "bull", "bear", "crash", etc.
            else:
                # Fallback to sentiment-based
                if market_mood.score > 0.3:
                    regime = "bull"
                elif market_mood.score < -0.3:
                    regime = "bear"
                else:
                    regime = "sideways"
        except Exception:
            pass

        elapsed_ms = (time.time() - start) * 1000

        report = IntelligenceReport(
            timestamp=datetime.now(),
            top_events=top_events,
            per_symbol=per_symbol,
            cross_impacts=cross_impacts,
            market_mood=market_mood,
            regime=regime,
            total_articles=len(unique_articles),
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            f"[news_pipeline] Pipeline komplet: {len(unique_articles)} artikler, "
            f"{len(cross_impacts)} cross-impacts, mood={market_mood.label} "
            f"({elapsed_ms:.0f}ms)"
        )

        return report

    def _detect_cross_impacts(
        self, scored: list[ScoredArticle]
    ) -> list[CrossImpact]:
        """
        Detektér cross-impact relationer.

        Matcher nyheder mod CROSS_IMPACT_RULES for at finde artikler
        der påvirker andre symboler end dem de direkte handler om.
        """
        impacts: list[CrossImpact] = []

        for sa in scored:
            text = f"{sa.article.title} {sa.article.summary}".lower()

            for rule_name, rule in CROSS_IMPACT_RULES.items():
                keywords = rule["keywords"]
                matched = any(kw in text for kw in keywords)

                if matched:
                    # Bestem sentiment for de påvirkede symboler
                    if sa.sentiment.score > 0.1:
                        sentiment = "bullish"
                    elif sa.sentiment.score < -0.1:
                        sentiment = "bearish"
                    else:
                        sentiment = "neutral"

                    # Filtrer: kun symboler der IKKE allerede er i artiklen
                    already_mentioned = set(
                        s.upper() for s in (sa.article.symbols or [])
                    )
                    affected = [
                        s for s in rule["affected"]
                        if s.upper() not in already_mentioned
                    ]

                    if affected:
                        impacts.append(CrossImpact(
                            source_article=sa.article,
                            rule_name=rule_name,
                            affected_symbols=affected,
                            sentiment=sentiment,
                            confidence=min(0.8, sa.impact_score),
                        ))

        return impacts

    def _calculate_market_mood(
        self,
        scored: list[ScoredArticle],
        cross_impacts: list[CrossImpact],
    ) -> MarketMood:
        """Beregn samlet markedsstemning."""
        if not scored:
            return MarketMood(
                score=0.0, label="neutral", fear_greed=50.0,
                top_bullish="", top_bearish="", trending_themes=[],
            )

        # Vægtet gennemsnit af alle sentiment scores
        total_w = 0.0
        weighted_sum = 0.0
        top_bullish_score = -1.0
        top_bearish_score = 1.0
        top_bullish_title = ""
        top_bearish_title = ""

        for sa in scored:
            w = sa.impact_score + 0.1  # Minimum weight
            total_w += w
            weighted_sum += sa.sentiment.score * w

            if sa.sentiment.score > top_bullish_score:
                top_bullish_score = sa.sentiment.score
                top_bullish_title = sa.article.title
            if sa.sentiment.score < top_bearish_score:
                top_bearish_score = sa.sentiment.score
                top_bearish_title = sa.article.title

        mood_score = weighted_sum / total_w if total_w > 0 else 0.0
        mood_score = max(-1.0, min(1.0, mood_score))

        # Fear & Greed: -1 → 0 (extreme fear), +1 → 100 (extreme greed)
        fear_greed = (mood_score + 1) * 50

        # Label
        if mood_score > 0.2:
            label = "risk_on"
        elif mood_score < -0.2:
            label = "risk_off"
        else:
            label = "neutral"

        # Trending themes
        themes: dict[str, int] = {}
        for ci in cross_impacts:
            themes[ci.rule_name] = themes.get(ci.rule_name, 0) + 1
        trending = sorted(themes, key=themes.get, reverse=True)[:5]

        return MarketMood(
            score=mood_score,
            label=label,
            fear_greed=fear_greed,
            top_bullish=top_bullish_title[:100],
            top_bearish=top_bearish_title[:100],
            trending_themes=trending,
        )

    # ── Convenience metoder ──────────────────────────────────

    def get_symbol_summary(self, symbol: str, days_back: int = 3) -> dict:
        """Hurtig enkelt-symbol intelligence summary."""
        report = self.run(symbols=[symbol], days_back=days_back)
        intel = report.per_symbol.get(symbol)
        if not intel:
            return {"symbol": symbol, "status": "no_data"}
        return {
            "symbol": symbol,
            "sentiment": round(intel.sentiment_score, 3),
            "label": intel.sentiment_label,
            "articles": intel.article_count,
            "events": len(intel.events),
            "cross_impacts": len(intel.cross_impacts),
            "top_news": [
                sa.article.title for sa in intel.top_articles[:3]
            ],
        }

    def get_market_pulse(self) -> dict:
        """Hurtig markedspuls uden symboler."""
        report = self.run(symbols=[], include_market_news=True, days_back=1)
        mood = report.market_mood
        return {
            "mood": mood.label,
            "score": round(mood.score, 3),
            "fear_greed": round(mood.fear_greed, 1),
            "themes": mood.trending_themes,
            "total_articles": report.total_articles,
        }
