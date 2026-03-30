"""
Sentiment-modul – nyheder, NLP-analyse, event-detektion, social sentiment og kalender.

Komponenter:
  - NewsFetcher: Hent nyheder fra Finnhub, Alpha Vantage, RSS feeds
  - SentimentAnalyzer: NLP-analyse med FinBERT / keyword-fallback
  - EventDetector: Detektér corporate events (earnings, M&A, FDA, osv.)
  - SocialSentiment: Reddit-sentiment og trending tickers
  - EarningsTracker: Earnings-kalender, surprise-analyse og positionsjustering
  - MacroCalendar: Makroøkonomisk kalender (FOMC, NFP, CPI, GDP, PMI)
"""

from src.sentiment.news_fetcher import (
    NewsFetcher,
    NewsArticle,
    EarningsEvent,
    EconomicEvent,
    NewsCache,
    RateLimiter,
    RSS_FEEDS,
    SOURCE_CREDIBILITY,
    get_source_credibility,
)
from src.sentiment.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentScore,
    WeightedSentiment,
    AggregatedSentiment,
    keyword_sentiment,
    is_finbert_available,
)
from src.sentiment.event_detector import (
    EventDetector,
    DetectedEvent,
    EventType,
    EventSentiment,
    EventImpact,
)
from src.sentiment.social_sentiment import (
    SocialSentiment,
    SocialSentimentResult,
    RedditPost,
    TrendingTicker,
    extract_tickers,
    SUBREDDITS,
)
from src.sentiment.earnings_tracker import (
    EarningsTracker,
    EarningsAnalysis,
    EarningsCalendar,
    PositionAdjustment,
    SurpriseType,
    classify_surprise,
)
from src.sentiment.macro_calendar import (
    MacroCalendar,
    MacroAnalysis,
    MacroCalendarView,
    MacroEventType,
    MacroImpact,
    ExposureAdjustment,
    classify_macro_event,
    get_event_impact,
)

__all__ = [
    # news_fetcher
    "NewsFetcher",
    "NewsArticle",
    "EarningsEvent",
    "EconomicEvent",
    "NewsCache",
    "RateLimiter",
    "RSS_FEEDS",
    "SOURCE_CREDIBILITY",
    "get_source_credibility",
    # sentiment_analyzer
    "SentimentAnalyzer",
    "SentimentScore",
    "WeightedSentiment",
    "AggregatedSentiment",
    "keyword_sentiment",
    "is_finbert_available",
    # event_detector
    "EventDetector",
    "DetectedEvent",
    "EventType",
    "EventSentiment",
    "EventImpact",
    # social_sentiment
    "SocialSentiment",
    "SocialSentimentResult",
    "RedditPost",
    "TrendingTicker",
    "extract_tickers",
    "SUBREDDITS",
    # earnings_tracker
    "EarningsTracker",
    "EarningsAnalysis",
    "EarningsCalendar",
    "PositionAdjustment",
    "SurpriseType",
    "classify_surprise",
    # macro_calendar
    "MacroCalendar",
    "MacroAnalysis",
    "MacroCalendarView",
    "MacroEventType",
    "MacroImpact",
    "ExposureAdjustment",
    "classify_macro_event",
    "get_event_impact",
]
