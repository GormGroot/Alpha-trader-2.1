"""
Tests for sentiment-modulet: news_fetcher, sentiment_analyzer, event_detector, social_sentiment.
"""

import tempfile
import time
from datetime import datetime, timedelta

import pytest

from src.sentiment.news_fetcher import (
    NewsFetcher,
    NewsArticle,
    EarningsEvent,
    EconomicEvent,
    NewsCache,
    RateLimiter,
    get_source_credibility,
    _categorize_economic_event,
    RSS_FEEDS,
    SOURCE_CREDIBILITY,
)
from src.sentiment.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentScore,
    WeightedSentiment,
    AggregatedSentiment,
    keyword_sentiment,
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
    _NOT_TICKERS,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_article(
    title: str = "Test Article",
    summary: str = "Some summary text",
    source: str = "Reuters",
    symbol: str = "AAPL",
    hours_ago: float = 2.0,
    relevance: float = 1.0,
    sentiment_api: float | None = None,
) -> NewsArticle:
    pub = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
    return NewsArticle(
        title=title,
        summary=summary,
        url=f"https://example.com/{title.replace(' ', '-')}",
        source=source,
        published=pub,
        symbols=[symbol] if symbol else [],
        sentiment_api=sentiment_api,
        relevance=relevance,
    )


def _make_reddit_post(
    title: str = "AAPL to the moon",
    body: str = "",
    subreddit: str = "wallstreetbets",
    score: int = 100,
    num_comments: int = 50,
    tickers: list[str] | None = None,
) -> RedditPost:
    return RedditPost(
        title=title,
        body=body,
        subreddit=subreddit,
        author="testuser",
        score=score,
        num_comments=num_comments,
        url="https://reddit.com/r/test/123",
        created_utc=time.time() - 3600,  # 1 time siden
        tickers_mentioned=tickers or ["AAPL"],
    )


# ════════════════════════════════════════════════════════════
# NEWS FETCHER
# ════════════════════════════════════════════════════════════

class TestNewsArticle:
    def test_age_hours(self):
        a = _make_article(hours_ago=5.0)
        assert 4.9 < a.age_hours < 5.5

    def test_id_is_hash(self):
        a = _make_article()
        assert len(a.id) == 12
        assert a.id.isalnum()

    def test_unique_ids(self):
        a1 = _make_article(title="A")
        a2 = _make_article(title="B")
        assert a1.id != a2.id

    def test_default_values(self):
        a = NewsArticle(title="T", summary="S", url="http://x", source="X", published="2026-01-01")
        assert a.category == "general"
        assert a.relevance == 1.0
        assert a.sentiment_api is None


class TestEarningsEvent:
    def test_has_reported_false(self):
        e = EarningsEvent(symbol="AAPL", date="2026-04-01", hour="amc")
        assert e.has_reported is False

    def test_has_reported_true(self):
        e = EarningsEvent(symbol="AAPL", date="2026-04-01", hour="amc",
                          eps_estimate=1.50, eps_actual=1.65)
        assert e.has_reported is True

    def test_eps_surprise_beat(self):
        e = EarningsEvent(symbol="AAPL", date="2026-04-01", hour="amc",
                          eps_estimate=1.50, eps_actual=1.65)
        assert e.eps_surprise_pct == pytest.approx(0.10)

    def test_eps_surprise_miss(self):
        e = EarningsEvent(symbol="AAPL", date="2026-04-01", hour="amc",
                          eps_estimate=1.50, eps_actual=1.35)
        assert e.eps_surprise_pct == pytest.approx(-0.10)

    def test_eps_surprise_none(self):
        e = EarningsEvent(symbol="AAPL", date="2026-04-01", hour="amc")
        assert e.eps_surprise_pct is None


class TestEconomicEvent:
    def test_categorize_interest_rate(self):
        assert _categorize_economic_event("FOMC Interest Rate Decision") == "interest_rate"

    def test_categorize_employment(self):
        assert _categorize_economic_event("Nonfarm Payrolls") == "employment"

    def test_categorize_inflation(self):
        assert _categorize_economic_event("CPI Year-over-Year") == "inflation"

    def test_categorize_gdp(self):
        assert _categorize_economic_event("GDP Growth Rate") == "gdp"

    def test_categorize_unknown(self):
        assert _categorize_economic_event("Random Event") == "other"


class TestSourceCredibility:
    def test_reuters_high(self):
        assert get_source_credibility("Reuters") >= 0.9

    def test_reddit_low(self):
        assert get_source_credibility("Reddit") <= 0.35

    def test_unknown_source(self):
        score = get_source_credibility("RandomBlog123")
        assert score == SOURCE_CREDIBILITY["Unknown"]

    def test_case_insensitive(self):
        assert get_source_credibility("reuters business") >= 0.9


class TestNewsCache:
    def test_set_and_get(self):
        cache = NewsCache(tempfile.mkdtemp())
        cache.set_cached_response("test_key", '{"data": 1}', ttl_minutes=60)
        result = cache.get_cached_response("test_key")
        assert result == '{"data": 1}'

    def test_expired_returns_none(self):
        cache = NewsCache(tempfile.mkdtemp())
        cache.set_cached_response("test_key", '{"data": 1}', ttl_minutes=0)
        # TTL 0 → udløbet med det samme
        import time
        time.sleep(0.1)
        result = cache.get_cached_response("test_key")
        assert result is None

    def test_missing_returns_none(self):
        cache = NewsCache(tempfile.mkdtemp())
        assert cache.get_cached_response("nonexistent") is None

    def test_cleanup(self):
        cache = NewsCache(tempfile.mkdtemp())
        cache.set_cached_response("old", "data", ttl_minutes=0)
        cache.set_cached_response("new", "data", ttl_minutes=60)
        import time
        time.sleep(0.1)
        deleted = cache.cleanup()
        assert deleted >= 1
        assert cache.get_cached_response("new") == "data"


class TestRateLimiter:
    def test_first_call_no_wait(self):
        rl = RateLimiter(calls_per_minute=600)  # Hurtig
        start = time.time()
        rl.wait("test")
        assert time.time() - start < 0.5


class TestNewsFetcher:
    def test_init_without_keys(self):
        fetcher = NewsFetcher()
        assert fetcher.has_finnhub is False
        assert fetcher.has_alpha_vantage is False

    def test_rss_feeds_defined(self):
        assert len(RSS_FEEDS) >= 3
        for name, url in RSS_FEEDS.items():
            assert url.startswith("http")

    def test_fetch_company_news_no_keys(self):
        """Uden API-keys returneres tom liste (ingen crash)."""
        fetcher = NewsFetcher(cache_dir=tempfile.mkdtemp())
        articles = fetcher.fetch_company_news("AAPL", include_rss=False)
        assert isinstance(articles, list)

    def test_fetch_earnings_no_client(self):
        fetcher = NewsFetcher()
        assert fetcher.fetch_earnings_calendar("AAPL") == []

    def test_fetch_economic_no_client(self):
        fetcher = NewsFetcher()
        assert fetcher.fetch_economic_calendar() == []

    def test_is_near_earnings_no_client(self):
        fetcher = NewsFetcher()
        assert fetcher.is_near_earnings("AAPL") is False


# ════════════════════════════════════════════════════════════
# SENTIMENT ANALYZER
# ════════════════════════════════════════════════════════════

class TestKeywordSentiment:
    def test_positive_text(self):
        s = keyword_sentiment("Apple beats earnings estimates, revenue surge, stock rally")
        assert s.label == "positive"
        assert s.score > 0

    def test_negative_text(self):
        s = keyword_sentiment("Company faces lawsuit, CEO resign, massive layoffs announced")
        assert s.label == "negative"
        assert s.score < 0

    def test_neutral_text(self):
        s = keyword_sentiment("The weather is nice today")
        assert s.label == "neutral"
        assert s.score == 0.0

    def test_empty_text(self):
        s = keyword_sentiment("")
        assert s.label == "neutral"

    def test_confidence_increases_with_keywords(self):
        s1 = keyword_sentiment("surge")
        s2 = keyword_sentiment("surge rally growth profit gains upgrade beat")
        assert s2.confidence >= s1.confidence

    def test_method_is_keyword(self):
        s = keyword_sentiment("test text")
        assert s.method == "keyword"


class TestSentimentScore:
    def test_is_positive(self):
        s = SentimentScore("test", "positive", 0.5, 0.9)
        assert s.is_positive is True
        assert s.is_negative is False

    def test_is_negative(self):
        s = SentimentScore("test", "negative", -0.5, 0.9)
        assert s.is_negative is True
        assert s.is_positive is False

    def test_neutral_is_neither(self):
        s = SentimentScore("test", "neutral", 0.0, 0.5)
        assert s.is_positive is False
        assert s.is_negative is False


class TestSentimentAnalyzer:
    def test_init_keyword_fallback(self):
        # Undgå at loade FinBERT i tests
        analyzer = SentimentAnalyzer(use_finbert=False)
        assert analyzer.method == "keyword"

    def test_analyze_text_positive(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        s = analyzer.analyze_text("Stock surges after beat earnings upgrade")
        assert s.score > 0

    def test_analyze_text_negative(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        s = analyzer.analyze_text("Company faces lawsuit after massive fraud investigation")
        assert s.score < 0

    def test_analyze_empty(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        s = analyzer.analyze_text("")
        assert s.score == 0.0
        assert s.method == "none"

    def test_quick_score(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        score = analyzer.quick_score("Great earnings beat profit surge")
        assert isinstance(score, float)
        assert score > 0

    def test_batch_scores(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        scores = analyzer.batch_scores(["good growth", "bad loss", "normal day"])
        assert len(scores) == 3

    def test_analyze_articles(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        articles = [
            _make_article(title="Stock surges on beat"),
            _make_article(title="Massive layoffs announced"),
        ]
        scores = analyzer.analyze_articles(articles)
        assert len(scores) == 2


class TestWeightedSentiment:
    def test_total_weight(self):
        ws = WeightedSentiment(
            article_id="abc",
            title="Test",
            source="Reuters",
            raw_score=0.5,
            source_weight=0.9,
            age_weight=0.8,
            relevance_weight=0.7,
            weighted_score=0.5 * 0.9 * 0.8 * 0.7,
        )
        assert ws.total_weight == pytest.approx(0.9 * 0.8 * 0.7)

    def test_compute_weighted(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        article = _make_article(title="Stock surges", source="Reuters", hours_ago=1)
        sentiment = analyzer.analyze_text(article.title)
        ws = analyzer.compute_weighted_sentiment(article, sentiment)
        assert ws.source_weight >= 0.9  # Reuters
        assert ws.age_weight > 0.9      # Kun 1 time gammel


class TestAggregatedSentiment:
    def test_aggregate_positive(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        articles = [
            _make_article(title="AAPL beats earnings with surge", source="Reuters"),
            _make_article(title="AAPL growth profit rally", source="CNBC"),
            _make_article(title="AAPL gains upgrade strong", source="Yahoo Finance"),
        ]
        agg = analyzer.aggregate_sentiment("AAPL", articles)
        assert agg.symbol == "AAPL"
        assert agg.score > 0
        assert agg.label == "bullish"
        assert agg.positive_count >= 2
        assert agg.article_count >= 2

    def test_aggregate_negative(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        articles = [
            _make_article(title="AAPL faces lawsuit scandal fine"),
            _make_article(title="AAPL layoffs crash decline loss"),
            _make_article(title="AAPL fraud investigation downgrade"),
        ]
        agg = analyzer.aggregate_sentiment("AAPL", articles)
        assert agg.score < 0
        assert agg.label == "bearish"

    def test_aggregate_empty(self):
        analyzer = SentimentAnalyzer(use_finbert=False)
        agg = analyzer.aggregate_sentiment("AAPL", [])
        assert agg.score == 0.0
        assert agg.label == "neutral"
        assert agg.article_count == 0

    def test_older_articles_weighted_less(self):
        analyzer = SentimentAnalyzer(use_finbert=False, age_halflife_hours=12)
        recent = _make_article(title="Stock surges gain", hours_ago=1)
        old = _make_article(title="Stock surges gain", hours_ago=48)

        agg_recent = analyzer.aggregate_sentiment("AAPL", [recent])
        agg_old = analyzer.aggregate_sentiment("AAPL", [old])

        # Nyere bør have højere detaljeret score
        if agg_recent.details and agg_old.details:
            assert agg_recent.details[0].age_weight > agg_old.details[0].age_weight

    def test_sentiment_ratio(self):
        agg = AggregatedSentiment(
            symbol="AAPL", score=0.3, label="bullish",
            article_count=10, positive_count=7,
            negative_count=2, neutral_count=1,
        )
        assert agg.sentiment_ratio == pytest.approx(3.5)

    def test_sentiment_ratio_no_negatives(self):
        agg = AggregatedSentiment(
            symbol="AAPL", score=0.5, label="bullish",
            article_count=5, positive_count=5,
            negative_count=0, neutral_count=0,
        )
        assert agg.sentiment_ratio == float("inf")


# ════════════════════════════════════════════════════════════
# EVENT DETECTOR
# ════════════════════════════════════════════════════════════

class TestEventDetector:
    def test_detect_earnings_beat(self):
        detector = EventDetector()
        events = detector.detect_events("Apple beats earnings estimates by a wide margin")
        assert any(e[0] == EventType.EARNINGS_BEAT for e in events)

    def test_detect_earnings_miss(self):
        detector = EventDetector()
        events = detector.detect_events("Company misses earnings expectations significantly")
        assert any(e[0] == EventType.EARNINGS_MISS for e in events)

    def test_detect_ceo_change(self):
        detector = EventDetector()
        events = detector.detect_events("CEO steps down after 10 years leading the company")
        assert any(e[0] == EventType.CEO_CHANGE for e in events)

    def test_detect_merger(self):
        detector = EventDetector()
        events = detector.detect_events("Company announces acquisition of rival firm for $5 billion")
        assert any(e[0] == EventType.MERGER_ACQUISITION for e in events)

    def test_detect_fda_approval(self):
        detector = EventDetector()
        events = detector.detect_events("FDA approves new cancer drug from Pfizer")
        assert any(e[0] == EventType.FDA_APPROVAL for e in events)

    def test_detect_fda_rejection(self):
        detector = EventDetector()
        events = detector.detect_events("FDA rejects drug application citing safety concerns")
        assert any(e[0] == EventType.FDA_REJECTION for e in events)

    def test_detect_lawsuit(self):
        detector = EventDetector()
        events = detector.detect_events("Class action lawsuit filed against the company")
        assert any(e[0] == EventType.LAWSUIT for e in events)

    def test_detect_buyback(self):
        detector = EventDetector()
        events = detector.detect_events("Company announces $10B share repurchase program")
        assert any(e[0] == EventType.BUYBACK for e in events)

    def test_detect_dividend_increase(self):
        detector = EventDetector()
        events = detector.detect_events("Board approves dividend increase of 10%")
        types = [e[0] for e in events]
        assert EventType.DIVIDEND in types

    def test_detect_upgrade(self):
        detector = EventDetector()
        events = detector.detect_events("Goldman Sachs upgrades Apple, price target raised")
        types = [e[0] for e in events]
        assert EventType.UPGRADE in types

    def test_detect_downgrade(self):
        detector = EventDetector()
        events = detector.detect_events("Morgan Stanley downgrades stock to underperform")
        types = [e[0] for e in events]
        assert EventType.DOWNGRADE in types

    def test_detect_bankruptcy(self):
        detector = EventDetector()
        events = detector.detect_events("Company files for Chapter 11 bankruptcy")
        types = [e[0] for e in events]
        assert EventType.BANKRUPTCY in types

    def test_no_events_in_neutral_text(self):
        detector = EventDetector()
        events = detector.detect_events("The weather is nice today")
        assert len(events) == 0

    def test_empty_text(self):
        detector = EventDetector()
        assert detector.detect_events("") == []

    def test_multiple_events_in_text(self):
        detector = EventDetector()
        text = "CEO steps down after FDA rejects drug and company files lawsuit"
        events = detector.detect_events(text)
        types = {e[0] for e in events}
        assert len(types) >= 2


class TestEventDetectorArticles:
    def test_detect_from_article(self):
        detector = EventDetector()
        article = _make_article(
            title="Apple beats earnings estimates",
            summary="Q4 revenue topped Wall Street consensus by 5%",
        )
        events = detector.detect_from_article(article)
        assert len(events) >= 1
        assert events[0].event_type == EventType.EARNINGS_BEAT
        assert events[0].sentiment == EventSentiment.BULLISH

    def test_detect_from_articles(self):
        detector = EventDetector()
        articles = [
            _make_article(title="FDA approves Pfizer drug"),
            _make_article(title="CEO resigns from tech company"),
            _make_article(title="Normal weather report"),
        ]
        events = detector.detect_from_articles(articles)
        assert len(events) >= 2

    def test_no_duplicate_types_per_article(self):
        detector = EventDetector()
        article = _make_article(
            title="Company beats earnings, beats all estimates, topped consensus"
        )
        events = detector.detect_from_article(article)
        types = [e.event_type for e in events]
        assert len(types) == len(set(types))


class TestEarningsSurprise:
    def test_detect_beat(self):
        detector = EventDetector()
        earnings = [
            EarningsEvent("AAPL", "2026-04-01", "amc", eps_estimate=1.50, eps_actual=1.80),
        ]
        events = detector.detect_earnings_surprise(earnings)
        assert len(events) == 1
        assert events[0].event_type == EventType.EARNINGS_BEAT
        assert events[0].sentiment == EventSentiment.BULLISH

    def test_detect_miss(self):
        detector = EventDetector()
        earnings = [
            EarningsEvent("AAPL", "2026-04-01", "amc", eps_estimate=1.50, eps_actual=1.20),
        ]
        events = detector.detect_earnings_surprise(earnings)
        assert len(events) == 1
        assert events[0].event_type == EventType.EARNINGS_MISS
        assert events[0].sentiment == EventSentiment.BEARISH

    def test_no_surprise_in_line(self):
        detector = EventDetector()
        earnings = [
            EarningsEvent("AAPL", "2026-04-01", "amc", eps_estimate=1.50, eps_actual=1.52),
        ]
        events = detector.detect_earnings_surprise(earnings, surprise_threshold=0.05)
        assert len(events) == 0  # Kun 1.3% surprise

    def test_unreported_ignored(self):
        detector = EventDetector()
        earnings = [
            EarningsEvent("AAPL", "2026-04-01", "amc", eps_estimate=1.50),
        ]
        events = detector.detect_earnings_surprise(earnings)
        assert len(events) == 0


class TestEventFiltering:
    def test_filter_by_impact(self):
        events = [
            DetectedEvent(EventType.EARNINGS_BEAT, EventSentiment.BULLISH, EventImpact.HIGH, "A", ""),
            DetectedEvent(EventType.UPGRADE, EventSentiment.BULLISH, EventImpact.LOW, "B", ""),
            DetectedEvent(EventType.CEO_CHANGE, EventSentiment.NEUTRAL, EventImpact.MEDIUM, "C", ""),
        ]
        high = EventDetector.filter_by_impact(events, EventImpact.HIGH)
        assert len(high) == 1
        medium_plus = EventDetector.filter_by_impact(events, EventImpact.MEDIUM)
        assert len(medium_plus) == 2

    def test_filter_by_sentiment(self):
        events = [
            DetectedEvent(EventType.EARNINGS_BEAT, EventSentiment.BULLISH, EventImpact.HIGH, "A", ""),
            DetectedEvent(EventType.LAWSUIT, EventSentiment.BEARISH, EventImpact.MEDIUM, "B", ""),
        ]
        bullish = EventDetector.filter_by_sentiment(events, EventSentiment.BULLISH)
        assert len(bullish) == 1

    def test_summarize_events(self):
        detector = EventDetector()
        events = [
            DetectedEvent(EventType.EARNINGS_BEAT, EventSentiment.BULLISH, EventImpact.HIGH, "A", "", "AAPL"),
            DetectedEvent(EventType.LAWSUIT, EventSentiment.BEARISH, EventImpact.MEDIUM, "B", "", "AAPL"),
            DetectedEvent(EventType.CEO_CHANGE, EventSentiment.NEUTRAL, EventImpact.HIGH, "C", "", "MSFT"),
        ]
        summary = detector.summarize_events(events)
        assert summary["total"] == 3
        assert summary["bullish"] == 1
        assert summary["bearish"] == 1
        assert summary["neutral"] == 1
        assert len(summary["high_impact"]) == 2

    def test_summarize_empty(self):
        detector = EventDetector()
        summary = detector.summarize_events([])
        assert summary["total"] == 0


# ════════════════════════════════════════════════════════════
# SOCIAL SENTIMENT
# ════════════════════════════════════════════════════════════

class TestExtractTickers:
    def test_dollar_sign(self):
        tickers = extract_tickers("Just bought $AAPL and $TSLA!")
        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_plain_ticker(self):
        tickers = extract_tickers("MSFT is looking great today")
        assert "MSFT" in tickers

    def test_filters_common_words(self):
        tickers = extract_tickers("I think THE CEO IS good for ALL")
        for word in ["THE", "CEO", "IS", "ALL"]:
            assert word not in tickers

    def test_empty_text(self):
        assert extract_tickers("") == []

    def test_no_single_letter(self):
        tickers = extract_tickers("I bought A share")
        assert "I" not in tickers
        assert "A" not in tickers

    def test_filters_reddit_slang(self):
        tickers = extract_tickers("YOLO into HODL mode FOMO")
        assert "YOLO" not in tickers
        assert "HODL" not in tickers
        assert "FOMO" not in tickers


class TestRedditPost:
    def test_age_hours(self):
        post = _make_reddit_post()
        assert 0.9 < post.age_hours < 1.5  # ~1 time

    def test_engagement(self):
        post = _make_reddit_post(score=100, num_comments=50)
        assert post.engagement == 200  # 100 + 50*2


class TestTrendingTicker:
    def test_is_trending(self):
        t = TrendingTicker(symbol="AAPL", mention_count=10, avg_sentiment=0.5,
                           top_subreddit="wsb", momentum=3.0)
        assert t.is_trending is True

    def test_not_trending(self):
        t = TrendingTicker(symbol="AAPL", mention_count=2, avg_sentiment=0.0,
                           top_subreddit="wsb", momentum=1.0)
        assert t.is_trending is False


class TestSocialSentiment:
    def test_init_unconfigured(self):
        social = SocialSentiment()
        assert social.is_configured is False

    def test_unconfigured_returns_empty(self):
        social = SocialSentiment()
        posts = social.fetch_subreddit_posts("stocks")
        assert posts == []

    def test_symbol_sentiment_no_posts(self):
        social = SocialSentiment()
        result = social.get_symbol_sentiment("AAPL", posts=[])
        assert result.score == 0.0
        assert result.mention_count == 0

    def test_symbol_sentiment_with_posts(self):
        social = SocialSentiment()
        posts = [
            _make_reddit_post(title="AAPL is great", score=200, tickers=["AAPL"]),
            _make_reddit_post(title="AAPL going up", score=150, tickers=["AAPL"]),
            _make_reddit_post(title="TSLA discussion", score=50, tickers=["TSLA"]),
        ]
        result = social.get_symbol_sentiment("AAPL", posts=posts)
        assert result.mention_count == 2
        assert result.score > 0  # Positive scores

    def test_trending_tickers(self):
        social = SocialSentiment()
        posts = [
            _make_reddit_post(title="AAPL buy", tickers=["AAPL"], score=100),
            _make_reddit_post(title="AAPL moon", tickers=["AAPL"], score=200),
            _make_reddit_post(title="AAPL up", tickers=["AAPL"], score=150),
            _make_reddit_post(title="TSLA ok", tickers=["TSLA"], score=50),
        ]
        trending = social.get_trending_tickers(posts=posts, min_mentions=2)
        assert len(trending) >= 1
        assert trending[0].symbol == "AAPL"
        assert trending[0].mention_count == 3

    def test_social_sentiment_result(self):
        result = SocialSentimentResult(
            symbol="AAPL", score=0.3, label="bullish",
            mention_count=15, avg_engagement=200,
        )
        assert result.symbol == "AAPL"
        assert result.label == "bullish"

    def test_manipulation_warning(self):
        social = SocialSentiment()
        posts = [
            _make_reddit_post(tickers=["AAPL"], score=1000, num_comments=200)
            for _ in range(25)
        ]
        result = social.get_symbol_sentiment("AAPL", posts=posts)
        assert result.warning != ""


# ════════════════════════════════════════════════════════════
# INTEGRATION
# ════════════════════════════════════════════════════════════

class TestModuleImports:
    def test_import_all(self):
        from src.sentiment import (
            NewsFetcher, SentimentAnalyzer, EventDetector, SocialSentiment,
            NewsArticle, SentimentScore, DetectedEvent, RedditPost,
            EventType, EventSentiment, EventImpact,
            extract_tickers, keyword_sentiment,
        )
        assert NewsFetcher is not None
        assert SentimentAnalyzer is not None
        assert EventDetector is not None
        assert SocialSentiment is not None
