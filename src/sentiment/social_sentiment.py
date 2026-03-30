"""
SocialSentiment – hent sentiment fra Reddit.

Tracker:
  - r/wallstreetbets, r/stocks, r/investing
  - Trending tickers (pludselig omtale)
  - Sentiment fra kommentarer

ADVARSEL: Social media sentiment kan være manipuleret.
Vægt det lavt i den samlede analyse (max 15-20%).
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from loguru import logger


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class RedditPost:
    """Et Reddit-opslag med metadata."""
    title: str
    body: str
    subreddit: str
    author: str
    score: int                  # Upvotes - downvotes
    num_comments: int
    url: str
    created_utc: float
    tickers_mentioned: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0

    @property
    def age_hours(self) -> float:
        return max(0, (time.time() - self.created_utc) / 3600)

    @property
    def engagement(self) -> float:
        """Engagement-score baseret på votes og kommentarer."""
        return self.score + self.num_comments * 2


@dataclass
class TrendingTicker:
    """En ticker der pludselig omtales meget."""
    symbol: str
    mention_count: int
    avg_sentiment: float          # -1 til 1
    top_subreddit: str
    posts: list[RedditPost] = field(default_factory=list)
    momentum: float = 0.0        # Stigning i omtaler vs. normalt

    @property
    def is_trending(self) -> bool:
        return self.momentum > 2.0  # 2x normal omtale


@dataclass
class SocialSentimentResult:
    """Samlet social sentiment for et symbol."""
    symbol: str
    score: float                  # -1 til 1
    label: str                    # "bullish", "bearish", "neutral"
    mention_count: int
    avg_engagement: float
    subreddit_breakdown: dict[str, float] = field(default_factory=dict)
    warning: str = ""             # Manipulation-advarsel


# ── Subreddit Config ─────────────────────────────────────────

SUBREDDITS = {
    "wallstreetbets": {"weight": 0.5, "risk": "high"},    # Mest støjende
    "stocks": {"weight": 0.8, "risk": "medium"},
    "investing": {"weight": 0.9, "risk": "low"},
    "options": {"weight": 0.6, "risk": "high"},
    "stockmarket": {"weight": 0.7, "risk": "medium"},
}

# Standard US ticker symbols (til ticker-detektion)
_TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b")

# Ord der IKKE er tickers
_NOT_TICKERS = {
    "I", "A", "AN", "THE", "AND", "OR", "NOT", "IS", "IT", "TO", "IN",
    "FOR", "ON", "AT", "BY", "UP", "SO", "IF", "OF", "DD", "PM", "AM",
    "CEO", "CFO", "CTO", "COO", "IPO", "SEC", "FDA", "GDP", "IMO",
    "YOLO", "HODL", "FOMO", "TIL", "ELI5", "TLDR", "WTF", "FYI",
    "USA", "UK", "EU", "US", "API", "ETF", "RSI", "SMA", "EMA",
    "ATH", "ATL", "OTM", "ITM", "DCA", "PE", "EPS", "PT", "TA",
    "ALL", "NEW", "LOW", "HIGH", "BIG", "OLD", "OUR", "ANY",
    "ARE", "WAS", "HAS", "HAD", "BUT", "CAN", "NOW", "HOW",
    "DID", "GET", "HIS", "HER", "WHO", "OUT", "MAY", "SAY",
    "GOOD", "JUST", "LIKE", "OVER", "ALSO", "MOST", "MANY",
    "WILL", "BEEN", "SOME", "THAN", "WHEN", "WHAT", "WITH",
    "FROM", "HAVE", "MORE", "VERY", "LONG", "SHORT", "SELL", "BUY",
    "PUTS", "CALL", "HOLD",
}


def extract_tickers(text: str) -> list[str]:
    """Udtræk ticker-symboler fra tekst."""
    matches = _TICKER_PATTERN.findall(text)
    tickers = []
    for dollar_match, word_match in matches:
        ticker = dollar_match or word_match
        if ticker and ticker not in _NOT_TICKERS and len(ticker) >= 2:
            tickers.append(ticker)
    return list(set(tickers))


# ── SocialSentiment ──────────────────────────────────────────

class SocialSentiment:
    """
    Hent og analysér sentiment fra Reddit.

    Kræver PRAW (Python Reddit API Wrapper) med credentials:
      - REDDIT_CLIENT_ID
      - REDDIT_CLIENT_SECRET
      - REDDIT_USER_AGENT

    Brug:
        social = SocialSentiment(client_id="...", client_secret="...", user_agent="...")
        result = social.get_symbol_sentiment("AAPL")
        trending = social.get_trending_tickers()
    """

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "AlphaTradingPlatform/1.0",
        subreddits: dict[str, dict] | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._subreddits = subreddits or SUBREDDITS
        self._reddit = None
        self._configured = bool(client_id and client_secret)

        if self._configured:
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )
                # Sæt til read-only
                self._reddit.read_only = True
            except ImportError:
                logger.warning("[social] praw ikke installeret")
            except Exception as exc:
                logger.error(f"[social] Reddit-forbindelse fejl: {exc}")

    @property
    def is_configured(self) -> bool:
        return self._reddit is not None

    # ── Hent Posts ────────────────────────────────────────────

    def fetch_subreddit_posts(
        self,
        subreddit_name: str,
        limit: int = 50,
        time_filter: str = "day",
    ) -> list[RedditPost]:
        """
        Hent top-posts fra en subreddit.

        Args:
            subreddit_name: Subreddit-navn (uden r/).
            limit: Max antal posts.
            time_filter: "hour", "day", "week".

        Returns:
            Liste af RedditPost.
        """
        if not self._reddit:
            return []

        try:
            subreddit = self._reddit.subreddit(subreddit_name)
            posts = []

            for submission in subreddit.hot(limit=limit):
                text = f"{submission.title} {submission.selftext or ''}"
                tickers = extract_tickers(text)

                posts.append(RedditPost(
                    title=submission.title,
                    body=(submission.selftext or "")[:500],
                    subreddit=subreddit_name,
                    author=str(submission.author) if submission.author else "[deleted]",
                    score=submission.score,
                    num_comments=submission.num_comments,
                    url=f"https://reddit.com{submission.permalink}",
                    created_utc=submission.created_utc,
                    tickers_mentioned=tickers,
                ))

            logger.debug(f"[social] r/{subreddit_name}: {len(posts)} posts hentet")
            return posts

        except Exception as exc:
            logger.error(f"[social] r/{subreddit_name} fejl: {exc}")
            return []

    def fetch_all_subreddits(
        self,
        limit: int = 50,
    ) -> list[RedditPost]:
        """Hent posts fra alle konfigurerede subreddits."""
        all_posts: list[RedditPost] = []
        for name in self._subreddits:
            posts = self.fetch_subreddit_posts(name, limit)
            all_posts.extend(posts)
        return all_posts

    # ── Symbol Sentiment ──────────────────────────────────────

    def get_symbol_sentiment(
        self,
        symbol: str,
        posts: list[RedditPost] | None = None,
    ) -> SocialSentimentResult:
        """
        Beregn social sentiment for et specifikt symbol.

        Args:
            symbol: Aktiesymbol.
            posts: Præ-hentede posts (eller hent automatisk).

        Returns:
            SocialSentimentResult med score, label og statistik.
        """
        if posts is None:
            posts = self.fetch_all_subreddits()

        # Filtrér posts der nævner symbolet
        symbol_posts = [p for p in posts if symbol in p.tickers_mentioned]

        if not symbol_posts:
            return SocialSentimentResult(
                symbol=symbol, score=0.0, label="neutral",
                mention_count=0, avg_engagement=0.0,
            )

        # Enkel sentiment baseret på post-score (upvotes)
        # Positive posts med høj score = bullish, negative = bearish
        total_engagement = sum(p.engagement for p in symbol_posts)
        avg_engagement = total_engagement / len(symbol_posts)

        # Subreddit-breakdown
        sub_scores: dict[str, list[float]] = {}
        for p in symbol_posts:
            sub = p.subreddit
            if sub not in sub_scores:
                sub_scores[sub] = []
            # Simpel: positiv score = bullish indikation
            norm_score = min(1, max(-1, p.score / 100))
            sub_scores[sub].append(norm_score)

        subreddit_breakdown = {}
        weighted_score = 0.0
        total_weight = 0.0

        for sub, scores in sub_scores.items():
            avg = sum(scores) / len(scores)
            subreddit_breakdown[sub] = avg
            weight = self._subreddits.get(sub, {}).get("weight", 0.5)
            weighted_score += avg * weight * len(scores)
            total_weight += weight * len(scores)

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        final_score = max(-1.0, min(1.0, final_score))

        if final_score > 0.15:
            label = "bullish"
        elif final_score < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        # Manipulation-advarsel
        warning = ""
        if len(symbol_posts) > 20 and avg_engagement > 500:
            warning = ("Høj aktivitet detekteret – social sentiment kan være "
                       "manipuleret. Vægt denne kilde lavt.")

        return SocialSentimentResult(
            symbol=symbol,
            score=final_score,
            label=label,
            mention_count=len(symbol_posts),
            avg_engagement=avg_engagement,
            subreddit_breakdown=subreddit_breakdown,
            warning=warning,
        )

    # ── Trending Tickers ──────────────────────────────────────

    def get_trending_tickers(
        self,
        posts: list[RedditPost] | None = None,
        min_mentions: int = 3,
        top_n: int = 10,
    ) -> list[TrendingTicker]:
        """
        Find tickers der pludselig omtales meget.

        Args:
            posts: Præ-hentede posts (eller hent automatisk).
            min_mentions: Minimum antal omtaler for at inkludere.
            top_n: Antal trending tickers at returnere.

        Returns:
            Liste af TrendingTicker, sorteret efter omtaler.
        """
        if posts is None:
            posts = self.fetch_all_subreddits()

        # Tæl ticker-omtaler
        ticker_counter: Counter[str] = Counter()
        ticker_posts: dict[str, list[RedditPost]] = {}

        for post in posts:
            for ticker in post.tickers_mentioned:
                ticker_counter[ticker] += 1
                if ticker not in ticker_posts:
                    ticker_posts[ticker] = []
                ticker_posts[ticker].append(post)

        # Find trending (over min_mentions)
        trending = []
        for ticker, count in ticker_counter.most_common(top_n * 2):
            if count < min_mentions:
                continue

            relevant_posts = ticker_posts[ticker]

            # Simpel sentiment fra engagement
            sentiments = []
            for p in relevant_posts:
                norm = min(1, max(-1, p.score / 100))
                sentiments.append(norm)

            avg_sentiment = sum(sentiments) / len(sentiments)
            top_sub = Counter(p.subreddit for p in relevant_posts).most_common(1)[0][0]

            trending.append(TrendingTicker(
                symbol=ticker,
                mention_count=count,
                avg_sentiment=avg_sentiment,
                top_subreddit=top_sub,
                posts=relevant_posts[:5],
                momentum=count / max(1, min_mentions),
            ))

        # Sortér efter omtaler
        trending.sort(key=lambda t: t.mention_count, reverse=True)
        return trending[:top_n]
