"""
EventDetector – detektér vigtige corporate events fra nyhedstekster.

Detekterer:
  - Earnings surprises (bedre/dårre end forventet)
  - CEO/ledelsesskift
  - Opkøb og fusioner (M&A)
  - FDA-godkendelser (pharma)
  - Produktlanceringer
  - Retssager og regulatoriske tiltag
  - Aktietilbagekøb og udbytte
  - Analytiker-opjusteringer/nedjusteringer

Klassificering:
  - Sentiment: BULLISH / BEARISH / NEUTRAL
  - Påvirkning: LOW / MEDIUM / HIGH
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from src.sentiment.news_fetcher import NewsArticle, EarningsEvent


# ── Enums ────────────────────────────────────────────────────

class EventSentiment(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class EventImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EventType(Enum):
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    CEO_CHANGE = "ceo_change"
    EXECUTIVE_CHANGE = "executive_change"
    MERGER_ACQUISITION = "merger_acquisition"
    FDA_APPROVAL = "fda_approval"
    FDA_REJECTION = "fda_rejection"
    PRODUCT_LAUNCH = "product_launch"
    LAWSUIT = "lawsuit"
    REGULATORY = "regulatory"
    BUYBACK = "buyback"
    DIVIDEND = "dividend"
    UPGRADE = "analyst_upgrade"
    DOWNGRADE = "analyst_downgrade"
    BANKRUPTCY = "bankruptcy"
    PARTNERSHIP = "partnership"
    STOCK_SPLIT = "stock_split"
    INSIDER_TRADE = "insider_trade"
    UNKNOWN = "unknown"


# ── Detected Event ───────────────────────────────────────────

@dataclass
class DetectedEvent:
    """Et detekteret corporate event."""
    event_type: EventType
    sentiment: EventSentiment
    impact: EventImpact
    title: str
    summary: str
    symbol: str = ""
    source: str = ""
    published: str = ""
    confidence: float = 0.0     # 0-1 hvor sikker er detektionen
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DetectedEvent({self.event_type.value}, {self.sentiment.value}, "
            f"impact={self.impact.value}, {self.title[:50]!r})"
        )


# ── Pattern Definitions ──────────────────────────────────────

# Hvert pattern: (regex_mønster, EventType, default_sentiment, default_impact)
_EVENT_PATTERNS: list[tuple[str, EventType, EventSentiment, EventImpact]] = [
    # Earnings
    (r"\b(beats?|exceeds?|exceeded|topped?|surpass\w*)\b.*\b(estimates?|expectations?|forecasts?|consensus)\b",
     EventType.EARNINGS_BEAT, EventSentiment.BULLISH, EventImpact.HIGH),
    (r"\b(miss(?:es|ed)?|fell short|below)\b.*\b(estimates?|expectations?|forecasts?|consensus)\b",
     EventType.EARNINGS_MISS, EventSentiment.BEARISH, EventImpact.HIGH),
    (r"\b(earnings|revenue|profit)\b.*\b(surprise|beats?|tops?)\b",
     EventType.EARNINGS_BEAT, EventSentiment.BULLISH, EventImpact.HIGH),

    # CEO / Executive
    (r"\b(ceo|chief executive)\b.*\b(steps? down|resigns?|retires?|leaves?|departs?|replace\w*)\b",
     EventType.CEO_CHANGE, EventSentiment.NEUTRAL, EventImpact.HIGH),
    (r"\b(appoints?|names?|hires?)\w*\b.*\b(ceo|chief executive|president)\b",
     EventType.CEO_CHANGE, EventSentiment.NEUTRAL, EventImpact.HIGH),
    (r"\b(cfo|cto|coo)\b.*\b(steps? down|resigns?|retires?|leaves?|appoints?|names?)\b",
     EventType.EXECUTIVE_CHANGE, EventSentiment.NEUTRAL, EventImpact.MEDIUM),

    # M&A
    (r"\b(acquir\w*|acquisition|takeover|buyout|merger)\b",
     EventType.MERGER_ACQUISITION, EventSentiment.NEUTRAL, EventImpact.HIGH),
    (r"\b(merg(?:e|es|ing|er))\b.*\b(with|deal|agreement)\b",
     EventType.MERGER_ACQUISITION, EventSentiment.NEUTRAL, EventImpact.HIGH),

    # FDA
    (r"\bfda\b.*\b(approv\w*|clear\w*|green.?light\w*|authoriz\w*)\b",
     EventType.FDA_APPROVAL, EventSentiment.BULLISH, EventImpact.HIGH),
    (r"\bfda\b.*\b(reject\w*|den\w+|refus\w*|fail\w*|warning|recall\w*)\b",
     EventType.FDA_REJECTION, EventSentiment.BEARISH, EventImpact.HIGH),

    # Product launch
    (r"\b(launch\w*|unveil\w*|introduc\w*|announc\w*|releas\w*)\b.*\b(product|device|platform|service|iphone|chip)\b",
     EventType.PRODUCT_LAUNCH, EventSentiment.BULLISH, EventImpact.MEDIUM),

    # Legal / Regulatory
    (r"\b(lawsuit|sues?|sued|litigation|legal action|class.?action)\b",
     EventType.LAWSUIT, EventSentiment.BEARISH, EventImpact.MEDIUM),
    (r"\b(fined?|fines|penalty|sanctions?|violation\w*|probe|investigat\w*)\b",
     EventType.REGULATORY, EventSentiment.BEARISH, EventImpact.MEDIUM),
    (r"\b(regulat\w*|antitrust|anti.?trust|monopoly)\b.*\b(action|scrutiny|probe|fine)\b",
     EventType.REGULATORY, EventSentiment.BEARISH, EventImpact.HIGH),

    # Buyback / Dividend
    (r"\b(buyback|share repurchase|stock repurchase)\b",
     EventType.BUYBACK, EventSentiment.BULLISH, EventImpact.MEDIUM),
    (r"\b(dividend)\b.*\b(increas\w*|rais\w*|hik\w*|boost\w*|special)\b",
     EventType.DIVIDEND, EventSentiment.BULLISH, EventImpact.MEDIUM),
    (r"\b(dividend)\b.*\b(cut|slash\w*|suspend\w*|eliminat\w*|cancel\w*)\b",
     EventType.DIVIDEND, EventSentiment.BEARISH, EventImpact.HIGH),

    # Analyst ratings
    (r"\b(upgrades?|price target.*rais\w*|outperform\w*|overweight)\b",
     EventType.UPGRADE, EventSentiment.BULLISH, EventImpact.LOW),
    (r"\b(downgrades?|price target.*cut|underperform\w*|underweight)\b",
     EventType.DOWNGRADE, EventSentiment.BEARISH, EventImpact.LOW),

    # Bankruptcy
    (r"\b(bankrupt\w*|chapter 11|chapter 7|insolvency|default\w*)\b",
     EventType.BANKRUPTCY, EventSentiment.BEARISH, EventImpact.HIGH),

    # Partnership
    (r"\b(partner\w*|collaborat\w*|joint venture|alliance|deal)\b.*\b(with|sign\w*|announc\w*)\b",
     EventType.PARTNERSHIP, EventSentiment.BULLISH, EventImpact.MEDIUM),

    # Stock split
    (r"\b(stock split|share split)\b",
     EventType.STOCK_SPLIT, EventSentiment.NEUTRAL, EventImpact.LOW),

    # Insider trading
    (r"\b(insider|executive|director)\b.*\b(buys?|bought|purchas\w*|sells?|sold|dump\w*)\b.*\b(shares?|stock)\b",
     EventType.INSIDER_TRADE, EventSentiment.NEUTRAL, EventImpact.LOW),
]


# ── EventDetector ────────────────────────────────────────────

class EventDetector:
    """
    Detektér vigtige corporate events fra nyhedstekster.

    Bruger regex-patterns til at identificere event-typer og
    klassificere dem som BULLISH/BEARISH/NEUTRAL med LOW/MEDIUM/HIGH impact.

    Brug:
        detector = EventDetector()
        events = detector.detect_from_articles(articles)
        earnings_events = detector.detect_earnings_surprise(earnings_data)
    """

    def __init__(
        self,
        custom_patterns: list[tuple[str, EventType, EventSentiment, EventImpact]] | None = None,
    ) -> None:
        self._patterns = _EVENT_PATTERNS.copy()
        if custom_patterns:
            self._patterns.extend(custom_patterns)

        # Pre-compile regex
        self._compiled = [
            (re.compile(pat, re.IGNORECASE), etype, sent, imp)
            for pat, etype, sent, imp in self._patterns
        ]

    def detect_events(self, text: str) -> list[tuple[EventType, EventSentiment, EventImpact, float]]:
        """
        Detektér events i en tekst.

        Returns:
            Liste af (EventType, EventSentiment, EventImpact, confidence) tuples.
        """
        if not text:
            return []

        detected = []
        for regex, etype, sentiment, impact in self._compiled:
            match = regex.search(text)
            if match:
                # Confidence baseret på match-kvalitet
                match_len = match.end() - match.start()
                confidence = min(1.0, match_len / 30)  # Længere match = højere confidence
                detected.append((etype, sentiment, impact, confidence))

        return detected

    def detect_from_article(self, article: NewsArticle) -> list[DetectedEvent]:
        """Detektér events fra én nyhedsartikel."""
        text = f"{article.title}. {article.summary}"
        raw_events = self.detect_events(text)

        results = []
        seen_types: set[EventType] = set()

        for etype, sentiment, impact, confidence in raw_events:
            # Undgå duplikater af samme type
            if etype in seen_types:
                continue
            seen_types.add(etype)

            results.append(DetectedEvent(
                event_type=etype,
                sentiment=sentiment,
                impact=impact,
                title=article.title,
                summary=article.summary[:200],
                symbol=article.symbols[0] if article.symbols else "",
                source=article.source,
                published=article.published,
                confidence=confidence,
            ))

        return results

    def detect_from_articles(self, articles: list[NewsArticle]) -> list[DetectedEvent]:
        """Detektér events fra en liste af artikler."""
        all_events: list[DetectedEvent] = []
        for article in articles:
            events = self.detect_from_article(article)
            all_events.extend(events)

        # Sortér efter impact (HIGH først) og confidence
        impact_order = {EventImpact.HIGH: 0, EventImpact.MEDIUM: 1, EventImpact.LOW: 2}
        all_events.sort(key=lambda e: (impact_order.get(e.impact, 3), -e.confidence))

        logger.debug(f"[events] Detekteret {len(all_events)} events fra {len(articles)} artikler")
        return all_events

    # ── Earnings Surprise ─────────────────────────────────────

    def detect_earnings_surprise(
        self,
        earnings: list[EarningsEvent],
        surprise_threshold: float = 0.05,
    ) -> list[DetectedEvent]:
        """
        Detektér earnings surprises fra earnings-data.

        Args:
            earnings: Liste af EarningsEvent.
            surprise_threshold: Min. pct. afvigelse for at tælle som surprise (5%).

        Returns:
            Liste af DetectedEvent for earnings beats/misses.
        """
        events = []
        for e in earnings:
            if not e.has_reported:
                continue

            surprise = e.eps_surprise_pct
            if surprise is None:
                continue

            if surprise > surprise_threshold:
                events.append(DetectedEvent(
                    event_type=EventType.EARNINGS_BEAT,
                    sentiment=EventSentiment.BULLISH,
                    impact=EventImpact.HIGH if surprise > 0.15 else EventImpact.MEDIUM,
                    title=f"{e.symbol} overgår forventningerne med {surprise:.1%}",
                    summary=f"EPS: {e.eps_actual} vs. forventet {e.eps_estimate} ({surprise:+.1%})",
                    symbol=e.symbol,
                    published=e.date,
                    confidence=min(1.0, abs(surprise) * 5),
                    details={"eps_actual": e.eps_actual, "eps_estimate": e.eps_estimate,
                             "surprise_pct": surprise},
                ))
            elif surprise < -surprise_threshold:
                events.append(DetectedEvent(
                    event_type=EventType.EARNINGS_MISS,
                    sentiment=EventSentiment.BEARISH,
                    impact=EventImpact.HIGH if surprise < -0.15 else EventImpact.MEDIUM,
                    title=f"{e.symbol} skuffer med {surprise:.1%} under forventning",
                    summary=f"EPS: {e.eps_actual} vs. forventet {e.eps_estimate} ({surprise:+.1%})",
                    symbol=e.symbol,
                    published=e.date,
                    confidence=min(1.0, abs(surprise) * 5),
                    details={"eps_actual": e.eps_actual, "eps_estimate": e.eps_estimate,
                             "surprise_pct": surprise},
                ))

        return events

    # ── Filtrering ────────────────────────────────────────────

    @staticmethod
    def filter_by_impact(
        events: list[DetectedEvent],
        min_impact: EventImpact = EventImpact.MEDIUM,
    ) -> list[DetectedEvent]:
        """Filtrér events efter minimum impact."""
        impact_level = {EventImpact.LOW: 0, EventImpact.MEDIUM: 1, EventImpact.HIGH: 2}
        min_level = impact_level[min_impact]
        return [e for e in events if impact_level.get(e.impact, 0) >= min_level]

    @staticmethod
    def filter_by_sentiment(
        events: list[DetectedEvent],
        sentiment: EventSentiment,
    ) -> list[DetectedEvent]:
        """Filtrér events efter sentiment."""
        return [e for e in events if e.sentiment == sentiment]

    @staticmethod
    def get_symbol_events(
        events: list[DetectedEvent],
        symbol: str,
    ) -> list[DetectedEvent]:
        """Hent events for ét specifikt symbol."""
        return [e for e in events if e.symbol == symbol]

    # ── Opsummering ───────────────────────────────────────────

    def summarize_events(self, events: list[DetectedEvent]) -> dict:
        """
        Opsummér events i en dict.

        Returns:
            Dict med total, fordeling og vigtigste events.
        """
        if not events:
            return {"total": 0, "bullish": 0, "bearish": 0, "neutral": 0,
                    "high_impact": [], "types": {}}

        bullish = sum(1 for e in events if e.sentiment == EventSentiment.BULLISH)
        bearish = sum(1 for e in events if e.sentiment == EventSentiment.BEARISH)
        neutral = sum(1 for e in events if e.sentiment == EventSentiment.NEUTRAL)

        high_impact = [e for e in events if e.impact == EventImpact.HIGH]

        type_counts: dict[str, int] = {}
        for e in events:
            t = e.event_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total": len(events),
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "high_impact": [
                {"type": e.event_type.value, "title": e.title,
                 "sentiment": e.sentiment.value, "symbol": e.symbol}
                for e in high_impact[:5]
            ],
            "types": type_counts,
        }
