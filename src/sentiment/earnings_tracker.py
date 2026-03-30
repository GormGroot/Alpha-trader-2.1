"""
EarningsTracker – intelligent earnings-kalender og analyse.

Features:
  - Hent earnings-datoer for alle aktier i universet
  - Vis EPS-estimater vs. sidste år
  - Beregn earnings surprise (actual vs expected)
  - Historisk analyse: typisk kursreaktion på earnings
  - Handelsregel: reducér positionsstørrelse 50% dagen før earnings

Bruges af TradingBot til automatisk risikostyring omkring earnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from src.sentiment.news_fetcher import EarningsEvent, NewsFetcher, NewsCache


# ── Enums & Dataklasser ──────────────────────────────────────

class SurpriseType(Enum):
    BIG_BEAT = "big_beat"        # > +10%
    BEAT = "beat"                # +5% til +10%
    SMALL_BEAT = "small_beat"    # +0% til +5%
    INLINE = "inline"            # -2% til +2%
    SMALL_MISS = "small_miss"    # -5% til 0%
    MISS = "miss"                # -10% til -5%
    BIG_MISS = "big_miss"        # < -10%


def classify_surprise(surprise_pct: float) -> SurpriseType:
    """Klassificér earnings surprise."""
    if surprise_pct > 0.10:
        return SurpriseType.BIG_BEAT
    elif surprise_pct > 0.05:
        return SurpriseType.BEAT
    elif surprise_pct > 0.02:
        return SurpriseType.SMALL_BEAT
    elif surprise_pct > -0.02:
        return SurpriseType.INLINE
    elif surprise_pct > -0.05:
        return SurpriseType.SMALL_MISS
    elif surprise_pct > -0.10:
        return SurpriseType.MISS
    else:
        return SurpriseType.BIG_MISS


@dataclass
class EarningsAnalysis:
    """Analyse af ét earnings-event."""
    symbol: str
    date: str
    hour: str
    eps_estimate: float | None = None
    eps_actual: float | None = None
    eps_previous: float | None = None  # Sidste kvartal eller år
    revenue_estimate: float | None = None
    revenue_actual: float | None = None
    surprise_pct: float | None = None
    surprise_type: SurpriseType | None = None
    is_upcoming: bool = True
    days_until: int = 0

    # Historisk reaktion
    avg_move_on_beat: float = 0.0     # Gennemsnitlig kursbevægelse ved beat
    avg_move_on_miss: float = 0.0     # Gennemsnitlig kursbevægelse ved miss
    historical_beat_rate: float = 0.0  # Andel gange firmaet har slået

    @property
    def has_reported(self) -> bool:
        return self.eps_actual is not None

    @property
    def yoy_growth(self) -> float | None:
        """Year-over-year EPS-vækst."""
        if self.eps_actual is None or self.eps_previous is None or self.eps_previous == 0:
            return None
        return (self.eps_actual - self.eps_previous) / abs(self.eps_previous)


@dataclass
class EarningsCalendar:
    """Samlet earnings-kalender."""
    upcoming: list[EarningsAnalysis] = field(default_factory=list)
    recent: list[EarningsAnalysis] = field(default_factory=list)
    today: list[EarningsAnalysis] = field(default_factory=list)
    next_event: EarningsAnalysis | None = None
    days_to_next: int = 999

    @property
    def has_earnings_today(self) -> bool:
        return len(self.today) > 0

    @property
    def symbols_reporting_soon(self) -> list[str]:
        """Symboler der rapporterer indenfor 3 dage."""
        return [e.symbol for e in self.upcoming if e.days_until <= 3]


@dataclass
class PositionAdjustment:
    """Anbefalet positions-justering pga. earnings."""
    symbol: str
    reason: str
    reduction_pct: float         # 0-1, f.eks. 0.5 = reducér 50%
    days_until_earnings: int
    earnings_date: str
    earnings_hour: str

    def __repr__(self) -> str:
        return (
            f"PositionAdjustment({self.symbol}: reducér {self.reduction_pct:.0%}, "
            f"earnings om {self.days_until_earnings}d)"
        )


# ── EarningsTracker ──────────────────────────────────────────

class EarningsTracker:
    """
    Intelligent earnings-kalender med handelsregler.

    Tracker earnings-datoer, beregner surprises, analyserer
    historisk kursreaktion og anbefaler positionsjusteringer.

    Brug:
        tracker = EarningsTracker(fetcher)
        calendar = tracker.get_calendar(["AAPL", "MSFT", "GOOGL"])
        adjustments = tracker.get_position_adjustments(["AAPL", "MSFT"])
    """

    def __init__(
        self,
        fetcher: NewsFetcher | None = None,
        pre_earnings_reduction: float = 0.50,
        pre_earnings_days: int = 1,
        cache_dir: str = "data_cache",
    ) -> None:
        """
        Args:
            fetcher: NewsFetcher til at hente earnings-data.
            pre_earnings_reduction: Positionsreduktion før earnings (0.50 = 50%).
            pre_earnings_days: Dage før earnings med reduceret position.
            cache_dir: Mappe til cache.
        """
        self._fetcher = fetcher or NewsFetcher(cache_dir=cache_dir)
        self._reduction = pre_earnings_reduction
        self._pre_days = pre_earnings_days
        self._cache = NewsCache(cache_dir)

    # ── Kalender ──────────────────────────────────────────────

    def get_calendar(
        self,
        symbols: list[str],
        days_ahead: int = 30,
        days_back: int = 7,
    ) -> EarningsCalendar:
        """
        Hent earnings-kalender for en liste symboler.

        Args:
            symbols: Liste af aktiesymboler.
            days_ahead: Antal dage fremad at kigge.
            days_back: Antal dage bagud (nylige earnings).

        Returns:
            EarningsCalendar med upcoming, recent og today.
        """
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        from_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        all_analyses: list[EarningsAnalysis] = []

        for symbol in symbols:
            events = self._fetcher.fetch_earnings_calendar(
                symbol=symbol, from_date=from_date, to_date=to_date,
            )
            for event in events:
                analysis = self._analyze_event(event, today_str)
                all_analyses.append(analysis)

        # Opdel i upcoming, today og recent
        upcoming = sorted(
            [a for a in all_analyses if not a.has_reported and a.days_until > 0],
            key=lambda a: a.days_until,
        )
        today = [a for a in all_analyses if a.days_until == 0 and not a.has_reported]
        recent = sorted(
            [a for a in all_analyses if a.has_reported],
            key=lambda a: a.date, reverse=True,
        )

        next_event = upcoming[0] if upcoming else None
        days_to_next = next_event.days_until if next_event else 999

        return EarningsCalendar(
            upcoming=upcoming,
            recent=recent,
            today=today,
            next_event=next_event,
            days_to_next=days_to_next,
        )

    def _analyze_event(self, event: EarningsEvent, today_str: str) -> EarningsAnalysis:
        """Konvertér EarningsEvent til EarningsAnalysis med beregninger."""
        try:
            event_date = datetime.strptime(event.date, "%Y-%m-%d")
            today = datetime.strptime(today_str, "%Y-%m-%d")
            days_until = (event_date - today).days
        except (ValueError, TypeError):
            days_until = 999

        surprise_pct = event.eps_surprise_pct
        surprise_type = classify_surprise(surprise_pct) if surprise_pct is not None else None

        return EarningsAnalysis(
            symbol=event.symbol,
            date=event.date,
            hour=event.hour,
            eps_estimate=event.eps_estimate,
            eps_actual=event.eps_actual,
            revenue_estimate=event.revenue_estimate,
            revenue_actual=event.revenue_actual,
            surprise_pct=surprise_pct,
            surprise_type=surprise_type,
            is_upcoming=not event.has_reported,
            days_until=max(0, days_until),
        )

    # ── Historisk Analyse ─────────────────────────────────────

    def analyze_historical_reaction(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        earnings_dates: list[str],
        window_days: int = 5,
    ) -> dict:
        """
        Analysér historisk kursreaktion på earnings.

        Args:
            symbol: Aktiesymbol.
            price_data: OHLCV DataFrame med dato-index.
            earnings_dates: Liste af earnings-datoer (YYYY-MM-DD).
            window_days: Antal dage efter earnings at måle.

        Returns:
            Dict med avg_move, beat_rate, moves_list, osv.
        """
        if price_data is None or price_data.empty or not earnings_dates:
            return {
                "symbol": symbol,
                "earnings_count": 0,
                "avg_move_pct": 0.0,
                "avg_abs_move_pct": 0.0,
                "max_positive_move": 0.0,
                "max_negative_move": 0.0,
                "positive_reaction_pct": 0.0,
                "moves": [],
            }

        moves = []
        close = price_data["Close"]

        for date_str in earnings_dates:
            try:
                earnings_date = pd.Timestamp(date_str)
                # Find nærmeste handelsdag
                idx = close.index.searchsorted(earnings_date)
                if idx >= len(close) or idx + window_days >= len(close):
                    continue

                price_before = float(close.iloc[idx])
                price_after = float(close.iloc[min(idx + window_days, len(close) - 1)])

                if price_before > 0:
                    move_pct = (price_after - price_before) / price_before
                    moves.append(move_pct)
            except (ValueError, IndexError, KeyError):
                continue

        if not moves:
            return {
                "symbol": symbol,
                "earnings_count": 0,
                "avg_move_pct": 0.0,
                "avg_abs_move_pct": 0.0,
                "max_positive_move": 0.0,
                "max_negative_move": 0.0,
                "positive_reaction_pct": 0.0,
                "moves": [],
            }

        moves_arr = np.array(moves)
        pos_pct = float(np.mean(moves_arr > 0))

        return {
            "symbol": symbol,
            "earnings_count": len(moves),
            "avg_move_pct": float(np.mean(moves_arr)),
            "avg_abs_move_pct": float(np.mean(np.abs(moves_arr))),
            "max_positive_move": float(np.max(moves_arr)),
            "max_negative_move": float(np.min(moves_arr)),
            "positive_reaction_pct": pos_pct,
            "moves": [float(m) for m in moves],
        }

    # ── Position Adjustments ──────────────────────────────────

    def get_position_adjustments(
        self,
        symbols: list[str],
        days_ahead: int = 5,
    ) -> list[PositionAdjustment]:
        """
        Beregn anbefalede positionsjusteringer baseret på earnings.

        Regel: Reducér position med pre_earnings_reduction (default 50%)
        indenfor pre_earnings_days (default 1 dag) af earnings.

        Args:
            symbols: Aktier at tjekke.
            days_ahead: Antal dage fremad.

        Returns:
            Liste af PositionAdjustment for aktier nær earnings.
        """
        calendar = self.get_calendar(symbols, days_ahead=days_ahead, days_back=0)
        adjustments = []

        for event in calendar.upcoming + calendar.today:
            if event.days_until <= self._pre_days:
                adjustments.append(PositionAdjustment(
                    symbol=event.symbol,
                    reason=(
                        f"Earnings {event.date} ({event.hour.upper()}) – "
                        f"reducér positionsstørrelse {self._reduction:.0%}"
                    ),
                    reduction_pct=self._reduction,
                    days_until_earnings=event.days_until,
                    earnings_date=event.date,
                    earnings_hour=event.hour,
                ))

        return adjustments

    def should_reduce_position(self, symbol: str) -> tuple[bool, float, str]:
        """
        Tjek om en position bør reduceres pga. nær earnings.

        Returns:
            (should_reduce, reduction_pct, reason)
        """
        adjustments = self.get_position_adjustments([symbol])
        if adjustments:
            adj = adjustments[0]
            return True, adj.reduction_pct, adj.reason
        return False, 0.0, ""

    # ── Earnings Surprise Analyse ─────────────────────────────

    def analyze_surprises(
        self,
        events: list[EarningsEvent],
    ) -> dict:
        """
        Analysér en samling af earnings events for surprise-mønstre.

        Returns:
            Dict med beat_rate, avg_surprise, distribution, osv.
        """
        reported = [e for e in events if e.has_reported and e.eps_surprise_pct is not None]

        if not reported:
            return {
                "total_reported": 0,
                "beat_rate": 0.0,
                "miss_rate": 0.0,
                "avg_surprise_pct": 0.0,
                "median_surprise_pct": 0.0,
                "distribution": {},
            }

        surprises = [e.eps_surprise_pct for e in reported]
        surprises_arr = np.array(surprises)

        beats = sum(1 for s in surprises if s > 0.02)
        misses = sum(1 for s in surprises if s < -0.02)
        inline = len(surprises) - beats - misses

        # Fordeling per SurpriseType
        dist: dict[str, int] = {}
        for s in surprises:
            st = classify_surprise(s)
            dist[st.value] = dist.get(st.value, 0) + 1

        return {
            "total_reported": len(reported),
            "beat_rate": beats / len(reported),
            "miss_rate": misses / len(reported),
            "inline_rate": inline / len(reported),
            "avg_surprise_pct": float(np.mean(surprises_arr)),
            "median_surprise_pct": float(np.median(surprises_arr)),
            "distribution": dist,
        }

    # ── Print ─────────────────────────────────────────────────

    def print_calendar(self, calendar: EarningsCalendar) -> None:
        """Print earnings-kalender til konsol."""
        print("=" * 65)
        print("EARNINGS KALENDER")
        print("=" * 65)

        if calendar.today:
            print("\n📅 I DAG:")
            for e in calendar.today:
                hour = "Før åbning" if e.hour == "bmo" else "Efter lukke" if e.hour == "amc" else e.hour
                est = f"EPS est: ${e.eps_estimate:.2f}" if e.eps_estimate else "Intet estimat"
                print(f"  {e.symbol:<8s} {hour:<16s} {est}")

        if calendar.upcoming:
            print(f"\n📅 KOMMENDE ({len(calendar.upcoming)} events):")
            for e in calendar.upcoming[:15]:
                hour = "BMO" if e.hour == "bmo" else "AMC" if e.hour == "amc" else "?"
                est = f"EPS est: ${e.eps_estimate:.2f}" if e.eps_estimate else ""
                print(f"  {e.date}  {e.symbol:<8s} {hour:<5s} {est:<20s} (om {e.days_until} dage)")

        if calendar.recent:
            print(f"\n📊 NYLIGE RESULTATER ({len(calendar.recent)} events):")
            for e in calendar.recent[:10]:
                if e.surprise_pct is not None:
                    icon = "✅" if e.surprise_pct > 0.02 else "❌" if e.surprise_pct < -0.02 else "➖"
                    print(
                        f"  {icon} {e.symbol:<8s} {e.date}  "
                        f"EPS: ${e.eps_actual:.2f} vs ${e.eps_estimate:.2f} "
                        f"({e.surprise_pct:+.1%})"
                    )

        print("=" * 65)
