"""
MacroCalendar – track vigtige makroøkonomiske events.

Events:
  - FOMC (Federal Reserve rentemøder)
  - NFP (Non-Farm Payrolls / jobtal)
  - CPI / PPI (inflation)
  - GDP-tal
  - PMI (Purchasing Managers Index)

Features:
  - Vis forventede vs. faktiske tal
  - Historisk markedsreaktion på events
  - Handelsregel: reducér samlet eksponering 25% på FOMC-dage

Bruges af TradingBot til automatisk risikostyring omkring makro-events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from src.sentiment.news_fetcher import EconomicEvent, NewsFetcher, NewsCache


# ── Enums & Dataklasser ──────────────────────────────────────

class MacroEventType(Enum):
    FOMC = "fomc"                    # Federal Reserve rentemøde
    NFP = "nfp"                      # Non-Farm Payrolls
    CPI = "cpi"                      # Consumer Price Index
    PPI = "ppi"                      # Producer Price Index
    GDP = "gdp"                      # Gross Domestic Product
    PMI = "pmi"                      # Purchasing Managers Index
    RETAIL_SALES = "retail_sales"    # Detailsalg
    UNEMPLOYMENT = "unemployment"    # Arbejdsløshedstal
    OTHER = "other"


class MacroImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"  # FOMC, NFP


# Standard impact-niveau per event-type
_EVENT_IMPACT: dict[MacroEventType, MacroImpact] = {
    MacroEventType.FOMC: MacroImpact.CRITICAL,
    MacroEventType.NFP: MacroImpact.CRITICAL,
    MacroEventType.CPI: MacroImpact.HIGH,
    MacroEventType.PPI: MacroImpact.MEDIUM,
    MacroEventType.GDP: MacroImpact.HIGH,
    MacroEventType.PMI: MacroImpact.MEDIUM,
    MacroEventType.RETAIL_SALES: MacroImpact.MEDIUM,
    MacroEventType.UNEMPLOYMENT: MacroImpact.MEDIUM,
    MacroEventType.OTHER: MacroImpact.LOW,
}

# Nøgleord til at klassificere events fra tekst
_EVENT_KEYWORDS: dict[MacroEventType, list[str]] = {
    MacroEventType.FOMC: [
        "fomc", "federal reserve", "fed rate", "interest rate decision",
        "fed meeting", "monetary policy", "powell", "federal funds rate",
    ],
    MacroEventType.NFP: [
        "non-farm payrolls", "nonfarm payrolls", "nfp", "jobs report",
        "employment situation", "payroll",
    ],
    MacroEventType.CPI: [
        "consumer price index", "cpi", "inflation rate", "core inflation",
        "consumer prices",
    ],
    MacroEventType.PPI: [
        "producer price index", "ppi", "producer prices", "wholesale prices",
    ],
    MacroEventType.GDP: [
        "gross domestic product", "gdp", "economic growth", "gdp growth",
    ],
    MacroEventType.PMI: [
        "purchasing managers", "pmi", "ism manufacturing", "ism services",
        "manufacturing index",
    ],
    MacroEventType.RETAIL_SALES: [
        "retail sales", "consumer spending",
    ],
    MacroEventType.UNEMPLOYMENT: [
        "unemployment rate", "jobless claims", "initial claims",
        "continuing claims",
    ],
}


def classify_macro_event(event_name: str) -> MacroEventType:
    """Klassificér en makro-event baseret på navn."""
    name_lower = event_name.lower()
    for etype, keywords in _EVENT_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return etype
    return MacroEventType.OTHER


def get_event_impact(event_type: MacroEventType) -> MacroImpact:
    """Hent standard impact-niveau for en event-type."""
    return _EVENT_IMPACT.get(event_type, MacroImpact.LOW)


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class MacroAnalysis:
    """Analyse af ét makro-event."""
    event_name: str
    event_type: MacroEventType
    impact: MacroImpact
    date: str
    country: str = "US"
    estimate: float | None = None
    actual: float | None = None
    previous: float | None = None
    surprise: float | None = None       # actual - estimate
    surprise_pct: float | None = None   # (actual - estimate) / |estimate|
    is_upcoming: bool = True
    days_until: int = 0

    # Historisk reaktion
    avg_market_move: float = 0.0        # Gennemsnitlig S&P 500 bevægelse
    avg_move_on_hawkish: float = 0.0    # Bevægelse ved hawkish/hot data
    avg_move_on_dovish: float = 0.0     # Bevægelse ved dovish/cold data
    historical_volatility: float = 0.0   # Typisk volatilitet omkring event

    @property
    def has_reported(self) -> bool:
        return self.actual is not None

    @property
    def is_hot(self) -> bool:
        """Er data varmere end forventet? (inflations-risiko)."""
        if self.surprise is None:
            return False
        # For CPI/PPI: højere end forventet = hot
        if self.event_type in (MacroEventType.CPI, MacroEventType.PPI):
            return self.surprise > 0
        # For NFP: flere jobs end forventet = hot
        if self.event_type == MacroEventType.NFP:
            return self.surprise > 0
        # For PMI: højere = hot
        if self.event_type == MacroEventType.PMI:
            return self.surprise > 0
        return False

    @property
    def is_cold(self) -> bool:
        """Er data koldere end forventet? (recession-risiko)."""
        if self.surprise is None:
            return False
        if self.event_type in (MacroEventType.GDP, MacroEventType.RETAIL_SALES):
            return self.surprise < 0
        if self.event_type == MacroEventType.NFP:
            return self.surprise < 0
        return False


@dataclass
class MacroCalendarView:
    """Samlet makro-kalender."""
    upcoming: list[MacroAnalysis] = field(default_factory=list)
    recent: list[MacroAnalysis] = field(default_factory=list)
    today: list[MacroAnalysis] = field(default_factory=list)
    next_critical: MacroAnalysis | None = None
    days_to_next_critical: int = 999

    @property
    def has_critical_today(self) -> bool:
        return any(
            e.impact in (MacroImpact.CRITICAL, MacroImpact.HIGH)
            for e in self.today
        )

    @property
    def critical_events_this_week(self) -> list[MacroAnalysis]:
        """Kritiske events indenfor 7 dage."""
        return [
            e for e in self.upcoming
            if e.days_until <= 7 and e.impact in (MacroImpact.CRITICAL, MacroImpact.HIGH)
        ]


@dataclass
class ExposureAdjustment:
    """Anbefalet eksponerings-justering pga. makro-event."""
    reason: str
    reduction_pct: float        # 0-1, f.eks. 0.25 = reducér 25%
    event_type: MacroEventType
    event_date: str
    days_until: int
    impact: MacroImpact

    def __repr__(self) -> str:
        return (
            f"ExposureAdjustment({self.event_type.value}: "
            f"reducér {self.reduction_pct:.0%}, om {self.days_until}d)"
        )


# ── MacroCalendar ────────────────────────────────────────────

class MacroCalendar:
    """
    Intelligent makroøkonomisk kalender.

    Tracker FOMC, NFP, CPI, GDP, PMI og andre vigtige events.
    Beregner historiske markedsreaktioner og anbefaler eksponerings-justeringer.

    Brug:
        calendar = MacroCalendar(fetcher)
        view = calendar.get_calendar(days_ahead=30)
        adjustments = calendar.get_exposure_adjustments()
    """

    # Standardreduktioner per impact-niveau
    _REDUCTION_MAP: dict[MacroImpact, float] = {
        MacroImpact.CRITICAL: 0.25,   # 25% reduktion for FOMC, NFP
        MacroImpact.HIGH: 0.15,       # 15% for CPI, GDP
        MacroImpact.MEDIUM: 0.10,     # 10% for PPI, PMI
        MacroImpact.LOW: 0.00,        # Ingen reduktion
    }

    def __init__(
        self,
        fetcher: NewsFetcher | None = None,
        reduction_map: dict[MacroImpact, float] | None = None,
        pre_event_days: int = 1,
        cache_dir: str = "data_cache",
    ) -> None:
        """
        Args:
            fetcher: NewsFetcher til at hente økonomisk kalender.
            reduction_map: Custom reduktioner per impact-niveau.
            pre_event_days: Dage før event med reduceret eksponering.
            cache_dir: Mappe til cache.
        """
        self._fetcher = fetcher or NewsFetcher(cache_dir=cache_dir)
        self._reductions = reduction_map or self._REDUCTION_MAP.copy()
        self._pre_days = pre_event_days
        self._cache = NewsCache(cache_dir)

    # ── Kalender ──────────────────────────────────────────────

    def get_calendar(
        self,
        days_ahead: int = 30,
        days_back: int = 7,
        country: str = "US",
    ) -> MacroCalendarView:
        """
        Hent makro-kalender.

        Args:
            days_ahead: Antal dage fremad.
            days_back: Antal dage bagud.
            country: Landekode (default US).

        Returns:
            MacroCalendarView med upcoming, recent og today.
        """
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        from_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        events = self._fetcher.fetch_economic_calendar(
            from_date=from_date, to_date=to_date,
        )

        all_analyses: list[MacroAnalysis] = []
        for event in events:
            if event.country and event.country.upper() != country.upper():
                continue
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

        # Find næste kritiske event
        critical = [
            a for a in upcoming
            if a.impact in (MacroImpact.CRITICAL, MacroImpact.HIGH)
        ]
        next_critical = critical[0] if critical else None
        days_to_next = next_critical.days_until if next_critical else 999

        return MacroCalendarView(
            upcoming=upcoming,
            recent=recent,
            today=today,
            next_critical=next_critical,
            days_to_next_critical=days_to_next,
        )

    @staticmethod
    def _parse_float(val: str | float | None) -> float | None:
        """Konvertér string eller float til float, returnér None ved fejl."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = str(val).strip().replace("%", "").replace(",", "")
            if not s:
                return None
            return float(s)
        except (ValueError, TypeError):
            return None

    def _analyze_event(
        self, event: EconomicEvent, today_str: str,
    ) -> MacroAnalysis:
        """Konvertér EconomicEvent til MacroAnalysis."""
        try:
            event_date = datetime.strptime(event.date, "%Y-%m-%d")
            today = datetime.strptime(today_str, "%Y-%m-%d")
            days_until = (event_date - today).days
        except (ValueError, TypeError):
            days_until = 999

        event_type = classify_macro_event(event.name)
        impact = get_event_impact(event_type)

        # Parse numeric values (EconomicEvent bruger strings)
        actual_val = self._parse_float(event.actual)
        estimate_val = self._parse_float(event.estimate)
        previous_val = self._parse_float(event.previous)

        # Beregn surprise
        surprise = None
        surprise_pct = None
        if actual_val is not None and estimate_val is not None:
            surprise = actual_val - estimate_val
            if abs(estimate_val) > 0.001:
                surprise_pct = surprise / abs(estimate_val)

        return MacroAnalysis(
            event_name=event.name,
            event_type=event_type,
            impact=impact,
            date=event.date,
            country=event.country or "US",
            estimate=estimate_val,
            actual=actual_val,
            previous=previous_val,
            surprise=surprise,
            surprise_pct=surprise_pct,
            is_upcoming=actual_val is None,
            days_until=max(0, days_until),
        )

    # ── Historisk Analyse ─────────────────────────────────────

    def analyze_historical_reaction(
        self,
        event_type: MacroEventType,
        market_data: pd.DataFrame,
        event_dates: list[str],
        window_days: int = 3,
    ) -> dict:
        """
        Analysér historisk markedsreaktion på en bestemt makro-event.

        Args:
            event_type: Type af makro-event.
            market_data: OHLCV DataFrame (f.eks. S&P 500) med dato-index.
            event_dates: Liste af event-datoer (YYYY-MM-DD).
            window_days: Antal dage efter event at måle.

        Returns:
            Dict med avg_move, volatility, positive_pct, osv.
        """
        if market_data is None or market_data.empty or not event_dates:
            return {
                "event_type": event_type.value,
                "event_count": 0,
                "avg_move_pct": 0.0,
                "avg_abs_move_pct": 0.0,
                "max_positive_move": 0.0,
                "max_negative_move": 0.0,
                "positive_reaction_pct": 0.0,
                "avg_volatility": 0.0,
                "moves": [],
            }

        moves = []
        close = market_data["Close"]

        for date_str in event_dates:
            try:
                event_date = pd.Timestamp(date_str)
                idx = close.index.searchsorted(event_date)
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
                "event_type": event_type.value,
                "event_count": 0,
                "avg_move_pct": 0.0,
                "avg_abs_move_pct": 0.0,
                "max_positive_move": 0.0,
                "max_negative_move": 0.0,
                "positive_reaction_pct": 0.0,
                "avg_volatility": 0.0,
                "moves": [],
            }

        moves_arr = np.array(moves)

        return {
            "event_type": event_type.value,
            "event_count": len(moves),
            "avg_move_pct": float(np.mean(moves_arr)),
            "avg_abs_move_pct": float(np.mean(np.abs(moves_arr))),
            "max_positive_move": float(np.max(moves_arr)),
            "max_negative_move": float(np.min(moves_arr)),
            "positive_reaction_pct": float(np.mean(moves_arr > 0)),
            "avg_volatility": float(np.std(moves_arr)),
            "moves": [float(m) for m in moves],
        }

    # ── Exposure Adjustments ─────────────────────────────────

    def get_exposure_adjustments(
        self,
        days_ahead: int = 3,
    ) -> list[ExposureAdjustment]:
        """
        Beregn anbefalede eksponeringsjusteringer baseret på kommende makro-events.

        Regel: Reducér samlet eksponering baseret på event-impact:
          - CRITICAL (FOMC, NFP): 25%
          - HIGH (CPI, GDP): 15%
          - MEDIUM (PPI, PMI): 10%

        Args:
            days_ahead: Antal dage fremad at tjekke.

        Returns:
            Liste af ExposureAdjustment, sorteret efter vigtighed.
        """
        view = self.get_calendar(days_ahead=days_ahead, days_back=0)
        adjustments = []

        for event in view.upcoming + view.today:
            if event.days_until > self._pre_days:
                continue

            reduction = self._reductions.get(event.impact, 0.0)
            if reduction <= 0:
                continue

            adjustments.append(ExposureAdjustment(
                reason=(
                    f"{event.event_name} ({event.event_type.value.upper()}) "
                    f"den {event.date} – reducér eksponering {reduction:.0%}"
                ),
                reduction_pct=reduction,
                event_type=event.event_type,
                event_date=event.date,
                days_until=event.days_until,
                impact=event.impact,
            ))

        # Sortér: højeste impact først
        impact_order = {
            MacroImpact.CRITICAL: 0,
            MacroImpact.HIGH: 1,
            MacroImpact.MEDIUM: 2,
            MacroImpact.LOW: 3,
        }
        adjustments.sort(key=lambda a: impact_order.get(a.impact, 4))

        return adjustments

    def get_total_reduction(self, days_ahead: int = 3) -> tuple[float, list[str]]:
        """
        Beregn samlet eksponeringsreduktion.

        Tager den HØJESTE reduktion (ikke summen) for at undgå overreduktion.

        Returns:
            (total_reduction_pct, list_of_reasons)
        """
        adjustments = self.get_exposure_adjustments(days_ahead)
        if not adjustments:
            return 0.0, []

        # Tag den højeste reduktion
        max_reduction = max(a.reduction_pct for a in adjustments)
        reasons = [a.reason for a in adjustments]

        return max_reduction, reasons

    def should_reduce_exposure(self) -> tuple[bool, float, str]:
        """
        Tjek om samlet eksponering bør reduceres.

        Returns:
            (should_reduce, reduction_pct, reason)
        """
        reduction, reasons = self.get_total_reduction()
        if reduction > 0:
            return True, reduction, " | ".join(reasons)
        return False, 0.0, ""

    # ── FOMC-specifik ────────────────────────────────────────

    def is_fomc_day(self, date: str | None = None) -> bool:
        """Tjek om en dato er en FOMC-dag."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        view = self.get_calendar(days_ahead=1, days_back=0)
        for event in view.today + view.upcoming:
            if event.event_type == MacroEventType.FOMC and event.date == date:
                return True
        return False

    def next_fomc_date(self) -> str | None:
        """Hent næste FOMC-mødedato."""
        view = self.get_calendar(days_ahead=90, days_back=0)
        for event in view.upcoming:
            if event.event_type == MacroEventType.FOMC:
                return event.date
        return None

    # ── Print ─────────────────────────────────────────────────

    def print_calendar(self, view: MacroCalendarView) -> None:
        """Print makro-kalender til konsol."""
        print("=" * 70)
        print("MAKROØKONOMISK KALENDER")
        print("=" * 70)

        impact_icon = {
            MacroImpact.CRITICAL: "🔴",
            MacroImpact.HIGH: "🟠",
            MacroImpact.MEDIUM: "🟡",
            MacroImpact.LOW: "🟢",
        }

        if view.today:
            print("\n📅 I DAG:")
            for e in view.today:
                icon = impact_icon.get(e.impact, "⚪")
                est = f"Est: {e.estimate}" if e.estimate is not None else ""
                print(f"  {icon} {e.event_name:<40s} {est}")

        if view.upcoming:
            print(f"\n📅 KOMMENDE ({len(view.upcoming)} events):")
            for e in view.upcoming[:20]:
                icon = impact_icon.get(e.impact, "⚪")
                est = f"Est: {e.estimate}" if e.estimate is not None else ""
                print(
                    f"  {icon} {e.date}  {e.event_name:<35s} "
                    f"{est:<15s} (om {e.days_until} dage)"
                )

        if view.recent:
            print(f"\n📊 NYLIGE RESULTATER ({len(view.recent)} events):")
            for e in view.recent[:10]:
                icon = impact_icon.get(e.impact, "⚪")
                if e.actual is not None and e.estimate is not None:
                    diff = "✅" if e.actual >= e.estimate else "❌"
                    print(
                        f"  {icon} {diff} {e.event_name:<35s} "
                        f"Actual: {e.actual} vs Est: {e.estimate} "
                        f"(Prev: {e.previous})"
                    )
                elif e.actual is not None:
                    print(f"  {icon} {e.event_name:<35s} Actual: {e.actual}")

        if view.next_critical:
            print(
                f"\n⏰ Næste kritiske event: {view.next_critical.event_name} "
                f"den {view.next_critical.date} "
                f"(om {view.days_to_next_critical} dage)"
            )

        print("=" * 70)
