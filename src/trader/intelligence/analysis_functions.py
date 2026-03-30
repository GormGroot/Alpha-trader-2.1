"""
Analysis Functions — Claude API-drevne markedsanalyser.

Funktioner:
  1. analyze_earnings — Earnings rapport analyse (beat/miss, guidance, trading implikation)
  2. analyze_event_impact — Nyhedsevents påvirkning på porteføljen
  3. compare_stocks — Sammenlign aktier på specifikke kriterier
  4. portfolio_review — Fuld portefølje-gennemgang med anbefalinger
  5. sector_outlook — Sektorudsigt baseret på makro og nyheder
  6. ad_hoc_query — "Spørg Alpha" — svar på ethvert markedsspørgsmål

Alle funktioner bruger LLMClient og er fallback-kompatible (template-svar uden API).
Resultater caches i 1 time for at spare tokens.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from src.trader.intelligence.llm_client import LLMClient


# ── Response Cache ───────────────────────────────────────────

class AnalysisCache:
    """Cache for analyse-resultater (undgå gentagne API-kald)."""

    def __init__(self, db_path: str = "data_cache/analysis_cache.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    purpose TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)

    def get(self, key: str) -> str | None:
        """Hent cached analyse hvis den ikke er udløbet."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT response FROM analysis_cache "
                "WHERE cache_key = ? AND expires_at > ?",
                (key, datetime.now().isoformat()),
            ).fetchone()
            return row[0] if row else None

    def set(self, key: str, response: str, purpose: str = "",
            ttl_minutes: int = 60) -> None:
        """Gem analyse i cache."""
        expires = (datetime.now() + timedelta(minutes=ttl_minutes)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO analysis_cache
                   (cache_key, response, purpose, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, response, purpose, datetime.now().isoformat(), expires),
            )

    def make_key(self, *args) -> str:
        """Generér cache-key fra argumenter."""
        content = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()

    def cleanup(self) -> int:
        """Slet udløbne poster."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM analysis_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),),
            )
            return cursor.rowcount or 0


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Resultat af en analyse."""
    text: str
    purpose: str
    symbol: str | None = None
    data_used: dict = field(default_factory=dict)
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


# ── System Prompt ────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """Du er en erfaren markedsanalytiker der arbejder for Alpha Vision,
et dansk investeringsselskab. Du analyserer globale markeder med fokus på
europæiske og amerikanske aktier, ETF'er, råvarer, crypto og forex.

Regler:
- Vær direkte og konkret. Ingen generelle floskel.
- Kvantificér når muligt (%, beløb, niveauer).
- Referér til specifikke aktier, niveauer og events.
- Advar eksplicit om risici.
- Platformen handler for et dansk selskab — husk skattemæssige implikationer.
- Svar på dansk medmindre data er på engelsk.
- DISCLAIMER: Du giver analyse, ikke investeringsrådgivning."""


# ── Analysis Functions ───────────────────────────────────────

class MarketAnalyst:
    """
    Claude API-drevne markedsanalyser.

    Brug:
        analyst = MarketAnalyst(llm_client=client)

        # Earnings
        result = analyst.analyze_earnings("AAPL", earnings_data)

        # Event impact
        result = analyst.analyze_event_impact(event, portfolio)

        # Sammenlign aktier
        result = analyst.compare_stocks(["NOVO-B.CO", "LLY"], "GLP-1 markedsposition")

        # Portefølje review
        result = analyst.portfolio_review(positions, market_data, alpha_scores)

        # Sektorudsigt
        result = analyst.sector_outlook("technology", macro_data)

        # Spørg Alpha (ad-hoc)
        result = analyst.ask("Hvad sker der med europæisk forsvar?", context)
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        cache_ttl_minutes: int = 60,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._cache = AnalysisCache()
        self._cache_ttl = cache_ttl_minutes

    # ── 1. Earnings Analyse ──────────────────────────────────

    def analyze_earnings(
        self,
        symbol: str,
        earnings_data: dict[str, Any],
        price_reaction: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """
        Analysér en earnings rapport.

        Args:
            symbol: Aktiesymbol.
            earnings_data: Dict med EPS actual/estimate, revenue, guidance osv.
            price_reaction: Kursreaktion (before/after, volume).

        Returns:
            AnalysisResult med tekstanalyse.
        """
        cache_key = self._cache.make_key("earnings", symbol, earnings_data)
        cached = self._cache.get(cache_key)
        if cached:
            return AnalysisResult(text=cached, purpose="earnings",
                                  symbol=symbol, cached=True)

        prompt = f"""Analysér denne earnings rapport for {symbol}:

EARNINGS DATA:
{json.dumps(earnings_data, indent=2, default=str)}

{f"KURSREAKTION: {json.dumps(price_reaction, indent=2, default=str)}" if price_reaction else ""}

Giv en kort analyse der dækker:
1. Beat/miss på EPS og revenue (med beløb og %)
2. Guidance (hævet/sænket/fastholdt og implikation)
3. Nøgletal der skiller sig ud
4. Trading-implikation: Bør man købe, holde eller sælge?
5. Risici og hvad der kan ændre billedet
6. Skattemæssig note (lagerbeskatning ved årsopgørelse)

Hold det under 300 ord. Vær specifik med tal."""

        text = self._llm.analyze(
            prompt=prompt,
            purpose="earnings_analysis",
            important=True,  # Sonnet for vigtig analyse
            max_tokens=1500,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        self._cache.set(cache_key, text, "earnings", self._cache_ttl)

        return AnalysisResult(
            text=text,
            purpose="earnings",
            symbol=symbol,
            data_used=earnings_data,
        )

    # ── 2. Event Impact ──────────────────────────────────────

    def analyze_event_impact(
        self,
        event: dict[str, Any],
        portfolio_positions: list[dict] | None = None,
    ) -> AnalysisResult:
        """
        Vurdér en nyhedsevents påvirkning på porteføljen.

        Args:
            event: Dict med event-info (title, type, symbols, sentiment).
            portfolio_positions: Nuværende positioner.
        """
        cache_key = self._cache.make_key("event_impact", event.get("title", ""))
        cached = self._cache.get(cache_key)
        if cached:
            return AnalysisResult(text=cached, purpose="event_impact", cached=True)

        prompt = f"""Vurdér denne nyhedsbegivenheds påvirkning:

EVENT:
{json.dumps(event, indent=2, default=str, ensure_ascii=False)}

PORTEFØLJE:
{json.dumps(portfolio_positions, indent=2, default=str) if portfolio_positions else "Ingen positioner angivet"}

Giv en analyse der dækker:
1. Hvad skete og hvorfor det er vigtigt
2. Direkte påvirkede aktier/sektorer
3. Indirekte påvirkning (cross-impact: f.eks. oliepris → energi, shipping, airlines)
4. Påvirkning af porteføljen (hvis positioner er angivet)
5. Anbefalet handling: Køb/sælg/hold/afvent?
6. Tidshorisont: Er det en kort- eller langsigtet effekt?

Vær specifik. Nævn konkrete aktier og niveauer. Under 250 ord."""

        text = self._llm.analyze(
            prompt=prompt,
            purpose="event_impact",
            important=False,  # Haiku er fint
            max_tokens=1200,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        self._cache.set(cache_key, text, "event_impact", self._cache_ttl)

        return AnalysisResult(
            text=text, purpose="event_impact", data_used=event,
        )

    # ── 3. Sammenlign Aktier ─────────────────────────────────

    def compare_stocks(
        self,
        symbols: list[str],
        criteria: str = "",
        data: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """
        Sammenlign aktier på specifikke kriterier.

        Args:
            symbols: Liste af 2-5 symboler.
            criteria: Hvad skal sammenlignes (f.eks. "vækst", "value", "GLP-1").
            data: Valgfrit — nøgletal per symbol.
        """
        cache_key = self._cache.make_key("compare", sorted(symbols), criteria)
        cached = self._cache.get(cache_key)
        if cached:
            return AnalysisResult(text=cached, purpose="compare", cached=True)

        symbols_str = ", ".join(symbols)
        prompt = f"""Sammenlign disse aktier: {symbols_str}

{f"FOKUS: {criteria}" if criteria else "Giv en overordnet sammenligning."}

{f"DATA: {json.dumps(data, indent=2, default=str)}" if data else ""}

Dæk:
1. Vigtigste forskelle (vurdering, vækst, momentum, risiko)
2. Hvem er bedst positioneret lige nu og hvorfor
3. Hvem har mest upside/downside
4. Konklusion: Hvis du kun kan eje én, hvilken og hvorfor?

Brug en tabel-format til nøgletal hvis muligt. Under 400 ord."""

        text = self._llm.analyze(
            prompt=prompt,
            purpose="stock_comparison",
            important=True,
            max_tokens=1800,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        self._cache.set(cache_key, text, "compare", self._cache_ttl)

        return AnalysisResult(
            text=text, purpose="compare",
            data_used={"symbols": symbols, "criteria": criteria},
        )

    # ── 4. Portefølje Review ─────────────────────────────────

    def portfolio_review(
        self,
        positions: list[dict],
        market_data: dict[str, Any] | None = None,
        alpha_scores: list[dict] | None = None,
    ) -> AnalysisResult:
        """
        Fuld portefølje-gennemgang med anbefalinger.

        Args:
            positions: Liste af positioner (symbol, qty, avg_price, current, pnl).
            market_data: Aktuel markedsdata.
            alpha_scores: Alpha Scores for positionerne.
        """
        # Ingen caching — altid frisk analyse
        total_value = sum(p.get("market_value", 0) for p in positions)
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)

        prompt = f"""Gennemgå denne portefølje:

PORTEFØLJE (total: {total_value:,.0f} DKK, P&L: {total_pnl:+,.0f} DKK):
{json.dumps(positions, indent=2, default=str)}

{f"ALPHA SCORES: {json.dumps(alpha_scores, indent=2, default=str)}" if alpha_scores else ""}

{f"MARKEDSDATA: {json.dumps(market_data, indent=2, default=str)}" if market_data else ""}

Giv en gennemgang der dækker:
1. OVERBLIK: Porteføljens samlede tilstand (diversificering, eksponering, risiko)
2. VINDERE: Positioner der performer godt — bør de trimmes eller rides?
3. TABERE: Positioner med tab — bør de cut'es eller er der recovery-potentiale?
4. RISICI: Overeksponering, korrelation, sektor-koncentration
5. ANBEFALINGER: Top 3 konkrete handlinger lige nu
6. SKAT: Estimeret lagerbeskatning og evt. tab-realisering muligheder

Vær specifik med procenter og beløb. Under 500 ord."""

        text = self._llm.analyze(
            prompt=prompt,
            purpose="portfolio_review",
            important=True,  # Sonnet for vigtig analyse
            max_tokens=2500,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        return AnalysisResult(
            text=text, purpose="portfolio_review",
            data_used={"positions_count": len(positions), "total_value": total_value},
        )

    # ── 5. Sektorudsigt ─────────────────────────────────────

    def sector_outlook(
        self,
        sector: str,
        macro_data: dict[str, Any] | None = None,
        news_summary: list[str] | None = None,
    ) -> AnalysisResult:
        """
        Sektorudsigt baseret på makro, nyheder og trends.

        Args:
            sector: Sektornavn (f.eks. "technology", "healthcare", "energy").
            macro_data: Makro-indikatorer (renter, inflation, PMI).
            news_summary: Top nyheder for sektoren.
        """
        cache_key = self._cache.make_key("sector", sector,
                                          datetime.now().strftime("%Y-%m-%d"))
        cached = self._cache.get(cache_key)
        if cached:
            return AnalysisResult(text=cached, purpose="sector_outlook",
                                  cached=True)

        prompt = f"""Giv en udsigt for {sector}-sektoren:

{f"MAKRO-DATA: {json.dumps(macro_data, indent=2, default=str)}" if macro_data else ""}

{f"SENESTE NYHEDER: {chr(10).join(f'- {n}' for n in (news_summary or []))}" if news_summary else ""}

Dæk:
1. Nuværende tilstand: Er sektoren i med- eller modvind?
2. Drivers: Hvad driver sektoren lige nu (renter, earnings, regulering)?
3. Top picks: 3-5 aktier der er bedst positioneret
4. Risici: Hvad kan gå galt?
5. Tidshorisont: Kort- vs langsigtet outlook
6. Relevante ETF'er som alternativ til enkeltaktier

Under 350 ord. Vær specifik med navne og niveauer."""

        text = self._llm.analyze(
            prompt=prompt,
            purpose="sector_outlook",
            important=False,
            max_tokens=1500,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        self._cache.set(cache_key, text, "sector", self._cache_ttl)

        return AnalysisResult(
            text=text, purpose="sector_outlook",
            data_used={"sector": sector},
        )

    # ── 6. Spørg Alpha (Ad-hoc) ─────────────────────────────

    def ask(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """
        "Spørg Alpha" — svar på ethvert markedsspørgsmål.

        Args:
            question: Brugerens spørgsmål (dansk eller engelsk).
            context: Valgfrit — portefølje, Alpha Scores, markedsdata.
        """
        # Korte spørgsmål caches ikke — de er typisk unikke
        prompt = f"""Spørgsmål: {question}

{f"KONTEKST: {json.dumps(context, indent=2, default=str)}" if context else ""}

Svar direkte og konkret. Hvis spørgsmålet handler om en specifik aktie,
inkludér nøgletal og anbefalinger. Hvis det er et markedsspørgsmål,
relater til hvordan det påvirker porteføljen.

Under 300 ord. Vær specifik."""

        text = self._llm.analyze(
            prompt=prompt,
            purpose="ad_hoc_query",
            important=False,  # Haiku for hurtige svar
            max_tokens=1200,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        return AnalysisResult(
            text=text,
            purpose="ad_hoc",
            data_used={"question": question},
        )

    # ── Convenience ──────────────────────────────────────────

    def quick_take(self, symbol: str) -> AnalysisResult:
        """
        Hurtig take på én aktie — 2-3 sætninger.

        Bruger Haiku for minimal cost.
        """
        prompt = f"""Giv en 2-3 sætningers take på {symbol} lige nu.
Inkludér: kursudvikling, vigtigste nyhed, og om det er køb/hold/sælg.
Vær direkte."""

        text = self._llm.quick_analyze(prompt)

        return AnalysisResult(
            text=text, purpose="quick_take", symbol=symbol,
        )

    def daily_market_narrative(
        self,
        market_data: dict[str, Any],
    ) -> AnalysisResult:
        """
        "Hvad skete i dag og hvorfor?" — daglig markedsnarrative.
        """
        prompt = f"""Beskriv dagens markedsbevægelser:

{json.dumps(market_data, indent=2, default=str)}

Skriv en kort narrative (150-200 ord) der forklarer:
1. Hvad skete (hovedbevægelser)
2. Hvorfor (drivere)
3. Hvad det betyder for i morgen"""

        text = self._llm.analyze(
            prompt=prompt,
            purpose="daily_narrative",
            max_tokens=800,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

        return AnalysisResult(
            text=text, purpose="daily_narrative",
            data_used=market_data,
        )
