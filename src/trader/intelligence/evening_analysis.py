"""
Evening Analysis — daglig performance review og strategi-evaluering.

Kører dagligt kl 22:00 CET (efter US close). Opsummerer dagens:
  - P&L per position og totalt
  - Hvad gik godt / hvad gik galt
  - Vigtigste events og deres impact
  - Alpha Score ændringer
  - Strategiernes performance
  - Foreslåede justeringer til i morgen
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any

from loguru import logger

from src.trader.intelligence.llm_client import LLMClient
from src.trader.intelligence.news_pipeline import NewsPipeline


@dataclass
class DayPerformance:
    """Samlet performance for en handelsdag."""
    date: date
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    positions_opened: int = 0
    positions_closed: int = 0
    best_trade: dict = field(default_factory=dict)
    worst_trade: dict = field(default_factory=dict)
    winning_trades: int = 0
    losing_trades: int = 0


@dataclass
class EveningReport:
    """Output fra evening analysis."""
    date: date
    analysis_text: str
    performance: DayPerformance
    alpha_score_changes: list[dict] = field(default_factory=list)
    key_events: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class EveningAnalysis:
    """
    Daglig evening performance review.

    Brug:
        analysis = EveningAnalysis(llm_client=client)
        report = analysis.generate(
            portfolio=current_portfolio,
            trades_today=todays_trades,
            alpha_scores=current_scores,
        )
        print(report.analysis_text)
    """

    SYSTEM_PROMPT = """Du er Alpha Traders handelsanalytiker. Du laver daglig
performance-evaluering for Ole (dansk selskab der handler med firmaets kapital).

Regler:
- Skriv på dansk
- Vær ærlig om fejl — ingen sugarcoating
- Fokusér på hvad der kan forbedres
- Nævn skatte-implikationer ved lukkede positioner
- Foreslå justeringer til i morgen
- Hold det under 400 ord"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        news_pipeline: NewsPipeline | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._pipeline = news_pipeline or NewsPipeline()

    def generate(
        self,
        portfolio: list[dict] | None = None,
        trades_today: list[dict] | None = None,
        alpha_scores: list[dict] | None = None,
        alpha_score_changes: list[dict] | None = None,
    ) -> EveningReport:
        """Generér aften-analyse."""
        logger.info("[evening_analysis] Genererer aften-rapport...")

        # 1. Beregn performance
        performance = self._calculate_performance(portfolio, trades_today)

        # 2. Saml nyheds-events fra i dag
        key_events = []
        try:
            watchlist = [p.get("symbol", "") for p in (portfolio or [])]
            if watchlist:
                report = self._pipeline.run(symbols=watchlist, days_back=1)
                key_events = [
                    {
                        "title": sa.article.title[:100],
                        "sentiment": sa.sentiment.label,
                        "impact": sa.impact_score,
                    }
                    for sa in report.top_events[:5]
                ]
        except Exception as exc:
            logger.warning(f"[evening_analysis] Nyheder fejl: {exc}")

        # 3. Generér analyse
        if self._llm.is_available:
            analysis_text = self._generate_with_llm(
                performance, portfolio, trades_today,
                alpha_scores, key_events,
            )
        else:
            analysis_text = self._generate_fallback(
                performance, portfolio, trades_today, key_events,
            )

        return EveningReport(
            date=date.today(),
            analysis_text=analysis_text,
            performance=performance,
            alpha_score_changes=alpha_score_changes or [],
            key_events=key_events,
        )

    def _calculate_performance(
        self,
        portfolio: list[dict] | None,
        trades: list[dict] | None,
    ) -> DayPerformance:
        """Beregn dagens performance."""
        perf = DayPerformance(date=date.today())

        if not trades:
            return perf

        total_pnl = 0.0
        best = {"pnl": -999999, "symbol": ""}
        worst = {"pnl": 999999, "symbol": ""}

        for trade in trades:
            pnl = trade.get("pnl", 0)
            total_pnl += pnl

            if trade.get("action") == "open":
                perf.positions_opened += 1
            elif trade.get("action") == "close":
                perf.positions_closed += 1

            if pnl > 0:
                perf.winning_trades += 1
            elif pnl < 0:
                perf.losing_trades += 1

            if pnl > best["pnl"]:
                best = {"pnl": pnl, "symbol": trade.get("symbol", "")}
            if pnl < worst["pnl"]:
                worst = {"pnl": pnl, "symbol": trade.get("symbol", "")}

        perf.total_pnl = total_pnl
        perf.best_trade = best
        perf.worst_trade = worst

        # P&L %
        if portfolio:
            total_value = sum(p.get("market_value", 0) for p in portfolio)
            if total_value > 0:
                perf.total_pnl_pct = total_pnl / total_value * 100

        return perf

    def _generate_with_llm(
        self,
        performance: DayPerformance,
        portfolio: list[dict] | None,
        trades: list[dict] | None,
        alpha_scores: list[dict] | None,
        key_events: list[dict],
    ) -> str:
        """Generér analyse med Claude API."""
        prompt = f"""Lav en aften-analyse for {date.today().strftime('%d. %B %Y')}.

PERFORMANCE:
- Total P&L: {performance.total_pnl:+,.2f} DKK ({performance.total_pnl_pct:+.2f}%)
- Åbnet: {performance.positions_opened}, Lukket: {performance.positions_closed}
- Winners: {performance.winning_trades}, Losers: {performance.losing_trades}
- Bedste trade: {performance.best_trade}
- Værste trade: {performance.worst_trade}

PORTEFØLJE:
{portfolio if portfolio else 'Ingen positioner'}

DAGENS TRADES:
{trades if trades else 'Ingen trades'}

ALPHA SCORES:
{alpha_scores if alpha_scores else 'Ikke tilgængelig'}

VIGTIGSTE NYHEDER:
{key_events if key_events else 'Ingen events'}

Skriv analyse nu. Fokusér på hvad der gik godt/dårligt og hvad Ole bør gøre i morgen:"""

        return self._llm.analyze(
            prompt=prompt,
            purpose="evening_analysis",
            important=False,  # Haiku er fint til daglig review
            max_tokens=1500,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def _generate_fallback(
        self,
        performance: DayPerformance,
        portfolio: list[dict] | None,
        trades: list[dict] | None,
        key_events: list[dict],
    ) -> str:
        """Simpel template-baseret rapport."""
        lines = [
            f"═══ AFTEN-RAPPORT — {date.today().strftime('%d. %B %Y')} ═══",
            "",
            f"PERFORMANCE: {performance.total_pnl:+,.2f} DKK "
            f"({performance.total_pnl_pct:+.2f}%)",
            f"  Åbnet: {performance.positions_opened} | "
            f"Lukket: {performance.positions_closed}",
            f"  Winners: {performance.winning_trades} | "
            f"Losers: {performance.losing_trades}",
        ]

        if performance.best_trade.get("symbol"):
            lines.append(f"  Bedste: {performance.best_trade['symbol']} "
                         f"({performance.best_trade['pnl']:+,.2f})")
        if performance.worst_trade.get("symbol"):
            lines.append(f"  Værste: {performance.worst_trade['symbol']} "
                         f"({performance.worst_trade['pnl']:+,.2f})")

        if key_events:
            lines.extend(["", "VIGTIGE EVENTS:"])
            for ev in key_events[:3]:
                lines.append(f"  • {ev['title']} ({ev['sentiment']})")

        lines.append("\n[LLM ikke tilgængelig — template rapport]")
        return "\n".join(lines)
