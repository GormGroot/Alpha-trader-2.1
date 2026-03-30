"""
Morning Briefing — daglig markedsbriefing genereret via Claude API.

Kører dagligt kl 07:30 CET. Samler overnight data, nyheder, Alpha Scores
og sender en intelligent sammenfatning til Ole.

Indhold:
  - Overnight US-markeder (S&P, Nasdaq, Dow)
  - Asien-session (Nikkei, Hang Seng)
  - EU pre-market (STOXX futures)
  - FX bevægelser (EUR/USD, DKK)
  - Crypto (BTC, ETH)
  - Råvarer (olie, guld)
  - Top nyheder med sentiment
  - Dagens kalender (earnings, ECB, Fed)
  - Top Alpha Scores
  - Porteføljeeksponering
  - Anbefalet handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any

from loguru import logger

from src.trader.intelligence.llm_client import LLMClient
from src.trader.intelligence.news_pipeline import NewsPipeline


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class BriefingData:
    """Alle data samlet til briefing."""
    overnight_us: dict[str, Any] = field(default_factory=dict)
    asia_session: dict[str, Any] = field(default_factory=dict)
    eu_premarket: dict[str, Any] = field(default_factory=dict)
    fx_moves: dict[str, Any] = field(default_factory=dict)
    crypto: dict[str, Any] = field(default_factory=dict)
    commodities: dict[str, Any] = field(default_factory=dict)
    top_news: list[dict] = field(default_factory=list)
    todays_calendar: list[dict] = field(default_factory=list)
    portfolio_positions: list[dict] = field(default_factory=list)
    top_alpha_scores: list[dict] = field(default_factory=list)
    market_mood: dict[str, Any] = field(default_factory=dict)
    regime: str = "unknown"


@dataclass
class MorningBriefingResult:
    """Output fra morning briefing."""
    date: date
    briefing_text: str
    data: BriefingData
    generated_at: datetime = field(default_factory=datetime.now)
    llm_used: bool = False


# ── Morning Briefing Generator ───────────────────────────────

class MorningBriefing:
    """
    Genererer daglig markedsbriefing.

    Brug:
        briefing = MorningBriefing(llm_client=client, pipeline=pipeline)
        result = briefing.generate(watchlist=["AAPL", "MSFT", "NOVO-B.CO"])
        print(result.briefing_text)
    """

    # System prompt til Claude
    SYSTEM_PROMPT = """Du er Alpha Trader's markedsanalytiker. Du skriver en kort,
actionable morgen-briefing til Ole, der handler med firmaets kapital (dansk selskab).

Regler:
- Skriv på dansk
- Vær konkret og actionable (ikke generelle råd)
- Fremhæv de vigtigste 3-5 ting Ole bør vide
- Angiv risici tydeligt
- Hold det under 500 ord
- Brug tal og procenter
- Slut med 1-3 specifikke handlingsforslag baseret på data"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        news_pipeline: NewsPipeline | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._pipeline = news_pipeline or NewsPipeline()

    def generate(
        self,
        watchlist: list[str] | None = None,
        portfolio: list[dict] | None = None,
        alpha_scores: list[dict] | None = None,
    ) -> MorningBriefingResult:
        """
        Generér morgen-briefing.

        Args:
            watchlist: Symboler at overvåge.
            portfolio: Nuværende positioner (valgfrit).
            alpha_scores: Forudberegnede Alpha Scores (valgfrit).

        Returns:
            MorningBriefingResult med tekst og underliggende data.
        """
        watchlist = watchlist or ["AAPL", "MSFT", "GOOGL", "NOVO-B.CO",
                                   "ASML.AS", "EQNR.OL"]

        logger.info("[morning_briefing] Genererer briefing...")

        # 1. Saml data
        data = self._gather_data(watchlist, portfolio, alpha_scores)

        # 2. Generér via LLM
        if self._llm.is_available:
            briefing_text = self._generate_with_llm(data)
            llm_used = True
        else:
            briefing_text = self._generate_fallback(data)
            llm_used = False

        result = MorningBriefingResult(
            date=date.today(),
            briefing_text=briefing_text,
            data=data,
            llm_used=llm_used,
        )

        logger.info(f"[morning_briefing] Briefing genereret (LLM={llm_used})")
        return result

    def _gather_data(
        self,
        watchlist: list[str],
        portfolio: list[dict] | None,
        alpha_scores: list[dict] | None,
    ) -> BriefingData:
        """Saml alle data til briefing."""
        data = BriefingData()

        # Overnight markets via yfinance
        try:
            import yfinance as yf

            # US indices
            indices = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones"}
            for ticker, name in indices.items():
                try:
                    info = yf.Ticker(ticker)
                    hist = info.history(period="2d")
                    if len(hist) >= 2:
                        close = hist["Close"].iloc[-1]
                        prev = hist["Close"].iloc[-2]
                        change_pct = (close - prev) / prev * 100
                        data.overnight_us[name] = {
                            "close": round(close, 2),
                            "change_pct": round(change_pct, 2),
                        }
                except Exception:
                    pass

            # FX
            fx_pairs = {"EURUSD=X": "EUR/USD", "DKKUSD=X": "DKK/USD",
                        "GBPUSD=X": "GBP/USD"}
            for ticker, name in fx_pairs.items():
                try:
                    info = yf.Ticker(ticker)
                    hist = info.history(period="2d")
                    if len(hist) >= 2:
                        close = hist["Close"].iloc[-1]
                        prev = hist["Close"].iloc[-2]
                        change_pct = (close - prev) / prev * 100
                        data.fx_moves[name] = {
                            "rate": round(close, 4),
                            "change_pct": round(change_pct, 2),
                        }
                except Exception:
                    pass

            # Crypto
            for ticker, name in {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum"}.items():
                try:
                    info = yf.Ticker(ticker)
                    hist = info.history(period="2d")
                    if len(hist) >= 2:
                        close = hist["Close"].iloc[-1]
                        prev = hist["Close"].iloc[-2]
                        change_pct = (close - prev) / prev * 100
                        data.crypto[name] = {
                            "price": round(close, 2),
                            "change_pct": round(change_pct, 2),
                        }
                except Exception:
                    pass

            # Commodities
            for ticker, name in {"GC=F": "Guld", "CL=F": "Olie (WTI)"}.items():
                try:
                    info = yf.Ticker(ticker)
                    hist = info.history(period="2d")
                    if len(hist) >= 2:
                        close = hist["Close"].iloc[-1]
                        prev = hist["Close"].iloc[-2]
                        change_pct = (close - prev) / prev * 100
                        data.commodities[name] = {
                            "price": round(close, 2),
                            "change_pct": round(change_pct, 2),
                        }
                except Exception:
                    pass

        except ImportError:
            logger.warning("[morning_briefing] yfinance ikke installeret")

        # Nyheder via pipeline
        try:
            report = self._pipeline.run(symbols=watchlist, days_back=1)
            data.top_news = [
                {
                    "title": sa.article.title[:100],
                    "source": sa.article.source,
                    "sentiment": sa.sentiment.label,
                    "score": round(sa.sentiment.score, 2),
                }
                for sa in report.top_events[:10]
            ]
            data.market_mood = {
                "score": round(report.market_mood.score, 2),
                "label": report.market_mood.label,
                "fear_greed": round(report.market_mood.fear_greed, 1),
                "themes": report.market_mood.trending_themes,
            }
            data.regime = report.regime
        except Exception as exc:
            logger.warning(f"[morning_briefing] News pipeline fejl: {exc}")

        # Earnings kalender
        try:
            from src.sentiment.news_fetcher import NewsFetcher
            fetcher = NewsFetcher()
            today = datetime.now().strftime("%Y-%m-%d")
            earnings = fetcher.fetch_earnings_calendar(from_date=today, to_date=today)
            data.todays_calendar = [
                {"symbol": e.symbol, "hour": e.hour,
                 "eps_estimate": e.eps_estimate}
                for e in (earnings or [])[:10]
            ]
        except Exception:
            pass

        # Portfolio
        if portfolio:
            data.portfolio_positions = portfolio

        # Alpha scores
        if alpha_scores:
            data.top_alpha_scores = alpha_scores

        return data

    def _generate_with_llm(self, data: BriefingData) -> str:
        """Generér briefing med Claude API."""
        prompt = f"""Generér morgen-briefing for {date.today().strftime('%d. %B %Y')}.

Baseret på følgende markedsdata:

OVERNIGHT US:
{_format_dict(data.overnight_us)}

FX:
{_format_dict(data.fx_moves)}

CRYPTO:
{_format_dict(data.crypto)}

RÅVARER:
{_format_dict(data.commodities)}

MARKEDSSTEMNING:
{_format_dict(data.market_mood)}

TOP NYHEDER:
{_format_list(data.top_news)}

DAGENS KALENDER:
{_format_list(data.todays_calendar)}

PORTEFØLJE:
{_format_list(data.portfolio_positions) if data.portfolio_positions else 'Ingen positioner tilgængelige'}

TOP ALPHA SCORES:
{_format_list(data.top_alpha_scores) if data.top_alpha_scores else 'Ikke beregnet endnu'}

Skriv briefing nu:"""

        return self._llm.analyze(
            prompt=prompt,
            purpose="morning_briefing",
            important=True,  # Brug Sonnet til vigtig analyse
            max_tokens=2000,
            system_prompt=self.SYSTEM_PROMPT,
        )

    def _generate_fallback(self, data: BriefingData) -> str:
        """Generér briefing uden LLM (simpel template)."""
        lines = [
            f"═══ MORGEN-BRIEFING — {date.today().strftime('%d. %B %Y')} ═══",
            "",
        ]

        if data.overnight_us:
            lines.append("📊 OVERNIGHT US:")
            for name, info in data.overnight_us.items():
                arrow = "↑" if info["change_pct"] > 0 else "↓"
                lines.append(f"  {name}: {info['close']:,.0f} ({arrow}{info['change_pct']:+.1f}%)")
            lines.append("")

        if data.fx_moves:
            lines.append("💱 FX:")
            for name, info in data.fx_moves.items():
                lines.append(f"  {name}: {info['rate']:.4f} ({info['change_pct']:+.2f}%)")
            lines.append("")

        if data.crypto:
            lines.append("₿ CRYPTO:")
            for name, info in data.crypto.items():
                lines.append(f"  {name}: ${info['price']:,.0f} ({info['change_pct']:+.1f}%)")
            lines.append("")

        if data.commodities:
            lines.append("🛢️ RÅVARER:")
            for name, info in data.commodities.items():
                lines.append(f"  {name}: ${info['price']:,.2f} ({info['change_pct']:+.1f}%)")
            lines.append("")

        if data.market_mood:
            mood = data.market_mood
            lines.append(f"🎭 MARKEDSSTEMNING: {mood.get('label', '?')} "
                         f"(Fear/Greed: {mood.get('fear_greed', 50):.0f}/100)")
            lines.append("")

        if data.top_news:
            lines.append("📰 TOP NYHEDER:")
            for n in data.top_news[:5]:
                emoji = "🟢" if n["sentiment"] == "positive" else "🔴" if n["sentiment"] == "negative" else "⚪"
                lines.append(f"  {emoji} {n['title']} [{n['source']}]")
            lines.append("")

        if data.todays_calendar:
            lines.append("📅 DAGENS KALENDER:")
            for ev in data.todays_calendar[:5]:
                lines.append(f"  {ev['symbol']} — {ev['hour']} "
                             f"(EPS est: {ev.get('eps_estimate', '?')})")
            lines.append("")

        lines.append("[LLM ikke tilgængelig — dette er en template-briefing]")
        return "\n".join(lines)


# ── Hjælpefunktioner ─────────────────────────────────────────

def _format_dict(d: dict) -> str:
    """Formattér dict til tekst."""
    if not d:
        return "Ingen data"
    return "\n".join(f"  {k}: {v}" for k, v in d.items())


def _format_list(lst: list) -> str:
    """Formattér liste af dicts til tekst."""
    if not lst:
        return "Ingen data"
    return "\n".join(str(item) for item in lst[:10])
