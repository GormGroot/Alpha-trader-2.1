"""
Theme Tracker — sektorrotation og markedstema-tracking.

Funktionalitet:
  - Track investmenttemaer (AI, GLP-1, energy transition, defense, etc.)
  - Detektér sektorrotation (capital flow: tech → energy → healthcare)
  - ETF flow analyse som proxy for institutional positioning
  - "Hvad er hot, hvad er ikke" — dagligt overblik
  - Tema-scoring: styrke, momentum, varighed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger


# ── Tema-definitioner ────────────────────────────────────────

@dataclass
class ThemeDefinition:
    """Definition af et investeringstema."""
    name: str
    description: str
    keywords: list[str]
    etf_proxies: list[str]       # ETF'er der tracker temaet
    key_stocks: list[str]        # Nøgleaktier i temaet
    sector: str


# Predefinerede temaer
THEMES: dict[str, ThemeDefinition] = {
    "ai_infrastructure": ThemeDefinition(
        name="AI Infrastructure",
        description="AI chips, data centers, cloud computing",
        keywords=["artificial intelligence", "gpu", "data center", "ai chip",
                  "large language model", "generative ai", "machine learning"],
        etf_proxies=["BOTZ", "ROBT", "AIQ", "SMH"],
        key_stocks=["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD",
                     "CRM", "SNOW", "PLTR"],
        sector="technology",
    ),
    "glp1_obesity": ThemeDefinition(
        name="GLP-1 & Obesity",
        description="Vægttabsmedicin, diabetes, GLP-1 receptoragonister",
        keywords=["glp-1", "ozempic", "wegovy", "mounjaro", "obesity",
                  "weight loss", "semaglutide", "tirzepatide"],
        etf_proxies=["XLV", "IBB", "XBI"],
        key_stocks=["NOVO-B.CO", "NVO", "LLY", "AMGN", "VKTX"],
        sector="healthcare",
    ),
    "energy_transition": ThemeDefinition(
        name="Energy Transition",
        description="Vedvarende energi, solceller, vindkraft, EVs",
        keywords=["renewable", "solar", "wind energy", "electric vehicle",
                  "ev", "battery", "green energy", "hydrogen",
                  "grøn omstilling", "vedvarende"],
        etf_proxies=["ICLN", "TAN", "FAN", "LIT", "QCLN"],
        key_stocks=["ENPH", "SEDG", "FSLR", "NEE", "ORSTED.CO",
                     "VESTAS.CO", "TSLA", "RIVN"],
        sector="clean_energy",
    ),
    "defense_reshoring": ThemeDefinition(
        name="Defense & Reshoring",
        description="Oprustning, forsyningskæde-flytning, protektionisme",
        keywords=["defense spending", "military", "nato", "reshoring",
                  "onshoring", "supply chain", "tariff", "trade war",
                  "forsvarsbudget", "oprustning"],
        etf_proxies=["ITA", "XAR", "DFEN"],
        key_stocks=["LMT", "RTX", "NOC", "GD", "BA", "SAAB-B.ST",
                     "CHEMM.CO"],
        sector="defense",
    ),
    "crypto_defi": ThemeDefinition(
        name="Crypto & DeFi",
        description="Kryptovaluta, blockchain, decentraliseret finans",
        keywords=["bitcoin", "ethereum", "crypto", "blockchain", "defi",
                  "web3", "nft", "stablecoin", "btc", "eth"],
        etf_proxies=["BITO", "GBTC", "ETHE"],
        key_stocks=["COIN", "MSTR", "RIOT", "MARA", "HUT"],
        sector="crypto",
    ),
    "nuclear_energy": ThemeDefinition(
        name="Nuclear Energy",
        description="Atomkraft, SMR, uranium",
        keywords=["nuclear", "uranium", "smr", "small modular reactor",
                  "atomic", "fission", "atomkraft"],
        etf_proxies=["URA", "URNM", "NLR"],
        key_stocks=["CCJ", "LEU", "SMR", "NNE", "OKLO"],
        sector="energy",
    ),
    "india_growth": ThemeDefinition(
        name="India Growth",
        description="Indiens økonomiske vækst og demografisk dividend",
        keywords=["india gdp", "indian market", "sensex", "nifty",
                  "india economy", "modi", "make in india"],
        etf_proxies=["INDA", "EPI", "INDY", "SMIN"],
        key_stocks=["INFY", "WIT", "HDB", "IBN", "RELIANCE.NS"],
        sector="emerging_markets",
    ),
    "nordic_quality": ThemeDefinition(
        name="Nordic Quality",
        description="Nordiske kvalitetsaktier, C25, skandinavisk industri",
        keywords=["nordic", "scandinavian", "c25", "omx", "danske",
                  "norsk", "svensk", "nordisk"],
        etf_proxies=["GXF", "EDEN"],
        key_stocks=["NOVO-B.CO", "MAERSK-B.CO", "DANSKE.CO", "CARL-B.CO",
                     "VESTAS.CO", "ORSTED.CO", "VOLVO-B.ST", "ABB.ST",
                     "EQNR.OL", "DNB.OL"],
        sector="nordic",
    ),
}


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class MarketTheme:
    """Analyseret markedstema med scoring."""
    name: str
    description: str
    strength: float        # 0-100 (samlet tema-styrke)
    momentum: float        # -100 til +100 (ændring i styrke)
    news_count: int        # Antal nyheder relateret til temaet
    sentiment: float       # -1 til 1 (aggregeret sentiment)
    key_movers: list[dict] # Top aktier i temaet med performance
    etf_performance: dict  # ETF performance (1d, 1w, 1m)
    status: str            # "hot", "warming", "cooling", "cold"
    confidence: float      # 0-1


@dataclass
class SectorRotation:
    """Sektorrotationsanalyse."""
    from_sectors: list[str]   # Sektorer kapital flyder FRA
    to_sectors: list[str]     # Sektorer kapital flyder TIL
    confidence: float
    evidence: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


# ── Theme Tracker ────────────────────────────────────────────

class ThemeTracker:
    """
    Track markedstemaer og sektorrotation.

    Brug:
        tracker = ThemeTracker()
        themes = tracker.analyze_all()

        for theme in themes:
            print(f"{theme.name}: {theme.strength:.0f}/100 ({theme.status})")

        rotation = tracker.detect_rotation()
        print(f"Flow: {rotation.from_sectors} → {rotation.to_sectors}")
    """

    def __init__(self) -> None:
        self._themes = THEMES
        self._history: dict[str, list[tuple[datetime, float]]] = {}

    def analyze_theme(self, theme_id: str) -> MarketTheme | None:
        """Analysér ét tema."""
        defn = self._themes.get(theme_id)
        if not defn:
            return None

        scores: list[float] = []
        details: dict[str, Any] = {}

        # 1. ETF performance som proxy
        etf_perf = {}
        try:
            import yfinance as yf
            for etf in defn.etf_proxies[:3]:
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="1mo")
                    if len(hist) >= 2:
                        perf_1d = (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100
                        perf_1w = (hist["Close"].iloc[-1] / hist["Close"].iloc[-5] - 1) * 100 if len(hist) >= 5 else 0
                        perf_1m = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                        etf_perf[etf] = {
                            "1d": round(perf_1d, 2),
                            "1w": round(perf_1w, 2),
                            "1m": round(perf_1m, 2),
                        }
                        # Score baseret på momentum
                        momentum_score = 50 + perf_1m * 2  # +10% i måneden = 70
                        scores.append(max(0, min(100, momentum_score)))
                except Exception:
                    pass
        except ImportError:
            pass

        # 2. Nyheds-volume og sentiment
        news_count = 0
        sentiment = 0.0
        try:
            from src.sentiment.news_fetcher import NewsFetcher
            from src.sentiment.sentiment_analyzer import SentimentAnalyzer
            fetcher = NewsFetcher()
            analyzer = SentimentAnalyzer()

            # Søg efter tema-keywords i RSS
            articles = fetcher.fetch_rss_news()
            theme_articles = []
            for a in articles:
                text = f"{a.title} {a.summary}".lower()
                if any(kw in text for kw in defn.keywords):
                    theme_articles.append(a)

            news_count = len(theme_articles)
            if theme_articles:
                agg = analyzer.aggregate_sentiment("THEME", theme_articles)
                sentiment = agg.score

                # Score: mange nyheder + positivt sentiment = hot tema
                news_score = min(100, news_count * 10)  # 10+ artikler = 100
                sentiment_score = (sentiment + 1) * 50  # -1→0, 0→50, 1→100
                scores.append(news_score * 0.5 + sentiment_score * 0.5)
        except Exception as exc:
            logger.debug(f"[themes] Nyheder fejl for {theme_id}: {exc}")

        # 3. Key stock performance
        key_movers = []
        try:
            import yfinance as yf
            for sym in defn.key_stocks[:5]:
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(period="5d")
                    if len(hist) >= 2:
                        perf = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                        key_movers.append({
                            "symbol": sym,
                            "performance_5d": round(perf, 2),
                        })
                except Exception:
                    pass

            if key_movers:
                avg_perf = np.mean([m["performance_5d"] for m in key_movers])
                perf_score = 50 + avg_perf * 5
                scores.append(max(0, min(100, perf_score)))
        except ImportError:
            pass

        # Saml
        strength = np.mean(scores) if scores else 50.0
        strength = max(0, min(100, strength))

        # Momentum: sammenlign med historik
        momentum = 0.0
        history = self._history.get(theme_id, [])
        if history:
            old_score = history[-1][1]
            momentum = strength - old_score

        # Gem historik
        if theme_id not in self._history:
            self._history[theme_id] = []
        self._history[theme_id].append((datetime.now(), strength))
        self._history[theme_id] = self._history[theme_id][-100:]

        # Status
        if strength > 70 and momentum > 5:
            status = "hot"
        elif strength > 55 or momentum > 10:
            status = "warming"
        elif strength < 35 or momentum < -10:
            status = "cold"
        elif momentum < -5:
            status = "cooling"
        else:
            status = "neutral"

        confidence = min(1.0, len(scores) / 3)

        return MarketTheme(
            name=defn.name,
            description=defn.description,
            strength=strength,
            momentum=momentum,
            news_count=news_count,
            sentiment=sentiment,
            key_movers=sorted(key_movers,
                              key=lambda m: m["performance_5d"], reverse=True),
            etf_performance=etf_perf,
            status=status,
            confidence=confidence,
        )

    def analyze_all(self) -> list[MarketTheme]:
        """Analysér alle temaer og sorter efter styrke."""
        themes = []
        for theme_id in self._themes:
            theme = self.analyze_theme(theme_id)
            if theme:
                themes.append(theme)

        themes.sort(key=lambda t: t.strength, reverse=True)
        return themes

    def get_hot_themes(self, min_strength: float = 65) -> list[MarketTheme]:
        """Returnér kun hot/warming temaer."""
        return [
            t for t in self.analyze_all()
            if t.strength >= min_strength or t.status in ("hot", "warming")
        ]

    def detect_rotation(self) -> SectorRotation | None:
        """
        Detektér sektorrotation baseret på ETF flows.

        Returnerer hvilke sektorer der modtager/mister kapital.
        """
        try:
            import yfinance as yf

            # Sektor-ETF'er
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Energy": "XLE",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Communication": "XLC",
            }

            performances: dict[str, float] = {}
            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="1mo")
                    if len(hist) >= 5:
                        # Seneste 5 dage relative performance
                        recent = (hist["Close"].iloc[-1] / hist["Close"].iloc[-5] - 1) * 100
                        performances[sector] = recent
                except Exception:
                    pass

            if len(performances) < 5:
                return None

            # Sortér: top = kapital flyder TIL, bund = kapital flyder FRA
            sorted_perf = sorted(performances.items(), key=lambda x: x[1])
            from_sectors = [s[0] for s in sorted_perf[:3]]
            to_sectors = [s[0] for s in sorted_perf[-3:]]

            # Evidence
            evidence = [
                f"{sector}: {perf:+.1f}% (5d)"
                for sector, perf in sorted_perf
            ]

            # Confidence baseret på spredning
            values = list(performances.values())
            spread = max(values) - min(values)
            confidence = min(1.0, spread / 10)  # 10%+ spread = høj confidence

            return SectorRotation(
                from_sectors=from_sectors,
                to_sectors=to_sectors,
                confidence=confidence,
                evidence=evidence,
            )

        except ImportError:
            logger.warning("[themes] yfinance ikke tilgængelig for rotation-analyse")
            return None
        except Exception as exc:
            logger.warning(f"[themes] Rotation fejl: {exc}")
            return None

    def get_daily_overview(self) -> dict:
        """
        Dagligt overblik: hvad er hot, hvad er ikke.

        Returns:
            Dict med hot themes, cold themes, rotation status.
        """
        themes = self.analyze_all()
        rotation = self.detect_rotation()

        return {
            "hot": [
                {"name": t.name, "strength": round(t.strength, 1),
                 "momentum": round(t.momentum, 1), "status": t.status}
                for t in themes if t.status in ("hot", "warming")
            ],
            "cold": [
                {"name": t.name, "strength": round(t.strength, 1),
                 "momentum": round(t.momentum, 1), "status": t.status}
                for t in themes if t.status in ("cold", "cooling")
            ],
            "rotation": {
                "from": rotation.from_sectors if rotation else [],
                "to": rotation.to_sectors if rotation else [],
                "confidence": rotation.confidence if rotation else 0,
            },
            "timestamp": datetime.now().isoformat(),
        }
