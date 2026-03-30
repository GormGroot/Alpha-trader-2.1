"""
Intelligence Engine — Alpha Traders analytiske hjerne.

Moduler:
  - alpha_score: Composite 0-100 score per aktie (6 dimensioner)
  - news_pipeline: Nyhedsaggregering med cross-impact detection
  - llm_client: Claude API wrapper med token tracking og budget management
  - analysis_functions: Claude-drevne markedsanalyser (earnings, events, portefølje)
  - morning_briefing: Daglig markedsbriefing via Claude API
  - evening_analysis: Daglig performance review
  - alert_system: Real-time alerts på pris, sentiment, events
  - watchlist: Intelligent watchlist med Alpha Score ranking
  - theme_tracker: Sektor-rotation og tema-tracking
"""

from src.trader.intelligence.alpha_score import AlphaScoreEngine, AlphaScore
from src.trader.intelligence.news_pipeline import NewsPipeline, IntelligenceReport
from src.trader.intelligence.llm_client import LLMClient, TokenTracker
from src.trader.intelligence.analysis_functions import MarketAnalyst, AnalysisResult
from src.trader.intelligence.alert_system import AlertSystem, Alert, AlertType
from src.trader.intelligence.watchlist import WatchlistManager, WatchlistEntry
from src.trader.intelligence.theme_tracker import ThemeTracker, MarketTheme

__all__ = [
    # Core scoring
    "AlphaScoreEngine",
    "AlphaScore",
    # News intelligence
    "NewsPipeline",
    "IntelligenceReport",
    # Claude API
    "LLMClient",
    "TokenTracker",
    "MarketAnalyst",
    "AnalysisResult",
    # Alerts
    "AlertSystem",
    "Alert",
    "AlertType",
    # Watchlist
    "WatchlistManager",
    "WatchlistEntry",
    # Themes
    "ThemeTracker",
    "MarketTheme",
]
