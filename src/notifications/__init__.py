"""
Notifikationsmodul – email, alerts og påmindelser.

Sender proaktive advarsler om skat, risiko, handler og deadlines.
"""

from src.notifications.notifier import (
    Notifier,
    NotificationChannel,
    EmailChannel,
    LogChannel,
    CallbackChannel,
)
from src.notifications.trading_notifier import (
    TradingNotifier,
    TradingEventConfig,
    TRADE_EXECUTED,
    STOP_LOSS_HIT,
    TRAILING_STOP_HIT,
    DAILY_REPORT,
    DRAWDOWN_WARNING,
    TAKE_PROFIT_HIT,
    REGIME_SHIFT,
    CIRCUIT_BREAKER,
    WEEKLY_SUMMARY,
    STRATEGY_DECAY,
    CORRELATION_WARNING,
    SYSTEM_ERROR,
    TAX_WARNING,
)

__all__ = [
    "Notifier",
    "NotificationChannel",
    "EmailChannel",
    "LogChannel",
    "CallbackChannel",
    "TradingNotifier",
    "TradingEventConfig",
    "TRADE_EXECUTED",
    "STOP_LOSS_HIT",
    "TRAILING_STOP_HIT",
    "DAILY_REPORT",
    "DRAWDOWN_WARNING",
    "TAKE_PROFIT_HIT",
    "REGIME_SHIFT",
    "CIRCUIT_BREAKER",
    "WEEKLY_SUMMARY",
    "STRATEGY_DECAY",
    "CORRELATION_WARNING",
    "SYSTEM_ERROR",
    "TAX_WARNING",
]
