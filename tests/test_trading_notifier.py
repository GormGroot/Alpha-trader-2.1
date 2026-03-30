"""
Tests for TradingNotifier – handelsspecifikke notifikationer.
"""

import tempfile

import pytest

from src.notifications.notifier import Notifier, CallbackChannel
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
    _trade_executed_html,
    _stop_loss_html,
    _daily_report_html,
    _drawdown_warning_html,
    _regime_shift_html,
    _circuit_breaker_html,
    _weekly_summary_html,
    _strategy_decay_html,
    _system_error_html,
    _tax_warning_html,
    _format_pnl,
    _format_pct,
    _pnl_class,
)
from src.risk.portfolio_tracker import PortfolioTracker, ClosedTrade, Position


# ── Helpers ──────────────────────────────────────────────────

def _make_notifier() -> Notifier:
    tmpdir = tempfile.mkdtemp()
    return Notifier(cache_dir=tmpdir)


def _make_trading_notifier(**kwargs) -> tuple[TradingNotifier, list]:
    """Returnér TradingNotifier + captured messages."""
    notifier = _make_notifier()
    captured = []
    notifier.add_channel(CallbackChannel(
        lambda s, t, m, c: captured.append({"severity": s, "title": t, "message": m, "category": c})
    ))
    tn = TradingNotifier(notifier=notifier, **kwargs)
    return tn, captured


def _make_closed_trade(**overrides) -> ClosedTrade:
    defaults = dict(
        symbol="AAPL",
        side="long",
        qty=100,
        entry_price=150.0,
        exit_price=145.0,
        entry_time="2026-03-15T10:00:00",
        exit_time="2026-03-15T14:00:00",
        exit_reason="stop_loss",
    )
    defaults.update(overrides)
    return ClosedTrade(**defaults)


def _make_tracker_with_position() -> PortfolioTracker:
    """Tracker med en åben position og en lukket trade."""
    tracker = PortfolioTracker(initial_capital=100_000)
    tracker.open_position("AAPL", "long", 100, 150.0, "2026-03-15T10:00:00")
    tracker.update_prices({"AAPL": 155.0})
    return tracker


# ── Test Formatering ─────────────────────────────────────────

class TestFormatHelpers:
    def test_format_pnl_positive(self):
        assert _format_pnl(500.0) == "+$500.00"

    def test_format_pnl_negative(self):
        assert _format_pnl(-250.0) == "-$250.00"

    def test_format_pnl_zero(self):
        assert _format_pnl(0.0) == "+$0.00"

    def test_format_pct_positive(self):
        assert _format_pct(0.05) == "+5.00%"

    def test_format_pct_negative(self):
        assert _format_pct(-0.03) == "-3.00%"

    def test_pnl_class_positive(self):
        assert _pnl_class(100) == "positive"

    def test_pnl_class_negative(self):
        assert _pnl_class(-100) == "negative"

    def test_pnl_class_zero(self):
        assert _pnl_class(0) == "neutral"


# ── Test TradingEventConfig ──────────────────────────────────

class TestTradingEventConfig:
    def test_defaults_all_enabled(self):
        config = TradingEventConfig()
        assert config.on_trade_executed is True
        assert config.on_stop_loss is True
        assert config.on_trailing_stop is True
        assert config.on_take_profit is True
        assert config.on_daily_report is True
        assert config.on_drawdown_warning is True

    def test_custom_config(self):
        config = TradingEventConfig(on_trade_executed=False, drawdown_threshold_pct=0.10)
        assert config.on_trade_executed is False
        assert config.drawdown_threshold_pct == 0.10


# ── Test Trade Alert ─────────────────────────────────────────

class TestTradeAlert:
    def test_send_buy_alert(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_trade_alert("BUY", "AAPL", 100, 150.0, 100_000, 85_000)
        assert sent >= 1
        assert len(captured) == 1
        assert captured[0]["category"] == TRADE_EXECUTED
        assert "BUY" in captured[0]["title"]
        assert "AAPL" in captured[0]["title"]

    def test_send_sell_alert(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_trade_alert("SELL", "TSLA", 50, 200.0)
        assert sent >= 1
        assert "SELL" in captured[0]["title"]
        assert "TSLA" in captured[0]["title"]

    def test_disabled_trade_alert(self):
        config = TradingEventConfig(on_trade_executed=False)
        tn, captured = _make_trading_notifier(event_config=config)
        sent = tn.send_trade_alert("BUY", "AAPL", 100, 150.0)
        assert sent == 0
        assert len(captured) == 0

    def test_trade_alert_message_contains_details(self):
        tn, captured = _make_trading_notifier()
        tn.send_trade_alert("BUY", "MSFT", 50, 400.0, 200_000, 180_000)
        msg = captured[0]["message"]
        assert "MSFT" in msg
        assert "$20,000.00" in msg  # 50 * 400
        assert "$200,000.00" in msg
        assert "$180,000.00" in msg


# ── Test Stop-Loss Alert ─────────────────────────────────────

class TestStopLossAlert:
    def test_send_stop_loss_alert(self):
        tn, captured = _make_trading_notifier()
        trade = _make_closed_trade()
        sent = tn.send_stop_loss_alert(trade, 95_000, 5)
        assert sent >= 1
        assert captured[0]["category"] == STOP_LOSS_HIT
        assert "STOP-LOSS" in captured[0]["title"]
        assert captured[0]["severity"] == "WARNING"

    def test_stop_loss_message_has_pnl(self):
        tn, captured = _make_trading_notifier()
        trade = _make_closed_trade(entry_price=150.0, exit_price=145.0, qty=100)
        tn.send_stop_loss_alert(trade)
        msg = captured[0]["message"]
        assert "-$500.00" in msg

    def test_disabled_stop_loss(self):
        config = TradingEventConfig(on_stop_loss=False)
        tn, captured = _make_trading_notifier(event_config=config)
        trade = _make_closed_trade()
        sent = tn.send_stop_loss_alert(trade)
        assert sent == 0


# ── Test Trailing Stop Alert ─────────────────────────────────

class TestTrailingStopAlert:
    def test_send_trailing_stop_profitable(self):
        tn, captured = _make_trading_notifier()
        trade = _make_closed_trade(entry_price=100.0, exit_price=120.0, exit_reason="trailing_stop")
        sent = tn.send_trailing_stop_alert(trade)
        assert sent >= 1
        assert captured[0]["category"] == TRAILING_STOP_HIT
        assert captured[0]["severity"] == "INFO"  # Profitabel

    def test_send_trailing_stop_losing(self):
        tn, captured = _make_trading_notifier()
        trade = _make_closed_trade(entry_price=100.0, exit_price=95.0, exit_reason="trailing_stop")
        tn.send_trailing_stop_alert(trade)
        assert captured[0]["severity"] == "WARNING"  # Tab

    def test_disabled_trailing_stop(self):
        config = TradingEventConfig(on_trailing_stop=False)
        tn, captured = _make_trading_notifier(event_config=config)
        trade = _make_closed_trade()
        sent = tn.send_trailing_stop_alert(trade)
        assert sent == 0


# ── Test Take Profit Alert ───────────────────────────────────

class TestTakeProfitAlert:
    def test_send_take_profit(self):
        tn, captured = _make_trading_notifier()
        trade = _make_closed_trade(entry_price=100.0, exit_price=110.0, exit_reason="take_profit")
        sent = tn.send_take_profit_alert(trade)
        assert sent >= 1
        assert captured[0]["category"] == TAKE_PROFIT_HIT
        assert captured[0]["severity"] == "INFO"

    def test_disabled_take_profit(self):
        config = TradingEventConfig(on_take_profit=False)
        tn, captured = _make_trading_notifier(event_config=config)
        trade = _make_closed_trade()
        sent = tn.send_take_profit_alert(trade)
        assert sent == 0


# ── Test Daily Report ────────────────────────────────────────

class TestDailyReport:
    def test_send_daily_report(self):
        tn, captured = _make_trading_notifier()
        tracker = _make_tracker_with_position()
        sent = tn.send_daily_report(tracker, trades_today=3)
        assert sent >= 1
        assert captured[0]["category"] == DAILY_REPORT

    def test_daily_report_has_metrics(self):
        tn, captured = _make_trading_notifier()
        tracker = _make_tracker_with_position()
        tn.send_daily_report(tracker, trades_today=5)
        msg = captured[0]["message"]
        assert "Porteføljeværdi" in msg
        assert "Kontanter" in msg
        assert "Win rate" in msg
        assert "Sharpe ratio" in msg
        assert "Handler i dag" in msg

    def test_daily_report_contains_positions(self):
        tn, captured = _make_trading_notifier()
        tracker = _make_tracker_with_position()
        tn.send_daily_report(tracker)
        msg = captured[0]["message"]
        assert "AAPL" in msg

    def test_daily_report_positive_is_info(self):
        tn, captured = _make_trading_notifier()
        tracker = _make_tracker_with_position()
        # Equity er steget pga prisændring
        tracker.start_new_day()
        tracker.update_prices({"AAPL": 160.0})
        tn.send_daily_report(tracker)
        assert captured[0]["severity"] == "INFO"

    def test_daily_report_negative_is_warning(self):
        tn, captured = _make_trading_notifier()
        tracker = _make_tracker_with_position()
        tracker.start_new_day()
        tracker.update_prices({"AAPL": 140.0})
        tn.send_daily_report(tracker)
        assert captured[0]["severity"] == "WARNING"

    def test_disabled_daily_report(self):
        config = TradingEventConfig(on_daily_report=False)
        tn, captured = _make_trading_notifier(event_config=config)
        tracker = PortfolioTracker(initial_capital=100_000)
        sent = tn.send_daily_report(tracker)
        assert sent == 0


# ── Test Drawdown Warning ────────────────────────────────────

class TestDrawdownWarning:
    def test_send_drawdown_warning(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_drawdown_warning(0.06, 100_000, 94_000)
        assert sent >= 1
        assert captured[0]["category"] == DRAWDOWN_WARNING
        assert "6.0%" in captured[0]["title"]

    def test_no_alert_below_threshold(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_drawdown_warning(0.03, 100_000, 97_000)
        assert sent == 0
        assert len(captured) == 0

    def test_custom_threshold(self):
        config = TradingEventConfig(drawdown_threshold_pct=0.10)
        tn, captured = _make_trading_notifier(event_config=config)
        # 8% – under ny grænse
        sent = tn.send_drawdown_warning(0.08, 100_000, 92_000)
        assert sent == 0
        # 11% – over grænse
        sent = tn.send_drawdown_warning(0.11, 100_000, 89_000)
        assert sent >= 1

    def test_no_spam_same_level(self):
        tn, captured = _make_trading_notifier()
        tn.send_drawdown_warning(0.06, 100_000, 94_000)
        assert len(captured) == 1
        # Samme niveau igen
        tn.send_drawdown_warning(0.06, 100_000, 94_000)
        assert len(captured) == 1  # Ingen ny alert

    def test_new_alert_at_higher_level(self):
        tn, captured = _make_trading_notifier()
        tn.send_drawdown_warning(0.06, 100_000, 94_000)
        assert len(captured) == 1
        # Værre drawdown → ny alert
        tn.send_drawdown_warning(0.08, 100_000, 92_000)
        assert len(captured) == 2

    def test_critical_severity_at_double_threshold(self):
        tn, captured = _make_trading_notifier()
        # Default threshold 5%, dobbbelt = 10%
        tn.send_drawdown_warning(0.12, 100_000, 88_000)
        assert captured[0]["severity"] == "CRITICAL"

    def test_reset_drawdown_tracker(self):
        tn, captured = _make_trading_notifier()
        tn.send_drawdown_warning(0.06, 100_000, 94_000)
        assert len(captured) == 1
        tn.reset_drawdown_tracker()
        tn.send_drawdown_warning(0.06, 100_000, 94_000)
        assert len(captured) == 2  # Ny alert efter reset

    def test_disabled_drawdown(self):
        config = TradingEventConfig(on_drawdown_warning=False)
        tn, captured = _make_trading_notifier(event_config=config)
        sent = tn.send_drawdown_warning(0.15, 100_000, 85_000)
        assert sent == 0

    def test_check_portfolio_alerts(self):
        tn, captured = _make_trading_notifier(
            event_config=TradingEventConfig(drawdown_threshold_pct=0.01)
        )
        tracker = PortfolioTracker(initial_capital=100_000)
        tracker.open_position("AAPL", "long", 100, 150.0, "2026-03-15T10:00:00")
        tracker.update_prices({"AAPL": 140.0})  # Tab → drawdown
        sent = tn.check_portfolio_alerts(tracker)
        assert sent >= 1


# ── Test HTML Templates ──────────────────────────────────────

class TestHtmlTemplates:
    def test_trade_executed_html(self):
        html = _trade_executed_html(
            action="BUY", symbol="AAPL", qty=100, price=150.0,
            portfolio_value=100_000, cash=85_000,
        )
        assert "AAPL" in html
        assert "BUY" in html
        assert "$150.00" in html
        assert "$15,000.00" in html  # handelsværdi
        assert "Alpha Trading Platform" in html

    def test_stop_loss_html(self):
        trade = _make_closed_trade()
        html = _stop_loss_html(trade, 95_000, 5)
        assert "AAPL" in html
        assert "STOP-LOSS" in html
        assert "$150.00" in html
        assert "$145.00" in html
        assert "-$500.00" in html

    def test_daily_report_html(self):
        html = _daily_report_html(
            total_equity=105_000,
            cash=50_000,
            daily_pnl=500,
            daily_pnl_pct=0.005,
            total_return_pct=0.05,
            open_positions=3,
            unrealized_pnl=2_500,
            realized_pnl=2_000,
            win_rate=0.65,
            max_drawdown_pct=0.03,
            sharpe_ratio=1.5,
            trades_today=5,
            positions=[
                {"symbol": "AAPL", "side": "long", "qty": 100,
                 "current_price": 155.0, "pnl": 500, "pnl_pct": 0.033},
            ],
        )
        assert "AAPL" in html
        assert "$105,000.00" in html
        assert "Win Rate" in html
        assert "Sharpe" in html
        assert "65%" in html

    def test_daily_report_html_no_positions(self):
        html = _daily_report_html(
            total_equity=100_000, cash=100_000,
            daily_pnl=0, daily_pnl_pct=0,
            total_return_pct=0, open_positions=0,
            unrealized_pnl=0, realized_pnl=0,
            win_rate=0, max_drawdown_pct=0,
            sharpe_ratio=0, trades_today=0,
        )
        assert "$100,000.00" in html
        assert "Åbne Positioner" not in html  # Ingen positioner

    def test_drawdown_warning_html(self):
        html = _drawdown_warning_html(0.08, 100_000, 92_000, 0.05)
        assert "8.0%" in html
        assert "$100,000.00" in html
        assert "$92,000.00" in html
        assert "Anbefaling" in html

    def test_static_html_getters(self):
        """Test at statiske HTML-getters virker."""
        html = TradingNotifier.get_trade_html("BUY", "TSLA", 50, 300.0)
        assert "TSLA" in html

        trade = _make_closed_trade()
        html = TradingNotifier.get_stop_loss_html(trade)
        assert "AAPL" in html

        html = TradingNotifier.get_drawdown_html(0.10, 100_000, 90_000)
        assert "10.0%" in html

    def test_daily_report_html_getter_with_tracker(self):
        tracker = _make_tracker_with_position()
        html = TradingNotifier.get_daily_report_html(tracker, trades_today=2)
        assert "AAPL" in html
        assert "Alpha Trading Platform" in html


# ── Test History ─────────────────────────────────────────────

class TestNotificationHistory:
    def test_trade_alerts_saved_to_history(self):
        tn, _ = _make_trading_notifier()
        tn.send_trade_alert("BUY", "AAPL", 100, 150.0)
        tn.send_stop_loss_alert(_make_closed_trade())
        history = tn.notifier.get_history()
        assert len(history) >= 2
        categories = [h["category"] for h in history]
        assert TRADE_EXECUTED in categories
        assert STOP_LOSS_HIT in categories

    def test_filter_by_trading_category(self):
        tn, _ = _make_trading_notifier()
        tn.send_trade_alert("BUY", "AAPL", 100, 150.0)
        tn.send_trade_alert("SELL", "MSFT", 50, 400.0)
        tn.send_stop_loss_alert(_make_closed_trade())

        trade_history = tn.notifier.get_history(category=TRADE_EXECUTED)
        assert all(h["category"] == TRADE_EXECUTED for h in trade_history)
        assert len(trade_history) == 2

    def test_unread_count_after_alerts(self):
        tn, _ = _make_trading_notifier()
        tn.send_trade_alert("BUY", "AAPL", 100, 150.0)
        tn.send_drawdown_warning(0.06, 100_000, 94_000)
        count = tn.notifier.get_unread_count()
        assert count >= 2


# ── Test Properties ──────────────────────────────────────────

class TestProperties:
    def test_notifier_property(self):
        tn, _ = _make_trading_notifier()
        assert isinstance(tn.notifier, Notifier)

    def test_config_property(self):
        config = TradingEventConfig(on_trade_executed=False)
        tn, _ = _make_trading_notifier(event_config=config)
        assert tn.config.on_trade_executed is False

    def test_default_config(self):
        notifier = _make_notifier()
        tn = TradingNotifier(notifier=notifier)
        assert tn.config.on_trade_executed is True
        assert tn.config.drawdown_threshold_pct == 0.05


# ── Test Regime Shift Alert ─────────────────────────────────

class TestRegimeShiftAlert:
    def test_send_regime_shift(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_regime_shift_alert("BULL", "BEAR", 85.0)
        assert sent >= 1
        assert captured[0]["category"] == REGIME_SHIFT
        assert "BULL" in captured[0]["title"]
        assert "BEAR" in captured[0]["title"]

    def test_bear_is_critical(self):
        tn, captured = _make_trading_notifier()
        tn.send_regime_shift_alert("BULL", "BEAR", 90.0)
        assert captured[0]["severity"] == "CRITICAL"

    def test_crash_is_critical(self):
        tn, captured = _make_trading_notifier()
        tn.send_regime_shift_alert("BULL", "CRASH", 95.0)
        assert captured[0]["severity"] == "CRITICAL"

    def test_sideways_is_warning(self):
        tn, captured = _make_trading_notifier()
        tn.send_regime_shift_alert("BULL", "SIDEWAYS", 70.0)
        assert captured[0]["severity"] == "WARNING"

    def test_with_recommended_action(self):
        tn, captured = _make_trading_notifier()
        tn.send_regime_shift_alert(
            "BULL", "BEAR", 85.0,
            recommended_action="Reducér aktieeksponering til 30%",
        )
        assert "Reducér" in captured[0]["message"]

    def test_with_extra_details(self):
        tn, captured = _make_trading_notifier()
        tn.send_regime_shift_alert(
            "BULL", "BEAR", 85.0,
            details={"VIX": 35.0, "Trend": "Negativ"},
        )
        assert "VIX" in captured[0]["message"]

    def test_disabled(self):
        config = TradingEventConfig(on_regime_shift=False)
        tn, captured = _make_trading_notifier(event_config=config)
        sent = tn.send_regime_shift_alert("BULL", "BEAR", 85.0)
        assert sent == 0
        assert len(captured) == 0


# ── Test Circuit Breaker Alert ──────────────────────────────

class TestCircuitBreakerAlert:
    def test_send_circuit_breaker(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_circuit_breaker_alert("HALT", "Max drawdown nået")
        assert sent >= 1
        assert captured[0]["category"] == CIRCUIT_BREAKER
        assert captured[0]["severity"] == "CRITICAL"
        assert "HALT" in captured[0]["title"]

    def test_with_drawdown(self):
        tn, captured = _make_trading_notifier()
        tn.send_circuit_breaker_alert("WARNING", "Dagligt tab", current_drawdown_pct=0.08)
        assert "8.0%" in captured[0]["message"]

    def test_with_actions_taken(self):
        tn, captured = _make_trading_notifier()
        tn.send_circuit_breaker_alert(
            "HALT", "Max drawdown",
            actions_taken=["Alle positioner lukket", "Trading stoppet"],
        )
        assert "Alle positioner lukket" in captured[0]["message"]
        assert "Trading stoppet" in captured[0]["message"]

    def test_disabled(self):
        config = TradingEventConfig(on_circuit_breaker=False)
        tn, captured = _make_trading_notifier(event_config=config)
        sent = tn.send_circuit_breaker_alert("HALT", "Test")
        assert sent == 0


# ── Test Weekly Summary ─────────────────────────────────────

class TestWeeklySummary:
    def test_send_weekly_summary(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_weekly_summary(
            week_pnl=2500.0, week_pnl_pct=0.025,
            total_equity=102_500, trades_count=15,
            win_rate=0.60, sharpe_ratio=1.3,
            max_drawdown_pct=0.03, regime="BULL",
        )
        assert sent >= 1
        assert captured[0]["category"] == WEEKLY_SUMMARY
        assert "Ugentlig Opsummering" in captured[0]["title"]

    def test_positive_is_info(self):
        tn, captured = _make_trading_notifier()
        tn.send_weekly_summary(
            week_pnl=1000.0, week_pnl_pct=0.01,
            total_equity=101_000, trades_count=10,
            win_rate=0.55,
        )
        assert captured[0]["severity"] == "INFO"

    def test_negative_is_warning(self):
        tn, captured = _make_trading_notifier()
        tn.send_weekly_summary(
            week_pnl=-500.0, week_pnl_pct=-0.005,
            total_equity=99_500, trades_count=8,
            win_rate=0.40,
        )
        assert captured[0]["severity"] == "WARNING"

    def test_includes_all_metrics(self):
        tn, captured = _make_trading_notifier()
        tn.send_weekly_summary(
            week_pnl=2000.0, week_pnl_pct=0.02,
            total_equity=102_000, trades_count=12,
            win_rate=0.58, sharpe_ratio=1.1,
            max_drawdown_pct=0.02, regime="BULL",
            best_trade="AAPL +$500", worst_trade="TSLA -$200",
        )
        msg = captured[0]["message"]
        assert "Win rate" in msg
        assert "Sharpe" in msg
        assert "BULL" in msg
        assert "AAPL" in msg
        assert "TSLA" in msg

    def test_disabled(self):
        config = TradingEventConfig(on_weekly_summary=False)
        tn, captured = _make_trading_notifier(event_config=config)
        sent = tn.send_weekly_summary(
            week_pnl=0, week_pnl_pct=0,
            total_equity=100_000, trades_count=0, win_rate=0,
        )
        assert sent == 0


# ── Test Strategy Decay Alert ───────────────────────────────

class TestStrategyDecayAlert:
    def test_send_decay_alert(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_strategy_decay_alert(
            strategy_name="Momentum_v2",
            current_sharpe=0.3, previous_sharpe=1.5,
            win_rate=0.40, previous_win_rate=0.65,
        )
        assert sent >= 1
        assert captured[0]["category"] == STRATEGY_DECAY
        assert "Momentum_v2" in captured[0]["title"]

    def test_sharpe_change_in_message(self):
        tn, captured = _make_trading_notifier()
        tn.send_strategy_decay_alert(
            strategy_name="RSI",
            current_sharpe=0.5, previous_sharpe=1.8,
            win_rate=0.45, previous_win_rate=0.60,
        )
        msg = captured[0]["message"]
        assert "1.80" in msg  # previous
        assert "0.50" in msg  # current
        assert "-1.30" in msg  # change

    def test_critical_severity(self):
        tn, captured = _make_trading_notifier()
        tn.send_strategy_decay_alert(
            strategy_name="Test",
            current_sharpe=-0.5, previous_sharpe=2.0,
            win_rate=0.30, previous_win_rate=0.70,
            decay_severity="CRITICAL",
        )
        assert captured[0]["severity"] == "CRITICAL"

    def test_with_recommendation(self):
        tn, captured = _make_trading_notifier()
        tn.send_strategy_decay_alert(
            strategy_name="Test",
            current_sharpe=0.2, previous_sharpe=1.0,
            win_rate=0.40, previous_win_rate=0.55,
            recommendation="Deaktivér strategien og genoptimér parametre.",
        )
        assert "Deaktivér" in captured[0]["message"]

    def test_disabled(self):
        config = TradingEventConfig(on_strategy_decay=False)
        tn, captured = _make_trading_notifier(event_config=config)
        sent = tn.send_strategy_decay_alert(
            "Test", 0.2, 1.0, 0.40, 0.55,
        )
        assert sent == 0


# ── Test Correlation Warning ────────────────────────────────

class TestCorrelationWarning:
    def test_send_correlation_warning(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_correlation_warning("AAPL", "MSFT", 0.92)
        assert sent >= 1
        assert captured[0]["category"] == CORRELATION_WARNING
        assert "AAPL" in captured[0]["title"]
        assert "MSFT" in captured[0]["title"]
        assert captured[0]["severity"] == "WARNING"

    def test_includes_correlation_value(self):
        tn, captured = _make_trading_notifier()
        tn.send_correlation_warning("AAPL", "MSFT", 0.95, threshold=0.85)
        msg = captured[0]["message"]
        assert "0.95" in msg
        assert "0.85" in msg

    def test_with_exposure(self):
        tn, captured = _make_trading_notifier()
        tn.send_correlation_warning(
            "AAPL", "MSFT", 0.90,
            combined_exposure_pct=0.45,
        )
        assert "45.0%" in captured[0]["message"]

    def test_disabled(self):
        config = TradingEventConfig(on_correlation_warning=False)
        tn, captured = _make_trading_notifier(event_config=config)
        sent = tn.send_correlation_warning("AAPL", "MSFT", 0.95)
        assert sent == 0


# ── Test System Error Alert ─────────────────────────────────

class TestSystemErrorAlert:
    def test_send_system_error(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_system_error_alert("Data Pipeline", "Timeout fra API")
        assert sent >= 1
        assert captured[0]["category"] == SYSTEM_ERROR
        assert "Data Pipeline" in captured[0]["title"]

    def test_recoverable_is_warning(self):
        tn, captured = _make_trading_notifier()
        tn.send_system_error_alert(
            "Broker", "Connection lost",
            is_recoverable=True,
        )
        assert captured[0]["severity"] == "WARNING"

    def test_non_recoverable_is_critical(self):
        tn, captured = _make_trading_notifier()
        tn.send_system_error_alert(
            "Database", "Korrupt database",
            is_recoverable=False,
        )
        assert captured[0]["severity"] == "CRITICAL"

    def test_with_error_type_and_trace(self):
        tn, captured = _make_trading_notifier()
        tn.send_system_error_alert(
            "Broker", "API timeout",
            error_type="ConnectionError",
            stack_trace="Traceback... line 42...",
        )
        msg = captured[0]["message"]
        assert "ConnectionError" in msg
        assert "Traceback" in msg

    def test_disabled(self):
        config = TradingEventConfig(on_system_error=False)
        tn, captured = _make_trading_notifier(event_config=config)
        sent = tn.send_system_error_alert("Test", "Error")
        assert sent == 0


# ── Test Tax Warning ────────────────────────────────────────

class TestTaxWarning:
    def test_send_tax_warning(self):
        tn, captured = _make_trading_notifier()
        sent = tn.send_tax_warning(
            warning_type="Progressionsgrænse",
            realized_gains=60_000,
            tax_threshold=61_000,
            estimated_tax=16_200,
        )
        assert sent >= 1
        assert captured[0]["category"] == TAX_WARNING
        assert "Progressionsgrænse" in captured[0]["title"]

    def test_under_threshold_is_warning(self):
        tn, captured = _make_trading_notifier()
        tn.send_tax_warning(
            "Progressionsgrænse",
            realized_gains=50_000,
            tax_threshold=61_000,
            estimated_tax=13_500,
        )
        assert captured[0]["severity"] == "WARNING"

    def test_over_threshold_is_critical(self):
        tn, captured = _make_trading_notifier()
        tn.send_tax_warning(
            "Progressionsgrænse",
            realized_gains=70_000,
            tax_threshold=61_000,
            estimated_tax=20_000,
        )
        assert captured[0]["severity"] == "CRITICAL"

    def test_includes_amounts(self):
        tn, captured = _make_trading_notifier()
        tn.send_tax_warning(
            "Tab-fradrag",
            realized_gains=45_000,
            tax_threshold=61_000,
            estimated_tax=12_150,
            message="Du nærmer dig progressionsgrænsen.",
        )
        msg = captured[0]["message"]
        assert "45,000" in msg
        assert "61,000" in msg
        assert "12,150" in msg
        assert "Du nærmer dig" in msg
        assert "verificér med revisor" in msg

    def test_disabled(self):
        config = TradingEventConfig(on_tax_warning=False)
        tn, captured = _make_trading_notifier(event_config=config)
        sent = tn.send_tax_warning("Test", 50_000, 61_000, 13_500)
        assert sent == 0


# ── Test New HTML Templates ─────────────────────────────────

class TestNewHtmlTemplates:
    def test_regime_shift_html(self):
        html = _regime_shift_html("BULL", "BEAR", 85.0, "Reducér eksponering")
        assert "BULL" in html
        assert "BEAR" in html
        assert "85%" in html
        assert "Reducér eksponering" in html
        assert "Alpha Trading Platform" in html

    def test_circuit_breaker_html(self):
        html = _circuit_breaker_html(
            "HALT", "Max drawdown", 0.12,
            actions_taken=["Positioner lukket"],
        )
        assert "HALT" in html
        assert "Positioner lukket" in html
        assert "12.0%" in html

    def test_weekly_summary_html(self):
        html = _weekly_summary_html(
            week_pnl=2500.0, week_pnl_pct=0.025,
            total_equity=102_500, trades_count=15,
            win_rate=0.60, sharpe_ratio=1.3,
            max_drawdown_pct=0.03, regime="BULL",
        )
        assert "+$2,500.00" in html
        assert "$102,500.00" in html
        assert "60%" in html
        assert "BULL" in html

    def test_strategy_decay_html(self):
        html = _strategy_decay_html(
            "Momentum_v2", 0.3, 1.5, 0.40, 0.65,
            "Deaktivér strategien",
        )
        assert "Momentum_v2" in html
        assert "1.50" in html
        assert "0.30" in html
        assert "Deaktivér" in html

    def test_system_error_html(self):
        html = _system_error_html(
            "Data Pipeline", "API timeout",
            error_type="ConnectionError", is_recoverable=True,
        )
        assert "Data Pipeline" in html
        assert "API timeout" in html
        assert "ConnectionError" in html
        assert "genopretning" in html

    def test_system_error_html_non_recoverable(self):
        html = _system_error_html(
            "Database", "Korrupt", is_recoverable=False,
        )
        assert "Manuel indgriben" in html

    def test_tax_warning_html(self):
        html = _tax_warning_html(
            "Progressionsgrænse", 60_000, 61_000, 16_200,
        )
        assert "60,000" in html
        assert "61,000" in html
        assert "16,200" in html
        assert "Vejledende" in html

    def test_tax_warning_html_over_threshold(self):
        html = _tax_warning_html(
            "Over grænse", 70_000, 61_000, 20_000,
        )
        assert "Over grænse!" in html

    def test_static_html_getters_new(self):
        """Test nye statiske HTML-getters."""
        html = TradingNotifier.get_regime_shift_html("BULL", "BEAR", 85.0)
        assert "BEAR" in html

        html = TradingNotifier.get_circuit_breaker_html("HALT", "Test")
        assert "HALT" in html

        html = TradingNotifier.get_weekly_summary_html(
            1000, 0.01, 101_000, 5, 0.60,
        )
        assert "$101,000.00" in html

        html = TradingNotifier.get_strategy_decay_html(
            "Test", 0.3, 1.5, 0.40, 0.65,
        )
        assert "Test" in html

        html = TradingNotifier.get_system_error_html("Broker", "Timeout")
        assert "Broker" in html

        html = TradingNotifier.get_tax_warning_html(
            "Skat", 50_000, 61_000, 13_500,
        )
        assert "50,000" in html


# ── Test New EventConfig Defaults ───────────────────────────

class TestNewEventConfigDefaults:
    def test_new_config_fields_all_enabled(self):
        config = TradingEventConfig()
        assert config.on_regime_shift is True
        assert config.on_circuit_breaker is True
        assert config.on_weekly_summary is True
        assert config.on_strategy_decay is True
        assert config.on_correlation_warning is True
        assert config.on_system_error is True
        assert config.on_tax_warning is True

    def test_selective_disable(self):
        config = TradingEventConfig(
            on_regime_shift=False,
            on_tax_warning=False,
        )
        assert config.on_regime_shift is False
        assert config.on_tax_warning is False
        assert config.on_circuit_breaker is True  # Stadig enabled
