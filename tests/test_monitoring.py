"""Tests for src/monitoring/ – health, performance, anomaly, audit."""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest

from src.monitoring.audit_log import (
    AuditCategory,
    AuditEntry,
    AuditLog,
    AuditSeverity,
    AuditSummary,
)
from src.monitoring.health_monitor import (
    ComponentStatus,
    HealthMonitor,
    HealthStatus,
    SystemHealth,
)
from src.monitoring.performance_tracker import (
    DailyReport,
    DecayAlert,
    PerformanceTracker,
    StrategyPerformance,
    TradeRecord,
)
from src.monitoring.anomaly_detector import (
    Anomaly,
    AnomalyDetector,
    AnomalySeverity,
    AnomalyType,
)


# ══════════════════════════════════════════════════════════════
#  AuditLog
# ══════════════════════════════════════════════════════════════


class TestAuditLog:
    def _make_log(self) -> AuditLog:
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        return AuditLog(db_path=path, session_id="test_session")

    def test_record_and_query(self):
        log = self._make_log()
        entry_id = log.record(AuditCategory.TRADE, "BUY AAPL 10 @ $175")
        assert entry_id >= 1
        entries = log.query(category=AuditCategory.TRADE)
        assert len(entries) == 1
        assert "AAPL" in entries[0].action

    def test_multiple_categories(self):
        log = self._make_log()
        log.record(AuditCategory.TRADE, "BUY AAPL")
        log.record(AuditCategory.REGIME, "BULL → BEAR")
        log.record(AuditCategory.SYSTEM, "System start")
        assert log.count() == 3
        assert log.count(AuditCategory.TRADE) == 1
        assert log.count(AuditCategory.REGIME) == 1

    def test_checksum_chain(self):
        log = self._make_log()
        log.record(AuditCategory.TRADE, "Trade 1")
        log.record(AuditCategory.TRADE, "Trade 2")
        log.record(AuditCategory.TRADE, "Trade 3")
        assert log.verify_integrity()

    def test_log_trade_convenience(self):
        log = self._make_log()
        entry_id = log.log_trade("AAPL", "buy", 50, 175.0, "SMA crossover signal")
        assert entry_id >= 1
        entries = log.query(category=AuditCategory.TRADE)
        assert entries[0].details["symbol"] == "AAPL"
        assert entries[0].details["qty"] == 50

    def test_log_regime_shift(self):
        log = self._make_log()
        log.log_regime_shift("BULL", "BEAR", 85.0)
        entries = log.query(category=AuditCategory.REGIME)
        assert len(entries) == 1
        assert "BULL" in entries[0].action
        assert entries[0].severity == "warning"

    def test_log_circuit_breaker(self):
        log = self._make_log()
        log.log_circuit_breaker("DAILY", "3% dagligt tab")
        entries = log.query(category=AuditCategory.CIRCUIT_BREAKER)
        assert len(entries) == 1
        assert entries[0].severity == "critical"

    def test_log_error(self):
        log = self._make_log()
        log.log_error("Connection timeout", {"module": "broker"})
        entries = log.query(severity=AuditSeverity.ERROR)
        assert len(entries) == 1

    def test_query_with_since(self):
        log = self._make_log()
        log.record(AuditCategory.TRADE, "Old trade")
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        entries = log.query(since=future)
        assert len(entries) == 0

    def test_summary(self):
        log = self._make_log()
        log.record(AuditCategory.TRADE, "Trade 1")
        log.record(AuditCategory.TRADE, "Trade 2")
        log.log_error("Error 1")
        summary = log.summary()
        assert isinstance(summary, AuditSummary)
        assert summary.total_entries == 3
        assert summary.trade_count == 2
        assert summary.error_count == 1
        assert summary.integrity_ok

    def test_export_csv(self):
        log = self._make_log()
        log.record(AuditCategory.TRADE, "Trade 1")
        log.record(AuditCategory.TRADE, "Trade 2")
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        count = log.export_csv(csv_path)
        assert count == 2
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # Header + 2 rows

    def test_get_recent(self):
        log = self._make_log()
        for i in range(5):
            log.record(AuditCategory.SYSTEM, f"Event {i}")
        recent = log.get_recent(limit=3)
        assert len(recent) == 3

    def test_session_id_stored(self):
        log = self._make_log()
        log.record(AuditCategory.SYSTEM, "Test")
        entries = log.query()
        assert entries[0].session_id == "test_session"


class TestAuditEntry:
    def test_category_enum(self):
        entry = AuditEntry(
            id=1, timestamp="", category="trade", severity="info",
            action="", details={}, checksum="",
        )
        assert entry.category_enum == AuditCategory.TRADE

    def test_severity_enum(self):
        entry = AuditEntry(
            id=1, timestamp="", category="system", severity="critical",
            action="", details={}, checksum="",
        )
        assert entry.severity_enum == AuditSeverity.CRITICAL

    def test_unknown_category(self):
        entry = AuditEntry(
            id=1, timestamp="", category="unknown", severity="info",
            action="", details={}, checksum="",
        )
        assert entry.category_enum == AuditCategory.SYSTEM


# ══════════════════════════════════════════════════════════════
#  HealthMonitor
# ══════════════════════════════════════════════════════════════


class TestHealthMonitor:
    def test_default_init(self):
        m = HealthMonitor()
        assert m.check_count == 0
        assert m.uptime_seconds >= 0

    def test_check_all_returns_system_health(self):
        m = HealthMonitor()
        health = m.check_all()
        assert isinstance(health, SystemHealth)
        assert len(health.components) == 6
        assert m.check_count == 1

    def test_unconfigured_shows_unknown(self):
        m = HealthMonitor()
        health = m.check_all()
        for c in health.components:
            assert c.status in (HealthStatus.UNKNOWN, HealthStatus.HEALTHY,
                                HealthStatus.DEGRADED)

    def test_database_check_healthy(self):
        m = HealthMonitor()
        status = m.check_database()
        assert status.status == HealthStatus.HEALTHY
        assert status.response_time_ms >= 0

    def test_broker_configured_healthy(self):
        mock_broker = MagicMock()
        mock_broker.name = "paper"
        mock_broker.get_account.return_value = MagicMock(
            equity=100_000, cash=50_000,
        )
        m = HealthMonitor()
        m.set_broker(mock_broker)
        status = m.check_broker()
        assert status.status == HealthStatus.HEALTHY
        assert "paper" in status.message

    def test_broker_failure_unhealthy(self):
        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = Exception("Connection refused")
        m = HealthMonitor()
        m.set_broker(mock_broker)
        status = m.check_broker()
        assert status.status == HealthStatus.UNHEALTHY

    def test_risk_manager_halted(self):
        mock_rm = MagicMock()
        type(mock_rm).is_trading_halted = PropertyMock(return_value=True)
        m = HealthMonitor()
        m.set_risk_manager(mock_rm)
        status = m.check_risk_manager()
        assert status.status == HealthStatus.DEGRADED
        assert "stoppet" in status.message.lower()

    def test_data_pipeline_fresh_data(self):
        m = HealthMonitor(data_freshness_minutes=5)
        mock_pipeline = MagicMock()
        m.set_data_pipeline(mock_pipeline)
        m.report_data_received()
        status = m.check_data_pipeline()
        assert status.status == HealthStatus.HEALTHY

    def test_signal_engine_fresh(self):
        m = HealthMonitor()
        mock_engine = MagicMock()
        m.set_signal_engine(mock_engine)
        m.report_signal_generated()
        status = m.check_signal_engine()
        assert status.status == HealthStatus.HEALTHY

    def test_portfolio_check(self):
        mock_portfolio = MagicMock()
        mock_portfolio.summary.return_value = {
            "total_equity": 100_000, "positions": 3,
        }
        m = HealthMonitor()
        m.set_portfolio(mock_portfolio)
        status = m.check_portfolio()
        assert status.status == HealthStatus.HEALTHY
        assert "$100,000" in status.message

    def test_overall_status_unhealthy_if_any_unhealthy(self):
        m = HealthMonitor()
        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = Exception("Fail")
        m.set_broker(mock_broker)
        health = m.check_all()
        assert health.overall_status in (HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN,
                                          HealthStatus.DEGRADED)

    def test_history_tracked(self):
        m = HealthMonitor()
        m.check_all()
        m.check_all()
        history = m.get_history(limit=50)
        assert len(history) >= 6  # 6 components × 2 checks min

    def test_summary_table(self):
        m = HealthMonitor()
        health = m.check_all()
        table = health.summary_table()
        assert "SYSTEM HEALTH" in table

    def test_fail_count_increments(self):
        m = HealthMonitor()
        mock_broker = MagicMock()
        mock_broker.get_account.side_effect = Exception("Fail")
        m.set_broker(mock_broker)
        m.check_all()
        assert m.fail_count >= 1


class TestComponentStatus:
    def test_is_ok_healthy(self):
        c = ComponentStatus(name="Test", status=HealthStatus.HEALTHY,
                            message="OK", last_check="")
        assert c.is_ok

    def test_is_ok_degraded(self):
        c = ComponentStatus(name="Test", status=HealthStatus.DEGRADED,
                            message="Slow", last_check="")
        assert c.is_ok

    def test_is_not_ok_unhealthy(self):
        c = ComponentStatus(name="Test", status=HealthStatus.UNHEALTHY,
                            message="Error", last_check="")
        assert not c.is_ok

    def test_status_icon(self):
        c = ComponentStatus(name="Test", status=HealthStatus.HEALTHY,
                            message="OK", last_check="")
        assert c.status_icon == "🟢"


# ══════════════════════════════════════════════════════════════
#  PerformanceTracker
# ══════════════════════════════════════════════════════════════


class TestPerformanceTracker:
    def test_record_trade(self):
        pt = PerformanceTracker(initial_equity=100_000)
        pt.record_trade("AAPL", pnl=500, strategy="SMA")
        assert pt.trade_count == 1
        assert pt.current_equity == 100_500

    def test_multiple_strategies(self):
        pt = PerformanceTracker()
        pt.record_trade("AAPL", pnl=300, strategy="SMA")
        pt.record_trade("MSFT", pnl=-100, strategy="RSI")
        pt.record_trade("GOOGL", pnl=200, strategy="SMA")
        assert "SMA" in pt.strategy_names
        assert "RSI" in pt.strategy_names

    def test_strategy_performance_ranking(self):
        pt = PerformanceTracker()
        for i in range(15):
            pt.record_trade("AAPL", pnl=100, strategy="Good",
                            return_pct=1.0)
            pt.record_trade("MSFT", pnl=-50, strategy="Bad",
                            return_pct=-0.5)
        perf = pt.strategy_performance()
        assert len(perf) == 2
        assert perf[0].rank == 1
        assert perf[0].name == "Good"

    def test_win_rate(self):
        pt = PerformanceTracker()
        pt.record_trade("AAPL", pnl=100, strategy="A")
        pt.record_trade("AAPL", pnl=200, strategy="A")
        pt.record_trade("AAPL", pnl=-50, strategy="A")
        perf = pt.strategy_performance()
        assert perf[0].win_rate == pytest.approx(66.67, abs=1)

    def test_profit_factor(self):
        pt = PerformanceTracker()
        pt.record_trade("AAPL", pnl=300, strategy="A")
        pt.record_trade("AAPL", pnl=-100, strategy="A")
        perf = pt.strategy_performance()
        assert perf[0].profit_factor == pytest.approx(3.0)

    def test_record_snapshot(self):
        pt = PerformanceTracker(initial_equity=100_000)
        pt.record_snapshot(equity=101_000)
        pt.record_snapshot(equity=102_000)
        assert pt.current_equity == 102_000

    def test_benchmark_return(self):
        pt = PerformanceTracker()
        pt.add_benchmark_return(0.01)
        pt.add_benchmark_return(0.02)
        report = pt.daily_report()
        assert isinstance(report, DailyReport)

    def test_daily_report_structure(self):
        pt = PerformanceTracker()
        pt.record_trade("AAPL", pnl=500, strategy="SMA")
        report = pt.daily_report()
        assert report.total_trades == 1
        assert report.pnl_today >= 0
        assert report.best_trade is not None

    def test_daily_report_summary_table(self):
        pt = PerformanceTracker()
        pt.record_trade("AAPL", pnl=500, strategy="SMA")
        report = pt.daily_report()
        table = report.summary_table()
        assert "DAGLIG RAPPORT" in table
        assert "Sharpe" in table


class TestStrategyDecay:
    def test_no_decay_with_few_trades(self):
        pt = PerformanceTracker()
        pt.record_trade("AAPL", pnl=100, strategy="A")
        alerts = pt.detect_decay()
        assert len(alerts) == 0

    def test_decay_with_negative_sharpe(self):
        pt = PerformanceTracker(initial_equity=100_000)
        for i in range(20):
            pt.record_trade("AAPL", pnl=-100, strategy="Losing",
                            return_pct=-1.0)
        alerts = pt.detect_decay()
        critical = [a for a in alerts if a.severity == "CRITICAL"]
        assert len(critical) >= 1
        assert "Losing" in critical[0].strategy

    def test_decay_alert_has_recommendation(self):
        pt = PerformanceTracker()
        for i in range(20):
            pt.record_trade("AAPL", pnl=-50, strategy="Bad",
                            return_pct=-0.5)
        alerts = pt.detect_decay()
        for a in alerts:
            assert a.recommendation
            assert a.current_sharpe <= 0.5

    def test_strategy_status(self):
        sp = StrategyPerformance(
            name="Test", total_trades=10, winning_trades=7,
            total_pnl=1000, win_rate=70, avg_pnl=100,
            profit_factor=2.0, sharpe_30d=1.5, sharpe_1y=1.2,
            max_drawdown_pct=-5, is_decaying=False, decay_reason="",
        )
        assert sp.status == "STÆRK"


class TestABTest:
    def test_ab_test_basic(self):
        pt = PerformanceTracker()
        for i in range(10):
            pt.record_trade("AAPL", pnl=100, strategy="A",
                            return_pct=1.0)
            pt.record_trade("MSFT", pnl=-20, strategy="B",
                            return_pct=-0.2)
        result = pt.ab_test("A", "B", days=30)
        assert result["winner"] == "A"
        assert result["pnl_a"] > result["pnl_b"]
        assert "recommendation" in result

    def test_ab_test_empty_strategy(self):
        pt = PerformanceTracker()
        result = pt.ab_test("A", "B", days=30)
        assert result["trades_a"] == 0


# ══════════════════════════════════════════════════════════════
#  AnomalyDetector
# ══════════════════════════════════════════════════════════════


class TestAnomalyDetector:
    def test_normal_trade_no_anomaly(self):
        d = AnomalyDetector()
        result = d.check_trade("AAPL", pnl=100, avg_pnl=100, std_pnl=50)
        assert result is None

    def test_large_loss_detected(self):
        d = AnomalyDetector(loss_threshold_std=2.0)
        result = d.check_trade("AAPL", pnl=-500, avg_pnl=100, std_pnl=50)
        assert result is not None
        assert result.anomaly_type == AnomalyType.LARGE_LOSS
        assert result.severity in (AnomalySeverity.HIGH, AnomalySeverity.CRITICAL)

    def test_large_win_detected(self):
        d = AnomalyDetector(loss_threshold_std=2.0)
        result = d.check_trade("AAPL", pnl=500, avg_pnl=100, std_pnl=50)
        assert result is not None
        assert result.anomaly_type == AnomalyType.LARGE_WIN

    def test_auto_stats(self):
        d = AnomalyDetector(loss_threshold_std=3.0)
        # Byg historik op
        for _ in range(20):
            d.check_trade("AAPL", pnl=100)
        # Nu en outlier
        result = d.check_trade("AAPL", pnl=-10000)
        assert result is not None
        assert result.anomaly_type == AnomalyType.LARGE_LOSS

    def test_trade_burst_not_triggered(self):
        d = AnomalyDetector(trade_burst_max=5)
        for _ in range(3):
            d._trade_timestamps.append(datetime.now().isoformat())
        result = d.check_trade_burst()
        assert result is None

    def test_trade_burst_triggered(self):
        d = AnomalyDetector(trade_burst_window_minutes=60, trade_burst_max=5)
        for _ in range(10):
            d._trade_timestamps.append(datetime.now().isoformat())
        result = d.check_trade_burst()
        assert result is not None
        assert result.anomaly_type == AnomalyType.TRADE_BURST


class TestDataQuality:
    def test_no_anomalies_clean_data(self):
        d = AnomalyDetector()
        ts = list(range(10))
        prices = [100 + i * 0.5 for i in range(10)]
        anomalies = d.check_data_quality("AAPL", ts, prices)
        assert len(anomalies) == 0

    def test_duplicate_detected(self):
        d = AnomalyDetector()
        ts = [1, 2, 2, 3, 4]
        prices = [100, 101, 101, 102, 103]
        anomalies = d.check_data_quality("AAPL", ts, prices)
        types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.DATA_DUPLICATE in types

    def test_price_spike_detected(self):
        d = AnomalyDetector(price_spike_pct=5.0)
        ts = list(range(5))
        prices = [100, 101, 102, 150, 103]  # 150 er en spike
        anomalies = d.check_data_quality("AAPL", ts, prices)
        spikes = [a for a in anomalies if a.anomaly_type == AnomalyType.DATA_SPIKE]
        assert len(spikes) >= 1

    def test_volume_anomaly(self):
        d = AnomalyDetector()
        ts = list(range(10))
        prices = [100 + i for i in range(10)]
        volumes = [0.0] * 5 + [1000.0] * 5  # 50% nul
        anomalies = d.check_data_quality("AAPL", ts, prices, volumes)
        vol_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME_ANOMALY]
        assert len(vol_anomalies) >= 1

    def test_mismatched_lengths(self):
        d = AnomalyDetector()
        anomalies = d.check_data_quality("AAPL", [1, 2, 3], [100, 101])
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.DATA_GAP

    def test_empty_data(self):
        d = AnomalyDetector()
        anomalies = d.check_data_quality("AAPL", [], [])
        assert len(anomalies) == 0


class TestStrategyShift:
    def test_no_shift_few_signals(self):
        d = AnomalyDetector()
        result = d.check_strategy_shift("SMA", "BUY")
        assert result is None  # Kun 1 signal – for lidt

    def test_shift_detected(self):
        d = AnomalyDetector()
        # 25 SELL-signaler – 0% buy-rate vs forventet 50% → afvigelse 0.5 > 0.4
        results = []
        for _ in range(25):
            r = d.check_strategy_shift("SMA", "SELL", expected_buy_rate=0.5)
            if r is not None:
                results.append(r)
        assert len(results) >= 1
        assert results[0].anomaly_type == AnomalyType.STRATEGY_SHIFT

    def test_normal_buy_rate_no_shift(self):
        d = AnomalyDetector()
        signals = ["BUY"] * 6 + ["SELL"] * 14  # 30% buy-rate
        for s in signals:
            d.check_strategy_shift("SMA", s)
        result = d.check_strategy_shift("SMA", "BUY")
        assert result is None


class TestAnomalyManagement:
    def test_resolve_anomaly(self):
        d = AnomalyDetector(loss_threshold_std=2.0)
        d.check_trade("AAPL", pnl=-500, avg_pnl=100, std_pnl=50)
        assert d.active_count == 1
        d.resolve(0)
        assert d.active_count == 0

    def test_resolve_all(self):
        d = AnomalyDetector(loss_threshold_std=2.0)
        d.check_trade("AAPL", pnl=-500, avg_pnl=100, std_pnl=50)
        d.check_trade("MSFT", pnl=-600, avg_pnl=100, std_pnl=50)
        assert d.active_count == 2
        count = d.resolve_all()
        assert count == 2
        assert d.active_count == 0

    def test_report(self):
        d = AnomalyDetector(loss_threshold_std=2.0)
        d.check_trade("AAPL", pnl=-500, avg_pnl=100, std_pnl=50)
        report = d.report()
        assert report.active_anomalies == 1
        assert "large_loss" in report.by_type

    def test_anomaly_severity_icon(self):
        a = Anomaly(
            anomaly_type=AnomalyType.LARGE_LOSS,
            severity=AnomalySeverity.CRITICAL,
            title="Test", description="Test",
        )
        assert a.severity_icon == "🔴"


# ══════════════════════════════════════════════════════════════
#  Integration
# ══════════════════════════════════════════════════════════════


class TestIntegration:
    def test_all_modules_import(self):
        from src.monitoring import (
            HealthMonitor, PerformanceTracker, AnomalyDetector, AuditLog,
        )
        assert HealthMonitor is not None
        assert PerformanceTracker is not None
        assert AnomalyDetector is not None
        assert AuditLog is not None

    def test_audit_and_performance_together(self):
        """Test at audit-log og performance tracker kan bruges sammen."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        audit = AuditLog(db_path=path)
        perf = PerformanceTracker(initial_equity=100_000)

        # Simulér en handel
        perf.record_trade("AAPL", pnl=500, strategy="SMA")
        audit.log_trade("AAPL", "buy", 10, 175.0, "SMA crossover")

        assert perf.trade_count == 1
        assert audit.count(AuditCategory.TRADE) == 1

    def test_health_and_anomaly_together(self):
        """Test at health monitor og anomaly detector kan bruges sammen."""
        monitor = HealthMonitor()
        detector = AnomalyDetector()

        health = monitor.check_all()
        anomaly = detector.check_trade("AAPL", pnl=-500, avg_pnl=100, std_pnl=50)

        assert isinstance(health, SystemHealth)
        # Anomaly kan være None eller Anomaly
        if anomaly:
            assert isinstance(anomaly, Anomaly)
