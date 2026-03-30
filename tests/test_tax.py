"""
Tests for skatteindberetningsmodulet:
  - CurrencyConverter
  - TransactionLog
  - DanishTaxCalculator
  - TaxReportGenerator
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.risk.portfolio_tracker import ClosedTrade
from src.tax.currency import CurrencyConverter
from src.tax.transaction_log import TransactionLog, TransactionRecord
from src.tax.tax_calculator import DanishTaxCalculator, TaxResult, TaxLot
from src.tax.tax_report import TaxReportGenerator, AnnualReport


# ══════════════════════════════════════════════════════════════
#  CurrencyConverter
# ══════════════════════════════════════════════════════════════


class TestCurrencyConverter:
    def _make(self, fallback_rate: float = 6.90) -> CurrencyConverter:
        tmpdir = tempfile.mkdtemp()
        return CurrencyConverter(cache_dir=tmpdir, fallback_rate=fallback_rate)

    def test_fallback_rate_when_no_data(self):
        cc = self._make(fallback_rate=7.00)
        rate = cc.get_rate("2026-01-15")
        assert rate == 7.00

    def test_convert_usd_to_dkk(self):
        cc = self._make(fallback_rate=6.50)
        result = cc.convert_usd_to_dkk(100.0, "2026-01-15")
        assert result == pytest.approx(650.0)

    def test_manual_save_and_retrieve(self):
        cc = self._make()
        cc._save_rate("2026-06-15", 6.85, "test")
        rate = cc.get_rate("2026-06-15")
        assert rate == pytest.approx(6.85)

    def test_memory_cache(self):
        cc = self._make()
        cc._save_rate("2026-03-01", 6.92, "test")
        cc.get_rate("2026-03-01")  # Populates memory
        assert "2026-03-01" in cc._memory
        assert cc._memory["2026-03-01"] == pytest.approx(6.92)

    def test_weekend_falls_back_to_friday(self):
        cc = self._make()
        cc._save_rate("2026-03-13", 6.88, "test")  # Fredag
        rate = cc.get_rate("2026-03-14")  # Lørdag
        assert rate == pytest.approx(6.88)

    def test_bulk_fetch_returns_count(self):
        cc = self._make()
        # Mock the HTTP call
        with patch.object(cc, "_fetch_and_parse_ecb", return_value=250):
            count = cc.bulk_fetch(2026)
            assert count == 250


# ══════════════════════════════════════════════════════════════
#  TransactionLog
# ══════════════════════════════════════════════════════════════


def _make_closed_trade(**kwargs) -> ClosedTrade:
    defaults = {
        "symbol": "AAPL",
        "side": "long",
        "qty": 10,
        "entry_price": 150.0,
        "exit_price": 160.0,
        "entry_time": "2026-01-15T10:00:00",
        "exit_time": "2026-06-15T15:00:00",
        "exit_reason": "signal",
    }
    defaults.update(kwargs)
    return ClosedTrade(**defaults)


class TestTransactionLog:
    def _make_log(self) -> TransactionLog:
        tmpdir = tempfile.mkdtemp()
        cc = CurrencyConverter(cache_dir=tmpdir, fallback_rate=6.90)
        return TransactionLog(currency=cc, cache_dir=tmpdir)

    def test_log_trade_returns_record(self):
        log = self._make_log()
        trade = _make_closed_trade()
        record = log.log_trade(trade)
        assert isinstance(record, TransactionRecord)
        assert record.symbol == "AAPL"
        assert record.qty == 10

    def test_log_trade_calculates_dkk(self):
        log = self._make_log()
        trade = _make_closed_trade(entry_price=100.0, exit_price=110.0, qty=5)
        record = log.log_trade(trade)
        # 5 * 100 * 6.90 = 3450, 5 * 110 * 6.90 = 3795
        assert record.entry_value_dkk == pytest.approx(3450.0)
        assert record.exit_value_dkk == pytest.approx(3795.0)
        assert record.realized_pnl_dkk == pytest.approx(345.0)

    def test_get_transactions_returns_dataframe(self):
        log = self._make_log()
        log.log_trade(_make_closed_trade())
        df = log.get_transactions()
        assert len(df) == 1
        assert "symbol" in df.columns

    def test_get_transactions_filter_by_year(self):
        log = self._make_log()
        log.log_trade(_make_closed_trade(exit_time="2026-06-15T10:00:00"))
        log.log_trade(_make_closed_trade(exit_time="2025-06-15T10:00:00"))
        df = log.get_transactions(year=2026)
        assert len(df) == 1

    def test_get_transactions_filter_by_symbol(self):
        log = self._make_log()
        log.log_trade(_make_closed_trade(symbol="AAPL"))
        log.log_trade(_make_closed_trade(symbol="MSFT"))
        df = log.get_transactions(symbol="AAPL")
        assert len(df) == 1

    def test_yearly_summary(self):
        log = self._make_log()
        log.log_trade(_make_closed_trade(entry_price=100, exit_price=110))
        log.log_trade(_make_closed_trade(entry_price=100, exit_price=90, symbol="MSFT"))
        summary = log.get_yearly_summary(2026)
        assert summary["num_trades"] == 2
        assert summary["gains_dkk"] > 0
        assert summary["losses_dkk"] < 0

    def test_cumulative_pnl(self):
        log = self._make_log()
        r1 = log.log_trade(_make_closed_trade(entry_price=100, exit_price=110))
        r2 = log.log_trade(_make_closed_trade(entry_price=100, exit_price=120, symbol="MSFT"))
        assert r2.cumulative_pnl_dkk > r1.cumulative_pnl_dkk

    def test_export_csv(self):
        log = self._make_log()
        log.log_trade(_make_closed_trade())
        tmpfile = os.path.join(tempfile.mkdtemp(), "test.csv")
        path = log.export_csv(tmpfile)
        assert os.path.exists(path)

    def test_log_trades_batch(self):
        log = self._make_log()
        trades = [
            _make_closed_trade(symbol="AAPL"),
            _make_closed_trade(symbol="MSFT"),
        ]
        records = log.log_trades(trades)
        assert len(records) == 2


# ══════════════════════════════════════════════════════════════
#  DanishTaxCalculator
# ══════════════════════════════════════════════════════════════


class TestDanishTaxCalculator:
    def test_zero_gain_zero_tax(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        result = calc.calculate(transactions=[], year=2026)
        assert result.total_tax_dkk == 0.0

    def test_gain_under_limit_27_pct(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        txs = [{"symbol": "AAPL", "qty": 10, "entry_value_dkk": 10_000,
                "exit_value_dkk": 40_000, "realized_pnl_dkk": 30_000,
                "entry_date": "2026-01-01", "trade_date": "2026-06-01"}]
        result = calc.calculate(txs, year=2026)
        assert result.total_tax_dkk == pytest.approx(30_000 * 0.27)
        assert result.tax_high_bracket == 0.0

    def test_gain_over_limit_progressive(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        txs = [{"symbol": "AAPL", "qty": 10, "entry_value_dkk": 10_000,
                "exit_value_dkk": 110_000, "realized_pnl_dkk": 100_000,
                "entry_date": "2026-01-01", "trade_date": "2026-06-01"}]
        result = calc.calculate(txs, year=2026)
        expected = 61_000 * 0.27 + 39_000 * 0.42
        assert result.total_tax_dkk == pytest.approx(expected)

    def test_net_loss_zero_tax(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        txs = [{"symbol": "AAPL", "qty": 10, "entry_value_dkk": 50_000,
                "exit_value_dkk": 30_000, "realized_pnl_dkk": -20_000,
                "entry_date": "2026-01-01", "trade_date": "2026-06-01"}]
        result = calc.calculate(txs, year=2026)
        assert result.total_tax_dkk == 0.0
        assert result.remaining_loss_dkk == pytest.approx(20_000)

    def test_carried_losses_reduce_taxable(self):
        calc = DanishTaxCalculator(progression_limit=61_000, carried_losses=10_000)
        txs = [{"symbol": "AAPL", "qty": 10, "entry_value_dkk": 10_000,
                "exit_value_dkk": 40_000, "realized_pnl_dkk": 30_000,
                "entry_date": "2026-01-01", "trade_date": "2026-06-01"}]
        result = calc.calculate(txs, year=2026)
        assert result.taxable_gain_dkk == pytest.approx(20_000)
        assert result.loss_utilized_dkk == pytest.approx(10_000)
        assert result.remaining_loss_dkk == 0.0

    def test_carried_losses_larger_than_gain(self):
        calc = DanishTaxCalculator(progression_limit=61_000, carried_losses=50_000)
        txs = [{"symbol": "AAPL", "qty": 10, "entry_value_dkk": 10_000,
                "exit_value_dkk": 30_000, "realized_pnl_dkk": 20_000,
                "entry_date": "2026-01-01", "trade_date": "2026-06-01"}]
        result = calc.calculate(txs, year=2026)
        assert result.taxable_gain_dkk == 0.0
        assert result.total_tax_dkk == 0.0
        assert result.remaining_loss_dkk == pytest.approx(30_000)

    def test_rubrik_66_on_gain(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        txs = [{"symbol": "AAPL", "qty": 10, "entry_value_dkk": 10_000,
                "exit_value_dkk": 30_000, "realized_pnl_dkk": 20_000,
                "entry_date": "2026-01-01", "trade_date": "2026-06-01"}]
        result = calc.calculate(txs, year=2026)
        assert result.rubrik_66 == pytest.approx(20_000)
        assert result.rubrik_67 == 0.0

    def test_rubrik_67_on_loss(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        txs = [{"symbol": "AAPL", "qty": 10, "entry_value_dkk": 30_000,
                "exit_value_dkk": 10_000, "realized_pnl_dkk": -20_000,
                "entry_date": "2026-01-01", "trade_date": "2026-06-01"}]
        result = calc.calculate(txs, year=2026)
        assert result.rubrik_66 == 0.0
        assert result.rubrik_67 == pytest.approx(20_000)

    def test_per_symbol_breakdown(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        txs = [
            {"symbol": "AAPL", "qty": 5, "entry_value_dkk": 5_000,
             "exit_value_dkk": 8_000, "realized_pnl_dkk": 3_000,
             "entry_date": "2026-01-01", "trade_date": "2026-06-01"},
            {"symbol": "MSFT", "qty": 3, "entry_value_dkk": 6_000,
             "exit_value_dkk": 4_000, "realized_pnl_dkk": -2_000,
             "entry_date": "2026-01-01", "trade_date": "2026-06-01"},
        ]
        result = calc.calculate(txs, year=2026)
        assert result.per_symbol["AAPL"]["gains"] == 3_000
        assert result.per_symbol["MSFT"]["losses"] == -2_000

    def test_estimate_tax_shortcut(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        assert calc.estimate_tax(50_000) == pytest.approx(50_000 * 0.27)
        assert calc.estimate_tax(-1_000) == 0.0

    def test_dividends_and_credit(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        divs = [{"symbol": "AAPL", "gross_dkk": 10_000, "us_tax_dkk": 1_500}]
        result = calc.calculate([], dividends=divs, year=2026)
        assert result.dividend_gross_dkk == 10_000
        assert result.dividend_us_tax_dkk == 1_500
        assert result.dividend_dk_credit == pytest.approx(min(1_500, 10_000 * 0.27))
        assert result.rubrik_68 == 10_000

    def test_lots_created(self):
        calc = DanishTaxCalculator(progression_limit=61_000)
        txs = [{"symbol": "AAPL", "qty": 10, "entry_value_dkk": 10_000,
                "exit_value_dkk": 15_000, "realized_pnl_dkk": 5_000,
                "entry_date": "2026-01-01", "trade_date": "2026-06-01"}]
        result = calc.calculate(txs, year=2026)
        assert len(result.lots) == 1
        assert result.lots[0].gain_dkk == 5_000

    def test_disclaimer_present(self):
        calc = DanishTaxCalculator()
        result = calc.calculate([], year=2026)
        assert "vejledende" in result.disclaimer.lower()


# ══════════════════════════════════════════════════════════════
#  TaxReportGenerator
# ══════════════════════════════════════════════════════════════


class TestTaxReportGenerator:
    def _make_generator(self) -> TaxReportGenerator:
        tmpdir = tempfile.mkdtemp()
        gen = TaxReportGenerator(
            progression_limit=61_000,
            carried_losses=0,
            cache_dir=tmpdir,
        )
        gen._reports_dir = os.path.join(tmpdir, "reports")
        os.makedirs(gen._reports_dir, exist_ok=True)
        from pathlib import Path
        gen._reports_dir = Path(gen._reports_dir)
        return gen

    def test_generate_empty_year(self):
        gen = self._make_generator()
        report = gen.generate(year=2026)
        assert isinstance(report, AnnualReport)
        assert report.tax_result.total_tax_dkk == 0.0

    def test_generate_with_trades(self):
        gen = self._make_generator()
        trade = _make_closed_trade(entry_price=100, exit_price=120)
        gen.transaction_log.log_trade(trade)
        report = gen.generate(year=2026)
        assert report.tax_result.net_gain_dkk > 0
        assert report.tax_result.total_tax_dkk > 0

    def test_summary_lines_format(self):
        gen = self._make_generator()
        trade = _make_closed_trade()
        gen.transaction_log.log_trade(trade)
        report = gen.generate(year=2026)
        lines = report.summary_lines
        assert any("SKATTEINDBERETNING" in l for l in lines)
        assert any("Rubrik 66" in l for l in lines)
        assert any("Rubrik 67" in l for l in lines)
        assert any("DISCLAIMER" in l or "vejledende" in l.lower() for l in lines)

    def test_csv_exported(self):
        gen = self._make_generator()
        trade = _make_closed_trade()
        gen.transaction_log.log_trade(trade)
        report = gen.generate(year=2026)
        assert os.path.exists(report.transactions_csv_path)

    def test_txt_exported(self):
        gen = self._make_generator()
        trade = _make_closed_trade()
        gen.transaction_log.log_trade(trade)
        report = gen.generate(year=2026)
        assert os.path.exists(report.report_txt_path)
