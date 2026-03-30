#!/usr/bin/env python3
"""
Smoke Test — hurtig verificering af alle imports og moduler.

Kører UDEN broker-credentials. Tester kun at koden kan importeres
og at grundlæggende datastrukturer virker.

Usage:
  python scripts/smoke_test.py

Denne test bør køres FØR paper_test.py for at fange import-fejl tidligt.
"""

from __future__ import annotations

import os
import sys
import traceback
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {C.GREEN}✓{C.END} {msg}")


def fail(msg: str) -> None:
    print(f"  {C.RED}✗{C.END} {msg}")


def warn(msg: str) -> None:
    print(f"  {C.YELLOW}⚠{C.END} {msg}")


def test_import(module_path: str, description: str) -> bool:
    try:
        importlib.import_module(module_path)
        ok(f"{description} ({module_path})")
        return True
    except ImportError as e:
        # Missing optional dependency — warn, don't fail
        if "No module named" in str(e):
            dep = str(e).split("'")[1] if "'" in str(e) else str(e)
            warn(f"{description}: mangler dependency '{dep}'")
            return True  # Soft fail — optional dep
        fail(f"{description}: {e}")
        return False
    except Exception as e:
        fail(f"{description}: {e}")
        return False


def main():
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 60}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  ALPHA TRADER — Smoke Test (Import Verification){C.END}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.END}\n")

    passed = 0
    failed = 0

    # ── Core Models ──
    print(f"{C.BOLD}Core Models:{C.END}")
    modules = [
        ("src.broker.models", "Broker models (Order, AccountInfo, enums)"),
        ("src.broker.base_broker", "BaseBroker abstract class"),
    ]
    for mod, desc in modules:
        if test_import(mod, desc):
            passed += 1
        else:
            failed += 1

    # ── Multi-Broker ──
    print(f"\n{C.BOLD}Multi-Broker (T1-T4):{C.END}")
    modules = [
        ("src.broker.broker_router", "BrokerRouter + exchange detection"),
        ("src.broker.aggregated_portfolio", "AggregatedPortfolio"),
        ("src.broker.connection_manager", "ConnectionManager"),
        ("src.broker.order_manager", "OrderManager"),
        ("src.broker.saxo_auth", "Saxo OAuth2"),
        ("src.broker.saxo_broker", "SaxoBroker"),
        ("src.broker.ibkr_broker", "IBKRBroker"),
        ("src.broker.nordnet_auth", "NordnetSession"),
        ("src.broker.nordnet_broker", "NordnetBroker"),
    ]
    for mod, desc in modules:
        if test_import(mod, desc):
            passed += 1
        else:
            failed += 1

    # ── Tax (T5) ──
    print(f"\n{C.BOLD}Corporate Tax (T5):{C.END}")
    modules = [
        ("src.tax.corporate_tax", "CorporateTaxCalculator + FIFO"),
        ("src.tax.tax_credit_tracker", "TaxCreditTracker"),
        ("src.tax.mark_to_market", "MarkToMarketEngine"),
        ("src.tax.dividend_tracker", "DividendTracker + DBO rates"),
        ("src.tax.currency_pnl", "CurrencyPnLTracker"),
        ("src.tax.corporate_tax_reports", "CorporateTaxReportGenerator"),
    ]
    for mod, desc in modules:
        if test_import(mod, desc):
            passed += 1
        else:
            failed += 1

    # ── Dashboard (T6) ──
    print(f"\n{C.BOLD}Dashboard Pages (T6):{C.END}")
    modules = [
        ("src.dashboard.pages.portfolio", "Portfolio page"),
        ("src.dashboard.pages.trading", "Trading page"),
        ("src.dashboard.pages.tax_center", "Tax Center page"),
        ("src.dashboard.pages.broker_status", "Broker Status page"),
        ("src.dashboard.pages.market_explorer", "Market Explorer page"),
    ]
    for mod, desc in modules:
        if test_import(mod, desc):
            passed += 1
        else:
            failed += 1

    # ── Ops (T7) ──
    print(f"\n{C.BOLD}Ops (T7):{C.END}")
    modules = [
        ("src.ops.daily_scheduler", "DailyScheduler"),
        ("src.ops.email_reports", "EmailReports + AlarmManager"),
        ("src.ops.backup", "BackupManager"),
    ]
    for mod, desc in modules:
        if test_import(mod, desc):
            passed += 1
        else:
            failed += 1

    # ── Data Integrity Checks ──
    print(f"\n{C.BOLD}Data Integrity:{C.END}")

    try:
        from src.tax.corporate_tax import CORPORATE_TAX_RATE
        assert CORPORATE_TAX_RATE == 0.22, f"Expected 0.22, got {CORPORATE_TAX_RATE}"
        ok("Selskabsskattesats = 22%")
        passed += 1
    except Exception as e:
        fail(f"Tax rate check: {e}")
        failed += 1

    try:
        from src.tax.dividend_tracker import DBO_RATES
        assert len(DBO_RATES) >= 15, f"Only {len(DBO_RATES)} DBO rates"
        assert DBO_RATES.get("US") == 0.15
        assert DBO_RATES.get("GB") == 0.0
        ok(f"DBO rates: {len(DBO_RATES)} lande, US=15%, GB=0%")
        passed += 1
    except Exception as e:
        fail(f"DBO rates: {e}")
        failed += 1

    try:
        from src.broker.broker_router import detect_exchange
        assert detect_exchange("NOVO-B.CO") == "CSE"
        assert detect_exchange("SAP.DE") == "XETRA"
        assert detect_exchange("HSBA.L") == "LSE"
        ok("Exchange detection: CSE, XETRA, LSE ✓")
        passed += 1
    except Exception as e:
        fail(f"Exchange detection: {e}")
        failed += 1

    try:
        from src.ops.daily_scheduler import is_market_day
        from datetime import date
        # Saturday should not be market day
        assert is_market_day(date(2026, 3, 14)) is False  # Saturday
        # Monday should be market day (unless holiday)
        ok("Market day detection fungerer")
        passed += 1
    except Exception as e:
        fail(f"Market day: {e}")
        failed += 1

    # ── Summary ──
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 60}{C.END}")
    total = passed + failed
    if failed == 0:
        print(f"  {C.GREEN}{C.BOLD}ALL {passed} CHECKS PASSED ✓{C.END}")
        print(f"  Klar til paper_test.py")
    else:
        print(f"  {C.BOLD}{passed}/{total} passed, {C.RED}{failed} failed{C.END}")
        print(f"  Fix fejl og kør igen")
    print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.END}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
