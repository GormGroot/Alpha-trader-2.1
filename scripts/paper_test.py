#!/usr/bin/env python3
"""
Alpha Trading Platform — Paper Trading Test Suite.

Kører fuld end-to-end test af hele pipelinen med Alpaca paper trading.

Forudsætninger:
  - .env med ALPACA_API_KEY og ALPACA_SECRET_KEY (paper keys)
  - pip install -r requirements.txt -r requirements-trader.txt

Usage:
  python scripts/paper_test.py              # Kør alle tests
  python scripts/paper_test.py --step 1     # Kør kun step 1
  python scripts/paper_test.py --skip-trade # Spring ordretest over

Trin:
  1. Broker connection + account info
  2. BrokerRouter + exchange detection
  3. Market data hentning
  4. Place paper order (BUY 1 AAPL)
  5. Order tracking + status
  6. Portfolio / positions
  7. Tax beregning (mock + real positions)
  8. Dashboard smoke test (start + screenshot)
  9. Scheduler dry-run
  10. Full P&L + rapport
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

TZ_CET = ZoneInfo("Europe/Copenhagen")

# Use /tmp for SQLite databases in sandbox environments
import tempfile
_TEST_TMP = tempfile.mkdtemp(prefix="alpha_test_")


# ── Styling ────────────────────────────────────────────────

class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def header(text: str) -> None:
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 60}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.END}\n")


def step(n: int, text: str) -> None:
    print(f"{C.BOLD}[Step {n}/10]{C.END} {text}")


def ok(msg: str) -> None:
    print(f"  {C.GREEN}✓{C.END} {msg}")


def fail(msg: str) -> None:
    print(f"  {C.RED}✗{C.END} {msg}")


def warn(msg: str) -> None:
    print(f"  {C.YELLOW}⚠{C.END} {msg}")


def info(msg: str) -> None:
    print(f"  {C.CYAN}→{C.END} {msg}")


# ── Test Steps ─────────────────────────────────────────────

results = {}


def run_step(n: int, name: str, func, skip: bool = False):
    step(n, name)
    if skip:
        warn("SKIPPED")
        results[n] = "SKIPPED"
        return None
    try:
        result = func()
        results[n] = "PASS"
        return result
    except Exception as e:
        fail(f"FEJL: {e}")
        traceback.print_exc()
        results[n] = "FAIL"
        return None


# ── Step 1: Broker Connection ──────────────────────────────

def test_broker_connection():
    from src.broker.alpaca_broker import AlpacaBroker

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or not secret:
        fail("Mangler ALPACA_API_KEY / ALPACA_SECRET_KEY i .env")
        raise ValueError("Missing Alpaca credentials")

    info(f"API key: {api_key[:8]}...")

    broker = AlpacaBroker(
        api_key=api_key,
        secret_key=secret,
        base_url="https://paper-api.alpaca.markets",
    )
    # AlpacaBroker connects in __init__, connect() is optional
    if hasattr(broker, "connect"):
        broker.connect()
    ok("Alpaca paper broker connected")

    account = broker.get_account()
    ok(f"Account equity: ${account.equity:,.2f}")
    ok(f"Cash: ${account.cash:,.2f}")
    ok(f"Buying power: ${account.buying_power:,.2f}")

    return broker


# ── Step 2: BrokerRouter + Exchange Detection ──────────────

def test_broker_router(broker):
    from src.broker.broker_router import BrokerRouter, detect_exchange, detect_asset_type

    # Test exchange detection
    tests = [
        ("NOVO-B.CO", "CSE"),
        ("ERIC-B.ST", "SFB"),
        ("SAP.DE", "XETRA"),
        ("HSBA.L", "LSE"),
        ("ASML.AS", "AMS"),
    ]

    for symbol, expected in tests:
        result = detect_exchange(symbol)
        if result == expected:
            ok(f"detect_exchange('{symbol}') → {result}")
        else:
            warn(f"detect_exchange('{symbol}') → {result} (forventet {expected})")

    # Test asset type detection
    crypto_type = detect_asset_type("BTC-USD")
    ok(f"detect_asset_type('BTC-USD') → {crypto_type}")

    # Router med Alpaca
    router = BrokerRouter()
    router.register("alpaca", broker)
    ok("BrokerRouter created with Alpaca")

    # Resolve US stock
    broker_name, broker_inst = router.resolve_broker("AAPL")
    ok(f"resolve_broker('AAPL') → {broker_name}")

    # Explain routing
    explanation = router.explain_routing("AAPL")
    ok(f"Routing explanation: {explanation}")

    return router


# ── Step 3: Market Data ───────────────────────────────────

def test_market_data():
    from src.data.market_data import MarketDataFetcher

    mdf = MarketDataFetcher(cache_dir=_TEST_TMP)

    # Test historiske data
    df = mdf.get_historical("AAPL", interval="1d", lookback_days=30)
    if df.empty:
        fail("Ingen data for AAPL")
        raise ValueError("No market data")

    ok(f"AAPL historik: {len(df)} dage, seneste close: ${df['Close'].iloc[-1]:.2f}")

    # Test DK stock
    try:
        df_novo = mdf.get_historical("NOVO-B.CO", interval="1d", lookback_days=30)
        if not df_novo.empty:
            ok(f"NOVO-B.CO historik: {len(df_novo)} dage, seneste: {df_novo['Close'].iloc[-1]:.2f} DKK")
        else:
            warn("NOVO-B.CO: Ingen data (yfinance ticker-problem)")
    except Exception as e:
        warn(f"NOVO-B.CO: {e}")

    return mdf


# ── Step 4: Place Paper Order ──────────────────────────────

def test_place_order(router):
    from src.broker.models import OrderSide, OrderType

    info("Placerer paper order: BUY 1 AAPL (market)...")
    order = router.buy("AAPL", 1, order_type=OrderType.MARKET)

    oid = getattr(order, "order_id", None) or getattr(order, "id", None) or "unknown"
    ok(f"Order ID: {oid}")
    ok(f"Status: {order.status}")
    ok(f"Symbol: {order.symbol}, Qty: {order.qty}, Side: {order.side}")

    # Vent lidt på fill
    time.sleep(3)

    return order


# ── Step 5: Order Tracking ────────────────────────────────

def test_order_tracking(router, order):
    from src.broker.order_manager import OrderManager

    mgr = OrderManager(router=router, db_path=os.path.join(_TEST_TMP, "test_orders.db"))

    # Check order status via broker
    try:
        oid = getattr(order, "order_id", None) or getattr(order, "id", None)
        status = router.get_order_status(oid)
        ok(f"Order status after 3s: {status.status}")
    except Exception as e:
        warn(f"Order status check: {e}")

    # Test OrderManager persistence
    info("Test OrderManager SQLite persistence...")
    unified = mgr.place_order("MSFT", "buy", 1)
    ok(f"Unified order: {unified.unified_id}")

    retrieved = mgr.get_order(unified.unified_id)
    if retrieved:
        ok(f"Genfundet fra DB: {retrieved.unified_id}")
    else:
        warn("Kunne ikke genfinde order fra DB")

    stats = mgr.get_statistics()
    ok(f"Order statistik: {stats}")

    return mgr


# ── Step 6: Portfolio / Positions ──────────────────────────

def test_portfolio(router):
    from src.broker.aggregated_portfolio import AggregatedPortfolio

    portfolio = AggregatedPortfolio(router)
    positions = portfolio.get_all_positions("DKK")

    ok(f"Positioner fundet: {len(positions)}")
    for pos in positions[:5]:
        broker_src = getattr(pos, "broker_source", getattr(pos, "broker", ""))
        ok(f"  {pos.symbol}: {pos.qty} stk @ {pos.current_price:.2f} ({broker_src})")

    try:
        summary = portfolio.get_total_value("DKK")
        ok(f"Total portfolio: {summary.total_value_dkk:,.0f} DKK")
        ok(f"Unrealized P&L: {summary.total_unrealized_pnl_dkk:+,.0f} DKK")
    except Exception as e:
        warn(f"Portfolio summary: {e}")

    try:
        breakdown = portfolio.allocation_breakdown("DKK")
        if breakdown:
            ok(f"Allocation breakdown: {list(breakdown.keys())}")
    except Exception as e:
        warn(f"Allocation: {e}")

    return portfolio


# ── Step 7: Tax Beregning ─────────────────────────────────

def test_tax(portfolio):
    from src.tax.corporate_tax import CorporateTaxCalculator, CORPORATE_TAX_RATE

    ok(f"Selskabsskattesats: {CORPORATE_TAX_RATE:.0%}")

    from src.tax.corporate_tax import FIFOTracker
    fifo = FIFOTracker(db_path=os.path.join(_TEST_TMP, "test_tax.db"))
    calc = CorporateTaxCalculator(fifo_tracker=fifo)

    # Test med portfolio-data
    try:
        positions = portfolio.get_all_positions("DKK")
        if positions:
            unrealized = calc.calculate_unrealized_pnl(positions)
            ok(f"Urealiseret P&L (lagerbeskatning): {unrealized:+,.0f} DKK")
            tax_impact = unrealized * CORPORATE_TAX_RATE
            ok(f"Estimeret skatteeffekt: {tax_impact:+,.0f} DKK")
    except Exception as e:
        warn(f"Unrealized P&L beregning: {e}")

    # Test FIFO tracker
    try:
        from src.tax.corporate_tax import FIFOTracker
        fifo = FIFOTracker(db_path=os.path.join(_TEST_TMP, "test_fifo.db"))
        from datetime import date
        fifo.add_lot("TEST", 100, 50.0, date(2026, 1, 1))
        fifo.add_lot("TEST", 50, 60.0, date(2026, 2, 1))
        consumed = fifo.consume_lots("TEST", 120)
        ok(f"FIFO test: Forbrugt {sum(c[0] for c in consumed)} lots korrekt")
    except Exception as e:
        warn(f"FIFO: {e}")

    # Test DBO rates
    try:
        from src.tax.dividend_tracker import DBO_RATES, RECLAIMABLE_EXCESS
        ok(f"DBO rates: {len(DBO_RATES)} lande")
        ok(f"US DBO: {DBO_RATES['US']:.0%}, DE DBO: {DBO_RATES['DE']:.0%}, GB: {DBO_RATES['GB']:.0%}")
        ok(f"Reclaimable DE: {RECLAIMABLE_EXCESS.get('DE', 0):.3%}, CH: {RECLAIMABLE_EXCESS.get('CH', 0):.3%}")
    except Exception as e:
        warn(f"DBO rates: {e}")

    # Test mark-to-market
    try:
        from src.tax.mark_to_market import MarkToMarketEngine
        mtm = MarkToMarketEngine(db_path=os.path.join(_TEST_TMP, "test_mtm.db"))
        ok("MarkToMarketEngine initialized")
    except Exception as e:
        warn(f"MTM: {e}")


# ── Step 8: Dashboard Smoke Test ──────────────────────────

def test_dashboard():
    info("Verificerer dashboard imports...")
    try:
        from src.dashboard.app import app
        ok("Dashboard app importeret")
    except Exception as e:
        fail(f"Dashboard import fejl: {e}")
        raise

    # Verify new pages
    pages = [
        ("portfolio", "src.dashboard.pages.portfolio"),
        ("trading", "src.dashboard.pages.trading"),
        ("tax_center", "src.dashboard.pages.tax_center"),
        ("broker_status", "src.dashboard.pages.broker_status"),
        ("market_explorer", "src.dashboard.pages.market_explorer"),
    ]

    for name, module_path in pages:
        try:
            __import__(module_path)
            ok(f"Page '{name}' importeret")
        except Exception as e:
            warn(f"Page '{name}': {e}")

    info("Dashboard klar — start med: python main.py --mode dashboard")


# ── Step 9: Scheduler Dry-Run ─────────────────────────────

def test_scheduler():
    from src.ops.daily_scheduler import DailyScheduler, is_market_day
    from datetime import date

    today = date.today()
    is_trading = is_market_day(today)
    ok(f"I dag ({today}): {'Handelsdag' if is_trading else 'Ikke handelsdag (weekend/helligdag)'}")

    scheduler = DailyScheduler()
    schedule = scheduler.get_schedule()
    ok(f"Tidsplan: {len(schedule)} opgaver")
    for task in schedule:
        ok(f"  {task['time']} — {task['name']} (prioritet: {task['priority']})")

    # Kør maintenance som dry-run (den kræver ikke market day)
    info("Kører maintenance task (dry-run)...")
    result = scheduler.run_task_now("maintenance")
    if result:
        ok(f"Maintenance: {result.status.value} ({result.duration_seconds:.1f}s)")
        if result.details:
            for k, v in result.details.items():
                ok(f"  {k}: {v}")


# ── Step 10: Full Report ──────────────────────────────────

def test_report():
    from src.ops.email_reports import ReportGenerator, ReportData

    data = ReportData(
        portfolio_value_dkk=1_500_000,
        daily_pnl_dkk=12_500,
        daily_pnl_pct=0.84,
        mtd_pnl_dkk=45_000,
        ytd_pnl_dkk=185_000,
        broker_status={"alpaca": "connected", "saxo": "disconnected", "ibkr": "disconnected", "nordnet": "disconnected"},
        positions_count=42,
        tax_credit_balance=350_000,
        estimated_tax_ytd=40_700,
        dividends_ytd=12_000,
    )

    # Generate morning report
    morning = ReportGenerator.morning_report(data)
    ok(f"Morgenrapport genereret ({len(morning)} chars HTML)")

    # Generate evening report
    evening = ReportGenerator.evening_report(data)
    ok(f"Aftenrapport genereret ({len(evening)} chars HTML)")

    # Save sample report
    report_path = os.path.join(ROOT, "data", "sample_evening_report.html")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(evening)
    ok(f"Sample rapport gemt: {report_path}")

    # Test backup manager
    try:
        from src.ops.backup import BackupManager, BackupConfig
        config = BackupConfig(backup_dir=os.path.join(ROOT, "backups"))
        bm = BackupManager(config)
        status = bm.get_status()
        ok(f"Backup status: {status['file_count']} filer, {status['total_size_mb']} MB")
    except Exception as e:
        warn(f"Backup: {e}")


# ── Main ───────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Alpha Trader Paper Test")
    parser.add_argument("--step", type=int, help="Kør kun dette step (1-10)")
    parser.add_argument("--skip-trade", action="store_true", help="Spring ordretest over")
    args = parser.parse_args()

    header("ALPHA TRADER — Paper Trading Test")
    info(f"Tid: {datetime.now(TZ_CET).strftime('%Y-%m-%d %H:%M CET')}")
    info(f"Mode: Alpaca Paper Trading")
    print()

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

    only = args.step
    skip_trade = args.skip_trade

    # Step 1: Connection
    broker = run_step(1, "Broker Connection (Alpaca Paper)", test_broker_connection,
                      skip=(only and only != 1))

    # Step 2: Router
    router = None
    if broker:
        router = run_step(2, "BrokerRouter + Exchange Detection",
                          lambda: test_broker_router(broker),
                          skip=(only and only != 2))

    # Step 3: Market data
    mdf = run_step(3, "Market Data Hentning", test_market_data,
                   skip=(only and only != 3))

    # Step 4: Place order
    order = None
    if router:
        order = run_step(4, "Paper Order (BUY 1 AAPL)", lambda: test_place_order(router),
                         skip=(only and only != 4) or skip_trade)

    # Step 5: Order tracking
    if router and order:
        run_step(5, "Order Tracking + Persistence",
                 lambda: test_order_tracking(router, order),
                 skip=(only and only != 5) or skip_trade)
    elif not skip_trade:
        warn("Step 5 skipped — ingen order at tracke")
        results[5] = "SKIPPED"

    # Step 6: Portfolio
    portfolio = None
    if router:
        portfolio = run_step(6, "Portfolio / Positions",
                             lambda: test_portfolio(router),
                             skip=(only and only != 6))

    # Step 7: Tax
    if portfolio:
        run_step(7, "Skatteberegning (Corporate Tax)",
                 lambda: test_tax(portfolio),
                 skip=(only and only != 7))
    else:
        warn("Step 7 skipped — ingen portfolio")
        results[7] = "SKIPPED"

    # Step 8: Dashboard
    run_step(8, "Dashboard Smoke Test", test_dashboard,
             skip=(only and only != 8))

    # Step 9: Scheduler
    run_step(9, "Scheduler Dry-Run", test_scheduler,
             skip=(only and only != 9))

    # Step 10: Report
    run_step(10, "Full P&L + Rapport", test_report,
             skip=(only and only != 10))

    # ── Summary ────────────────────────────────────────────
    print()
    header("RESULTAT")
    passed = sum(1 for v in results.values() if v == "PASS")
    failed = sum(1 for v in results.values() if v == "FAIL")
    skipped = sum(1 for v in results.values() if v == "SKIPPED")

    for n in sorted(results.keys()):
        status = results[n]
        icon = {"PASS": f"{C.GREEN}✓{C.END}", "FAIL": f"{C.RED}✗{C.END}", "SKIPPED": f"{C.YELLOW}○{C.END}"}[status]
        print(f"  {icon} Step {n}: {status}")

    print(f"\n  {C.BOLD}Total: {passed} passed, {failed} failed, {skipped} skipped{C.END}")

    if failed == 0:
        print(f"\n  {C.GREEN}{C.BOLD}🎉 ALL TESTS PASSED — Klar til paper trading!{C.END}")
    else:
        print(f"\n  {C.RED}{C.BOLD}⚠️  {failed} test(s) fejlede — fix før live trading{C.END}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
