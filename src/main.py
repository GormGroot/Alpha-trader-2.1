"""
Alpha Trading Platform – Hovedprogram.

CLI-kommandoer:
    python -m src.main run          Kør botten (paper trading)
    python -m src.main backtest     Kør backtests for alle strategier
    python -m src.main dashboard    Start web-dashboardet
    python -m src.main status       Vis positioner og P&L
    python -m src.main simulate     Kør én cyklus og vis resultater
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Sørg for at projektets rod er i sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

# Log-mappe
(_ROOT / "logs").mkdir(exist_ok=True)
logger.add(
    str(_ROOT / "logs" / "trading_{time}.log"),
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

from config.settings import settings
from src.data.market_data import MarketDataFetcher
from src.data.indicators import add_all_indicators
from src.data.pipeline import DataPipeline
from src.strategy.base_strategy import Signal
from src.strategy.sma_crossover import SMACrossoverStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.combined_strategy import CombinedStrategy
from src.strategy.signal_engine import SignalEngine, SymbolSignal
from src.risk.portfolio_tracker import PortfolioTracker
from src.risk.risk_manager import RiskManager
from src.broker.paper_broker import PaperBroker
from src.broker.models import OrderStatus


# ══════════════════════════════════════════════════════════════
#  TradingBot
# ══════════════════════════════════════════════════════════════


class TradingBot:
    """
    Hovedklasse der binder alle moduler sammen:
      DataPipeline → SignalEngine → RiskManager → PaperBroker

    Kan køre én cyklus (cycle) eller kontinuerligt (run).
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        initial_capital: float = 100_000,
    ) -> None:
        self.symbols = symbols or settings.trading.symbols
        self._running = False
        self._stop_event = threading.Event()

        # Markedsdata
        self._market_data = MarketDataFetcher()
        self._pipeline = DataPipeline(symbols=self.symbols)

        # Portfolio & risiko
        self._portfolio = PortfolioTracker(initial_capital)
        self._risk = RiskManager(self._portfolio)

        # Broker (paper)
        self._broker = PaperBroker(
            initial_capital=initial_capital,
            market_data=self._market_data,
            portfolio=self._portfolio,
        )

        # Strategier
        sma = SMACrossoverStrategy(
            short_window=settings.strategy.sma_short_window,
            long_window=settings.strategy.sma_long_window,
        )
        rsi = RSIStrategy(
            oversold=settings.strategy.rsi_oversold,
            overbought=settings.strategy.rsi_overbought,
        )
        self._strategies = [(sma, 0.6), (rsi, 0.4)]
        self._engine = SignalEngine(
            strategies=self._strategies,
            min_agreement=1,
            portfolio_value=initial_capital,
            max_position_pct=settings.risk.max_position_pct,
        )

        logger.info(
            f"TradingBot initialiseret: {len(self.symbols)} aktier, "
            f"${initial_capital:,.0f} kapital, "
            f"{len(self._strategies)} strategier"
        )

    # ── Én handelscyklus ──────────────────────────────────────

    def cycle(self) -> dict:
        """
        Kør én komplet handelscyklus:
          1. Hent markedsdata
          2. Generer signaler
          3. Tjek risiko for exits
          4. Udfør nye handler
          5. Returnér resultater

        Returns:
            Dict med info om cyklen.
        """
        cycle_start = time.time()
        result = {
            "timestamp": datetime.now().isoformat(),
            "signals": [],
            "orders": [],
            "exits": [],
            "errors": [],
        }

        # 1 ─ Hent data
        try:
            data = self._pipeline.run_once()
            logger.info(f"Data hentet for {len(data)} symboler")
        except Exception as exc:
            msg = f"Fejl ved datahentning: {exc}"
            logger.error(msg)
            result["errors"].append(msg)
            return result

        # 2 ─ Generer signaler
        try:
            engine_result = self._engine.process(
                data, portfolio_value=self._portfolio.total_equity
            )
            for sig in engine_result.signals:
                result["signals"].append({
                    "symbol": sig.symbol,
                    "signal": sig.signal.value,
                    "confidence": sig.confidence,
                    "reason": sig.reason,
                })
            logger.info(
                f"Signaler: {len(engine_result.buys)} KØB, "
                f"{len(engine_result.sells)} SÆLG, "
                f"{len(engine_result.signals) - len(engine_result.actionable)} HOLD"
            )
        except Exception as exc:
            msg = f"Fejl i signalmotor: {exc}"
            logger.error(msg)
            result["errors"].append(msg)
            return result

        # 3 ─ Tjek eksisterende positioner for exits
        try:
            prices = {}
            for sym in self._portfolio.positions:
                try:
                    prices[sym] = self._market_data.get_latest_price(sym)
                except Exception:
                    pass

            if prices:
                self._portfolio.update_prices(prices)
                exits = self._risk.check_positions(prices)
                for exit_sig in exits:
                    try:
                        order = self._broker.sell(
                            exit_sig.symbol,
                            self._portfolio.positions[exit_sig.symbol].qty
                            if exit_sig.symbol in self._portfolio.positions
                            else 0,
                        )
                        result["exits"].append({
                            "symbol": exit_sig.symbol,
                            "reason": exit_sig.reason,
                            "price": exit_sig.price,
                            "order_id": order.order_id,
                        })
                        logger.warning(
                            f"EXIT {exit_sig.symbol}: {exit_sig.reason} "
                            f"@ ${exit_sig.price:.2f}"
                        )
                    except Exception as exc:
                        msg = f"Exit-fejl for {exit_sig.symbol}: {exc}"
                        logger.error(msg)
                        result["errors"].append(msg)
        except Exception as exc:
            msg = f"Fejl ved positions-tjek: {exc}"
            logger.error(msg)
            result["errors"].append(msg)

        # 4 ─ Udfør nye handler (KØB)
        for sig in engine_result.buys:
            try:
                price = self._market_data.get_latest_price(sig.symbol)
                decision = self._risk.check_order(
                    sig.symbol, "long", sig.position_size_usd, price
                )

                if not decision.approved:
                    logger.info(
                        f"AFVIST {sig.symbol}: {decision.message}"
                    )
                    continue

                qty = decision.adjusted_qty
                if qty <= 0:
                    continue

                order = self._broker.buy(sig.symbol, qty)
                result["orders"].append({
                    "symbol": sig.symbol,
                    "side": "BUY",
                    "qty": qty,
                    "price": order.filled_avg_price,
                    "order_id": order.order_id,
                    "confidence": sig.confidence,
                })
                logger.info(
                    f"KØB {qty} {sig.symbol} @ ${order.filled_avg_price:.2f} "
                    f"(confidence={sig.confidence:.0f}%)"
                )
            except Exception as exc:
                msg = f"Ordre-fejl for {sig.symbol}: {exc}"
                logger.error(msg)
                result["errors"].append(msg)

        # 5 ─ Udfør SÆLG-signaler
        for sig in engine_result.sells:
            if sig.symbol not in self._portfolio.positions:
                continue
            try:
                pos = self._portfolio.positions[sig.symbol]
                order = self._broker.sell(sig.symbol, pos.qty)
                result["orders"].append({
                    "symbol": sig.symbol,
                    "side": "SELL",
                    "qty": pos.qty,
                    "price": order.filled_avg_price,
                    "order_id": order.order_id,
                    "confidence": sig.confidence,
                })
                logger.info(
                    f"SÆLG {pos.qty} {sig.symbol} @ ${order.filled_avg_price:.2f} "
                    f"(confidence={sig.confidence:.0f}%)"
                )
            except Exception as exc:
                msg = f"Sælg-fejl for {sig.symbol}: {exc}"
                logger.error(msg)
                result["errors"].append(msg)

        elapsed = (time.time() - cycle_start) * 1000
        result["duration_ms"] = elapsed
        logger.info(
            f"Cyklus færdig: {len(result['orders'])} handler, "
            f"{len(result['exits'])} exits, {elapsed:.0f}ms"
        )
        return result

    # ── Kontinuerlig kørsel ───────────────────────────────────

    def run(self, interval_seconds: int | None = None) -> None:
        """
        Kør botten kontinuerligt med interval mellem cykler.
        Stop med Ctrl+C eller stop()-metoden.
        """
        interval = interval_seconds or settings.trading.check_interval_seconds
        self._running = True
        self._stop_event.clear()

        logger.info(
            f"Bot startet – kører hvert {interval}. sekund "
            f"({len(self.symbols)} aktier)"
        )

        cycle_count = 0
        while self._running and not self._stop_event.is_set():
            cycle_count += 1
            logger.info(f"═══ Cyklus #{cycle_count} ═══")

            try:
                self.cycle()
            except Exception as exc:
                logger.error(f"Uventet fejl i cyklus #{cycle_count}: {exc}")
                if self._risk.is_trading_halted:
                    logger.critical("Trading er stoppet af risikostyring!")
                    break

            self._log_status()
            self._stop_event.wait(timeout=interval)

        logger.info(f"Bot stoppet efter {cycle_count} cykler")

    def stop(self) -> None:
        """Stop botten pænt."""
        logger.info("Stopper botten...")
        self._running = False
        self._stop_event.set()

    # ── Status ────────────────────────────────────────────────

    def status(self) -> dict:
        """Returnér aktuel status som dict."""
        return self._portfolio.summary()

    def print_status(self) -> None:
        """Print formateret status til konsollen."""
        s = self.status()
        acct = self._broker.get_account()

        print("\n" + "═" * 60)
        print("  ALPHA TRADING PLATFORM – STATUS")
        print("═" * 60)
        print(f"  Tidspunkt:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Broker:           {self._broker.name}")
        print(f"  Trading stoppet:  {'JA ⚠️' if self._risk.is_trading_halted else 'Nej ✓'}")
        print()
        print(f"  💰 Total equity:  ${s['total_equity']:>12,.2f}")
        print(f"  💵 Kontanter:     ${s['cash']:>12,.2f}")
        print(f"  📈 Urealiseret:   ${s['unrealized_pnl']:>+12,.2f}")
        print(f"  📊 Realiseret:    ${s['realized_pnl']:>+12,.2f}")
        print(f"  📅 Daglig P&L:    ${s['daily_pnl']:>+12,.2f} ({s['daily_pnl_pct']:+.2f}%)")
        print(f"  📉 Max drawdown:  {s['max_drawdown_pct']:>11.2f}%")
        print(f"  🏆 Total afkast:  {s['total_return_pct']:>+11.2f}%")
        print()

        positions = self._broker.get_positions()
        if positions:
            print(f"  Åbne positioner ({len(positions)}):")
            print(f"  {'Symbol':<8} {'Side':<6} {'Antal':>6} {'Entry':>10} {'Nu':>10} {'P&L':>12}")
            print(f"  {'─'*8} {'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*12}")
            for p in positions:
                pnl = p.unrealized_pnl
                pnl_str = f"${pnl:>+,.2f}"
                print(
                    f"  {p.symbol:<8} {p.side:<6} {p.qty:>6.0f} "
                    f"${p.entry_price:>9.2f} ${p.current_price:>9.2f} {pnl_str:>12}"
                )
        else:
            print("  Ingen åbne positioner")

        trades = self._portfolio.closed_trades
        if trades:
            print(f"\n  Lukkede handler ({len(trades)}):")
            print(f"  {'Symbol':<8} {'Ind':>10} {'Ud':>10} {'P&L':>12} {'Årsag':<15}")
            print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*12} {'─'*15}")
            for t in trades[-10:]:
                pnl_str = f"${t.realized_pnl:>+,.2f}"
                print(
                    f"  {t.symbol:<8} ${t.entry_price:>9.2f} "
                    f"${t.exit_price:>9.2f} {pnl_str:>12} {t.exit_reason:<15}"
                )

        print(f"\n  Win rate:         {s['win_rate']:.0%}")
        print(f"  Profit factor:    {s['profit_factor']:.2f}")
        print(f"  Sharpe ratio:     {s['sharpe_ratio']:.2f}")
        print("═" * 60 + "\n")

    def _log_status(self) -> None:
        """Log kort status."""
        s = self.status()
        logger.info(
            f"Status: equity=${s['total_equity']:,.0f}, "
            f"cash=${s['cash']:,.0f}, "
            f"positioner={s['positions']}, "
            f"P&L=${s['daily_pnl']:+,.2f}"
        )


# ══════════════════════════════════════════════════════════════
#  CLI-kommandoer
# ══════════════════════════════════════════════════════════════


def cmd_run(args: argparse.Namespace) -> None:
    """Kør botten kontinuerligt."""
    bot = TradingBot(
        symbols=args.symbols.split(",") if args.symbols else None,
        initial_capital=args.capital,
    )

    # Graceful shutdown
    def _shutdown(signum, frame):
        logger.info(f"Signal {signum} modtaget – stopper...")
        bot.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start dashboard i separat tråd
    if not args.no_dashboard:
        dash_thread = threading.Thread(target=_start_dashboard, daemon=True)
        dash_thread.start()
        logger.info("Dashboard startet på http://localhost:8050")

    bot.run(interval_seconds=args.interval)
    bot.print_status()


def cmd_simulate(args: argparse.Namespace) -> None:
    """Kør én cyklus og vis resultater."""
    symbols = args.symbols.split(",") if args.symbols else ["AAPL", "MSFT", "GOOGL"]
    bot = TradingBot(symbols=symbols, initial_capital=args.capital)

    print(f"\n🤖 Kører simulering med {len(bot.symbols)} aktier...")
    print(f"   Aktier: {', '.join(bot.symbols)}")
    print(f"   Kapital: ${args.capital:,.0f}\n")

    result = bot.cycle()

    # Vis signaler
    print("📡 Signaler:")
    for sig in result["signals"]:
        icon = "🟢" if sig["signal"] == "BUY" else "🔴" if sig["signal"] == "SELL" else "⚪"
        print(f"   {icon} {sig['symbol']}: {sig['signal']} (confidence={sig['confidence']:.0f}%)")
        print(f"      {sig['reason']}")

    # Vis handler
    if result["orders"]:
        print(f"\n📋 Handler ({len(result['orders'])}):")
        for o in result["orders"]:
            icon = "🟢" if o["side"] == "BUY" else "🔴"
            print(f"   {icon} {o['side']} {o['qty']:.0f} {o['symbol']} @ ${o['price']:.2f}")
    else:
        print("\n📋 Ingen handler udført (strategierne signalerer HOLD)")

    # Vis exits
    if result["exits"]:
        print(f"\n🚪 Exits ({len(result['exits'])}):")
        for e in result["exits"]:
            print(f"   {e['symbol']}: {e['reason']} @ ${e['price']:.2f}")

    # Vis fejl
    if result["errors"]:
        print(f"\n⚠️  Fejl ({len(result['errors'])}):")
        for err in result["errors"]:
            print(f"   {err}")

    # Vis risiko-advarsler
    if bot._risk.is_trading_halted:
        print("\n🚨 ADVARSEL: Trading er stoppet af risikostyring!")

    bot.print_status()
    print(f"⏱  Cyklus tog {result.get('duration_ms', 0):.0f}ms\n")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Kør backtests for alle strategier."""
    from src.backtest.backtester import Backtester

    symbols = args.symbols.split(",") if args.symbols else settings.trading.symbols
    mdf = MarketDataFetcher()

    strategies = [
        SMACrossoverStrategy(
            short_window=settings.strategy.sma_short_window,
            long_window=settings.strategy.sma_long_window,
        ),
        RSIStrategy(
            oversold=settings.strategy.rsi_oversold,
            overbought=settings.strategy.rsi_overbought,
        ),
        CombinedStrategy(
            strategies=[
                (SMACrossoverStrategy(short_window=20, long_window=50), 0.6),
                (RSIStrategy(oversold=30, overbought=70), 0.4),
            ],
            min_agreement=1,
        ),
    ]

    print(f"\n📊 Kører backtests: {len(strategies)} strategier × {len(symbols)} aktier")
    print(f"   Periode: {args.start} → {args.end}")
    print(f"   Kapital: ${args.capital:,.0f}\n")

    for strat in strategies:
        bt = Backtester(
            strategy=strat,
            symbols=symbols,
            start=args.start,
            end=args.end,
            initial_capital=args.capital,
            commission_pct=0.001,
            spread_pct=0.0005,
            market_data=mdf,
        )
        result = bt.run()
        print(result.summary_table())
        print()


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Start dashboardet."""
    print("🖥  Starter dashboard på http://localhost:8050 ...")
    _start_dashboard()


def cmd_status(args: argparse.Namespace) -> None:
    """Vis status (kræver kørende bot – viser demo-status)."""
    bot = TradingBot(
        symbols=args.symbols.split(",") if args.symbols else ["AAPL", "MSFT", "GOOGL"],
        initial_capital=args.capital,
    )
    bot.print_status()


def cmd_tax(args: argparse.Namespace) -> None:
    """Generer skatteindberetningsrapport."""
    from src.tax.tax_report import TaxReportGenerator
    from src.tax.transaction_log import TransactionLog
    from src.tax.currency import CurrencyConverter
    from src.backtest.backtester import Backtester

    year = args.year
    print(f"\n🧾 Genererer skatteindberetning for {year}...")

    generator = TaxReportGenerator(
        progression_limit=settings.tax.progression_limit,
        carried_losses=settings.tax.carried_losses,
    )

    # Hvis der ikke er nogen loggede transaktioner, kør backtest for demo-data
    df = generator.transaction_log.get_transactions(year=year)
    if df.empty:
        print(f"   Ingen loggede handler for {year}.")
        print(f"   Kører backtest for at generere demo-data...\n")

        symbols = args.symbols.split(",") if args.symbols else settings.trading.symbols
        mdf = MarketDataFetcher()

        strat = CombinedStrategy(
            strategies=[
                (SMACrossoverStrategy(short_window=20, long_window=50), 0.6),
                (RSIStrategy(oversold=30, overbought=70), 0.4),
            ],
            min_agreement=1,
        )
        bt = Backtester(
            strategy=strat, symbols=symbols,
            start=f"{year}-01-01", end=f"{year}-12-31",
            initial_capital=args.capital,
            commission_pct=0.001, spread_pct=0.0005,
            market_data=mdf,
        )
        result = bt.run()

        # Log handler til transaktionsloggen
        from src.risk.portfolio_tracker import ClosedTrade
        for t in result.trades:
            ct = ClosedTrade(
                symbol=t.symbol, side=t.side, qty=t.qty,
                entry_price=t.entry_price, exit_price=t.exit_price,
                entry_time=t.entry_date, exit_time=t.exit_date,
                exit_reason=t.exit_reason,
            )
            generator.transaction_log.log_trade(ct)

    # Generer rapport
    report = generator.generate(year=year)
    generator.print_report(report)

    print(f"\n📁 Filer gemt:")
    print(f"   CSV: {report.transactions_csv_path}")
    print(f"   TXT: {report.report_txt_path}")
    print()


def cmd_tax_advisor(args: argparse.Namespace) -> None:
    """Kør skatterådgiver med proaktive advarsler."""
    from src.tax.tax_advisor import TaxAdvisor
    from src.tax.tax_report import TaxReportGenerator
    from src.tax.tax_calculator import DanishTaxCalculator
    from src.backtest.backtester import Backtester
    from src.notifications.notifier import Notifier

    year = args.year
    print(f"\n🧾 Skatterådgiver – analyse for {year}...")
    print("   ⚠️ Vejledende beregning – verificér med revisor.\n")

    # ── 1. Hent transaktionsdata ──
    generator = TaxReportGenerator(
        progression_limit=settings.tax.progression_limit,
        carried_losses=settings.tax.carried_losses,
    )

    df = generator.transaction_log.get_transactions(year=year)
    if df.empty:
        print(f"   Ingen loggede handler for {year}.")
        print(f"   Kører backtest for at generere demo-data...\n")

        symbols = args.symbols.split(",") if args.symbols else settings.trading.symbols
        mdf = MarketDataFetcher()
        strat = CombinedStrategy(
            strategies=[
                (SMACrossoverStrategy(short_window=20, long_window=50), 0.6),
                (RSIStrategy(oversold=30, overbought=70), 0.4),
            ],
            min_agreement=1,
        )
        bt = Backtester(
            strategy=strat, symbols=symbols,
            start=f"{year}-01-01", end=f"{year}-12-31",
            initial_capital=args.capital,
            commission_pct=0.001, spread_pct=0.0005,
            market_data=mdf,
        )
        result = bt.run()

        from src.risk.portfolio_tracker import ClosedTrade
        for t in result.trades:
            ct = ClosedTrade(
                symbol=t.symbol, side=t.side, qty=t.qty,
                entry_price=t.entry_price, exit_price=t.exit_price,
                entry_time=t.entry_date, exit_time=t.exit_date,
                exit_reason=t.exit_reason,
            )
            generator.transaction_log.log_trade(ct)

        df = generator.transaction_log.get_transactions(year=year)

    transactions = df.to_dict("records") if not df.empty else []

    # ── 2. Opret rådgiver ──
    advisor = TaxAdvisor(
        progression_limit=settings.tax.progression_limit,
        carried_losses=settings.tax.carried_losses,
        fx_rate=settings.tax.fallback_fx_rate,
    )

    # ── 3. Kvartalsestimat ──
    print("─" * 65)
    print("  KVARTALSESTIMAT")
    print("─" * 65)
    q_est = advisor.quarterly_estimate(transactions, year=year)
    print(f"  Kvartal:            Q{q_est.quarter} {year}")
    print(f"  Handler YTD:        {q_est.num_trades_ytd}")
    print(f"  Netto YTD:          {q_est.net_ytd_dkk:+,.2f} DKK")
    print(f"  Skat YTD:           {q_est.tax_ytd_dkk:,.2f} DKK")
    print()
    print(f"  📊 Projektion for hele {year}:")
    print(f"  Forventet gevinst:  {q_est.projected_annual_gain_dkk:+,.2f} DKK")
    print(f"  Forventet skat:     {q_est.projected_annual_tax_dkk:,.2f} DKK")
    print(f"  Effektiv sats:      {q_est.projected_effective_rate:.1f}%")
    print()
    print(f"  Progressionsgrænse: {q_est.pct_of_limit_used:.0f}% brugt "
          f"({q_est.remaining_before_42pct:,.0f} DKK tilbage)")
    if q_est.projected_hits_limit:
        print(f"  ⚠️ Du rammer sandsynligvis progressionsgrænsen!")
        if q_est.projected_limit_date:
            print(f"     Estimeret dato: {q_est.projected_limit_date}")
    print()

    # ── 4. Advarsler ──
    alerts = advisor.collect_pending_alerts(transactions, year=year)
    if alerts:
        print("─" * 65)
        print("  ADVARSLER")
        print("─" * 65)
        for alert in alerts:
            icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(
                alert.severity, "⚪"
            )
            print(f"  {icon} [{alert.severity}] {alert.title}")
            for line in alert.message.split("\n")[:4]:
                print(f"     {line}")
            print()

    # ── 5. Årsafslutning (altid vis) ──
    print("─" * 65)
    print("  ÅRSAFSLUTNINGSANALYSE")
    print("─" * 65)
    year_report = advisor.year_end_report(
        transactions=transactions, year=year,
    )
    for action in year_report.actions:
        print(f"  → {action}")
    print()

    if year_report.deadlines:
        print("─" * 65)
        print("  VIGTIGE DEADLINES")
        print("─" * 65)
        for dl in year_report.deadlines:
            print(f"  📅 {dl}")
        print()

    # ── 6. Audit trail ──
    print("─" * 65)
    print("  AUDIT TRAIL (til revisor)")
    print("─" * 65)
    for note in q_est.audit_notes:
        print(f"  📝 {note}")
    for note in year_report.audit_notes:
        print(f"  📝 {note}")
    print()

    # ── 7. Send notifikationer ──
    notifier = Notifier()
    for alert in alerts:
        notifier.send_tax_alert(alert)

    print("═" * 65)
    print("  ⚠️ Alle beregninger er vejledende – verificér med revisor.")
    print("═" * 65)
    print()


def cmd_universe(args: argparse.Namespace) -> None:
    """Vis og administrér aktivuniverset."""
    from src.data.universe import AssetUniverse

    # Opret univers fra config
    enabled = [
        cat for cat, active in settings.universe.categories.items()
        if active
    ]
    universe = AssetUniverse(
        enabled_categories=enabled,
        watchlist=settings.universe.watchlist or None,
    )

    if args.category:
        # Vis symboler for én kategori
        symbols = universe.get_symbols_for_category(args.category)
        cat = universe.get_category(args.category)
        if not cat:
            print(f"❌ Ukendt kategori: {args.category}")
            print(f"   Tilgængelige: {', '.join(c.name for c in universe.all_categories)}")
            return
        print(f"\n{cat.display_name} ({cat.symbol_count} symboler):")
        print(f"  {cat.description}")
        for sub_name, items in cat.subcategories.items():
            syms = [
                item[0] if isinstance(item, tuple) else item
                for item in items
            ]
            print(f"\n  [{sub_name}] ({len(syms)} stk.):")
            for i in range(0, len(syms), 10):
                print(f"    {', '.join(syms[i:i+10])}")
    elif args.scan:
        # Vis scan-symboler
        symbols = universe.scan_universe(
            max_symbols=settings.universe.max_symbols_per_scan,
        )
        print(f"\n🔍 Scan-mode: {len(symbols)} symboler klar til screening")
        for i in range(0, len(symbols), 15):
            print(f"  {', '.join(symbols[i:i+15])}")
    else:
        # Vis oversigt
        universe.print_summary()

    print()


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan markeder med MarketScanner."""
    from src.strategy.market_scanner import (
        MarketScanner, SECTOR_ETF_MAP,
        VIX_SYMBOL, DXY_SYMBOL, GOLD_SYMBOL, OIL_SYMBOL,
        SP500_SYMBOL, YIELD_2Y, YIELD_10Y,
    )
    from src.data.universe import AssetUniverse

    import pandas as _pd  # noqa: F811 – lokal alias for klarhed

    mdf = MarketDataFetcher()
    scanner = MarketScanner()

    # Bestem hvilke symboler der skal scannes
    if args.sector:
        sector_map = {v.lower(): k for k, v in SECTOR_ETF_MAP.items()}
        etf = sector_map.get(args.sector.lower())
        if etf:
            scan_symbols = [etf]
            print(f"\n🔍 Scanner sektor: {SECTOR_ETF_MAP[etf]} ({etf})")
        else:
            etf_upper = args.sector.upper()
            if etf_upper in SECTOR_ETF_MAP:
                scan_symbols = [etf_upper]
                print(f"\n🔍 Scanner sektor: {SECTOR_ETF_MAP[etf_upper]}")
            else:
                print(f"❌ Ukendt sektor: {args.sector}")
                print(f"   Tilgængelige: {', '.join(SECTOR_ETF_MAP.values())}")
                return
    elif args.asset_class:
        enabled = [
            cat for cat, active in settings.universe.categories.items()
            if active
        ]
        universe = AssetUniverse(enabled_categories=enabled)
        scan_symbols = universe.filter_by_asset_class(args.asset_class)
        if not scan_symbols:
            print(f"❌ Ukendt aktivklasse: {args.asset_class}")
            print("   Tilgængelige: stocks, etfs, commodities, crypto, bonds, forex")
            return
        scan_symbols = scan_symbols[:50]
        print(f"\n🔍 Scanner {args.asset_class}: {len(scan_symbols)} symboler")
    else:
        enabled = [
            cat for cat, active in settings.universe.categories.items()
            if active
        ]
        universe = AssetUniverse(enabled_categories=enabled)
        scan_symbols = universe.scan_universe(
            max_symbols=settings.universe.max_symbols_per_scan,
        )
        print(f"\n🔍 Fuld markedsscanning: {len(scan_symbols)} symboler")

    print("   Henter data...\n")

    # Hent asset-data
    asset_data: dict = {}
    for sym in scan_symbols:
        try:
            df = mdf.get_historical(sym, interval="1d", lookback_days=365)
            if not df.empty:
                df = add_all_indicators(df)
                asset_data[sym] = df
        except Exception as exc:
            logger.debug(f"Kunne ikke hente {sym}: {exc}")

    # Hent sektor-data
    sector_data: dict = {}
    for etf in SECTOR_ETF_MAP:
        if etf in asset_data:
            sector_data[etf] = asset_data[etf]
        else:
            try:
                df = mdf.get_historical(etf, interval="1d", lookback_days=365)
                if not df.empty:
                    df = add_all_indicators(df)
                sector_data[etf] = df
            except Exception:
                sector_data[etf] = _pd.DataFrame()

    # Hent makro-data
    macro_symbols = [VIX_SYMBOL, DXY_SYMBOL, GOLD_SYMBOL, OIL_SYMBOL,
                     SP500_SYMBOL, YIELD_2Y, YIELD_10Y]
    macro_data: dict = {}
    for sym in macro_symbols:
        try:
            macro_data[sym] = mdf.get_historical(sym, interval="1d", lookback_days=365)
        except Exception:
            macro_data[sym] = _pd.DataFrame()

    benchmark = macro_data.get(SP500_SYMBOL)
    result = scanner.full_scan(
        asset_data, sector_data, macro_data, benchmark=benchmark,
    )
    scanner.print_scan_result(result)


def cmd_macro(args: argparse.Namespace) -> None:
    """Vis makro-dashboard."""
    from src.strategy.market_scanner import (
        MarketScanner, SECTOR_ETF_MAP,
        VIX_SYMBOL, DXY_SYMBOL, GOLD_SYMBOL, OIL_SYMBOL,
        SP500_SYMBOL, YIELD_2Y, YIELD_10Y,
    )

    import pandas as _pd  # noqa: F811

    mdf = MarketDataFetcher()
    scanner = MarketScanner()

    print("\n🌍 Henter makro-data...\n")

    macro_symbols = [VIX_SYMBOL, DXY_SYMBOL, GOLD_SYMBOL, OIL_SYMBOL,
                     SP500_SYMBOL, YIELD_2Y, YIELD_10Y]
    macro_data: dict = {}
    for sym in macro_symbols:
        try:
            macro_data[sym] = mdf.get_historical(sym, interval="1d", lookback_days=365)
        except Exception:
            macro_data[sym] = _pd.DataFrame()

    macro = scanner.macro_snapshot(macro_data)
    scanner.print_macro(macro)

    # Vis sektor-analyse
    sector_data: dict = {}
    for etf in SECTOR_ETF_MAP:
        try:
            df = mdf.get_historical(etf, interval="1d", lookback_days=365)
            if not df.empty:
                df = add_all_indicators(df)
            sector_data[etf] = df
        except Exception:
            sector_data[etf] = _pd.DataFrame()

    benchmark = macro_data.get(SP500_SYMBOL)
    sectors = scanner.analyze_sectors(sector_data, benchmark=benchmark)

    print("\n" + "─" * 70)
    print("  📊 SEKTOR-PERFORMANCE")
    print("─" * 70)
    print(f"  {'Sektor':<22} {'1d':>6} {'1u':>6} {'1m':>6} {'3m':>6} {'RS':>6} {'Trend':>8}")
    print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
    for s in sectors:
        trend_icon = "↑" if s.trend == "up" else "↓" if s.trend == "down" else "→"
        print(
            f"  {s.name:<22} {s.change_1d:>+5.1f}% {s.change_1w:>+5.1f}% "
            f"{s.change_1m:>+5.1f}% {s.change_3m:>+5.1f}% "
            f"{s.relative_strength_1m:>+5.1f}  {trend_icon:>7}"
        )

    advice = scanner.sector_rotation_advice(sectors)
    if advice:
        print("\n  Rotation-forslag:")
        for a in advice:
            print(f"    → {a}")

    alloc = scanner.recommend_allocation(macro, sectors)
    print(f"\n  💼 Anbefalet allokering: "
          f"Aktier {alloc.stocks_pct:.0f}% | Obligationer {alloc.bonds_pct:.0f}% | "
          f"Råstoffer {alloc.commodities_pct:.0f}% | Krypto {alloc.crypto_pct:.0f}% | "
          f"Kontanter {alloc.cash_pct:.0f}%")
    print(f"     {alloc.rationale}")
    print("═" * 70 + "\n")


def cmd_stress_test(args: argparse.Namespace) -> None:
    """Kør stress-test mod porteføljen."""
    from src.backtest.stress_test import (
        StressTester, HISTORICAL_CRISES, SYNTHETIC_SCENARIOS,
    )

    symbols = args.symbols.split(",") if args.symbols else settings.trading.symbols
    weights = {s.upper(): 1.0 / len(symbols) for s in symbols}

    tester = StressTester(
        portfolio_weights=weights,
        initial_value=args.capital,
    )

    if args.scenario:
        # Kør ét specifikt scenarie
        key = args.scenario.lower().replace("-", "_").replace(" ", "_")
        result = tester.run_single(key)
        if result is None:
            print(f"\n❌ Ukendt scenarie: {args.scenario}")
            print(f"   Tilgængelige historiske: {', '.join(HISTORICAL_CRISES.keys())}")
            print(f"   Tilgængelige syntetiske: {', '.join(SYNTHETIC_SCENARIOS.keys())}")
            return

        print(f"\n{'═' * 65}")
        print(f"  STRESS-TEST: {result.scenario.name}")
        print(f"{'═' * 65}")
        print(f"  {result.scenario.description}")
        print()
        print(f"  Portefølje:      {', '.join(f'{s} ({w:.0%})' for s, w in weights.items())}")
        print(f"  Startværdi:      ${tester.initial_value:,.0f}")
        print(f"  Slutværdi:       ${result.portfolio_value_end:,.0f}")
        print(f"  Max drawdown:    {result.max_drawdown_pct:+.1f}%")
        print(f"  Worst dag:       {result.worst_day_pct:+.1f}% ({result.worst_day_date})")
        print(f"  Recovery tid:    ~{result.recovery_days} dage")
        print()
        print(f"  📊 Med risikostyring:    ${result.with_risk_mgmt_end:>10,.0f}")
        print(f"  📉 Uden risikostyring:   ${result.without_risk_mgmt_end:>10,.0f}")
        print(f"  💰 RM reddede:           {result.risk_mgmt_saved_pct:+.1f}%")
        print()
        print(f"  Regime-aktioner:")
        for action in result.regime_actions:
            print(f"    → {action}")
        print()
        print(f"  Nøgle-events:")
        for event in result.scenario.key_events:
            print(f"    📅 {event}")
        print(f"{'═' * 65}\n")

    elif args.monte_carlo:
        # Kør Monte Carlo
        runs = args.runs
        print(f"\n🎲 Monte Carlo simulation: {runs:,} scenarier...")
        print(f"   Portefølje: {', '.join(f'{s} ({w:.0%})' for s, w in weights.items())}")
        print(f"   Kapital: ${args.capital:,.0f}")
        print(f"   Horisont: 252 handelsdage (1 år)\n")

        mc = tester.monte_carlo(
            num_simulations=runs,
            horizon_days=252,
            seed=None,  # Forskellige resultater hver gang
        )

        print(f"{'═' * 55}")
        print(f"  MONTE CARLO RESULTATER")
        print(f"{'═' * 55}")
        print(f"  Worst case (1%):    ${mc.worst_case:>12,.0f}  "
              f"({(mc.worst_case / mc.initial_value - 1) * 100:+.1f}%)")
        print(f"  5. percentil:       ${mc.percentile_5:>12,.0f}  "
              f"({(mc.percentile_5 / mc.initial_value - 1) * 100:+.1f}%)")
        print(f"  25. percentil:      ${mc.percentile_25:>12,.0f}  "
              f"({(mc.percentile_25 / mc.initial_value - 1) * 100:+.1f}%)")
        print(f"  Median:             ${mc.median:>12,.0f}  "
              f"({(mc.median / mc.initial_value - 1) * 100:+.1f}%)")
        print(f"  75. percentil:      ${mc.percentile_75:>12,.0f}  "
              f"({(mc.percentile_75 / mc.initial_value - 1) * 100:+.1f}%)")
        print(f"  95. percentil:      ${mc.percentile_95:>12,.0f}  "
              f"({(mc.percentile_95 / mc.initial_value - 1) * 100:+.1f}%)")
        print(f"  Best case (99%):    ${mc.best_case:>12,.0f}  "
              f"({(mc.best_case / mc.initial_value - 1) * 100:+.1f}%)")
        print()
        print(f"  VaR (95%):          ${mc.var_95:>12,.0f}")
        print(f"  VaR (99%):          ${mc.var_99:>12,.0f}")
        print(f"  P(tab):             {mc.prob_loss_pct:>11.1f}%")
        print(f"  P(tab > 10%):       {mc.prob_loss_10_pct:>11.1f}%")
        print(f"  P(tab > 20%):       {mc.prob_loss_20_pct:>11.1f}%")
        print(f"  Max DD (gns):       {mc.max_drawdown_mean:>11.1f}%")
        print(f"  Max DD (worst):     {mc.max_drawdown_worst:>11.1f}%")
        print(f"{'═' * 55}\n")

    else:
        # Kør alle scenarier
        print(f"\n⚡ Fuld stress-test: {len(weights)} positioner, ${args.capital:,.0f}")
        print(f"   Portefølje: {', '.join(f'{s} ({w:.0%})' for s, w in weights.items())}")
        print(f"   Kører {len(HISTORICAL_CRISES)} historiske + "
              f"{len(SYNTHETIC_SCENARIOS)} syntetiske scenarier + Monte Carlo...\n")

        report = tester.run_all(
            include_monte_carlo=True,
            monte_carlo_runs=args.runs,
        )
        print(report.summary_table())

        # Skat-impact for worst scenario
        worst = min(report.scenario_results, key=lambda r: r.max_drawdown_pct)
        tax = tester.tax_impact_in_crash(worst.scenario)
        print(f"\n  💰 SKAT VED KRAK ({worst.scenario.name}):")
        print(f"  Tab i USD:           ${tax['loss_usd']:>+12,.0f}")
        print(f"  Tab i DKK:           {tax['loss_dkk']:>+12,.0f} DKK")
        print(f"  Fradrag (27%):       {tax['tax_deduction_27pct']:>12,.0f} DKK")
        print(f"  Fradrag (42%):       {tax['tax_deduction_42pct']:>12,.0f} DKK")
        print(f"  Netto tab (27%):     {tax['net_loss_after_tax_27']:>+12,.0f} DKK")
        print(f"  {tax['advice']}")
        print()


def _start_dashboard() -> None:
    """Start Dash-appen."""
    from src.dashboard.app import app
    app.run(
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        debug=False,
    )


# ══════════════════════════════════════════════════════════════
#  Argument Parser
# ══════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alpha Trading Platform – automatiseret aktiehandel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Eksempler:
  python -m src.main run                          Kør botten
  python -m src.main simulate --symbols AAPL,MSFT Simulér én cyklus
  python -m src.main backtest --start 2024-03-01  Kør backtest
  python -m src.main dashboard                    Start dashboard
  python -m src.main status                       Vis status
  python -m src.main tax --year 2026              Skatteindberetning
  python -m src.main tax-advisor --year 2026      Skatterådgiver
  python -m src.main universe                     Vis aktivunivers
  python -m src.main scan                          Fuld markedsscanning
  python -m src.main scan --sector tech            Scan kun tech-sektor
  python -m src.main scan --asset-class commodities Scan kun råstoffer
  python -m src.main macro                         Makro-dashboard
  python -m src.main stress-test                   Fuld stress-test
  python -m src.main stress-test --scenario covid  Test ét scenarie
  python -m src.main stress-test --monte-carlo --runs 10000  Monte Carlo
        """,
    )

    # Fælles argumenter
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--symbols", type=str, default=None,
                        help="Kommaseparerede aktiesymboler (f.eks. AAPL,MSFT)")
    parent.add_argument("--capital", type=float, default=100_000,
                        help="Startkapital i USD (default: 100000)")

    sub = parser.add_subparsers(dest="command", help="Kommando")

    # run
    p_run = sub.add_parser("run", parents=[parent],
                           help="Kør botten kontinuerligt")
    p_run.add_argument("--interval", type=int, default=60,
                       help="Sekunder mellem cykler (default: 60)")
    p_run.add_argument("--no-dashboard", action="store_true",
                       help="Start uden dashboard")
    p_run.set_defaults(func=cmd_run)

    # simulate
    p_sim = sub.add_parser("simulate", parents=[parent],
                           help="Kør én cyklus og vis resultater")
    p_sim.set_defaults(func=cmd_simulate)

    # backtest
    p_bt = sub.add_parser("backtest", parents=[parent],
                          help="Kør backtests")
    p_bt.add_argument("--start", type=str, default="2024-03-01",
                      help="Startdato (YYYY-MM-DD)")
    p_bt.add_argument("--end", type=str, default="2026-03-01",
                      help="Slutdato (YYYY-MM-DD)")
    p_bt.set_defaults(func=cmd_backtest)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Start web-dashboardet")
    p_dash.set_defaults(func=cmd_dashboard)

    # status
    p_status = sub.add_parser("status", parents=[parent],
                              help="Vis nuværende status")
    p_status.set_defaults(func=cmd_status)

    # tax
    p_tax = sub.add_parser("tax", parents=[parent],
                           help="Generer skatteindberetning")
    p_tax.add_argument("--year", type=int, default=2026,
                       help="Skatteår (default: 2026)")
    p_tax.set_defaults(func=cmd_tax)

    # tax-advisor
    p_advisor = sub.add_parser("tax-advisor", parents=[parent],
                               help="Skatterådgiver med advarsler og optimering")
    p_advisor.add_argument("--year", type=int, default=2026,
                           help="Skatteår (default: 2026)")
    p_advisor.set_defaults(func=cmd_tax_advisor)

    # universe
    p_uni = sub.add_parser("universe", help="Vis aktivunivers")
    p_uni.add_argument("--category", type=str, default=None,
                       help="Vis symboler for én kategori (f.eks. us_stocks)")
    p_uni.add_argument("--scan", action="store_true",
                       help="Vis alle symboler klar til scanning")
    p_uni.set_defaults(func=cmd_universe)

    # scan
    p_scan = sub.add_parser("scan", help="Scan markeder og find muligheder")
    p_scan.add_argument("--sector", type=str, default=None,
                        help="Scan kun én sektor (f.eks. tech, energi, XLK)")
    p_scan.add_argument("--asset-class", type=str, default=None,
                        help="Scan kun én aktivklasse (stocks, etfs, commodities, crypto)")
    p_scan.set_defaults(func=cmd_scan)

    # macro
    p_macro = sub.add_parser("macro", help="Vis makro-dashboard")
    p_macro.set_defaults(func=cmd_macro)

    # stress-test
    p_stress = sub.add_parser("stress-test", parents=[parent],
                               help="Kør stress-test mod porteføljen")
    p_stress.add_argument("--scenario", type=str, default=None,
                          help="Kør ét specifikt scenarie (f.eks. covid, dotcom, "
                               "financial_crisis, flash_crash, market_crash_20)")
    p_stress.add_argument("--monte-carlo", action="store_true",
                          help="Kør kun Monte Carlo simulation")
    p_stress.add_argument("--runs", type=int, default=10_000,
                          help="Antal Monte Carlo simuleringer (default: 10000)")
    p_stress.set_defaults(func=cmd_stress_test)

    return parser


# ══════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logger.info(f"Alpha Trading Platform – kommando: {args.command}")
    args.func(args)


if __name__ == "__main__":
    main()
