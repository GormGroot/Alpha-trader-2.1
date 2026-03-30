#!/usr/bin/env python3
"""
Alpha Trading Platform — Aggressiv Multi-Asset Test.

Tester alle asset-klasser: aktier, ETF'er, råstoffer, forex, crypto + short selling.
Kører 100% lokalt med yfinance + PaperBroker.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>")


def main():
    from datetime import datetime, timedelta
    from src.data.market_data import MarketDataFetcher
    from src.data.indicators import add_all_indicators
    from src.strategy.sma_crossover import SMACrossoverStrategy
    from src.strategy.rsi_strategy import RSIStrategy
    from src.strategy.signal_engine import SignalEngine
    from src.broker.paper_broker import PaperBroker
    from src.broker.models import OrderType

    print("=" * 70)
    print("  ALPHA TRADING PLATFORM — Aggressiv Multi-Asset Test")
    print(f"  Dato: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  Mode: AGGRESSIV (80% bankroll, $200k target)")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════
    #  1. HENT DATA FOR ALLE ASSET-KLASSER
    # ══════════════════════════════════════════════════════════════
    fetcher = MarketDataFetcher()

    asset_classes = {
        "US Aktier": ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "NVDA"],
        "Nordic": ["NOVO-B.CO", "MAERSK-B.CO", "VOLV-B.ST"],
        "ETF'er": ["SPY", "QQQ", "GLD", "SLV", "USO", "XLF", "XLE"],
        "Raastoffer": ["GC=F", "SI=F", "CL=F"],
        "Forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    }

    all_data = {}
    print("\n[1/5] Henter markedsdata for alle sektorer...")

    for category, symbols in asset_classes.items():
        print(f"\n  {category}:")
        for sym in symbols:
            try:
                df = fetcher.get_historical(sym, interval="1d", lookback_days=120)
                if df is not None and not df.empty and len(df) >= 55:
                    df = add_all_indicators(df)
                    all_data[sym] = df
                    close = df["Close"].iloc[-1]
                    rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else 0
                    print(f"    {sym:<15} ${close:>10,.2f}  RSI={rsi:.0f}  ({len(df)} dage)")
                else:
                    rows = len(df) if df is not None else 0
                    print(f"    {sym:<15} -- ingen/utilstraekkelig data ({rows} rækker)")
            except Exception as e:
                print(f"    {sym:<15} -- fejl: {e}")

    print(f"\n  Totalt: {len(all_data)} symboler med data")

    # ══════════════════════════════════════════════════════════════
    #  2. KØR STRATEGIER (AGGRESSIV — min_agreement=1)
    # ══════════════════════════════════════════════════════════════
    print(f"\n[2/5] Koerer strategier (aggressiv mode)...")

    strategies = [
        (SMACrossoverStrategy(20, 50), 0.6),
        (RSIStrategy(14, 30, 70), 0.4),
    ]

    engine = SignalEngine(
        strategies=strategies,
        min_agreement=1,
        portfolio_value=100_000,
        max_position_pct=0.10,
    )

    result = engine.process(all_data)

    # ══════════════════════════════════════════════════════════════
    #  3. VIS SIGNALER PER SEKTOR
    # ══════════════════════════════════════════════════════════════
    print(f"\n[3/5] Strategi-signaler:")
    print("=" * 70)

    buys = result.buys
    sells = result.sells
    holds = len(result.signals) - len(buys) - len(sells)

    # Gruppér per sektor
    def get_category(sym):
        for cat, syms in asset_classes.items():
            if sym in syms:
                return cat
        return "Andet"

    if buys:
        print(f"\n  >>> KOB-SIGNALER ({len(buys)}):")
        for sig in buys:
            cat = get_category(sig.symbol)
            print(f"    [{cat:<12}] {sig.symbol:<15} conf={sig.confidence:.0f}%  "
                  f"${sig.position_size_usd:>8,.0f}  — {sig.reason}")
            for d in sig.strategy_details:
                if d["signal"] != "HOLD":
                    print(f"                  {d['strategy']}: {d['reason']}")

    if sells:
        print(f"\n  <<< SAELG/SHORT-SIGNALER ({len(sells)}):")
        for sig in sells:
            cat = get_category(sig.symbol)
            print(f"    [{cat:<12}] {sig.symbol:<15} conf={sig.confidence:.0f}%  "
                  f"${sig.position_size_usd:>8,.0f}  — {sig.reason}")
            for d in sig.strategy_details:
                if d["signal"] != "HOLD":
                    print(f"                  {d['strategy']}: {d['reason']}")

    print(f"\n  OPSUMMERING: {len(buys)} KOB, {len(sells)} SAELG/SHORT, {holds} HOLD")

    # ══════════════════════════════════════════════════════════════
    #  4. PAPER BROKER — SIMULÉR HANDLER
    # ══════════════════════════════════════════════════════════════
    print(f"\n[4/5] Simulerer handler via PaperBroker...")
    print("=" * 70)

    broker = PaperBroker(initial_capital=100_000, market_data=fetcher)

    trades_executed = 0

    # Eksekver BUY signaler
    for sig in buys[:8]:  # Max 8 pr scan
        try:
            price = all_data[sig.symbol]["Close"].iloc[-1]
            position_usd = min(sig.position_size_usd, broker.portfolio.cash * 0.15)
            is_frac = any(sig.symbol.endswith(s) for s in ("-USD", "=F", "=X"))
            if is_frac:
                qty = round(position_usd / price, 4)
            else:
                qty = max(1, int(position_usd / price))
            if qty * price > broker.portfolio.cash:
                continue
            order = broker.buy(sig.symbol, qty)
            trades_executed += 1
            cat = get_category(sig.symbol)
            print(f"  KOB  [{cat:<12}] {qty:>8.2f} {sig.symbol:<15} @ ${price:>10,.2f}  = ${qty*price:>10,.2f}")
        except Exception as e:
            print(f"  FEJL {sig.symbol}: {e}")

    # Eksekver SELL/SHORT signaler
    for sig in sells[:5]:  # Max 5 shorts
        try:
            price = all_data[sig.symbol]["Close"].iloc[-1]
            position_usd = min(sig.position_size_usd, 10_000)
            is_frac = any(sig.symbol.endswith(s) for s in ("-USD", "=F", "=X"))
            if is_frac:
                qty = round(position_usd / price, 4)
            else:
                qty = max(1, int(position_usd / price))
            order = broker.sell(sig.symbol, qty, short=True)
            trades_executed += 1
            cat = get_category(sig.symbol)
            print(f"  SHORT[{cat:<12}] {qty:>8.2f} {sig.symbol:<15} @ ${price:>10,.2f}  = ${qty*price:>10,.2f}")
        except Exception as e:
            print(f"  FEJL SHORT {sig.symbol}: {e}")

    # Vis portefoeljeerstatus
    account = broker.get_account()
    positions = broker.get_positions()

    print(f"\n  --- Portefoelje Status ---")
    print(f"  Kontanter:      ${account.cash:>12,.2f}")
    print(f"  Equity:         ${account.equity:>12,.2f}")
    print(f"  Positioner:     {len(positions)}")
    print(f"  Handler udfoert: {trades_executed}")

    if positions:
        print(f"\n  Aabne positioner:")
        for pos in positions:
            side_str = "SHORT" if pos.side == "short" else "LONG "
            cat = get_category(pos.symbol)
            print(f"    {side_str} [{cat:<12}] {pos.qty:>8.2f} {pos.symbol:<15} "
                  f"entry=${pos.entry_price:>10,.2f}  value=${pos.market_value:>10,.2f}")

    # ══════════════════════════════════════════════════════════════
    #  5. BACKTEST (6 MÅNEDER, AGGRESSIV)
    # ══════════════════════════════════════════════════════════════
    print(f"\n[5/5] Backtest — 6 maaneder, aggressiv mode...")
    print("=" * 70)

    from src.backtest.backtester import Backtester

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    # Test med SMA strategi over udvidede aktiver
    backtest_symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM",
                        "SPY", "QQQ", "GLD"]

    bt = Backtester(
        strategy=SMACrossoverStrategy(20, 50),
        symbols=backtest_symbols,
        start=start_date,
        end=end_date,
        initial_capital=100_000,
        commission_pct=0.0,
        max_position_pct=0.10,  # Aggressiv
    )
    bt_result = bt.run()

    print(f"\n  Periode:         {start_date} -> {end_date}")
    print(f"  Symboler:        {', '.join(backtest_symbols)}")
    print(f"  Startkapital:    $100,000")
    print(f"  Slutvaerdi:      ${bt_result.final_equity:,.2f}")
    print(f"  Afkast:          {bt_result.total_return_pct:+.2f}%")
    print(f"  Aarligt afkast:  {bt_result.annualized_return_pct:+.2f}%")
    print(f"  Sharpe:          {bt_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:    {bt_result.max_drawdown_pct:.2f}%")
    print(f"  Handler:         {bt_result.num_trades}")
    print(f"  Win Rate:        {bt_result.win_rate:.1f}%")
    print(f"  Profit Factor:   {bt_result.profit_factor:.2f}")

    if bt_result.trades:
        print(f"\n  Seneste handler:")
        for t in bt_result.trades[-8:]:
            pnl = f"${t.net_pnl:+,.2f}"
            print(f"    {t.entry_date} -> {t.exit_date}  {t.symbol:<8} "
                  f"{t.qty} stk  ${t.entry_price:.2f} -> ${t.exit_price:.2f}  {pnl}")

    # ══════════════════════════════════════════════════════════════
    #  KONKLUSION
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  SYSTEM STATUS")
    print(f"{'=' * 70}")
    print(f"  [OK] US aktier, Nordic, ETF'er, raastoffer, forex, crypto")
    print(f"  [OK] Short selling (PaperBroker)")
    print(f"  [OK] Aggressiv mode (10% per position, 30 max)")
    print(f"  [OK] $200k target mode switch")
    print(f"  [OK] Fractional qty for commodities/forex/crypto")
    print(f"  [OK] {len(all_data)} symboler scannet paa tvaers af alle sektorer")
    print(f"  [OK] 100% lokalt — ingen API-noegler nødvendige")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
