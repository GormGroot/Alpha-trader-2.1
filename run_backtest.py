"""
Kør backtest af alle strategier mod 2 års historisk data.

Symboler: AAPL, MSFT, GOOGL, AMZN, TSLA
Periode:  2024-03-01 → 2026-03-01
Benchmark: SPY (S&P 500)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.strategy.sma_crossover import SMACrossoverStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.combined_strategy import CombinedStrategy
from src.backtest.backtester import Backtester
from src.backtest.comparison import StrategyComparison

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
START = "2024-03-01"
END = "2026-03-01"
INITIAL_CAPITAL = 100_000
COMMISSION = 0.001    # 0.1% per handel
SPREAD = 0.0005       # 0.05% spread

# Opret strategier
sma = SMACrossoverStrategy(short_window=20, long_window=50)
rsi = RSIStrategy(oversold=30, overbought=70)
combined = CombinedStrategy(
    strategies=[
        (SMACrossoverStrategy(short_window=20, long_window=50), 0.6),
        (RSIStrategy(oversold=30, overbought=70), 0.4),
    ],
    min_agreement=1,
)

strategies = [sma, rsi, combined]

print("\n" + "=" * 65)
print("  BACKTEST – Alle strategier mod 2 års historisk data")
print(f"  Symboler: {', '.join(SYMBOLS)}")
print(f"  Periode:  {START} → {END}")
print(f"  Kapital:  ${INITIAL_CAPITAL:,}")
print(f"  Kommission: {COMMISSION*100:.1f}%  |  Spread: {SPREAD*100:.3f}%")
print("=" * 65)

# Kør sammenligning
comp = StrategyComparison(
    strategies=strategies,
    symbols=SYMBOLS,
    start=START,
    end=END,
    initial_capital=INITIAL_CAPITAL,
    benchmark_symbol="SPY",
    commission_pct=COMMISSION,
    spread_pct=SPREAD,
)

report = comp.run()

# Vis resultater
print(report.detailed_report())

# Individuelle rapporter
print("\n" + "=" * 65)
print("  INDIVIDUELLE STRATEGI-RAPPORTER")
print("=" * 65)

for result in report.results:
    print(result.summary_table())
    if result.trades:
        print(f"\n  Top-5 bedste handler ({result.strategy_name}):")
        best = sorted(result.trades, key=lambda t: t.net_pnl, reverse=True)[:5]
        for t in best:
            print(f"    {t.symbol}: {t.entry_date} → {t.exit_date} | "
                  f"${t.net_pnl:+,.2f} ({t.return_pct:+.1f}%)")
        print(f"\n  Top-5 værste handler ({result.strategy_name}):")
        worst = sorted(result.trades, key=lambda t: t.net_pnl)[:5]
        for t in worst:
            print(f"    {t.symbol}: {t.entry_date} → {t.exit_date} | "
                  f"${t.net_pnl:+,.2f} ({t.return_pct:+.1f}%)")
    print()
