"""
Strategi-sammenligning – sammenlign performance af flere strategier.

Kører backtests for flere strategier, sammenligner med benchmark
(f.eks. S&P 500) og genererer en sammenligningsrapport.

Brug:
    from src.backtest.comparison import StrategyComparison
    comp = StrategyComparison(strategies, symbols, start, end)
    report = comp.run()
    print(report.comparison_table())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from src.backtest.backtester import Backtester, BacktestResult
from src.data.market_data import MarketDataFetcher
from src.strategy.base_strategy import BaseStrategy


# ── Dataklasser ───────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Buy-and-hold benchmark resultat."""

    name: str
    initial_capital: float
    final_equity: float
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return ((self.final_equity - self.initial_capital) / self.initial_capital) * 100

    @property
    def annualized_return_pct(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        days = len(self.equity_curve)
        total = self.final_equity / self.initial_capital
        if total <= 0:
            return -100.0
        years = days / 252
        if years == 0:
            return 0.0
        return (total ** (1 / years) - 1) * 100

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.expanding().max()
        dd = (self.equity_curve - peak) / peak
        return abs(dd.min()) * 100

    @property
    def sharpe_ratio(self) -> float:
        if self.daily_returns.empty or len(self.daily_returns) < 2:
            return 0.0
        mean = self.daily_returns.mean()
        std = self.daily_returns.std()
        if std == 0:
            return 0.0
        return (mean / std) * (252 ** 0.5)


@dataclass
class ComparisonReport:
    """Resultat af strategisammenligning."""

    results: list[BacktestResult]
    benchmark: BenchmarkResult | None = None

    def comparison_table(self) -> str:
        """Generér en pæn sammenligningstabel."""
        rows = []

        # Header
        headers = [
            "Strategi", "Afkast", "Årligt", "Sharpe",
            "Max DD", "Handler", "Win%", "Gns P/L",
        ]
        rows.append(headers)

        # Benchmark først
        if self.benchmark:
            b = self.benchmark
            rows.append([
                f"📊 {b.name}",
                f"{b.total_return_pct:+.1f}%",
                f"{b.annualized_return_pct:+.1f}%",
                f"{b.sharpe_ratio:.2f}",
                f"{b.max_drawdown_pct:.1f}%",
                "–",
                "–",
                "–",
            ])

        # Strategier
        for r in self.results:
            rows.append([
                f"🤖 {r.strategy_name}",
                f"{r.total_return_pct:+.1f}%",
                f"{r.annualized_return_pct:+.1f}%",
                f"{r.sharpe_ratio:.2f}",
                f"{r.max_drawdown_pct:.1f}%",
                str(r.num_trades),
                f"{r.win_rate:.0f}%",
                f"${r.avg_profit_per_trade:,.0f}",
            ])

        # Formatér som tabel
        col_widths = [
            max(len(str(row[i])) for row in rows) + 2
            for i in range(len(headers))
        ]

        lines = [""]
        sep = "─" * sum(col_widths)
        lines.append(f"╔{'═' * sum(col_widths)}╗")
        lines.append(f"║{'STRATEGISAMMENLIGNING':^{sum(col_widths)}}║")
        lines.append(f"╠{'═' * sum(col_widths)}╣")

        # Header
        header_line = "║"
        for i, h in enumerate(headers):
            header_line += f"{h:^{col_widths[i]}}"
        header_line += "║"
        lines.append(header_line)
        lines.append(f"╟{sep}╢")

        # Data
        for row in rows[1:]:
            line = "║"
            for i, cell in enumerate(row):
                if i == 0:
                    line += f"{cell:<{col_widths[i]}}"
                else:
                    line += f"{cell:>{col_widths[i]}}"
            line += "║"
            lines.append(line)

        lines.append(f"╚{'═' * sum(col_widths)}╝")
        lines.append("")

        return "\n".join(lines)

    def ranking(self) -> list[tuple[str, float]]:
        """Rangér strategier efter Sharpe ratio."""
        ranked = sorted(self.results, key=lambda r: r.sharpe_ratio, reverse=True)
        return [(r.strategy_name, r.sharpe_ratio) for r in ranked]

    def best_strategy(self) -> BacktestResult | None:
        """Returnér strategien med højeste Sharpe ratio."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.sharpe_ratio)

    def detailed_report(self) -> str:
        """Generér en detaljeret rapport med alle metrics."""
        lines = [self.comparison_table()]

        # Ranking
        lines.append("  RANGERING (efter Sharpe ratio):")
        for i, (name, sharpe) in enumerate(self.ranking(), 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
            lines.append(f"    {medal} {name} (Sharpe: {sharpe:.2f})")

        # Benchmark sammenligning
        if self.benchmark:
            lines.append("")
            lines.append("  VS. BENCHMARK:")
            for r in self.results:
                alpha = r.annualized_return_pct - self.benchmark.annualized_return_pct
                lines.append(
                    f"    {r.strategy_name}: alpha = {alpha:+.2f}% p.a."
                )

        # Bedste strategi detaljer
        best = self.best_strategy()
        if best:
            lines.append("")
            lines.append(f"  BEDSTE STRATEGI: {best.strategy_name}")
            lines.append(best.summary_table())

        return "\n".join(lines)


# ── StrategyComparison ────────────────────────────────────────


class StrategyComparison:
    """
    Sammenlign flere strategier mod historisk data.

    Args:
        strategies: Liste af strategier der testes.
        symbols: Aktie-symboler.
        start: Startdato.
        end: Slutdato.
        initial_capital: Startkapital.
        benchmark_symbol: Benchmark (default: SPY for S&P 500).
        commission_pct: Kommission per handel.
        spread_pct: Simuleret spread.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        initial_capital: float = 100_000,
        benchmark_symbol: str = "SPY",
        commission_pct: float = 0.001,
        spread_pct: float = 0.0005,
        market_data: MarketDataFetcher | None = None,
    ) -> None:
        self.strategies = strategies
        self.symbols = symbols
        self.start = start
        self.end = end
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol
        self.commission_pct = commission_pct
        self.spread_pct = spread_pct
        self._market_data = market_data or MarketDataFetcher()

    def run(self) -> ComparisonReport:
        """Kør backtests for alle strategier + benchmark."""
        results: list[BacktestResult] = []

        for strat in self.strategies:
            logger.info(f"Kører backtest: {strat.name}")
            bt = Backtester(
                strategy=strat,
                symbols=self.symbols,
                start=self.start,
                end=self.end,
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct,
                spread_pct=self.spread_pct,
                market_data=self._market_data,
            )
            result = bt.run()
            results.append(result)

        # Benchmark
        benchmark = self._run_benchmark()

        report = ComparisonReport(results=results, benchmark=benchmark)

        # Log ranking
        for i, (name, sharpe) in enumerate(report.ranking(), 1):
            logger.info(f"#{i} {name}: Sharpe={sharpe:.2f}")

        return report

    def _run_benchmark(self) -> BenchmarkResult | None:
        """Kør buy-and-hold benchmark."""
        try:
            df = self._market_data.get_historical(
                symbol=self.benchmark_symbol,
                interval="1d",
                start=self.start,
                end=self.end,
            )
            if df.empty:
                logger.warning(f"Ingen benchmark-data for {self.benchmark_symbol}")
                return None

            # Simulér buy-and-hold
            initial_price = df.iloc[0]["Close"]
            shares = self.initial_capital / initial_price
            equity_values = df["Close"] * shares

            eq_series = pd.Series(equity_values.values, dtype=float)
            daily_ret = eq_series.pct_change().dropna()

            return BenchmarkResult(
                name=f"Buy & Hold {self.benchmark_symbol}",
                initial_capital=self.initial_capital,
                final_equity=float(eq_series.iloc[-1]),
                equity_curve=eq_series,
                daily_returns=daily_ret,
            )
        except Exception as exc:
            logger.error(f"Benchmark-fejl: {exc}")
            return None
