"""
PerformanceTracker – daglig performance-rapport og strategy decay detektion.

Features:
  - P&L: dag, uge, måned, YTD
  - Sharpe ratio (rullende 30d og 1år)
  - Win rate, profit factor, bedste/dårligste handler
  - Benchmark-sammenligning (S&P 500)
  - Strategy decay: advar ved dårlig performance over 30d
  - A/B test af strategier

Brug:
    from src.monitoring.performance_tracker import PerformanceTracker
    tracker = PerformanceTracker()
    tracker.record_trade(symbol="AAPL", pnl=150.0, strategy="SMA Crossover")
    report = tracker.daily_report()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from loguru import logger


# ══════════════════════════════════════════════════════════════
#  Dataklasser
# ══════════════════════════════════════════════════════════════


@dataclass
class TradeRecord:
    """Én registreret handel."""
    timestamp: str
    symbol: str
    side: str
    pnl: float
    return_pct: float
    strategy: str
    hold_time_hours: float = 0.0


@dataclass
class PerformanceSnapshot:
    """Øjebliksbillede af performance."""
    timestamp: str
    equity: float
    daily_pnl: float
    daily_return_pct: float


@dataclass
class StrategyPerformance:
    """Performance for én strategi."""
    name: str
    total_trades: int
    winning_trades: int
    total_pnl: float
    win_rate: float
    avg_pnl: float
    profit_factor: float
    sharpe_30d: float
    sharpe_1y: float
    max_drawdown_pct: float
    is_decaying: bool               # Strategy decay detekteret?
    decay_reason: str
    rank: int = 0

    @property
    def status(self) -> str:
        if self.is_decaying:
            return "DECAY"
        if self.sharpe_30d >= 1.0:
            return "STÆRK"
        if self.sharpe_30d >= 0.5:
            return "OK"
        return "SVAG"


@dataclass
class DecayAlert:
    """Advarsel om strategy decay."""
    strategy: str
    reason: str
    severity: str           # "WARNING" eller "CRITICAL"
    recommendation: str
    current_sharpe: float
    days_underperforming: int


@dataclass
class DailyReport:
    """Daglig performance-rapport."""
    timestamp: str
    # P&L
    pnl_today: float
    pnl_week: float
    pnl_month: float
    pnl_ytd: float
    return_today_pct: float
    return_week_pct: float
    return_month_pct: float
    return_ytd_pct: float
    # Metrics
    sharpe_30d: float
    sharpe_1y: float
    win_rate: float
    profit_factor: float
    total_trades: int
    trades_today: int
    # Best/worst
    best_trade: TradeRecord | None
    worst_trade: TradeRecord | None
    # Benchmark
    benchmark_return_pct: float
    alpha_pct: float                # Portefølje minus benchmark
    # Strategier
    strategy_ranking: list[StrategyPerformance]
    decay_alerts: list[DecayAlert]

    def summary_table(self) -> str:
        lines = [
            f"{'═' * 60}",
            f"  DAGLIG RAPPORT – {self.timestamp}",
            f"{'═' * 60}",
            f"  P&L i dag:       ${self.pnl_today:>+10,.2f}  ({self.return_today_pct:+.2f}%)",
            f"  P&L denne uge:   ${self.pnl_week:>+10,.2f}  ({self.return_week_pct:+.2f}%)",
            f"  P&L denne måned: ${self.pnl_month:>+10,.2f}  ({self.return_month_pct:+.2f}%)",
            f"  P&L YTD:         ${self.pnl_ytd:>+10,.2f}  ({self.return_ytd_pct:+.2f}%)",
            f"",
            f"  Sharpe (30d):    {self.sharpe_30d:>10.2f}",
            f"  Sharpe (1år):    {self.sharpe_1y:>10.2f}",
            f"  Win rate:        {self.win_rate:>9.1f}%",
            f"  Profit factor:   {self.profit_factor:>10.2f}",
            f"  Handler i dag:   {self.trades_today:>10}",
            f"  Handler totalt:  {self.total_trades:>10}",
            f"",
            f"  Benchmark:       {self.benchmark_return_pct:>+9.2f}%",
            f"  Alpha:           {self.alpha_pct:>+9.2f}%",
        ]
        if self.best_trade:
            lines.append(
                f"  Bedste handel:   {self.best_trade.symbol} "
                f"${self.best_trade.pnl:+,.2f}"
            )
        if self.worst_trade:
            lines.append(
                f"  Dårligste:       {self.worst_trade.symbol} "
                f"${self.worst_trade.pnl:+,.2f}"
            )

        if self.strategy_ranking:
            lines += [
                f"",
                f"  {'─' * 50}",
                f"  STRATEGI-RANGERING",
                f"  {'Strategi':<22} {'Sharpe':>7} {'Win%':>6} {'P&L':>12} {'Status':>8}",
            ]
            for s in self.strategy_ranking:
                lines.append(
                    f"  {s.name:<22} {s.sharpe_30d:>+6.2f} "
                    f"{s.win_rate:>5.1f}% ${s.total_pnl:>+10,.0f} {s.status:>8}"
                )

        if self.decay_alerts:
            lines += [f"", f"  ⚠️ DECAY ALERTS:"]
            for a in self.decay_alerts:
                lines.append(f"    [{a.severity}] {a.strategy}: {a.reason}")
                lines.append(f"      → {a.recommendation}")

        lines.append(f"{'═' * 60}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  PerformanceTracker
# ══════════════════════════════════════════════════════════════


class PerformanceTracker:
    """
    Spor og analysér trading-performance.

    Args:
        initial_equity: Startkapital.
        benchmark_returns: Daglige benchmark-afkast (S&P 500).
        decay_window_days: Vindue for decay-detektion.
        min_sharpe_threshold: Minimum Sharpe ratio før decay-advarsel.
    """

    def __init__(
        self,
        initial_equity: float = 100_000,
        benchmark_returns: list[float] | None = None,
        decay_window_days: int = 30,
        min_sharpe_threshold: float = 0.5,
    ) -> None:
        self._initial_equity = initial_equity
        self._current_equity = initial_equity
        self._trades: list[TradeRecord] = []
        self._snapshots: list[PerformanceSnapshot] = []
        self._benchmark_returns = benchmark_returns or []
        self._decay_window = decay_window_days
        self._min_sharpe = min_sharpe_threshold

        # Memory limits — keep at most 90 days of trades/snapshots in RAM
        self._max_trades = 50_000
        self._max_snapshots = 10_000

        # Strategy-specifikke spor
        self._strategy_trades: dict[str, list[TradeRecord]] = {}
        self._strategy_daily_returns: dict[str, list[float]] = {}

    @property
    def current_equity(self) -> float:
        return self._current_equity

    @current_equity.setter
    def current_equity(self, value: float) -> None:
        self._current_equity = max(0, value)

    # ── Registrering ────────────────────────────────────────────

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        strategy: str,
        side: str = "long",
        return_pct: float = 0.0,
        hold_time_hours: float = 0.0,
    ) -> None:
        """Registrér en afsluttet handel."""
        record = TradeRecord(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=side,
            pnl=pnl,
            return_pct=return_pct,
            strategy=strategy,
            hold_time_hours=hold_time_hours,
        )
        self._trades.append(record)
        self._current_equity += pnl

        # Prune oldest trades when limit exceeded
        if len(self._trades) > self._max_trades:
            self._trades = self._trades[-self._max_trades:]

        # Per strategi
        if strategy not in self._strategy_trades:
            self._strategy_trades[strategy] = []
        self._strategy_trades[strategy].append(record)
        if len(self._strategy_trades[strategy]) > self._max_trades:
            self._strategy_trades[strategy] = self._strategy_trades[strategy][-self._max_trades:]

    def record_snapshot(self, equity: float | None = None) -> None:
        """Registrér daglig equity-snapshot."""
        eq = equity if equity is not None else self._current_equity
        prev = self._snapshots[-1].equity if self._snapshots else self._initial_equity
        daily_pnl = eq - prev
        daily_ret = daily_pnl / prev if prev > 0 else 0.0

        self._snapshots.append(PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            equity=eq,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_ret * 100,
        ))
        self._current_equity = eq

        # Prune oldest snapshots when limit exceeded
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]

    def add_benchmark_return(self, daily_return: float) -> None:
        """Tilføj dagligt benchmark-afkast."""
        self._benchmark_returns.append(daily_return)
        if len(self._benchmark_returns) > self._max_snapshots:
            self._benchmark_returns = self._benchmark_returns[-self._max_snapshots:]

    # ── Performance-beregning ──────────────────────────────────

    def _pnl_since(self, days: int) -> float:
        """Beregn P&L for seneste N dage."""
        cutoff = datetime.now() - timedelta(days=days)
        return sum(
            t.pnl for t in self._trades
            if t.timestamp >= cutoff.isoformat()
        )

    def _return_since(self, days: int) -> float:
        """Beregn afkast for seneste N dage."""
        pnl = self._pnl_since(days)
        base = self._initial_equity
        if self._snapshots:
            # Find equity N dage siden
            cutoff = datetime.now() - timedelta(days=days)
            older = [s for s in self._snapshots if s.timestamp < cutoff.isoformat()]
            if older:
                base = older[-1].equity
        return (pnl / base * 100) if base > 0 else 0.0

    def _sharpe_ratio(self, returns: list[float]) -> float:
        """Beregn annualiseret Sharpe ratio."""
        if len(returns) < 5:
            return 0.0
        arr = np.array(returns)
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0.0
        return float((mean / std) * np.sqrt(252))

    def _get_daily_returns(self, days: int) -> list[float]:
        """Hent daglige afkast for seneste N dage."""
        if not self._snapshots:
            return []
        n = min(days, len(self._snapshots))
        return [s.daily_return_pct / 100 for s in self._snapshots[-n:]]

    def _win_rate(self, trades: list[TradeRecord]) -> float:
        if not trades:
            return 0.0
        return sum(1 for t in trades if t.pnl > 0) / len(trades) * 100

    def _profit_factor(self, trades: list[TradeRecord]) -> float:
        gross_wins = sum(t.pnl for t in trades if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    # ── Strategy decay ──────────────────────────────────────────

    def detect_decay(self) -> list[DecayAlert]:
        """
        Detektér strategier der performer dårligt.

        Regler:
          1. Sharpe < 0.5 over 30 dage → WARNING
          2. Sharpe < 0.0 over 30 dage → CRITICAL, foreslå deaktivering
          3. Win rate faldet >20% fra gennemsnit → WARNING
        """
        alerts: list[DecayAlert] = []

        for strat_name, trades in self._strategy_trades.items():
            # Seneste 30 dages trades
            cutoff = datetime.now() - timedelta(days=self._decay_window)
            recent = [t for t in trades if t.timestamp >= cutoff.isoformat()]

            if len(recent) < 5:
                continue

            # Sharpe fra daglige P&L'er
            pnls = [t.pnl for t in recent]
            daily_returns = [t.return_pct / 100 for t in recent if t.return_pct != 0]

            if daily_returns:
                sharpe = self._sharpe_ratio(daily_returns)
            else:
                sharpe = self._sharpe_ratio([p / self._initial_equity for p in pnls])

            # Win rate sammenligning
            all_wr = self._win_rate(trades)
            recent_wr = self._win_rate(recent)

            if sharpe < 0.0:
                alerts.append(DecayAlert(
                    strategy=strat_name,
                    reason=f"Sharpe {sharpe:.2f} < 0 over {self._decay_window}d",
                    severity="CRITICAL",
                    recommendation=f"Overvej at deaktivere {strat_name} midlertidigt",
                    current_sharpe=sharpe,
                    days_underperforming=self._decay_window,
                ))
            elif sharpe < self._min_sharpe:
                alerts.append(DecayAlert(
                    strategy=strat_name,
                    reason=f"Sharpe {sharpe:.2f} < {self._min_sharpe} over {self._decay_window}d",
                    severity="WARNING",
                    recommendation=f"Monitorér {strat_name} tæt, overvej re-optimering",
                    current_sharpe=sharpe,
                    days_underperforming=self._decay_window,
                ))

            if all_wr > 0 and (all_wr - recent_wr) > 20:
                alerts.append(DecayAlert(
                    strategy=strat_name,
                    reason=f"Win rate faldet: {all_wr:.0f}% → {recent_wr:.0f}%",
                    severity="WARNING",
                    recommendation="Markedsforhold har ændret sig – tjek regime",
                    current_sharpe=sharpe,
                    days_underperforming=self._decay_window,
                ))

        return alerts

    # ── Strategy performance ────────────────────────────────────

    def strategy_performance(self) -> list[StrategyPerformance]:
        """Beregn performance per strategi, rangeret efter Sharpe."""
        results: list[StrategyPerformance] = []

        for strat_name, trades in self._strategy_trades.items():
            if not trades:
                continue

            wins = [t for t in trades if t.pnl > 0]
            total_pnl = sum(t.pnl for t in trades)
            wr = self._win_rate(trades)
            pf = self._profit_factor(trades)
            avg_pnl = total_pnl / len(trades) if trades else 0

            # Sharpe
            pnls = [t.pnl / self._initial_equity for t in trades]
            sharpe_all = self._sharpe_ratio(pnls)

            cutoff_30 = datetime.now() - timedelta(days=30)
            recent_30 = [t for t in trades if t.timestamp >= cutoff_30.isoformat()]
            pnls_30 = [t.pnl / self._initial_equity for t in recent_30]
            sharpe_30 = self._sharpe_ratio(pnls_30)

            # Drawdown
            cumsum = np.cumsum([t.pnl for t in trades])
            if len(cumsum) > 0:
                peak = np.maximum.accumulate(cumsum)
                dd = (cumsum - peak)
                max_dd = float(dd.min() / self._initial_equity * 100) if self._initial_equity > 0 else 0
            else:
                max_dd = 0.0

            # Decay
            decay_alerts = [a for a in self.detect_decay() if a.strategy == strat_name]
            is_decay = len(decay_alerts) > 0
            decay_reason = decay_alerts[0].reason if decay_alerts else ""

            results.append(StrategyPerformance(
                name=strat_name,
                total_trades=len(trades),
                winning_trades=len(wins),
                total_pnl=total_pnl,
                win_rate=wr,
                avg_pnl=avg_pnl,
                profit_factor=pf,
                sharpe_30d=sharpe_30,
                sharpe_1y=sharpe_all,
                max_drawdown_pct=max_dd,
                is_decaying=is_decay,
                decay_reason=decay_reason,
            ))

        # Rangér efter Sharpe (30d)
        results.sort(key=lambda x: x.sharpe_30d, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    # ── Daglig rapport ──────────────────────────────────────────

    def daily_report(self) -> DailyReport:
        """Generér fuld daglig performance-rapport."""
        now = datetime.now()

        pnl_today = self._pnl_since(1)
        pnl_week = self._pnl_since(7)
        pnl_month = self._pnl_since(30)
        days_ytd = (now - datetime(now.year, 1, 1)).days
        pnl_ytd = self._pnl_since(max(1, days_ytd))

        returns_30d = self._get_daily_returns(30)
        returns_1y = self._get_daily_returns(252)

        # Benchmark
        bench_ret = 0.0
        if self._benchmark_returns:
            bench_ret = sum(self._benchmark_returns[-days_ytd:]) * 100 if days_ytd > 0 else 0

        ytd_ret = self._return_since(max(1, days_ytd))
        alpha = ytd_ret - bench_ret

        # Trades i dag
        today_str = now.date().isoformat()
        trades_today = [t for t in self._trades if t.timestamp[:10] == today_str]

        # Best/worst
        best = max(self._trades, key=lambda t: t.pnl) if self._trades else None
        worst = min(self._trades, key=lambda t: t.pnl) if self._trades else None

        return DailyReport(
            timestamp=now.strftime("%Y-%m-%d %H:%M"),
            pnl_today=pnl_today,
            pnl_week=pnl_week,
            pnl_month=pnl_month,
            pnl_ytd=pnl_ytd,
            return_today_pct=self._return_since(1),
            return_week_pct=self._return_since(7),
            return_month_pct=self._return_since(30),
            return_ytd_pct=ytd_ret,
            sharpe_30d=self._sharpe_ratio(returns_30d),
            sharpe_1y=self._sharpe_ratio(returns_1y),
            win_rate=self._win_rate(self._trades),
            profit_factor=self._profit_factor(self._trades),
            total_trades=len(self._trades),
            trades_today=len(trades_today),
            best_trade=best,
            worst_trade=worst,
            benchmark_return_pct=bench_ret,
            alpha_pct=alpha,
            strategy_ranking=self.strategy_performance(),
            decay_alerts=self.detect_decay(),
        )

    # ── A/B Test ────────────────────────────────────────────────

    def ab_test(
        self,
        strategy_a: str,
        strategy_b: str,
        days: int = 30,
    ) -> dict:
        """
        Sammenlign to strategier over N dage.

        Returns:
            Dict med sammenligningsdata og anbefaling.
        """
        cutoff = datetime.now() - timedelta(days=days)

        trades_a = [
            t for t in self._strategy_trades.get(strategy_a, [])
            if t.timestamp >= cutoff.isoformat()
        ]
        trades_b = [
            t for t in self._strategy_trades.get(strategy_b, [])
            if t.timestamp >= cutoff.isoformat()
        ]

        pnl_a = sum(t.pnl for t in trades_a)
        pnl_b = sum(t.pnl for t in trades_b)
        wr_a = self._win_rate(trades_a)
        wr_b = self._win_rate(trades_b)

        sharpe_a = self._sharpe_ratio(
            [t.pnl / self._initial_equity for t in trades_a]
        )
        sharpe_b = self._sharpe_ratio(
            [t.pnl / self._initial_equity for t in trades_b]
        )

        winner = strategy_a if sharpe_a >= sharpe_b else strategy_b

        return {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "period_days": days,
            "trades_a": len(trades_a),
            "trades_b": len(trades_b),
            "pnl_a": pnl_a,
            "pnl_b": pnl_b,
            "win_rate_a": wr_a,
            "win_rate_b": wr_b,
            "sharpe_a": sharpe_a,
            "sharpe_b": sharpe_b,
            "winner": winner,
            "recommendation": (
                f"{winner} performer bedst (Sharpe: "
                f"{max(sharpe_a, sharpe_b):.2f} vs {min(sharpe_a, sharpe_b):.2f})"
            ),
        }

    # ── Utility ─────────────────────────────────────────────────

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    @property
    def all_trades(self) -> list[TradeRecord]:
        return list(self._trades)

    @property
    def strategy_names(self) -> list[str]:
        return list(self._strategy_trades.keys())
