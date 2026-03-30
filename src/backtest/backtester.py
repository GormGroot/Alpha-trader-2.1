"""
Backtesting-motor – kør strategier mod historisk data.

Simulerer handler bar-for-bar med realistiske omkostninger
(spread + kommission) og beregner performance metrics.

Brug:
    from src.backtest.backtester import Backtester
    bt = Backtester(strategy, symbols, start="2024-01-01", end="2025-12-31")
    result = bt.run()
    print(result.summary_table())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

import gc

from config.settings import settings
from src.data.indicators import add_all_indicators
from src.data.market_data import MarketDataFetcher
from src.fees.fee_calculator import FeeCalculator
from src.strategy.base_strategy import BaseStrategy, Signal


# ── Dataklasser ───────────────────────────────────────────────


@dataclass
class Trade:
    """En gennemført handel (åbnet + lukket)."""

    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    entry_reason: str
    exit_reason: str
    commission_cost: float = 0.0

    @property
    def gross_pnl(self) -> float:
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.qty
        return (self.entry_price - self.exit_price) * self.qty

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission_cost

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.qty
        if cost == 0:
            return 0.0
        return (self.net_pnl / cost) * 100


@dataclass
class BacktestResult:
    """Samlet resultat af en backtest."""

    strategy_name: str
    symbols: list[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # ── Beregnede metrics ─────────────────────────────────────

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return ((self.final_equity - self.initial_capital) / self.initial_capital) * 100

    @property
    def total_pnl(self) -> float:
        return self.final_equity - self.initial_capital

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.net_pnl > 0]

    @property
    def losing_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.net_pnl <= 0]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.winning_trades) / len(self.trades) * 100

    @property
    def avg_profit_per_trade(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.net_pnl for t in self.trades) / len(self.trades)

    @property
    def avg_win(self) -> float:
        wins = self.winning_trades
        if not wins:
            return 0.0
        return sum(t.net_pnl for t in wins) / len(wins)

    @property
    def avg_loss(self) -> float:
        losses = self.losing_trades
        if not losses:
            return 0.0
        return sum(t.net_pnl for t in losses) / len(losses)

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(t.net_pnl for t in self.winning_trades)
        gross_losses = abs(sum(t.net_pnl for t in self.losing_trades))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        return abs(drawdown.min()) * 100

    @property
    def annualized_return_pct(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        trading_days = len(self.equity_curve)
        total_return = self.final_equity / self.initial_capital
        if total_return <= 0:
            return -100.0
        years = trading_days / 252
        if years == 0:
            return 0.0
        return (total_return ** (1 / years) - 1) * 100

    @property
    def sharpe_ratio(self) -> float:
        if self.daily_returns.empty or len(self.daily_returns) < 2:
            return 0.0
        mean_ret = self.daily_returns.mean()
        std_ret = self.daily_returns.std()
        if std_ret == 0:
            return 0.0
        # Annualiseret: √252
        return (mean_ret / std_ret) * np.sqrt(252)

    @property
    def sortino_ratio(self) -> float:
        if self.daily_returns.empty or len(self.daily_returns) < 2:
            return 0.0
        mean_ret = self.daily_returns.mean()
        downside = self.daily_returns[self.daily_returns < 0]
        if downside.empty or downside.std() == 0:
            return 0.0
        return (mean_ret / downside.std()) * np.sqrt(252)

    @property
    def total_commission(self) -> float:
        return sum(t.commission_cost for t in self.trades)

    @property
    def calmar_ratio(self) -> float:
        if self.max_drawdown_pct == 0:
            return 0.0
        return self.annualized_return_pct / self.max_drawdown_pct

    # ── Rapportering ──────────────────────────────────────────

    def summary(self) -> dict:
        """Returnér alle metrics som dict."""
        return {
            "Strategi": self.strategy_name,
            "Symboler": ", ".join(self.symbols),
            "Periode": f"{self.start_date} → {self.end_date}",
            "Startkapital": f"${self.initial_capital:,.0f}",
            "Slutværdi": f"${self.final_equity:,.2f}",
            "Totalt afkast": f"{self.total_return_pct:+.2f}%",
            "Årligt afkast": f"{self.annualized_return_pct:+.2f}%",
            "Sharpe ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino ratio": f"{self.sortino_ratio:.2f}",
            "Max drawdown": f"{self.max_drawdown_pct:.2f}%",
            "Calmar ratio": f"{self.calmar_ratio:.2f}",
            "Antal handler": self.num_trades,
            "Win rate": f"{self.win_rate:.1f}%",
            "Gns. profit/handel": f"${self.avg_profit_per_trade:,.2f}",
            "Gns. gevinst": f"${self.avg_win:,.2f}",
            "Gns. tab": f"${self.avg_loss:,.2f}",
            "Profit factor": f"{self.profit_factor:.2f}",
            "Kommission totalt": f"${self.total_commission:,.2f}",
        }

    def summary_table(self) -> str:
        """Formatér metrics som en pæn tabel."""
        s = self.summary()
        lines = [
            f"{'═' * 55}",
            f"  BACKTEST: {s['Strategi']}",
            f"{'═' * 55}",
        ]
        for key, val in s.items():
            if key == "Strategi":
                continue
            lines.append(f"  {key:<25} {val:>25}")
        lines.append(f"{'═' * 55}")
        return "\n".join(lines)


# ── Backtester ────────────────────────────────────────────────


class Backtester:
    """
    Backtesting-motor der kører en strategi mod historisk data.

    Args:
        strategy: Strategien der testes.
        symbols: Liste af aktie-symboler.
        start: Startdato (YYYY-MM-DD).
        end: Slutdato (YYYY-MM-DD).
        initial_capital: Startkapital i USD.
        commission_pct: Kommission i procent per handel (0.001 = 0.1%).
        spread_pct: Simuleret spread i procent (0.0005 = 0.05%).
        max_position_pct: Max andel af porteføljen per position.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        initial_capital: float | None = None,
        commission_pct: float | None = None,
        spread_pct: float | None = None,
        max_position_pct: float | None = None,
        market_data: MarketDataFetcher | None = None,
        broker: str = "paper",
    ) -> None:
        self.strategy = strategy
        self.symbols = [s.upper() for s in (symbols or settings.trading.symbols)]
        self.start = start or settings.backtest.start_date
        self.end = end or settings.backtest.end_date
        self.initial_capital = initial_capital or settings.backtest.initial_capital
        self.max_position_pct = max_position_pct or settings.risk.max_position_pct
        self._market_data = market_data or MarketDataFetcher()

        # Fee calculator — uses realistic per-symbol fees when no override given
        self._fee_calc = FeeCalculator(broker=broker)
        self._override_commission = commission_pct
        self._override_spread = spread_pct

        # Legacy fallback: if explicit pct given, use flat rates (old behaviour)
        self.commission_pct = (
            commission_pct
            if commission_pct is not None
            else settings.backtest.commission_pct
        )
        self.spread_pct = spread_pct if spread_pct is not None else 0.0005
        self._use_realistic_fees = (commission_pct is None and spread_pct is None)

    def run(self) -> BacktestResult:
        """Kør backtesten og returnér resultater."""
        logger.info(
            f"Starter backtest: {self.strategy.name} | "
            f"{self.symbols} | {self.start} → {self.end}"
        )

        # 1. Hent historisk data
        data = self._fetch_data()
        if not data:
            logger.error("Ingen data hentet – afbryder backtest")
            return BacktestResult(
                strategy_name=self.strategy.name,
                symbols=self.symbols,
                start_date=self.start,
                end_date=self.end,
                initial_capital=self.initial_capital,
                final_equity=self.initial_capital,
            )

        # 2. Kør simulering
        trades, equity_curve, all_dates = self._simulate(data)

        # 3. Beregn daglige afkast
        eq_series = pd.Series(equity_curve, index=all_dates[:len(equity_curve)], dtype=float)
        daily_returns = eq_series.pct_change().dropna()

        result = BacktestResult(
            strategy_name=self.strategy.name,
            symbols=self.symbols,
            start_date=self.start,
            end_date=self.end,
            initial_capital=self.initial_capital,
            final_equity=eq_series.iloc[-1] if not eq_series.empty else self.initial_capital,
            trades=trades,
            equity_curve=eq_series,
            daily_returns=daily_returns,
        )

        logger.info(
            f"Backtest færdig: {result.num_trades} handler, "
            f"afkast={result.total_return_pct:+.2f}%, "
            f"sharpe={result.sharpe_ratio:.2f}"
        )
        return result

    # ── Data-hentning ─────────────────────────────────────────

    def _fetch_data(self) -> dict[str, pd.DataFrame]:
        """Hent og berig historisk data for alle symboler."""
        data: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            try:
                df = self._market_data.get_historical(
                    symbol=symbol,
                    interval="1d",
                    start=self.start,
                    end=self.end,
                )
                if df.empty:
                    logger.warning(f"Ingen data for {symbol}")
                    continue

                df = add_all_indicators(df)
                data[symbol] = df
                logger.debug(f"{symbol}: {len(df)} bars hentet")
            except Exception as exc:
                logger.error(f"Fejl ved hentning af {symbol}: {exc}")
        gc.collect()
        return data

    # ── Simulering ────────────────────────────────────────────

    def _simulate(
        self, data: dict[str, pd.DataFrame]
    ) -> tuple[list[Trade], list[float], list]:
        """
        Bar-for-bar simulering.

        For hver handelsdag:
          1. Opdatér åbne positioner med dagens lukkekurs
          2. Tjek exit-signaler (strategi siger SELL)
          3. Tjek entry-signaler (strategi siger BUY)
          4. Registrér equity
        """
        # Fælles datoer
        all_dates = sorted(
            set().union(*(df.index.tolist() for df in data.values()))
        )

        cash = self.initial_capital
        positions: dict[str, dict] = {}  # symbol → {qty, entry_price, entry_date, reason}
        trades: list[Trade] = []
        equity_curve: list[float] = []

        # Mindste lookback for strategi
        min_bars = 60

        for date in all_dates:
            # Beregn equity
            equity = cash
            for sym, pos in positions.items():
                if sym in data and date in data[sym].index:
                    price = data[sym].loc[date, "Close"]
                    equity += pos["qty"] * price
                else:
                    equity += pos["qty"] * pos["entry_price"]

            equity_curve.append(equity)

            # Analysér hvert symbol
            for symbol in self.symbols:
                if symbol not in data:
                    continue
                df = data[symbol]
                if date not in df.index:
                    continue

                # Brug data op til og med denne dag
                idx = df.index.get_loc(date)
                if idx < min_bars:
                    continue

                # Use a view (not copy) to avoid 41K+ DataFrame allocations
                df_slice = df.iloc[: idx + 1]
                close_price = df.iat[idx, df.columns.get_loc("Close")]

                # Kør strategi
                try:
                    result = self.strategy.analyze(df_slice)
                except Exception as exc:
                    logger.debug(f"Strategi-fejl {symbol} {date}: {exc}")
                    continue

                # ── EXIT: Sælg hvis vi har position og signal=SELL ──
                if symbol in positions and result.signal == Signal.SELL:
                    pos = positions.pop(symbol)
                    if self._use_realistic_fees:
                        spread = self._fee_calc.get_spread_pct(symbol)
                    else:
                        spread = self.spread_pct
                    sell_price = close_price * (1 - spread)
                    proceeds = pos["qty"] * sell_price

                    if self._use_realistic_fees:
                        fee = self._fee_calc.calculate(symbol, "sell", pos["qty"], sell_price)
                        commission = fee.total
                    else:
                        commission = proceeds * self.commission_pct
                    cash += proceeds - commission

                    trades.append(
                        Trade(
                            symbol=symbol,
                            side="long",
                            qty=pos["qty"],
                            entry_price=pos["entry_price"],
                            exit_price=sell_price,
                            entry_date=pos["entry_date"],
                            exit_date=str(date)[:10],
                            entry_reason=pos["reason"],
                            exit_reason=result.reason,
                            commission_cost=commission + pos.get("entry_commission", 0),
                        )
                    )

                # ── ENTRY: Køb hvis intet position og signal=BUY ──
                elif symbol not in positions and result.signal == Signal.BUY:
                    # Position sizing
                    budget = equity * self.max_position_pct
                    if self._use_realistic_fees:
                        spread = self._fee_calc.get_spread_pct(symbol)
                    else:
                        spread = self.spread_pct
                    buy_price = close_price * (1 + spread)

                    if buy_price <= 0:
                        continue

                    max_qty = int(budget / buy_price)
                    if max_qty <= 0 or max_qty * buy_price > cash:
                        max_qty = int(cash * 0.95 / buy_price)
                    if max_qty <= 0:
                        continue

                    cost = max_qty * buy_price
                    if self._use_realistic_fees:
                        fee = self._fee_calc.calculate(symbol, "buy", max_qty, buy_price)
                        commission = fee.total
                    else:
                        commission = cost * self.commission_pct
                    if cost + commission > cash:
                        continue

                    cash -= cost + commission
                    positions[symbol] = {
                        "qty": max_qty,
                        "entry_price": buy_price,
                        "entry_date": str(date)[:10],
                        "reason": result.reason,
                        "entry_commission": commission,
                    }

        # Luk alle åbne positioner ved slutning
        last_date = str(all_dates[-1])[:10] if all_dates else self.end
        for symbol, pos in list(positions.items()):
            if symbol in data and not data[symbol].empty:
                close_price = data[symbol].iloc[-1]["Close"]
            else:
                close_price = pos["entry_price"]

            if self._use_realistic_fees:
                spread = self._fee_calc.get_spread_pct(symbol)
            else:
                spread = self.spread_pct
            sell_price = close_price * (1 - spread)
            proceeds = pos["qty"] * sell_price

            if self._use_realistic_fees:
                fee = self._fee_calc.calculate(symbol, "sell", pos["qty"], sell_price)
                commission = fee.total
            else:
                commission = proceeds * self.commission_pct
            cash += proceeds - commission

            trades.append(
                Trade(
                    symbol=symbol,
                    side="long",
                    qty=pos["qty"],
                    entry_price=pos["entry_price"],
                    exit_price=sell_price,
                    entry_date=pos["entry_date"],
                    exit_date=last_date,
                    entry_reason=pos["reason"],
                    exit_reason="backtest_slut",
                    commission_cost=commission + pos.get("entry_commission", 0),
                )
            )

        # Opdatér sidste equity
        if equity_curve:
            equity_curve[-1] = cash

        return trades, equity_curve, all_dates
