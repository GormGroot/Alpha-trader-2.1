"""
CorrelationMonitor – overvaag korrelationer og koncentration i portefoeljen.

Features:
  - Korrelationsmatrix mellem positioner
  - Koncentrationsadvarsel (for mange korrelerede aktier)
  - Portfolio beta (foelsomhed overfor markedet)
  - Diversificeringsforslag naar korrelation er for hoej
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CorrelationWarning:
    """Advarsel om hoej korrelation."""
    symbol_a: str
    symbol_b: str
    correlation: float
    message: str


@dataclass
class ConcentrationWarning:
    """Advarsel om for hoej sektorkoncentration."""
    sector: str
    symbols: list[str]
    total_weight_pct: float
    message: str


@dataclass
class DiversificationSuggestion:
    """Forslag til diversificering."""
    reason: str
    suggestion: str
    current_correlation: float


@dataclass
class CorrelationReport:
    """Samlet korrelationsrapport."""
    portfolio_beta: float
    avg_correlation: float
    max_correlation: float
    highly_correlated_pairs: list[CorrelationWarning] = field(default_factory=list)
    concentration_warnings: list[ConcentrationWarning] = field(default_factory=list)
    diversification_suggestions: list[DiversificationSuggestion] = field(default_factory=list)
    correlation_matrix: pd.DataFrame | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def is_healthy(self) -> bool:
        return (
            len(self.highly_correlated_pairs) == 0
            and len(self.concentration_warnings) == 0
        )

    @property
    def risk_level(self) -> str:
        n_warnings = len(self.highly_correlated_pairs) + len(self.concentration_warnings)
        if n_warnings == 0:
            return "low"
        elif n_warnings <= 2:
            return "medium"
        return "high"


# Sektor-mapping for populaere aktier
_SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "META": "Technology", "NVDA": "Technology",
    "TSLA": "Consumer Discretionary", "NFLX": "Communication Services",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "WFC": "Financials", "MS": "Financials", "C": "Financials",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "CAT": "Industrials", "UPS": "Industrials", "BA": "Industrials",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "AMT": "Real Estate", "PLD": "Real Estate",
    "GLD": "Safe Haven", "TLT": "Safe Haven", "SHY": "Safe Haven",
    "SPY": "Index", "QQQ": "Index", "IWM": "Index",
}

# Sektor-diversificeringsforslag
_DIVERSIFICATION_MAP: dict[str, list[str]] = {
    "Technology": ["Healthcare", "Utilities", "Consumer Staples"],
    "Financials": ["Healthcare", "Technology", "Utilities"],
    "Healthcare": ["Technology", "Energy", "Industrials"],
    "Energy": ["Technology", "Healthcare", "Utilities"],
    "Consumer Discretionary": ["Utilities", "Healthcare", "Consumer Staples"],
    "Consumer Staples": ["Technology", "Financials", "Energy"],
}


class CorrelationMonitor:
    """
    Overvaag korrelationer, beta og koncentration i portefoeljen.

    Brug:
        monitor = CorrelationMonitor()
        report = monitor.analyze(price_data, positions, market_data)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.75,
        concentration_threshold: float = 0.40,
        min_data_points: int = 30,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            correlation_threshold: Advar hvis korrelation > dette (0-1).
            concentration_threshold: Advar hvis sektor-vaegt > dette (0-1).
            min_data_points: Minimum antal datapunkter for korrelation.
            sector_map: Custom symbol -> sektor mapping.
        """
        self._corr_threshold = correlation_threshold
        self._conc_threshold = concentration_threshold
        self._min_points = min_data_points
        self._sector_map = sector_map or _SECTOR_MAP.copy()

    # ── Hovedanalyse ─────────────────────────────────────────

    def analyze(
        self,
        price_data: dict[str, pd.DataFrame] | pd.DataFrame,
        positions: dict[str, float],
        market_data: pd.DataFrame | None = None,
    ) -> CorrelationReport:
        """
        Analysér korrelationer i portefoeljen.

        Args:
            price_data: Dict af symbol->DataFrame med OHLCV, eller
                        én DataFrame med Close-kolonner per symbol.
            positions: Dict af symbol -> vaegt (markedsvaerdi som fraction).
            market_data: Benchmark DataFrame (f.eks. SPY) for beta.

        Returns:
            CorrelationReport med advarsler og forslag.
        """
        symbols = list(positions.keys())
        if len(symbols) < 2:
            return CorrelationReport(
                portfolio_beta=0.0, avg_correlation=0.0, max_correlation=0.0,
            )

        # Byg returns-matrix
        returns_df = self._build_returns(price_data, symbols)
        if returns_df is None or returns_df.empty or len(returns_df) < self._min_points:
            return CorrelationReport(
                portfolio_beta=0.0, avg_correlation=0.0, max_correlation=0.0,
            )

        # Korrelationsmatrix
        corr_matrix = returns_df.corr()

        # Find hoejt korrelerede par
        warnings = self._find_correlated_pairs(corr_matrix, symbols)

        # Beregn gennemsnit og max korrelation
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        avg_corr = float(upper.stack().mean()) if not upper.stack().empty else 0.0
        max_corr = float(upper.stack().max()) if not upper.stack().empty else 0.0

        # Beregn portfolio beta
        beta = self._compute_beta(returns_df, positions, market_data)

        # Koncentrationsadvarsler
        conc_warnings = self._check_concentration(positions)

        # Diversificeringsforslag
        suggestions = self._suggest_diversification(
            warnings, conc_warnings, avg_corr,
        )

        return CorrelationReport(
            portfolio_beta=beta,
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            highly_correlated_pairs=warnings,
            concentration_warnings=conc_warnings,
            diversification_suggestions=suggestions,
            correlation_matrix=corr_matrix,
        )

    # ── Returns-bygning ──────────────────────────────────────

    def _build_returns(
        self,
        price_data: dict[str, pd.DataFrame] | pd.DataFrame,
        symbols: list[str],
    ) -> pd.DataFrame | None:
        """Byg daglige returns DataFrame fra prisdata."""
        if isinstance(price_data, pd.DataFrame):
            # Allerede en samlet DataFrame med Close-kolonner
            if all(s in price_data.columns for s in symbols):
                return price_data[symbols].pct_change().dropna()
            # Prøv Close-suffixed kolonner
            close_cols = {s: f"{s}_Close" for s in symbols}
            available = {s: c for s, c in close_cols.items() if c in price_data.columns}
            if len(available) >= 2:
                df = price_data[list(available.values())]
                df.columns = list(available.keys())
                return df.pct_change().dropna()
            return None

        # Dict af DataFrames
        close_series = {}
        for symbol in symbols:
            if symbol in price_data:
                df = price_data[symbol]
                if "Close" in df.columns:
                    close_series[symbol] = df["Close"]

        if len(close_series) < 2:
            return None

        combined = pd.DataFrame(close_series)
        return combined.pct_change().dropna()

    # ── Korrelationspar ──────────────────────────────────────

    def _find_correlated_pairs(
        self,
        corr_matrix: pd.DataFrame,
        symbols: list[str],
    ) -> list[CorrelationWarning]:
        """Find par med korrelation over threshold."""
        warnings = []
        for i, sym_a in enumerate(symbols):
            for j, sym_b in enumerate(symbols):
                if j <= i:
                    continue
                if sym_a not in corr_matrix.columns or sym_b not in corr_matrix.columns:
                    continue
                corr = corr_matrix.loc[sym_a, sym_b]
                if abs(corr) >= self._corr_threshold:
                    warnings.append(CorrelationWarning(
                        symbol_a=sym_a,
                        symbol_b=sym_b,
                        correlation=float(corr),
                        message=(
                            f"{sym_a} og {sym_b} har korrelation {corr:.2f} "
                            f"(grænse: {self._corr_threshold:.2f})"
                        ),
                    ))
        return sorted(warnings, key=lambda w: abs(w.correlation), reverse=True)

    # ── Portfolio Beta ───────────────────────────────────────

    def _compute_beta(
        self,
        returns_df: pd.DataFrame,
        positions: dict[str, float],
        market_data: pd.DataFrame | None,
    ) -> float:
        """Beregn portfolio beta mod markedet."""
        if market_data is None or market_data.empty:
            return 0.0

        market_returns = market_data["Close"].pct_change().dropna()

        # Portfolio-vægtet returns
        symbols = [s for s in positions if s in returns_df.columns]
        if not symbols:
            return 0.0

        weights = np.array([positions[s] for s in symbols])
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum

        portfolio_returns = (returns_df[symbols] * weights).sum(axis=1)

        # Align dates
        common = portfolio_returns.index.intersection(market_returns.index)
        if len(common) < self._min_points:
            return 0.0

        port_r = portfolio_returns.loc[common].values
        mkt_r = market_returns.loc[common].values

        # Beta = cov(portfolio, market) / var(market)
        cov = np.cov(port_r, mkt_r)
        if cov.shape == (2, 2) and cov[1, 1] > 0:
            return float(cov[0, 1] / cov[1, 1])
        return 0.0

    # ── Koncentration ────────────────────────────────────────

    def _check_concentration(
        self, positions: dict[str, float],
    ) -> list[ConcentrationWarning]:
        """Tjek sektor-koncentration."""
        sector_weights: dict[str, list[tuple[str, float]]] = {}
        for symbol, weight in positions.items():
            sector = self._sector_map.get(symbol, "Unknown")
            if sector not in sector_weights:
                sector_weights[sector] = []
            sector_weights[sector].append((symbol, weight))

        warnings = []
        for sector, entries in sector_weights.items():
            total = sum(w for _, w in entries)
            if total >= self._conc_threshold and len(entries) >= 2:
                syms = [s for s, _ in entries]
                warnings.append(ConcentrationWarning(
                    sector=sector,
                    symbols=syms,
                    total_weight_pct=total,
                    message=(
                        f"Sektor '{sector}' udgoer {total:.0%} af portefoeljen "
                        f"med {len(syms)} positioner ({', '.join(syms)}). "
                        f"Graense: {self._conc_threshold:.0%}."
                    ),
                ))

        return sorted(warnings, key=lambda w: w.total_weight_pct, reverse=True)

    # ── Diversificering ──────────────────────────────────────

    def _suggest_diversification(
        self,
        corr_warnings: list[CorrelationWarning],
        conc_warnings: list[ConcentrationWarning],
        avg_correlation: float,
    ) -> list[DiversificationSuggestion]:
        """Generér diversificeringsforslag."""
        suggestions = []

        # Generel korrelationsadvarsel
        if avg_correlation > 0.60:
            suggestions.append(DiversificationSuggestion(
                reason=f"Gennemsnitlig korrelation er hoej ({avg_correlation:.2f})",
                suggestion=(
                    "Tilfoej ukorrelerede aktiver som guld (GLD), "
                    "obligationer (TLT) eller internationale aktier"
                ),
                current_correlation=avg_correlation,
            ))

        # Sektor-specifikke forslag
        for warning in conc_warnings:
            alt_sectors = _DIVERSIFICATION_MAP.get(warning.sector, [])
            if alt_sectors:
                suggestions.append(DiversificationSuggestion(
                    reason=f"{warning.sector} er overrepresenteret ({warning.total_weight_pct:.0%})",
                    suggestion=f"Overvej at tilfoeje {', '.join(alt_sectors)} for bedre diversificering",
                    current_correlation=0.0,
                ))

        # Parvis korrelation
        for warning in corr_warnings[:3]:
            suggestions.append(DiversificationSuggestion(
                reason=f"{warning.symbol_a} og {warning.symbol_b} er hoejt korrelerede",
                suggestion=f"Overvej at reducere en af positionerne (korr={warning.correlation:.2f})",
                current_correlation=warning.correlation,
            ))

        return suggestions

    # ── Utility ──────────────────────────────────────────────

    def get_sector(self, symbol: str) -> str:
        """Hent sektor for et symbol."""
        return self._sector_map.get(symbol, "Unknown")

    def add_sector_mapping(self, symbol: str, sector: str) -> None:
        """Tilfoej custom sektor-mapping."""
        self._sector_map[symbol] = sector
