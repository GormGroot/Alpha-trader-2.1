"""
Stress-test modul – test porteføljen mod historiske og syntetiske krisescenarier.

Features:
  - 7 historiske kriser (Dot-com, Finanskrise, Flash Crash, COVID, osv.)
  - Syntetiske scenarier (markedsfald, rentehopp, oliekrise, FX-shock)
  - Monte Carlo simulation (10.000 tilfældige scenarier)
  - Regime-detektion under kriser (hvad ville systemet have gjort?)
  - Sammenligning: med vs. uden risikostyring
  - Anbefaling baseret på porteføljesårbarhed

Brug:
    from src.backtest.stress_test import StressTester, CrisisScenario
    tester = StressTester(portfolio_weights={"AAPL": 0.3, "MSFT": 0.3, "SPY": 0.4})
    report = tester.run_all()
    print(report.summary_table())

CLI:
    python -m src.main stress-test --scenario covid
    python -m src.main stress-test --monte-carlo --runs 10000
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


# ══════════════════════════════════════════════════════════════
#  Dataklasser
# ══════════════════════════════════════════════════════════════


class ScenarioType(Enum):
    """Type af stress-scenarie."""
    HISTORICAL = "historical"
    SYNTHETIC = "synthetic"
    MONTE_CARLO = "monte_carlo"


@dataclass
class CrisisScenario:
    """Definition af ét krisescenarie."""
    name: str
    scenario_type: ScenarioType
    description: str
    start_date: str                     # YYYY-MM-DD
    end_date: str
    peak_decline_pct: float             # F.eks. -78.0 for dot-com
    duration_days: int                  # Krisen varighed
    recovery_days: int                  # Dage til fuld recovery
    sector_impacts: dict[str, float]    # Sektor -> ekstra impact multiplier
    key_events: list[str] = field(default_factory=list)

    @property
    def severity(self) -> str:
        """Klassificér alvorlighed."""
        d = abs(self.peak_decline_pct)
        if d >= 40:
            return "EKSTREM"
        if d >= 20:
            return "ALVORLIG"
        if d >= 10:
            return "MODERAT"
        return "LET"


@dataclass
class ScenarioResult:
    """Resultat af én stress-test."""
    scenario: CrisisScenario
    portfolio_value_start: float
    portfolio_value_end: float
    max_drawdown_pct: float
    worst_day_pct: float
    worst_day_date: str
    recovery_days: int                  # Estimeret recovery tid
    daily_values: list[float]           # Dag-for-dag porteføljeværdi
    daily_returns: list[float]          # Daglige afkast
    regime_actions: list[str]           # Hvad regime-systemet ville have gjort
    with_risk_mgmt_end: float           # Slutværdi med risikostyring
    without_risk_mgmt_end: float        # Slutværdi uden risikostyring
    risk_mgmt_saved_pct: float          # Hvor meget risikostyring reddede

    @property
    def total_loss_pct(self) -> float:
        if self.portfolio_value_start == 0:
            return 0.0
        return ((self.portfolio_value_end - self.portfolio_value_start)
                / self.portfolio_value_start) * 100

    @property
    def total_loss_dollar(self) -> float:
        return self.portfolio_value_end - self.portfolio_value_start


@dataclass
class MonteCarloResult:
    """Resultat af Monte Carlo simulation."""
    num_simulations: int
    horizon_days: int
    initial_value: float
    # Percentiler af slutværdier
    worst_case: float           # 1. percentil
    percentile_5: float
    percentile_25: float
    median: float               # 50. percentil
    percentile_75: float
    percentile_95: float
    best_case: float            # 99. percentil
    # Statistik
    mean_return_pct: float
    std_return_pct: float
    prob_loss_pct: float        # Sandsynlighed for tab
    prob_loss_10_pct: float     # Sandsynlighed for >10% tab
    prob_loss_20_pct: float     # Sandsynlighed for >20% tab
    # Alle simulerede slutværdier (til histogrammer)
    final_values: np.ndarray = field(default_factory=lambda: np.array([]))
    # Drawdown-statistik
    max_drawdown_mean: float = 0.0
    max_drawdown_worst: float = 0.0

    @property
    def var_95(self) -> float:
        """Value-at-Risk (95%): max forventet tab ved 95% konfidens."""
        return self.initial_value - self.percentile_5

    @property
    def var_99(self) -> float:
        """Value-at-Risk (99%)."""
        return self.initial_value - self.worst_case


@dataclass
class Vulnerability:
    """Én identificeret sårbarhed."""
    area: str               # F.eks. "Rentestigninger", "Tech-krak"
    severity: str           # "HØJ", "MIDDEL", "LAV"
    description: str
    recommendation: str


@dataclass
class StressTestReport:
    """Samlet rapport fra stress-test."""
    timestamp: str
    portfolio_weights: dict[str, float]
    initial_value: float
    scenario_results: list[ScenarioResult]
    monte_carlo: MonteCarloResult | None
    vulnerabilities: list[Vulnerability]
    overall_risk_rating: str            # "LAV", "MIDDEL", "HØJ", "KRITISK"

    def summary_table(self) -> str:
        """Formatér som tabel."""
        lines = [
            f"{'═' * 90}",
            f"  STRESS-TEST RAPPORT  –  {self.timestamp}",
            f"{'═' * 90}",
            f"  Porteføljeværdi: ${self.initial_value:,.0f}",
            f"  Risiko-rating:   {self.overall_risk_rating}",
            f"",
            f"  {'Scenarie':<28} {'Max tab':>10} {'Recovery':>10} "
            f"{'Worst dag':>10} {'Med RM':>12} {'Uden RM':>12}",
            f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12}",
        ]
        for r in self.scenario_results:
            lines.append(
                f"  {r.scenario.name:<28} "
                f"{r.max_drawdown_pct:>+9.1f}% "
                f"{r.recovery_days:>8}d "
                f"{r.worst_day_pct:>+9.1f}% "
                f"${r.with_risk_mgmt_end:>10,.0f} "
                f"${r.without_risk_mgmt_end:>10,.0f}"
            )

        if self.monte_carlo:
            mc = self.monte_carlo
            lines += [
                f"",
                f"  {'─' * 60}",
                f"  MONTE CARLO ({mc.num_simulations:,} simuleringer, "
                f"{mc.horizon_days} dage)",
                f"  {'─' * 60}",
                f"  Worst case (1%):    ${mc.worst_case:>12,.0f}  "
                f"({(mc.worst_case / mc.initial_value - 1) * 100:+.1f}%)",
                f"  5. percentil:       ${mc.percentile_5:>12,.0f}  "
                f"({(mc.percentile_5 / mc.initial_value - 1) * 100:+.1f}%)",
                f"  Median:             ${mc.median:>12,.0f}  "
                f"({(mc.median / mc.initial_value - 1) * 100:+.1f}%)",
                f"  95. percentil:      ${mc.percentile_95:>12,.0f}  "
                f"({(mc.percentile_95 / mc.initial_value - 1) * 100:+.1f}%)",
                f"  Best case (99%):    ${mc.best_case:>12,.0f}  "
                f"({(mc.best_case / mc.initial_value - 1) * 100:+.1f}%)",
                f"  VaR (95%):          ${mc.var_95:>12,.0f}",
                f"  Prob(tab):          {mc.prob_loss_pct:>11.1f}%",
                f"  Prob(tab > 10%):    {mc.prob_loss_10_pct:>11.1f}%",
                f"  Prob(tab > 20%):    {mc.prob_loss_20_pct:>11.1f}%",
            ]

        if self.vulnerabilities:
            lines += [
                f"",
                f"  {'─' * 60}",
                f"  SÅRBARHEDER",
                f"  {'─' * 60}",
            ]
            for v in self.vulnerabilities:
                sev_icon = {"HØJ": "🔴", "MIDDEL": "🟡", "LAV": "🟢"}.get(
                    v.severity, "⚪"
                )
                lines.append(f"  {sev_icon} [{v.severity}] {v.area}")
                lines.append(f"     {v.description}")
                lines.append(f"     → {v.recommendation}")
                lines.append("")

        lines.append(f"{'═' * 90}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Historiske kriser
# ══════════════════════════════════════════════════════════════

# Sektor-kategorisering for impact-beregning
_SECTOR_MAP: dict[str, str] = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "AMZN": "tech",
    "META": "tech", "NVDA": "tech", "TSLA": "tech", "AMD": "tech",
    "NFLX": "tech", "CRM": "tech", "ADBE": "tech", "INTC": "tech",
    "ORCL": "tech", "CSCO": "tech", "AVGO": "tech",
    "JPM": "finance", "BAC": "finance", "GS": "finance", "MS": "finance",
    "WFC": "finance", "C": "finance", "BRK-B": "finance", "V": "finance",
    "MA": "finance", "AXP": "finance",
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "OXY": "energy", "MPC": "energy", "PSX": "energy",
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "MRK": "healthcare", "ABBV": "healthcare", "LLY": "healthcare",
    "PG": "consumer", "KO": "consumer", "PEP": "consumer", "WMT": "consumer",
    "COST": "consumer", "MCD": "consumer", "NKE": "consumer",
    "XLK": "tech", "XLF": "finance", "XLE": "energy", "XLV": "healthcare",
    "XLP": "consumer", "XLI": "industrial", "XLU": "utilities",
    "SPY": "market", "QQQ": "tech", "DIA": "market", "IWM": "market",
    "GLD": "commodities", "SLV": "commodities", "USO": "energy",
    "TLT": "bonds", "AGG": "bonds", "BND": "bonds",
}


HISTORICAL_CRISES: dict[str, CrisisScenario] = {
    "dotcom": CrisisScenario(
        name="Dot-com krakket (2000-2002)",
        scenario_type=ScenarioType.HISTORICAL,
        description="Teknologi-boblen brast. NASDAQ faldt 78% over 2.5 år.",
        start_date="2000-03-10",
        end_date="2002-10-09",
        peak_decline_pct=-78.0,
        duration_days=945,
        recovery_days=5400,           # ~15 år til NASDAQ genvandt toppen
        sector_impacts={
            "tech": -0.78, "finance": -0.35, "energy": -0.15,
            "healthcare": -0.20, "consumer": -0.25, "market": -0.49,
            "bonds": 0.15, "commodities": 0.05, "utilities": -0.10,
            "industrial": -0.30,
        },
        key_events=[
            "NASDAQ topper 10. marts 2000",
            "Fed hæver renten 6 gange i 1999-2000",
            "Pets.com, Webvan, eToys går konkurs",
            "9/11 lukker markederne i 4 dage",
            "Bund 9. oktober 2002: NASDAQ -78%",
        ],
    ),
    "financial_crisis": CrisisScenario(
        name="Finanskrisen (2008-2009)",
        scenario_type=ScenarioType.HISTORICAL,
        description="Subprime-boligkrise udløste global finanskrise. S&P 500 faldt 57%.",
        start_date="2007-10-09",
        end_date="2009-03-09",
        peak_decline_pct=-57.0,
        duration_days=517,
        recovery_days=1480,           # ~4 år til S&P genvandt
        sector_impacts={
            "finance": -0.83, "market": -0.57, "tech": -0.45,
            "energy": -0.55, "consumer": -0.40, "industrial": -0.55,
            "healthcare": -0.30, "bonds": 0.20, "commodities": -0.35,
            "utilities": -0.35,
        },
        key_events=[
            "S&P 500 topper 9. oktober 2007",
            "Bear Stearns kollapser marts 2008",
            "Lehman Brothers går konkurs 15. sept 2008",
            "Fed redningspakke $700 mia (TARP)",
            "Bund 9. marts 2009: S&P 500 på 666",
        ],
    ),
    "flash_crash": CrisisScenario(
        name="Flash Crash (maj 2010)",
        scenario_type=ScenarioType.HISTORICAL,
        description="S&P 500 faldt 9% på minutter pga. algoritmisk handel.",
        start_date="2010-05-06",
        end_date="2010-05-07",
        peak_decline_pct=-9.0,
        duration_days=1,
        recovery_days=4,              # Intraday recovery + næste par dage
        sector_impacts={
            "tech": -0.09, "finance": -0.10, "market": -0.09,
            "energy": -0.08, "consumer": -0.07, "healthcare": -0.06,
            "bonds": 0.02, "commodities": -0.05, "utilities": -0.05,
            "industrial": -0.08,
        },
        key_events=[
            "Kl 14:32 EST starter salgs-kaskade",
            "Dow Jones falder 1.000 pkt på 5 min",
            "Accenture handles til $0.01",
            "Circuit breakers stopper ikke nok",
            "Markedet genvinder det meste inden lukketid",
        ],
    ),
    "covid": CrisisScenario(
        name="COVID-krakket (marts 2020)",
        scenario_type=ScenarioType.HISTORICAL,
        description="Pandemi-frygt: S&P 500 faldt 34% på 23 handelsdage.",
        start_date="2020-02-19",
        end_date="2020-03-23",
        peak_decline_pct=-34.0,
        duration_days=23,
        recovery_days=148,            # ~5 måneder til nye highs
        sector_impacts={
            "energy": -0.60, "finance": -0.40, "tech": -0.25,
            "consumer": -0.35, "market": -0.34, "industrial": -0.40,
            "healthcare": -0.20, "bonds": 0.05, "commodities": -0.30,
            "utilities": -0.25,
        },
        key_events=[
            "S&P 500 topper 19. februar 2020",
            "WHO erklærer pandemi 11. marts",
            "Circuit breakers udløses 4 gange på 8 dage",
            "Fed sænker renten til 0% + QE unlimited",
            "Bund 23. marts 2020: S&P 500 -34%",
        ],
    ),
    "bear_2022": CrisisScenario(
        name="2022 Bear Market",
        scenario_type=ScenarioType.HISTORICAL,
        description="Inflation + aggressive rentestigninger. S&P 500 faldt 25%.",
        start_date="2022-01-03",
        end_date="2022-10-12",
        peak_decline_pct=-25.0,
        duration_days=282,
        recovery_days=450,
        sector_impacts={
            "tech": -0.33, "finance": -0.20, "market": -0.25,
            "energy": 0.30, "consumer": -0.28, "healthcare": -0.10,
            "bonds": -0.18, "commodities": 0.15, "utilities": -0.05,
            "industrial": -0.18,
        },
        key_events=[
            "Fed signalerer aggressiv stramning",
            "Inflation rammer 9.1% i juni 2022",
            "NASDAQ falder 33%",
            "Fed hæver renten 4.25% på ét år",
            "Energi-aktier stiger 30%+",
        ],
    ),
    "china_2015": CrisisScenario(
        name="Kina-krise (august 2015)",
        scenario_type=ScenarioType.HISTORICAL,
        description="Kina devaluerer yuan. Global nervøsitet spreder sig.",
        start_date="2015-08-17",
        end_date="2015-08-25",
        peak_decline_pct=-12.0,
        duration_days=6,
        recovery_days=130,
        sector_impacts={
            "tech": -0.12, "finance": -0.13, "market": -0.12,
            "energy": -0.15, "consumer": -0.10, "healthcare": -0.08,
            "bonds": 0.03, "commodities": -0.10, "utilities": -0.05,
            "industrial": -0.14,
        },
        key_events=[
            "PBOC devaluerer yuan 11. august",
            "Shanghai Composite falder 8.5% på én dag",
            "Dow Jones åbner -1000 pkt mandag 24. aug",
            "VIX springer fra 13 til 53",
            "Fed udskyder rentestigning",
        ],
    ),
    "brexit": CrisisScenario(
        name="Brexit (juni 2016)",
        scenario_type=ScenarioType.HISTORICAL,
        description="UK stemmer for at forlade EU. Overraskende politisk event.",
        start_date="2016-06-23",
        end_date="2016-06-27",
        peak_decline_pct=-5.3,
        duration_days=2,
        recovery_days=15,
        sector_impacts={
            "finance": -0.08, "market": -0.05, "tech": -0.04,
            "energy": -0.05, "consumer": -0.04, "healthcare": -0.03,
            "bonds": 0.04, "commodities": 0.05, "utilities": -0.02,
            "industrial": -0.05,
        },
        key_events=[
            "Afstemning 23. juni 2016",
            "Leave vinder 52-48%",
            "GBP falder 8% på timer",
            "S&P 500 falder 3.6% fredag 24. juni",
            "Markedet genvinder inden for 2 uger",
        ],
    ),
}


# ── Syntetiske scenarier ──────────────────────────────────────

SYNTHETIC_SCENARIOS: dict[str, CrisisScenario] = {
    "market_crash_20": CrisisScenario(
        name="Marked falder 20% på 1 uge",
        scenario_type=ScenarioType.SYNTHETIC,
        description="Syntetisk scenarie: bredt markedsfald 20% over 5 handelsdage.",
        start_date="syntetisk",
        end_date="syntetisk",
        peak_decline_pct=-20.0,
        duration_days=5,
        recovery_days=180,
        sector_impacts={
            "tech": -0.25, "finance": -0.22, "market": -0.20,
            "energy": -0.18, "consumer": -0.18, "healthcare": -0.15,
            "bonds": 0.03, "commodities": -0.10, "utilities": -0.12,
            "industrial": -0.20,
        },
        key_events=["Simuleret markedsfald: -4% per dag i 5 dage"],
    ),
    "rate_hike_2pct": CrisisScenario(
        name="Renten stiger 2% på 1 måned",
        scenario_type=ScenarioType.SYNTHETIC,
        description="Fed hæver renten aggressivt. Vækstaktier straffes.",
        start_date="syntetisk",
        end_date="syntetisk",
        peak_decline_pct=-15.0,
        duration_days=22,
        recovery_days=250,
        sector_impacts={
            "tech": -0.22, "finance": 0.05, "market": -0.12,
            "energy": 0.00, "consumer": -0.10, "healthcare": -0.08,
            "bonds": -0.15, "commodities": 0.02, "utilities": -0.18,
            "industrial": -0.08,
        },
        key_events=["Simuleret rentestigning: +2% over 22 dage"],
    ),
    "oil_triple": CrisisScenario(
        name="Olie tredobles i pris",
        scenario_type=ScenarioType.SYNTHETIC,
        description="Geopolitisk krise tredobler olieprisen. Inflations-chok.",
        start_date="syntetisk",
        end_date="syntetisk",
        peak_decline_pct=-18.0,
        duration_days=30,
        recovery_days=365,
        sector_impacts={
            "energy": 0.50, "tech": -0.15, "market": -0.18,
            "finance": -0.12, "consumer": -0.22, "healthcare": -0.10,
            "bonds": -0.08, "commodities": 0.40, "utilities": -0.15,
            "industrial": -0.20,
        },
        key_events=["Simuleret oliekrise: energi +50%, transport -30%"],
    ),
    "fx_dkk_shock": CrisisScenario(
        name="USD/DKK rykker 10%",
        scenario_type=ScenarioType.SYNTHETIC,
        description="Dollar svækkes 10% mod DKK. Valutaeksponering rammer.",
        start_date="syntetisk",
        end_date="syntetisk",
        peak_decline_pct=-10.0,
        duration_days=15,
        recovery_days=120,
        sector_impacts={
            "tech": -0.08, "finance": -0.05, "market": -0.10,
            "energy": -0.05, "consumer": -0.08, "healthcare": -0.06,
            "bonds": -0.03, "commodities": 0.05, "utilities": -0.04,
            "industrial": -0.07,
        },
        key_events=["Simuleret valuta-shock: USD svækkes 10% mod DKK"],
    ),
}


# ══════════════════════════════════════════════════════════════
#  StressTester
# ══════════════════════════════════════════════════════════════


class StressTester:
    """
    Kør stress-tests mod portefølje.

    Args:
        portfolio_weights: Dict af symbol -> vægt (0-1), summer til ~1.0.
        initial_value: Porteføljeværdi i USD.
        risk_mgmt_enabled: Simulér risikostyring under kriser.
    """

    def __init__(
        self,
        portfolio_weights: dict[str, float] | None = None,
        initial_value: float = 100_000,
        risk_mgmt_enabled: bool = True,
    ) -> None:
        self._weights = portfolio_weights or {"SPY": 1.0}
        self._initial = initial_value
        self._risk_mgmt = risk_mgmt_enabled

        # Normalisér vægte
        total = sum(self._weights.values())
        if total > 0:
            self._weights = {s: w / total for s, w in self._weights.items()}

    @property
    def portfolio_weights(self) -> dict[str, float]:
        return dict(self._weights)

    @property
    def initial_value(self) -> float:
        return self._initial

    # ── Sektor-eksponering ──────────────────────────────────────

    def _get_sector_exposure(self) -> dict[str, float]:
        """Beregn porteføljens sektoreksponering."""
        exposure: dict[str, float] = {}
        for symbol, weight in self._weights.items():
            sector = _SECTOR_MAP.get(symbol, "other")
            exposure[sector] = exposure.get(sector, 0.0) + weight
        return exposure

    # ── Scenarie-simulering ─────────────────────────────────────

    def run_scenario(self, scenario: CrisisScenario) -> ScenarioResult:
        """
        Simulér porteføljeudvikling under ét krisescenarie.

        1. Beregn daglig porteføljeindvirkning baseret på sektorvægte
        2. Simulér dag-for-dag med realistisk volatilitet
        3. Beregn hvad regime-detektionen ville gøre
        4. Sammenlign med/uden risikostyring
        """
        duration = max(1, scenario.duration_days)
        rng = np.random.RandomState(42)

        # Beregn portfolio-specifik impact
        sector_exp = self._get_sector_exposure()
        weighted_impact = 0.0
        for sector, exp in sector_exp.items():
            sector_impact = scenario.sector_impacts.get(sector, scenario.peak_decline_pct / 100)
            weighted_impact += exp * sector_impact

        # Generér daglige afkast der summer til total impact
        # Bruger Geometric Brownian Motion med drift mod target
        daily_target = (1 + weighted_impact) ** (1 / duration) - 1

        # Tilføj volatilitet proportionelt med krisens alvorlighed
        vol_scale = abs(weighted_impact) / max(1, duration) * 3
        vol_scale = max(0.005, min(vol_scale, 0.08))

        daily_returns_raw = rng.normal(daily_target, vol_scale, size=duration)

        # Justér så total impact matcher nogenlunde
        cumulative = np.cumprod(1 + daily_returns_raw)
        if len(cumulative) > 0 and cumulative[-1] != 0:
            adjustment = (1 + weighted_impact) / cumulative[-1]
            adjustment_per_day = adjustment ** (1 / duration)
            daily_returns_raw = (1 + daily_returns_raw) * adjustment_per_day - 1

        # Simulér porteføljeværdi – UDEN risikostyring
        values_no_rm = [self._initial]
        for ret in daily_returns_raw:
            values_no_rm.append(values_no_rm[-1] * (1 + ret))

        # Simulér MED risikostyring
        values_with_rm = [self._initial]
        rm_exposure = 1.0
        rm_actions: list[str] = []

        for i, ret in enumerate(daily_returns_raw):
            current_dd = (values_with_rm[-1] / self._initial - 1) * 100
            daily_pct = ret * 100

            # Regime-simulering: reducer eksponering ved store tab
            if daily_pct < -3.0 and rm_exposure > 0.5:
                old_exp = rm_exposure
                rm_exposure = max(0.3, rm_exposure * 0.6)
                rm_actions.append(
                    f"Dag {i+1}: Dagligt tab {daily_pct:.1f}% → "
                    f"Reducer eksponering {old_exp:.0%} → {rm_exposure:.0%}"
                )
            elif daily_pct < -1.5 and rm_exposure > 0.3:
                old_exp = rm_exposure
                rm_exposure = max(0.3, rm_exposure * 0.85)
                rm_actions.append(
                    f"Dag {i+1}: Tab {daily_pct:.1f}% → "
                    f"Reducer til {rm_exposure:.0%}"
                )

            if current_dd < -7.0 and rm_exposure > 0.1:
                rm_exposure = 0.1
                rm_actions.append(
                    f"Dag {i+1}: Circuit breaker: Drawdown {current_dd:.1f}% "
                    f"→ STOP handel (10% eksponering)"
                )
            elif current_dd < -3.0 and rm_exposure > 0.3:
                rm_exposure = min(rm_exposure, 0.3)

            # CRASH-regime: eksponering max 10%
            if current_dd < -15.0:
                rm_exposure = 0.1
                if i == 0 or (values_with_rm[-1] / self._initial - 1) * 100 >= -15:
                    rm_actions.append(
                        f"Dag {i+1}: CRASH regime detekteret → Max 10% eksponering"
                    )

            adjusted_ret = ret * rm_exposure
            values_with_rm.append(values_with_rm[-1] * (1 + adjusted_ret))

        if not rm_actions:
            rm_actions.append("Ingen indgreb nødvendig – tab under tærskler")

        # Beregn metrics
        values_arr = np.array(values_no_rm)
        peaks = np.maximum.accumulate(values_arr)
        drawdowns = (values_arr - peaks) / peaks * 100

        daily_rets_list = list(daily_returns_raw * 100)
        worst_day_idx = int(np.argmin(daily_returns_raw))
        worst_day_pct = float(daily_returns_raw[worst_day_idx]) * 100

        # Recovery estimation
        end_dd = (values_no_rm[-1] / self._initial - 1) * 100
        if end_dd < 0:
            # Approx recovery baseret på historisk gennemsnit (8% årligt)
            annual_recovery = 0.08
            loss_pct = abs(end_dd) / 100
            if annual_recovery > 0 and loss_pct > 0:
                recovery_years = loss_pct / annual_recovery
                est_recovery = int(recovery_years * 252)
            else:
                est_recovery = scenario.recovery_days
        else:
            est_recovery = 0

        return ScenarioResult(
            scenario=scenario,
            portfolio_value_start=self._initial,
            portfolio_value_end=values_no_rm[-1],
            max_drawdown_pct=float(drawdowns.min()),
            worst_day_pct=worst_day_pct,
            worst_day_date=f"Dag {worst_day_idx + 1}",
            recovery_days=est_recovery,
            daily_values=values_no_rm,
            daily_returns=daily_rets_list,
            regime_actions=rm_actions,
            with_risk_mgmt_end=values_with_rm[-1],
            without_risk_mgmt_end=values_no_rm[-1],
            risk_mgmt_saved_pct=(
                (values_with_rm[-1] - values_no_rm[-1]) / self._initial * 100
                if self._initial > 0 else 0.0
            ),
        )

    # ── Monte Carlo ──────────────────────────────────────────────

    def monte_carlo(
        self,
        num_simulations: int = 10_000,
        horizon_days: int = 252,
        annual_return: float = 0.08,
        annual_volatility: float = 0.20,
        seed: int | None = 42,
    ) -> MonteCarloResult:
        """
        Monte Carlo simulation baseret på historisk volatilitet.

        Simulerer GBM (Geometric Brownian Motion):
          dS = μ·S·dt + σ·S·dW

        Args:
            num_simulations: Antal simuleringer.
            horizon_days: Horisont i handelsdage (252 = 1 år).
            annual_return: Forventet årligt afkast.
            annual_volatility: Årlig volatilitet.
            seed: Random seed for reproducerbarhed.
        """
        rng = np.random.RandomState(seed)

        dt = 1 / 252
        daily_mu = (annual_return - 0.5 * annual_volatility**2) * dt
        daily_sigma = annual_volatility * np.sqrt(dt)

        # Generér alle stier på én gang (matrix: simulations × days)
        random_shocks = rng.normal(0, 1, size=(num_simulations, horizon_days))
        daily_returns = daily_mu + daily_sigma * random_shocks

        # Kumulerede afkast
        cumulative = np.exp(np.cumsum(daily_returns, axis=1))
        paths = self._initial * cumulative

        final_values = paths[:, -1]

        # Max drawdown per sti
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - running_max) / running_max
        max_drawdowns = drawdowns.min(axis=1) * 100

        # Statistik
        returns_pct = (final_values / self._initial - 1) * 100

        return MonteCarloResult(
            num_simulations=num_simulations,
            horizon_days=horizon_days,
            initial_value=self._initial,
            worst_case=float(np.percentile(final_values, 1)),
            percentile_5=float(np.percentile(final_values, 5)),
            percentile_25=float(np.percentile(final_values, 25)),
            median=float(np.percentile(final_values, 50)),
            percentile_75=float(np.percentile(final_values, 75)),
            percentile_95=float(np.percentile(final_values, 95)),
            best_case=float(np.percentile(final_values, 99)),
            mean_return_pct=float(returns_pct.mean()),
            std_return_pct=float(returns_pct.std()),
            prob_loss_pct=float((final_values < self._initial).mean() * 100),
            prob_loss_10_pct=float(
                (final_values < self._initial * 0.90).mean() * 100
            ),
            prob_loss_20_pct=float(
                (final_values < self._initial * 0.80).mean() * 100
            ),
            final_values=final_values,
            max_drawdown_mean=float(max_drawdowns.mean()),
            max_drawdown_worst=float(max_drawdowns.min()),
        )

    # ── Sårbarhedsanalyse ──────────────────────────────────────

    def _analyze_vulnerabilities(
        self, results: list[ScenarioResult],
    ) -> list[Vulnerability]:
        """Identificér porteføljens sårbarheder baseret på stress-test resultater."""
        vulns: list[Vulnerability] = []
        sector_exp = self._get_sector_exposure()

        # 1. Tech-eksponering
        tech_pct = sector_exp.get("tech", 0) * 100
        if tech_pct > 40:
            vulns.append(Vulnerability(
                area="Høj tech-eksponering",
                severity="HØJ",
                description=(
                    f"Porteføljen har {tech_pct:.0f}% tech-eksponering. "
                    f"Under dot-com faldt tech 78%."
                ),
                recommendation=(
                    "Overvej at diversificere med energi (XLE), "
                    "sundhed (XLV) og forbrugsvarer (XLP)."
                ),
            ))
        elif tech_pct > 25:
            vulns.append(Vulnerability(
                area="Moderat tech-eksponering",
                severity="MIDDEL",
                description=f"{tech_pct:.0f}% tech – sårbar overfor rentestigninger.",
                recommendation="Tilføj obligations-ETF'er (TLT, AGG) som hedge.",
            ))

        # 2. Finansiel eksponering
        fin_pct = sector_exp.get("finance", 0) * 100
        if fin_pct > 30:
            vulns.append(Vulnerability(
                area="Høj finansiel eksponering",
                severity="HØJ",
                description=(
                    f"{fin_pct:.0f}% i finanssektoren. "
                    f"Under finanskrisen faldt banker 83%."
                ),
                recommendation="Reducer bankeksponering, tilføj guld (GLD) som hedge.",
            ))

        # 3. Energi-afhængighed
        energy_pct = sector_exp.get("energy", 0) * 100
        if energy_pct > 20:
            vulns.append(Vulnerability(
                area="Energi-koncentration",
                severity="MIDDEL",
                description=f"{energy_pct:.0f}% energi – volatil sektor.",
                recommendation="Balancér med defensive sektorer (sundhed, forbrugsvarer).",
            ))

        # 4. Mangel på diversificering
        if len(self._weights) < 5:
            vulns.append(Vulnerability(
                area="Lav diversificering",
                severity="HØJ",
                description=(
                    f"Kun {len(self._weights)} positioner. "
                    f"Enkeltaktie-risiko er høj."
                ),
                recommendation=(
                    "Tilføj minimum 10-15 positioner eller brug "
                    "bred indeks-ETF (SPY, VTI)."
                ),
            ))

        # 5. Ingen obligationer/guld (hedge)
        bond_pct = sector_exp.get("bonds", 0) * 100
        gold_pct = sector_exp.get("commodities", 0) * 100
        if bond_pct < 5 and gold_pct < 5:
            vulns.append(Vulnerability(
                area="Ingen hedge-aktiver",
                severity="MIDDEL",
                description=(
                    "Ingen obligationer eller guld i porteføljen. "
                    "Ingen beskyttelse under krak."
                ),
                recommendation=(
                    "Allokér 10-20% til obligationer (TLT/AGG) "
                    "og 5% til guld (GLD)."
                ),
            ))

        # 6. Rentesårbarhed
        worst_rate_result = None
        for r in results:
            if "rent" in r.scenario.name.lower() or "rate" in r.scenario.name.lower():
                worst_rate_result = r
                break
        if worst_rate_result and worst_rate_result.max_drawdown_pct < -12:
            vulns.append(Vulnerability(
                area="Rentesårbarhed",
                severity="HØJ",
                description=(
                    f"Porteføljen falder {worst_rate_result.max_drawdown_pct:.1f}% "
                    f"ved rentestigninger. Høj duration-risiko."
                ),
                recommendation=(
                    "Reducer vækstaktier og lang-duration obligationer. "
                    "Tilføj value-aktier og korte obligationer."
                ),
            ))

        # 7. Crash-sårbarhed (COVID/flash)
        crash_results = [r for r in results if r.max_drawdown_pct < -25]
        if len(crash_results) >= 2:
            avg_crash = np.mean([r.max_drawdown_pct for r in crash_results])
            vulns.append(Vulnerability(
                area="Generel krak-sårbarhed",
                severity="HØJ",
                description=(
                    f"Porteføljen mister gennemsnitlig {abs(avg_crash):.0f}% "
                    f"i krak-scenarier."
                ),
                recommendation=(
                    "Aktiver regime-baseret risikostyring. "
                    "Circuit breakers reducerer tab markant."
                ),
            ))

        return vulns

    def _compute_risk_rating(
        self,
        results: list[ScenarioResult],
        mc: MonteCarloResult | None,
    ) -> str:
        """Beregn samlet risikovurdering."""
        score = 0

        # Worst scenarie
        worst = min(r.max_drawdown_pct for r in results) if results else 0
        if worst < -50:
            score += 4
        elif worst < -30:
            score += 3
        elif worst < -15:
            score += 2
        else:
            score += 1

        # Monte Carlo tab-sandsynlighed
        if mc:
            if mc.prob_loss_20_pct > 20:
                score += 3
            elif mc.prob_loss_10_pct > 20:
                score += 2
            elif mc.prob_loss_pct > 40:
                score += 1

        # Diversificering
        if len(self._weights) < 5:
            score += 2
        elif len(self._weights) < 10:
            score += 1

        if score >= 7:
            return "KRITISK"
        if score >= 5:
            return "HØJ"
        if score >= 3:
            return "MIDDEL"
        return "LAV"

    # ── Kør alt ──────────────────────────────────────────────────

    def run_historical(self) -> list[ScenarioResult]:
        """Kør alle historiske kriser."""
        results = []
        for key, scenario in HISTORICAL_CRISES.items():
            logger.info(f"Stress-test: {scenario.name}")
            results.append(self.run_scenario(scenario))
        return results

    def run_synthetic(self) -> list[ScenarioResult]:
        """Kør alle syntetiske scenarier."""
        results = []
        for key, scenario in SYNTHETIC_SCENARIOS.items():
            logger.info(f"Stress-test: {scenario.name}")
            results.append(self.run_scenario(scenario))
        return results

    def run_all(
        self,
        include_monte_carlo: bool = True,
        monte_carlo_runs: int = 10_000,
    ) -> StressTestReport:
        """
        Kør komplet stress-test suite.

        Returns:
            StressTestReport med alle resultater og anbefalinger.
        """
        logger.info(
            f"Starter stress-test: {len(self._weights)} positioner, "
            f"${self._initial:,.0f}"
        )

        # Historiske scenarier
        hist_results = self.run_historical()

        # Syntetiske scenarier
        synth_results = self.run_synthetic()

        all_results = hist_results + synth_results

        # Monte Carlo
        mc_result = None
        if include_monte_carlo:
            logger.info(f"Monte Carlo: {monte_carlo_runs:,} simuleringer...")
            mc_result = self.monte_carlo(
                num_simulations=monte_carlo_runs,
                horizon_days=252,
            )

        # Analyse
        vulnerabilities = self._analyze_vulnerabilities(all_results)
        risk_rating = self._compute_risk_rating(all_results, mc_result)

        report = StressTestReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            portfolio_weights=self._weights,
            initial_value=self._initial,
            scenario_results=all_results,
            monte_carlo=mc_result,
            vulnerabilities=vulnerabilities,
            overall_risk_rating=risk_rating,
        )

        logger.info(
            f"Stress-test færdig: {len(all_results)} scenarier, "
            f"rating={risk_rating}"
        )
        return report

    def run_single(self, scenario_key: str) -> ScenarioResult | None:
        """
        Kør ét specifikt scenarie.

        Args:
            scenario_key: Nøgle fra HISTORICAL_CRISES eller SYNTHETIC_SCENARIOS.
                          F.eks. "covid", "dotcom", "market_crash_20".
        """
        scenario = HISTORICAL_CRISES.get(scenario_key) or SYNTHETIC_SCENARIOS.get(scenario_key)
        if not scenario:
            logger.warning(f"Ukendt scenarie: {scenario_key}")
            return None
        return self.run_scenario(scenario)

    # ── Skat i krak ──────────────────────────────────────────────

    def tax_impact_in_crash(
        self,
        scenario: CrisisScenario,
        fx_rate: float = 6.85,
    ) -> dict:
        """
        Beregn skatteimplikation ved realisation under krise.

        Scenarie: Investor sælger alt under panik.
        Antag kostpris = initial_value (nyligt købt).
        """
        result = self.run_scenario(scenario)
        loss_usd = result.total_loss_dollar
        loss_dkk = loss_usd * fx_rate

        # Dansk skat: tab kan fratrækkes aktieindkomst 27%/42%
        tax_deduction_27 = abs(loss_dkk) * 0.27 if loss_dkk < 0 else 0
        tax_deduction_42 = abs(loss_dkk) * 0.42 if loss_dkk < 0 else 0

        return {
            "scenario": scenario.name,
            "loss_usd": loss_usd,
            "loss_dkk": loss_dkk,
            "tax_deduction_27pct": tax_deduction_27,
            "tax_deduction_42pct": tax_deduction_42,
            "net_loss_after_tax_27": loss_dkk + tax_deduction_27,
            "net_loss_after_tax_42": loss_dkk + tax_deduction_42,
            "advice": (
                "Tab på aktier kan fratrækkes i aktieindkomst. "
                "Ved tab > progressionsgrænsen (61.000 DKK) gives fradrag på 42%."
                if loss_dkk < 0
                else "Ingen tab – ingen skattefradrag nødvendig."
            ),
        }
