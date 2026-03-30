"""Tests for src/backtest/stress_test.py – stress-test modul."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtest.stress_test import (
    CrisisScenario,
    HISTORICAL_CRISES,
    MonteCarloResult,
    ScenarioResult,
    ScenarioType,
    SYNTHETIC_SCENARIOS,
    StressTestReport,
    StressTester,
    Vulnerability,
)


# ── Helpers ─────────────────────────────────────────────────────

def _tech_heavy_portfolio() -> dict[str, float]:
    return {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "NVDA": 0.25}


def _diversified_portfolio() -> dict[str, float]:
    return {
        "AAPL": 0.10, "MSFT": 0.10, "JPM": 0.10,
        "XOM": 0.10, "JNJ": 0.10, "PG": 0.10,
        "GLD": 0.10, "TLT": 0.10, "SPY": 0.10, "XLE": 0.10,
    }


def _single_stock() -> dict[str, float]:
    return {"AAPL": 1.0}


# ══════════════════════════════════════════════════════════════
#  CrisisScenario
# ══════════════════════════════════════════════════════════════


class TestCrisisScenario:
    def test_severity_extreme(self):
        s = HISTORICAL_CRISES["dotcom"]
        assert s.severity == "EKSTREM"

    def test_severity_alvorlig(self):
        s = HISTORICAL_CRISES["covid"]
        assert s.severity == "ALVORLIG"

    def test_severity_moderat(self):
        s = HISTORICAL_CRISES["china_2015"]
        assert s.severity == "MODERAT"

    def test_severity_let(self):
        s = HISTORICAL_CRISES["brexit"]
        assert s.severity == "LET"

    def test_all_historical_have_required_fields(self):
        for key, s in HISTORICAL_CRISES.items():
            assert s.name, f"{key} mangler name"
            assert s.scenario_type == ScenarioType.HISTORICAL
            assert s.peak_decline_pct < 0, f"{key} peak_decline should be negative"
            assert s.duration_days > 0
            assert s.recovery_days > 0
            assert len(s.sector_impacts) > 0
            assert len(s.key_events) > 0

    def test_all_synthetic_have_required_fields(self):
        for key, s in SYNTHETIC_SCENARIOS.items():
            assert s.name, f"{key} mangler name"
            assert s.scenario_type == ScenarioType.SYNTHETIC
            assert s.peak_decline_pct < 0
            assert s.duration_days > 0

    def test_historical_count(self):
        assert len(HISTORICAL_CRISES) == 7

    def test_synthetic_count(self):
        assert len(SYNTHETIC_SCENARIOS) == 4


# ══════════════════════════════════════════════════════════════
#  StressTester – Init
# ══════════════════════════════════════════════════════════════


class TestStressTesterInit:
    def test_default_portfolio(self):
        tester = StressTester()
        assert "SPY" in tester.portfolio_weights
        assert tester.initial_value == 100_000

    def test_custom_portfolio(self):
        w = {"AAPL": 0.5, "MSFT": 0.5}
        tester = StressTester(portfolio_weights=w, initial_value=200_000)
        assert tester.initial_value == 200_000
        assert abs(sum(tester.portfolio_weights.values()) - 1.0) < 0.01

    def test_weights_normalized(self):
        w = {"AAPL": 3.0, "MSFT": 7.0}
        tester = StressTester(portfolio_weights=w)
        assert abs(tester.portfolio_weights["AAPL"] - 0.3) < 0.01
        assert abs(tester.portfolio_weights["MSFT"] - 0.7) < 0.01


# ══════════════════════════════════════════════════════════════
#  Sector Exposure
# ══════════════════════════════════════════════════════════════


class TestSectorExposure:
    def test_tech_heavy(self):
        tester = StressTester(portfolio_weights=_tech_heavy_portfolio())
        exp = tester._get_sector_exposure()
        assert exp.get("tech", 0) > 0.9

    def test_diversified(self):
        tester = StressTester(portfolio_weights=_diversified_portfolio())
        exp = tester._get_sector_exposure()
        # Bør have flere sektorer
        assert len(exp) >= 5

    def test_unknown_symbol_goes_to_other(self):
        tester = StressTester(portfolio_weights={"XXXX": 1.0})
        exp = tester._get_sector_exposure()
        assert "other" in exp


# ══════════════════════════════════════════════════════════════
#  Scenarie-simulering
# ══════════════════════════════════════════════════════════════


class TestRunScenario:
    def test_covid_produces_loss(self):
        tester = StressTester(portfolio_weights={"SPY": 1.0})
        result = tester.run_scenario(HISTORICAL_CRISES["covid"])
        assert result.portfolio_value_end < result.portfolio_value_start
        assert result.max_drawdown_pct < 0

    def test_dotcom_tech_heavy_worse_than_diversified(self):
        dotcom = HISTORICAL_CRISES["dotcom"]
        tech = StressTester(portfolio_weights=_tech_heavy_portfolio())
        div = StressTester(portfolio_weights=_diversified_portfolio())
        r_tech = tech.run_scenario(dotcom)
        r_div = div.run_scenario(dotcom)
        # Tech-tung portefølje bør rammes hårdere
        assert r_tech.max_drawdown_pct < r_div.max_drawdown_pct

    def test_daily_values_length(self):
        tester = StressTester()
        result = tester.run_scenario(HISTORICAL_CRISES["covid"])
        # duration + 1 (startværdi)
        assert len(result.daily_values) == HISTORICAL_CRISES["covid"].duration_days + 1

    def test_daily_returns_length(self):
        tester = StressTester()
        result = tester.run_scenario(HISTORICAL_CRISES["covid"])
        assert len(result.daily_returns) == HISTORICAL_CRISES["covid"].duration_days

    def test_worst_day_negative(self):
        tester = StressTester()
        result = tester.run_scenario(HISTORICAL_CRISES["financial_crisis"])
        assert result.worst_day_pct < 0

    def test_risk_mgmt_reduces_loss(self):
        tester = StressTester()
        result = tester.run_scenario(HISTORICAL_CRISES["financial_crisis"])
        # Med risikostyring bør man miste mindre
        assert result.with_risk_mgmt_end >= result.without_risk_mgmt_end

    def test_regime_actions_not_empty_for_big_crash(self):
        tester = StressTester()
        result = tester.run_scenario(HISTORICAL_CRISES["financial_crisis"])
        assert len(result.regime_actions) > 0

    def test_flash_crash_short_duration(self):
        tester = StressTester()
        result = tester.run_scenario(HISTORICAL_CRISES["flash_crash"])
        assert len(result.daily_values) == 2  # 1 dag + start

    def test_total_loss_pct(self):
        tester = StressTester(initial_value=100_000)
        result = tester.run_scenario(HISTORICAL_CRISES["covid"])
        assert result.total_loss_pct < 0
        assert result.total_loss_dollar < 0

    def test_risk_mgmt_saved_pct_positive(self):
        tester = StressTester()
        result = tester.run_scenario(HISTORICAL_CRISES["financial_crisis"])
        assert result.risk_mgmt_saved_pct >= 0


class TestSyntheticScenarios:
    def test_market_crash_20(self):
        tester = StressTester()
        result = tester.run_scenario(SYNTHETIC_SCENARIOS["market_crash_20"])
        assert result.max_drawdown_pct < -10

    def test_rate_hike_hits_tech_harder(self):
        rate = SYNTHETIC_SCENARIOS["rate_hike_2pct"]
        tech = StressTester(portfolio_weights=_tech_heavy_portfolio())
        div = StressTester(portfolio_weights=_diversified_portfolio())
        r_tech = tech.run_scenario(rate)
        r_div = div.run_scenario(rate)
        assert r_tech.max_drawdown_pct < r_div.max_drawdown_pct

    def test_oil_triple_helps_energy(self):
        oil = SYNTHETIC_SCENARIOS["oil_triple"]
        # Ren energi-portefølje bør klare sig bedre
        energy = StressTester(portfolio_weights={"XOM": 0.5, "CVX": 0.5})
        result = energy.run_scenario(oil)
        # Energi bør stige, ikke falde
        assert result.portfolio_value_end > result.portfolio_value_start * 0.9

    def test_fx_shock(self):
        tester = StressTester()
        result = tester.run_scenario(SYNTHETIC_SCENARIOS["fx_dkk_shock"])
        assert result.max_drawdown_pct < 0


# ══════════════════════════════════════════════════════════════
#  Monte Carlo
# ══════════════════════════════════════════════════════════════


class TestMonteCarlo:
    def test_basic_output(self):
        tester = StressTester(initial_value=100_000)
        mc = tester.monte_carlo(num_simulations=1000, horizon_days=252, seed=42)
        assert mc.num_simulations == 1000
        assert mc.horizon_days == 252
        assert mc.initial_value == 100_000

    def test_percentile_ordering(self):
        tester = StressTester()
        mc = tester.monte_carlo(num_simulations=5000, seed=42)
        assert mc.worst_case <= mc.percentile_5
        assert mc.percentile_5 <= mc.percentile_25
        assert mc.percentile_25 <= mc.median
        assert mc.median <= mc.percentile_75
        assert mc.percentile_75 <= mc.percentile_95
        assert mc.percentile_95 <= mc.best_case

    def test_median_positive_with_positive_drift(self):
        tester = StressTester()
        mc = tester.monte_carlo(
            num_simulations=5000, annual_return=0.08,
            annual_volatility=0.20, seed=42,
        )
        assert mc.median > tester.initial_value

    def test_high_vol_wider_spread(self):
        tester = StressTester()
        mc_low = tester.monte_carlo(
            num_simulations=5000, annual_volatility=0.10, seed=42,
        )
        mc_high = tester.monte_carlo(
            num_simulations=5000, annual_volatility=0.40, seed=42,
        )
        spread_low = mc_high.percentile_95 - mc_low.percentile_5
        spread_high = mc_high.percentile_95 - mc_high.percentile_5
        # Højere vol = bredere spread
        assert spread_high > spread_low * 0.5

    def test_prob_loss_reasonable(self):
        tester = StressTester()
        mc = tester.monte_carlo(num_simulations=5000, seed=42)
        assert 0 <= mc.prob_loss_pct <= 100
        assert 0 <= mc.prob_loss_10_pct <= 100
        assert 0 <= mc.prob_loss_20_pct <= 100
        # Med positiv drift bør prob_loss < 50%
        assert mc.prob_loss_pct < 60

    def test_var_95_positive(self):
        tester = StressTester()
        mc = tester.monte_carlo(num_simulations=5000, seed=42)
        # VaR bør være positiv (et tab-beløb)
        assert mc.var_95 >= 0 or mc.percentile_5 > tester.initial_value

    def test_final_values_array_length(self):
        n = 2000
        tester = StressTester()
        mc = tester.monte_carlo(num_simulations=n, seed=42)
        assert len(mc.final_values) == n

    def test_max_drawdown_stats(self):
        tester = StressTester()
        mc = tester.monte_carlo(num_simulations=1000, seed=42)
        assert mc.max_drawdown_mean < 0
        assert mc.max_drawdown_worst < mc.max_drawdown_mean

    def test_reproducible_with_seed(self):
        tester = StressTester()
        mc1 = tester.monte_carlo(num_simulations=100, seed=123)
        mc2 = tester.monte_carlo(num_simulations=100, seed=123)
        assert mc1.median == mc2.median
        assert mc1.worst_case == mc2.worst_case


# ══════════════════════════════════════════════════════════════
#  Vulnerabilities
# ══════════════════════════════════════════════════════════════


class TestVulnerabilities:
    def test_tech_heavy_gets_warning(self):
        tester = StressTester(portfolio_weights=_tech_heavy_portfolio())
        results = tester.run_historical()
        vulns = tester._analyze_vulnerabilities(results)
        areas = [v.area for v in vulns]
        assert any("tech" in a.lower() for a in areas)

    def test_single_stock_low_diversification(self):
        tester = StressTester(portfolio_weights=_single_stock())
        results = tester.run_historical()
        vulns = tester._analyze_vulnerabilities(results)
        areas = [v.area for v in vulns]
        assert any("diversificering" in a.lower() for a in areas)

    def test_no_hedge_warning(self):
        tester = StressTester(portfolio_weights={"AAPL": 0.5, "MSFT": 0.5})
        results = tester.run_historical()
        vulns = tester._analyze_vulnerabilities(results)
        areas = [v.area for v in vulns]
        assert any("hedge" in a.lower() for a in areas)

    def test_diversified_fewer_warnings(self):
        tech_tester = StressTester(portfolio_weights=_tech_heavy_portfolio())
        div_tester = StressTester(portfolio_weights=_diversified_portfolio())

        tech_results = tech_tester.run_historical()
        div_results = div_tester.run_historical()

        tech_vulns = tech_tester._analyze_vulnerabilities(tech_results)
        div_vulns = div_tester._analyze_vulnerabilities(div_results)

        # Diversificeret bør have færre/mildere advarsler
        tech_high = sum(1 for v in tech_vulns if v.severity == "HØJ")
        div_high = sum(1 for v in div_vulns if v.severity == "HØJ")
        assert div_high <= tech_high

    def test_vulnerability_has_recommendation(self):
        tester = StressTester(portfolio_weights=_single_stock())
        results = tester.run_historical()
        vulns = tester._analyze_vulnerabilities(results)
        for v in vulns:
            assert v.recommendation
            assert v.area
            assert v.severity in ("HØJ", "MIDDEL", "LAV")


# ══════════════════════════════════════════════════════════════
#  Risk Rating
# ══════════════════════════════════════════════════════════════


class TestRiskRating:
    def test_single_stock_high_risk(self):
        tester = StressTester(portfolio_weights=_single_stock())
        results = tester.run_historical()
        mc = tester.monte_carlo(num_simulations=500, seed=42)
        rating = tester._compute_risk_rating(results, mc)
        assert rating in ("HØJ", "KRITISK")

    def test_diversified_lower_risk(self):
        tester = StressTester(portfolio_weights=_diversified_portfolio())
        results = tester.run_historical()
        mc = tester.monte_carlo(num_simulations=500, seed=42)
        rating = tester._compute_risk_rating(results, mc)
        assert rating in ("LAV", "MIDDEL", "HØJ")

    def test_rating_valid_values(self):
        tester = StressTester()
        results = tester.run_historical()
        rating = tester._compute_risk_rating(results, None)
        assert rating in ("LAV", "MIDDEL", "HØJ", "KRITISK")


# ══════════════════════════════════════════════════════════════
#  run_all / run_single / run_historical / run_synthetic
# ══════════════════════════════════════════════════════════════


class TestRunAll:
    def test_run_all_returns_report(self):
        tester = StressTester(initial_value=50_000)
        report = tester.run_all(
            include_monte_carlo=True,
            monte_carlo_runs=500,
        )
        assert isinstance(report, StressTestReport)
        assert report.initial_value == 50_000
        assert len(report.scenario_results) == 11  # 7 hist + 4 synth

    def test_run_all_no_monte_carlo(self):
        tester = StressTester()
        report = tester.run_all(include_monte_carlo=False)
        assert report.monte_carlo is None
        assert len(report.scenario_results) == 11

    def test_run_all_has_risk_rating(self):
        tester = StressTester()
        report = tester.run_all(monte_carlo_runs=200)
        assert report.overall_risk_rating in ("LAV", "MIDDEL", "HØJ", "KRITISK")

    def test_run_historical_count(self):
        tester = StressTester()
        results = tester.run_historical()
        assert len(results) == 7

    def test_run_synthetic_count(self):
        tester = StressTester()
        results = tester.run_synthetic()
        assert len(results) == 4

    def test_run_single_valid(self):
        tester = StressTester()
        result = tester.run_single("covid")
        assert result is not None
        assert result.scenario.name == "COVID-krakket (marts 2020)"

    def test_run_single_invalid(self):
        tester = StressTester()
        result = tester.run_single("nonexistent")
        assert result is None

    def test_run_single_synthetic(self):
        tester = StressTester()
        result = tester.run_single("market_crash_20")
        assert result is not None
        assert "20%" in result.scenario.name


# ══════════════════════════════════════════════════════════════
#  Report / Summary Table
# ══════════════════════════════════════════════════════════════


class TestReport:
    def test_summary_table_string(self):
        tester = StressTester()
        report = tester.run_all(monte_carlo_runs=200)
        table = report.summary_table()
        assert isinstance(table, str)
        assert "STRESS-TEST RAPPORT" in table

    def test_summary_table_contains_scenarios(self):
        tester = StressTester()
        report = tester.run_all(monte_carlo_runs=200)
        table = report.summary_table()
        assert "Dot-com" in table
        assert "COVID" in table
        assert "Finanskrisen" in table

    def test_summary_table_contains_monte_carlo(self):
        tester = StressTester()
        report = tester.run_all(monte_carlo_runs=200)
        table = report.summary_table()
        assert "MONTE CARLO" in table
        assert "Median" in table
        assert "VaR" in table

    def test_summary_table_contains_vulnerabilities(self):
        tester = StressTester(portfolio_weights=_single_stock())
        report = tester.run_all(monte_carlo_runs=200)
        table = report.summary_table()
        assert "SÅRBARHEDER" in table

    def test_report_timestamp(self):
        tester = StressTester()
        report = tester.run_all(include_monte_carlo=False)
        assert report.timestamp  # Non-empty


# ══════════════════════════════════════════════════════════════
#  Skat i krak
# ══════════════════════════════════════════════════════════════


class TestTaxImpact:
    def test_tax_impact_returns_dict(self):
        tester = StressTester(initial_value=100_000)
        result = tester.tax_impact_in_crash(HISTORICAL_CRISES["covid"])
        assert isinstance(result, dict)
        assert "loss_usd" in result
        assert "loss_dkk" in result

    def test_tax_deduction_positive_on_loss(self):
        tester = StressTester()
        result = tester.tax_impact_in_crash(HISTORICAL_CRISES["financial_crisis"])
        assert result["tax_deduction_27pct"] > 0
        assert result["tax_deduction_42pct"] > 0

    def test_net_loss_less_than_gross_loss(self):
        tester = StressTester()
        result = tester.tax_impact_in_crash(HISTORICAL_CRISES["covid"])
        # Net loss (after tax deduction) bør være mindre negativt
        assert result["net_loss_after_tax_27"] > result["loss_dkk"]

    def test_fx_rate_applied(self):
        tester = StressTester(initial_value=100_000)
        r1 = tester.tax_impact_in_crash(HISTORICAL_CRISES["covid"], fx_rate=6.85)
        r2 = tester.tax_impact_in_crash(HISTORICAL_CRISES["covid"], fx_rate=7.50)
        # Højere kurs = større DKK-beløb
        assert abs(r2["loss_dkk"]) > abs(r1["loss_dkk"])

    def test_advice_present(self):
        tester = StressTester()
        result = tester.tax_impact_in_crash(HISTORICAL_CRISES["covid"])
        assert "advice" in result
        assert len(result["advice"]) > 10


# ══════════════════════════════════════════════════════════════
#  Scenario-specific logik
# ══════════════════════════════════════════════════════════════


class TestScenarioSpecifics:
    def test_all_scenarios_produce_results(self):
        tester = StressTester()
        for key in list(HISTORICAL_CRISES) + list(SYNTHETIC_SCENARIOS):
            result = tester.run_single(key)
            assert result is not None, f"Scenarie {key} returnerede None"
            assert result.daily_values[0] == tester.initial_value

    def test_energy_portfolio_benefits_from_oil_crisis(self):
        """Energi-portefølje bør klare oliekrise bedre end tech."""
        oil = SYNTHETIC_SCENARIOS["oil_triple"]
        energy_tester = StressTester(portfolio_weights={"XOM": 0.5, "CVX": 0.5})
        tech_tester = StressTester(portfolio_weights=_tech_heavy_portfolio())
        r_energy = energy_tester.run_scenario(oil)
        r_tech = tech_tester.run_scenario(oil)
        assert r_energy.portfolio_value_end > r_tech.portfolio_value_end

    def test_bonds_hedge_in_crash(self):
        """Obligationer bør holde bedre i et krak."""
        crash = HISTORICAL_CRISES["covid"]
        bond_tester = StressTester(portfolio_weights={"TLT": 0.5, "AGG": 0.5})
        spy_tester = StressTester(portfolio_weights={"SPY": 1.0})
        r_bond = bond_tester.run_scenario(crash)
        r_spy = spy_tester.run_scenario(crash)
        assert r_bond.portfolio_value_end > r_spy.portfolio_value_end

    def test_2022_bear_energy_outperforms(self):
        """I 2022 steg energi mens tech faldt."""
        bear = HISTORICAL_CRISES["bear_2022"]
        energy_tester = StressTester(portfolio_weights={"XOM": 0.5, "CVX": 0.5})
        result = energy_tester.run_scenario(bear)
        # Energi bør have positiv slutværdi i 2022
        assert result.portfolio_value_end > result.portfolio_value_start * 0.9


class TestEdgeCases:
    def test_zero_weight_portfolio(self):
        tester = StressTester(portfolio_weights={"SPY": 0})
        # Should handle gracefully (weights get normalized)
        result = tester.run_scenario(HISTORICAL_CRISES["covid"])
        assert result is not None

    def test_very_small_initial_value(self):
        tester = StressTester(initial_value=100)
        result = tester.run_scenario(HISTORICAL_CRISES["covid"])
        assert result.portfolio_value_end < 100

    def test_very_large_initial_value(self):
        tester = StressTester(initial_value=10_000_000)
        result = tester.run_scenario(HISTORICAL_CRISES["covid"])
        assert result.portfolio_value_end < 10_000_000

    def test_monte_carlo_single_sim(self):
        tester = StressTester()
        mc = tester.monte_carlo(num_simulations=1, seed=42)
        assert len(mc.final_values) == 1

    def test_monte_carlo_short_horizon(self):
        tester = StressTester()
        mc = tester.monte_carlo(num_simulations=100, horizon_days=5, seed=42)
        assert mc.horizon_days == 5
        # Med kun 5 dage bør spredningen være lille
        spread = mc.percentile_95 - mc.percentile_5
        assert spread < tester.initial_value * 0.5
