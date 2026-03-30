"""
Tests for MarketScanner – scoring, sektor-rotation, makro, alerts, allokering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.market_scanner import (
    MarketScanner,
    ScoredAsset,
    SectorPerformance,
    MacroSnapshot,
    MarketAlert,
    AllocationRecommendation,
    ScanResult,
    SECTOR_ETF_MAP,
    VIX_SYMBOL,
    DXY_SYMBOL,
    GOLD_SYMBOL,
    OIL_SYMBOL,
    SP500_SYMBOL,
    YIELD_2Y,
    YIELD_10Y,
    _safe_pct_change,
    _clamp,
)
from src.data.indicators import add_all_indicators


# ── Hjælpere ──────────────────────────────────────────────────


def _make_df(
    close_prices: list[float],
    volumes: list[int] | None = None,
    with_indicators: bool = True,
) -> pd.DataFrame:
    """Byg en OHLCV DataFrame fra lukkekurser."""
    n = len(close_prices)
    dates = pd.date_range(end="2026-03-15", periods=n, freq="D")
    close = pd.Series(close_prices, index=dates, dtype=float)

    df = pd.DataFrame({
        "Open": close * 0.99,
        "High": close * 1.02,
        "Low": close * 0.98,
        "Close": close,
        "Volume": volumes or [1_000_000] * n,
    }, index=dates)

    if with_indicators and n >= 30:
        df = add_all_indicators(df)

    return df


def _uptrend(n: int = 250, start: float = 100.0) -> list[float]:
    """Generer en klar stigende prisrække med lille støj."""
    rng = np.random.default_rng(42)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + 0.003 + rng.normal(0, 0.005)))
    return prices


def _downtrend(n: int = 250, start: float = 100.0) -> list[float]:
    """Generer en klar faldende prisrække."""
    rng = np.random.default_rng(42)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 - 0.003 + rng.normal(0, 0.005)))
    return prices


def _flat(n: int = 250, base: float = 100.0) -> list[float]:
    """Generer en sidelæns prisrække."""
    rng = np.random.default_rng(42)
    return [base + rng.normal(0, 0.5) for _ in range(n)]


# ══════════════════════════════════════════════════════════════
#  _safe_pct_change og _clamp
# ══════════════════════════════════════════════════════════════


class TestHelpers:
    def test_pct_change_basic(self):
        s = pd.Series([100.0, 110.0])
        assert abs(_safe_pct_change(s, 1) - 10.0) < 0.01

    def test_pct_change_too_short(self):
        s = pd.Series([100.0])
        assert _safe_pct_change(s, 5) == 0.0

    def test_pct_change_zero(self):
        s = pd.Series([0.0, 100.0])
        assert _safe_pct_change(s, 1) == 0.0

    def test_clamp(self):
        assert _clamp(150) == 100
        assert _clamp(-10) == 0
        assert _clamp(50) == 50


# ══════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════


class TestScoring:
    def setup_method(self):
        self.scanner = MarketScanner()

    def test_uptrend_gets_high_score(self):
        df = _make_df(_uptrend(250))
        asset = self.scanner.score_asset("TEST", df)
        assert asset.score > 55
        assert asset.signal in ("BUY", "HOLD")

    def test_downtrend_gets_low_score(self):
        df = _make_df(_downtrend(250))
        asset = self.scanner.score_asset("TEST", df)
        assert asset.score < 45
        assert asset.signal in ("SELL", "HOLD")

    def test_flat_gets_neutral_score(self):
        df = _make_df(_flat(250))
        asset = self.scanner.score_asset("TEST", df)
        assert 25 < asset.score < 75
        assert asset.signal == "HOLD"

    def test_score_in_range(self):
        df = _make_df(_uptrend(250))
        asset = self.scanner.score_asset("TEST", df)
        assert 0 <= asset.score <= 100

    def test_empty_df_returns_hold(self):
        asset = self.scanner.score_asset("TEST", pd.DataFrame())
        assert asset.signal == "HOLD"
        assert asset.score == 50.0

    def test_short_df_returns_hold(self):
        df = _make_df([100, 101, 102], with_indicators=False)
        asset = self.scanner.score_asset("TEST", df)
        assert asset.signal == "HOLD"

    def test_sub_scores_populated(self):
        df = _make_df(_uptrend(250))
        asset = self.scanner.score_asset("TEST", df)
        assert asset.momentum_score >= 0
        assert asset.trend_score >= 0
        assert asset.rsi_score >= 0
        assert asset.macd_score >= 0

    def test_relative_strength_vs_benchmark(self):
        # Asset med stærkere afkast end benchmark
        strong = _make_df(_uptrend(250, start=100))
        weak_bench = _make_df(_downtrend(250, start=100))

        asset = self.scanner.score_asset("TEST", strong, benchmark=weak_bench)
        assert asset.relative_strength > 50

    def test_volume_anomaly_detected(self):
        prices = _uptrend(250)
        volumes = [1_000_000] * 249 + [5_000_000]  # spike sidste dag
        df = _make_df(prices, volumes=volumes)
        asset = self.scanner.score_asset("TEST", df)
        assert asset.volume_anomaly > 1.5

    def test_change_pct_populated(self):
        df = _make_df(_uptrend(250))
        asset = self.scanner.score_asset("TEST", df)
        assert isinstance(asset.change_pct, float)

    def test_reasons_populated(self):
        df = _make_df(_uptrend(250))
        asset = self.scanner.score_asset("TEST", df)
        # Uptrend burde generere mindst én årsag
        assert isinstance(asset.reasons, list)


class TestScoreUniverse:
    def test_score_multiple(self):
        scanner = MarketScanner()
        data = {
            "UP": _make_df(_uptrend(250)),
            "DOWN": _make_df(_downtrend(250)),
            "FLAT": _make_df(_flat(250)),
        }
        scored = scanner.score_universe(data)
        assert len(scored) == 3
        # Sorted by score descending
        assert scored[0].score >= scored[1].score >= scored[2].score

    def test_empty_data_skipped(self):
        scanner = MarketScanner()
        data = {
            "UP": _make_df(_uptrend(250)),
            "EMPTY": pd.DataFrame(),
        }
        scored = scanner.score_universe(data)
        assert len(scored) == 1

    def test_top_picks(self):
        scanner = MarketScanner(top_n=2)
        data = {
            f"SYM{i}": _make_df(_uptrend(250, start=100 + i * 10))
            for i in range(5)
        }
        scored = scanner.score_universe(data)
        buys, sells = scanner.get_top_picks(scored)
        assert len(buys) <= 2
        assert len(sells) <= 2


# ══════════════════════════════════════════════════════════════
#  Sektor-rotation
# ══════════════════════════════════════════════════════════════


class TestSectorRotation:
    def setup_method(self):
        self.scanner = MarketScanner()

    def test_analyze_sectors_basic(self):
        sector_data = {}
        for etf in SECTOR_ETF_MAP:
            sector_data[etf] = _make_df(_uptrend(250), with_indicators=True)

        sectors = self.scanner.analyze_sectors(sector_data)
        assert len(sectors) == len(SECTOR_ETF_MAP)
        assert all(isinstance(s, SectorPerformance) for s in sectors)

    def test_sector_names(self):
        sector_data = {"XLK": _make_df(_uptrend(250))}
        sectors = self.scanner.analyze_sectors(sector_data)
        xlk = [s for s in sectors if s.etf_symbol == "XLK"][0]
        assert xlk.name == "Teknologi"

    def test_trend_detection(self):
        # Stærk optrend
        up_data = {"XLE": _make_df(_uptrend(250))}
        sectors = self.scanner.analyze_sectors(up_data)
        xle = [s for s in sectors if s.etf_symbol == "XLE"][0]
        # Kan være up eller neutral afhængigt af volatilitet
        assert xle.trend in ("up", "neutral")

    def test_relative_strength_vs_benchmark(self):
        bench = _make_df(_downtrend(250))
        sector_data = {"XLK": _make_df(_uptrend(250))}
        sectors = self.scanner.analyze_sectors(sector_data, benchmark=bench)
        xlk = [s for s in sectors if s.etf_symbol == "XLK"][0]
        # Outperforming declining benchmark
        assert xlk.relative_strength_1m > 0

    def test_missing_sector_data(self):
        # Kun ét symbol leveret, resten mangler
        sectors = self.scanner.analyze_sectors({"XLK": _make_df(_uptrend(250))})
        assert len(sectors) == len(SECTOR_ETF_MAP)
        # De manglende har defaults
        empty = [s for s in sectors if s.etf_symbol != "XLK"]
        assert all(s.change_1d == 0 for s in empty)

    def test_rotation_advice(self):
        # Skab stærke og svage sektorer
        sectors = [
            SectorPerformance(
                etf_symbol="XLE", name="Energi",
                change_1m=8.0, relative_strength_1m=5.0, trend="up",
            ),
            SectorPerformance(
                etf_symbol="XLK", name="Teknologi",
                change_1m=-5.0, relative_strength_1m=-4.0, trend="down",
            ),
        ]
        advice = self.scanner.sector_rotation_advice(sectors)
        assert len(advice) > 0
        assert any("Energi" in a for a in advice)


# ══════════════════════════════════════════════════════════════
#  Makro-dashboard
# ══════════════════════════════════════════════════════════════


class TestMacro:
    def setup_method(self):
        self.scanner = MarketScanner()

    def _make_macro_data(
        self,
        vix: float = 20,
        dxy: float = 105,
        gold: float = 2000,
        oil: float = 75,
        sp500: float = 5000,
        yield_2y: float = 4.5,
        yield_10y: float = 4.0,
    ) -> dict[str, pd.DataFrame]:
        """Byg macro-data med specifikke slutværdier."""
        n = 100
        return {
            VIX_SYMBOL: _make_df([vix] * n, with_indicators=False),
            DXY_SYMBOL: _make_df([dxy] * n, with_indicators=False),
            GOLD_SYMBOL: _make_df([gold] * n, with_indicators=False),
            OIL_SYMBOL: _make_df([oil] * n, with_indicators=False),
            SP500_SYMBOL: _make_df([sp500] * n, with_indicators=False),
            YIELD_2Y: _make_df([yield_2y] * n, with_indicators=False),
            YIELD_10Y: _make_df([yield_10y] * n, with_indicators=False),
        }

    def test_basic_snapshot(self):
        data = self._make_macro_data()
        snap = self.scanner.macro_snapshot(data)
        assert isinstance(snap, MacroSnapshot)
        assert snap.vix == pytest.approx(20, abs=0.01)
        assert snap.timestamp != ""

    def test_vix_levels(self):
        for vix, expected in [(12, "low"), (17, "normal"), (22, "elevated"),
                               (30, "high"), (40, "extreme")]:
            data = self._make_macro_data(vix=vix)
            snap = self.scanner.macro_snapshot(data)
            assert snap.vix_level == expected, f"VIX={vix} should be {expected}"

    def test_yield_curve_inverted(self):
        data = self._make_macro_data(yield_2y=5.0, yield_10y=4.0)
        snap = self.scanner.macro_snapshot(data)
        assert snap.yield_curve_status == "inverted"
        assert snap.yield_spread < 0

    def test_yield_curve_normal(self):
        data = self._make_macro_data(yield_2y=3.0, yield_10y=4.5)
        snap = self.scanner.macro_snapshot(data)
        assert snap.yield_curve_status == "normal"

    def test_yield_curve_flat(self):
        data = self._make_macro_data(yield_2y=4.0, yield_10y=4.1)
        snap = self.scanner.macro_snapshot(data)
        assert snap.yield_curve_status == "flat"

    def test_dxy_trend(self):
        # Stabil dollar
        data = self._make_macro_data(dxy=105)
        snap = self.scanner.macro_snapshot(data)
        assert snap.dxy_trend == "neutral"

    def test_correlations_computed(self):
        data = self._make_macro_data()
        snap = self.scanner.macro_snapshot(data)
        # Korrelationer kan være tomme for flat data, men strukturen skal eksistere
        assert isinstance(snap.correlations, dict)

    def test_empty_macro_data(self):
        snap = self.scanner.macro_snapshot({})
        assert snap.vix == 0.0
        assert snap.yield_curve_status == "normal"  # default

    def test_partial_macro_data(self):
        data = {VIX_SYMBOL: _make_df([25] * 100, with_indicators=False)}
        snap = self.scanner.macro_snapshot(data)
        assert snap.vix == pytest.approx(25, abs=0.01)
        assert snap.gold_price == 0.0  # manglende


# ══════════════════════════════════════════════════════════════
#  Alerts
# ══════════════════════════════════════════════════════════════


class TestAlerts:
    def setup_method(self):
        self.scanner = MarketScanner()

    def test_flight_to_safety_alert(self):
        macro = MacroSnapshot(
            timestamp="",
            gold_change_1m=5.0,
            sp500_change_1m=-3.0,
        )
        alerts = self.scanner.generate_alerts([], [], macro)
        fts = [a for a in alerts if a.category == "flight_to_safety"]
        assert len(fts) == 1
        assert fts[0].severity == "HIGH"

    def test_yield_curve_alert(self):
        macro = MacroSnapshot(
            timestamp="",
            yield_curve_status="inverted",
            yield_2y=5.0, yield_10y=4.0, yield_spread=-1.0,
        )
        alerts = self.scanner.generate_alerts([], [], macro)
        yc = [a for a in alerts if a.category == "yield_curve"]
        assert len(yc) == 1

    def test_vix_alert(self):
        macro = MacroSnapshot(
            timestamp="",
            vix=35, vix_change=10, vix_level="high",
        )
        alerts = self.scanner.generate_alerts([], [], macro)
        vol = [a for a in alerts if a.category == "volatility"]
        assert len(vol) == 1

    def test_sector_breakout_alert(self):
        sectors = [
            SectorPerformance(
                etf_symbol="XLE", name="Energi",
                change_1w=4.0, above_sma50=True,
            ),
        ]
        alerts = self.scanner.generate_alerts([], sectors, MacroSnapshot(timestamp=""))
        bo = [a for a in alerts if a.category == "breakout"]
        assert len(bo) == 1
        assert "Energi" in bo[0].title

    def test_volume_spike_alert(self):
        scored = [
            ScoredAsset(
                symbol="NVDA", score=75, signal="BUY",
                volume_anomaly=4.0, change_pct=5.0,
            ),
        ]
        alerts = self.scanner.generate_alerts(scored, [], MacroSnapshot(timestamp=""))
        vs = [a for a in alerts if a.category == "volume_spike"]
        assert len(vs) == 1
        assert "NVDA" in vs[0].title

    def test_no_alerts_when_calm(self):
        macro = MacroSnapshot(
            timestamp="",
            vix=17, vix_level="normal",
            yield_curve_status="normal",
            gold_change_1m=1.0, sp500_change_1m=2.0,
            dxy_change=0.5, oil_change_1m=3.0,
        )
        alerts = self.scanner.generate_alerts([], [], macro)
        assert len(alerts) == 0

    def test_alerts_sorted_by_severity(self):
        macro = MacroSnapshot(
            timestamp="",
            vix=35, vix_level="high", vix_change=5,
            gold_change_1m=6, sp500_change_1m=-4,
            dxy_change=3.0,  # triggers medium
        )
        alerts = self.scanner.generate_alerts([], [], macro)
        if len(alerts) >= 2:
            # HIGH should come first
            sev_order = [a.severity for a in alerts]
            assert sev_order == sorted(
                sev_order, key=lambda s: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[s]
            )

    def test_oil_alert(self):
        macro = MacroSnapshot(
            timestamp="",
            oil_change_1m=15.0, oil_price=90.0,
        )
        alerts = self.scanner.generate_alerts([], [], macro)
        oil = [a for a in alerts if a.category == "commodities"]
        assert len(oil) == 1

    def test_dollar_alert(self):
        macro = MacroSnapshot(
            timestamp="",
            dxy_change=3.0, dxy=108.0,
        )
        alerts = self.scanner.generate_alerts([], [], macro)
        fx = [a for a in alerts if a.category == "fx"]
        assert len(fx) == 1


# ══════════════════════════════════════════════════════════════
#  Allokering
# ══════════════════════════════════════════════════════════════


class TestAllocation:
    def setup_method(self):
        self.scanner = MarketScanner()

    def test_base_allocation_sums_to_100(self):
        macro = MacroSnapshot(timestamp="", vix_level="normal",
                               yield_curve_status="normal")
        alloc = self.scanner.recommend_allocation(macro, [])
        total = alloc.stocks_pct + alloc.bonds_pct + alloc.commodities_pct + \
            alloc.crypto_pct + alloc.cash_pct
        assert abs(total - 100) < 1.0

    def test_high_vix_reduces_stocks(self):
        calm = MacroSnapshot(timestamp="", vix=15, vix_level="low",
                              yield_curve_status="normal")
        scared = MacroSnapshot(timestamp="", vix=35, vix_level="high",
                                yield_curve_status="normal")

        alloc_calm = self.scanner.recommend_allocation(calm, [])
        alloc_scared = self.scanner.recommend_allocation(scared, [])

        assert alloc_scared.stocks_pct < alloc_calm.stocks_pct

    def test_inverted_yield_reduces_stocks(self):
        normal = MacroSnapshot(timestamp="", vix_level="normal",
                                yield_curve_status="normal")
        inverted = MacroSnapshot(timestamp="", vix_level="normal",
                                  yield_curve_status="inverted")

        alloc_n = self.scanner.recommend_allocation(normal, [])
        alloc_i = self.scanner.recommend_allocation(inverted, [])

        assert alloc_i.stocks_pct < alloc_n.stocks_pct
        assert alloc_i.cash_pct > alloc_n.cash_pct

    def test_gold_surge_increases_commodities(self):
        calm = MacroSnapshot(timestamp="", vix_level="normal",
                              yield_curve_status="normal", gold_change_1m=1.0)
        gold_up = MacroSnapshot(timestamp="", vix_level="normal",
                                 yield_curve_status="normal", gold_change_1m=8.0)

        alloc_calm = self.scanner.recommend_allocation(calm, [])
        alloc_gold = self.scanner.recommend_allocation(gold_up, [])

        assert alloc_gold.commodities_pct > alloc_calm.commodities_pct

    def test_rebalance_actions_generated(self):
        macro = MacroSnapshot(timestamp="", vix_level="normal",
                               yield_curve_status="normal")
        current = {"stocks": 80, "bonds": 10, "commodities": 5, "crypto": 5}
        alloc = self.scanner.recommend_allocation(macro, [], current_allocation=current)
        # 80% stocks vs anbefalet ~60% → skal sælge
        assert len(alloc.rebalance_actions) > 0

    def test_rationale_populated(self):
        macro = MacroSnapshot(timestamp="", vix=35, vix_level="high",
                               yield_curve_status="inverted")
        alloc = self.scanner.recommend_allocation(macro, [])
        assert len(alloc.rationale) > 0

    def test_sector_weights_populated(self):
        sectors = [
            SectorPerformance(etf_symbol="XLE", name="Energi", trend="up", change_1m=5),
            SectorPerformance(etf_symbol="XLK", name="Teknologi", trend="down", change_1m=-3),
        ]
        macro = MacroSnapshot(timestamp="", vix_level="normal",
                               yield_curve_status="normal")
        alloc = self.scanner.recommend_allocation(macro, sectors)
        assert "Energi" in alloc.sector_weights
        assert alloc.sector_weights["Energi"] > alloc.sector_weights["Teknologi"]


# ══════════════════════════════════════════════════════════════
#  Full Scan
# ══════════════════════════════════════════════════════════════


class TestFullScan:
    def test_full_scan_returns_result(self):
        scanner = MarketScanner(top_n=3)

        asset_data = {
            "AAPL": _make_df(_uptrend(250)),
            "TSLA": _make_df(_downtrend(250)),
            "MSFT": _make_df(_flat(250)),
        }
        sector_data = {
            "XLK": _make_df(_uptrend(250)),
        }
        macro_data = {
            VIX_SYMBOL: _make_df([20] * 100, with_indicators=False),
            GOLD_SYMBOL: _make_df([2000] * 100, with_indicators=False),
            OIL_SYMBOL: _make_df([75] * 100, with_indicators=False),
            SP500_SYMBOL: _make_df([5000] * 100, with_indicators=False),
        }
        bench = _make_df(_flat(250))

        result = scanner.full_scan(asset_data, sector_data, macro_data, benchmark=bench)

        assert isinstance(result, ScanResult)
        assert len(result.all_scored) == 3
        assert result.scan_duration_ms > 0
        assert result.timestamp != ""

    def test_full_scan_empty_data(self):
        scanner = MarketScanner()
        result = scanner.full_scan({}, {}, {})
        assert len(result.all_scored) == 0
        assert len(result.top_buys) == 0

    def test_full_scan_top_picks(self):
        scanner = MarketScanner(top_n=2)
        data = {}
        for i in range(10):
            data[f"SYM{i}"] = _make_df(_uptrend(250, start=50 + i * 20))

        result = scanner.full_scan(data, {}, {})
        assert len(result.top_buys) <= 2
        assert len(result.top_sells) <= 2


# ══════════════════════════════════════════════════════════════
#  CLI-udskrift
# ══════════════════════════════════════════════════════════════


class TestPrint:
    def test_print_scan_result(self, capsys):
        scanner = MarketScanner()
        result = ScanResult(
            timestamp="2026-03-16T10:00:00",
            top_buys=[ScoredAsset(symbol="AAPL", score=80, signal="BUY",
                                   change_pct=2.1, reasons=["Stærkt momentum"])],
            top_sells=[ScoredAsset(symbol="BAD", score=20, signal="SELL",
                                    change_pct=-3.0, reasons=["Svagt momentum"])],
            all_scored=[],
            sector_performance=[
                SectorPerformance(etf_symbol="XLK", name="Teknologi",
                                   change_1d=0.5, change_1w=2.0, change_1m=5.0,
                                   change_3m=8.0, relative_strength_1m=2.0, trend="up"),
            ],
            macro=MacroSnapshot(
                timestamp="", vix=20, vix_level="normal", vix_change=-0.5,
                dxy=105, dxy_change=0.3, dxy_trend="neutral",
                gold_price=2000, gold_change_1m=2.0,
                oil_price=75, oil_change_1m=-1.0,
                yield_2y=4.0, yield_10y=4.5, yield_spread=0.5,
                yield_curve_status="normal",
            ),
            alerts=[],
            allocation=AllocationRecommendation(
                stocks_pct=60, bonds_pct=25, commodities_pct=10,
                crypto_pct=5, cash_pct=0, rationale="Standard",
            ),
        )
        scanner.print_scan_result(result)
        captured = capsys.readouterr()
        assert "MARKEDSSCANNING" in captured.out
        assert "AAPL" in captured.out
        assert "Teknologi" in captured.out

    def test_print_macro(self, capsys):
        scanner = MarketScanner()
        macro = MacroSnapshot(
            timestamp="", vix=25, vix_level="elevated", vix_change=1.5,
            dxy=105, dxy_change=0.3, dxy_trend="neutral",
            gold_price=2000, gold_change_1m=3.0,
            oil_price=80, oil_change_1m=5.0,
            yield_2y=4.0, yield_10y=4.3, yield_spread=0.3,
            yield_curve_status="normal", sp500_change_1m=2.0,
        )
        scanner.print_macro(macro)
        captured = capsys.readouterr()
        assert "MAKRO-DASHBOARD" in captured.out
        assert "VIX" in captured.out
        assert "Yield Curve" in captured.out


# ══════════════════════════════════════════════════════════════
#  Dataklasser
# ══════════════════════════════════════════════════════════════


class TestDataclasses:
    def test_scored_asset_defaults(self):
        a = ScoredAsset(symbol="TEST", score=50, signal="HOLD")
        assert a.momentum_score == 0.0
        assert a.reasons == []

    def test_sector_performance_defaults(self):
        s = SectorPerformance(etf_symbol="XLK", name="Tech")
        assert s.trend == "neutral"
        assert s.above_sma50 is False

    def test_macro_snapshot_defaults(self):
        m = MacroSnapshot(timestamp="now")
        assert m.vix == 0.0
        assert m.yield_curve_status == "normal"
        assert m.correlations == {}

    def test_market_alert(self):
        a = MarketAlert(
            severity="HIGH", category="test",
            title="Test", message="msg",
        )
        assert a.symbols == []

    def test_allocation_defaults(self):
        a = AllocationRecommendation()
        assert a.stocks_pct == 0.0
        assert a.rebalance_actions == []
