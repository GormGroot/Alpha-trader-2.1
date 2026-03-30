"""
Tests for MLStrategy – ML-baseret handelsstrategi.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.base_strategy import Signal, StrategyResult
from src.strategy.ml_strategy import (
    MLStrategy,
    MLMetrics,
    BacktestComparison,
    build_features,
    build_target,
    FEATURE_COLUMNS,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_df(n: int = 800, trend: float = 0.0003, noise: float = 0.015, seed: int = 42) -> pd.DataFrame:
    """
    Generér syntetisk OHLCV-data.

    Args:
        n: Antal datapunkter (800 ≈ 3+ år handelsdage).
        trend: Daglig drift (positiv = opadgående).
        noise: Daglig standardafvigelse.
        seed: Random seed for reproducerbarhed.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range(end="2026-03-15", periods=n, freq="D")

    # Geometrisk random walk med drift
    log_returns = trend + noise * rng.randn(n)
    log_returns[0] = 0
    prices = 100 * np.exp(np.cumsum(log_returns))

    # OHLCV
    high = prices * (1 + rng.uniform(0, 0.02, n))
    low = prices * (1 - rng.uniform(0, 0.02, n))
    volume = rng.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame({
        "Open": prices * (1 + rng.uniform(-0.005, 0.005, n)),
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    }, index=dates)


def _make_uptrend(n: int = 800) -> pd.DataFrame:
    return _make_df(n=n, trend=0.001, noise=0.01, seed=42)


def _make_downtrend(n: int = 800) -> pd.DataFrame:
    return _make_df(n=n, trend=-0.001, noise=0.01, seed=42)


def _make_sideways(n: int = 800) -> pd.DataFrame:
    return _make_df(n=n, trend=0.0, noise=0.01, seed=42)


# ── Test Feature Engineering ─────────────────────────────────

class TestBuildFeatures:
    def test_returns_all_feature_columns(self):
        df = _make_df()
        feat = build_features(df)
        for col in FEATURE_COLUMNS:
            assert col in feat.columns, f"Mangler kolonne: {col}"

    def test_feature_count(self):
        assert len(FEATURE_COLUMNS) == 16

    def test_sma_pct_around_zero(self):
        df = _make_sideways()
        feat = build_features(df)
        # SMA_20_pct bør være tæt på 0 for sideways-data
        sma_pct = feat["SMA_20_pct"].dropna()
        assert abs(sma_pct.mean()) < 0.05

    def test_bb_position_bounded(self):
        df = _make_df()
        feat = build_features(df)
        bb_pos = feat["BB_position"].dropna()
        # De fleste bør være mellem 0 og 1
        assert bb_pos.median() > 0.0
        assert bb_pos.median() < 1.0

    def test_rsi_present(self):
        df = _make_df()
        feat = build_features(df)
        rsi = feat["RSI"].dropna()
        assert len(rsi) > 0
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_return_columns(self):
        df = _make_df()
        feat = build_features(df)
        assert not feat["return_1d"].dropna().empty
        assert not feat["return_5d"].dropna().empty
        assert not feat["return_20d"].dropna().empty

    def test_volatility_positive(self):
        df = _make_df()
        feat = build_features(df)
        vol = feat["volatility_20d"].dropna()
        assert (vol >= 0).all()

    def test_idempotent(self):
        """Kald build_features to gange giver samme resultat."""
        df = _make_df()
        feat1 = build_features(df)
        feat2 = build_features(df)
        for col in FEATURE_COLUMNS:
            pd.testing.assert_series_equal(
                feat1[col].dropna(), feat2[col].dropna(),
                check_names=False,
            )


class TestBuildTarget:
    def test_binary_values(self):
        df = _make_df()
        target = build_target(df)
        valid = target.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_last_values_nan(self):
        df = _make_df()
        target = build_target(df, horizon=1)
        assert pd.isna(target.iloc[-1])

    def test_horizon_shifts_nans(self):
        df = _make_df()
        t1 = build_target(df, horizon=1)
        t5 = build_target(df, horizon=5)
        # Horizon 5 har 5 NaN'er i slutningen
        assert t5.isna().sum() > t1.isna().sum()

    def test_threshold_reduces_positives(self):
        df = _make_df()
        t_zero = build_target(df, threshold=0.0)
        t_high = build_target(df, threshold=0.01)  # Kræver 1% stigning
        assert t_high.dropna().sum() <= t_zero.dropna().sum()

    def test_uptrend_more_positives(self):
        df = _make_uptrend()
        target = build_target(df)
        valid = target.dropna()
        assert valid.mean() > 0.5  # Flere op-dage i uptrend


# ── Test MLMetrics ───────────────────────────────────────────

class TestMLMetrics:
    def test_repr(self):
        m = MLMetrics(accuracy=0.55, precision=0.60, recall=0.50, f1=0.545, auc_roc=0.58)
        s = repr(m)
        assert "55.0%" in s
        assert "0.580" in s

    def test_default_values(self):
        m = MLMetrics()
        assert m.accuracy == 0.0
        assert m.feature_importance == {}


# ── Test MLStrategy Training ─────────────────────────────────

class TestMLTraining:
    def test_train_returns_metrics(self):
        df = _make_df(800)
        ml = MLStrategy()
        metrics = ml.train(df)
        assert isinstance(metrics, MLMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.auc_roc <= 1
        assert metrics.n_train > 0
        assert metrics.n_test > 0

    def test_is_trained_flag(self):
        ml = MLStrategy()
        assert ml.is_trained is False
        ml.train(_make_df(800))
        assert ml.is_trained is True

    def test_feature_importance_has_all_features(self):
        ml = MLStrategy()
        metrics = ml.train(_make_df(800))
        assert len(metrics.feature_importance) == len(FEATURE_COLUMNS)
        for feat in FEATURE_COLUMNS:
            assert feat in metrics.feature_importance

    def test_metrics_stored(self):
        ml = MLStrategy()
        ml.train(_make_df(800))
        assert ml.metrics is not None
        assert ml.metrics.train_period != ""
        assert ml.metrics.test_period != ""

    def test_too_little_data_raises(self):
        ml = MLStrategy()
        with pytest.raises(ValueError, match="For lidt data"):
            ml.train(_make_df(50))

    def test_custom_hyperparameters(self):
        ml = MLStrategy(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
        )
        metrics = ml.train(_make_df(600))
        assert isinstance(metrics, MLMetrics)
        assert metrics.n_train > 0

    def test_different_horizon(self):
        ml = MLStrategy(horizon=5)
        metrics = ml.train(_make_df(800))
        assert isinstance(metrics, MLMetrics)

    def test_with_threshold(self):
        ml = MLStrategy(threshold=0.005)
        metrics = ml.train(_make_df(800))
        assert metrics.accuracy > 0


# ── Test MLStrategy Analyze ──────────────────────────────────

class TestMLAnalyze:
    def test_untrained_returns_hold(self):
        ml = MLStrategy()
        result = ml.analyze(_make_df(300))
        assert result.signal == Signal.HOLD
        assert "ikke trænet" in result.reason.lower()

    def test_too_little_data_returns_hold(self):
        ml = MLStrategy()
        ml.train(_make_df(800))
        result = ml.analyze(_make_df(100))
        assert result.signal == Signal.HOLD

    def test_returns_strategy_result(self):
        ml = MLStrategy()
        ml.train(_make_df(800))
        result = ml.analyze(_make_df(400))
        assert isinstance(result, StrategyResult)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_confidence_range(self):
        ml = MLStrategy(confidence_min=0.50)  # Lavere grænse → flere signaler
        ml.train(_make_df(800))
        result = ml.analyze(_make_df(400))
        assert 0 <= result.confidence <= 100

    def test_analyze_returns_valid_signal(self):
        """Træn og analysér → altid et gyldigt signal med korrekt confidence."""
        ml = MLStrategy(confidence_min=0.50)
        up = _make_uptrend(800)
        ml.train(up)
        result = ml.analyze(up)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        if result.signal != Signal.HOLD:
            assert result.confidence >= 50

    def test_reason_contains_ml(self):
        ml = MLStrategy(confidence_min=0.50)
        ml.train(_make_df(800))
        result = ml.analyze(_make_df(400))
        if result.signal != Signal.HOLD or result.confidence == 0:
            assert "ML" in result.reason or "confidence" in result.reason.lower()

    def test_name_property(self):
        ml = MLStrategy(horizon=3, confidence_min=0.60)
        assert "ML" in ml.name
        assert "h=3" in ml.name
        assert "60%" in ml.name


# ── Test Scale Confidence ────────────────────────────────────

class TestScaleConfidence:
    def test_min_probability_gives_50(self):
        ml = MLStrategy(confidence_min=0.55)
        assert ml._scale_confidence(0.55) == pytest.approx(50.0)

    def test_max_probability_gives_95(self):
        ml = MLStrategy(confidence_min=0.55)
        assert ml._scale_confidence(1.0) == pytest.approx(95.0)

    def test_mid_probability(self):
        ml = MLStrategy(confidence_min=0.50)
        # 0.75 er halvvejs mellem 0.50 og 1.0
        assert ml._scale_confidence(0.75) == pytest.approx(72.5)

    def test_monotonically_increasing(self):
        ml = MLStrategy(confidence_min=0.55)
        prev = 0
        for p in [0.55, 0.60, 0.70, 0.80, 0.90, 1.0]:
            c = ml._scale_confidence(p)
            assert c > prev
            prev = c


# ── Test Evaluate (Backtest) ─────────────────────────────────

class TestEvaluate:
    def test_evaluate_returns_comparison(self):
        ml = MLStrategy()
        df = _make_df(800)
        ml.train(df)
        comp = ml.evaluate(df)
        assert isinstance(comp, BacktestComparison)
        assert comp.test_period != ""

    def test_evaluate_untrained_raises(self):
        ml = MLStrategy()
        with pytest.raises(RuntimeError, match="ikke trænet"):
            ml.evaluate(_make_df(800))

    def test_returns_are_finite(self):
        ml = MLStrategy()
        df = _make_df(800)
        ml.train(df)
        comp = ml.evaluate(df)
        assert np.isfinite(comp.ml_return)
        assert np.isfinite(comp.buy_hold_return)

    def test_win_rate_bounded(self):
        ml = MLStrategy()
        df = _make_df(800)
        ml.train(df)
        comp = ml.evaluate(df)
        assert 0 <= comp.ml_win_rate <= 1

    def test_trades_non_negative(self):
        ml = MLStrategy()
        df = _make_df(800)
        ml.train(df)
        comp = ml.evaluate(df)
        assert comp.ml_trades >= 0


# ── Test Explain ─────────────────────────────────────────────

class TestExplain:
    def test_untrained_explain(self):
        ml = MLStrategy()
        text = ml.explain()
        assert "ikke trænet" in text.lower()

    def test_trained_explain_contains_metrics(self):
        ml = MLStrategy()
        ml.train(_make_df(800))
        text = ml.explain()
        assert "Accuracy" in text
        assert "AUC-ROC" in text
        assert "FEATURE" in text
        assert "VURDERING" in text

    def test_explain_has_feature_ranking(self):
        ml = MLStrategy()
        ml.train(_make_df(800))
        text = ml.explain()
        # Mindst 5 features nævnt
        count = sum(1 for feat in FEATURE_COLUMNS if feat in text)
        assert count >= 5

    def test_print_explanation(self, capsys):
        ml = MLStrategy()
        ml.train(_make_df(800))
        ml.print_explanation()
        captured = capsys.readouterr()
        assert "ML MODEL" in captured.out


# ── Test BaseStrategy Integration ────────────────────────────

class TestBaseStrategyIntegration:
    def test_inherits_base_strategy(self):
        ml = MLStrategy()
        assert isinstance(ml, MLStrategy)
        from src.strategy.base_strategy import BaseStrategy
        assert isinstance(ml, BaseStrategy)

    def test_validate_data(self):
        ml = MLStrategy()
        df = _make_df(300)
        assert ml.validate_data(df, 200) is True
        assert ml.validate_data(df, 400) is False

    def test_get_position_size(self):
        ml = MLStrategy(confidence_min=0.50)
        ml.train(_make_uptrend(800))
        result = ml.analyze(_make_uptrend(400))
        if result.signal != Signal.HOLD:
            size = ml.get_position_size(result, 100_000)
            assert size > 0

    def test_works_in_combined_strategy(self):
        """ML kan bruges i CombinedStrategy."""
        from src.strategy.combined_strategy import CombinedStrategy
        from src.strategy.rsi_strategy import RSIStrategy

        ml = MLStrategy(confidence_min=0.50)
        ml.train(_make_df(800))

        combined = CombinedStrategy(
            strategies=[(ml, 1.0), (RSIStrategy(), 1.0)],
            min_agreement=1,
        )
        result = combined.analyze(_make_df(400))
        assert isinstance(result, StrategyResult)


# ── Test Sharpe Calc ─────────────────────────────────────────

class TestSharpeCalc:
    def test_zero_std_returns_zero(self):
        returns = np.array([0.01, 0.01, 0.01])
        assert MLStrategy._calc_sharpe(returns) == 0.0

    def test_positive_returns(self):
        rng = np.random.RandomState(42)
        returns = 0.001 + 0.01 * rng.randn(252)
        sharpe = MLStrategy._calc_sharpe(returns)
        assert sharpe > 0

    def test_empty_returns(self):
        assert MLStrategy._calc_sharpe(np.array([])) == 0.0
