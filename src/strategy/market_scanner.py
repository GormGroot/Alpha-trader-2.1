"""
MarketScanner – kryds-aktivklasse analyse med scoring, sektor-rotation,
makro-dashboard, muligheds-alerts og porteføljeanbefaling.

Modulet analyserer hele universet og producerer:
  1. Scoring (0–100) per aktiv baseret på momentum, trend, RSI, MACD,
     relativ styrke, volume-anomalier og volatilitet.
  2. Top 10 KØB- og SÆLG-kandidater per dag.
  3. Sektor-rotation analyse med rebalanceringsforslag.
  4. Makro-dashboard: yield curve, VIX, dollar, guld/aktier, olie.
  5. Muligheds-alerts (flight-to-safety, breakouts, volume-spikes osv.).
  6. Porteføljeanbefaling med diversificeret allokering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


# ── Sektor-ETF mapping ────────────────────────────────────────

SECTOR_ETF_MAP: dict[str, str] = {
    "XLK": "Teknologi",
    "XLF": "Finans",
    "XLE": "Energi",
    "XLV": "Sundhed",
    "XLI": "Industri",
    "XLP": "Dagligvarer",
    "XLY": "Diskretionært Forbrug",
    "XLRE": "Ejendomme",
    "XLB": "Materialer",
    "XLC": "Kommunikation",
    "XLU": "Forsyning",
}

# Makro-symboler
VIX_SYMBOL = "^VIX"
DXY_SYMBOL = "DX-Y.NYB"
GOLD_SYMBOL = "GC=F"
OIL_SYMBOL = "CL=F"
SP500_SYMBOL = "^GSPC"

# Yield-symboler (US Treasury)
YIELD_2Y = "^IRX"   # 13-week proxy — vi bruger den vi har
YIELD_10Y = "^TNX"

# ── Dataklasser ───────────────────────────────────────────────


@dataclass
class ScoredAsset:
    """Et aktiv med score og detaljer."""
    symbol: str
    score: float                    # 0–100
    signal: str                     # "BUY", "SELL", "HOLD"
    momentum_score: float = 0.0
    trend_score: float = 0.0
    rsi_score: float = 0.0
    macd_score: float = 0.0
    relative_strength: float = 0.0
    volume_anomaly: float = 0.0     # volume ratio (>1 = over normalt)
    volatility_rank: float = 0.0    # 0–1 (0=lav, 1=høj)
    sector: str = ""
    change_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class SectorPerformance:
    """Performance for én sektor."""
    etf_symbol: str
    name: str
    change_1d: float = 0.0
    change_1w: float = 0.0
    change_1m: float = 0.0
    change_3m: float = 0.0
    relative_strength_1m: float = 0.0  # vs. SPY
    trend: str = "neutral"              # "up", "down", "neutral"
    above_sma50: bool = False


@dataclass
class MacroSnapshot:
    """Snapshot af makro-indikatorer."""
    timestamp: str
    vix: float = 0.0
    vix_change: float = 0.0
    vix_level: str = "normal"           # "low", "normal", "elevated", "high", "extreme"
    dxy: float = 0.0
    dxy_change: float = 0.0
    dxy_trend: str = "neutral"
    gold_price: float = 0.0
    gold_change_1m: float = 0.0
    oil_price: float = 0.0
    oil_change_1m: float = 0.0
    sp500_change_1m: float = 0.0
    yield_2y: float = 0.0
    yield_10y: float = 0.0
    yield_spread: float = 0.0
    yield_curve_status: str = "normal"  # "normal", "flat", "inverted"
    correlations: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class MarketAlert:
    """En markedsalert/mulighed."""
    severity: str       # "HIGH", "MEDIUM", "LOW"
    category: str       # "flight_to_safety", "breakout", "volume_spike" osv.
    title: str
    message: str
    symbols: list[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class AllocationRecommendation:
    """Anbefalet porteføljeallokering."""
    stocks_pct: float = 0.0
    bonds_pct: float = 0.0
    commodities_pct: float = 0.0
    crypto_pct: float = 0.0
    cash_pct: float = 0.0
    rationale: str = ""
    sector_weights: dict[str, float] = field(default_factory=dict)
    rebalance_actions: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """Komplet resultat af en markedsscanning."""
    timestamp: str
    top_buys: list[ScoredAsset]
    top_sells: list[ScoredAsset]
    all_scored: list[ScoredAsset]
    sector_performance: list[SectorPerformance]
    macro: MacroSnapshot
    alerts: list[MarketAlert]
    allocation: AllocationRecommendation
    scan_duration_ms: float = 0.0


# ── Hjælpefunktioner ──────────────────────────────────────────


def _safe_pct_change(series: pd.Series, periods: int) -> float:
    """Beregn procentvis ændring sikkert (håndterer NaN/manglende data)."""
    if len(series) < periods + 1:
        return 0.0
    old = series.iloc[-(periods + 1)]
    new = series.iloc[-1]
    if old == 0 or pd.isna(old) or pd.isna(new):
        return 0.0
    return (new - old) / abs(old) * 100


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


# ── MarketScanner ─────────────────────────────────────────────


class MarketScanner:
    """
    Analyserer hele universet og producerer scoring, alerts og anbefalinger.

    Brug:
        scanner = MarketScanner()
        result = scanner.full_scan(data_dict, benchmark_df)
        for asset in result.top_buys:
            print(f"{asset.symbol}: score={asset.score:.0f}")
    """

    def __init__(
        self,
        sector_etf_map: dict[str, str] | None = None,
        top_n: int = 10,
    ) -> None:
        self._sector_map = sector_etf_map or SECTOR_ETF_MAP
        self._top_n = top_n

    # ══════════════════════════════════════════════════════════
    #  1. SCORING
    # ══════════════════════════════════════════════════════════

    def score_asset(
        self,
        symbol: str,
        df: pd.DataFrame,
        benchmark: pd.DataFrame | None = None,
        sector_df: pd.DataFrame | None = None,
    ) -> ScoredAsset:
        """
        Beregn en score (0–100) for ét aktiv.

        Vægtning:
          - Momentum (20-dag afkast + ROC)       25%
          - Trend (SMA position + retning)        20%
          - RSI (reversals + styrke)               15%
          - MACD (histogram + crossovers)          15%
          - Relativ styrke vs. benchmark           10%
          - Volume-anomali                         10%
          - Volatilitet (lav = bedre for køb)       5%
        """
        if df.empty or len(df) < 30:
            return ScoredAsset(symbol=symbol, score=50.0, signal="HOLD",
                               reasons=["Utilstrækkelig data"])

        close = df["Close"]
        reasons: list[str] = []

        # ── Momentum (25%) ──
        mom_5d = _safe_pct_change(close, 5)
        mom_20d = _safe_pct_change(close, 20)
        mom_60d = _safe_pct_change(close, 60) if len(close) > 60 else 0.0

        # Normalisér: +10% = 80 score, -10% = 20 score
        mom_norm = _clamp(50 + mom_20d * 3, 0, 100)
        # Bonus for konsistent opad over kort + lang sigt
        if mom_5d > 0 and mom_20d > 0 and mom_60d > 0:
            mom_norm = min(100, mom_norm + 10)
            reasons.append(f"Stærkt momentum: +{mom_20d:.1f}% (20d)")
        elif mom_5d < 0 and mom_20d < 0 and mom_60d < 0:
            mom_norm = max(0, mom_norm - 10)
            reasons.append(f"Svagt momentum: {mom_20d:.1f}% (20d)")
        momentum_score = mom_norm

        # ── Trend (20%) ──
        trend_score = 50.0
        if "SMA_20" in df.columns and "SMA_50" in df.columns:
            price = close.iloc[-1]
            sma20 = df["SMA_20"].iloc[-1]
            sma50 = df["SMA_50"].iloc[-1]
            sma200 = df["SMA_200"].iloc[-1] if "SMA_200" in df.columns else None

            above_20 = price > sma20 if not pd.isna(sma20) else False
            above_50 = price > sma50 if not pd.isna(sma50) else False
            above_200 = price > sma200 if sma200 is not None and not pd.isna(sma200) else False
            golden = sma20 > sma50 if (not pd.isna(sma20) and not pd.isna(sma50)) else False

            pts = 0
            if above_20:
                pts += 20
            if above_50:
                pts += 20
            if above_200:
                pts += 25
            if golden:
                pts += 15
                reasons.append("Golden cross (SMA20 > SMA50)")

            # Retning: SMA_50 stigende?
            if len(df) > 55 and "SMA_50" in df.columns:
                sma50_5d_ago = df["SMA_50"].iloc[-6]
                if not pd.isna(sma50_5d_ago) and sma50 > sma50_5d_ago:
                    pts += 10
                    reasons.append("Stigende SMA_50")
                elif not pd.isna(sma50_5d_ago) and sma50 < sma50_5d_ago:
                    pts -= 10

            trend_score = _clamp(pts + 10, 0, 100)

        # ── RSI (15%) ──
        rsi_score = 50.0
        if "RSI" in df.columns:
            rsi = df["RSI"].iloc[-1]
            if not pd.isna(rsi):
                if rsi < 30:
                    rsi_score = 85  # Oversolgt → købs-mulighed
                    reasons.append(f"RSI oversolgt ({rsi:.0f})")
                elif rsi < 40:
                    rsi_score = 70
                elif rsi < 60:
                    rsi_score = 55  # Neutral
                elif rsi < 70:
                    rsi_score = 40
                else:
                    rsi_score = 15  # Overkøbt → salgs-signal
                    reasons.append(f"RSI overkøbt ({rsi:.0f})")

        # ── MACD (15%) ──
        macd_score = 50.0
        if "MACD" in df.columns and "MACD_Signal" in df.columns and "MACD_Hist" in df.columns:
            macd = df["MACD"].iloc[-1]
            macd_sig = df["MACD_Signal"].iloc[-1]
            hist = df["MACD_Hist"].iloc[-1]

            if not any(pd.isna(x) for x in [macd, macd_sig, hist]):
                # MACD over signal = bullish
                if macd > macd_sig:
                    macd_score = 65
                    if hist > 0 and len(df) > 2:
                        prev_hist = df["MACD_Hist"].iloc[-2]
                        if not pd.isna(prev_hist) and hist > prev_hist:
                            macd_score = 80
                            reasons.append("MACD accelererer opad")
                else:
                    macd_score = 35
                    if hist < 0 and len(df) > 2:
                        prev_hist = df["MACD_Hist"].iloc[-2]
                        if not pd.isna(prev_hist) and hist < prev_hist:
                            macd_score = 20
                            reasons.append("MACD accelererer nedad")

                # Crossover detection
                if len(df) > 2:
                    prev_macd = df["MACD"].iloc[-2]
                    prev_sig = df["MACD_Signal"].iloc[-2]
                    if not pd.isna(prev_macd) and not pd.isna(prev_sig):
                        if prev_macd <= prev_sig and macd > macd_sig:
                            macd_score = 90
                            reasons.append("MACD bullish crossover")
                        elif prev_macd >= prev_sig and macd < macd_sig:
                            macd_score = 10
                            reasons.append("MACD bearish crossover")

        # ── Relativ styrke (10%) ──
        rs_score = 50.0
        if benchmark is not None and len(benchmark) >= 20 and len(close) >= 20:
            asset_ret = _safe_pct_change(close, 20)
            bench_ret = _safe_pct_change(benchmark["Close"], 20)
            rs = asset_ret - bench_ret
            rs_score = _clamp(50 + rs * 3, 0, 100)
            if rs > 5:
                reasons.append(f"Outperformer benchmark med {rs:.1f}%")
            elif rs < -5:
                reasons.append(f"Underperformer benchmark med {rs:.1f}%")

        # ── Volume-anomali (10%) ──
        vol_score = 50.0
        vol_ratio = 1.0
        if "Volume_Ratio" in df.columns:
            vr = df["Volume_Ratio"].iloc[-1]
            if not pd.isna(vr):
                vol_ratio = vr
                # Høj volumen + prisopgang = stærkt signal
                daily_change = _safe_pct_change(close, 1)
                if vr > 2.0:
                    if daily_change > 0:
                        vol_score = 85
                        reasons.append(f"Usædvanlig volumen ({vr:.1f}x) + prisopgang")
                    else:
                        vol_score = 25
                        reasons.append(f"Usædvanlig volumen ({vr:.1f}x) + prisfald")
                elif vr > 1.5:
                    vol_score = 60 if daily_change > 0 else 40

        # ── Volatilitet (5%) ──
        vol_rank = 0.5
        vola_score = 50.0
        if "BB_Width" in df.columns:
            bb_width = df["BB_Width"].iloc[-1]
            if not pd.isna(bb_width) and len(df) > 60:
                bb_hist = df["BB_Width"].dropna()
                if len(bb_hist) > 20:
                    vol_rank = float(bb_hist.rank(pct=True).iloc[-1])
                    # Lav volatilitet lidt bedre for køb (squeeze → breakout)
                    if vol_rank < 0.2:
                        vola_score = 70
                        reasons.append("Lav volatilitet (squeeze mulig)")
                    elif vol_rank > 0.8:
                        vola_score = 30

        # ── Samlet score ──
        total = (
            momentum_score * 0.25
            + trend_score * 0.20
            + rsi_score * 0.15
            + macd_score * 0.15
            + rs_score * 0.10
            + vol_score * 0.10
            + vola_score * 0.05
        )
        total = round(_clamp(total, 0, 100), 1)

        # Signal
        if total >= 65:
            signal = "BUY"
        elif total <= 35:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Sektor
        sector = ""
        if symbol in self._sector_map:
            sector = self._sector_map[symbol]

        return ScoredAsset(
            symbol=symbol,
            score=total,
            signal=signal,
            momentum_score=round(momentum_score, 1),
            trend_score=round(trend_score, 1),
            rsi_score=round(rsi_score, 1),
            macd_score=round(macd_score, 1),
            relative_strength=round(rs_score, 1),
            volume_anomaly=round(vol_ratio, 2),
            volatility_rank=round(vol_rank, 2),
            sector=sector,
            change_pct=round(_safe_pct_change(close, 1), 2),
            reasons=reasons,
        )

    def score_universe(
        self,
        data: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame | None = None,
    ) -> list[ScoredAsset]:
        """Score alle aktiver i universet."""
        scored: list[ScoredAsset] = []
        for symbol, df in data.items():
            if df.empty:
                continue
            try:
                asset = self.score_asset(symbol, df, benchmark=benchmark)
                scored.append(asset)
            except Exception as exc:
                logger.warning(f"Scoring fejl for {symbol}: {exc}")

        scored.sort(key=lambda a: a.score, reverse=True)
        return scored

    def get_top_picks(
        self,
        scored: list[ScoredAsset],
        n: int | None = None,
    ) -> tuple[list[ScoredAsset], list[ScoredAsset]]:
        """Returnér top N KØB- og SÆLG-kandidater."""
        n = n or self._top_n
        buys = [a for a in scored if a.signal == "BUY"][:n]
        sells = [a for a in reversed(scored) if a.signal == "SELL"][:n]
        return buys, sells

    # ══════════════════════════════════════════════════════════
    #  2. SEKTOR-ROTATION
    # ══════════════════════════════════════════════════════════

    def analyze_sectors(
        self,
        sector_data: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame | None = None,
    ) -> list[SectorPerformance]:
        """Analysér sektor-performance og trends."""
        results: list[SectorPerformance] = []

        for etf, name in self._sector_map.items():
            df = sector_data.get(etf)
            if df is None or df.empty or len(df) < 5:
                results.append(SectorPerformance(etf_symbol=etf, name=name))
                continue

            close = df["Close"]
            perf = SectorPerformance(
                etf_symbol=etf,
                name=name,
                change_1d=round(_safe_pct_change(close, 1), 2),
                change_1w=round(_safe_pct_change(close, 5), 2),
                change_1m=round(_safe_pct_change(close, 21), 2),
                change_3m=round(_safe_pct_change(close, 63), 2) if len(close) > 63 else 0.0,
            )

            # Relativ styrke vs. benchmark (1 måned)
            if benchmark is not None and len(benchmark) > 21:
                bench_1m = _safe_pct_change(benchmark["Close"], 21)
                perf.relative_strength_1m = round(perf.change_1m - bench_1m, 2)

            # Trend (baseret på 1-uges og 1-måneds retning)
            if perf.change_1w > 1 and perf.change_1m > 2:
                perf.trend = "up"
            elif perf.change_1w < -1 and perf.change_1m < -2:
                perf.trend = "down"
            else:
                perf.trend = "neutral"

            # Over SMA_50?
            if "SMA_50" in df.columns:
                sma50 = df["SMA_50"].iloc[-1]
                perf.above_sma50 = bool(close.iloc[-1] > sma50) if not pd.isna(sma50) else False

            results.append(perf)

        results.sort(key=lambda s: s.change_1m, reverse=True)
        return results

    def sector_rotation_advice(
        self,
        sectors: list[SectorPerformance],
    ) -> list[str]:
        """Generér sektorrotations-forslag baseret på performance."""
        advice: list[str] = []

        strong = [s for s in sectors if s.trend == "up" and s.relative_strength_1m > 2]
        weak = [s for s in sectors if s.trend == "down" and s.relative_strength_1m < -2]

        if strong:
            names = ", ".join(s.name for s in strong[:3])
            advice.append(f"Stærke sektorer: {names} — overvej øget eksponering")
        if weak:
            names = ", ".join(s.name for s in weak[:3])
            advice.append(f"Svage sektorer: {names} — overvej reduceret eksponering")

        # Specifik rotation
        if strong and weak:
            advice.append(
                f"Sektorrotation: Flyt fra {weak[0].name} "
                f"({weak[0].change_1m:+.1f}%) til {strong[0].name} "
                f"({strong[0].change_1m:+.1f}%)"
            )

        # Bredde-tjek: er opgang bred eller snæver?
        up_sectors = sum(1 for s in sectors if s.change_1m > 0)
        total = len(sectors)
        if total > 0:
            breadth = up_sectors / total
            if breadth > 0.8:
                advice.append(
                    f"Bred markedsopgang: {up_sectors}/{total} sektorer positive — "
                    f"bullish signal"
                )
            elif breadth < 0.3:
                advice.append(
                    f"Smal markedsopgang: kun {up_sectors}/{total} sektorer positive — "
                    f"defensiv positionering anbefales"
                )

        return advice

    # ══════════════════════════════════════════════════════════
    #  3. MAKRO-DASHBOARD
    # ══════════════════════════════════════════════════════════

    def macro_snapshot(
        self,
        macro_data: dict[str, pd.DataFrame],
    ) -> MacroSnapshot:
        """Byg et makro-snapshot fra markedsdata."""
        now = datetime.now().isoformat()
        snap = MacroSnapshot(timestamp=now)

        # VIX
        if VIX_SYMBOL in macro_data and not macro_data[VIX_SYMBOL].empty:
            vix_df = macro_data[VIX_SYMBOL]
            snap.vix = float(vix_df["Close"].iloc[-1])
            snap.vix_change = _safe_pct_change(vix_df["Close"], 1)
            if snap.vix < 15:
                snap.vix_level = "low"
            elif snap.vix < 20:
                snap.vix_level = "normal"
            elif snap.vix < 25:
                snap.vix_level = "elevated"
            elif snap.vix < 35:
                snap.vix_level = "high"
            else:
                snap.vix_level = "extreme"

        # DXY (Dollar Index)
        if DXY_SYMBOL in macro_data and not macro_data[DXY_SYMBOL].empty:
            dxy_df = macro_data[DXY_SYMBOL]
            snap.dxy = float(dxy_df["Close"].iloc[-1])
            snap.dxy_change = _safe_pct_change(dxy_df["Close"], 5)
            if snap.dxy_change > 1:
                snap.dxy_trend = "strengthening"
            elif snap.dxy_change < -1:
                snap.dxy_trend = "weakening"

        # Guld
        if GOLD_SYMBOL in macro_data and not macro_data[GOLD_SYMBOL].empty:
            gold_df = macro_data[GOLD_SYMBOL]
            snap.gold_price = float(gold_df["Close"].iloc[-1])
            snap.gold_change_1m = _safe_pct_change(gold_df["Close"], 21)

        # Olie
        if OIL_SYMBOL in macro_data and not macro_data[OIL_SYMBOL].empty:
            oil_df = macro_data[OIL_SYMBOL]
            snap.oil_price = float(oil_df["Close"].iloc[-1])
            snap.oil_change_1m = _safe_pct_change(oil_df["Close"], 21)

        # S&P 500
        if SP500_SYMBOL in macro_data and not macro_data[SP500_SYMBOL].empty:
            sp_df = macro_data[SP500_SYMBOL]
            snap.sp500_change_1m = _safe_pct_change(sp_df["Close"], 21)

        # Yield curve
        if YIELD_2Y in macro_data and not macro_data[YIELD_2Y].empty:
            snap.yield_2y = float(macro_data[YIELD_2Y]["Close"].iloc[-1])
        if YIELD_10Y in macro_data and not macro_data[YIELD_10Y].empty:
            snap.yield_10y = float(macro_data[YIELD_10Y]["Close"].iloc[-1])

        snap.yield_spread = round(snap.yield_10y - snap.yield_2y, 2)
        if snap.yield_10y > 0 or snap.yield_2y > 0:
            if snap.yield_spread < -0.1:
                snap.yield_curve_status = "inverted"
            elif snap.yield_spread < 0.3:
                snap.yield_curve_status = "flat"
            else:
                snap.yield_curve_status = "normal"

        # Korrelationer
        snap.correlations = self._compute_correlations(macro_data)

        return snap

    def _compute_correlations(
        self,
        macro_data: dict[str, pd.DataFrame],
        window: int = 60,
    ) -> dict[str, dict[str, float]]:
        """Beregn korrelationer mellem aktivklasser (rolling 60 dage)."""
        labels = {
            SP500_SYMBOL: "S&P 500",
            GOLD_SYMBOL: "Guld",
            OIL_SYMBOL: "Olie",
            VIX_SYMBOL: "VIX",
            DXY_SYMBOL: "Dollar",
        }

        # Byg returns-matrix
        returns: dict[str, pd.Series] = {}
        for sym, label in labels.items():
            if sym in macro_data and not macro_data[sym].empty:
                df = macro_data[sym]
                if len(df) > window:
                    ret = df["Close"].pct_change().tail(window).dropna()
                    if len(ret) > 10:
                        returns[label] = ret

        if len(returns) < 2:
            return {}

        corr_matrix: dict[str, dict[str, float]] = {}
        for name_a, ret_a in returns.items():
            corr_matrix[name_a] = {}
            for name_b, ret_b in returns.items():
                # Align on common index
                aligned = pd.concat([ret_a, ret_b], axis=1, join="inner")
                if len(aligned) > 10:
                    c = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    corr_matrix[name_a][name_b] = round(c, 2) if not pd.isna(c) else 0.0
                else:
                    corr_matrix[name_a][name_b] = 0.0

        return corr_matrix

    # ══════════════════════════════════════════════════════════
    #  4. MULIGHEDS-ALERTS
    # ══════════════════════════════════════════════════════════

    def generate_alerts(
        self,
        scored: list[ScoredAsset],
        sectors: list[SectorPerformance],
        macro: MacroSnapshot,
    ) -> list[MarketAlert]:
        """Generér alerts baseret på scanning-resultater."""
        alerts: list[MarketAlert] = []
        now = datetime.now().isoformat()

        # ── Flight to Safety ──
        if macro.gold_change_1m > 3 and macro.sp500_change_1m < -2:
            alerts.append(MarketAlert(
                severity="HIGH",
                category="flight_to_safety",
                title="Guld stiger mens aktier falder — mulig flight to safety",
                message=(
                    f"Guld: +{macro.gold_change_1m:.1f}% (1m), "
                    f"S&P 500: {macro.sp500_change_1m:.1f}% (1m). "
                    f"Investorer flytter til sikre havne."
                ),
                symbols=[GOLD_SYMBOL, SP500_SYMBOL],
                timestamp=now,
            ))

        # ── Yield curve inverteret ──
        if macro.yield_curve_status == "inverted":
            alerts.append(MarketAlert(
                severity="HIGH",
                category="yield_curve",
                title="Yield curve inverteret — historisk recessions-signal",
                message=(
                    f"2Y={macro.yield_2y:.2f}%, 10Y={macro.yield_10y:.2f}%, "
                    f"spread={macro.yield_spread:.2f}%. "
                    f"Inverteret rentekurve har historisk forudset recession."
                ),
                timestamp=now,
            ))

        # ── VIX spike ──
        if macro.vix_level in ("high", "extreme"):
            alerts.append(MarketAlert(
                severity="HIGH",
                category="volatility",
                title=f"VIX er {macro.vix_level} ({macro.vix:.1f}) — markedet er nervøst",
                message=(
                    f"VIX: {macro.vix:.1f} ({macro.vix_change:+.1f}% i dag). "
                    f"Høj VIX indikerer frygt og kan skabe muligheder for moige investorer."
                ),
                symbols=[VIX_SYMBOL],
                timestamp=now,
            ))

        # ── Sektor breakout ──
        for sector in sectors:
            if sector.above_sma50 and sector.change_1w > 3:
                alerts.append(MarketAlert(
                    severity="MEDIUM",
                    category="breakout",
                    title=f"{sector.name}-sektoren har breakout over 50-dages SMA",
                    message=(
                        f"{sector.etf_symbol}: +{sector.change_1w:.1f}% (1 uge), "
                        f"+{sector.change_1m:.1f}% (1 måned). "
                        f"Stigende over SMA_50 tyder på ny optrend."
                    ),
                    symbols=[sector.etf_symbol],
                    timestamp=now,
                ))

        # ── Volume-spikes ──
        for asset in scored:
            if asset.volume_anomaly >= 3.0:
                alerts.append(MarketAlert(
                    severity="MEDIUM",
                    category="volume_spike",
                    title=(
                        f"Usædvanlig volumen i {asset.symbol} — "
                        f"{asset.volume_anomaly:.0f}x over gennemsnit"
                    ),
                    message=(
                        f"{asset.symbol}: volumen er {asset.volume_anomaly:.0f}x normalt niveau. "
                        f"Kursændring: {asset.change_pct:+.1f}%. "
                        f"Undersøg for nyheder eller institutionel aktivitet."
                    ),
                    symbols=[asset.symbol],
                    timestamp=now,
                ))

        # ── Dollar-bevægelse ──
        if abs(macro.dxy_change) > 2:
            direction = "styrkes" if macro.dxy_change > 0 else "svækkes"
            impact = "negativt for råstoffer og emerging markets" if macro.dxy_change > 0 \
                else "positivt for råstoffer og emerging markets"
            alerts.append(MarketAlert(
                severity="MEDIUM",
                category="fx",
                title=f"Dollaren {direction} markant ({macro.dxy_change:+.1f}% denne uge)",
                message=f"DXY: {macro.dxy:.1f}. En {direction}de dollar er typisk {impact}.",
                symbols=[DXY_SYMBOL],
                timestamp=now,
            ))

        # ── Oliepris ──
        if abs(macro.oil_change_1m) > 10:
            direction = "stiger" if macro.oil_change_1m > 0 else "falder"
            alerts.append(MarketAlert(
                severity="MEDIUM",
                category="commodities",
                title=f"Oliepris {direction} kraftigt ({macro.oil_change_1m:+.1f}% / 1m)",
                message=(
                    f"Råolie: ${macro.oil_price:.2f}. "
                    f"Påvirker energisektoren og inflationsforventninger."
                ),
                symbols=[OIL_SYMBOL],
                timestamp=now,
            ))

        # ── Korrelation breakdown ──
        corr = macro.correlations
        if "S&P 500" in corr and "Guld" in corr.get("S&P 500", {}):
            spx_gold = corr["S&P 500"]["Guld"]
            if spx_gold > 0.5:
                alerts.append(MarketAlert(
                    severity="LOW",
                    category="correlation",
                    title="Guld og aktier korrelerer positivt — usædvanligt",
                    message=(
                        f"Korrelation: {spx_gold:.2f}. "
                        f"Normalt er guld og aktier negativt korrelerede."
                    ),
                    timestamp=now,
                ))

        alerts.sort(key=lambda a: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[a.severity])
        return alerts

    # ══════════════════════════════════════════════════════════
    #  5. PORTEFØLJE-ANBEFALING
    # ══════════════════════════════════════════════════════════

    def recommend_allocation(
        self,
        macro: MacroSnapshot,
        sectors: list[SectorPerformance],
        current_allocation: dict[str, float] | None = None,
    ) -> AllocationRecommendation:
        """
        Foreslå en diversificeret porteføljeallokering baseret på
        aktuelle makro-signaler.

        Basisallokering: 60% aktier, 25% obligationer, 10% råstoffer, 5% krypto.
        Justeres baseret på VIX, yield curve og sektortrends.
        """
        # Start med basis
        stocks = 60.0
        bonds = 25.0
        commodities = 10.0
        crypto = 5.0
        cash = 0.0
        rationale_parts: list[str] = []

        # ── VIX-justering ──
        if macro.vix_level in ("high", "extreme"):
            stocks -= 10
            bonds += 5
            cash += 5
            rationale_parts.append(
                f"VIX er {macro.vix_level} ({macro.vix:.0f}) → reducér aktier, øg sikre havne"
            )
        elif macro.vix_level == "low":
            stocks += 5
            bonds -= 5
            rationale_parts.append(
                f"VIX er lav ({macro.vix:.0f}) → øg aktier"
            )

        # ── Yield curve-justering ──
        if macro.yield_curve_status == "inverted":
            stocks -= 10
            bonds += 5
            cash += 5
            rationale_parts.append(
                "Inverteret rentekurve → defensiv positionering"
            )
        elif macro.yield_curve_status == "flat":
            stocks -= 5
            bonds += 5
            rationale_parts.append(
                "Flad rentekurve → moderat defensiv"
            )

        # ── Flight-to-safety ──
        if macro.gold_change_1m > 5:
            commodities += 5
            stocks -= 5
            rationale_parts.append(
                f"Guld +{macro.gold_change_1m:.1f}% → øg råstoffer som hedge"
            )

        # ── Olie-inflation ──
        if macro.oil_change_1m > 10:
            commodities += 3
            bonds -= 3
            rationale_parts.append(
                "Stigende olie → inflation risk → mere råstoffer"
            )

        # Normaliser til 100%
        total = stocks + bonds + commodities + crypto + cash
        if total > 0:
            stocks = round(stocks / total * 100, 1)
            bonds = round(bonds / total * 100, 1)
            commodities = round(commodities / total * 100, 1)
            crypto = round(crypto / total * 100, 1)
            cash = round(100 - stocks - bonds - commodities - crypto, 1)

        # ── Sektor-vægte ──
        sector_weights: dict[str, float] = {}
        up_sectors = [s for s in sectors if s.trend == "up"]
        down_sectors = [s for s in sectors if s.trend == "down"]
        neutral_sectors = [s for s in sectors if s.trend == "neutral"]

        base_weight = stocks / max(len(sectors), 1)
        for s in up_sectors:
            sector_weights[s.name] = round(base_weight * 1.3, 1)
        for s in neutral_sectors:
            sector_weights[s.name] = round(base_weight, 1)
        for s in down_sectors:
            sector_weights[s.name] = round(base_weight * 0.7, 1)

        # ── Rebalanceringsforslag ──
        actions: list[str] = []
        if current_allocation:
            cur_stocks = current_allocation.get("stocks", 0)
            cur_bonds = current_allocation.get("bonds", 0)
            cur_commodities = current_allocation.get("commodities", 0)
            cur_crypto = current_allocation.get("crypto", 0)

            diff_stocks = stocks - cur_stocks
            diff_bonds = bonds - cur_bonds
            diff_commodities = commodities - cur_commodities
            diff_crypto = crypto - cur_crypto

            if abs(diff_stocks) > 3:
                verb = "Køb" if diff_stocks > 0 else "Sælg"
                actions.append(f"{verb} {abs(diff_stocks):.0f}% aktier")
            if abs(diff_bonds) > 3:
                verb = "Køb" if diff_bonds > 0 else "Sælg"
                actions.append(f"{verb} {abs(diff_bonds):.0f}% obligationer")
            if abs(diff_commodities) > 3:
                verb = "Køb" if diff_commodities > 0 else "Sælg"
                actions.append(f"{verb} {abs(diff_commodities):.0f}% råstoffer")
            if abs(diff_crypto) > 3:
                verb = "Køb" if diff_crypto > 0 else "Sælg"
                actions.append(f"{verb} {abs(diff_crypto):.0f}% krypto")

            # Sektor-rotation
            for s in up_sectors[:2]:
                actions.append(f"Overvej at øge {s.name} (+{s.change_1m:.1f}%)")
            for s in down_sectors[:2]:
                actions.append(f"Overvej at reducere {s.name} ({s.change_1m:.1f}%)")

        return AllocationRecommendation(
            stocks_pct=stocks,
            bonds_pct=bonds,
            commodities_pct=commodities,
            crypto_pct=crypto,
            cash_pct=cash,
            rationale=" | ".join(rationale_parts) if rationale_parts else "Standard allokering",
            sector_weights=sector_weights,
            rebalance_actions=actions,
        )

    # ══════════════════════════════════════════════════════════
    #  6. FULD SCANNING
    # ══════════════════════════════════════════════════════════

    def full_scan(
        self,
        asset_data: dict[str, pd.DataFrame],
        sector_data: dict[str, pd.DataFrame],
        macro_data: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame | None = None,
        current_allocation: dict[str, float] | None = None,
    ) -> ScanResult:
        """
        Kør komplet markedsscanning: scoring + sektor + makro + alerts.

        Args:
            asset_data: symbol → DataFrame med OHLCV + indikatorer.
            sector_data: sektor-ETF → DataFrame.
            macro_data: makro-symboler (VIX, DXY, guld, olie osv.) → DataFrame.
            benchmark: S&P 500 DataFrame.
            current_allocation: Nuværende allokering (til rebalancering).

        Returns:
            ScanResult med alle resultater.
        """
        start = pd.Timestamp.now()
        now = start.isoformat()
        logger.info(f"Fuld markedsscanning startet: {len(asset_data)} aktiver")

        # 1. Score alle aktiver
        all_scored = self.score_universe(asset_data, benchmark=benchmark)
        top_buys, top_sells = self.get_top_picks(all_scored)
        logger.info(
            f"Scoring færdig: {len(all_scored)} aktiver, "
            f"{len(top_buys)} KØB, {len(top_sells)} SÆLG"
        )

        # 2. Sektor-analyse
        sectors = self.analyze_sectors(sector_data, benchmark=benchmark)

        # 3. Makro-snapshot
        macro = self.macro_snapshot(macro_data)

        # 4. Alerts
        alerts = self.generate_alerts(all_scored, sectors, macro)
        logger.info(f"Genereret {len(alerts)} alerts")

        # 5. Allokering
        allocation = self.recommend_allocation(
            macro, sectors, current_allocation=current_allocation,
        )

        elapsed = (pd.Timestamp.now() - start).total_seconds() * 1000

        result = ScanResult(
            timestamp=now,
            top_buys=top_buys,
            top_sells=top_sells,
            all_scored=all_scored,
            sector_performance=sectors,
            macro=macro,
            alerts=alerts,
            allocation=allocation,
            scan_duration_ms=elapsed,
        )

        logger.info(f"Fuld scanning færdig på {elapsed:.0f}ms")
        return result

    # ══════════════════════════════════════════════════════════
    #  CLI-udskrift
    # ══════════════════════════════════════════════════════════

    def print_scan_result(self, result: ScanResult) -> None:
        """Print scan-resultater til konsol."""
        print("\n" + "═" * 70)
        print("  MARKEDSSCANNING")
        print("═" * 70)
        print(f"  Tidspunkt: {result.timestamp[:19]}")
        print(f"  Aktiver analyseret: {len(result.all_scored)}")
        print(f"  Tid: {result.scan_duration_ms:.0f}ms")

        # ── Top KØB ──
        print("\n" + "─" * 70)
        print("  🟢 TOP KØB-KANDIDATER")
        print("─" * 70)
        print(f"  {'#':<3} {'Symbol':<10} {'Score':>6} {'Ændring':>8} {'Årsag'}")
        print(f"  {'─'*3} {'─'*10} {'─'*6} {'─'*8} {'─'*35}")
        for i, a in enumerate(result.top_buys, 1):
            reason = a.reasons[0] if a.reasons else ""
            print(f"  {i:<3} {a.symbol:<10} {a.score:>5.0f}  {a.change_pct:>+7.1f}%  {reason}")

        # ── Top SÆLG ──
        print("\n" + "─" * 70)
        print("  🔴 TOP SÆLG-KANDIDATER")
        print("─" * 70)
        print(f"  {'#':<3} {'Symbol':<10} {'Score':>6} {'Ændring':>8} {'Årsag'}")
        print(f"  {'─'*3} {'─'*10} {'─'*6} {'─'*8} {'─'*35}")
        for i, a in enumerate(result.top_sells, 1):
            reason = a.reasons[0] if a.reasons else ""
            print(f"  {i:<3} {a.symbol:<10} {a.score:>5.0f}  {a.change_pct:>+7.1f}%  {reason}")

        # ── Sektorer ──
        print("\n" + "─" * 70)
        print("  📊 SEKTOR-PERFORMANCE")
        print("─" * 70)
        print(f"  {'Sektor':<22} {'1d':>6} {'1u':>6} {'1m':>6} {'3m':>6} {'RS':>6} {'Trend':>8}")
        print(f"  {'─'*22} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
        for s in result.sector_performance:
            trend_icon = "↑" if s.trend == "up" else "↓" if s.trend == "down" else "→"
            print(
                f"  {s.name:<22} {s.change_1d:>+5.1f}% {s.change_1w:>+5.1f}% "
                f"{s.change_1m:>+5.1f}% {s.change_3m:>+5.1f}% "
                f"{s.relative_strength_1m:>+5.1f}  {trend_icon:>7}"
            )

        # ── Makro ──
        print("\n" + "─" * 70)
        print("  🌍 MAKRO-INDIKATORER")
        print("─" * 70)
        m = result.macro
        vix_icon = "🟢" if m.vix_level == "low" else "🟡" if m.vix_level in ("normal", "elevated") \
            else "🔴"
        yc_icon = "🟢" if m.yield_curve_status == "normal" else "🟡" if m.yield_curve_status == "flat" \
            else "🔴"
        print(f"  {vix_icon} VIX:          {m.vix:.1f} ({m.vix_change:+.1f}%) — {m.vix_level}")
        print(f"  {'💲'} Dollar (DXY):  {m.dxy:.1f} ({m.dxy_change:+.1f}%) — {m.dxy_trend}")
        print(f"  {'🥇'} Guld:          ${m.gold_price:,.0f} ({m.gold_change_1m:+.1f}%/1m)")
        print(f"  {'🛢️ '} Olie:          ${m.oil_price:.2f} ({m.oil_change_1m:+.1f}%/1m)")
        print(f"  {yc_icon} Yield curve:   {m.yield_spread:+.2f}% — {m.yield_curve_status}")
        print(f"     2Y: {m.yield_2y:.2f}%  10Y: {m.yield_10y:.2f}%")

        # ── Alerts ──
        if result.alerts:
            print("\n" + "─" * 70)
            print("  ⚡ ALERTS")
            print("─" * 70)
            for alert in result.alerts:
                icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}[alert.severity]
                print(f"  {icon} [{alert.severity}] {alert.title}")
                print(f"     {alert.message}")
                print()

        # ── Allokering ──
        alloc = result.allocation
        print("─" * 70)
        print("  💼 ANBEFALET ALLOKERING")
        print("─" * 70)
        print(f"  Aktier:       {alloc.stocks_pct:.0f}%")
        print(f"  Obligationer: {alloc.bonds_pct:.0f}%")
        print(f"  Råstoffer:    {alloc.commodities_pct:.0f}%")
        print(f"  Krypto:       {alloc.crypto_pct:.0f}%")
        print(f"  Kontanter:    {alloc.cash_pct:.0f}%")
        print(f"  Rationale:    {alloc.rationale}")

        if alloc.rebalance_actions:
            print("\n  Rebalanceringsforslag:")
            for action in alloc.rebalance_actions:
                print(f"    → {action}")

        print("\n" + "═" * 70)

    def print_macro(self, macro: MacroSnapshot) -> None:
        """Print kun makro-dashboard til konsol."""
        print("\n" + "═" * 70)
        print("  MAKRO-DASHBOARD")
        print("═" * 70)

        m = macro
        vix_icon = "🟢" if m.vix_level == "low" else "🟡" if m.vix_level in ("normal", "elevated") \
            else "🔴"
        yc_icon = "🟢" if m.yield_curve_status == "normal" else "🟡" if m.yield_curve_status == "flat" \
            else "🔴"

        print(f"\n  {vix_icon} VIX (Frygtindeks)")
        print(f"     Niveau: {m.vix:.1f} ({m.vix_change:+.1f}%)")
        print(f"     Status: {m.vix_level.upper()}")
        desc = {
            "low": "Investorer er trygge. Ideel til at tage risiko.",
            "normal": "Normalt niveau. Ingen særlig frygt.",
            "elevated": "Noget forhøjet. Vær opmærksom.",
            "high": "Høj frygt. Overvej at afdække risiko.",
            "extreme": "Ekstremt. Panik i markedet. Muligheder for modige.",
        }
        print(f"     {desc.get(m.vix_level, '')}")

        print(f"\n  💲 Dollar (DXY)")
        print(f"     Niveau: {m.dxy:.1f}")
        print(f"     Trend:  {m.dxy_trend} ({m.dxy_change:+.1f}% denne uge)")

        print(f"\n  🥇 Guld")
        print(f"     Pris:   ${m.gold_price:,.0f}")
        print(f"     1 md:   {m.gold_change_1m:+.1f}%")

        print(f"\n  🛢️  Olie (Crude)")
        print(f"     Pris:   ${m.oil_price:.2f}")
        print(f"     1 md:   {m.oil_change_1m:+.1f}%")

        print(f"\n  {yc_icon} Yield Curve")
        print(f"     2-års rente:  {m.yield_2y:.2f}%")
        print(f"     10-års rente: {m.yield_10y:.2f}%")
        print(f"     Spread:       {m.yield_spread:+.2f}%")
        print(f"     Status:       {m.yield_curve_status.upper()}")
        yc_desc = {
            "normal": "Normal kurve. Markedet forventer vækst.",
            "flat": "Flad kurve. Usikkerhed om fremtiden.",
            "inverted": "INVERTERET! Historisk recessions-signal.",
        }
        print(f"     {yc_desc.get(m.yield_curve_status, '')}")

        print(f"\n  📈 S&P 500")
        print(f"     1 md:   {m.sp500_change_1m:+.1f}%")

        if m.correlations:
            print(f"\n  🔗 Korrelationer (60 dage)")
            labels = list(m.correlations.keys())
            header = f"  {'':>12}" + "".join(f"{l:>12}" for l in labels)
            print(header)
            for name in labels:
                row = f"  {name:>12}"
                for other in labels:
                    c = m.correlations.get(name, {}).get(other, 0)
                    row += f"{c:>12.2f}"
                print(row)

        print("\n" + "═" * 70)
