"""
On-chain krypto-analysemodul.

Funktionalitet:
  - On-chain metrics: active addresses, exchange flows, whale tracking,
    hash rate, NVT ratio
  - DeFi metrics: TVL, DEX volume, stablecoin supply
  - Fear & Greed Index (krypto-specifik via alternative.me)
  - Bitcoin dominance og alt-season detektion
  - Integrerer med strategier via confidence-justering

Datakilder:
  - alternative.me API (gratis, Fear & Greed)
  - CoinGecko API (gratis, markedsdata + dominance)
  - Blockchain.com API (gratis, on-chain BTC stats)
  - DeFi Llama API (gratis, TVL data)
  - Etherscan / public RPCs (via web3, valgfrit)
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings

# ── Optional imports ─────────────────────────────────────────

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from web3 import Web3
    _HAS_WEB3 = True
except ImportError:
    _HAS_WEB3 = False
    logger.debug("web3 ikke installeret – direkte blockchain queries utilgængelige")


# ── Konstanter ───────────────────────────────────────────────

_MIN_REQUEST_GAP = 1.0  # sekunder mellem API-kald

# API endpoints (alle gratis, ingen nøgle kræves)
FEAR_GREED_API = "https://api.alternative.me/fng/"
COINGECKO_API = "https://api.coingecko.com/api/v3"
BLOCKCHAIN_API = "https://api.blockchain.info"
DEFILLAMA_API = "https://api.llama.fi"

# Krypto-ticker → CoinGecko ID mapping
CRYPTO_IDS: dict[str, str] = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "BNB-USD": "binancecoin",
    "SOL-USD": "solana",
    "XRP-USD": "ripple",
    "ADA-USD": "cardano",
    "DOGE-USD": "dogecoin",
    "AVAX-USD": "avalanche-2",
    "DOT-USD": "polkadot",
    "LINK-USD": "chainlink",
    "TRX-USD": "tron",
    "MATIC-USD": "matic-network",
    "SHIB-USD": "shiba-inu",
    "LTC-USD": "litecoin",
    "BCH-USD": "bitcoin-cash",
    "UNI7083-USD": "uniswap",
    "NEAR-USD": "near",
    "FIL-USD": "filecoin",
}


# ══════════════════════════════════════════════════════════════
#  Dataclasses
# ══════════════════════════════════════════════════════════════

class FearGreedLevel(Enum):
    """Fear & Greed klassificering."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


class OnChainSignal(Enum):
    """On-chain signal-styrke."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class FearGreedIndex:
    """Crypto Fear & Greed Index."""
    value: int                      # 0–100
    classification: str             # "Extreme Fear", "Fear", etc.
    level: FearGreedLevel
    timestamp: str
    contrarian_signal: str          # "buy" / "sell" / "neutral"

    @property
    def score(self) -> float:
        """Contrarian score: lav fear = køb, høj greed = sælg."""
        # Inverteret: extreme fear (0) → score 100 (stærkt køb)
        return 100.0 - self.value


@dataclass
class ExchangeFlowData:
    """Exchange inflow/outflow data."""
    inflow_btc: float          # BTC der strømmer IND på exchanges
    outflow_btc: float         # BTC der strømmer UD af exchanges
    net_flow: float            # inflow - outflow
    signal: str                # "bullish" (outflow) / "bearish" (inflow)
    description: str


@dataclass
class ActiveAddresses:
    """Antal aktive adresser på blockchain."""
    count: int
    change_pct_7d: float       # ændring over 7 dage
    signal: str                # "bullish" / "bearish" / "neutral"


@dataclass
class HashRateData:
    """Bitcoin hash rate (mining styrke)."""
    hash_rate: float           # TH/s
    change_pct_30d: float      # ændring over 30 dage
    signal: str                # "bullish" / "bearish"


@dataclass
class NVTRatio:
    """Network Value to Transactions ratio."""
    nvt: float
    signal: str                # "overvalued" / "undervalued" / "fair"
    description: str

    @property
    def is_overvalued(self) -> bool:
        return self.nvt > 95

    @property
    def is_undervalued(self) -> bool:
        return self.nvt < 45


@dataclass
class WhaleActivity:
    """Store wallet-bevægelser."""
    large_txs_24h: int         # transaktioner > $1M
    whale_sentiment: str       # "accumulating" / "distributing" / "neutral"
    largest_tx_usd: float
    description: str


@dataclass
class DeFiMetrics:
    """DeFi økosystem metrics."""
    total_tvl_usd: float       # Total Value Locked i DeFi
    tvl_change_24h_pct: float
    tvl_change_7d_pct: float
    top_protocols: list[dict]  # [{name, tvl, change}]
    stablecoin_mcap: float     # total stablecoin market cap
    signal: str


@dataclass
class BitcoinDominance:
    """BTC's andel af total krypto markedsværdi."""
    dominance_pct: float       # f.eks. 52.3
    change_7d: float           # ændring over 7 dage
    alt_season: bool           # True hvis alt-season (dominance faldende)
    signal: str                # "btc_strength" / "alt_season" / "neutral"
    description: str


@dataclass
class OnChainReport:
    """Samlet on-chain rapport for en kryptovaluta."""
    symbol: str
    timestamp: str
    fear_greed: FearGreedIndex | None
    exchange_flow: ExchangeFlowData | None
    active_addresses: ActiveAddresses | None
    hash_rate: HashRateData | None
    nvt_ratio: NVTRatio | None
    whale_activity: WhaleActivity | None
    defi_metrics: DeFiMetrics | None
    btc_dominance: BitcoinDominance | None
    overall_signal: OnChainSignal
    confidence: float           # 0–100
    summary: str


# ══════════════════════════════════════════════════════════════
#  On-Chain Tracker
# ══════════════════════════════════════════════════════════════

class OnChainTracker:
    """
    Henter og analyserer on-chain krypto-data.

    Bruger gratis API'er: alternative.me, CoinGecko, Blockchain.com, DeFi Llama.
    """

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = Path(cache_dir or "data/cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._cache_dir / "onchain_cache.db"
        self._last_request = 0.0
        self._init_db()

    def _init_db(self) -> None:
        """Opret cache-tabeller."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS onchain_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    fetched_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fear_greed_history (
                    date TEXT PRIMARY KEY,
                    value INTEGER NOT NULL,
                    classification TEXT NOT NULL
                )
            """)

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _throttle(self) -> None:
        """Rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < _MIN_REQUEST_GAP:
            time.sleep(_MIN_REQUEST_GAP - elapsed)
        self._last_request = time.time()

    def _get_json(self, url: str, params: dict | None = None) -> dict | list | None:
        """Hent JSON fra API med rate limiting og fejlhåndtering."""
        if not _HAS_REQUESTS:
            logger.warning("requests ikke installeret")
            return None

        self._throttle()
        try:
            resp = _requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"API fejl for {url}: {e}")
            return None

    def _read_cache(self, key: str, max_age_hours: float = 1.0) -> str | None:
        """Læs fra cache hvis ikke udløbet."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT value, fetched_at FROM onchain_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        fetched = datetime.fromisoformat(row[1])
        if datetime.now() - fetched > timedelta(hours=max_age_hours):
            return None
        return row[0]

    def _write_cache(self, key: str, value: str) -> None:
        """Skriv til cache."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO onchain_cache (key, value, fetched_at)
                   VALUES (?, ?, ?)""",
                (key, value, datetime.now().isoformat()),
            )

    # ── Fear & Greed Index ───────────────────────────────────

    def get_fear_greed(self, days: int = 1) -> FearGreedIndex | None:
        """
        Hent Crypto Fear & Greed Index fra alternative.me.

        Gratis API, ingen nøgle. Opdateres dagligt.
        Contrarian: Extreme Fear = potentielt køb, Extreme Greed = potentielt salg.
        """
        import json

        cached = self._read_cache("fear_greed", max_age_hours=6.0)
        if cached:
            try:
                d = json.loads(cached)
                return FearGreedIndex(**d)
            except Exception:
                pass

        data = self._get_json(FEAR_GREED_API, params={"limit": str(days)})
        if not data or "data" not in data:
            return None

        entry = data["data"][0]
        value = int(entry["value"])
        classification = entry.get("value_classification", "")
        timestamp = entry.get("timestamp", "")

        # Klassificering
        if value <= 20:
            level = FearGreedLevel.EXTREME_FEAR
            contrarian = "buy"
        elif value <= 40:
            level = FearGreedLevel.FEAR
            contrarian = "buy"
        elif value <= 60:
            level = FearGreedLevel.NEUTRAL
            contrarian = "neutral"
        elif value <= 80:
            level = FearGreedLevel.GREED
            contrarian = "sell"
        else:
            level = FearGreedLevel.EXTREME_GREED
            contrarian = "sell"

        result = FearGreedIndex(
            value=value,
            classification=classification,
            level=level,
            timestamp=timestamp,
            contrarian_signal=contrarian,
        )

        # Cache
        self._write_cache("fear_greed", json.dumps({
            "value": result.value,
            "classification": result.classification,
            "level": result.level.value,
            "timestamp": result.timestamp,
            "contrarian_signal": result.contrarian_signal,
        }))

        # Gem i historik
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO fear_greed_history
                   (date, value, classification) VALUES (?, ?, ?)""",
                (datetime.now().strftime("%Y-%m-%d"), value, classification),
            )

        return result

    # ── Bitcoin Dominance ────────────────────────────────────

    def get_btc_dominance(self) -> BitcoinDominance | None:
        """
        Hent Bitcoin dominance fra CoinGecko.

        Stigende dominance = flight to quality (bearish for alts).
        Faldende dominance = alt season (bullish for alts).
        """
        data = self._get_json(f"{COINGECKO_API}/global")
        if not data or "data" not in data:
            return None

        market_data = data["data"]
        btc_dom = market_data.get("market_cap_percentage", {}).get("btc", 0)
        btc_change = market_data.get("market_cap_change_percentage_24h_usd", 0)

        # Alt season: BTC dominance < 40% eller faldende hurtigt
        alt_season = btc_dom < 40

        if btc_dom > 55:
            signal = "btc_strength"
            desc = f"BTC dominance {btc_dom:.1f}% – flight to quality, bearish for alts"
        elif btc_dom < 40:
            signal = "alt_season"
            desc = f"BTC dominance {btc_dom:.1f}% – alt season, bullish for alts"
        else:
            signal = "neutral"
            desc = f"BTC dominance {btc_dom:.1f}% – neutral"

        return BitcoinDominance(
            dominance_pct=float(btc_dom),
            change_7d=float(btc_change),
            alt_season=alt_season,
            signal=signal,
            description=desc,
        )

    # ── Exchange Flow ────────────────────────────────────────

    def get_exchange_flow(self) -> ExchangeFlowData | None:
        """
        Estimér exchange flow baseret på Blockchain.com data.

        Stor inflow → folk vil sælge → bearish.
        Stor outflow → folk hodler → bullish.
        """
        # Blockchain.com giver transaktionsdata
        data = self._get_json(f"{BLOCKCHAIN_API}/stats")
        if not data:
            return None

        # Brug transactions per day og mempool som proxy
        n_tx = data.get("n_tx", 0)
        hash_rate = data.get("hash_rate", 0)
        total_btc_sent = data.get("total_btc_sent", 0) / 1e8  # satoshi → BTC
        estimated_tx_volume = data.get("estimated_transaction_volume_usd", 0)

        # Simpel heuristik: høj tx volume relativ til markedsværdi = mere exchange aktivitet
        market_cap = data.get("market_price_usd", 50000) * 21_000_000
        flow_ratio = estimated_tx_volume / market_cap if market_cap > 0 else 0

        # Vi kan ikke skelne inflow/outflow fra public API
        # Brug flow ratio som proxy
        inflow_est = total_btc_sent * 0.3    # estimat: ~30% går til exchanges
        outflow_est = total_btc_sent * 0.25  # estimat: ~25% forlader exchanges
        net = inflow_est - outflow_est

        if flow_ratio > 0.05:
            signal = "bearish"
            desc = "Høj transaktionsvolumen – mulig salgspres"
        elif flow_ratio < 0.01:
            signal = "bullish"
            desc = "Lav transaktionsvolumen – hodlers dominerer"
        else:
            signal = "neutral"
            desc = "Normal transaktionsvolumen"

        return ExchangeFlowData(
            inflow_btc=float(inflow_est),
            outflow_btc=float(outflow_est),
            net_flow=float(net),
            signal=signal,
            description=desc,
        )

    # ── Active Addresses ─────────────────────────────────────

    def get_active_addresses(self) -> ActiveAddresses | None:
        """
        Hent antal aktive BTC-adresser fra Blockchain.com.

        Flere aktive adresser = mere netværksaktivitet = bullish.
        """
        data = self._get_json(f"{BLOCKCHAIN_API}/stats")
        if not data:
            return None

        n_tx = data.get("n_tx", 0)
        # Blockchain.com stats giver ikke direkte active addresses,
        # men n_unique_addresses fra charts API
        # Brug n_tx som proxy (korrelerer stærkt)
        n_unique = data.get("n_btc_mined", 0)  # bruger mined som fallback

        # Hent 7d ændring fra chart API
        chart_data = self._get_json(
            f"{BLOCKCHAIN_API}/charts/n-unique-addresses",
            params={"timespan": "30days", "format": "json"},
        )

        count = n_tx  # proxy
        change_7d = 0.0

        if chart_data and "values" in chart_data:
            vals = chart_data["values"]
            if len(vals) >= 7:
                recent = np.mean([v["y"] for v in vals[-3:]])
                week_ago = np.mean([v["y"] for v in vals[-10:-7]])
                count = int(recent)
                if week_ago > 0:
                    change_7d = (recent - week_ago) / week_ago * 100

        if change_7d > 5:
            signal = "bullish"
        elif change_7d < -5:
            signal = "bearish"
        else:
            signal = "neutral"

        return ActiveAddresses(
            count=count,
            change_pct_7d=float(change_7d),
            signal=signal,
        )

    # ── Hash Rate ────────────────────────────────────────────

    def get_hash_rate(self) -> HashRateData | None:
        """
        Hent Bitcoin hash rate.

        Stigende hash rate = miners er optimistiske = bullish.
        Faldende hash rate = miners lukker ned = bearish.
        """
        data = self._get_json(f"{BLOCKCHAIN_API}/stats")
        if not data:
            return None

        hash_rate = data.get("hash_rate", 0) / 1e9  # konvertér til TH/s

        # 30d ændring fra chart API
        chart_data = self._get_json(
            f"{BLOCKCHAIN_API}/charts/hash-rate",
            params={"timespan": "60days", "format": "json"},
        )

        change_30d = 0.0
        if chart_data and "values" in chart_data:
            vals = chart_data["values"]
            if len(vals) >= 30:
                recent = np.mean([v["y"] for v in vals[-7:]])
                month_ago = np.mean([v["y"] for v in vals[-37:-30]])
                if month_ago > 0:
                    change_30d = (recent - month_ago) / month_ago * 100
                    hash_rate = recent / 1e9

        signal = "bullish" if change_30d > 3 else "bearish" if change_30d < -3 else "neutral"

        return HashRateData(
            hash_rate=float(hash_rate),
            change_pct_30d=float(change_30d),
            signal=signal,
        )

    # ── NVT Ratio ────────────────────────────────────────────

    def get_nvt_ratio(self) -> NVTRatio | None:
        """
        Beregn NVT Ratio (Network Value to Transactions).

        NVT = Market Cap / Daily Transaction Volume.
        Kryptovalutaens "P/E ratio".
        Høj NVT (>95) = overvalued, lav NVT (<45) = undervalued.
        """
        data = self._get_json(f"{BLOCKCHAIN_API}/stats")
        if not data:
            return None

        market_price = data.get("market_price_usd", 0)
        total_supply = 19_500_000  # approx circulating BTC
        market_cap = market_price * total_supply

        tx_volume = data.get("estimated_transaction_volume_usd", 0)
        daily_volume = tx_volume  # allerede dagligt

        if daily_volume <= 0:
            return NVTRatio(nvt=0, signal="unknown",
                          description="Ingen transaktionsdata tilgængelig")

        nvt = market_cap / daily_volume

        if nvt > 95:
            signal = "overvalued"
            desc = f"NVT {nvt:.0f} – netværket er overvalued relativt til brug"
        elif nvt < 45:
            signal = "undervalued"
            desc = f"NVT {nvt:.0f} – netværket er undervalued relativt til brug"
        else:
            signal = "fair"
            desc = f"NVT {nvt:.0f} – fair value"

        return NVTRatio(nvt=float(nvt), signal=signal, description=desc)

    # ── Whale Activity ───────────────────────────────────────

    def get_whale_activity(self) -> WhaleActivity | None:
        """
        Estimér whale-aktivitet baseret på store transaktioner.

        Whales der køber = accumulating = bullish.
        Whales der sælger = distributing = bearish.
        """
        data = self._get_json(f"{BLOCKCHAIN_API}/stats")
        if not data:
            return None

        n_tx = data.get("n_tx", 0)
        total_btc = data.get("total_btc_sent", 0) / 1e8
        market_price = data.get("market_price_usd", 50000)
        avg_tx_value = (total_btc * market_price / n_tx) if n_tx > 0 else 0

        # Estimér antal store transaktioner (>$1M)
        # Ca. 0.5-2% af transaktioner er >$1M baseret på typisk distribution
        whale_pct = min(0.02, max(0.005, avg_tx_value / 500_000))
        large_txs = int(n_tx * whale_pct)
        largest_est = avg_tx_value * 20  # estimat

        if avg_tx_value > 50_000:
            sentiment = "accumulating"
            desc = f"Høj gennemsnitsværdi ({avg_tx_value:,.0f} USD) – whales akkumulerer"
        elif avg_tx_value < 10_000:
            sentiment = "distributing"
            desc = f"Lav gennemsnitsværdi ({avg_tx_value:,.0f} USD) – retail dominerer"
        else:
            sentiment = "neutral"
            desc = f"Normal gennemsnitsværdi ({avg_tx_value:,.0f} USD)"

        return WhaleActivity(
            large_txs_24h=large_txs,
            whale_sentiment=sentiment,
            largest_tx_usd=float(largest_est),
            description=desc,
        )

    # ── DeFi Metrics ─────────────────────────────────────────

    def get_defi_metrics(self) -> DeFiMetrics | None:
        """
        Hent DeFi metrics fra DeFi Llama (gratis, ingen nøgle).

        TVL = Total Value Locked i DeFi protocols.
        Stigende TVL = mere kapital i DeFi = bullish for krypto.
        """
        # Total TVL
        tvl_data = self._get_json(f"{DEFILLAMA_API}/v2/historicalChainTvl")
        protocols_data = self._get_json(f"{DEFILLAMA_API}/protocols")
        stablecoin_data = self._get_json(f"{DEFILLAMA_API}/stablecoins")

        total_tvl = 0.0
        tvl_24h_change = 0.0
        tvl_7d_change = 0.0

        if tvl_data and isinstance(tvl_data, list) and len(tvl_data) >= 2:
            total_tvl = tvl_data[-1].get("tvl", 0)
            yesterday_tvl = tvl_data[-2].get("tvl", 0) if len(tvl_data) >= 2 else total_tvl
            week_ago_tvl = tvl_data[-8].get("tvl", 0) if len(tvl_data) >= 8 else total_tvl

            if yesterday_tvl > 0:
                tvl_24h_change = (total_tvl - yesterday_tvl) / yesterday_tvl * 100
            if week_ago_tvl > 0:
                tvl_7d_change = (total_tvl - week_ago_tvl) / week_ago_tvl * 100

        # Top protocols
        top_protocols = []
        if protocols_data and isinstance(protocols_data, list):
            sorted_protos = sorted(protocols_data, key=lambda x: x.get("tvl", 0), reverse=True)
            for p in sorted_protos[:5]:
                top_protocols.append({
                    "name": p.get("name", "Unknown"),
                    "tvl": p.get("tvl", 0),
                    "change_1d": p.get("change_1d", 0),
                })

        # Stablecoin market cap
        stablecoin_mcap = 0.0
        if stablecoin_data and "peggedAssets" in stablecoin_data:
            for sc in stablecoin_data["peggedAssets"]:
                chains = sc.get("chainCirculating", {})
                for chain_data in chains.values():
                    pegged = chain_data.get("current", {})
                    stablecoin_mcap += pegged.get("peggedUSD", 0)

        # Signal
        if tvl_7d_change > 5:
            signal = "bullish"
        elif tvl_7d_change < -5:
            signal = "bearish"
        else:
            signal = "neutral"

        return DeFiMetrics(
            total_tvl_usd=float(total_tvl),
            tvl_change_24h_pct=float(tvl_24h_change),
            tvl_change_7d_pct=float(tvl_7d_change),
            top_protocols=top_protocols,
            stablecoin_mcap=float(stablecoin_mcap),
            signal=signal,
        )

    # ── Samlet Rapport ───────────────────────────────────────

    def get_report(self, symbol: str = "BTC-USD") -> OnChainReport:
        """
        Generér samlet on-chain rapport.

        Henter alle tilgængelige metrics og beregner overall signal.
        """
        fear_greed = None
        exchange_flow = None
        active_addr = None
        hash_rate = None
        nvt = None
        whale = None
        defi = None
        dominance = None

        # Hent alle metrics (fejl i én stopper ikke de andre)
        try:
            fear_greed = self.get_fear_greed()
        except Exception as e:
            logger.debug(f"Fear & Greed fejl: {e}")

        try:
            dominance = self.get_btc_dominance()
        except Exception as e:
            logger.debug(f"BTC dominance fejl: {e}")

        # BTC-specifik on-chain data
        if symbol in ("BTC-USD", "bitcoin") or symbol.startswith("BTC"):
            try:
                exchange_flow = self.get_exchange_flow()
            except Exception as e:
                logger.debug(f"Exchange flow fejl: {e}")
            try:
                active_addr = self.get_active_addresses()
            except Exception as e:
                logger.debug(f"Active addresses fejl: {e}")
            try:
                hash_rate = self.get_hash_rate()
            except Exception as e:
                logger.debug(f"Hash rate fejl: {e}")
            try:
                nvt = self.get_nvt_ratio()
            except Exception as e:
                logger.debug(f"NVT fejl: {e}")
            try:
                whale = self.get_whale_activity()
            except Exception as e:
                logger.debug(f"Whale activity fejl: {e}")

        try:
            defi = self.get_defi_metrics()
        except Exception as e:
            logger.debug(f"DeFi metrics fejl: {e}")

        # Beregn overall signal
        overall, confidence, summary = self._aggregate_signals(
            symbol, fear_greed, exchange_flow, active_addr,
            hash_rate, nvt, whale, defi, dominance,
        )

        return OnChainReport(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            fear_greed=fear_greed,
            exchange_flow=exchange_flow,
            active_addresses=active_addr,
            hash_rate=hash_rate,
            nvt_ratio=nvt,
            whale_activity=whale,
            defi_metrics=defi,
            btc_dominance=dominance,
            overall_signal=overall,
            confidence=confidence,
            summary=summary,
        )

    def _aggregate_signals(
        self,
        symbol: str,
        fear_greed: FearGreedIndex | None,
        exchange_flow: ExchangeFlowData | None,
        active_addr: ActiveAddresses | None,
        hash_rate: HashRateData | None,
        nvt: NVTRatio | None,
        whale: WhaleActivity | None,
        defi: DeFiMetrics | None,
        dominance: BitcoinDominance | None,
    ) -> tuple[OnChainSignal, float, str]:
        """Aggregér alle on-chain signaler til ét samlet signal."""
        bullish = 0
        bearish = 0
        total = 0
        reasons = []

        # Fear & Greed (contrarian – vægt 25%)
        if fear_greed:
            total += 1
            if fear_greed.contrarian_signal == "buy":
                bullish += 0.25
                reasons.append(f"Fear & Greed: {fear_greed.value} (Extreme Fear = contrarian køb)")
            elif fear_greed.contrarian_signal == "sell":
                bearish += 0.25
                reasons.append(f"Fear & Greed: {fear_greed.value} (Extreme Greed = contrarian sælg)")

        # Exchange flow (vægt 20%)
        if exchange_flow:
            total += 1
            if exchange_flow.signal == "bullish":
                bullish += 0.20
            elif exchange_flow.signal == "bearish":
                bearish += 0.20
                reasons.append("Høj exchange inflow (salgspres)")

        # Active addresses (vægt 15%)
        if active_addr:
            total += 1
            if active_addr.signal == "bullish":
                bullish += 0.15
                reasons.append(f"Aktive adresser +{active_addr.change_pct_7d:.1f}% (7d)")
            elif active_addr.signal == "bearish":
                bearish += 0.15

        # Hash rate (vægt 10%)
        if hash_rate:
            total += 1
            if hash_rate.signal == "bullish":
                bullish += 0.10
            elif hash_rate.signal == "bearish":
                bearish += 0.10

        # NVT (vægt 10%)
        if nvt:
            total += 1
            if nvt.signal == "undervalued":
                bullish += 0.10
                reasons.append(f"NVT {nvt.nvt:.0f} – undervalued")
            elif nvt.signal == "overvalued":
                bearish += 0.10
                reasons.append(f"NVT {nvt.nvt:.0f} – overvalued")

        # Whale activity (vægt 10%)
        if whale:
            total += 1
            if whale.whale_sentiment == "accumulating":
                bullish += 0.10
                reasons.append("Whales akkumulerer")
            elif whale.whale_sentiment == "distributing":
                bearish += 0.10

        # DeFi TVL (vægt 5%)
        if defi:
            total += 1
            if defi.signal == "bullish":
                bullish += 0.05
            elif defi.signal == "bearish":
                bearish += 0.05

        # BTC Dominance (vægt 5%)
        if dominance:
            total += 1
            is_alt = symbol != "BTC-USD"
            if dominance.alt_season and is_alt:
                bullish += 0.05
                reasons.append("Alt season (faldende BTC dominance)")
            elif not dominance.alt_season and not is_alt:
                bullish += 0.05

        # Bestem signal
        if total == 0:
            return OnChainSignal.NEUTRAL, 30.0, f"{symbol}: Ingen on-chain data tilgængelig"

        net = bullish - bearish
        if net > 0.3:
            signal = OnChainSignal.STRONG_BULLISH
        elif net > 0.1:
            signal = OnChainSignal.BULLISH
        elif net < -0.3:
            signal = OnChainSignal.STRONG_BEARISH
        elif net < -0.1:
            signal = OnChainSignal.BEARISH
        else:
            signal = OnChainSignal.NEUTRAL

        confidence = min(85, 30 + abs(net) * 150 + total * 3)
        top_reasons = "; ".join(reasons[:3]) if reasons else "Neutral on-chain data"
        summary = f"{symbol}: {signal.value} ({confidence:.0f}%) – {top_reasons}"

        return signal, confidence, summary

    # ── Strategy Integration ─────────────────────────────────

    def get_confidence_adjustment(self, report: OnChainReport) -> int:
        """
        Returnér confidence-justering til strategier (±15 points).

        Bruges til at booste/reducere strategi-confidence for krypto-handler.
        """
        if report.overall_signal in (OnChainSignal.STRONG_BULLISH, OnChainSignal.BULLISH):
            return min(15, int(report.confidence / 6))
        elif report.overall_signal in (OnChainSignal.STRONG_BEARISH, OnChainSignal.BEARISH):
            return max(-15, -int(report.confidence / 6))
        return 0

    def is_crypto(self, symbol: str) -> bool:
        """Tjek om et symbol er en kryptovaluta."""
        return symbol in CRYPTO_IDS or symbol.endswith("-USD")

    # ── Explain / Print ──────────────────────────────────────

    def explain(self, report: OnChainReport) -> str:
        """Generér human-readable forklaring."""
        lines = [
            f"=== On-Chain Analyse: {report.symbol} ===",
            f"Tidspunkt: {report.timestamp[:19]}",
            f"Samlet signal: {report.overall_signal.value} "
            f"(confidence: {report.confidence:.0f}%)",
            f"Sammenfatning: {report.summary}",
            "",
        ]

        if report.fear_greed:
            fg = report.fear_greed
            lines.append("── Fear & Greed Index ──")
            lines.append(f"  Værdi: {fg.value}/100 ({fg.classification})")
            lines.append(f"  Niveau: {fg.level.value}")
            lines.append(f"  Contrarian signal: {fg.contrarian_signal}")
            lines.append("")

        if report.btc_dominance:
            bd = report.btc_dominance
            lines.append("── Bitcoin Dominance ──")
            lines.append(f"  Dominance: {bd.dominance_pct:.1f}%")
            lines.append(f"  Alt season: {'Ja' if bd.alt_season else 'Nej'}")
            lines.append(f"  Signal: {bd.signal}")
            lines.append(f"  {bd.description}")
            lines.append("")

        if report.exchange_flow:
            ef = report.exchange_flow
            lines.append("── Exchange Flow ──")
            lines.append(f"  Inflow: {ef.inflow_btc:,.1f} BTC (estimat)")
            lines.append(f"  Outflow: {ef.outflow_btc:,.1f} BTC (estimat)")
            lines.append(f"  Net flow: {ef.net_flow:+,.1f} BTC")
            lines.append(f"  Signal: {ef.signal}")
            lines.append("")

        if report.active_addresses:
            aa = report.active_addresses
            lines.append("── Active Addresses ──")
            lines.append(f"  Antal: {aa.count:,}")
            lines.append(f"  7d ændring: {aa.change_pct_7d:+.1f}%")
            lines.append(f"  Signal: {aa.signal}")
            lines.append("")

        if report.hash_rate:
            hr = report.hash_rate
            lines.append("── Hash Rate ──")
            lines.append(f"  Hash rate: {hr.hash_rate:,.1f} TH/s")
            lines.append(f"  30d ændring: {hr.change_pct_30d:+.1f}%")
            lines.append(f"  Signal: {hr.signal}")
            lines.append("")

        if report.nvt_ratio:
            nv = report.nvt_ratio
            lines.append("── NVT Ratio ──")
            lines.append(f"  NVT: {nv.nvt:.0f}")
            lines.append(f"  Signal: {nv.signal}")
            lines.append(f"  {nv.description}")
            lines.append("")

        if report.whale_activity:
            wa = report.whale_activity
            lines.append("── Whale Activity ──")
            lines.append(f"  Store transaktioner (24h): ~{wa.large_txs_24h:,}")
            lines.append(f"  Sentiment: {wa.whale_sentiment}")
            lines.append(f"  {wa.description}")
            lines.append("")

        if report.defi_metrics:
            dm = report.defi_metrics
            lines.append("── DeFi Metrics ──")
            lines.append(f"  Total TVL: ${dm.total_tvl_usd / 1e9:,.1f}B")
            lines.append(f"  24h ændring: {dm.tvl_change_24h_pct:+.1f}%")
            lines.append(f"  7d ændring: {dm.tvl_change_7d_pct:+.1f}%")
            if dm.stablecoin_mcap > 0:
                lines.append(f"  Stablecoin MCap: ${dm.stablecoin_mcap / 1e9:,.1f}B")
            if dm.top_protocols:
                lines.append("  Top protocols:")
                for p in dm.top_protocols[:3]:
                    lines.append(f"    {p['name']}: ${p['tvl'] / 1e9:,.1f}B")
            lines.append("")

        return "\n".join(lines)

    def print_report(self, report: OnChainReport) -> None:
        """Print rapport til konsollen."""
        print(self.explain(report))
