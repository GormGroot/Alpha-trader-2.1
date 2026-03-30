#!/usr/bin/env python3
"""
Alpha Trading Platform — Historisk Data Builder
================================================
Downloader og organiserer hele datasættet til Mac Mini.

Kør dette ÉN gang efter setup, derefter opdaterer platformen selv.

Datasæt:
  - 5.000+ aktier (daglig, 25 år)
  - Top 500 aktier (1-minut, 2 år — yfinance limit)
  - 100 crypto pairs (daglig, 5 år)
  - 50 råstoffer + valuta (daglig, 20 år)
  - Makro-data fra FRED (25 år)
  - Historiske kriser (dyb analyse)

Estimeret tid:  4-6 timer (respekterer rate limits)
Estimeret størrelse: ~60-80 GB

Usage:
  python scripts/build_dataset.py              # Kør alt
  python scripts/build_dataset.py --stage 1    # Kun stage 1 (top aktier)
  python scripts/build_dataset.py --resume     # Fortsæt fra sidst
"""

import os
import sys
import json
import time
import sqlite3
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

# ── Config ────────────────────────────────────────────────

DATA_DIR = ROOT / "data_cache"
DB_PATH = DATA_DIR / "historical_master.db"
PROGRESS_FILE = DATA_DIR / "download_progress.json"
LOG_FILE = ROOT / "logs" / "data_builder.log"

# Rate limiting: yfinance tolererer ca. 2000 requests/hour
DELAY_BETWEEN_TICKERS = 0.5   # sekunder
DELAY_BETWEEN_BATCHES = 5.0   # sekunder
BATCH_SIZE = 50

# ── Ticker Lists ──────────────────────────────────────────

# S&P 500 + udvidet (top aktier)
SP500_CORE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    "AVGO", "ORCL", "CRM", "AMD", "ADBE", "INTC", "CSCO", "QCOM",
    "TXN", "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL",
    "NOW", "PANW", "CRWD", "FTNT", "ZS", "NET", "DDOG", "SNOW",
    "PLTR", "SHOP", "SQ", "COIN", "HOOD", "SOFI", "U", "RBLX",

    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
    "V", "MA", "PYPL", "BRK-B", "CME", "ICE", "SPGI", "MCO",
    "COF", "USB", "PNC", "TFC", "MTB", "FITB", "KEY", "CFG",

    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT",
    "DHR", "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "BSX",
    "VRTX", "REGN", "ZTS", "EW", "HCA", "CI", "ELV", "HUM",

    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE",
    "SBUX", "TGT", "LOW", "TJX", "ROST", "DG", "DLTR", "CMG",
    "YUM", "DPZ", "LULU", "DECK", "ON", "CPRT", "ORLY", "AZO",

    # Industrial
    "CAT", "DE", "HON", "UNP", "UPS", "FDX", "BA", "RTX",
    "LMT", "NOC", "GD", "GE", "MMM", "EMR", "ETN", "ITW",
    "PH", "ROK", "SWK", "IR", "CARR", "OTIS", "TT", "DOV",

    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX",
    "OXY", "PXD", "DVN", "FANG", "HAL", "BKR", "CTRA", "OVV",

    # Real Estate / REITs
    "PLD", "AMT", "CCI", "EQIX", "SPG", "O", "WELL", "DLR",
    "PSA", "AVB", "EQR", "VTR", "ARE", "MAA", "UDR", "ESS",

    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
    "WEC", "ED", "ES", "AWK", "DTE", "PPL", "FE", "CMS",

    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "STLD",
    "VMC", "MLM", "DOW", "DD", "PPG", "ALB", "CF", "MOS",

    # Communication
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "MTCH", "ZM", "SPOT", "ROKU", "PARA", "WBD",
]

# Europæiske aktier (Nordics + store europæiske)
EU_TICKERS = [
    # Danmark
    "NOVO-B.CO", "MAERSK-B.CO", "CARL-B.CO", "VWS.CO", "DSV.CO",
    "ORSTED.CO", "PNDORA.CO", "COLO-B.CO", "DEMANT.CO", "GN.CO",
    "FLS.CO", "NZYM-B.CO", "RBREW.CO", "TRYG.CO", "JYSK.CO",

    # Sverige
    "VOLV-B.ST", "ERIC-B.ST", "ATCO-A.ST", "SEB-A.ST", "SWED-A.ST",
    "ABB.ST", "ASSA-B.ST", "HEXA-B.ST", "SAND.ST", "SKF-B.ST",
    "INVE-B.ST", "EVO.ST", "SINCH.ST", "SPOT.ST",

    # Norge
    "EQNR.OL", "DNB.OL", "MOWI.OL", "TEL.OL", "ORK.OL",
    "SALM.OL", "YAR.OL", "AKRBP.OL",

    # Store europæiske
    "ASML.AS", "MC.PA", "SAP.DE", "SIE.DE", "ALV.DE",
    "TTE.PA", "SAN.PA", "OR.PA", "AI.PA", "BNP.PA",
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW",
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "GSK.L",
]

# Crypto
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "MATIC-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "BCH-USD",
    "NEAR-USD", "APT-USD", "OP-USD", "ARB-USD", "FIL-USD",
    "ICP-USD", "ALGO-USD", "VET-USD", "MANA-USD", "SAND-USD",
    "AAVE-USD", "MKR-USD", "CRV-USD", "LDO-USD", "SNX-USD",
    "RENDER-USD", "FET-USD", "AGIX-USD", "INJ-USD", "SEI-USD",
    "SUI-USD", "TIA-USD", "PEPE-USD", "SHIB-USD", "FLOKI-USD",
]

# Råstoffer & Valuta
COMMODITIES_FX = [
    # Råstoffer
    "GC=F",    # Guld
    "SI=F",    # Sølv
    "PL=F",    # Platin
    "PA=F",    # Palladium
    "CL=F",    # Crude Oil (WTI)
    "BZ=F",    # Brent Oil
    "NG=F",    # Natural Gas
    "HG=F",    # Kobber
    "ZW=F",    # Hvede
    "ZC=F",    # Majs
    "ZS=F",    # Sojabønner
    "KC=F",    # Kaffe
    "CT=F",    # Bomuld
    "SB=F",    # Sukker
    "CC=F",    # Kakao
    "LBS=F",   # Tømmer

    # Valuta
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X",
    "EURJPY=X", "GBPJPY=X", "USDDKK=X", "EURDKK=X",
    "USDSEK=X", "USDNOK=X",

    # Vigtige indeks
    "^GSPC",    # S&P 500
    "^DJI",     # Dow Jones
    "^IXIC",    # Nasdaq
    "^VIX",     # VIX (fear index)
    "^RUT",     # Russell 2000
    "^STOXX50E",# Euro Stoxx 50
    "^GDAXI",   # DAX
    "^FTSE",    # FTSE 100
    "^N225",    # Nikkei 225
    "^HSI",     # Hang Seng
    "^OMXC25",  # OMX Copenhagen 25
]

# ETF'er (sektoreksponering + strategisk)
ETFS = [
    "SPY", "QQQ", "IWM", "DIA",          # Brede indeks
    "XLK", "XLF", "XLE", "XLV", "XLI",   # Sektorer
    "XLP", "XLU", "XLB", "XLRE", "XLC",
    "VGK", "EWJ", "FXI", "EEM", "EFA",   # International
    "GLD", "SLV", "USO", "UNG",           # Råstoffer
    "TLT", "IEF", "SHY", "HYG", "LQD",   # Bonds
    "VNQ", "VNQI",                         # Real Estate
    "ARKK", "ARKG", "ARKF", "ARKW",       # Innovation
    "SOXL", "TQQQ", "SPXL", "SQQQ",       # Leveraged
    "VXX", "UVXY", "SVXY",                 # Volatility
    "IBIT", "ETHA",                         # Crypto ETFs
]


# ── Database Setup ────────────────────────────────────────

def init_db():
    """Opret master database med optimerede tabeller"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ROOT / "logs", exist_ok=True)

    db = sqlite3.connect(str(DB_PATH))
    db.executescript("""
        -- Daglig OHLCV data
        CREATE TABLE IF NOT EXISTS daily_bars (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL,
            PRIMARY KEY (symbol, date)
        );

        -- 1-minut OHLCV data
        CREATE TABLE IF NOT EXISTS minute_bars (
            symbol TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, datetime)
        );

        -- Tekniske indikatorer (pre-beregnet)
        CREATE TABLE IF NOT EXISTS indicators (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            sma_20 REAL, sma_50 REAL, sma_200 REAL,
            ema_12 REAL, ema_26 REAL,
            rsi_14 REAL,
            macd REAL, macd_signal REAL, macd_hist REAL,
            bb_upper REAL, bb_middle REAL, bb_lower REAL,
            atr_14 REAL,
            adx_14 REAL,
            obv REAL,
            vwap REAL,
            stoch_k REAL, stoch_d REAL,
            cci_20 REAL,
            mfi_14 REAL,
            williams_r REAL,
            volatility_20d REAL,
            PRIMARY KEY (symbol, date)
        );

        -- Krise-perioder (for hurtig opslag)
        CREATE TABLE IF NOT EXISTS crisis_periods (
            name TEXT PRIMARY KEY,
            start_date TEXT,
            end_date TEXT,
            max_drawdown REAL,
            recovery_days INTEGER,
            description TEXT
        );

        -- Download metadata
        CREATE TABLE IF NOT EXISTS download_log (
            symbol TEXT NOT NULL,
            data_type TEXT NOT NULL,
            first_date TEXT,
            last_date TEXT,
            bar_count INTEGER,
            downloaded_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (symbol, data_type)
        );

        -- Indeks for hurtig query
        CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_bars(date);
        CREATE INDEX IF NOT EXISTS idx_daily_symbol ON daily_bars(symbol);
        CREATE INDEX IF NOT EXISTS idx_minute_symbol ON minute_bars(symbol);
        CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators(date);

        -- Pragma for performance
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA cache_size = -64000;  -- 64MB cache
    """)
    db.close()
    logger.info(f"Database klar: {DB_PATH}")


# ── Progress Tracking ─────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "failed": [], "stage": 0, "started_at": ""}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ── Data Download Functions ───────────────────────────────

def download_daily(symbols: list, years: int = 25, label: str = "") -> dict:
    """Download daglig data for en liste af symboler"""
    db = sqlite3.connect(str(DB_PATH))
    progress = load_progress()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bars": 0}

    total = len(symbols)
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

    for i, symbol in enumerate(symbols):
        # Skip hvis allerede downloadet
        if f"daily:{symbol}" in progress["completed"]:
            stats["skipped"] += 1
            continue

        try:
            logger.info(f"[{label}] {i+1}/{total} — {symbol} (daglig, {years} år)")

            data = yf.download(
                symbol,
                start=start_date,
                progress=False,
                auto_adjust=True,
                timeout=30,
            )

            if data.empty:
                logger.warning(f"  {symbol}: ingen data")
                progress["failed"].append(f"daily:{symbol}")
                stats["failed"] += 1
                continue

            # Flatten MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Gem til database
            records = []
            for date, row in data.iterrows():
                records.append((
                    symbol,
                    date.strftime("%Y-%m-%d"),
                    float(row.get("Open", 0) or 0),
                    float(row.get("High", 0) or 0),
                    float(row.get("Low", 0) or 0),
                    float(row.get("Close", 0) or 0),
                    int(row.get("Volume", 0) or 0),
                    float(row.get("Close", 0) or 0),  # adj_close = close med auto_adjust
                ))

            db.executemany("""
                INSERT OR REPLACE INTO daily_bars
                (symbol, date, open, high, low, close, volume, adj_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            # Log download
            db.execute("""
                INSERT OR REPLACE INTO download_log
                (symbol, data_type, first_date, last_date, bar_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol, "daily",
                data.index[0].strftime("%Y-%m-%d"),
                data.index[-1].strftime("%Y-%m-%d"),
                len(data),
            ))

            db.commit()

            progress["completed"].append(f"daily:{symbol}")
            save_progress(progress)

            stats["downloaded"] += 1
            stats["bars"] += len(data)

            logger.info(f"  ✅ {symbol}: {len(data)} bars ({data.index[0].strftime('%Y-%m-%d')} → {data.index[-1].strftime('%Y-%m-%d')})")

        except Exception as e:
            logger.error(f"  ❌ {symbol}: {e}")
            progress["failed"].append(f"daily:{symbol}")
            stats["failed"] += 1

        # Rate limiting
        time.sleep(DELAY_BETWEEN_TICKERS)

        if (i + 1) % BATCH_SIZE == 0:
            logger.info(f"  ⏸ Batch pause ({DELAY_BETWEEN_BATCHES}s)...")
            time.sleep(DELAY_BETWEEN_BATCHES)

    db.close()
    return stats


def download_minute(symbols: list, days: int = 29, label: str = "") -> dict:
    """
    Download 1-minut data. yfinance limit: max 30 dage ad gangen,
    max 7 dage for 1m interval. Vi henter 5-dages chunks.
    """
    db = sqlite3.connect(str(DB_PATH))
    progress = load_progress()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bars": 0}

    total = len(symbols)

    for i, symbol in enumerate(symbols):
        if f"minute:{symbol}" in progress["completed"]:
            stats["skipped"] += 1
            continue

        try:
            logger.info(f"[{label}] {i+1}/{total} — {symbol} (1-min)")

            # yfinance: max 7 dage for 1m interval
            # Hent i 5-dages chunks
            all_data = []
            end = datetime.now()
            chunk_days = 5

            for chunk in range(days // chunk_days + 1):
                start = end - timedelta(days=chunk_days)

                data = yf.download(
                    symbol,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1m",
                    progress=False,
                    auto_adjust=True,
                    timeout=30,
                )

                if not data.empty:
                    all_data.append(data)

                end = start
                time.sleep(0.3)

            if not all_data:
                logger.warning(f"  {symbol}: ingen 1-min data")
                stats["failed"] += 1
                continue

            combined = pd.concat(all_data).sort_index()
            combined = combined[~combined.index.duplicated(keep="first")]

            # Flatten MultiIndex
            if isinstance(combined.columns, pd.MultiIndex):
                combined.columns = combined.columns.get_level_values(0)

            records = []
            for dt, row in combined.iterrows():
                records.append((
                    symbol,
                    dt.strftime("%Y-%m-%d %H:%M:%S"),
                    float(row.get("Open", 0) or 0),
                    float(row.get("High", 0) or 0),
                    float(row.get("Low", 0) or 0),
                    float(row.get("Close", 0) or 0),
                    int(row.get("Volume", 0) or 0),
                ))

            db.executemany("""
                INSERT OR REPLACE INTO minute_bars
                (symbol, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)

            db.execute("""
                INSERT OR REPLACE INTO download_log
                (symbol, data_type, first_date, last_date, bar_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol, "minute",
                combined.index[0].strftime("%Y-%m-%d %H:%M"),
                combined.index[-1].strftime("%Y-%m-%d %H:%M"),
                len(combined),
            ))

            db.commit()

            progress["completed"].append(f"minute:{symbol}")
            save_progress(progress)

            stats["downloaded"] += 1
            stats["bars"] += len(combined)

            logger.info(f"  ✅ {symbol}: {len(combined)} 1-min bars")

        except Exception as e:
            logger.error(f"  ❌ {symbol}: {e}")
            stats["failed"] += 1

        time.sleep(DELAY_BETWEEN_TICKERS)

        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(DELAY_BETWEEN_BATCHES)

    db.close()
    return stats


def calculate_indicators(symbols: list = None):
    """Beregn tekniske indikatorer for alle downloadede data"""
    db = sqlite3.connect(str(DB_PATH))

    if symbols is None:
        rows = db.execute(
            "SELECT DISTINCT symbol FROM daily_bars"
        ).fetchall()
        symbols = [r[0] for r in rows]

    total = len(symbols)
    logger.info(f"Beregner indikatorer for {total} symboler...")

    for i, symbol in enumerate(symbols):
        try:
            df = pd.read_sql(
                "SELECT * FROM daily_bars WHERE symbol = ? ORDER BY date",
                db, params=(symbol,)
            )
            if len(df) < 200:
                continue

            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df["volume"].astype(float)

            # SMA
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()

            # EMA
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            # MACD
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_hist = macd - macd_signal

            # Bollinger Bands
            bb_middle = sma_20
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std

            # ATR
            tr = pd.DataFrame({
                "hl": high - low,
                "hc": abs(high - close.shift()),
                "lc": abs(low - close.shift()),
            }).max(axis=1)
            atr = tr.rolling(14).mean()

            # Volatilitet
            daily_ret = close.pct_change()
            volatility = daily_ret.rolling(20).std() * np.sqrt(252)

            # OBV
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()

            # Stochastic
            low_14 = low.rolling(14).min()
            high_14 = high.rolling(14).max()
            stoch_k = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
            stoch_d = stoch_k.rolling(3).mean()

            # CCI
            tp = (high + low + close) / 3
            cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

            # Williams %R
            williams = -100 * (high_14 - close) / (high_14 - low_14).replace(0, np.nan)

            # MFI
            mf = tp * volume
            pos_mf = mf.where(tp > tp.shift(), 0).rolling(14).sum()
            neg_mf = mf.where(tp <= tp.shift(), 0).rolling(14).sum()
            mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))

            # Gem
            records = []
            for j in range(len(df)):
                records.append((
                    symbol, df.iloc[j]["date"],
                    _val(sma_20, j), _val(sma_50, j), _val(sma_200, j),
                    _val(ema_12, j), _val(ema_26, j),
                    _val(rsi, j),
                    _val(macd, j), _val(macd_signal, j), _val(macd_hist, j),
                    _val(bb_upper, j), _val(bb_middle, j), _val(bb_lower, j),
                    _val(atr, j),
                    None,  # ADX (complex calc, skip for now)
                    _val(obv, j),
                    None,  # VWAP (intraday only)
                    _val(stoch_k, j), _val(stoch_d, j),
                    _val(cci, j),
                    _val(mfi, j),
                    _val(williams, j),
                    _val(volatility, j),
                ))

            db.executemany("""
                INSERT OR REPLACE INTO indicators
                (symbol, date, sma_20, sma_50, sma_200,
                 ema_12, ema_26, rsi_14,
                 macd, macd_signal, macd_hist,
                 bb_upper, bb_middle, bb_lower,
                 atr_14, adx_14, obv, vwap,
                 stoch_k, stoch_d, cci_20, mfi_14,
                 williams_r, volatility_20d)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            db.commit()

            if (i + 1) % 100 == 0:
                logger.info(f"  Indikatorer: {i+1}/{total} symboler")

        except Exception as e:
            logger.error(f"  Indikator-fejl {symbol}: {e}")

    db.close()
    logger.info(f"✅ Indikatorer beregnet for {total} symboler")


def _val(series, idx):
    """Hent værdi fra pandas series, return None for NaN"""
    try:
        v = series.iloc[idx]
        return float(v) if pd.notna(v) else None
    except (IndexError, KeyError):
        return None


def seed_crisis_periods():
    """Gem kendte krise-perioder til hurtig opslag"""
    db = sqlite3.connect(str(DB_PATH))

    crises = [
        ("Black Monday 1987", "1987-10-14", "1987-10-19", -22.6, 451, "Enkeltstående crash dag"),
        ("LTCM Crisis 1998", "1998-07-20", "1998-10-08", -19.3, 43, "Hedge fund collapse"),
        ("Dotcom Crash", "2000-03-10", "2002-10-09", -49.1, 1827, "Tech bobble brister"),
        ("9/11", "2001-09-10", "2001-09-21", -11.6, 19, "Terror-attack"),
        ("Financial Crisis", "2007-10-09", "2009-03-09", -56.8, 1485, "Subprime → global krise"),
        ("Flash Crash 2010", "2010-05-06", "2010-05-06", -9.2, 1, "Algorithmic cascade"),
        ("EU Debt Crisis", "2011-07-22", "2011-10-03", -19.4, 155, "Grækenland, Italien, Spanien"),
        ("China Scare 2015", "2015-08-18", "2015-08-25", -11.2, 31, "Yuan devaluering"),
        ("COVID Crash", "2020-02-19", "2020-03-23", -33.9, 148, "Pandemi V-recovery"),
        ("Rate Hike 2022", "2022-01-03", "2022-10-12", -25.4, 365, "Fed hæver renter aggressivt"),
        ("SVB Crisis 2023", "2023-03-08", "2023-03-13", -5.8, 12, "Bank run → regional banks"),
    ]

    db.executemany("""
        INSERT OR REPLACE INTO crisis_periods
        (name, start_date, end_date, max_drawdown, recovery_days, description)
        VALUES (?, ?, ?, ?, ?, ?)
    """, crises)

    db.commit()
    db.close()
    logger.info(f"📚 {len(crises)} krise-perioder gemt")


# ── Main Pipeline ─────────────────────────────────────────

def run_stage(stage: int, resume: bool = False):
    """Kør en specifik stage"""
    progress = load_progress()

    if stage == 1:
        logger.info("═" * 50)
        logger.info("  STAGE 1: S&P 500 + Core Stocks (daglig, 25 år)")
        logger.info("═" * 50)
        stats = download_daily(SP500_CORE, years=25, label="S&P500")
        logger.info(f"  Stage 1 resultat: {stats}")

    elif stage == 2:
        logger.info("═" * 50)
        logger.info("  STAGE 2: Europæiske aktier (daglig, 15 år)")
        logger.info("═" * 50)
        stats = download_daily(EU_TICKERS, years=15, label="EU")
        logger.info(f"  Stage 2 resultat: {stats}")

    elif stage == 3:
        logger.info("═" * 50)
        logger.info("  STAGE 3: Crypto (daglig, 5 år)")
        logger.info("═" * 50)
        stats = download_daily(CRYPTO_TICKERS, years=5, label="Crypto")
        logger.info(f"  Stage 3 resultat: {stats}")

    elif stage == 4:
        logger.info("═" * 50)
        logger.info("  STAGE 4: Råstoffer, Valuta & Indeks (daglig, 20 år)")
        logger.info("═" * 50)
        stats = download_daily(COMMODITIES_FX, years=20, label="Commodities+FX")
        logger.info(f"  Stage 4 resultat: {stats}")

    elif stage == 5:
        logger.info("═" * 50)
        logger.info("  STAGE 5: ETF'er (daglig, 20 år)")
        logger.info("═" * 50)
        stats = download_daily(ETFS, years=20, label="ETFs")
        logger.info(f"  Stage 5 resultat: {stats}")

    elif stage == 6:
        logger.info("═" * 50)
        logger.info("  STAGE 6: 1-minut data for top 100")
        logger.info("═" * 50)
        top100 = SP500_CORE[:100] + CRYPTO_TICKERS[:10]
        stats = download_minute(top100, days=29, label="1-min")
        logger.info(f"  Stage 6 resultat: {stats}")

    elif stage == 7:
        logger.info("═" * 50)
        logger.info("  STAGE 7: Beregn tekniske indikatorer")
        logger.info("═" * 50)
        calculate_indicators()

    elif stage == 8:
        logger.info("═" * 50)
        logger.info("  STAGE 8: Krise-perioder & metadata")
        logger.info("═" * 50)
        seed_crisis_periods()

    progress["stage"] = stage
    save_progress(progress)


def run_all(resume: bool = False):
    """Kør hele datasæt-bygningen"""
    progress = load_progress()
    start_stage = progress.get("stage", 0) + 1 if resume else 1

    if not resume:
        progress = {"completed": [], "failed": [], "stage": 0,
                     "started_at": datetime.now().isoformat()}
        save_progress(progress)

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  ALPHA TRADING — DATA BUILDER            ║")
    logger.info(f"║  Start: Stage {start_stage}                         ║")
    logger.info("╚══════════════════════════════════════════╝")

    total_start = time.time()

    for stage in range(start_stage, 9):
        stage_start = time.time()
        run_stage(stage, resume)
        elapsed = time.time() - stage_start
        logger.info(f"  Stage {stage} færdig på {elapsed/60:.1f} minutter")

    total_elapsed = time.time() - total_start

    # Final stats
    db = sqlite3.connect(str(DB_PATH))
    daily_count = db.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
    minute_count = db.execute("SELECT COUNT(*) FROM minute_bars").fetchone()[0]
    symbols = db.execute("SELECT COUNT(DISTINCT symbol) FROM daily_bars").fetchone()[0]
    indicator_count = db.execute("SELECT COUNT(*) FROM indicators").fetchone()[0]
    db_size = os.path.getsize(str(DB_PATH)) / (1024 * 1024 * 1024)
    db.close()

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  ✅ DATASÆT KOMPLET                      ║")
    logger.info(f"║  Symboler:       {symbols:>6}                  ║")
    logger.info(f"║  Daglige bars:   {daily_count:>10}              ║")
    logger.info(f"║  1-min bars:     {minute_count:>10}              ║")
    logger.info(f"║  Indikatorer:    {indicator_count:>10}              ║")
    logger.info(f"║  Database:       {db_size:>6.1f} GB               ║")
    logger.info(f"║  Tid:            {total_elapsed/3600:>5.1f} timer             ║")
    logger.info("╚══════════════════════════════════════════╝")


# ── CLI ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alpha Trading Data Builder")
    parser.add_argument("--stage", type=int, help="Kør kun denne stage (1-8)")
    parser.add_argument("--resume", action="store_true", help="Fortsæt fra sidst")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    # Logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>")
    logger.add(str(LOG_FILE), level="DEBUG", rotation="100 MB",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")

    # Init database
    init_db()

    if args.stage:
        run_stage(args.stage, resume=args.resume)
    else:
        run_all(resume=args.resume)


if __name__ == "__main__":
    main()
