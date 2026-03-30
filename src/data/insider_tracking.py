"""
Insider Tracking Module – SEC EDGAR integration for smart money signals.

Funktionalitet:
  - Form 4 filings: insider-handler (CEO, CFO, bestyrelse)
  - 13F filings: institutionel ejerskab (store hedgefonde)
  - Short interest tracking: squeeze potential, days to cover
  - Insider sentiment score per aktie
  - Cluster-buying detektion (flere insidere køber samtidig)
  - SQLite-cache for alle data

SEC EDGAR API:
  - Gratis, kræver User-Agent header
  - Rate limit: 10 requests/sekund
  - Base URL: https://efts.sec.gov/LATEST/ (full-text search)
  - Company search: https://efts.sec.gov/LATEST/search-index?q=...
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from config.settings import settings

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    logger.warning("requests ikke installeret – SEC EDGAR API utilgængelig")


# ── Konstanter ───────────────────────────────────────────────

SEC_EDGAR_BASE = "https://efts.sec.gov/LATEST"
SEC_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
SEC_EDGAR_COMPANY = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_EDGAR_FILINGS = "https://efts.sec.gov/LATEST/search-index"
SEC_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"

# User-Agent krævet af SEC (brug firmanavn + email)
SEC_USER_AGENT = "AlphaTradingPlatform/1.0 (https://github.com/alpha-trading-platform)"

# Rate limiting: SEC tillader 10 req/s, vi bruger 0.12s gap
_MIN_SEC_REQUEST_GAP = 0.12

# Kendte store hedgefonde CIK-numre (13F filings)
MAJOR_FUNDS: dict[str, str] = {
    "Berkshire Hathaway": "0001067983",
    "Bridgewater Associates": "0001350694",
    "Renaissance Technologies": "0001037389",
    "Citadel Advisors": "0001423053",
    "Two Sigma": "0001179392",
    "DE Shaw": "0001009207",
    "BlackRock": "0001364742",
    "Vanguard": "0000102909",
    "ARK Invest": "0001603466",
    "Soros Fund Management": "0001029160",
}

# Insider roller der er vigtige
INSIDER_ROLES = {
    "CEO", "CFO", "COO", "CTO", "President", "Chairman",
    "Director", "VP", "EVP", "SVP", "Officer", "10% Owner",
}


# ── Enums ────────────────────────────────────────────────────

class TransactionType(Enum):
    """Type insider-transaktion."""
    PURCHASE = "P"       # Køb på åbent marked
    SALE = "S"           # Salg på åbent marked
    GRANT = "A"          # Award/Grant
    EXERCISE = "M"       # Option exercise
    CONVERSION = "C"     # Konvertering
    OTHER = "X"


class InsiderSentiment(Enum):
    """Samlet insider-sentiment for en aktie."""
    VERY_BULLISH = "very_bullish"   # Cluster buying
    BULLISH = "bullish"             # Netto køb
    NEUTRAL = "neutral"             # Ingen aktivitet / blandet
    BEARISH = "bearish"             # Netto salg
    VERY_BEARISH = "very_bearish"   # Cluster selling


# ── Dataclasses ──────────────────────────────────────────────

@dataclass
class InsiderTrade:
    """Én insider-transaktion fra Form 4."""
    symbol: str
    insider_name: str
    insider_title: str
    transaction_type: TransactionType
    shares: float
    price: float
    value: float
    date: datetime
    filing_date: datetime
    ownership_after: float = 0.0
    is_direct: bool = True

    @property
    def is_purchase(self) -> bool:
        return self.transaction_type == TransactionType.PURCHASE

    @property
    def is_sale(self) -> bool:
        return self.transaction_type == TransactionType.SALE

    @property
    def is_c_suite(self) -> bool:
        import re
        title_upper = self.insider_title.upper()
        # Brug word boundary for at undgå falske matches (f.eks. "DIRECTOR" → "CTO")
        return any(
            re.search(rf"\b{r.upper()}\b", title_upper)
            for r in ("CEO", "CFO", "COO", "CTO", "PRESIDENT",
                       "CHIEF EXECUTIVE", "CHIEF FINANCIAL", "CHIEF OPERATING")
        )


@dataclass
class InsiderSentimentScore:
    """Samlet insider-sentiment for en aktie."""
    symbol: str
    sentiment: InsiderSentiment
    score: float               # -100 til +100
    net_purchases: int         # Antal netto køb
    net_sales: int             # Antal netto salg
    total_buy_value: float     # Samlet købs-værdi USD
    total_sell_value: float    # Samlet salgs-værdi USD
    cluster_buying: bool       # ≥3 insidere køber inden for 14 dage
    cluster_selling: bool      # ≥3 insidere sælger inden for 14 dage
    c_suite_buying: bool       # CEO/CFO/COO køber
    period_days: int = 90      # Analyse-periode
    last_trade_date: datetime | None = None
    trades: list[InsiderTrade] = field(default_factory=list)

    @property
    def confidence_boost(self) -> int:
        """Confidence boost for strategier (−15 til +15 points)."""
        boost = 0
        if self.cluster_buying:
            boost += 10
        if self.c_suite_buying:
            boost += 5
        if self.sentiment == InsiderSentiment.VERY_BULLISH:
            boost = max(boost, 15)
        elif self.sentiment == InsiderSentiment.BULLISH:
            boost = max(boost, 8)
        elif self.sentiment == InsiderSentiment.BEARISH:
            boost = min(boost, -8)
        elif self.sentiment == InsiderSentiment.VERY_BEARISH:
            boost = min(boost, -15)
        return max(-15, min(15, boost))


@dataclass
class InstitutionalHolding:
    """Én institutionel holding fra 13F filing."""
    fund_name: str
    symbol: str
    shares: int
    value_usd: float          # Markedsværdi
    pct_portfolio: float      # % af fondens portefølje
    quarter: str              # "2025Q4"
    change_shares: int = 0    # Ændring fra forrige kvartal
    change_pct: float = 0.0   # Procentuel ændring
    is_new_position: bool = False
    is_closed: bool = False


@dataclass
class SmartMoneyFlow:
    """Aggregeret smart money flow for en aktie."""
    symbol: str
    institutional_holders: int         # Antal store fonde
    total_institutional_value: float   # Samlet inst. værdi
    net_institutional_change: float    # Netto ændring i værdi
    new_positions: list[str]           # Fonde der har købt ind
    closed_positions: list[str]        # Fonde der har solgt ud
    increased: list[str]              # Fonde der har øget position
    decreased: list[str]              # Fonde der har reduceret
    top_holders: list[InstitutionalHolding] = field(default_factory=list)


@dataclass
class ShortInterestData:
    """Short interest data for en aktie."""
    symbol: str
    short_interest: int            # Antal shortede aktier
    short_pct_float: float         # % af float
    short_ratio: float             # Days to cover
    avg_volume: float              # Gennemsnitligt dagligt volumen
    previous_short_interest: int = 0
    change_pct: float = 0.0        # Ændring fra forrige rapport
    squeeze_potential: str = "low"  # "low", "medium", "high", "extreme"
    date: datetime | None = None

    @property
    def is_heavily_shorted(self) -> bool:
        return self.short_pct_float > 20.0

    @property
    def days_to_cover(self) -> float:
        if self.avg_volume <= 0:
            return 0.0
        return self.short_interest / self.avg_volume


@dataclass
class SmartMoneyReport:
    """Samlet smart money rapport for en aktie."""
    symbol: str
    insider_sentiment: InsiderSentimentScore | None
    smart_money_flow: SmartMoneyFlow | None
    short_interest: ShortInterestData | None
    overall_signal: str = "neutral"    # "bullish", "bearish", "neutral"
    confidence_adjustment: int = 0     # Samlet confidence boost/reduction
    warnings: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


# ── SEC EDGAR Client ─────────────────────────────────────────

class SECEdgarClient:
    """
    Klient til SEC EDGAR API.

    Henter Form 4 (insider trades) og 13F (institutional holdings)
    med rate limiting og caching.
    """

    def __init__(self, cache_dir: str | None = None, user_agent: str | None = None) -> None:
        self._last_request_time: float = 0.0
        self._user_agent = user_agent or SEC_USER_AGENT
        self._session: requests.Session | None = None

        # SQLite cache
        cache_path = Path(cache_dir or settings.market_data.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_path / "insider_tracking.db"
        self._init_db()

    def _get_session(self) -> requests.Session:
        """Lazy-init HTTP session med korrekt User-Agent."""
        if not _HAS_REQUESTS:
            raise RuntimeError("requests-biblioteket er ikke installeret")
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": self._user_agent,
                "Accept": "application/json",
            })
        return self._session

    def _throttle(self) -> None:
        """Rate limiting – max 10 req/s til SEC."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_SEC_REQUEST_GAP:
            time.sleep(_MIN_SEC_REQUEST_GAP - elapsed)
        self._last_request_time = time.monotonic()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        """Opret cache-tabeller."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insider_trades (
                    symbol          TEXT NOT NULL,
                    insider_name    TEXT NOT NULL,
                    insider_title   TEXT,
                    transaction_type TEXT,
                    shares          REAL,
                    price           REAL,
                    value           REAL,
                    trade_date      TEXT,
                    filing_date     TEXT,
                    ownership_after REAL,
                    is_direct       INTEGER DEFAULT 1,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (symbol, insider_name, trade_date, transaction_type)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS institutional_holdings (
                    fund_name       TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    shares          INTEGER,
                    value_usd       REAL,
                    pct_portfolio   REAL,
                    quarter         TEXT NOT NULL,
                    change_shares   INTEGER DEFAULT 0,
                    change_pct      REAL DEFAULT 0,
                    is_new_position INTEGER DEFAULT 0,
                    is_closed       INTEGER DEFAULT 0,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (fund_name, symbol, quarter)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS short_interest (
                    symbol              TEXT NOT NULL,
                    short_interest      INTEGER,
                    short_pct_float     REAL,
                    short_ratio         REAL,
                    avg_volume          REAL,
                    previous_short      INTEGER DEFAULT 0,
                    change_pct          REAL DEFAULT 0,
                    squeeze_potential   TEXT DEFAULT 'low',
                    report_date         TEXT,
                    fetched_at          TEXT NOT NULL,
                    PRIMARY KEY (symbol, report_date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cik_lookup (
                    symbol  TEXT PRIMARY KEY,
                    cik     TEXT NOT NULL,
                    name    TEXT,
                    fetched_at TEXT NOT NULL
                )
            """)

    # ── CIK Lookup ────────────────────────────────────────────

    def _get_cik(self, symbol: str) -> str | None:
        """
        Slå CIK-nummer op for et ticker-symbol.

        Tjekker cache først, derefter SEC EDGAR.
        """
        # Cache check
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT cik FROM cik_lookup WHERE symbol = ?", (symbol.upper(),)
            ).fetchone()
            if row:
                return row[0]

        # Hent fra SEC
        try:
            self._throttle()
            session = self._get_session()
            url = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "company": symbol,
                "CIK": symbol,
                "type": "4",
                "dateb": "",
                "owner": "include",
                "count": "1",
                "search_text": "",
                "output": "atom",
            }
            resp = session.get(url, params=params, timeout=15)

            # Alternativ: brug company tickers JSON
            if resp.status_code != 200:
                return self._lookup_cik_from_tickers(symbol)

            # Parse CIK fra response
            cik = self._extract_cik_from_response(resp.text, symbol)
            if cik:
                self._cache_cik(symbol, cik)
            return cik

        except Exception as exc:
            logger.debug(f"CIK lookup fejl for {symbol}: {exc}")
            return self._lookup_cik_from_tickers(symbol)

    def _lookup_cik_from_tickers(self, symbol: str) -> str | None:
        """Alternativ CIK-lookup via SEC company_tickers.json."""
        try:
            self._throttle()
            session = self._get_session()
            resp = session.get(
                "https://www.sec.gov/files/company_tickers.json",
                timeout=15,
            )
            if resp.status_code != 200:
                return None

            tickers = resp.json()
            for _key, entry in tickers.items():
                if entry.get("ticker", "").upper() == symbol.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    name = entry.get("title", "")
                    self._cache_cik(symbol, cik, name)
                    return cik
            return None

        except Exception as exc:
            logger.debug(f"Tickers lookup fejl: {exc}")
            return None

    def _extract_cik_from_response(self, text: str, symbol: str) -> str | None:
        """Ekstraher CIK fra SEC EDGAR Atom response."""
        import re
        match = re.search(r"CIK=(\d{10})", text)
        if match:
            return match.group(1)
        # Prøv alternativt format
        match = re.search(r"/cgi-bin/browse-edgar\?action=getcompany&CIK=(\d+)", text)
        if match:
            return match.group(1).zfill(10)
        return None

    def _cache_cik(self, symbol: str, cik: str, name: str = "") -> None:
        """Gem CIK i cache."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cik_lookup (symbol, cik, name, fetched_at)
                   VALUES (?, ?, ?, ?)""",
                (symbol.upper(), cik, name, datetime.now().isoformat()),
            )

    # ── Form 4 – Insider Trades ──────────────────────────────

    def get_insider_trades(
        self,
        symbol: str,
        lookback_days: int = 90,
        use_cache: bool = True,
    ) -> list[InsiderTrade]:
        """
        Hent insider-handler (Form 4) for et symbol.

        Args:
            symbol: Ticker symbol (f.eks. "AAPL").
            lookback_days: Antal dage tilbage at søge.
            use_cache: Brug SQLite cache.

        Returns:
            Liste af InsiderTrade sorteret efter dato (nyeste først).
        """
        symbol = symbol.upper()

        # Cache check
        if use_cache:
            cached = self._read_insider_cache(symbol, lookback_days)
            if cached:
                logger.debug(f"[insider] Cache hit: {symbol} – {len(cached)} trades")
                return cached

        # Hent fra SEC EDGAR
        trades = self._fetch_insider_trades_sec(symbol, lookback_days)

        # Cache resultater
        if trades:
            self._write_insider_cache(trades)

        return sorted(trades, key=lambda t: t.date, reverse=True)

    def _fetch_insider_trades_sec(self, symbol: str, lookback_days: int) -> list[InsiderTrade]:
        """Hent insider trades fra SEC EDGAR Full-Text Search."""
        trades: list[InsiderTrade] = []

        try:
            cik = self._get_cik(symbol)
            if not cik:
                logger.warning(f"[insider] Kunne ikke finde CIK for {symbol}")
                return trades

            # Brug SEC EDGAR submissions endpoint
            self._throttle()
            session = self._get_session()
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = session.get(url, timeout=15)

            if resp.status_code != 200:
                logger.warning(f"[insider] SEC returnerede {resp.status_code} for {symbol}")
                return trades

            data = resp.json()
            filings = data.get("filings", {}).get("recent", {})

            if not filings:
                return trades

            # Filtrer Form 4 filings
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            primary_docs = filings.get("primaryDocument", [])

            cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

            for i, form_type in enumerate(forms):
                if form_type != "4":
                    continue
                if i >= len(dates) or dates[i] < cutoff:
                    continue

                # Parse individuelt Form 4 filing
                filing_trades = self._parse_form4_filing(
                    symbol=symbol,
                    accession=accessions[i] if i < len(accessions) else "",
                    filing_date=dates[i],
                    cik=cik,
                )
                trades.extend(filing_trades)

                # Begræns antal API-kald
                if len(trades) > 100:
                    break

        except Exception as exc:
            logger.error(f"[insider] Fejl ved hentning for {symbol}: {exc}")

        return trades

    def _parse_form4_filing(
        self,
        symbol: str,
        accession: str,
        filing_date: str,
        cik: str,
    ) -> list[InsiderTrade]:
        """Parse et enkelt Form 4 filing fra SEC EDGAR."""
        trades: list[InsiderTrade] = []

        try:
            # Hent Form 4 XML
            acc_clean = accession.replace("-", "")
            self._throttle()
            session = self._get_session()
            url = f"https://data.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_clean}"

            # Prøv at hente filing index
            index_url = f"{url}/index.json"
            resp = session.get(index_url, timeout=15)

            if resp.status_code != 200:
                return trades

            index_data = resp.json()
            directory = index_data.get("directory", {})
            items = directory.get("item", [])

            # Find XML-filen i filingen
            xml_file = None
            for item in items:
                name = item.get("name", "")
                if name.endswith(".xml") and "primary_doc" not in name.lower():
                    xml_file = name
                    break

            if not xml_file:
                # Prøv at parse fra index
                return self._parse_form4_from_index(
                    symbol, filing_date, items, url
                )

            # Hent og parse XML
            self._throttle()
            xml_resp = session.get(f"{url}/{xml_file}", timeout=15)
            if xml_resp.status_code == 200:
                trades = self._parse_form4_xml(symbol, xml_resp.text, filing_date)

        except Exception as exc:
            logger.debug(f"[insider] Form 4 parse fejl: {exc}")

        return trades

    def _parse_form4_xml(self, symbol: str, xml_text: str, filing_date: str) -> list[InsiderTrade]:
        """Parse Form 4 XML og ekstraher transaktioner."""
        import xml.etree.ElementTree as ET
        trades: list[InsiderTrade] = []

        try:
            root = ET.fromstring(xml_text)

            # Hent insider info
            reporter = root.find(".//reportingOwner")
            name = ""
            title = ""
            if reporter is not None:
                name_elem = reporter.find(".//rptOwnerName")
                name = name_elem.text if name_elem is not None and name_elem.text else "Unknown"
                title_elem = reporter.find(".//officerTitle")
                title = title_elem.text if title_elem is not None and title_elem.text else ""
                if not title:
                    # Tjek for director / 10% owner
                    is_director = reporter.find(".//isDirector")
                    is_officer = reporter.find(".//isOfficer")
                    is_ten_pct = reporter.find(".//isTenPercentOwner")
                    if is_director is not None and is_director.text == "1":
                        title = "Director"
                    elif is_officer is not None and is_officer.text == "1":
                        title = "Officer"
                    elif is_ten_pct is not None and is_ten_pct.text == "1":
                        title = "10% Owner"

            # Parse non-derivative transaktioner
            for txn in root.findall(".//nonDerivativeTransaction"):
                trade = self._extract_transaction(txn, symbol, name, title, filing_date)
                if trade:
                    trades.append(trade)

            # Parse derivative transaktioner (optioner osv.)
            for txn in root.findall(".//derivativeTransaction"):
                trade = self._extract_transaction(txn, symbol, name, title, filing_date, derivative=True)
                if trade:
                    trades.append(trade)

        except ET.ParseError as exc:
            logger.debug(f"[insider] XML parse fejl: {exc}")

        return trades

    def _extract_transaction(
        self,
        txn_elem: Any,
        symbol: str,
        name: str,
        title: str,
        filing_date: str,
        derivative: bool = False,
    ) -> InsiderTrade | None:
        """Ekstraher én transaktion fra Form 4 XML element."""
        try:
            # Transaction code
            code_elem = txn_elem.find(".//transactionCode")
            if code_elem is None or not code_elem.text:
                return None
            code = code_elem.text.upper()

            tx_type = {
                "P": TransactionType.PURCHASE,
                "S": TransactionType.SALE,
                "A": TransactionType.GRANT,
                "M": TransactionType.EXERCISE,
                "C": TransactionType.CONVERSION,
            }.get(code, TransactionType.OTHER)

            # Shares
            shares_elem = txn_elem.find(".//transactionShares/value")
            shares = float(shares_elem.text) if shares_elem is not None and shares_elem.text else 0

            # Price
            price_elem = txn_elem.find(".//transactionPricePerShare/value")
            price = float(price_elem.text) if price_elem is not None and price_elem.text else 0

            # Date
            date_elem = txn_elem.find(".//transactionDate/value")
            trade_date = date_elem.text if date_elem is not None and date_elem.text else filing_date

            # Ownership after
            ownership_elem = txn_elem.find(".//sharesOwnedFollowingTransaction/value")
            ownership = float(ownership_elem.text) if ownership_elem is not None and ownership_elem.text else 0

            # Direct/indirect
            ownership_nature = txn_elem.find(".//directOrIndirectOwnership/value")
            is_direct = True
            if ownership_nature is not None and ownership_nature.text:
                is_direct = ownership_nature.text.upper() == "D"

            value = shares * price

            return InsiderTrade(
                symbol=symbol,
                insider_name=name,
                insider_title=title,
                transaction_type=tx_type,
                shares=shares,
                price=price,
                value=value,
                date=datetime.strptime(trade_date[:10], "%Y-%m-%d"),
                filing_date=datetime.strptime(filing_date[:10], "%Y-%m-%d"),
                ownership_after=ownership,
                is_direct=is_direct,
            )
        except (ValueError, AttributeError) as exc:
            logger.debug(f"[insider] Transaktion parse fejl: {exc}")
            return None

    def _parse_form4_from_index(
        self,
        symbol: str,
        filing_date: str,
        items: list[dict],
        base_url: str,
    ) -> list[InsiderTrade]:
        """Fallback: parse Form 4 fra filing index metadata."""
        # Minimal fallback – returnér tom liste
        return []

    # ── Cache helpers ─────────────────────────────────────────

    def _read_insider_cache(self, symbol: str, lookback_days: int) -> list[InsiderTrade]:
        """Læs insider trades fra cache."""
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        cache_cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT symbol, insider_name, insider_title, transaction_type,
                          shares, price, value, trade_date, filing_date,
                          ownership_after, is_direct
                   FROM insider_trades
                   WHERE symbol = ? AND trade_date >= ? AND fetched_at >= ?
                   ORDER BY trade_date DESC""",
                (symbol, cutoff, cache_cutoff),
            ).fetchall()

        if not rows:
            return []

        trades = []
        for r in rows:
            try:
                tx_type = {
                    "P": TransactionType.PURCHASE,
                    "S": TransactionType.SALE,
                    "A": TransactionType.GRANT,
                    "M": TransactionType.EXERCISE,
                    "C": TransactionType.CONVERSION,
                }.get(r[3], TransactionType.OTHER)

                trades.append(InsiderTrade(
                    symbol=r[0],
                    insider_name=r[1],
                    insider_title=r[2] or "",
                    transaction_type=tx_type,
                    shares=r[4] or 0,
                    price=r[5] or 0,
                    value=r[6] or 0,
                    date=datetime.strptime(r[7][:10], "%Y-%m-%d") if r[7] else datetime.now(),
                    filing_date=datetime.strptime(r[8][:10], "%Y-%m-%d") if r[8] else datetime.now(),
                    ownership_after=r[9] or 0,
                    is_direct=bool(r[10]),
                ))
            except (ValueError, IndexError):
                continue

        return trades

    def _write_insider_cache(self, trades: list[InsiderTrade]) -> None:
        """Skriv insider trades til cache."""
        now = datetime.now().isoformat()
        rows = [
            (
                t.symbol, t.insider_name, t.insider_title,
                t.transaction_type.value, t.shares, t.price, t.value,
                t.date.strftime("%Y-%m-%d"), t.filing_date.strftime("%Y-%m-%d"),
                t.ownership_after, int(t.is_direct), now,
            )
            for t in trades
        ]
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO insider_trades
                   (symbol, insider_name, insider_title, transaction_type,
                    shares, price, value, trade_date, filing_date,
                    ownership_after, is_direct, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        logger.debug(f"[insider] Cached {len(rows)} trades")

    # ── 13F – Institutional Holdings ──────────────────────────

    def get_institutional_holdings(
        self,
        symbol: str,
        funds: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> list[InstitutionalHolding]:
        """
        Hent institutionelle holdings for et symbol.

        Tjekker 13F filings fra store hedgefonde.

        Args:
            symbol: Ticker symbol.
            funds: Dict med {fund_name: cik}. Default = MAJOR_FUNDS.
            use_cache: Brug SQLite cache.

        Returns:
            Liste af InstitutionalHolding for seneste kvartal.
        """
        symbol = symbol.upper()
        funds = funds or MAJOR_FUNDS

        # Cache check
        if use_cache:
            cached = self._read_holdings_cache(symbol)
            if cached:
                logger.debug(f"[insider] Holdings cache hit: {symbol} – {len(cached)} funds")
                return cached

        holdings: list[InstitutionalHolding] = []

        for fund_name, fund_cik in funds.items():
            try:
                holding = self._fetch_13f_holding(symbol, fund_name, fund_cik)
                if holding:
                    holdings.append(holding)
            except Exception as exc:
                logger.debug(f"[insider] 13F fejl for {fund_name}: {exc}")

        # Cache
        if holdings:
            self._write_holdings_cache(holdings)

        return holdings

    def _fetch_13f_holding(
        self, symbol: str, fund_name: str, fund_cik: str,
    ) -> InstitutionalHolding | None:
        """Hent 13F holding for ét fund + symbol."""
        try:
            self._throttle()
            session = self._get_session()
            url = f"https://data.sec.gov/submissions/CIK{fund_cik}.json"
            resp = session.get(url, timeout=15)

            if resp.status_code != 200:
                return None

            data = resp.json()
            filings = data.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])

            # Find seneste 13F-HR
            for i, form in enumerate(forms):
                if form in ("13F-HR", "13F-HR/A"):
                    filing_date = dates[i] if i < len(dates) else ""
                    # Beregn kvartal
                    if filing_date:
                        dt = datetime.strptime(filing_date[:10], "%Y-%m-%d")
                        quarter = f"{dt.year}Q{(dt.month - 1) // 3 + 1}"
                    else:
                        quarter = "unknown"

                    # Note: Faktisk parsing af 13F kræver InfoTable XML
                    # Her returnerer vi metadata – fuld implementation
                    # kræver parsing af 13F InfoTable
                    return InstitutionalHolding(
                        fund_name=fund_name,
                        symbol=symbol,
                        shares=0,  # Kræver InfoTable parse
                        value_usd=0.0,
                        pct_portfolio=0.0,
                        quarter=quarter,
                    )

            return None

        except Exception as exc:
            logger.debug(f"[insider] 13F hentning fejl: {exc}")
            return None

    def _read_holdings_cache(self, symbol: str) -> list[InstitutionalHolding]:
        """Læs institutional holdings fra cache."""
        cache_cutoff = (datetime.now() - timedelta(days=7)).isoformat()

        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT fund_name, symbol, shares, value_usd, pct_portfolio,
                          quarter, change_shares, change_pct, is_new_position, is_closed
                   FROM institutional_holdings
                   WHERE symbol = ? AND fetched_at >= ?
                   ORDER BY value_usd DESC""",
                (symbol, cache_cutoff),
            ).fetchall()

        return [
            InstitutionalHolding(
                fund_name=r[0], symbol=r[1], shares=r[2], value_usd=r[3],
                pct_portfolio=r[4], quarter=r[5], change_shares=r[6],
                change_pct=r[7], is_new_position=bool(r[8]), is_closed=bool(r[9]),
            )
            for r in rows
        ]

    def _write_holdings_cache(self, holdings: list[InstitutionalHolding]) -> None:
        """Skriv institutional holdings til cache."""
        now = datetime.now().isoformat()
        rows = [
            (
                h.fund_name, h.symbol, h.shares, h.value_usd,
                h.pct_portfolio, h.quarter, h.change_shares, h.change_pct,
                int(h.is_new_position), int(h.is_closed), now,
            )
            for h in holdings
        ]
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO institutional_holdings
                   (fund_name, symbol, shares, value_usd, pct_portfolio,
                    quarter, change_shares, change_pct, is_new_position,
                    is_closed, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    # ── Short Interest ────────────────────────────────────────

    def get_short_interest(
        self,
        symbol: str,
        use_cache: bool = True,
    ) -> ShortInterestData | None:
        """
        Hent short interest data for et symbol.

        Bruger yfinance som datakilde (SEC data er forsinket).

        Args:
            symbol: Ticker symbol.
            use_cache: Brug cache.

        Returns:
            ShortInterestData eller None.
        """
        symbol = symbol.upper()

        # Cache check
        if use_cache:
            cached = self._read_short_cache(symbol)
            if cached:
                return cached

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            short_interest = info.get("sharesShort", 0) or 0
            short_pct = info.get("shortPercentOfFloat", 0) or 0
            short_ratio = info.get("shortRatio", 0) or 0
            avg_volume = info.get("averageDailyVolume10Day", 0) or info.get("averageVolume", 0) or 0
            prev_short = info.get("sharesShortPriorMonth", 0) or 0

            # Beregn ændring
            change_pct = 0.0
            if prev_short > 0:
                change_pct = (short_interest - prev_short) / prev_short * 100

            # Squeeze potential
            squeeze = self._calc_squeeze_potential(short_pct * 100, short_ratio, change_pct)

            data = ShortInterestData(
                symbol=symbol,
                short_interest=short_interest,
                short_pct_float=short_pct * 100 if short_pct < 1 else short_pct,
                short_ratio=short_ratio,
                avg_volume=avg_volume,
                previous_short_interest=prev_short,
                change_pct=change_pct,
                squeeze_potential=squeeze,
                date=datetime.now(),
            )

            # Cache
            self._write_short_cache(data)
            return data

        except Exception as exc:
            logger.error(f"[insider] Short interest fejl for {symbol}: {exc}")
            return None

    @staticmethod
    def _calc_squeeze_potential(
        short_pct: float, days_to_cover: float, change_pct: float,
    ) -> str:
        """Beregn short squeeze potential."""
        score = 0

        # Short % of float
        if short_pct > 40:
            score += 4
        elif short_pct > 25:
            score += 3
        elif short_pct > 15:
            score += 2
        elif short_pct > 10:
            score += 1

        # Days to cover
        if days_to_cover > 10:
            score += 3
        elif days_to_cover > 5:
            score += 2
        elif days_to_cover > 3:
            score += 1

        # Increasing short interest
        if change_pct > 20:
            score += 2
        elif change_pct > 10:
            score += 1

        if score >= 7:
            return "extreme"
        elif score >= 5:
            return "high"
        elif score >= 3:
            return "medium"
        return "low"

    def _read_short_cache(self, symbol: str) -> ShortInterestData | None:
        """Læs short interest fra cache."""
        cache_cutoff = (datetime.now() - timedelta(hours=12)).isoformat()

        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT short_interest, short_pct_float, short_ratio,
                          avg_volume, previous_short, change_pct,
                          squeeze_potential, report_date
                   FROM short_interest
                   WHERE symbol = ? AND fetched_at >= ?
                   ORDER BY report_date DESC LIMIT 1""",
                (symbol, cache_cutoff),
            ).fetchone()

        if not row:
            return None

        return ShortInterestData(
            symbol=symbol,
            short_interest=row[0] or 0,
            short_pct_float=row[1] or 0,
            short_ratio=row[2] or 0,
            avg_volume=row[3] or 0,
            previous_short_interest=row[4] or 0,
            change_pct=row[5] or 0,
            squeeze_potential=row[6] or "low",
            date=datetime.strptime(row[7][:10], "%Y-%m-%d") if row[7] else datetime.now(),
        )

    def _write_short_cache(self, data: ShortInterestData) -> None:
        """Skriv short interest til cache."""
        now = datetime.now().isoformat()
        report_date = data.date.strftime("%Y-%m-%d") if data.date else now[:10]

        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO short_interest
                   (symbol, short_interest, short_pct_float, short_ratio,
                    avg_volume, previous_short, change_pct,
                    squeeze_potential, report_date, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data.symbol, data.short_interest, data.short_pct_float,
                    data.short_ratio, data.avg_volume, data.previous_short_interest,
                    data.change_pct, data.squeeze_potential, report_date, now,
                ),
            )


# ── InsiderTracker (Hovedklasse) ─────────────────────────────

class InsiderTracker:
    """
    Samlet insider tracking – kombinerer Form 4, 13F og short interest.

    Beregner insider sentiment, detekterer cluster buying,
    og genererer smart money rapporter.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        cluster_window_days: int = 14,
        cluster_min_insiders: int = 3,
        sentiment_lookback_days: int = 90,
    ) -> None:
        self._client = SECEdgarClient(cache_dir=cache_dir)
        self._cluster_window = cluster_window_days
        self._cluster_min = cluster_min_insiders
        self._sentiment_lookback = sentiment_lookback_days

    # ── Insider Sentiment ─────────────────────────────────────

    def get_insider_sentiment(
        self,
        symbol: str,
        lookback_days: int | None = None,
    ) -> InsiderSentimentScore:
        """
        Beregn insider sentiment score for et symbol.

        Analyserer Form 4 filings og returnerer sentiment
        baseret på køb/salg, cluster buying og C-suite aktivitet.
        """
        lookback = lookback_days or self._sentiment_lookback
        trades = self._client.get_insider_trades(symbol, lookback_days=lookback)

        # Filtrer kun køb og salg (ignorer grants/exercises)
        purchases = [t for t in trades if t.is_purchase]
        sales = [t for t in trades if t.is_sale]

        total_buy_value = sum(t.value for t in purchases)
        total_sell_value = sum(t.value for t in sales)

        # Cluster detection
        cluster_buying = self._detect_cluster(purchases, is_buy=True)
        cluster_selling = self._detect_cluster(sales, is_buy=False)

        # C-suite buying
        c_suite_buying = any(t.is_c_suite for t in purchases)

        # Beregn score (-100 til +100)
        score = self._calc_sentiment_score(
            purchases=len(purchases),
            sales=len(sales),
            buy_value=total_buy_value,
            sell_value=total_sell_value,
            cluster_buying=cluster_buying,
            cluster_selling=cluster_selling,
            c_suite_buying=c_suite_buying,
        )

        # Kategorisér sentiment
        sentiment = self._score_to_sentiment(score, cluster_buying, cluster_selling)

        last_date = trades[0].date if trades else None

        return InsiderSentimentScore(
            symbol=symbol,
            sentiment=sentiment,
            score=score,
            net_purchases=len(purchases),
            net_sales=len(sales),
            total_buy_value=total_buy_value,
            total_sell_value=total_sell_value,
            cluster_buying=cluster_buying,
            cluster_selling=cluster_selling,
            c_suite_buying=c_suite_buying,
            period_days=lookback,
            last_trade_date=last_date,
            trades=trades,
        )

    def _detect_cluster(self, trades: list[InsiderTrade], is_buy: bool) -> bool:
        """Detektér cluster buying/selling (≥N insidere inden for window)."""
        if len(trades) < self._cluster_min:
            return False

        # Gruppér unikke insidere per vindue
        trades_sorted = sorted(trades, key=lambda t: t.date)

        for i, trade in enumerate(trades_sorted):
            window_start = trade.date
            window_end = window_start + timedelta(days=self._cluster_window)

            insiders_in_window = set()
            for t in trades_sorted[i:]:
                if t.date > window_end:
                    break
                insiders_in_window.add(t.insider_name)

            if len(insiders_in_window) >= self._cluster_min:
                return True

        return False

    @staticmethod
    def _calc_sentiment_score(
        purchases: int,
        sales: int,
        buy_value: float,
        sell_value: float,
        cluster_buying: bool,
        cluster_selling: bool,
        c_suite_buying: bool,
    ) -> float:
        """
        Beregn sentiment score fra -100 til +100.

        Vægter:
          - Antal transaktioner (25%)
          - Værdi af transaktioner (35%)
          - Cluster buying/selling (25%)
          - C-suite aktivitet (15%)
        """
        total_txn = purchases + sales
        if total_txn == 0:
            return 0.0

        # 1. Transaktions-ratio (-1 til +1)
        txn_ratio = (purchases - sales) / total_txn

        # 2. Værdi-ratio (-1 til +1)
        total_value = buy_value + sell_value
        if total_value > 0:
            value_ratio = (buy_value - sell_value) / total_value
        else:
            value_ratio = 0.0

        # 3. Cluster bonus (-1 til +1)
        cluster_score = 0.0
        if cluster_buying:
            cluster_score = 1.0
        elif cluster_selling:
            cluster_score = -1.0

        # 4. C-suite bonus (0 til +1)
        csuite_score = 1.0 if c_suite_buying else 0.0

        # Weighted sum
        score = (
            txn_ratio * 25
            + value_ratio * 35
            + cluster_score * 25
            + csuite_score * 15
        )

        return max(-100.0, min(100.0, score))

    @staticmethod
    def _score_to_sentiment(
        score: float,
        cluster_buying: bool,
        cluster_selling: bool,
    ) -> InsiderSentiment:
        """Konvertér numerisk score til InsiderSentiment enum."""
        if cluster_buying and score > 20:
            return InsiderSentiment.VERY_BULLISH
        if cluster_selling and score < -20:
            return InsiderSentiment.VERY_BEARISH

        if score > 30:
            return InsiderSentiment.BULLISH
        if score < -30:
            return InsiderSentiment.BEARISH

        return InsiderSentiment.NEUTRAL

    # ── Smart Money Flow ──────────────────────────────────────

    def get_smart_money_flow(self, symbol: str) -> SmartMoneyFlow:
        """
        Beregn smart money flow for et symbol.

        Aggregerer institutionelle holdings fra store hedgefonde.
        """
        holdings = self._client.get_institutional_holdings(symbol)

        new_positions = [h.fund_name for h in holdings if h.is_new_position]
        closed_positions = [h.fund_name for h in holdings if h.is_closed]
        increased = [h.fund_name for h in holdings if h.change_shares > 0 and not h.is_new_position]
        decreased = [h.fund_name for h in holdings if h.change_shares < 0 and not h.is_closed]

        total_value = sum(h.value_usd for h in holdings)
        net_change = sum(h.change_shares * (h.value_usd / h.shares if h.shares > 0 else 0) for h in holdings)

        return SmartMoneyFlow(
            symbol=symbol,
            institutional_holders=len(holdings),
            total_institutional_value=total_value,
            net_institutional_change=net_change,
            new_positions=new_positions,
            closed_positions=closed_positions,
            increased=increased,
            decreased=decreased,
            top_holders=sorted(holdings, key=lambda h: h.value_usd, reverse=True)[:10],
        )

    # ── Short Interest ────────────────────────────────────────

    def get_short_interest(self, symbol: str) -> ShortInterestData | None:
        """Hent short interest data for et symbol."""
        return self._client.get_short_interest(symbol)

    # ── Full Report ───────────────────────────────────────────

    def get_smart_money_report(self, symbol: str) -> SmartMoneyReport:
        """
        Generér komplet smart money rapport for et symbol.

        Kombinerer insider trades, institutional holdings og short interest.
        """
        warnings: list[str] = []

        # 1. Insider sentiment
        try:
            insider = self.get_insider_sentiment(symbol)
        except Exception as exc:
            logger.error(f"[insider] Sentiment fejl for {symbol}: {exc}")
            insider = None

        # 2. Smart money flow
        try:
            flow = self.get_smart_money_flow(symbol)
        except Exception as exc:
            logger.error(f"[insider] Smart money flow fejl for {symbol}: {exc}")
            flow = None

        # 3. Short interest
        try:
            short = self.get_short_interest(symbol)
        except Exception as exc:
            logger.error(f"[insider] Short interest fejl for {symbol}: {exc}")
            short = None

        # Samlet signal
        overall = "neutral"
        confidence_adj = 0

        if insider:
            confidence_adj += insider.confidence_boost
            if insider.cluster_buying:
                warnings.append(f"🟢 Cluster buying detekteret i {symbol}")
            if insider.cluster_selling:
                warnings.append(f"🔴 Cluster selling detekteret i {symbol}")
            if insider.c_suite_buying:
                warnings.append(f"🟢 C-suite insider buying i {symbol}")

        if short:
            if short.is_heavily_shorted:
                warnings.append(
                    f"⚠️ {symbol} er tungt shortet ({short.short_pct_float:.1f}% af float)"
                )
                confidence_adj -= 5
            if short.squeeze_potential in ("high", "extreme"):
                warnings.append(
                    f"🔥 Short squeeze potential: {short.squeeze_potential} for {symbol}"
                )
            if short.days_to_cover > 5:
                warnings.append(
                    f"⏰ Days to cover: {short.days_to_cover:.1f} dage for {symbol}"
                )

        # Bestem overall signal
        if confidence_adj > 5:
            overall = "bullish"
        elif confidence_adj < -5:
            overall = "bearish"

        return SmartMoneyReport(
            symbol=symbol,
            insider_sentiment=insider,
            smart_money_flow=flow,
            short_interest=short,
            overall_signal=overall,
            confidence_adjustment=max(-15, min(15, confidence_adj)),
            warnings=warnings,
        )

    # ── Batch Operations ──────────────────────────────────────

    def scan_symbols(self, symbols: list[str]) -> dict[str, SmartMoneyReport]:
        """
        Scan flere symboler for smart money signals.

        Returns:
            Dict med symbol → SmartMoneyReport.
        """
        results: dict[str, SmartMoneyReport] = {}

        for symbol in symbols:
            try:
                results[symbol] = self.get_smart_money_report(symbol)
                logger.info(f"[insider] Scannet {symbol}: {results[symbol].overall_signal}")
            except Exception as exc:
                logger.error(f"[insider] Scan fejl for {symbol}: {exc}")

        return results

    def get_top_insider_buys(
        self,
        symbols: list[str],
        lookback_days: int = 30,
        min_value: float = 50_000,
    ) -> list[InsiderTrade]:
        """
        Find de vigtigste insider-køb på tværs af symboler.

        Args:
            symbols: Symboler at scanne.
            lookback_days: Periode.
            min_value: Minimum transaktionsværdi USD.

        Returns:
            Sorteret liste af insider-køb (største først).
        """
        all_buys: list[InsiderTrade] = []

        for symbol in symbols:
            try:
                trades = self._client.get_insider_trades(symbol, lookback_days=lookback_days)
                buys = [t for t in trades if t.is_purchase and t.value >= min_value]
                all_buys.extend(buys)
            except Exception as exc:
                logger.debug(f"[insider] Fejl for {symbol}: {exc}")

        return sorted(all_buys, key=lambda t: t.value, reverse=True)

    # ── Strategy Integration ──────────────────────────────────

    def get_confidence_adjustment(self, symbol: str) -> int:
        """
        Beregn confidence-justering for et symbol.

        Bruges af SignalEngine til at booste/reducere confidence.

        Returns:
            -15 til +15 points adjustment.
        """
        report = self.get_smart_money_report(symbol)
        return report.confidence_adjustment

    def get_short_interest_warning(self, symbol: str, threshold: float = 20.0) -> str | None:
        """
        Returnér advarsel hvis short interest > threshold %.

        Returns:
            Advarsels-streng eller None.
        """
        short = self._client.get_short_interest(symbol)
        if short and short.short_pct_float > threshold:
            return (
                f"⚠️ {symbol}: Short interest {short.short_pct_float:.1f}% "
                f"(days to cover: {short.days_to_cover:.1f})"
            )
        return None

    # ── Explain ───────────────────────────────────────────────

    def explain(self, symbol: str) -> str:
        """
        Forklar smart money data i simple termer.

        Returns:
            Menneskelæselig forklaring.
        """
        report = self.get_smart_money_report(symbol)
        lines = [
            f"═══ SMART MONEY RAPPORT: {symbol} ═══",
            "",
        ]

        # Insider Sentiment
        if report.insider_sentiment:
            s = report.insider_sentiment
            lines.append("📊 INSIDER AKTIVITET")
            lines.append(f"   Sentiment: {s.sentiment.value.upper()} (score: {s.score:+.0f})")
            lines.append(f"   Køb: {s.net_purchases} handler (${s.total_buy_value:,.0f})")
            lines.append(f"   Salg: {s.net_sales} handler (${s.total_sell_value:,.0f})")
            if s.cluster_buying:
                lines.append("   🟢 CLUSTER BUYING: Flere insidere køber samtidig!")
            if s.c_suite_buying:
                lines.append("   🟢 C-SUITE BUYING: Ledelsen køber egne aktier")
            lines.append("")

        # Short Interest
        if report.short_interest:
            si = report.short_interest
            lines.append("📉 SHORT INTEREST")
            lines.append(f"   Short % af float: {si.short_pct_float:.1f}%")
            lines.append(f"   Days to cover: {si.days_to_cover:.1f}")
            lines.append(f"   Squeeze potential: {si.squeeze_potential.upper()}")
            if si.change_pct != 0:
                direction = "↑" if si.change_pct > 0 else "↓"
                lines.append(f"   Ændring: {direction} {abs(si.change_pct):.1f}%")
            lines.append("")

        # Smart Money Flow
        if report.smart_money_flow:
            f = report.smart_money_flow
            lines.append("🏦 INSTITUTIONELLE INVESTORER")
            lines.append(f"   Trackede fonde: {f.institutional_holders}")
            if f.new_positions:
                lines.append(f"   Nye positioner: {', '.join(f.new_positions)}")
            if f.closed_positions:
                lines.append(f"   Lukkede positioner: {', '.join(f.closed_positions)}")
            lines.append("")

        # Samlet
        lines.append("📋 SAMLET VURDERING")
        lines.append(f"   Signal: {report.overall_signal.upper()}")
        lines.append(f"   Confidence justering: {report.confidence_adjustment:+d} points")

        if report.warnings:
            lines.append("")
            lines.append("⚠️ ADVARSLER:")
            for w in report.warnings:
                lines.append(f"   {w}")

        return "\n".join(lines)

    def print_report(self, symbol: str) -> None:
        """Print smart money rapport til konsollen."""
        print(self.explain(symbol))
