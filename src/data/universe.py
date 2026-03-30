"""
AssetUniverse – organiserer alle handlebare aktiver i kategorier.

Understøtter:
  - Aktier (US, Norden, Europa, Asien, Emerging Markets)
  - ETF'er (indeks, sektor, obligation, råstof, region, tema)
  - Råstoffer (ædelmetaller, energi, landbrug, industrimetaller)
  - Kryptovaluta (top 20 efter markedsværdi)
  - Obligationer & renter (US Treasury yields, yield curve)
  - Valuta / Forex (hovedvalutaer)

Alle symboler bruger yfinance-format.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from loguru import logger


# ══════════════════════════════════════════════════════════════
#  Statiske univers-definitioner
# ══════════════════════════════════════════════════════════════

# ── 1. Aktier ────────────────────────────────────────────────

US_LARGE_CAP_CORE = [
    # Top 50 S&P 500 efter markedsværdi
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "XOM", "V", "AVGO", "JNJ", "MA", "PG", "HD",
    "COST", "ABBV", "MRK", "ADBE", "CRM", "WMT", "BAC", "NFLX", "CVX",
    "KO", "AMD", "PEP", "TMO", "LIN", "CSCO", "MCD", "ABT", "ORCL",
    "DHR", "ACN", "WFC", "TXN", "PM", "NEE", "CMCSA", "INTC", "RTX",
    "IBM", "AMGN", "HON", "UNP", "COP",
]

US_LARGE_CAP_EXTENDED = [
    # S&P 500 nr. 51-100
    "LOW", "CAT", "ISRG", "GE", "SPGI", "GS", "INTU", "ELV", "AMAT",
    "BKNG", "BLK", "ADP", "SYK", "MDLZ", "PLD", "T", "CB", "VRTX",
    "MMC", "REGN", "SCHW", "DE", "GILD", "LRCX", "AMT", "ETN", "ADI",
    "CI", "ZTS", "MO", "SO", "BSX", "PGR", "KLAC", "FI", "PANW",
    "DUK", "CME", "SNPS", "CDNS", "HUM", "CL", "EOG", "ICE", "SHW",
    "SLB", "MCK", "MPC", "FCX",
]

US_MID_SMALL_CAP = [
    # Russell 2000 top 200 (repræsentativt udvalg – top 50)
    "SMCI", "MSTR", "CELH", "CORT", "EXEL", "LNTH", "HALO",
    "RMBS", "ENSG", "PI", "CVLT", "GSHD", "BOOT", "CRVL",
    "AIT", "KRYS", "SAIA", "ELF", "SFM", "PCVX", "CSWI",
    "FROG", "AEHR", "FSS", "BLD", "ARCB", "WDFC", "MTH",
    "LBRT", "CALM", "AEIS", "PRIM", "IIPR", "KTOS", "CPRX",
    "SPSC", "SWX", "UFPT", "ROAD", "PRGS", "RXO", "NEOG",
    "ALKT", "VCYT", "TMDX", "SIG", "CRS", "PGNY", "FN", "CARG",
]

# Nordiske aktier (yfinance: .CO = København, .ST = Stockholm, .OL = Oslo)
NORDIC_OMX_C25 = [
    # OMX Copenhagen 25
    "NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "NZYM-B.CO", "VWS.CO",
    "CARL-B.CO", "ORSTED.CO", "COLO-B.CO", "PNDORA.CO", "GN.CO",
    "DEMANT.CO", "RBREW.CO", "ISS.CO", "TRYG.CO", "JYSK.CO",
    "FLS.CO", "ROCK-B.CO", "AMBU-B.CO", "GMAB.CO", "SIM.CO",
    "BAVA.CO", "NETC.CO", "TOP.CO", "NDA-DK.CO", "DANSKE.CO",
]

NORDIC_OMX_S30 = [
    # OMX Stockholm 30
    "VOLV-B.ST", "ASSA-B.ST", "ATCO-A.ST", "HEXA-B.ST", "INVE-B.ST",
    "SAND.ST", "ABB.ST", "ERIC-B.ST", "SEB-A.ST", "SHB-A.ST",
    "ALFA.ST", "ESSITY-B.ST", "SWED-A.ST", "KINV-B.ST", "SKF-B.ST",
    "TELIA.ST", "BOL.ST", "SINCH.ST", "NDA-SE.ST", "ELUX-B.ST",
    "HM-B.ST", "GETI-B.ST", "SSAB-A.ST", "TEL2-B.ST", "SAAB-B.ST",
    "NIBE-B.ST", "SCA-B.ST", "SWMA.ST", "SOBI.ST", "EVO.ST",
]

NORDIC_OBX_OSLO = [
    # OBX Oslo 25
    "EQNR.OL", "DNB.OL", "MOWI.OL", "TEL.OL", "ORK.OL",
    "AKRBP.OL", "YAR.OL", "SALM.OL", "SUBC.OL", "AKER.OL",
    "SCHA.OL", "KOG.OL", "BAKKA.OL", "GJF.OL", "NHYDY",
    "FRO.OL", "RECSI.OL", "BWLPG.OL", "AKSO.OL", "GOGL.OL",
    "HAFNI.OL", "AUSS.OL", "KIT.OL", "CRAYN.OL", "ENTRA.OL",
]

EUROPEAN_STOCKS = [
    # EURO STOXX 50 (top 30)
    "ASML.AS", "MC.PA", "SAP.DE", "TTE.PA", "SIE.DE", "SAN.PA",
    "AIR.PA", "ALV.DE", "OR.PA", "DTE.DE", "BNP.PA", "SU.PA",
    "ABI.BR", "INGA.AS", "IBE.MC", "MUV2.DE", "DG.PA", "BAS.DE",
    "AI.PA", "ADS.DE", "ENEL.MI", "ISP.MI", "CS.PA", "EL.PA",
    "BMW.DE", "VOW3.DE", "BAYN.DE", "KER.PA", "PHIA.AS", "ENI.MI",
    # FTSE 100 (top 20)
    "SHEL.L", "AZN.L", "ULVR.L", "HSBA.L", "BP.L", "GSK.L",
    "RIO.L", "DGE.L", "LSEG.L", "REL.L", "BA.L", "NG.L",
    "BATS.L", "AAL.L", "VOD.L", "GLEN.L", "BHP.L", "PRU.L",
    "CPG.L", "RKT.L",
    # DAX 40 (tilføj udover STOXX)
    "MBG.DE", "DB1.DE", "HEN3.DE", "IFX.DE", "RHM.DE",
    "FRE.DE", "MTX.DE", "SHL.DE", "PUM.DE", "ZAL.DE",
]

ASIAN_STOCKS = [
    # Nikkei 225 (top 50)
    "7203.T", "6758.T", "6861.T", "8306.T", "9984.T",  # Toyota, Sony, Keyence, MUFG, SoftBank
    "9432.T", "6902.T", "8035.T", "4063.T", "6501.T",  # NTT, Denso, TEL, Shin-Etsu, Hitachi
    "6367.T", "8058.T", "7741.T", "4502.T", "4503.T",  # Daikin, Mitsub Corp, Hoya, Takeda, Astellas
    "8766.T", "7267.T", "9433.T", "6098.T", "8801.T",  # Tokio Marine, Honda, KDDI, Recruit, Mitsui F
    "6971.T", "7974.T", "3382.T", "8031.T", "2914.T",  # Kyocera, Nintendo, 7-Eleven, Mitsui, JT
    "4568.T", "6273.T", "7751.T", "2802.T", "6954.T",  # Daiichi, SMC, Canon, Ajinomoto, Fanuc
    "8316.T", "6702.T", "4901.T", "6326.T", "9020.T",  # SMFG, Fujitsu, Fuji Film, Kubota, JR East
    "4543.T", "2413.T", "7832.T", "9613.T", "8002.T",  # Terumo, M3, Bandai, NTT Data, Marubeni
    "6594.T", "5108.T", "6988.T", "8411.T", "7752.T",  # Nidec, Bridgestone, Nitto, Mizuho, Ricoh
    "4578.T", "3659.T", "1605.T", "7201.T", "8053.T",  # Otsuka, Nexon, INPEX, Nissan, Sumitomo
    # Hang Seng (top 30)
    "0700.HK", "9988.HK", "0941.HK", "1299.HK", "0005.HK",
    "2318.HK", "0388.HK", "1810.HK", "0011.HK", "0003.HK",
    "0883.HK", "0027.HK", "1398.HK", "0016.HK", "0002.HK",
    "3968.HK", "0006.HK", "0012.HK", "2628.HK", "0688.HK",
    "1109.HK", "0823.HK", "2269.HK", "0175.HK", "0017.HK",
    "1038.HK", "0267.HK", "0066.HK", "0001.HK", "0019.HK",
]

EMERGING_MARKETS = [
    # Top 30 MSCI Emerging Markets (ADR'er / direkte)
    "TSM", "BABA", "TCEHY", "RELIANCE.NS", "INFY",
    "PDD", "HDB", "VALE", "NU", "JD",
    "LI", "BIDU", "WIT", "IBN", "KB",
    "SHG", "ITUB", "BEKE", "TME", "SE",
    "GRAB", "DIDIY", "ZTO", "YUMC", "XPEV",
    "NIO", "VNET", "CPNG", "MELI", "STNE",
]


# ── 2. ETF'er ────────────────────────────────────────────────

ETFS_BROAD_INDEX = [
    "SPY", "QQQ", "VTI", "VXUS", "VEA", "VWO",
    "IWM", "DIA", "VOO", "VT",
]

ETFS_SECTOR = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
    "XLY", "XLRE", "XLB", "XLC", "XLU",
]

ETFS_BONDS = [
    "BND", "TLT", "AGG", "HYG", "LQD",
    "IEF", "SHY", "TIPS", "EMB", "BNDX",
]

ETFS_COMMODITIES = [
    "GLD", "SLV", "USO", "DBA", "DBC",
    "PDBC", "PPLT", "PALL", "WEAT", "CORN",
]

ETFS_REGIONS = [
    "EWD", "EWQ", "EWG", "EWU", "EWJ", "FXI",
    "EWZ", "INDA", "EWT", "EWY", "EWA", "EWC",
]

ETFS_THEMATIC = [
    "ARKK", "ICLN", "BOTZ", "HACK", "SOXX",
    "TAN", "LIT", "DRIV", "ESPO", "BLOK",
    "FINX", "GNOM", "ARKG", "BETZ", "JETS",
]


# ── 3. Råstoffer ─────────────────────────────────────────────

COMMODITIES_PRECIOUS = [
    ("GC=F", "Guld"),
    ("SI=F", "Sølv"),
    ("PL=F", "Platin"),
    ("PA=F", "Palladium"),
]

COMMODITIES_ENERGY = [
    ("CL=F", "Råolie WTI"),
    ("BZ=F", "Brent Crude"),
    ("NG=F", "Naturgas"),
    ("HO=F", "Fyringsolie"),
    ("RB=F", "Benzin"),
]

COMMODITIES_AGRICULTURE = [
    ("ZW=F", "Hvede"),
    ("ZC=F", "Majs"),
    ("ZS=F", "Sojabønner"),
    ("KC=F", "Kaffe"),
    ("CC=F", "Kakao"),
    ("CT=F", "Bomuld"),
    ("SB=F", "Sukker"),
    ("OJ=F", "Appelsinjuice"),
]

COMMODITIES_INDUSTRIAL = [
    ("HG=F", "Kobber"),
    ("ALI=F", "Aluminium"),
]


# ── 4. Kryptovaluta ──────────────────────────────────────────

CRYPTO_TOP_20 = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "TRX-USD", "MATIC-USD", "SHIB-USD", "TON11419-USD", "UNI7083-USD",
    "LTC-USD", "BCH-USD", "NEAR-USD", "APT21794-USD", "FIL-USD",
]


# ── 5. Obligationer & renter ─────────────────────────────────

TREASURY_YIELDS = [
    ("^IRX", "US 3M T-Bill"),
    ("^FVX", "US 5Y Treasury"),
    ("^TNX", "US 10Y Treasury"),
    ("^TYX", "US 30Y Treasury"),
]


# ── 6. Forex ─────────────────────────────────────────────────

FOREX_PAIRS = [
    ("EURUSD=X", "EUR/USD"),
    ("GBPUSD=X", "GBP/USD"),
    ("USDJPY=X", "USD/JPY"),
    ("USDCHF=X", "USD/CHF"),
    ("AUDUSD=X", "AUD/USD"),
    ("USDCAD=X", "USD/CAD"),
    ("USDDKK=X", "USD/DKK"),
    ("EURDKK=X", "EUR/DKK"),
    ("USDSEK=X", "USD/SEK"),
    ("USDNOK=X", "USD/NOK"),
]


# ══════════════════════════════════════════════════════════════
#  Kategorier og metadata
# ══════════════════════════════════════════════════════════════


@dataclass
class AssetCategory:
    """Én aktivkategori med symboler og metadata."""

    name: str                           # Intern nøgle
    display_name: str                   # Visningsnavn (dansk)
    subcategories: dict[str, list]      # Underkategori → symboler
    tradeable: bool = True              # Kan handles af strategier
    supports_24h: bool = False          # Handler 24/7 (krypto)
    default_interval: str = "1d"        # Standard datahentnings-interval
    description: str = ""

    @property
    def all_symbols(self) -> list[str]:
        """Alle symboler i kategorien (flat liste)."""
        symbols = []
        for items in self.subcategories.values():
            for item in items:
                if isinstance(item, tuple):
                    symbols.append(item[0])
                else:
                    symbols.append(item)
        return symbols

    @property
    def symbol_count(self) -> int:
        return len(self.all_symbols)


# ══════════════════════════════════════════════════════════════
#  AssetUniverse
# ══════════════════════════════════════════════════════════════


class AssetUniverse:
    """
    Organiserer alle handlebare aktiver i kategorier.

    Features:
      - Slå hele kategorier til/fra via config
      - Watchlist-mode: kun brugerens udvalgte aktiver
      - Scan-mode: screener hele universet for signaler
      - Filtrering på region, sektor, aktivklasse
      - Persistent cache af univers-metadata
    """

    # Standard kategori-definitioner
    _DEFAULT_CATEGORIES: dict[str, AssetCategory] = {
        "us_stocks": AssetCategory(
            name="us_stocks",
            display_name="US Aktier",
            subcategories={
                "large_cap_core": US_LARGE_CAP_CORE,
                "large_cap_extended": US_LARGE_CAP_EXTENDED,
                "mid_small_cap": US_MID_SMALL_CAP,
            },
            description="Amerikanske aktier: S&P 500 + Russell 2000",
        ),
        "nordic_stocks": AssetCategory(
            name="nordic_stocks",
            display_name="Nordiske Aktier",
            subcategories={
                "omx_c25": NORDIC_OMX_C25,
                "omx_s30": NORDIC_OMX_S30,
                "obx_oslo": NORDIC_OBX_OSLO,
            },
            description="OMX C25 (Danmark), OMX Stockholm 30, OBX (Norge)",
        ),
        "european_stocks": AssetCategory(
            name="european_stocks",
            display_name="Europæiske Aktier",
            subcategories={
                "euro_stoxx_ftse_dax": EUROPEAN_STOCKS,
            },
            description="EURO STOXX 50, FTSE 100, DAX 40",
        ),
        "asian_stocks": AssetCategory(
            name="asian_stocks",
            display_name="Asiatiske Aktier",
            subcategories={
                "nikkei_hang_seng": ASIAN_STOCKS,
            },
            description="Nikkei 225 (top 50), Hang Seng (top 30)",
        ),
        "emerging_markets": AssetCategory(
            name="emerging_markets",
            display_name="Emerging Markets",
            subcategories={
                "msci_em_top30": EMERGING_MARKETS,
            },
            description="Top 30 fra MSCI Emerging Markets",
        ),
        "etfs": AssetCategory(
            name="etfs",
            display_name="ETF'er",
            subcategories={
                "broad_index": ETFS_BROAD_INDEX,
                "sector": ETFS_SECTOR,
                "bonds": ETFS_BONDS,
                "commodities": ETFS_COMMODITIES,
                "regions": ETFS_REGIONS,
                "thematic": ETFS_THEMATIC,
            },
            description="Exchange Traded Funds – indeks, sektor, obligationer, råstoffer, tema",
        ),
        "commodities": AssetCategory(
            name="commodities",
            display_name="Råstoffer",
            subcategories={
                "precious_metals": COMMODITIES_PRECIOUS,
                "energy": COMMODITIES_ENERGY,
                "agriculture": COMMODITIES_AGRICULTURE,
                "industrial_metals": COMMODITIES_INDUSTRIAL,
            },
            description="Futures: ædelmetaller, energi, landbrug, industrimetaller",
        ),
        "crypto": AssetCategory(
            name="crypto",
            display_name="Kryptovaluta",
            subcategories={
                "top_20": CRYPTO_TOP_20,
            },
            supports_24h=True,
            description="Top 20 kryptovalutaer efter markedsværdi",
        ),
        "bonds": AssetCategory(
            name="bonds",
            display_name="Obligationer & Renter",
            subcategories={
                "treasury_yields": TREASURY_YIELDS,
            },
            tradeable=False,
            description="US Treasury yields – til yield curve analyse",
        ),
        "forex": AssetCategory(
            name="forex",
            display_name="Valuta (Forex)",
            subcategories={
                "major_pairs": FOREX_PAIRS,
            },
            supports_24h=True,
            description="Hovedvalutaer inkl. USD/DKK, EUR/DKK",
        ),
    }

    def __init__(
        self,
        enabled_categories: list[str] | None = None,
        watchlist: list[str] | None = None,
        cache_dir: str = "data_cache",
    ) -> None:
        """
        Args:
            enabled_categories: Kategorier der er aktive (None = alle).
            watchlist: Brugerens personlige watchlist (overskriver kategori-filter).
            cache_dir: Mappe til metadata-cache.
        """
        self._categories = dict(self._DEFAULT_CATEGORIES)
        self._enabled = set(
            self._categories.keys() if enabled_categories is None
            else enabled_categories
        )
        self._watchlist = list(watchlist or [])
        self._excluded: set[str] = set()

        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Kategori-management ───────────────────────────────────

    def enable_category(self, name: str) -> None:
        """Aktivér en kategori."""
        if name in self._categories:
            self._enabled.add(name)
            logger.info(f"[universe] Aktiveret: {name}")

    def disable_category(self, name: str) -> None:
        """Deaktivér en kategori."""
        self._enabled.discard(name)
        logger.info(f"[universe] Deaktiveret: {name}")

    def get_category(self, name: str) -> AssetCategory | None:
        """Hent en kategori efter navn."""
        return self._categories.get(name)

    @property
    def enabled_categories(self) -> list[AssetCategory]:
        """Alle aktive kategorier."""
        return [
            self._categories[name]
            for name in sorted(self._enabled)
            if name in self._categories
        ]

    @property
    def all_categories(self) -> list[AssetCategory]:
        """Alle kategorier (også deaktiverede)."""
        return list(self._categories.values())

    # ── Symbol-opslag ─────────────────────────────────────────

    @property
    def active_symbols(self) -> list[str]:
        """Alle aktive symboler fra aktiverede kategorier."""
        if self._watchlist:
            return [s for s in self._watchlist if s not in self._excluded]

        symbols = []
        for cat in self.enabled_categories:
            symbols.extend(cat.all_symbols)

        # Fjern duplikater og ekskluderede
        seen = set()
        unique = []
        for s in symbols:
            if s not in seen and s not in self._excluded:
                seen.add(s)
                unique.append(s)
        return unique

    @property
    def tradeable_symbols(self) -> list[str]:
        """Kun symboler der kan handles (ekskl. yields, forex-reference)."""
        symbols = []
        for cat in self.enabled_categories:
            if cat.tradeable:
                symbols.extend(cat.all_symbols)

        seen = set()
        unique = []
        for s in symbols:
            if s not in seen and s not in self._excluded:
                seen.add(s)
                unique.append(s)
        return unique

    def get_symbols_for_category(self, category: str) -> list[str]:
        """Hent symboler for en specifik kategori."""
        cat = self._categories.get(category)
        return cat.all_symbols if cat else []

    def get_symbols_for_subcategory(
        self, category: str, subcategory: str,
    ) -> list[str]:
        """Hent symboler for en specifik underkategori."""
        cat = self._categories.get(category)
        if not cat:
            return []
        items = cat.subcategories.get(subcategory, [])
        return [
            item[0] if isinstance(item, tuple) else item
            for item in items
        ]

    # ── Watchlist ─────────────────────────────────────────────

    @property
    def watchlist(self) -> list[str]:
        return list(self._watchlist)

    def set_watchlist(self, symbols: list[str]) -> None:
        """Sæt personlig watchlist (overskriver kategori-filter)."""
        self._watchlist = list(symbols)
        logger.info(f"[universe] Watchlist sat: {len(symbols)} symboler")

    def add_to_watchlist(self, symbol: str) -> None:
        if symbol not in self._watchlist:
            self._watchlist.append(symbol)

    def remove_from_watchlist(self, symbol: str) -> None:
        if symbol in self._watchlist:
            self._watchlist.remove(symbol)

    def clear_watchlist(self) -> None:
        self._watchlist.clear()

    # ── Ekskludering ──────────────────────────────────────────

    def exclude(self, symbol: str) -> None:
        self._excluded.add(symbol)

    def include(self, symbol: str) -> None:
        self._excluded.discard(symbol)

    # ── Filtrering ────────────────────────────────────────────

    def filter_by_region(self, region: str) -> list[str]:
        """
        Filtrér symboler efter region.

        Regioner: us, nordic, europe, asia, emerging, global
        """
        region_map = {
            "us": ["us_stocks"],
            "nordic": ["nordic_stocks"],
            "europe": ["european_stocks"],
            "asia": ["asian_stocks"],
            "emerging": ["emerging_markets"],
            "global": list(self._categories.keys()),
        }
        cats = region_map.get(region.lower(), [])
        symbols = []
        for cat_name in cats:
            cat = self._categories.get(cat_name)
            if cat:
                symbols.extend(cat.all_symbols)
        return symbols

    def filter_by_asset_class(self, asset_class: str) -> list[str]:
        """
        Filtrér symboler efter aktivklasse.

        Klasser: stocks, etfs, commodities, crypto, bonds, forex
        """
        class_map = {
            "stocks": [
                "us_stocks", "nordic_stocks", "european_stocks",
                "asian_stocks", "emerging_markets",
            ],
            "etfs": ["etfs"],
            "commodities": ["commodities"],
            "crypto": ["crypto"],
            "bonds": ["bonds"],
            "forex": ["forex"],
        }
        cats = class_map.get(asset_class.lower(), [])
        symbols = []
        for cat_name in cats:
            cat = self._categories.get(cat_name)
            if cat:
                symbols.extend(cat.all_symbols)
        return symbols

    def get_24h_symbols(self) -> list[str]:
        """Symboler der handler 24/7 (krypto, forex)."""
        symbols = []
        for cat in self.enabled_categories:
            if cat.supports_24h:
                symbols.extend(cat.all_symbols)
        return symbols

    # ── Scan-mode ─────────────────────────────────────────────

    def scan_universe(
        self,
        categories: list[str] | None = None,
        max_symbols: int = 0,
    ) -> list[str]:
        """
        Returnér symboler til fuld univers-scanning.

        Args:
            categories: Specifik kategori-liste (default: alle aktive).
            max_symbols: Maks antal symboler (0 = ubegrænset).

        Returns:
            Liste af symboler klar til screening.
        """
        if categories:
            symbols = []
            for cat_name in categories:
                cat = self._categories.get(cat_name)
                if cat:
                    symbols.extend(cat.all_symbols)
        else:
            symbols = self.active_symbols

        if max_symbols > 0:
            symbols = symbols[:max_symbols]

        logger.info(
            f"[universe] Scan: {len(symbols)} symboler "
            f"fra {len(categories or self._enabled)} kategorier"
        )
        return symbols

    # ── Statistik ─────────────────────────────────────────────

    def summary(self) -> dict:
        """Returnér oversigt over universet."""
        total = 0
        by_category = {}
        for cat in self.all_categories:
            count = cat.symbol_count
            total += count
            by_category[cat.name] = {
                "display_name": cat.display_name,
                "symbols": count,
                "enabled": cat.name in self._enabled,
                "tradeable": cat.tradeable,
                "24h": cat.supports_24h,
            }

        return {
            "total_symbols": total,
            "active_symbols": len(self.active_symbols),
            "enabled_categories": len(self._enabled),
            "total_categories": len(self._categories),
            "watchlist_size": len(self._watchlist),
            "excluded": len(self._excluded),
            "categories": by_category,
        }

    def print_summary(self) -> None:
        """Print en oversigt over universet."""
        s = self.summary()
        print(f"\n{'═' * 65}")
        print(f"  ASSET UNIVERSE – {s['total_symbols']} aktiver i alt")
        print(f"{'═' * 65}")
        print(f"  Aktive symboler:    {s['active_symbols']}")
        print(f"  Aktive kategorier:  {s['enabled_categories']}/{s['total_categories']}")
        if s['watchlist_size']:
            print(f"  Watchlist:          {s['watchlist_size']} symboler")
        print()

        for name, info in s["categories"].items():
            status = "✅" if info["enabled"] else "❌"
            extra = ""
            if info["24h"]:
                extra += " 🕐24/7"
            if not info["tradeable"]:
                extra += " 📊reference"
            print(
                f"  {status} {info['display_name']:<25} "
                f"{info['symbols']:>5} symboler{extra}"
            )

        print(f"{'═' * 65}\n")

    # ── Iterator ──────────────────────────────────────────────

    def __iter__(self) -> Iterator[str]:
        return iter(self.active_symbols)

    def __len__(self) -> int:
        return len(self.active_symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.active_symbols
