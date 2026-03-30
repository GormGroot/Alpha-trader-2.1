"""
Currency display service for the trading dashboard.

Manages the user's display currency preference (USD, EUR, DKK), fetches
live FX rates from yfinance every 30 minutes, and provides conversion
and formatting helpers used across all dashboard pages.

All internal platform values are denominated in USD (from PaperBroker).
"""

import json
import os
import time
import threading
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_CURRENCIES = ("USD", "EUR", "DKK")
DEFAULT_CURRENCY = "DKK"

FALLBACK_USD_DKK = 6.90
FALLBACK_USD_EUR = 0.92

CACHE_TTL_SECONDS = 30 * 60  # 30 minutes

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PERSISTENCE_FILE = CONFIG_DIR / "display_currency.json"

# ---------------------------------------------------------------------------
# Module-level state (singleton)
# ---------------------------------------------------------------------------

_display_currency: str = DEFAULT_CURRENCY

_rates: dict = {
    "usd_dkk": FALLBACK_USD_DKK,
    "usd_eur": FALLBACK_USD_EUR,
    "eur_dkk": FALLBACK_USD_DKK / FALLBACK_USD_EUR,
}

_rates_timestamp: float = 0.0
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _load_persisted_currency() -> str:
    """Load the saved currency preference from disk, or return the default."""
    try:
        if PERSISTENCE_FILE.exists():
            data = json.loads(PERSISTENCE_FILE.read_text(encoding="utf-8"))
            code = data.get("currency", DEFAULT_CURRENCY).upper()
            if code in SUPPORTED_CURRENCIES:
                return code
            logger.warning("Persisted currency '{}' not supported, using {}", code, DEFAULT_CURRENCY)
    except Exception as exc:
        logger.warning("Failed to load display currency config: {}", exc)
    return DEFAULT_CURRENCY


def _save_currency(code: str) -> None:
    """Persist the currency choice to config/display_currency.json."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PERSISTENCE_FILE.write_text(
            json.dumps({"currency": code}, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Display currency saved: {}", code)
    except Exception as exc:
        logger.error("Failed to save display currency: {}", exc)


# ---------------------------------------------------------------------------
# Rate fetching
# ---------------------------------------------------------------------------


def refresh_rates() -> None:
    """Fetch fresh FX rates from yfinance and update the cache.

    Rates fetched:
      - DKK=X   -> USD/DKK (how many DKK per 1 USD)
      - EURUSD=X -> EUR/USD (how many USD per 1 EUR), inverted to get USD/EUR
    """
    global _rates, _rates_timestamp

    try:
        import yfinance as yf

        usd_dkk = FALLBACK_USD_DKK
        usd_eur = FALLBACK_USD_EUR

        # Fetch USD/DKK -------------------------------------------------
        try:
            ticker_dkk = yf.Ticker("DKK=X")
            info = ticker_dkk.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            if price and price > 0:
                usd_dkk = float(price)
                logger.debug("Fetched USD/DKK: {:.4f}", usd_dkk)
            else:
                logger.warning("USD/DKK price unavailable, using fallback {}", FALLBACK_USD_DKK)
        except Exception as exc:
            logger.warning("Failed to fetch USD/DKK: {}, using fallback", exc)

        # Fetch EUR/USD (then invert to USD/EUR) ------------------------
        try:
            ticker_eur = yf.Ticker("EURUSD=X")
            info = ticker_eur.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            if price and price > 0:
                usd_eur = 1.0 / float(price)
                logger.debug("Fetched USD/EUR: {:.4f} (EUR/USD={:.4f})", usd_eur, float(price))
            else:
                logger.warning("EUR/USD price unavailable, using fallback {}", FALLBACK_USD_EUR)
        except Exception as exc:
            logger.warning("Failed to fetch EUR/USD: {}, using fallback", exc)

        eur_dkk = usd_dkk / usd_eur if usd_eur > 0 else FALLBACK_USD_DKK / FALLBACK_USD_EUR

        with _lock:
            _rates = {
                "usd_dkk": usd_dkk,
                "usd_eur": usd_eur,
                "eur_dkk": eur_dkk,
            }
            _rates_timestamp = time.time()

        logger.info(
            "FX rates refreshed — USD/DKK={:.4f}  USD/EUR={:.4f}  EUR/DKK={:.4f}",
            usd_dkk, usd_eur, eur_dkk,
        )

    except ImportError:
        logger.error("yfinance not installed — using fallback FX rates")
    except Exception as exc:
        logger.error("Unexpected error refreshing FX rates: {}", exc)


def _ensure_rates() -> None:
    """Refresh rates if the cache has expired."""
    if time.time() - _rates_timestamp > CACHE_TTL_SECONDS:
        refresh_rates()


# ---------------------------------------------------------------------------
# Public API — currency selection
# ---------------------------------------------------------------------------


def get_display_currency() -> str:
    """Return the current display currency code (USD, EUR, or DKK)."""
    return _display_currency


def set_display_currency(code: str, persist: bool = True) -> None:
    """Set the display currency.

    Args:
        code: One of "USD", "EUR", "DKK" (case-insensitive).
        persist: If True, save to config file. False for syncing from browser store.
    """
    global _display_currency
    code = code.upper().strip()
    if code not in SUPPORTED_CURRENCIES:
        logger.error("Unsupported currency '{}'. Supported: {}", code, SUPPORTED_CURRENCIES)
        return
    if _display_currency == code:
        return  # no change
    _display_currency = code
    if persist:
        _save_currency(code)
    logger.info("Display currency set to {}", code)


def get_currency_symbol() -> str:
    """Return the symbol for the current display currency."""
    return {"USD": "$", "EUR": "\u20ac", "DKK": "kr"}[_display_currency]


def get_currency_label() -> str:
    """Return the ISO code for the current display currency."""
    return _display_currency


# ---------------------------------------------------------------------------
# Public API — rates & conversion
# ---------------------------------------------------------------------------


def get_fx_rates() -> dict:
    """Return the cached FX rates, refreshing first if stale.

    Returns:
        dict with keys ``usd_dkk``, ``usd_eur``, ``eur_dkk``.
    """
    _ensure_rates()
    with _lock:
        return dict(_rates)


def convert_from_usd(amount: float) -> float:
    """Convert a USD amount to the current display currency.

    Args:
        amount: Value in USD.

    Returns:
        Equivalent value in the display currency.
    """
    if _display_currency == "USD":
        return amount
    _ensure_rates()
    with _lock:
        if _display_currency == "DKK":
            return amount * _rates["usd_dkk"]
        if _display_currency == "EUR":
            return amount * _rates["usd_eur"]
    return amount


def convert_from_dkk(amount: float) -> float:
    """Convert a DKK amount to the current display currency.

    Args:
        amount: Value in DKK.

    Returns:
        Equivalent value in the display currency.
    """
    if _display_currency == "DKK":
        return amount
    _ensure_rates()
    with _lock:
        usd_dkk = _rates["usd_dkk"]
        if _display_currency == "USD":
            return amount / usd_dkk if usd_dkk > 0 else amount
        if _display_currency == "EUR":
            eur_dkk = _rates["eur_dkk"]
            return amount / eur_dkk if eur_dkk > 0 else amount
    return amount


# ---------------------------------------------------------------------------
# Public API — formatting
# ---------------------------------------------------------------------------


def _format_number(value: float, decimals: int) -> str:
    """Format a number with thousand separators appropriate to the currency.

    DKK/EUR use dot as thousand separator and comma for decimals.
    USD uses comma as thousand separator and dot for decimals.
    """
    if _display_currency in ("DKK", "EUR"):
        # European formatting: 1.234,56
        if decimals > 0:
            formatted = f"{value:,.{decimals}f}"
            # Swap separators: comma -> @, dot -> comma, @ -> dot
            formatted = formatted.replace(",", "@").replace(".", ",").replace("@", ".")
        else:
            formatted = f"{value:,.0f}"
            formatted = formatted.replace(",", ".")
    else:
        # US formatting: 1,234.56
        if decimals > 0:
            formatted = f"{value:,.{decimals}f}"
        else:
            formatted = f"{value:,.0f}"
    return formatted


def format_value(amount_usd: float, decimals: int = 0) -> str:
    """Convert a USD amount to the display currency and return a formatted string.

    Examples (depending on display currency):
        format_value(1234.5)       -> "8.518 kr"  (DKK)
        format_value(1234.5)       -> "$1,235"     (USD)
        format_value(1234.5, 2)    -> "1.135,74 €" ... no, euro symbol before -> "€1,135.74" ...

    Symbol placement:
        - USD: "$1,234"
        - EUR: "€1.234"  (symbol before, European number format)
        - DKK: "1.234 kr" (symbol after)

    Args:
        amount_usd: Value in USD.
        decimals: Number of decimal places (default 0).

    Returns:
        Formatted string with currency symbol.
    """
    converted = convert_from_usd(amount_usd)
    formatted = _format_number(converted, decimals)

    if _display_currency == "USD":
        return f"${formatted}"
    elif _display_currency == "EUR":
        return f"\u20ac{formatted}"
    else:  # DKK
        return f"{formatted} kr"


def format_value_dkk(amount_dkk: float, decimals: int = 0) -> str:
    """Convert a DKK amount to the display currency and return a formatted string.

    Same formatting rules as :func:`format_value` but the input is in DKK.

    Args:
        amount_dkk: Value in DKK.
        decimals: Number of decimal places (default 0).

    Returns:
        Formatted string with currency symbol.
    """
    converted = convert_from_dkk(amount_dkk)
    formatted = _format_number(converted, decimals)

    if _display_currency == "USD":
        return f"${formatted}"
    elif _display_currency == "EUR":
        return f"\u20ac{formatted}"
    else:  # DKK
        return f"{formatted} kr"


# ---------------------------------------------------------------------------
# Module initialisation
# ---------------------------------------------------------------------------

# Load persisted preference on import
_display_currency = _load_persisted_currency()
logger.debug("Currency service initialised — display currency: {}", _display_currency)
