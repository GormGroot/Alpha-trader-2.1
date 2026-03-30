"""
SaxoBroker — Saxo Bank OpenAPI integration.

Primært til: EU ETF'er, obligationer, danske investeringsforeninger.
Auth: OAuth2 via SaxoAuthManager (saxo_auth.py).
API: https://gateway.saxobank.com/openapi/ (live) / sim/openapi/ (simulation)

Rate limit: 120 req/min per endpoint group.
Vi holder os konservativt til max 2 req/sec.
"""

from __future__ import annotations

import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from loguru import logger

from src.broker.base_broker import BaseBroker
from src.broker.models import (
    AccountInfo,
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
)
from src.broker.saxo_auth import AuthError, SaxoAuthManager, SaxoConfig
from src.risk.portfolio_tracker import Position


# ── Rate Limiter ────────────────────────────────────────────

class _RateLimiter:
    """Simpel rate limiter: max N requests per sekund."""

    def __init__(self, max_per_second: float = 2.0) -> None:
        self._min_interval = 1.0 / max_per_second
        self._last_request = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()


# ── Retry Decorator ─────────────────────────────────────────

def _retry_saxo(
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Callable:
    """Retry med exponential backoff for Saxo API-kald."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except AuthError:
                    # Auth-fejl: retry hjælper ikke
                    raise
                except BrokerError as exc:
                    if "422" in str(exc) or "400" in str(exc):
                        raise  # Validation fejl
                    last_exc = exc
                except Exception as exc:
                    last_exc = exc

                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"[saxo] API-fejl (forsøg {attempt + 1}/{max_retries}): "
                        f"{last_exc} — retry om {delay:.1f}s"
                    )
                    time.sleep(delay)

            raise BrokerError(
                f"Saxo API-fejl efter {max_retries} forsøg: {last_exc}"
            ) from last_exc

        return wrapper
    return decorator


# ── Status Mapping ──────────────────────────────────────────

_SAXO_STATUS_MAP: dict[str, OrderStatus] = {
    "working": OrderStatus.SUBMITTED,
    "filled": OrderStatus.FILLED,
    "cancelled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "partiallyfilled": OrderStatus.PARTIALLY_FILLED,
    "placed": OrderStatus.SUBMITTED,
    "notworking": OrderStatus.PENDING,
}

# Saxo AssetType mapping
_SAXO_ASSET_TYPES: dict[str, str] = {
    "stock": "Stock",
    "etf": "Etf",
    "bond": "Bond",
    "forex": "FxSpot",
    "cfd": "CfdOnStock",
    "future": "ContractFutures",
    "fund": "MutualFund",
    "option": "StockOption",
}


# ── SaxoBroker ──────────────────────────────────────────────

class SaxoBroker(BaseBroker):
    """
    Saxo Bank broker-integration via OpenAPI.

    Brug:
        config = SaxoConfig.from_env()
        auth = SaxoAuthManager(config)
        broker = SaxoBroker(auth, config)

        # Forbind (kræver gyldige tokens)
        broker.connect()

        # Brug standard BaseBroker interface
        positions = broker.get_positions()
        account = broker.get_account()
        order = broker.buy("VWCE.DE", qty=5, order_type=OrderType.LIMIT, limit_price=105.0)
    """

    def __init__(
        self,
        auth: SaxoAuthManager | None = None,
        config: SaxoConfig | None = None,
    ) -> None:
        self._config = config or SaxoConfig.from_env()
        self._auth = auth or SaxoAuthManager(self._config)
        self._rate_limiter = _RateLimiter(max_per_second=2.0)

        # Account info (populated after connect)
        self._account_key: str = self._config.account_key
        self._client_key: str = ""

        # Instrument cache: symbol → Uic mapping
        self._instrument_cache: dict[str, dict] = {}

        # Session
        self._connected = False

    @property
    def name(self) -> str:
        return "saxo"

    @property
    def base_url(self) -> str:
        return self._config.base_url

    # ── HTTP Layer ──────────────────────────────────────────

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict:
        """
        Authenticated HTTP request til Saxo OpenAPI.

        Inkluderer rate limiting og auto-refresh af tokens.
        """
        import requests

        self._rate_limiter.wait()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._auth.get_headers()

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=10,
            )

            if response.status_code == 429:
                # Rate limited — vent og retry
                retry_after = int(response.headers.get("Retry-After", "5"))
                logger.warning(f"[saxo] Rate limited — venter {retry_after}s")
                time.sleep(retry_after)
                return self._request(method, endpoint, params, json_data)

            if response.status_code == 401:
                # Token expired — refresh og retry
                logger.info("[saxo] Token expired — refresher...")
                self._auth.refresh_access_token()
                headers = self._auth.get_headers()
                response = requests.request(
                    method=method, url=url, headers=headers,
                    params=params, json=json_data, timeout=10,
                )

            if response.status_code >= 400:
                error_body = response.text[:500]
                raise BrokerError(
                    f"Saxo API {response.status_code}: {error_body}"
                )

            if response.status_code == 204:
                return {}  # No content (f.eks. delete)

            return response.json()

        except AuthError:
            raise
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"Saxo API request fejl: {exc}") from exc

    # ── Connection ──────────────────────────────────────────

    def connect(self) -> dict:
        """
        Forbind til Saxo og verificér tokens.

        Returns:
            Account info dict.

        Raises:
            AuthError: Hvis ikke authenticated.
            BrokerError: Hvis forbindelse fejler.
        """
        if not self._auth.is_authenticated():
            raise AuthError(
                "Saxo ikke authenticated. "
                f"Login her: {self._auth.get_authorization_url()}"
            )

        # Hent account info
        data = self._request("GET", "/port/v1/accounts/me")

        if "Data" in data and len(data["Data"]) > 0:
            account = data["Data"][0]
            self._account_key = account.get("AccountKey", self._account_key)
            self._client_key = account.get("ClientKey", "")
        elif "AccountKey" in data:
            self._account_key = data["AccountKey"]
            self._client_key = data.get("ClientKey", "")

        self._connected = True
        self._auth.start_auto_refresh()

        logger.info(
            f"[saxo] Forbundet — account: {self._account_key}, "
            f"env: {self._config.environment}"
        )

        return {
            "account_key": self._account_key,
            "client_key": self._client_key,
            "environment": self._config.environment,
        }

    # ── Instrument Lookup ───────────────────────────────────

    @_retry_saxo()
    def search_instruments(
        self,
        query: str,
        asset_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Søg efter instrumenter i Saxo.

        Args:
            query: Søgeterm (f.eks. "VWCE", "NOVO", "US Treasury").
            asset_types: Filter (f.eks. ["Stock", "Etf"]).

        Returns:
            Liste af instrument-dicts med Uic, Description, Symbol, etc.
        """
        params: dict[str, Any] = {
            "Keywords": query,
            "IncludeNonTradable": False,
        }
        if asset_types:
            params["AssetTypes"] = ",".join(asset_types)

        data = self._request("GET", "/ref/v1/instruments", params=params)
        instruments = data.get("Data", [])

        # Cache results
        for inst in instruments:
            symbol = inst.get("Symbol", "")
            if symbol:
                self._instrument_cache[symbol.upper()] = inst

        return instruments

    def _resolve_instrument(self, symbol: str) -> dict:
        """
        Resolve symbol til Saxo instrument med Uic.

        Prøver cache først, søger ellers.
        """
        upper = symbol.upper()

        # Check cache
        if upper in self._instrument_cache:
            return self._instrument_cache[upper]

        # Søg
        results = self.search_instruments(
            symbol,
            asset_types=["Stock", "Etf", "Bond", "MutualFund"],
        )

        if not results:
            raise OrderValidationError(
                f"Instrument '{symbol}' ikke fundet hos Saxo"
            )

        # Prøv exact match først
        for inst in results:
            if inst.get("Symbol", "").upper() == upper:
                self._instrument_cache[upper] = inst
                return inst

        # Tag første result
        best = results[0]
        self._instrument_cache[upper] = best
        logger.info(
            f"[saxo] Resolved '{symbol}' → "
            f"Uic={best.get('Identifier')}, "
            f"Desc={best.get('Description', '')[:50]}"
        )
        return best

    # ── BaseBroker Implementation ───────────────────────────

    @_retry_saxo()
    def buy(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        """Placér en købsordre via Saxo OpenAPI."""
        self._validate_order(symbol, qty, order_type, limit_price)
        instrument = self._resolve_instrument(symbol)

        order_data: dict[str, Any] = {
            "AccountKey": self._account_key,
            "Uic": instrument.get("Identifier"),
            "AssetType": instrument.get("AssetType", "Stock"),
            "BuySell": "Buy",
            "Amount": qty,
            "OrderType": "Market" if order_type == OrderType.MARKET else "Limit",
            "OrderDuration": {"DurationType": "DayOrder"},
            "ManualOrder": True,
        }
        if limit_price is not None:
            order_data["OrderPrice"] = limit_price

        data = self._request("POST", "/trade/v2/orders", json_data=order_data)
        order = self._map_order_response(data, symbol, OrderSide.BUY, qty, order_type, limit_price)

        logger.info(
            f"[saxo] KØB {qty} {symbol} ({order_type.value}) → {order.order_id}"
        )
        return order

    @_retry_saxo()
    def sell(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        """Placér en salgsordre via Saxo OpenAPI."""
        self._validate_order(symbol, qty, order_type, limit_price)
        instrument = self._resolve_instrument(symbol)

        order_data: dict[str, Any] = {
            "AccountKey": self._account_key,
            "Uic": instrument.get("Identifier"),
            "AssetType": instrument.get("AssetType", "Stock"),
            "BuySell": "Sell",
            "Amount": qty,
            "OrderType": "Market" if order_type == OrderType.MARKET else "Limit",
            "OrderDuration": {"DurationType": "DayOrder"},
            "ManualOrder": True,
        }
        if limit_price is not None:
            order_data["OrderPrice"] = limit_price

        data = self._request("POST", "/trade/v2/orders", json_data=order_data)
        order = self._map_order_response(data, symbol, OrderSide.SELL, qty, order_type, limit_price)

        logger.info(
            f"[saxo] SÆLG {qty} {symbol} ({order_type.value}) → {order.order_id}"
        )
        return order

    @_retry_saxo()
    def get_positions(self) -> list[Position]:
        """Hent alle åbne positioner fra Saxo."""
        data = self._request("GET", "/port/v1/positions/me")
        positions_data = data.get("Data", [])

        positions: list[Position] = []
        for p in positions_data:
            pos_base = p.get("PositionBase", {})
            pos_view = p.get("PositionView", {})

            symbol = p.get("DisplayAndFormat", {}).get("Symbol", "")
            if not symbol:
                symbol = str(pos_base.get("Uic", ""))

            amount = float(pos_base.get("Amount", 0))
            side = "long" if amount > 0 else "short"

            positions.append(Position(
                symbol=symbol,
                side=side,
                qty=abs(amount),
                entry_price=float(pos_view.get("AverageOpenPrice", 0)),
                entry_time="",
                current_price=float(pos_view.get("CurrentPrice", 0)),
            ))

        return positions

    @_retry_saxo()
    def get_account(self) -> AccountInfo:
        """Hent kontoinformation fra Saxo."""
        # Hent balancer
        balance_data = self._request("GET", "/port/v1/balances/me")

        cash = float(balance_data.get("CashBalance", 0))
        total_value = float(balance_data.get("TotalValue", 0))
        margin = float(balance_data.get("MarginAvailable", 0))
        currency = balance_data.get("Currency", "DKK")

        return AccountInfo(
            account_id=self._account_key or "saxo",
            cash=cash,
            portfolio_value=total_value,
            buying_power=margin,
            equity=total_value,
            currency=currency,
        )

    @_retry_saxo()
    def get_order_status(self, order_id: str) -> Order:
        """Hent status for en specifik ordre."""
        data = self._request("GET", "/port/v1/orders/me")
        orders = data.get("Data", [])

        for o in orders:
            if str(o.get("OrderId", "")) == str(order_id):
                return self._map_saxo_order(o)

        raise BrokerError(f"Ordre {order_id} ikke fundet hos Saxo")

    @_retry_saxo()
    def cancel_order(self, order_id: str) -> bool:
        """Annullér en ordre via Saxo."""
        try:
            self._request(
                "DELETE",
                f"/trade/v2/orders/{order_id}",
                params={"AccountKey": self._account_key},
            )
            logger.info(f"[saxo] Ordre {order_id} annulleret")
            return True
        except BrokerError as exc:
            logger.warning(f"[saxo] Cancel fejl for {order_id}: {exc}")
            return False

    # ── Mapping Helpers ─────────────────────────────────────

    def _map_order_response(
        self,
        data: dict,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType,
        limit_price: float | None,
    ) -> Order:
        """Map Saxo order-response til vores Order model."""
        order_id = str(data.get("OrderId", data.get("OrderIds", ["unknown"])[0]
                        if isinstance(data.get("OrderIds"), list) else "unknown"))

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            status=OrderStatus.SUBMITTED,
            limit_price=limit_price,
            submitted_at=datetime.now().isoformat(),
        )

    def _map_saxo_order(self, saxo_order: dict) -> Order:
        """Map en Saxo ordre-dict til vores Order model."""
        status_str = str(saxo_order.get("Status", "")).lower()
        status = _SAXO_STATUS_MAP.get(status_str, OrderStatus.PENDING)

        buy_sell = saxo_order.get("BuySell", "Buy")
        side = OrderSide.BUY if buy_sell == "Buy" else OrderSide.SELL

        order_type_str = saxo_order.get("OrderType", "Market")
        order_type = (
            OrderType.LIMIT if order_type_str == "Limit" else OrderType.MARKET
        )

        return Order(
            order_id=str(saxo_order.get("OrderId", "")),
            symbol=saxo_order.get("DisplayAndFormat", {}).get(
                "Symbol", str(saxo_order.get("Uic", ""))
            ),
            side=side,
            order_type=order_type,
            qty=float(saxo_order.get("Amount", 0)),
            status=status,
            limit_price=saxo_order.get("Price"),
            filled_qty=float(saxo_order.get("FilledAmount", 0)),
            filled_avg_price=float(saxo_order.get("FilledPrice", 0)),
        )

    # ── Extra Functionality ─────────────────────────────────

    def get_instruments(
        self,
        query: str,
        asset_types: list[str] | None = None,
    ) -> list[dict]:
        """Søg instrumenter (convenience wrapper)."""
        return self.search_instruments(query, asset_types)

    def status(self) -> dict[str, Any]:
        """Broker-status til dashboard."""
        return {
            "broker": "saxo",
            "connected": self._connected,
            "environment": self._config.environment,
            "account_key": self._account_key,
            "auth_status": self._auth.status(),
            "instrument_cache_size": len(self._instrument_cache),
        }
