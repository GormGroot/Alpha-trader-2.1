"""
Abstrakt base-klasse for broker-integrationer.

Alle brokers arver fra BaseBroker og implementerer:
  - buy()              → Order
  - sell()             → Order
  - get_positions()    → list[Position]
  - get_account()      → AccountInfo
  - get_order_status() → Order
  - cancel_order()     → bool
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.broker.models import (
    AccountInfo,
    Order,
    OrderType,
    OrderValidationError,
)
from src.risk.portfolio_tracker import Position


class BaseBroker(ABC):
    """Abstrakt base-klasse for alle broker-implementationer."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unikt navn for brokeren (f.eks. 'paper', 'alpaca')."""

    @abstractmethod
    def buy(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        """Placér en købsordre."""

    @abstractmethod
    def sell(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        short: bool = False,
    ) -> Order:
        """Placér en salgsordre. Hvis short=True, åbn en short position."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Hent alle åbne positioner."""

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Hent kontoinformation (cash, equity, buying power)."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Hent status for en specifik ordre."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Annullér en ventende ordre. Returnerer True hvis succesfuldt."""

    # ── Fælles validering ─────────────────────────────────────

    def _validate_order(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType,
        limit_price: float | None,
    ) -> None:
        """Validér ordre-parametre før afsendelse."""
        if not symbol or not symbol.strip():
            raise OrderValidationError("Symbol må ikke være tomt")
        if qty <= 0:
            raise OrderValidationError(f"Antal skal være > 0, fik {qty}")
        if order_type == OrderType.LIMIT and limit_price is None:
            raise OrderValidationError("Limit-pris kræves for limit-ordrer")
        if limit_price is not None and limit_price <= 0:
            raise OrderValidationError(f"Limit-pris skal være > 0, fik {limit_price}")
