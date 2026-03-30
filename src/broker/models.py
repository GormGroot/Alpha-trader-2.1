"""
Broker-datamodeller – ordrer, kontostatus og fejltyper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# ── Dataklasser ───────────────────────────────────────────────

@dataclass
class Order:
    """Repræsenterer en ordre i systemet."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    status: OrderStatus = OrderStatus.PENDING
    limit_price: float | None = None
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    submitted_at: str = ""
    filled_at: str = ""
    error_message: str = ""


@dataclass
class AccountInfo:
    """Kontoinformation fra broker."""

    account_id: str
    cash: float
    portfolio_value: float
    buying_power: float
    equity: float
    currency: str = "USD"


# ── Exceptions ────────────────────────────────────────────────

class BrokerError(Exception):
    """Generel broker-fejl."""


class OrderValidationError(BrokerError):
    """Fejl ved ordre-validering (ugyldigt symbol, qty <= 0, osv.)."""


class InsufficientFundsError(BrokerError):
    """Ikke nok købekraft til ordren."""
