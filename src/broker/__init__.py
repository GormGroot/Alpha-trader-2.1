"""
Broker-modul: abstraktionslag for handelsudførelse.

Brug create_broker() til at oprette den korrekte broker baseret på config:

    from src.broker import create_broker
    broker = create_broker()        # Læser settings.broker.provider
    broker = create_broker("paper") # Tvinger paper-broker

Skift mellem paper og live:
    1. config/default_config.yaml  →  broker.provider: paper | alpaca
    2. .env                        →  ALPACA_PROVIDER=paper
    3. I koden                     →  create_broker("alpaca")
"""

from src.broker.base_broker import BaseBroker
from src.broker.models import (
    AccountInfo,
    BrokerError,
    InsufficientFundsError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
)
from src.broker.paper_broker import PaperBroker
from src.broker.alpaca_broker import AlpacaBroker


def create_broker(provider: str | None = None, **kwargs) -> BaseBroker:
    """
    Factory: opret broker baseret på provider-navn.

    Args:
        provider: "paper" eller "alpaca". Default: fra settings.broker.provider.
        **kwargs: Videresendes til broker-konstruktøren.

    Returns:
        BaseBroker-instans.

    Raises:
        ValueError: Ukendt provider.
    """
    from config.settings import settings

    name = (provider or settings.broker.provider).lower().strip()

    if name == "paper":
        return PaperBroker(**kwargs)
    elif name == "alpaca":
        return AlpacaBroker(**kwargs)
    else:
        raise ValueError(
            f"Ukendt broker provider: '{name}'. Brug 'paper' eller 'alpaca'."
        )


__all__ = [
    "BaseBroker",
    "PaperBroker",
    "AlpacaBroker",
    "create_broker",
    "Order",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "AccountInfo",
    "BrokerError",
    "OrderValidationError",
    "InsufficientFundsError",
]
