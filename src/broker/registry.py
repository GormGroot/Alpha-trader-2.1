"""
Global broker registry — deler BrokerRouter mellem trading engine og dashboard.

Brug:
    # I main.py (trading engine):
    from src.broker.registry import set_router
    router = setup_brokers(paper=True)
    set_router(router)

    # I dashboard pages:
    from src.broker.registry import get_router
    router = get_router()  # Samme instans med alle positioner
"""

from __future__ import annotations

_router = None
_auto_trader = None


def set_router(router) -> None:
    """Registrer den aktive BrokerRouter globalt."""
    global _router
    _router = router


def get_router():
    """Hent den aktive BrokerRouter. Returnerer None hvis ikke sat."""
    return _router


def set_auto_trader(trader) -> None:
    """Registrer den aktive AutoTrader globalt."""
    global _auto_trader
    _auto_trader = trader


def get_auto_trader():
    """Hent den aktive AutoTrader. Returnerer None hvis ikke sat."""
    return _auto_trader
