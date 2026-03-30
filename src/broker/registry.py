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

import threading

_lock = threading.Lock()
_router = None
_auto_trader = None
_order_manager = None


def set_router(router) -> None:
    """Registrer den aktive BrokerRouter globalt."""
    global _router
    with _lock:
        _router = router


def get_router():
    """Hent den aktive BrokerRouter. Returnerer None hvis ikke sat."""
    with _lock:
        return _router


def set_auto_trader(trader) -> None:
    """Registrer den aktive AutoTrader globalt."""
    global _auto_trader
    with _lock:
        _auto_trader = trader


def get_auto_trader():
    """Hent den aktive AutoTrader. Returnerer None hvis ikke sat."""
    with _lock:
        return _auto_trader


def set_order_manager(mgr) -> None:
    """Registrer den aktive OrderManager globalt."""
    global _order_manager
    with _lock:
        _order_manager = mgr


def get_order_manager():
    """Hent den aktive OrderManager. Returnerer None hvis ikke sat."""
    with _lock:
        return _order_manager
