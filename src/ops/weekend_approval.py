"""
Weekend Crypto Rollover Approval — user must accept/reject before activation.

On Friday before session close, the system calculates the estimated extra
fees for rolling into crypto over the weekend. A notification is sent and
the dashboard shows an accept/reject modal. The rollover only proceeds
if the user explicitly approves.

State is persisted in data_cache/weekend_approval.json so it survives
restarts and can be read by both the scheduler and the dashboard.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

_STATE_PATH = Path("data_cache/weekend_approval.json")


def _read_state() -> dict:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text())
    except Exception:
        pass
    return {}


def _write_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def request_approval(
    estimated_fees: float,
    crypto_allocation_pct: int,
    positions_to_close: list[dict],
    crypto_symbols: list[str],
    reopen_info: str,
) -> None:
    """Create a pending approval request for the weekend rollover."""
    state = {
        "status": "pending",
        "requested_at": datetime.now().isoformat(),
        "estimated_fees": estimated_fees,
        "crypto_allocation_pct": crypto_allocation_pct,
        "positions_to_close": positions_to_close,
        "crypto_symbols": crypto_symbols,
        "reopen_info": reopen_info,
        "responded_at": None,
    }
    _write_state(state)
    logger.info(
        f"[weekend] Approval requested — estimated fees: ${estimated_fees:,.2f}, "
        f"crypto allocation: {crypto_allocation_pct}%"
    )


def get_approval_state() -> dict:
    """Read the current approval state. Returns empty dict if none."""
    return _read_state()


def approve() -> None:
    """User accepts the weekend crypto rollover."""
    state = _read_state()
    state["status"] = "approved"
    state["responded_at"] = datetime.now().isoformat()
    _write_state(state)
    logger.info("[weekend] User APPROVED weekend crypto rollover")


def reject() -> None:
    """User rejects the weekend crypto rollover."""
    state = _read_state()
    state["status"] = "rejected"
    state["responded_at"] = datetime.now().isoformat()
    _write_state(state)
    logger.info("[weekend] User REJECTED weekend crypto rollover")


def clear() -> None:
    """Clear approval state (after weekend ends)."""
    if _STATE_PATH.exists():
        _STATE_PATH.unlink()
    logger.debug("[weekend] Approval state cleared")


def is_pending() -> bool:
    state = _read_state()
    return state.get("status") == "pending"


def is_approved() -> bool:
    state = _read_state()
    return state.get("status") == "approved"


def is_rejected() -> bool:
    state = _read_state()
    return state.get("status") == "rejected"
