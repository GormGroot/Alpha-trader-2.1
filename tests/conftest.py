"""
Shared test fixtures.

Ensures tests use isolated temp databases instead of the live
paper_portfolio.db used by the running trader.
"""

import pytest
import src.risk.portfolio_tracker as pt_mod


@pytest.fixture(autouse=True)
def _isolate_portfolio_db(tmp_path, monkeypatch):
    """Redirect PortfolioTracker's default DB to a temp directory."""
    tmp_db = str(tmp_path / "paper_portfolio.db")
    monkeypatch.setattr(pt_mod, "_DEFAULT_DB_PATH", tmp_db)
    # Also patch the default arg already bound in __init__ signatures
    monkeypatch.setattr(pt_mod.PortfolioTracker.__init__, "__defaults__",
                        (100_000, tmp_db))
    monkeypatch.setattr(pt_mod.PortfolioDB.__init__, "__defaults__",
                        (tmp_db,))
