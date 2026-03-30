"""
Dansk skatteberegning for aktiehandel.

Beregner realiseret gevinst/tab efter FIFO-princippet og
anvender danske skattesatser for aktieindkomst:
  - 27% op til progressionsgrænsen
  - 42% over progressionsgrænsen

VIGTIGT: Denne beregning er vejledende. Konsultér altid en revisor.
Skatteregler kan ændre sig fra år til år.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger


# ── Dataklasser ───────────────────────────────────────────────

@dataclass
class TaxLot:
    """Én skattemæssig post (FIFO-lot)."""

    symbol: str
    qty: float
    acquisition_price_dkk: float   # Anskaffelsessum per aktie i DKK
    acquisition_date: str
    disposal_price_dkk: float = 0.0
    disposal_date: str = ""
    gain_dkk: float = 0.0


@dataclass
class TaxResult:
    """
    Samlet skatteberegning for et år.

    BEMÆRK: Beregningen er vejledende og bør verificeres af en revisor.
    """

    year: int

    # Bruttobeløb
    total_gains_dkk: float = 0.0      # Samlede gevinster
    total_losses_dkk: float = 0.0     # Samlede tab (negativt tal)
    net_gain_dkk: float = 0.0         # Netto gevinst/tab

    # Tab fra tidligere år
    carried_loss_dkk: float = 0.0     # Uudnyttede tab overført
    loss_utilized_dkk: float = 0.0    # Tab modregnet i årets gevinst
    remaining_loss_dkk: float = 0.0   # Tab til fremførsel

    # Skattepligtig indkomst
    taxable_gain_dkk: float = 0.0     # Efter modregning af tab

    # Skatteberegning (aktieindkomst)
    progression_limit: float = 0.0    # Progressionsgrænsen
    tax_low_bracket: float = 0.0      # 27% af beløb under grænsen
    tax_high_bracket: float = 0.0     # 42% af beløb over grænsen
    total_tax_dkk: float = 0.0        # Samlet estimeret skat

    # Udbytte
    dividend_gross_dkk: float = 0.0   # Bruttobeløb udbytte
    dividend_us_tax_dkk: float = 0.0  # Tilbageholdt US kildeskat
    dividend_dk_credit: float = 0.0   # Creditlempelse

    # Rubrikker til årsopgørelsen
    rubrik_66: float = 0.0            # Aktieindkomst – gevinst
    rubrik_67: float = 0.0            # Aktieindkomst – tab
    rubrik_68: float = 0.0            # Udenlandsk udbytte

    # Detaljer
    lots: list[TaxLot] = field(default_factory=list)
    per_symbol: dict = field(default_factory=dict)

    # Meta
    disclaimer: str = (
        "⚠️ VEJLEDENDE BEREGNING – Denne skatteberegning er kun vejledende "
        "og erstatter ikke professionel rådgivning. Skatteregler kan ændre "
        "sig. Verificér altid med en revisor eller SKAT inden indberetning."
    )


# ── Kalkulator ────────────────────────────────────────────────

class DanishTaxCalculator:
    """
    Beregner dansk skat på aktiehandel.

    Anvender FIFO-princippet (First In, First Out) og danske
    skattesatser for aktieindkomst.
    """

    # Skattesatser
    LOW_RATE = 0.27    # 27% under progressionsgrænsen
    HIGH_RATE = 0.42   # 42% over progressionsgrænsen

    # US kildeskat på udbytte
    US_WITHHOLDING_RATE = 0.15  # 15% (med W-8BEN)

    def __init__(
        self,
        progression_limit: float = 61_000,
        carried_losses: float = 0.0,
    ) -> None:
        """
        Args:
            progression_limit: Progressionsgrænsen i DKK (2026: ca. 61.000 kr).
            carried_losses: Uudnyttede tab fra tidligere år i DKK (positivt tal).
        """
        self._progression_limit = progression_limit
        self._carried_losses = abs(carried_losses)

    # ── Hovedberegning ────────────────────────────────────────

    def calculate(
        self,
        transactions: list[dict],
        dividends: list[dict] | None = None,
        year: int = 2026,
    ) -> TaxResult:
        """
        Beregn skat for et skatteår.

        Args:
            transactions: Liste af dicts med nøgler:
                - symbol, qty, entry_value_dkk, exit_value_dkk,
                  entry_date, trade_date, realized_pnl_dkk
            dividends: Liste af dicts med nøgler:
                - symbol, gross_dkk, us_tax_dkk, date
            year: Skatteåret.

        Returns:
            TaxResult med komplet beregning.
        """
        result = TaxResult(year=year, progression_limit=self._progression_limit)

        # ── 1. Beregn gevinst/tab per handel (FIFO) ──
        per_symbol: dict[str, dict] = {}
        for tx in transactions:
            sym = tx["symbol"]
            pnl = tx["realized_pnl_dkk"]

            if sym not in per_symbol:
                per_symbol[sym] = {
                    "gains": 0.0, "losses": 0.0,
                    "num_trades": 0, "total_qty": 0.0,
                }

            per_symbol[sym]["num_trades"] += 1
            per_symbol[sym]["total_qty"] += tx["qty"]

            if pnl >= 0:
                result.total_gains_dkk += pnl
                per_symbol[sym]["gains"] += pnl
            else:
                result.total_losses_dkk += pnl
                per_symbol[sym]["losses"] += pnl

            result.lots.append(TaxLot(
                symbol=sym,
                qty=tx["qty"],
                acquisition_price_dkk=tx["entry_value_dkk"] / tx["qty"] if tx["qty"] > 0 else 0,
                acquisition_date=tx.get("entry_date", ""),
                disposal_price_dkk=tx["exit_value_dkk"] / tx["qty"] if tx["qty"] > 0 else 0,
                disposal_date=tx.get("trade_date", ""),
                gain_dkk=pnl,
            ))

        result.per_symbol = per_symbol

        # ── 2. Netto gevinst/tab ──
        result.net_gain_dkk = result.total_gains_dkk + result.total_losses_dkk

        # ── 3. Modregning af tab fra tidligere år ──
        result.carried_loss_dkk = self._carried_losses

        if result.net_gain_dkk > 0 and self._carried_losses > 0:
            utilizable = min(self._carried_losses, result.net_gain_dkk)
            result.loss_utilized_dkk = utilizable
            result.remaining_loss_dkk = self._carried_losses - utilizable
            result.taxable_gain_dkk = result.net_gain_dkk - utilizable
        elif result.net_gain_dkk > 0:
            result.taxable_gain_dkk = result.net_gain_dkk
            result.remaining_loss_dkk = 0.0
        else:
            # Nettotab – intet at beskatte, tab fremføres
            result.taxable_gain_dkk = 0.0
            result.remaining_loss_dkk = self._carried_losses + abs(result.net_gain_dkk)

        # ── 4. Skatteberegning (aktieindkomst) ──
        if result.taxable_gain_dkk > 0:
            if result.taxable_gain_dkk <= self._progression_limit:
                result.tax_low_bracket = result.taxable_gain_dkk * self.LOW_RATE
                result.tax_high_bracket = 0.0
            else:
                result.tax_low_bracket = self._progression_limit * self.LOW_RATE
                over = result.taxable_gain_dkk - self._progression_limit
                result.tax_high_bracket = over * self.HIGH_RATE

            result.total_tax_dkk = result.tax_low_bracket + result.tax_high_bracket

        # ── 5. Udbytte ──
        if dividends:
            for div in dividends:
                result.dividend_gross_dkk += div.get("gross_dkk", 0.0)
                result.dividend_us_tax_dkk += div.get("us_tax_dkk", 0.0)

            # Creditlempelse: US kildeskat kan modregnes i dansk skat
            # (max. den danske skat på udbyttet)
            dk_tax_on_dividends = result.dividend_gross_dkk * self.LOW_RATE
            result.dividend_dk_credit = min(
                result.dividend_us_tax_dkk,
                dk_tax_on_dividends,
            )

        # ── 6. Rubrikker til årsopgørelsen ──
        if result.net_gain_dkk > 0:
            result.rubrik_66 = result.net_gain_dkk
            result.rubrik_67 = 0.0
        else:
            result.rubrik_66 = 0.0
            result.rubrik_67 = abs(result.net_gain_dkk)

        result.rubrik_68 = result.dividend_gross_dkk

        logger.info(
            f"[skat] {year}: netto={result.net_gain_dkk:+,.0f} DKK, "
            f"skat={result.total_tax_dkk:,.0f} DKK, "
            f"tab fremført={result.remaining_loss_dkk:,.0f} DKK"
        )

        return result

    # ── Hjælpemetoder ─────────────────────────────────────────

    def estimate_tax(self, gain_dkk: float) -> float:
        """Hurtig estimering af skat på en given gevinst i DKK."""
        if gain_dkk <= 0:
            return 0.0
        if gain_dkk <= self._progression_limit:
            return gain_dkk * self.LOW_RATE
        return (
            self._progression_limit * self.LOW_RATE
            + (gain_dkk - self._progression_limit) * self.HIGH_RATE
        )

    @property
    def progression_limit(self) -> float:
        return self._progression_limit

    @property
    def carried_losses(self) -> float:
        return self._carried_losses
