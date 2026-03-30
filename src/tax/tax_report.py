"""
Skatteindberetningsrapport – årsrapport til SKAT.

Genererer detaljeret rapport med:
  - Oversigt over alle handler
  - Gevinst/tab per aktie
  - Skatteberegning med rubrik-angivelser
  - Eksport som PDF, CSV og dashboard-visning
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.tax.tax_calculator import DanishTaxCalculator, TaxResult
from src.tax.transaction_log import TransactionLog
from src.tax.currency import CurrencyConverter


@dataclass
class AnnualReport:
    """Samlet årsrapport for skatteindberetning."""

    year: int
    generated_at: str
    tax_result: TaxResult
    transactions_csv_path: str = ""
    report_txt_path: str = ""

    @property
    def summary_lines(self) -> list[str]:
        """Returnér rapport som liste af linjer."""
        r = self.tax_result
        lines = [
            "═" * 65,
            "  SKATTEINDBERETNING – VEJLEDENDE BEREGNING",
            "═" * 65,
            f"  Skatteår:               {self.year}",
            f"  Genereret:              {self.generated_at}",
            "",
            "  ⚠️  DISCLAIMER: Denne beregning er kun vejledende og",
            "     erstatter ikke professionel rådgivning fra revisor.",
            "",
            "─" * 65,
            "  HANDLER",
            "─" * 65,
            f"  Antal handler:          {len(r.lots)}",
            f"  Samlede gevinster:      {r.total_gains_dkk:>+14,.2f} DKK",
            f"  Samlede tab:            {r.total_losses_dkk:>+14,.2f} DKK",
            f"  Netto gevinst/tab:      {r.net_gain_dkk:>+14,.2f} DKK",
            "",
        ]

        # Per aktie
        if r.per_symbol:
            lines.append("─" * 65)
            lines.append("  GEVINST/TAB PER AKTIE")
            lines.append("─" * 65)
            lines.append(f"  {'Aktie':<10} {'Handler':>8} {'Gevinst':>14} {'Tab':>14} {'Netto':>14}")
            for sym, data in sorted(r.per_symbol.items()):
                net = data["gains"] + data["losses"]
                lines.append(
                    f"  {sym:<10} {data['num_trades']:>8} "
                    f"{data['gains']:>+14,.2f} {data['losses']:>+14,.2f} "
                    f"{net:>+14,.2f}"
                )
            lines.append("")

        # Tab fra tidligere år
        if r.carried_loss_dkk > 0:
            lines.append("─" * 65)
            lines.append("  TABSFREMFØRSEL")
            lines.append("─" * 65)
            lines.append(f"  Tab fra tidligere år:   {r.carried_loss_dkk:>14,.2f} DKK")
            lines.append(f"  Modregnet i år:         {r.loss_utilized_dkk:>14,.2f} DKK")
            lines.append(f"  Tab til fremførsel:     {r.remaining_loss_dkk:>14,.2f} DKK")
            lines.append("")

        # Skatteberegning
        lines.extend([
            "─" * 65,
            "  SKATTEBEREGNING (AKTIEINDKOMST)",
            "─" * 65,
            f"  Skattepligtig gevinst:  {r.taxable_gain_dkk:>14,.2f} DKK",
            f"  Progressionsgrænse:     {r.progression_limit:>14,.0f} DKK",
            "",
            f"  27% af beløb under grænsen:  {r.tax_low_bracket:>10,.2f} DKK",
            f"  42% af beløb over grænsen:   {r.tax_high_bracket:>10,.2f} DKK",
            f"  ────────────────────────────────────────",
            f"  Estimeret skat i alt:        {r.total_tax_dkk:>10,.2f} DKK",
            "",
        ])

        # Udbytte
        if r.dividend_gross_dkk > 0:
            lines.extend([
                "─" * 65,
                "  UDBYTTE",
                "─" * 65,
                f"  Bruttoudbytte:          {r.dividend_gross_dkk:>14,.2f} DKK",
                f"  US kildeskat (15%):     {r.dividend_us_tax_dkk:>14,.2f} DKK",
                f"  Creditlempelse:         {r.dividend_dk_credit:>14,.2f} DKK",
                "",
            ])

        # Rubrikker
        lines.extend([
            "─" * 65,
            "  RUBRIKKER TIL ÅRSOPGØRELSEN",
            "─" * 65,
            f"  Rubrik 66 (aktieindkomst – gevinst):    {r.rubrik_66:>10,.2f} DKK",
            f"  Rubrik 67 (aktieindkomst – tab):         {r.rubrik_67:>10,.2f} DKK",
            f"  Rubrik 68 (udenlandsk udbytte):          {r.rubrik_68:>10,.2f} DKK",
            "",
            "═" * 65,
            "  ⚠️  Husk: Verificér altid med revisor inden indberetning.",
            "═" * 65,
        ])

        return lines


class TaxReportGenerator:
    """
    Genererer skatteindberetningsrapporter.

    Samler data fra TransactionLog, beregner skat via DanishTaxCalculator,
    og eksporterer i flere formater.
    """

    def __init__(
        self,
        progression_limit: float = 61_000,
        carried_losses: float = 0.0,
        cache_dir: str = "data_cache",
    ) -> None:
        self._currency = CurrencyConverter(cache_dir=cache_dir)
        self._tx_log = TransactionLog(
            currency=self._currency, cache_dir=cache_dir,
        )
        self._calculator = DanishTaxCalculator(
            progression_limit=progression_limit,
            carried_losses=carried_losses,
        )
        self._reports_dir = Path("reports")
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    @property
    def transaction_log(self) -> TransactionLog:
        return self._tx_log

    @property
    def calculator(self) -> DanishTaxCalculator:
        return self._calculator

    @property
    def currency(self) -> CurrencyConverter:
        return self._currency

    # ── Generer rapport ───────────────────────────────────────

    def generate(
        self,
        year: int = 2026,
        dividends: list[dict] | None = None,
    ) -> AnnualReport:
        """
        Generer en komplet årsrapport.

        Args:
            year: Skatteåret.
            dividends: Evt. udbyttedata.

        Returns:
            AnnualReport med TaxResult og filstier.
        """
        # Hent transaktioner for året
        df = self._tx_log.get_transactions(year=year)

        if df.empty:
            logger.warning(f"[skat] Ingen transaktioner fundet for {year}")
            transactions = []
        else:
            transactions = df.to_dict("records")

        # Beregn skat
        tax_result = self._calculator.calculate(
            transactions=transactions,
            dividends=dividends,
            year=year,
        )

        report = AnnualReport(
            year=year,
            generated_at=datetime.now().isoformat(),
            tax_result=tax_result,
        )

        # Eksportér automatisk
        report.transactions_csv_path = self._export_csv(year, df)
        report.report_txt_path = self._export_txt(report)

        logger.info(
            f"[skat] Rapport genereret for {year}: "
            f"{len(transactions)} handler, "
            f"skat={tax_result.total_tax_dkk:,.0f} DKK"
        )

        return report

    # ── Eksport ───────────────────────────────────────────────

    def _export_csv(self, year: int, df) -> str:
        """Eksportér transaktionslog som CSV."""
        path = self._reports_dir / f"transaktioner_{year}.csv"
        if not df.empty:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            logger.info(f"[skat] CSV eksporteret: {path}")
        return str(path)

    def _export_txt(self, report: AnnualReport) -> str:
        """Eksportér rapport som tekstfil."""
        path = self._reports_dir / f"skatterapport_{report.year}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for line in report.summary_lines:
                f.write(line + "\n")
        logger.info(f"[skat] Rapport eksporteret: {path}")
        return str(path)

    def print_report(self, report: AnnualReport) -> None:
        """Print rapporten til konsollen."""
        for line in report.summary_lines:
            print(line)
