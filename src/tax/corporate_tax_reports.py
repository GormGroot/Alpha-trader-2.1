"""
Corporate Tax Reports — rapporter til revisor.

Genererer:
  1. Årsrapport (data klar til Excel-export via xlsx skill)
  2. Kvartalsrapport (summary)
  3. CSV-export for revisor
  4. Dashboard data

Ark-struktur for årsrapport:
  Ark 1: Overblik (total P&L, skat, tilgodehavende)
  Ark 2: Alle handler (transaktionslog)
  Ark 3: Lagerbeskatning per position
  Ark 4: Udbytter og kildeskat
  Ark 5: Valutakursgevinster
  Ark 6: Skattetilgodehavende bevægelse
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# ── Dataklasser ─────────────────────────────────────────────

@dataclass
class AnnualReportData:
    """Alle data for årsrapporten i ét objekt."""
    year: int
    generated_at: str = ""

    # Ark 1: Overblik
    overview: dict[str, Any] = field(default_factory=dict)

    # Ark 2: Transaktionslog
    transactions: list[dict] = field(default_factory=list)

    # Ark 3: Lagerbeskatning
    mark_to_market: list[dict] = field(default_factory=list)

    # Ark 4: Udbytter
    dividends: list[dict] = field(default_factory=list)

    # Ark 5: Valutakursgevinster
    fx_gains: list[dict] = field(default_factory=list)

    # Ark 6: Skattetilgodehavende
    tax_credit_movements: list[dict] = field(default_factory=list)


# ── Report Generator ───────────────────────────────────────

class CorporateTaxReportGenerator:
    """
    Genererer skatterapporter for dansk selskab.

    Brug:
        from src.tax.corporate_tax import CorporateTaxCalculator
        from src.tax.mark_to_market import MarkToMarketEngine
        from src.tax.dividend_tracker import DividendTracker
        from src.tax.currency_pnl import CurrencyPnLTracker
        from src.tax.tax_credit_tracker import TaxCreditTracker

        reporter = CorporateTaxReportGenerator(
            tax_calc=calculator,
            mtm_engine=mtm,
            dividend_tracker=dividends,
            fx_tracker=fx,
            credit_tracker=credit,
        )

        # Årsrapport data (klar til Excel-export)
        report = reporter.generate_annual_report(2025)

        # CSV-export
        reporter.export_csv(2025, output_dir="reports/")

        # Dashboard data
        dashboard = reporter.get_dashboard_data(2025)
    """

    DISCLAIMER = (
        "DISCLAIMER: Denne rapport er vejledende og erstatter ikke "
        "professionel rådgivning. Konsultér altid din revisor."
    )

    def __init__(
        self,
        tax_calc: Any = None,
        mtm_engine: Any = None,
        dividend_tracker: Any = None,
        fx_tracker: Any = None,
        credit_tracker: Any = None,
    ) -> None:
        self._tax = tax_calc
        self._mtm = mtm_engine
        self._dividends = dividend_tracker
        self._fx = fx_tracker
        self._credit = credit_tracker

    # ── Annual Report ───────────────────────────────────────

    def generate_annual_report(
        self,
        year: int,
        tax_result: Any = None,
        mtm_summary: Any = None,
    ) -> AnnualReportData:
        """
        Generér fuld årsrapport.

        Samler data fra alle skattemodulerne.
        Output er struktureret data klar til Excel-export.
        """
        report = AnnualReportData(
            year=year,
            generated_at=datetime.now().isoformat(),
        )

        # Ark 1: Overblik
        report.overview = self._build_overview(year, tax_result)

        # Ark 2: Transaktionslog
        if self._tax and hasattr(self._tax, "fifo"):
            report.transactions = self._build_transaction_log(year)

        # Ark 3: Lagerbeskatning
        if mtm_summary:
            report.mark_to_market = self._build_mtm_sheet(mtm_summary)
        elif self._mtm:
            history = self._mtm.get_calculation_history(year)
            if history:
                report.mark_to_market = [
                    {
                        "dato": h["date"][:10],
                        "primo_dkk": h["primo"],
                        "ultimo_dkk": h["ultimo"],
                        "pnl_dkk": h["pnl"],
                        "estimeret_skat_dkk": h["tax"],
                        "antal_positioner": h["positions"],
                        "endelig": h["is_final"],
                    }
                    for h in history
                ]

        # Ark 4: Udbytter
        if self._dividends:
            report.dividends = self._build_dividend_sheet(year)

        # Ark 5: Valutakursgevinster
        if self._fx:
            report.fx_gains = self._build_fx_sheet(year)

        # Ark 6: Skattetilgodehavende
        if self._credit:
            report.tax_credit_movements = self._build_credit_sheet()

        logger.info(
            f"[tax-report] Årsrapport {year} genereret: "
            f"{len(report.transactions)} handler, "
            f"{len(report.dividends)} udbytter"
        )

        return report

    def _build_overview(self, year: int, tax_result: Any) -> dict:
        """Byg overblik-ark."""
        overview: dict[str, Any] = {
            "regnskabsår": year,
            "selskabsskat_sats": "22%",
            "genereret": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "disclaimer": self.DISCLAIMER,
        }

        if tax_result:
            if hasattr(tax_result, "to_dict"):
                overview.update(tax_result.to_dict())
            elif isinstance(tax_result, dict):
                overview.update(tax_result)

        # Tilføj credit info
        if self._credit:
            overview["skattetilgodehavende_balance"] = round(
                self._credit.balance, 2
            )

        # Tilføj dividend summary
        if self._dividends:
            div_summary = self._dividends.get_annual_summary(year)
            overview["udbytter_brutto_dkk"] = round(
                div_summary.total_gross_dkk, 2
            )
            overview["kildeskat_betalt_dkk"] = round(
                div_summary.total_withholding_dkk, 2
            )
            overview["kildeskat_kredit_dkk"] = round(
                div_summary.total_creditable_dkk, 2
            )

        # Tilføj FX summary
        if self._fx:
            fx_summary = self._fx.get_annual_summary(year)
            overview["valutakursgevinst_netto_dkk"] = round(
                fx_summary.net_fx_pnl_dkk, 2
            )

        return overview

    def _build_transaction_log(self, year: int) -> list[dict]:
        """Byg transaktionslog-ark."""
        # Hent fra FIFO lots database
        try:
            import sqlite3
            db_path = "data_cache/fifo_lots.db"
            if not Path(db_path).exists():
                return []

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, qty, price_dkk, acquired_at, broker "
                    "FROM lots WHERE acquired_at LIKE ? "
                    "ORDER BY acquired_at",
                    (f"{year}%",),
                ).fetchall()

            return [
                {
                    "symbol": r[0],
                    "antal": r[1],
                    "pris_dkk": round(r[2], 2),
                    "dato": r[3][:10] if r[3] else "",
                    "broker": r[4],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning(f"[tax-report] Transaktionslog fejl: {exc}")
            return []

    def _build_mtm_sheet(self, mtm_summary: Any) -> list[dict]:
        """Byg lagerbeskatnings-ark fra MTMSummary."""
        if not hasattr(mtm_summary, "positions"):
            return []

        return [
            {
                "symbol": p.symbol,
                "antal": p.qty,
                "primo_pris_dkk": round(p.primo_price_dkk, 2),
                "primo_værdi_dkk": round(p.primo_value_dkk, 2),
                "ultimo_pris_dkk": round(p.ultimo_price_dkk, 2),
                "ultimo_værdi_dkk": round(p.ultimo_value_dkk, 2),
                "lagerbeskatning_pnl_dkk": round(p.mtm_pnl_dkk, 2),
                "skat_22pct_dkk": round(p.tax_22pct_dkk, 2),
                "ny_position": p.is_new_position,
                "broker": p.broker,
                "valuta": p.currency,
            }
            for p in mtm_summary.positions
        ]

    def _build_dividend_sheet(self, year: int) -> list[dict]:
        """Byg udbytte-ark."""
        dividends = self._dividends.get_dividends(year=year)
        return [
            {
                "symbol": d.symbol,
                "dato": d.pay_date,
                "brutto": round(d.gross_amount, 2),
                "valuta": d.currency,
                "fx_kurs": round(d.fx_rate, 4),
                "brutto_dkk": round(d.gross_dkk, 2),
                "kildeskat_pct": round(d.withholding_pct * 100, 1),
                "kildeskat_dkk": round(d.withholding_dkk, 2),
                "netto_dkk": round(d.net_dkk, 2),
                "land": d.country,
                "dbo_sats": round(d.dbo_rate * 100, 1),
                "kredit_dkk": round(d.creditable_dkk, 2),
                "refunderbar_dkk": round(d.reclaimable_dkk, 2),
                "broker": d.broker,
            }
            for d in dividends
        ]

    def _build_fx_sheet(self, year: int) -> list[dict]:
        """Byg valutakursgevinst-ark."""
        summary = self._fx.get_annual_summary(year)
        rows = []

        # Per currency summary
        for ccy, data in summary.by_currency.items():
            rows.append({
                "valuta": ccy,
                "gevinster_dkk": round(data.get("gains_dkk", 0), 2),
                "tab_dkk": round(data.get("losses_dkk", 0), 2),
                "netto_dkk": round(data.get("net_dkk", 0), 2),
                "antal_transaktioner": data.get("transactions", 0),
            })

        return rows

    def _build_credit_sheet(self) -> list[dict]:
        """Byg skattetilgodehavende-ark."""
        history = self._credit.get_history(limit=100)
        return [
            {
                "dato": m.date[:10],
                "type": m.movement_type,
                "beløb_dkk": round(m.amount_dkk, 2),
                "saldo_dkk": round(m.balance_after, 2),
                "beskrivelse": m.description,
                "år": m.year,
            }
            for m in history
        ]

    # ── CSV Export ──────────────────────────────────────────

    def export_csv(
        self,
        year: int,
        output_dir: str = "reports",
        tax_result: Any = None,
        mtm_summary: Any = None,
    ) -> list[str]:
        """
        Eksportér skatterapport som CSV-filer.

        Returns:
            Liste af genererede filstier.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report = self.generate_annual_report(year, tax_result, mtm_summary)
        files: list[str] = []

        # Overblik
        overview_path = out / f"skat_overblik_{year}.csv"
        self._write_dict_csv(overview_path, report.overview)
        files.append(str(overview_path))

        # Transaktioner
        if report.transactions:
            tx_path = out / f"transaktioner_{year}.csv"
            self._write_list_csv(tx_path, report.transactions)
            files.append(str(tx_path))

        # Lagerbeskatning
        if report.mark_to_market:
            mtm_path = out / f"lagerbeskatning_{year}.csv"
            self._write_list_csv(mtm_path, report.mark_to_market)
            files.append(str(mtm_path))

        # Udbytter
        if report.dividends:
            div_path = out / f"udbytter_{year}.csv"
            self._write_list_csv(div_path, report.dividends)
            files.append(str(div_path))

        # Valutakursgevinster
        if report.fx_gains:
            fx_path = out / f"valutakursgevinster_{year}.csv"
            self._write_list_csv(fx_path, report.fx_gains)
            files.append(str(fx_path))

        # Skattetilgodehavende
        if report.tax_credit_movements:
            credit_path = out / f"skattetilgodehavende_{year}.csv"
            self._write_list_csv(credit_path, report.tax_credit_movements)
            files.append(str(credit_path))

        logger.info(
            f"[tax-report] CSV-export {year}: {len(files)} filer til {output_dir}"
        )
        return files

    @staticmethod
    def _write_dict_csv(path: Path, data: dict) -> None:
        """Skriv en dict som key-value CSV."""
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Felt", "Værdi"])
            for key, value in data.items():
                writer.writerow([key, value])

    @staticmethod
    def _write_list_csv(path: Path, data: list[dict]) -> None:
        """Skriv en liste af dicts som CSV."""
        if not data:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=data[0].keys(), delimiter=";"
            )
            writer.writeheader()
            writer.writerows(data)

    # ── Quarterly Report ────────────────────────────────────

    def generate_quarterly_summary(
        self,
        year: int,
        quarter: int,
        positions: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Kvartalsvis summary.

        Args:
            year: Regnskabsår.
            quarter: 1-4.
            positions: Nuværende positioner (til YTD estimat).
        """
        summary: dict[str, Any] = {
            "year": year,
            "quarter": quarter,
            "period": f"Q{quarter} {year}",
            "generated": datetime.now().isoformat(),
        }

        # YTD estimat
        if self._tax and positions:
            ytd = self._tax.ytd_estimated_tax(positions)
            summary["ytd_estimated"] = ytd

        # Udbytter YTD
        if self._dividends:
            div_summary = self._dividends.get_annual_summary(year)
            summary["dividends_ytd"] = {
                "gross_dkk": round(div_summary.total_gross_dkk, 2),
                "net_dkk": round(div_summary.total_net_dkk, 2),
                "count": div_summary.dividend_count,
            }

        # FX YTD
        if self._fx:
            fx_summary = self._fx.get_annual_summary(year)
            summary["fx_pnl_ytd"] = round(fx_summary.net_fx_pnl_dkk, 2)

        # Tilgodehavende
        if self._credit:
            summary["tax_credit_balance"] = round(self._credit.balance, 2)

        summary["disclaimer"] = self.DISCLAIMER
        return summary

    # ── Dashboard Data ──────────────────────────────────────

    def get_dashboard_data(
        self,
        year: int | None = None,
        positions: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Data til skat-dashboard widgets."""
        current_year = year or datetime.now().year

        dashboard: dict[str, Any] = {
            "year": current_year,
        }

        # Tilgodehavende gauge
        if self._credit:
            dashboard["tax_credit"] = self._credit.dashboard_data()

        # YTD skat estimat
        if self._tax and positions:
            dashboard["ytd_tax"] = self._tax.ytd_estimated_tax(positions)

        # Seneste udbytter
        if self._dividends:
            recent = self._dividends.get_dividends(limit=5)
            dashboard["recent_dividends"] = [
                {
                    "symbol": d.symbol,
                    "date": d.pay_date[:10],
                    "net_dkk": round(d.net_dkk, 2),
                    "country": d.country,
                }
                for d in recent
            ]

        # Åben FX-eksponering
        if self._fx:
            dashboard["fx_exposure"] = self._fx.get_open_fx_positions()

        dashboard["disclaimer"] = self.DISCLAIMER
        return dashboard
