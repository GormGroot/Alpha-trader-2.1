"""
Skatteindberetningsmodul – dansk skat på aktiehandel.

Modulet beregner realiseret gevinst/tab, valutaomregning og
estimeret dansk skat (aktieindkomst). Inkluderer:
  - CurrencyConverter: USD/DKK dagskurser fra ECB
  - TransactionLog: Fuld dokumentation af alle handler
  - DanishTaxCalculator: Skatteberegning efter FIFO-princippet
  - TaxReportGenerator: Årsrapporter til SKAT

⚠️ VIGTIGT: Alle skatteberegninger er vejledende.
   Verificér altid med en revisor inden indberetning.
"""

from src.tax.currency import CurrencyConverter
from src.tax.transaction_log import TransactionLog, TransactionRecord
from src.tax.tax_calculator import DanishTaxCalculator, TaxResult, TaxLot
from src.tax.tax_report import TaxReportGenerator, AnnualReport
from src.tax.tax_advisor import (
    TaxAdvisor,
    QuarterlyEstimate,
    TaxLossCandidate,
    WashSaleWarning,
    YearEndReport,
    TaxAlert,
)

__all__ = [
    "CurrencyConverter",
    "TransactionLog",
    "TransactionRecord",
    "DanishTaxCalculator",
    "TaxResult",
    "TaxLot",
    "TaxReportGenerator",
    "AnnualReport",
    "TaxAdvisor",
    "QuarterlyEstimate",
    "TaxLossCandidate",
    "WashSaleWarning",
    "YearEndReport",
    "TaxAlert",
]
