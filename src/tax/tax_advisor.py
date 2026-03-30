"""
Proaktiv skatterådgiver – advarsler, projektion og optimering.

Modulet giver:
  - Kvartalsvise estimater med projektion for resten af året
  - Tax-loss harvesting muligheder
  - Wash-sale advarsel (30-dages regel for US-aktier)
  - Årsafslutningsrapport (december)
  - Progressionsgrænse-monitorering

Alle beregninger logges med detaljeret forklaring
(audit trail) så en revisor kan følge dem.

⚠️ VIGTIGT: Alle anbefalinger er vejledende.
   Konsultér altid en revisor inden du handler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger

from src.risk.portfolio_tracker import Position
from src.tax.tax_calculator import DanishTaxCalculator, TaxResult


# ── Dataklasser ───────────────────────────────────────────────


@dataclass
class QuarterlyEstimate:
    """Kvartalsoversigt med projektion."""

    year: int
    quarter: int                         # 1–4
    as_of_date: str                      # Beregningsdato

    # Hidtil i år (year-to-date)
    realized_gain_ytd_dkk: float = 0.0
    realized_loss_ytd_dkk: float = 0.0
    net_ytd_dkk: float = 0.0
    num_trades_ytd: int = 0

    # Projektion (lineær fremskrivning)
    projected_annual_gain_dkk: float = 0.0
    projected_annual_tax_dkk: float = 0.0
    projected_effective_rate: float = 0.0

    # Progressionsgrænse
    progression_limit: float = 61_000
    pct_of_limit_used: float = 0.0
    remaining_before_42pct: float = 0.0
    projected_hits_limit: bool = False
    projected_limit_date: str = ""       # Estimeret dato for at ramme grænsen

    # Aktuel skat YTD
    tax_ytd_dkk: float = 0.0

    # Audit trail
    audit_notes: list[str] = field(default_factory=list)


@dataclass
class TaxLossCandidate:
    """Aktie med urealiseret tab – kandidat til tax-loss harvesting."""

    symbol: str
    qty: float
    entry_price: float
    current_price: float
    unrealized_pnl_usd: float
    unrealized_pnl_dkk: float
    potential_tax_saving_dkk: float       # Besparelse ved salg nu
    holding_days: int
    recommendation: str                   # Kort anbefaling


@dataclass
class WashSaleWarning:
    """Advarsel om potentiel wash sale (genkøb inden 30 dage)."""

    symbol: str
    sell_date: str
    sell_price: float
    days_since_sell: int
    loss_amount_dkk: float
    warning: str


@dataclass
class YearEndReport:
    """Skatteforberedelses-rapport til årsafslutning (december)."""

    year: int
    generated_at: str

    # Status
    tax_result: TaxResult | None = None
    quarterly_estimate: QuarterlyEstimate | None = None

    # Handlingsanbefalinger
    actions: list[str] = field(default_factory=list)

    # Tax-loss harvesting
    harvest_candidates: list[TaxLossCandidate] = field(default_factory=list)
    potential_saving_dkk: float = 0.0

    # Udbytte
    dividend_reminder: str = ""

    # Deadlines
    deadlines: list[str] = field(default_factory=list)

    # Audit trail
    audit_notes: list[str] = field(default_factory=list)

    @property
    def summary_lines(self) -> list[str]:
        """Returnér rapport som liste af linjer."""
        lines = [
            "═" * 65,
            "  SKATTEFORBEREDELSE – ÅRSAFSLUTNING",
            "  ⚠️ Vejledende beregning – verificér med revisor",
            "═" * 65,
            f"  Skatteår:     {self.year}",
            f"  Genereret:    {self.generated_at}",
            "",
        ]

        if self.tax_result:
            r = self.tax_result
            lines.extend([
                "─" * 65,
                "  AKTUEL SKATTESTATUS",
                "─" * 65,
                f"  Realiserede gevinster:    {r.total_gains_dkk:>+14,.2f} DKK",
                f"  Realiserede tab:          {r.total_losses_dkk:>+14,.2f} DKK",
                f"  Netto gevinst/tab:        {r.net_gain_dkk:>+14,.2f} DKK",
                f"  Estimeret skat:           {r.total_tax_dkk:>14,.2f} DKK",
                "",
            ])

        if self.quarterly_estimate:
            q = self.quarterly_estimate
            lines.extend([
                "─" * 65,
                "  PROJEKTION FOR RESTEN AF ÅRET",
                "─" * 65,
                f"  Forventet årsgevinst:     {q.projected_annual_gain_dkk:>+14,.2f} DKK",
                f"  Forventet årsskat:        {q.projected_annual_tax_dkk:>14,.2f} DKK",
                f"  Forventet effektiv sats:  {q.projected_effective_rate:>13.1f}%",
                f"  Progressionsgrænse brugt: {q.pct_of_limit_used:>13.1f}%",
                "",
            ])
            if q.projected_hits_limit:
                lines.append(
                    f"  ⚠️ ADVARSEL: Du rammer sandsynligvis progressionsgrænsen!"
                )
                if q.projected_limit_date:
                    lines.append(
                        f"     Estimeret dato: {q.projected_limit_date}"
                    )
                lines.append("")

        if self.actions:
            lines.extend([
                "─" * 65,
                "  ANBEFALEDE HANDLINGER FØR 31. DECEMBER",
                "─" * 65,
            ])
            for i, action in enumerate(self.actions, 1):
                lines.append(f"  {i}. {action}")
            lines.append("")

        if self.harvest_candidates:
            lines.extend([
                "─" * 65,
                "  TAX-LOSS HARVESTING MULIGHEDER",
                "─" * 65,
                f"  {'Aktie':<8} {'Urealiseret tab':>16} {'Potentiel besparelse':>20}",
            ])
            for c in self.harvest_candidates:
                lines.append(
                    f"  {c.symbol:<8} {c.unrealized_pnl_dkk:>+16,.2f} DKK"
                    f" {c.potential_tax_saving_dkk:>+16,.2f} DKK"
                )
            lines.append(
                f"\n  Samlet potentiel besparelse: "
                f"{self.potential_saving_dkk:,.2f} DKK"
            )
            lines.append("")

        if self.dividend_reminder:
            lines.extend([
                "─" * 65,
                "  UDBYTTE",
                "─" * 65,
                f"  {self.dividend_reminder}",
                "",
            ])

        if self.deadlines:
            lines.extend([
                "─" * 65,
                "  VIGTIGE DEADLINES",
                "─" * 65,
            ])
            for dl in self.deadlines:
                lines.append(f"  📅 {dl}")
            lines.append("")

        lines.extend([
            "═" * 65,
            "  ⚠️ Alle anbefalinger er vejledende.",
            "  Konsultér altid en revisor inden indberetning.",
            "═" * 65,
        ])

        return lines


@dataclass
class TaxAlert:
    """
    En skatte-alert til notifikationssystemet.

    Severity:
      - INFO: Status-besked (månedlig opdatering)
      - WARNING: Noget kræver opmærksomhed (tæt på grænse)
      - CRITICAL: Handling påkrævet (wash sale, deadline)
    """

    severity: str           # "INFO", "WARNING", "CRITICAL"
    title: str
    message: str
    category: str           # "progression", "wash_sale", "year_end", "deadline", "monthly"
    timestamp: str = ""
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ── Rådgiver ─────────────────────────────────────────────────


class TaxAdvisor:
    """
    Proaktiv skatterådgiver med advarsler, projektioner og optimering.

    Alle beregninger logges med forklaring så en revisor kan følge dem.
    """

    def __init__(
        self,
        calculator: DanishTaxCalculator | None = None,
        progression_limit: float = 61_000,
        carried_losses: float = 0.0,
        fx_rate: float = 6.90,
    ) -> None:
        self._calculator = calculator or DanishTaxCalculator(
            progression_limit=progression_limit,
            carried_losses=carried_losses,
        )
        self._progression_limit = progression_limit
        self._carried_losses = carried_losses
        self._fx_rate = fx_rate

        # Historik over solgte aktier (til wash sale detektion)
        self._recent_sells: list[dict] = []

    # ══════════════════════════════════════════════════════════
    #  1. Kvartalsvise estimater
    # ══════════════════════════════════════════════════════════

    def quarterly_estimate(
        self,
        transactions: list[dict],
        year: int = 2026,
        as_of_date: str | None = None,
    ) -> QuarterlyEstimate:
        """
        Beregn kvartalsestimat med lineær projektion for resten af året.

        Args:
            transactions: Handler hidtil i år (dicts med realized_pnl_dkk osv.)
            year: Skatteåret.
            as_of_date: Beregningsdato (default: i dag).

        Returns:
            QuarterlyEstimate med projektion og audit trail.
        """
        now = datetime.now()
        as_of = as_of_date or now.strftime("%Y-%m-%d")
        quarter = (now.month - 1) // 3 + 1

        audit: list[str] = []
        audit.append(f"Kvartalsoversigt beregnet {as_of} for skatteår {year}")

        # ── YTD tal ──
        gains = sum(tx["realized_pnl_dkk"] for tx in transactions
                     if tx["realized_pnl_dkk"] > 0)
        losses = sum(tx["realized_pnl_dkk"] for tx in transactions
                      if tx["realized_pnl_dkk"] < 0)
        net_ytd = gains + losses
        num_trades = len(transactions)

        audit.append(
            f"YTD: {num_trades} handler, gevinster={gains:+,.2f} DKK, "
            f"tab={losses:+,.2f} DKK, netto={net_ytd:+,.2f} DKK"
        )

        # ── YTD skat ──
        ytd_tax_result = self._calculator.calculate(transactions, year=year)
        tax_ytd = ytd_tax_result.total_tax_dkk

        # ── Lineær projektion ──
        day_of_year = now.timetuple().tm_yday
        days_in_year = 366 if _is_leap_year(year) else 365
        fraction_passed = max(day_of_year / days_in_year, 0.01)  # Undgå /0

        projected_gain = net_ytd / fraction_passed
        projected_tax = self._calculator.estimate_tax(
            max(projected_gain - self._carried_losses, 0)
        )

        effective_rate = 0.0
        if projected_gain > 0:
            effective_rate = projected_tax / projected_gain * 100

        audit.append(
            f"Projektion: {fraction_passed:.1%} af året forløbet → "
            f"forventet årsgevinst={projected_gain:+,.2f} DKK, "
            f"skat={projected_tax:,.2f} DKK ({effective_rate:.1f}%)"
        )

        # ── Progressionsgrænse ──
        taxable_ytd = max(net_ytd - self._carried_losses, 0)
        pct_used = (taxable_ytd / self._progression_limit * 100
                    if self._progression_limit > 0 else 0)
        remaining = max(self._progression_limit - taxable_ytd, 0)

        hits_limit = projected_gain - self._carried_losses > self._progression_limit
        limit_date = ""

        if net_ytd > 0 and self._progression_limit > 0:
            daily_rate = net_ytd / max(day_of_year, 1)
            if daily_rate > 0:
                days_to_limit = (
                    (self._progression_limit + self._carried_losses - net_ytd)
                    / daily_rate
                )
                if 0 < days_to_limit < (days_in_year - day_of_year):
                    limit_date_dt = now + timedelta(days=int(days_to_limit))
                    limit_date = limit_date_dt.strftime("%Y-%m-%d")
                    audit.append(
                        f"Progressionsgrænse: estimeret rammes {limit_date} "
                        f"(ved {daily_rate:,.0f} DKK/dag)"
                    )

        if hits_limit:
            audit.append(
                "⚠️ Projektion viser overskridelse af progressionsgrænsen "
                f"({self._progression_limit:,.0f} DKK). Satsen stiger fra 27% til 42%."
            )

        logger.info(
            f"[skat-advisor] Q{quarter} {year}: "
            f"YTD={net_ytd:+,.0f} DKK, proj={projected_gain:+,.0f} DKK, "
            f"skat={projected_tax:,.0f} DKK, grænse={pct_used:.0f}% brugt"
        )

        return QuarterlyEstimate(
            year=year,
            quarter=quarter,
            as_of_date=as_of,
            realized_gain_ytd_dkk=gains,
            realized_loss_ytd_dkk=losses,
            net_ytd_dkk=net_ytd,
            num_trades_ytd=num_trades,
            projected_annual_gain_dkk=projected_gain,
            projected_annual_tax_dkk=projected_tax,
            projected_effective_rate=effective_rate,
            progression_limit=self._progression_limit,
            pct_of_limit_used=pct_used,
            remaining_before_42pct=remaining,
            projected_hits_limit=hits_limit,
            projected_limit_date=limit_date,
            tax_ytd_dkk=tax_ytd,
            audit_notes=audit,
        )

    # ══════════════════════════════════════════════════════════
    #  2. Skatteoptimering (vejledende)
    # ══════════════════════════════════════════════════════════

    def check_progression_warning(
        self,
        current_gain_dkk: float,
        planned_sale_gain_dkk: float = 0.0,
    ) -> TaxAlert | None:
        """
        Advar hvis du er tæt på eller overskrider progressionsgrænsen.

        Args:
            current_gain_dkk: Realiseret gevinst hidtil i år.
            planned_sale_gain_dkk: Forventet gevinst ved næste salg.

        Returns:
            TaxAlert hvis relevant, ellers None.
        """
        taxable = max(current_gain_dkk - self._carried_losses, 0)
        remaining = self._progression_limit - taxable

        # Allerede over grænsen
        if remaining <= 0:
            over = abs(remaining)
            extra_tax = over * (self._calculator.HIGH_RATE - self._calculator.LOW_RATE)
            alert = TaxAlert(
                severity="WARNING",
                title="Progressionsgrænse overskredet",
                message=(
                    f"Du har overskredet progressionsgrænsen med "
                    f"{over:,.0f} DKK. Yderligere gevinster beskattes med 42% "
                    f"i stedet for 27%. Ekstra skat: {extra_tax:,.0f} DKK."
                ),
                category="progression",
                data={"taxable": taxable, "over_limit": over, "extra_tax": extra_tax},
            )
            logger.warning(f"[skat-advisor] {alert.title}: {over:,.0f} DKK over grænsen")
            return alert

        # Tæt på grænsen (< 20% tilbage)
        pct_remaining = remaining / self._progression_limit * 100
        if pct_remaining < 20:
            after_sale = taxable + planned_sale_gain_dkk
            will_exceed = after_sale > self._progression_limit

            msg = (
                f"Du er tæt på progressionsgrænsen! "
                f"Kun {remaining:,.0f} DKK ({pct_remaining:.0f}%) tilbage "
                f"inden satsen stiger fra 27% til 42%."
            )
            if will_exceed and planned_sale_gain_dkk > 0:
                over = after_sale - self._progression_limit
                extra = over * (self._calculator.HIGH_RATE - self._calculator.LOW_RATE)
                msg += (
                    f"\n\nHvis du realiserer yderligere {planned_sale_gain_dkk:,.0f} DKK, "
                    f"vil {over:,.0f} DKK beskattes med 42% "
                    f"(ekstra skat: {extra:,.0f} DKK)."
                )

            alert = TaxAlert(
                severity="WARNING",
                title="Tæt på progressionsgrænsen",
                message=msg,
                category="progression",
                data={
                    "remaining": remaining,
                    "pct_remaining": pct_remaining,
                    "will_exceed_with_sale": will_exceed,
                },
            )
            logger.info(
                f"[skat-advisor] Progressionsgrænse: "
                f"{pct_remaining:.0f}% tilbage ({remaining:,.0f} DKK)"
            )
            return alert

        return None

    def find_tax_loss_candidates(
        self,
        positions: list[Position],
        current_gain_dkk: float = 0.0,
    ) -> list[TaxLossCandidate]:
        """
        Find aktier med urealiseret tab til tax-loss harvesting.

        Args:
            positions: Åbne positioner fra PortfolioTracker.
            current_gain_dkk: Realiserede gevinster hidtil i år.

        Returns:
            Liste af TaxLossCandidate sorteret efter størst besparelse.
        """
        candidates = []

        for pos in positions:
            if pos.unrealized_pnl >= 0:
                continue  # Kun tab er relevant

            loss_usd = pos.unrealized_pnl
            loss_dkk = loss_usd * self._fx_rate

            # Beregn potentiel besparelse
            taxable_now = max(current_gain_dkk - self._carried_losses, 0)
            taxable_after = max(
                current_gain_dkk + loss_dkk - self._carried_losses, 0
            )
            tax_now = self._calculator.estimate_tax(taxable_now)
            tax_after = self._calculator.estimate_tax(taxable_after)
            saving = tax_now - tax_after

            # Holdingperiode
            try:
                entry_dt = datetime.fromisoformat(pos.entry_time)
                holding_days = (datetime.now() - entry_dt).days
            except (ValueError, TypeError):
                holding_days = 0

            # Anbefaling
            if saving > 1_000:
                rec = (
                    f"Overvej salg – potentiel skattebesparelse: "
                    f"{saving:,.0f} DKK. ⚠️ Vejledende."
                )
            elif saving > 0:
                rec = (
                    f"Mulig besparelse: {saving:,.0f} DKK. "
                    f"Overvej om det er det værd."
                )
            else:
                rec = "Ingen skattebesparelse på nuværende tidspunkt."

            candidates.append(TaxLossCandidate(
                symbol=pos.symbol,
                qty=pos.qty,
                entry_price=pos.entry_price,
                current_price=pos.current_price,
                unrealized_pnl_usd=loss_usd,
                unrealized_pnl_dkk=loss_dkk,
                potential_tax_saving_dkk=saving,
                holding_days=holding_days,
                recommendation=rec,
            ))

            logger.debug(
                f"[skat-advisor] Tax-loss kandidat: {pos.symbol} "
                f"tab={loss_dkk:+,.0f} DKK, besparelse={saving:,.0f} DKK"
            )

        # Sortér efter størst besparelse
        candidates.sort(key=lambda c: c.potential_tax_saving_dkk, reverse=True)
        return candidates

    def check_wash_sale(
        self,
        symbol: str,
        buy_date: str,
        recent_sells: list[dict] | None = None,
    ) -> WashSaleWarning | None:
        """
        Tjek om genkøb af en aktie falder inden for 30-dages wash sale perioden.

        US wash sale-reglen: Hvis du sælger en aktie med tab og genkøber
        den samme (eller "substantially identical") aktie inden for 30 dage
        før eller efter salget, kan tabet ikke fratrækkes.

        BEMÆRK: Wash sale-reglen er primært en US-regel, men kan være
        relevant for aktier handlet på US-børser.

        Args:
            symbol: Aktiesymbolet der købes.
            buy_date: Købsdato (YYYY-MM-DD).
            recent_sells: Liste af seneste salg ({symbol, date, price, pnl_dkk}).

        Returns:
            WashSaleWarning hvis relevant, ellers None.
        """
        sells = recent_sells or self._recent_sells

        try:
            buy_dt = datetime.strptime(buy_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None

        for sell in sells:
            if sell.get("symbol") != symbol:
                continue
            if sell.get("pnl_dkk", 0) >= 0:
                continue  # Kun tab udløser wash sale

            try:
                sell_dt = datetime.strptime(sell["date"], "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            days_diff = abs((buy_dt - sell_dt).days)
            if days_diff <= 30:
                warning = WashSaleWarning(
                    symbol=symbol,
                    sell_date=sell["date"],
                    sell_price=sell.get("price", 0),
                    days_since_sell=days_diff,
                    loss_amount_dkk=abs(sell.get("pnl_dkk", 0)),
                    warning=(
                        f"⚠️ WASH SALE ADVARSEL: Du genkøber {symbol} kun "
                        f"{days_diff} dage efter et salg med tab "
                        f"({abs(sell.get('pnl_dkk', 0)):,.0f} DKK). "
                        f"Tabet kan muligvis IKKE fratrækkes efter US wash "
                        f"sale-reglen (30 dage). Konsultér en revisor."
                    ),
                )
                logger.warning(
                    f"[skat-advisor] Wash sale! {symbol} genkøbt "
                    f"{days_diff}d efter tabssalg"
                )
                return warning

        return None

    def register_sell(
        self,
        symbol: str,
        date: str,
        price: float,
        pnl_dkk: float,
    ) -> None:
        """Registrer et salg til wash sale-detektion."""
        self._recent_sells.append({
            "symbol": symbol,
            "date": date,
            "price": price,
            "pnl_dkk": pnl_dkk,
        })
        # Behold kun salg fra de seneste 60 dage
        cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        self._recent_sells = [
            s for s in self._recent_sells if s["date"] >= cutoff
        ]

    # ══════════════════════════════════════════════════════════
    #  3. Årsafslutning
    # ══════════════════════════════════════════════════════════

    def year_end_report(
        self,
        transactions: list[dict],
        positions: list[Position] | None = None,
        dividends: list[dict] | None = None,
        year: int = 2026,
    ) -> YearEndReport:
        """
        Generer skatteforberedelses-rapport til årsafslutning.

        Bør køres i december for at give tid til at handle inden 31/12.

        Args:
            transactions: Alle handler i året (dicts).
            positions: Åbne positioner (til tax-loss harvesting).
            dividends: Udbyttedata.
            year: Skatteåret.

        Returns:
            YearEndReport med anbefalinger og audit trail.
        """
        now = datetime.now()
        audit: list[str] = []
        audit.append(
            f"Årsafslutningsrapport genereret {now.isoformat()} for {year}"
        )

        # ── Beregn aktuel status ──
        tax_result = self._calculator.calculate(
            transactions, dividends=dividends, year=year,
        )
        audit.append(
            f"Skatteberegning: netto={tax_result.net_gain_dkk:+,.2f} DKK, "
            f"skat={tax_result.total_tax_dkk:,.2f} DKK"
        )

        # ── Kvartalsestimat ──
        q_est = self.quarterly_estimate(transactions, year=year)

        # ── Handlingsanbefalinger ──
        actions: list[str] = []

        # A) Progressionsgrænse
        taxable = tax_result.taxable_gain_dkk
        remaining = max(self._progression_limit - taxable, 0)
        if taxable > self._progression_limit:
            over = taxable - self._progression_limit
            actions.append(
                f"Du har overskredet progressionsgrænsen med {over:,.0f} DKK. "
                f"Overvej at udskyde yderligere salg til næste år."
            )
        elif remaining < 10_000 and taxable > 0:
            actions.append(
                f"Du er tæt på progressionsgrænsen ({remaining:,.0f} DKK "
                f"tilbage). Overvej om du vil realisere flere gevinster i år."
            )

        # B) Tab modregning
        if tax_result.total_losses_dkk < 0 and tax_result.total_gains_dkk > 0:
            net_after = tax_result.total_gains_dkk + tax_result.total_losses_dkk
            saved = (
                self._calculator.estimate_tax(tax_result.total_gains_dkk)
                - self._calculator.estimate_tax(max(net_after, 0))
            )
            actions.append(
                f"Dine tab ({tax_result.total_losses_dkk:+,.0f} DKK) reducerer "
                f"skatten med ca. {saved:,.0f} DKK."
            )
            audit.append(f"Tab modregnet: besparelse={saved:,.2f} DKK")

        # C) Uudnyttede tab til fremførsel
        if tax_result.remaining_loss_dkk > 0:
            actions.append(
                f"Tab til fremførsel: {tax_result.remaining_loss_dkk:,.0f} DKK "
                f"kan modregnes i fremtidige gevinster."
            )

        # D) Tax-loss harvesting
        harvest_candidates = []
        total_saving = 0.0
        if positions:
            harvest_candidates = self.find_tax_loss_candidates(
                positions, current_gain_dkk=tax_result.net_gain_dkk,
            )
            total_saving = sum(
                c.potential_tax_saving_dkk for c in harvest_candidates
                if c.potential_tax_saving_dkk > 0
            )
            if total_saving > 0:
                actions.append(
                    f"Tax-loss harvesting: Du kan potentielt spare "
                    f"{total_saving:,.0f} DKK i skat ved at sælge aktier "
                    f"med urealiseret tab. Se detaljeret oversigt nedenfor."
                )
                audit.append(
                    f"Tax-loss harvesting: {len(harvest_candidates)} kandidater, "
                    f"potentiel besparelse={total_saving:,.2f} DKK"
                )

        # E) Udbytte-påmindelse
        dividend_reminder = ""
        if tax_result.dividend_us_tax_dkk > 0:
            dividend_reminder = (
                f"Husk: Du har betalt {tax_result.dividend_us_tax_dkk:,.0f} DKK "
                f"i US kildeskat på udbytte. Op til "
                f"{tax_result.dividend_dk_credit:,.0f} DKK kan fratrækkes "
                f"via creditlempelse (rubrik 68)."
            )
            actions.append(dividend_reminder)
        elif tax_result.dividend_gross_dkk == 0:
            dividend_reminder = (
                "Tjek om du har modtaget udbytte fra US-aktier i løbet af året. "
                "Kildeskat tilbageholdt i USA kan fratrækkes i dansk skat."
            )

        # ── Deadlines ──
        deadlines = [
            f"31. december {year}: Sidste dag for handler i skatteår {year}",
            f"1. marts {year + 1}: Årsopgørelsen åbner hos SKAT",
            f"1. maj {year + 1}: Frist for rettelser til årsopgørelsen",
            f"1. juli {year + 1}: Frist for indbetaling af restskat",
        ]

        # ── Altid tilføj disclaimer ──
        actions.append(
            "⚠️ Alle anbefalinger er vejledende. Konsultér altid en revisor."
        )

        logger.info(
            f"[skat-advisor] Årsrapport {year}: "
            f"{len(actions)} anbefalinger, "
            f"{len(harvest_candidates)} harvest-kandidater"
        )

        return YearEndReport(
            year=year,
            generated_at=now.isoformat(),
            tax_result=tax_result,
            quarterly_estimate=q_est,
            actions=actions,
            harvest_candidates=harvest_candidates,
            potential_saving_dkk=total_saving,
            dividend_reminder=dividend_reminder,
            deadlines=deadlines,
            audit_notes=audit,
        )

    # ══════════════════════════════════════════════════════════
    #  4. Alerts til notifikationssystemet
    # ══════════════════════════════════════════════════════════

    def generate_monthly_status(
        self,
        transactions: list[dict],
        year: int = 2026,
    ) -> TaxAlert:
        """Generer månedlig statusbesked til email/notifikation."""
        est = self.quarterly_estimate(transactions, year=year)

        msg_lines = [
            f"Månedlig skattestatus for {year}",
            f"",
            f"Realiseret YTD: {est.net_ytd_dkk:+,.0f} DKK ({est.num_trades_ytd} handler)",
            f"Estimeret skat YTD: {est.tax_ytd_dkk:,.0f} DKK",
            f"",
            f"Projektion for hele året:",
            f"  Forventet gevinst: {est.projected_annual_gain_dkk:+,.0f} DKK",
            f"  Forventet skat: {est.projected_annual_tax_dkk:,.0f} DKK",
            f"  Effektiv sats: {est.projected_effective_rate:.1f}%",
            f"",
            f"Progressionsgrænse: {est.pct_of_limit_used:.0f}% brugt "
            f"({est.remaining_before_42pct:,.0f} DKK tilbage)",
        ]

        if est.projected_hits_limit:
            msg_lines.append(
                f"\n⚠️ Du rammer sandsynligvis progressionsgrænsen i år!"
            )

        msg_lines.append(
            f"\n⚠️ Vejledende beregning – verificér med revisor."
        )

        return TaxAlert(
            severity="INFO",
            title=f"Skattestatus – {datetime.now().strftime('%B %Y')}",
            message="\n".join(msg_lines),
            category="monthly",
            data={
                "net_ytd": est.net_ytd_dkk,
                "tax_ytd": est.tax_ytd_dkk,
                "projected_gain": est.projected_annual_gain_dkk,
                "projected_tax": est.projected_annual_tax_dkk,
                "pct_limit_used": est.pct_of_limit_used,
            },
        )

    def generate_progression_alert(
        self,
        current_gain_dkk: float,
    ) -> TaxAlert | None:
        """Generér alert når progressionsgrænsen nærmer sig."""
        return self.check_progression_warning(current_gain_dkk)

    def generate_march_reminder(self, year: int) -> TaxAlert:
        """Påmindelse om indberetning til SKAT i marts."""
        return TaxAlert(
            severity="CRITICAL",
            title=f"📋 Husk skatteindberetning for {year - 1}",
            message=(
                f"Årsopgørelsen for {year - 1} er åben hos SKAT.\n\n"
                f"Tjek at følgende er korrekt:\n"
                f"• Rubrik 66: Gevinst på aktieindkomst\n"
                f"• Rubrik 67: Tab på aktieindkomst\n"
                f"• Rubrik 68: Udenlandsk udbytte\n\n"
                f"Frist for rettelser: 1. maj {year}\n"
                f"Frist for restskat: 1. juli {year}\n\n"
                f"⚠️ Verificér altid med en revisor."
            ),
            category="deadline",
        )

    def collect_pending_alerts(
        self,
        transactions: list[dict],
        year: int = 2026,
    ) -> list[TaxAlert]:
        """
        Saml alle relevante alerts baseret på aktuel status.

        Returns:
            Liste af TaxAlert sorteret efter severity.
        """
        alerts: list[TaxAlert] = []
        now = datetime.now()

        # Beregn aktuel status
        if transactions:
            net_gain = sum(tx["realized_pnl_dkk"] for tx in transactions)
        else:
            net_gain = 0.0

        # 1. Progressionsgrænse
        prog_alert = self.check_progression_warning(net_gain)
        if prog_alert:
            alerts.append(prog_alert)

        # 2. Marts-påmindelse
        if now.month == 3:
            alerts.append(self.generate_march_reminder(now.year))

        # 3. December – årsafslutning
        if now.month == 12:
            alerts.append(TaxAlert(
                severity="WARNING",
                title=f"Skatteforberedelse – {year}",
                message=(
                    f"December er sidste chance for at optimere din "
                    f"skattesituation for {year}.\n\n"
                    f"Overvej:\n"
                    f"• Tax-loss harvesting (sælg tabsgivende aktier)\n"
                    f"• Udskyd salg af vinderaktier til januar\n"
                    f"• Tjek udbyttedokumentation\n\n"
                    f"Kør 'python -m src.main tax-advisor --year {year}' "
                    f"for fuld analyse."
                ),
                category="year_end",
            ))

        # Sortér: CRITICAL først, derefter WARNING, derefter INFO
        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

        return alerts


# ── Hjælpefunktioner ──────────────────────────────────────────


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
