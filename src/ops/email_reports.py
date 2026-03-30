"""
Email Reports — morgen-, aften- og ugerapporter + alarm-emails.

Rapporttyper:
  - Morgenrapport (kort): Portfolio value + overnight changes
  - Aftenrapport (detaljeret): Full P&L + signaler + skat
  - Ugentlig rapport: Performance summary + skat YTD
  - Alarm emails: Drawdown > 5%, broker disconnected, etc.

Konfiguration via environment:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, REPORT_EMAIL_TO
"""

from __future__ import annotations

import smtplib
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Any
from zoneinfo import ZoneInfo

from loguru import logger

TZ_CET = ZoneInfo("Europe/Copenhagen")


@dataclass
class SMTPConfig:
    """SMTP-konfiguration fra environment."""
    host: str = ""
    port: int = 587
    user: str = ""
    password: str = ""
    use_tls: bool = True
    from_email: str = ""
    to_emails: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> SMTPConfig:
        to_raw = os.getenv("REPORT_EMAIL_TO", "")
        to_list = [e.strip() for e in to_raw.split(",") if e.strip()]
        return cls(
            host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            port=int(os.getenv("SMTP_PORT", "587")),
            user=os.getenv("SMTP_USER", ""),
            password=os.getenv("SMTP_PASS", ""),
            use_tls=os.getenv("SMTP_TLS", "true").lower() == "true",
            from_email=os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "")),
            to_emails=to_list,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.user and self.password and self.to_emails)


@dataclass
class ReportData:
    """Data-container til rapport-generering."""
    portfolio_value_dkk: float = 0
    daily_pnl_dkk: float = 0
    daily_pnl_pct: float = 0
    mtd_pnl_dkk: float = 0
    ytd_pnl_dkk: float = 0
    broker_status: dict[str, str] = field(default_factory=dict)
    positions_count: int = 0
    signals: list[dict] = field(default_factory=list)
    tax_credit_balance: float = 0
    estimated_tax_ytd: float = 0
    tax_suggestions: list[str] = field(default_factory=list)
    top_winners: list[dict] = field(default_factory=list)
    top_losers: list[dict] = field(default_factory=list)
    dividends_ytd: float = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(TZ_CET))


# ── Email Sender ───────────────────────────────────────────

class EmailSender:
    """Lavniveau email-afsender via SMTP."""

    def __init__(self, config: SMTPConfig | None = None):
        self._config = config or SMTPConfig.from_env()

    def send(
        self,
        subject: str,
        html_body: str,
        attachments: list[tuple[str, bytes]] | None = None,
    ) -> bool:
        """Send email. Returns True on success."""
        if not self._config.is_configured:
            logger.warning("[email] SMTP not configured — skipping email")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self._config.from_email
            msg["To"] = ", ".join(self._config.to_emails)

            # HTML body
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            # Attachments
            if attachments:
                for filename, data in attachments:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(data)
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={filename}")
                    msg.attach(part)

            # Send
            with smtplib.SMTP(self._config.host, self._config.port) as server:
                if self._config.use_tls:
                    server.starttls()
                server.login(self._config.user, self._config.password)
                server.sendmail(
                    self._config.from_email,
                    self._config.to_emails,
                    msg.as_string(),
                )

            logger.info(f"[email] Sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"[email] Failed to send '{subject}': {e}")
            return False


# ── HTML Templates ─────────────────────────────────────────

_BASE_STYLE = """
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background-color: #0f1117; color: #e2e8f0; padding: 20px; }
    .container { max-width: 600px; margin: 0 auto; background: #1a1c24;
                 border-radius: 12px; padding: 24px; border: 1px solid #2d3748; }
    .header { color: #00d4aa; font-size: 24px; font-weight: 700; margin-bottom: 4px; }
    .subtitle { color: #64748b; font-size: 14px; margin-bottom: 20px; }
    .kpi-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
    .kpi { flex: 1; min-width: 120px; background: #0f1117; border-radius: 8px;
           padding: 12px; border: 1px solid #2d3748; }
    .kpi-label { color: #64748b; font-size: 12px; margin-bottom: 4px; }
    .kpi-value { color: #e2e8f0; font-size: 20px; font-weight: 700; }
    .positive { color: #2ed573; }
    .negative { color: #ff4757; }
    .section { margin-top: 20px; }
    .section-title { color: #00d4aa; font-size: 16px; font-weight: 600; margin-bottom: 8px; }
    table { width: 100%; border-collapse: collapse; }
    th { color: #64748b; font-size: 12px; text-align: left; padding: 8px; border-bottom: 1px solid #2d3748; }
    td { color: #e2e8f0; font-size: 13px; padding: 8px; border-bottom: 1px solid #2d3748; }
    .footer { color: #64748b; font-size: 11px; margin-top: 20px; text-align: center; }
    .alert { background: #ff4757; color: white; padding: 12px; border-radius: 8px; margin-bottom: 16px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
    .badge-green { background: #2ed573; color: #0f1117; }
    .badge-red { background: #ff4757; color: white; }
    .badge-orange { background: #ffa502; color: #0f1117; }
    .badge-gray { background: #64748b; color: white; }
</style>
"""


def _pnl_class(value: float) -> str:
    return "positive" if value >= 0 else "negative"


def _pnl_sign(value: float) -> str:
    return f"+{value:,.0f}" if value >= 0 else f"{value:,.0f}"


def _broker_badge(status: str) -> str:
    color = {"connected": "green", "degraded": "orange", "disconnected": "red"}.get(status, "gray")
    return f'<span class="badge badge-{color}">{status.upper()}</span>'


# ── Report Generators ──────────────────────────────────────

class ReportGenerator:
    """Generér HTML-rapporter fra ReportData."""

    @staticmethod
    def morning_report(data: ReportData) -> str:
        """Morgenrapport — kort overblik."""
        brokers_html = "".join(
            f"<tr><td>{name}</td><td>{_broker_badge(status)}</td></tr>"
            for name, status in data.broker_status.items()
        )

        return f"""
        <html><head>{_BASE_STYLE}</head><body>
        <div class="container">
            <div class="header">☀️ Morgenrapport</div>
            <div class="subtitle">{data.timestamp.strftime('%A %d. %B %Y, %H:%M CET')}</div>

            <div class="kpi-row">
                <div class="kpi">
                    <div class="kpi-label">Portfolio</div>
                    <div class="kpi-value">{data.portfolio_value_dkk:,.0f} DKK</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Overnight P&L</div>
                    <div class="kpi-value {_pnl_class(data.daily_pnl_dkk)}">
                        {_pnl_sign(data.daily_pnl_dkk)} DKK
                    </div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Positioner</div>
                    <div class="kpi-value">{data.positions_count}</div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">Broker Status</div>
                <table>{brokers_html}</table>
            </div>

            <div class="footer">Alpha Trading Platform — automatisk morgenrapport</div>
        </div>
        </body></html>
        """

    @staticmethod
    def evening_report(data: ReportData) -> str:
        """Aftenrapport — detaljeret dagsoverblik."""
        signals_html = ""
        for sig in data.signals[:10]:
            signals_html += (
                f"<tr><td>{sig.get('symbol', '')}</td>"
                f"<td>{sig.get('action', '')}</td>"
                f"<td>{sig.get('strategy', '')}</td>"
                f"<td>{sig.get('confidence', 0):.0f}%</td></tr>"
            )

        tax_html = "".join(f"<li>{s}</li>" for s in data.tax_suggestions[:5])

        winners_html = "".join(
            f"<tr><td>{w.get('symbol', '')}</td>"
            f'<td class="positive">+{w.get("pnl_pct", 0):.1f}%</td></tr>'
            for w in data.top_winners[:5]
        )
        losers_html = "".join(
            f"<tr><td>{l.get('symbol', '')}</td>"
            f'<td class="negative">{l.get("pnl_pct", 0):.1f}%</td></tr>'
            for l in data.top_losers[:5]
        )

        return f"""
        <html><head>{_BASE_STYLE}</head><body>
        <div class="container">
            <div class="header">🌙 Aftenrapport</div>
            <div class="subtitle">{data.timestamp.strftime('%A %d. %B %Y, %H:%M CET')}</div>

            <div class="kpi-row">
                <div class="kpi">
                    <div class="kpi-label">Portfolio</div>
                    <div class="kpi-value">{data.portfolio_value_dkk:,.0f} DKK</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Daglig P&L</div>
                    <div class="kpi-value {_pnl_class(data.daily_pnl_dkk)}">
                        {_pnl_sign(data.daily_pnl_dkk)} DKK ({data.daily_pnl_pct:+.2f}%)
                    </div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">YTD P&L</div>
                    <div class="kpi-value {_pnl_class(data.ytd_pnl_dkk)}">
                        {_pnl_sign(data.ytd_pnl_dkk)} DKK
                    </div>
                </div>
            </div>

            <div class="kpi-row">
                <div class="kpi">
                    <div class="kpi-label">Skattetilgodehavende</div>
                    <div class="kpi-value">{data.tax_credit_balance:,.0f} DKK</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Estimeret skat YTD</div>
                    <div class="kpi-value">{data.estimated_tax_ytd:,.0f} DKK</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Udbytter YTD</div>
                    <div class="kpi-value">{data.dividends_ytd:,.0f} DKK</div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">🏆 Top Winners</div>
                <table>{winners_html if winners_html else '<tr><td>Ingen data</td></tr>'}</table>
            </div>

            <div class="section">
                <div class="section-title">📉 Top Losers</div>
                <table>{losers_html if losers_html else '<tr><td>Ingen data</td></tr>'}</table>
            </div>

            {"<div class='section'><div class='section-title'>📡 Signaler</div><table><th>Symbol</th><th>Action</th><th>Strategi</th><th>Konfidens</th>" + signals_html + "</table></div>" if signals_html else ""}

            {"<div class='section'><div class='section-title'>💡 Skatteoptimering</div><ul>" + tax_html + "</ul></div>" if tax_html else ""}

            <div class="section">
                <div class="section-title">Broker Status</div>
                <table>{"".join(f"<tr><td>{n}</td><td>{_broker_badge(s)}</td></tr>" for n, s in data.broker_status.items())}</table>
            </div>

            <div class="footer">Alpha Trading Platform — automatisk aftenrapport</div>
        </div>
        </body></html>
        """

    @staticmethod
    def weekly_report(data: ReportData, week_pnl: float = 0, week_pnl_pct: float = 0) -> str:
        """Ugentlig rapport — performance summary."""
        return f"""
        <html><head>{_BASE_STYLE}</head><body>
        <div class="container">
            <div class="header">📊 Ugentlig Rapport</div>
            <div class="subtitle">Uge {data.timestamp.isocalendar()[1]}, {data.timestamp.year}</div>

            <div class="kpi-row">
                <div class="kpi">
                    <div class="kpi-label">Portfolio</div>
                    <div class="kpi-value">{data.portfolio_value_dkk:,.0f} DKK</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Ugens P&L</div>
                    <div class="kpi-value {_pnl_class(week_pnl)}">
                        {_pnl_sign(week_pnl)} DKK ({week_pnl_pct:+.2f}%)
                    </div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">YTD P&L</div>
                    <div class="kpi-value {_pnl_class(data.ytd_pnl_dkk)}">
                        {_pnl_sign(data.ytd_pnl_dkk)} DKK
                    </div>
                </div>
            </div>

            <div class="kpi-row">
                <div class="kpi">
                    <div class="kpi-label">Skattetilgodehavende</div>
                    <div class="kpi-value">{data.tax_credit_balance:,.0f} DKK</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Estimeret skat YTD</div>
                    <div class="kpi-value">{data.estimated_tax_ytd:,.0f} DKK</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Positioner</div>
                    <div class="kpi-value">{data.positions_count}</div>
                </div>
            </div>

            <div class="footer">Alpha Trading Platform — automatisk ugerapport</div>
        </div>
        </body></html>
        """


# ── Alarm Emails ───────────────────────────────────────────

class AlarmManager:
    """Send alarm-emails ved kritiske hændelser."""

    def __init__(self, sender: EmailSender | None = None):
        self._sender = sender or EmailSender()
        self._cooldowns: dict[str, datetime] = {}
        self._cooldown_minutes = 30  # Min 30 min mellem alarmer af samme type

    def _should_send(self, alarm_type: str) -> bool:
        last = self._cooldowns.get(alarm_type)
        if last and (datetime.now(TZ_CET) - last).total_seconds() < self._cooldown_minutes * 60:
            return False
        return True

    def _mark_sent(self, alarm_type: str) -> None:
        self._cooldowns[alarm_type] = datetime.now(TZ_CET)

    def drawdown_alarm(self, drawdown_pct: float, portfolio_value: float) -> bool:
        """Alarm: Drawdown over tærskel."""
        alarm_type = "drawdown"
        if not self._should_send(alarm_type):
            return False

        html = f"""
        <html><head>{_BASE_STYLE}</head><body>
        <div class="container">
            <div class="alert">⚠️ DRAWDOWN ALARM</div>
            <div class="header">Drawdown: {drawdown_pct:.1f}%</div>
            <div class="subtitle">{datetime.now(TZ_CET).strftime('%H:%M CET')}</div>
            <div class="kpi-row">
                <div class="kpi">
                    <div class="kpi-label">Portfolio</div>
                    <div class="kpi-value">{portfolio_value:,.0f} DKK</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Drawdown</div>
                    <div class="kpi-value negative">{drawdown_pct:.1f}%</div>
                </div>
            </div>
            <div class="footer">Alpha Trading Platform — automatisk alarm</div>
        </div>
        </body></html>
        """

        sent = self._sender.send(
            f"⚠️ DRAWDOWN {drawdown_pct:.1f}% — Alpha Trader",
            html,
        )
        if sent:
            self._mark_sent(alarm_type)
        return sent

    def broker_disconnected(self, broker_name: str, error: str = "") -> bool:
        """Alarm: Broker disconnected."""
        alarm_type = f"broker_disconnect_{broker_name}"
        if not self._should_send(alarm_type):
            return False

        html = f"""
        <html><head>{_BASE_STYLE}</head><body>
        <div class="container">
            <div class="alert">🔌 BROKER DISCONNECTED</div>
            <div class="header">{broker_name} er disconnected</div>
            <div class="subtitle">{datetime.now(TZ_CET).strftime('%H:%M CET')}</div>
            <p style="color: #e2e8f0;">Fejl: {error or 'Ingen detaljer'}</p>
            <div class="footer">Alpha Trading Platform — automatisk alarm</div>
        </div>
        </body></html>
        """

        sent = self._sender.send(
            f"🔌 {broker_name} DISCONNECTED — Alpha Trader",
            html,
        )
        if sent:
            self._mark_sent(alarm_type)
        return sent

    def tax_credit_low(self, balance: float) -> bool:
        """Alarm: Skattetilgodehavende lavt."""
        alarm_type = "tax_credit_low"
        if not self._should_send(alarm_type):
            return False

        html = f"""
        <html><head>{_BASE_STYLE}</head><body>
        <div class="container">
            <div class="alert">💰 SKATTETILGODEHAVENDE LAVT</div>
            <div class="header">Resterende: {balance:,.0f} DKK</div>
            <div class="subtitle">{datetime.now(TZ_CET).strftime('%H:%M CET')}</div>
            <p style="color: #e2e8f0;">Overvej at justere din strategi for at bevare skattetilgodehavende.</p>
            <div class="footer">Alpha Trading Platform — automatisk alarm</div>
        </div>
        </body></html>
        """

        sent = self._sender.send(
            f"💰 Skattetilgodehavende: {balance:,.0f} DKK — Alpha Trader",
            html,
        )
        if sent:
            self._mark_sent(alarm_type)
        return sent

    def custom_alarm(self, title: str, message: str) -> bool:
        """Send custom alarm."""
        html = f"""
        <html><head>{_BASE_STYLE}</head><body>
        <div class="container">
            <div class="alert">🚨 {title}</div>
            <div class="subtitle">{datetime.now(TZ_CET).strftime('%H:%M CET')}</div>
            <p style="color: #e2e8f0;">{message}</p>
            <div class="footer">Alpha Trading Platform — automatisk alarm</div>
        </div>
        </body></html>
        """
        return self._sender.send(f"🚨 {title} — Alpha Trader", html)


# ── High-Level Report Runner ──────────────────────────────

class EmailReportRunner:
    """
    Integrerer med DailyScheduler til at sende rapporter.

    Usage:
        runner = EmailReportRunner()
        runner.send_morning_report()
        runner.send_evening_report()
    """

    def __init__(self, sender: EmailSender | None = None):
        self._sender = sender or EmailSender()
        self._generator = ReportGenerator()
        self._alarm = AlarmManager(self._sender)

    def _gather_data(self) -> ReportData:
        """Hent aktuel data til rapport."""
        data = ReportData()

        try:
            from src.broker.broker_router import BrokerRouter
            from src.broker.aggregated_portfolio import AggregatedPortfolio
            router = BrokerRouter()
            portfolio = AggregatedPortfolio(router)
            summary = portfolio.get_total_value("DKK")
            data.portfolio_value_dkk = summary.total_value_dkk
            data.daily_pnl_dkk = summary.total_unrealized_pnl_dkk
            if summary.total_value_dkk:
                data.daily_pnl_pct = data.daily_pnl_dkk / summary.total_value_dkk * 100
            data.positions_count = len(portfolio.get_all_positions("DKK"))
        except Exception as e:
            logger.warning(f"[email-report] Portfolio data unavailable: {e}")

        try:
            from src.broker.connection_manager import ConnectionManager
            cm = ConnectionManager()
            dashboard = cm.get_dashboard_status()
            data.broker_status = {
                b: info.get("status", "unknown")
                for b, info in dashboard.get("brokers", {}).items()
            }
        except Exception:
            pass

        try:
            from src.tax.tax_credit_tracker import TaxCreditTracker
            tracker = TaxCreditTracker()
            data.tax_credit_balance = tracker.balance
        except Exception:
            pass

        try:
            from src.tax.corporate_tax import CorporateTaxCalculator
            calc = CorporateTaxCalculator()
            ytd = calc.ytd_estimated_tax([])
            data.estimated_tax_ytd = ytd.get("estimated_gross_tax", 0)
        except Exception:
            pass

        return data

    def send_morning_report(self) -> bool:
        data = self._gather_data()
        html = self._generator.morning_report(data)
        return self._sender.send(
            f"☀️ Morgen: {data.portfolio_value_dkk:,.0f} DKK — Alpha Trader",
            html,
        )

    def send_evening_report(self) -> bool:
        data = self._gather_data()
        html = self._generator.evening_report(data)
        return self._sender.send(
            f"🌙 Aften: {data.daily_pnl_dkk:+,.0f} DKK ({data.daily_pnl_pct:+.2f}%) — Alpha Trader",
            html,
        )

    def send_weekly_report(self) -> bool:
        data = self._gather_data()
        html = self._generator.weekly_report(data)
        return self._sender.send(
            f"📊 Uge {data.timestamp.isocalendar()[1]}: {data.portfolio_value_dkk:,.0f} DKK — Alpha Trader",
            html,
        )

    @property
    def alarm(self) -> AlarmManager:
        return self._alarm
