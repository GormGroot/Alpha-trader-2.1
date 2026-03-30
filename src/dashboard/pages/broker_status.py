"""
Broker Status — connection monitoring dashboard.

Dashboard-side: /status

Features:
  - Connection status per broker (grøn/gul/rød)
  - Response time og uptime
  - Cash balance og positions count
  - Reconnect-knapper
"""

from __future__ import annotations

from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
from datetime import datetime

from loguru import logger
from src.dashboard.i18n import t

COLORS = {
    "bg": "#0f1117", "card": "#1a1c24", "accent": "#00d4aa",
    "red": "#ff4757", "green": "#2ed573", "blue": "#3498db",
    "orange": "#ffa502", "text": "#e2e8f0", "muted": "#64748b",
    "border": "#2d3748",
}

# Broker display config
BROKER_INFO = {
    "alpaca": {"name": "Alpaca", "icon": "🇺🇸", "desc_key": "broker_status.us_stocks_crypto"},
    "saxo": {"name": "Saxo Bank", "icon": "🇩🇰", "desc_key": "broker_status.eu_etfs_bonds"},
    "ibkr": {"name": "Interactive Brokers", "icon": "🌍", "desc_key": "broker_status.eu_stocks_forex"},
    "nordnet": {"name": "Nordnet", "icon": "🇩🇰", "desc_key": "broker_status.nordic_stocks"},
}


def _broker_status_card(
    broker_key: str,
    status: str = "unknown",
    response_ms: float = 0,
    uptime: float = 100,
    positions: int = 0,
    cash: str = "—",
    last_check: str = "—",
) -> dbc.Card:
    """Generér broker status kort."""
    info = BROKER_INFO.get(broker_key, {"name": broker_key, "icon": "📊", "desc_key": ""})

    status_colors = {
        "connected": COLORS["green"],
        "degraded": COLORS["orange"],
        "disconnected": COLORS["red"],
        "unknown": COLORS["muted"],
    }
    status_color = status_colors.get(status, COLORS["muted"])
    status_badge = {
        "connected": "success",
        "degraded": "warning",
        "disconnected": "danger",
        "unknown": "secondary",
    }.get(status, "secondary")

    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4([
                        html.Span(info["icon"], className="me-2"),
                        info["name"],
                    ], style={"color": COLORS["text"]}, className="mb-1"),
                    html.P(t(info["desc_key"]) if info["desc_key"] else "", style={
                        "color": COLORS["muted"], "fontSize": "0.85rem",
                    }, className="mb-2"),
                    dbc.Badge(status.upper(), color=status_badge, className="me-2"),
                    html.Span(
                        f"{response_ms:.0f}ms" if response_ms > 0 else "",
                        style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                    ),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Span(t('broker_status.uptime'), style={"color": COLORS["muted"]}),
                            html.Span(f"{uptime:.1f}%", style={"color": status_color}),
                        ]),
                        html.Div([
                            html.Span(t('broker_status.positions'), style={"color": COLORS["muted"]}),
                            html.Span(str(positions), style={"color": COLORS["text"]}),
                        ]),
                        html.Div([
                            html.Span(t('broker_status.cash'), style={"color": COLORS["muted"]}),
                            html.Span(cash, style={"color": COLORS["text"]}),
                        ]),
                        html.Div([
                            html.Span(t('broker_status.last_checked'), style={"color": COLORS["muted"]}),
                            html.Span(last_check, style={
                                "color": COLORS["muted"], "fontSize": "0.8rem",
                            }),
                        ]),
                    ], style={"textAlign": "right", "fontSize": "0.9rem"}),
                ], width=6),
            ]),
        ]),
    ], style={
        "backgroundColor": COLORS["card"],
        "border": f"1px solid {COLORS['border']}",
        "borderLeft": f"4px solid {status_color}",
        "borderRadius": "8px",
    }, className="mb-3")


# ── Layout ──────────────────────────────────────────────────

def broker_status_layout() -> html.Div:
    return html.Div([
        dcc.Interval(id="status-refresh", interval=15_000, n_intervals=0),

        dbc.Row([
            dbc.Col(html.H2(t('broker_status.title'), style={"color": COLORS["text"]}), width=8),
            dbc.Col(html.Div(
                id="status-overall",
                className="text-end",
                style={"paddingTop": "8px"},
            ), width=4),
        ], className="mb-4"),

        # Broker cards
        html.Div(id="status-broker-cards"),

        # Connection log
        dbc.Card([
            dbc.CardHeader(
                html.H5(t('broker_status.connection_log'), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody(
                html.Div(id="status-connection-log"),
                style={"backgroundColor": COLORS["card"]},
            ),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mt-4"),

    ], style={"padding": "20px", "backgroundColor": COLORS["bg"]})


# ── Callbacks ───────────────────────────────────────────────

def register_status_callbacks(app: object) -> None:

    @app.callback(
        [
            Output("status-broker-cards", "children"),
            Output("status-overall", "children"),
            Output("status-connection-log", "children"),
        ],
        Input("status-refresh", "n_intervals"),
    )
    def update_broker_status(_n: int) -> tuple:
        cards = []

        try:
            from src.broker.connection_manager import ConnectionManager
            manager = ConnectionManager()
            dashboard = manager.get_dashboard_status()

            for broker_key, health in dashboard.get("brokers", {}).items():
                cards.append(_broker_status_card(
                    broker_key=broker_key,
                    status=health.get("status", "unknown"),
                    response_ms=health.get("response_time_ms", 0),
                    uptime=health.get("uptime_pct", 100),
                    last_check=health.get("last_check", "—"),
                ))

            overall = dashboard.get("overall_status", "unknown")
            connected = dashboard.get("connected_count", 0)
            total = dashboard.get("total_count", 0)

            overall_badge = dbc.Badge(
                f"{connected}/{total} Connected",
                color="success" if connected == total else "warning",
            )
        except Exception:
            overall_badge = dbc.Badge(t('broker_status.not_initialized'), color="secondary")

        if not cards:
            # Show placeholder cards for all 4 brokers
            for bk in ["alpaca", "saxo", "ibkr", "nordnet"]:
                cards.append(_broker_status_card(bk, status="unknown"))

        # Connection log
        log_entries = []
        try:
            from src.broker.connection_manager import ConnectionManager
            manager = ConnectionManager()
            events = manager.get_recent_events(limit=20) if hasattr(manager, "get_recent_events") else []
            for evt in events:
                ts = evt.get("timestamp", "")
                msg = evt.get("message", "")
                level = evt.get("level", "info")
                color = {
                    "error": COLORS["red"],
                    "warning": COLORS["orange"],
                    "info": COLORS["muted"],
                }.get(level, COLORS["muted"])
                log_entries.append(html.Div([
                    html.Span(f"[{ts}] ", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                    html.Span(msg, style={"color": color, "fontSize": "0.85rem"}),
                ], style={"padding": "2px 0"}))
        except Exception:
            pass

        if not log_entries:
            log_entries = [html.Div(
                t('broker_status.no_connection_events'),
                style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"},
            )]

        connection_log = html.Div(log_entries, style={
            "maxHeight": "300px", "overflowY": "auto",
            "fontFamily": "monospace",
        })

        return cards, overall_badge, connection_log
