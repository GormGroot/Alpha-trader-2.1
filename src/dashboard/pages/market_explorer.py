"""
Market Explorer — markedsheatmap, instrumentsøgning, watchlists.

Dashboard-side: /markets

Features:
  - Europæisk + US markeds-heatmap
  - Instrument-søg på tværs af brokers
  - Watchlists med live priser
  - Markedsnyheder med sentiment
"""

from __future__ import annotations

from dash import dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime

from loguru import logger

from src.dashboard.i18n import t

COLORS = {
    "bg": "#0f1117", "card": "#1a1c24", "accent": "#00d4aa",
    "red": "#ff4757", "green": "#2ed573", "blue": "#3498db",
    "orange": "#ffa502", "purple": "#a855f7", "text": "#e2e8f0",
    "muted": "#64748b", "border": "#2d3748",
}

# ── Market indices config ──────────────────────────────────

MARKET_INDICES = {
    "🇺🇸 S&P 500": "^GSPC",
    "🇺🇸 Nasdaq": "^IXIC",
    "🇺🇸 Dow Jones": "^DJI",
    "🇩🇰 OMX C25": "^OMXC25",
    "🇸🇪 OMX S30": "^OMX",
    "🇩🇪 DAX": "^GDAXI",
    "🇬🇧 FTSE 100": "^FTSE",
    "🇪🇺 STOXX 50": "^STOXX50E",
    "🇫🇷 CAC 40": "^FCHI",
    "🇯🇵 Nikkei 225": "^N225",
    "🇭🇰 Hang Seng": "^HSI",
    "₿ Bitcoin": "BTC-USD",
}

# Default watchlist symbols
DEFAULT_WATCHLIST = [
    {"symbol": "NOVO-B.CO", "name": "Novo Nordisk", "broker": "nordnet"},
    {"symbol": "AAPL", "name": "Apple", "broker": "alpaca"},
    {"symbol": "MSFT", "name": "Microsoft", "broker": "alpaca"},
    {"symbol": "ASML.AS", "name": "ASML", "broker": "ibkr"},
    {"symbol": "SAP.DE", "name": "SAP", "broker": "ibkr"},
    {"symbol": "DSV.CO", "name": "DSV", "broker": "nordnet"},
    {"symbol": "MAERSK-B.CO", "name": "Mærsk", "broker": "nordnet"},
    {"symbol": "BTC-USD", "name": "Bitcoin", "broker": "alpaca"},
]


def _index_card(
    name: str,
    value: str = "—",
    change_pct: float = 0.0,
) -> dbc.Card:
    """Generér indeks-kort med daglig ændring."""
    change_color = COLORS["green"] if change_pct >= 0 else COLORS["red"]
    arrow = "▲" if change_pct >= 0 else "▼"

    return dbc.Card(
        dbc.CardBody([
            html.P(name, className="mb-1", style={
                "color": COLORS["muted"], "fontSize": "0.8rem",
                "whiteSpace": "nowrap", "overflow": "hidden",
                "textOverflow": "ellipsis",
            }),
            html.H5(value, className="mb-0", style={"color": COLORS["text"]}),
            html.Span(
                f"{arrow} {change_pct:+.2f}%",
                style={"color": change_color, "fontSize": "0.85rem"},
            ),
        ]),
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "minWidth": "160px",
        },
        className="me-2 mb-2",
    )


def _watchlist_row(
    symbol: str,
    name: str,
    price: str = "—",
    change_pct: float = 0.0,
    broker: str = "",
) -> html.Tr:
    """Generér watchlist-tabelrække."""
    change_color = COLORS["green"] if change_pct >= 0 else COLORS["red"]
    broker_colors = {
        "alpaca": "primary", "saxo": "warning",
        "ibkr": "info", "nordnet": "success",
    }

    return html.Tr([
        html.Td(
            html.Strong(symbol, style={"color": COLORS["accent"]}),
            style={"padding": "8px"},
        ),
        html.Td(name, style={"color": COLORS["text"], "padding": "8px"}),
        html.Td(price, style={"color": COLORS["text"], "padding": "8px", "textAlign": "right"}),
        html.Td(
            html.Span(f"{change_pct:+.2f}%", style={"color": change_color}),
            style={"padding": "8px", "textAlign": "right"},
        ),
        html.Td(
            dbc.Badge(broker.upper(), color=broker_colors.get(broker, "secondary"), pill=True)
            if broker else html.Span(),
            style={"padding": "8px", "textAlign": "center"},
        ),
    ], style={"borderBottom": f"1px solid {COLORS['border']}"})


# ── Layout ──────────────────────────────────────────────────

def market_explorer_layout() -> html.Div:
    return html.Div([
        dcc.Interval(id="market-refresh", interval=30_000, n_intervals=0),

        html.H2(t('markets.title'), style={"color": COLORS["text"]}, className="mb-2"),
        html.P(
            t('markets.subtitle'),
            style={"color": COLORS["muted"]},
            className="mb-4",
        ),

        # ── Market Indices Row ──
        dbc.Card([
            dbc.CardHeader(
                html.H5(t('markets.market_indices'), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody(
                html.Div(id="market-indices-grid"),
                style={"backgroundColor": COLORS["card"]},
            ),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # ── Heatmap ──
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(html.H5(t('markets.sector_heatmap'), className="mb-0"), width=6),
                    dbc.Col(
                        dbc.RadioItems(
                            id="heatmap-region",
                            options=[
                                {"label": "US", "value": "us"},
                                {"label": "EU", "value": "eu"},
                                {"label": t('markets.north'), "value": "nordic"},
                            ],
                            value="us",
                            inline=True,
                            style={"color": COLORS["text"]},
                        ),
                        width=6,
                        className="text-end",
                    ),
                ]),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody(
                dcc.Graph(id="market-heatmap"),
                style={"backgroundColor": COLORS["card"]},
            ),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # ── Search + Watchlist ──
        dbc.Row([
            # Instrument søgning
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H5(t('markets.instrument_search'), className="mb-0"),
                        style={"backgroundColor": COLORS["card"]},
                    ),
                    dbc.CardBody([
                        dbc.InputGroup([
                            dbc.Input(
                                id="market-search-input",
                                placeholder=t('markets.search_placeholder'),
                                type="text",
                                style={
                                    "backgroundColor": COLORS["bg"],
                                    "color": COLORS["text"],
                                    "border": f"1px solid {COLORS['border']}",
                                },
                            ),
                            dbc.Button(
                                t('common.search'),
                                id="market-search-btn",
                                color="secondary",
                                outline=True,
                            ),
                        ], className="mb-3"),
                        html.Div(id="market-search-results"),
                    ], style={"backgroundColor": COLORS["card"]}),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], width=5),

            # Watchlist
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Row([
                            dbc.Col(html.H5(t('markets.watchlist'), className="mb-0"), width=8),
                            dbc.Col(
                                html.Span(
                                    id="watchlist-count",
                                    style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                                ),
                                width=4,
                                className="text-end",
                            ),
                        ]),
                        style={"backgroundColor": COLORS["card"]},
                    ),
                    dbc.CardBody(
                        html.Div(id="market-watchlist"),
                        style={
                            "backgroundColor": COLORS["card"],
                            "maxHeight": "400px",
                            "overflowY": "auto",
                        },
                    ),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], width=7),
        ], className="mb-4"),

        # ── Markedsnyheder ──
        dbc.Card([
            dbc.CardHeader(
                html.H5(t('markets.latest_news'), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody(
                html.Div(id="market-news-feed"),
                style={"backgroundColor": COLORS["card"]},
            ),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),

    ], style={"padding": "20px", "backgroundColor": COLORS["bg"]})


# ── Callbacks ───────────────────────────────────────────────

def register_market_callbacks(app: object) -> None:
    """Registrér market explorer callbacks."""

    @app.callback(
        Output("market-indices-grid", "children"),
        Input("market-refresh", "n_intervals"),
    )
    def update_indices(_n: int) -> list:
        cards = []
        try:
            import yfinance as yf
            for display_name, ticker in MARKET_INDICES.items():
                try:
                    tk = yf.Ticker(ticker)
                    info = tk.fast_info
                    price = getattr(info, "last_price", None) or 0
                    prev = getattr(info, "previous_close", None) or price
                    change_pct = ((price - prev) / prev * 100) if prev else 0
                    cards.append(_index_card(
                        display_name,
                        f"{price:,.2f}",
                        change_pct,
                    ))
                except Exception:
                    cards.append(_index_card(display_name))
        except ImportError:
            for name in MARKET_INDICES:
                cards.append(_index_card(name))

        if not cards:
            for name in MARKET_INDICES:
                cards.append(_index_card(name))

        # Wrap in responsive grid
        return dbc.Row([
            dbc.Col(card, xs=6, sm=4, md=3, lg=2) for card in cards
        ], className="g-2")

    @app.callback(
        Output("market-heatmap", "figure"),
        [
            Input("market-refresh", "n_intervals"),
            Input("heatmap-region", "value"),
        ],
    )
    def update_heatmap(_n: int, region: str) -> go.Figure:
        # Sector definitions per region
        # Sector display names come from translation files
        us_sectors = t('heatmap.sectors_us')
        eu_sectors = t('heatmap.sectors_eu')
        nordic_sectors = t('heatmap.sectors_nordic')

        # Symbol lists keyed by sector index position
        us_symbols = [
            ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
            ["JNJ", "UNH", "PFE", "LLY", "ABBV"],
            ["JPM", "BAC", "GS", "MS", "BLK"],
            ["XOM", "CVX", "COP", "SLB", "EOG"],
            ["AMZN", "TSLA", "HD", "NKE", "SBUX"],
            ["CAT", "BA", "HON", "GE", "MMM"],
        ]
        eu_symbols = [
            ["ASML.AS", "SAP.DE", "ERIC-B.ST"],
            ["ROG.SW", "NOVN.SW", "AZN.L"],
            ["HSBA.L", "BNP.PA", "SAN.PA"],
            ["SHEL.L", "TTE.PA", "EQNR.OL"],
            ["MC.PA", "RMS.PA", "KER.PA"],
            ["SIE.DE", "ABB.ST", "AIR.PA"],
        ]
        nordic_symbols = [
            ["NOVO-B.CO", "LUND-B.ST", "ORNBV.HE"],
            ["MAERSK-B.CO", "FRO.OL", "GOGL.OL"],
            ["DANSKE.CO", "NDA-DK.CO", "SEB-A.ST"],
            ["DSV.CO", "SAND.ST", "ATCO-A.ST"],
            ["EQNR.OL", "ORSTED.CO", "VWS.CO"],
            ["NETS.CO", "SINCH.ST", "SIM.CO"],
        ]

        def _build_sector_data(sectors, symbols_list):
            return {
                "sectors": sectors,
                "symbols": {sectors[i]: symbols_list[i] for i in range(len(sectors))},
            }

        sector_data = {
            "us": _build_sector_data(us_sectors, us_symbols),
            "eu": _build_sector_data(eu_sectors, eu_symbols),
            "nordic": _build_sector_data(nordic_sectors, nordic_symbols),
        }

        rd = sector_data.get(region, sector_data["us"])
        sectors = rd["sectors"]

        # Build treemap data
        labels, parents, values, colors_list = [], [], [], []

        for sector in sectors:
            labels.append(sector)
            parents.append("")
            values.append(0)
            colors_list.append(0)

            for sym in rd["symbols"].get(sector, []):
                labels.append(sym)
                parents.append(sector)
                # Placeholder random performance (live data will replace)
                import random
                perf = random.uniform(-3, 3)
                values.append(abs(perf) + 1)
                colors_list.append(perf)

        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors_list,
                colorscale=[[0, COLORS["red"]], [0.5, COLORS["border"]], [1, COLORS["green"]]],
                cmid=0,
                line=dict(width=1, color=COLORS["border"]),
            ),
            textinfo="label+text",
            texttemplate="%{label}<br>%{color:+.1f}%",
            textfont=dict(color=COLORS["text"]),
        ))

        fig.update_layout(
            paper_bgcolor="#1a1c24",
            plot_bgcolor="#1a1c24",
            height=400,
            margin=dict(l=5, r=5, t=5, b=5),
        )
        return fig

    @app.callback(
        Output("market-search-results", "children"),
        Input("market-search-btn", "n_clicks"),
        State("market-search-input", "value"),
        prevent_initial_call=True,
    )
    def search_instruments(_clicks: int, query: str | None) -> html.Div:
        if not query or len(query) < 2:
            return html.Div(
                t('markets.min_2_chars'),
                style={"color": COLORS["muted"], "textAlign": "center", "padding": "10px"},
            )

        results = []

        # Search via broker router if available
        try:
            from src.broker.broker_router import BrokerRouter, detect_exchange, detect_asset_type
            exchange = detect_exchange(query.upper())
            asset_type = detect_asset_type(query.upper())

            results.append(dbc.ListGroupItem([
                html.Strong(query.upper(), style={"color": COLORS["accent"]}),
                html.Span(f" — {t('trading.exchange')}: {exchange or 'auto'}, Type: {asset_type or 'auto'}",
                          style={"color": COLORS["muted"]}),
                dbc.Button(
                    t('markets.add_to_watchlist'),
                    size="sm",
                    color="outline-success",
                    className="float-end",
                    disabled=True,
                    title=t('markets.watchlist_coming_soon'),
                ),
            ], style={
                "backgroundColor": COLORS["bg"],
                "border": f"1px solid {COLORS['border']}",
            }))
        except Exception:
            pass

        # Try yfinance search
        try:
            import yfinance as yf
            tk = yf.Ticker(query.upper())
            info = tk.info or {}
            name = info.get("shortName", info.get("longName", query.upper()))
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            currency = info.get("currency", "")

            results.append(dbc.ListGroupItem([
                html.Div([
                    html.Strong(query.upper(), style={"color": COLORS["accent"]}),
                    html.Span(f" — {name}", style={"color": COLORS["text"]}),
                ]),
                html.Div([
                    html.Span(f"{price:,.2f} {currency}", style={"color": COLORS["text"]}),
                    html.Span(
                        f" · {t('common.sector')}: {info.get('sector', '—')}",
                        style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                    ),
                ]),
            ], style={
                "backgroundColor": COLORS["bg"],
                "border": f"1px solid {COLORS['border']}",
            }))
        except Exception:
            pass

        if not results:
            return html.Div(
                f"{t('markets.no_results')} '{query}'",
                style={"color": COLORS["muted"], "textAlign": "center", "padding": "10px"},
            )

        return dbc.ListGroup(results)

    @app.callback(
        [
            Output("market-watchlist", "children"),
            Output("watchlist-count", "children"),
        ],
        Input("market-refresh", "n_intervals"),
    )
    def update_watchlist(_n: int) -> tuple:
        headers = html.Thead(html.Tr([
            html.Th(t('common.symbol'), style={"color": COLORS["muted"], "padding": "8px", "fontSize": "0.8rem"}),
            html.Th(t('common.name'), style={"color": COLORS["muted"], "padding": "8px", "fontSize": "0.8rem"}),
            html.Th(t('common.price'), style={"color": COLORS["muted"], "padding": "8px", "textAlign": "right", "fontSize": "0.8rem"}),
            html.Th(t('common.change'), style={"color": COLORS["muted"], "padding": "8px", "textAlign": "right", "fontSize": "0.8rem"}),
            html.Th(t('common.broker'), style={"color": COLORS["muted"], "padding": "8px", "textAlign": "center", "fontSize": "0.8rem"}),
        ]))

        rows = []
        for item in DEFAULT_WATCHLIST:
            price_str = "—"
            change = 0.0

            try:
                import yfinance as yf
                tk = yf.Ticker(item["symbol"])
                fi = tk.fast_info
                price = getattr(fi, "last_price", None) or 0
                prev = getattr(fi, "previous_close", None) or price
                change = ((price - prev) / prev * 100) if prev else 0
                price_str = f"{price:,.2f}"
            except Exception:
                pass

            rows.append(_watchlist_row(
                symbol=item["symbol"],
                name=item["name"],
                price=price_str,
                change_pct=change,
                broker=item.get("broker", ""),
            ))

        body = html.Tbody(rows) if rows else html.Tbody([
            html.Tr([html.Td(
                t('markets.no_instruments'),
                colSpan=5,
                style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"},
            )])
        ])

        table = html.Table([headers, body], style={
            "width": "100%", "color": COLORS["text"], "fontSize": "0.9rem",
        })

        count_text = f"{len(DEFAULT_WATCHLIST)} {t('markets.instruments')}"

        return table, count_text

    @app.callback(
        Output("market-news-feed", "children"),
        Input("market-refresh", "n_intervals"),
    )
    def update_news(_n: int) -> html.Div:
        # Placeholder — will integrate with news pipeline when available
        try:
            from src.data.news_fetcher import NewsFetcher
            fetcher = NewsFetcher()
            articles = fetcher.get_latest(limit=5)
            items = []
            for a in articles:
                sentiment_color = COLORS["green"] if a.get("sentiment", 0) > 0.2 else (
                    COLORS["red"] if a.get("sentiment", 0) < -0.2 else COLORS["muted"]
                )
                items.append(dbc.ListGroupItem([
                    html.Div([
                        html.Strong(a.get("title", ""), style={"color": COLORS["text"]}),
                        dbc.Badge(
                            f"{a.get('sentiment', 0):+.2f}",
                            color="success" if a.get("sentiment", 0) > 0.2 else (
                                "danger" if a.get("sentiment", 0) < -0.2 else "secondary"
                            ),
                            className="ms-2",
                        ),
                    ]),
                    html.P(
                        a.get("summary", "")[:200],
                        style={"color": COLORS["muted"], "fontSize": "0.85rem", "marginBottom": 0},
                    ),
                ], style={
                    "backgroundColor": COLORS["bg"],
                    "border": f"1px solid {COLORS['border']}",
                }))
            return dbc.ListGroup(items)
        except Exception:
            return html.Div(
                t('markets.news_unavailable'),
                style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"},
            )
