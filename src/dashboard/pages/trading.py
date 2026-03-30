"""
Trading Page — smart ordre-panel med multi-broker routing.

Dashboard-side: /trading

Features:
  - Symbol-søg med autocomplete
  - Smart broker-routing preview
  - BUY/SELL + Market/Limit ordre
  - FX preview (DKK ↔ foreign currency)
  - Skatteimpact preview
  - Open orders med cancel
  - Recent trades feed
"""

from __future__ import annotations

import dash
from dash import dcc, html, callback, Input, Output, State, no_update
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


# ── Layout ──────────────────────────────────────────────────

def _build_recent_trades():
    """Build the recent trades table from closed_trades DB."""
    try:
        import sqlite3
        from pathlib import Path

        portfolio_db = Path("data_cache/paper_portfolio.db")
        if not portfolio_db.exists():
            return html.Div(t('trading.no_trades_today'),
                            style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"})

        with sqlite3.connect(portfolio_db) as db:
            db.row_factory = sqlite3.Row
            closed = db.execute("""
                SELECT symbol, side, qty, entry_price, exit_price,
                       exit_reason, exit_time
                FROM closed_trades
                ORDER BY exit_time DESC
                LIMIT 20
            """).fetchall()

        if not closed:
            return html.Div(t('trading.no_trades_today'),
                            style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"})

        rows = []
        for tr in closed:
            pnl = (tr["exit_price"] - tr["entry_price"]) * tr["qty"]
            if tr["side"] == "short":
                pnl = -pnl
            exit_ts = tr["exit_time"] or ""
            time_str = exit_ts[11:16] if len(exit_ts) > 16 else ""
            side_label = "SELL" if tr["side"] == "long" else "BUY"
            side_color = COLORS["red"] if side_label == "SELL" else COLORS["green"]
            pnl_color = COLORS["green"] if pnl >= 0 else COLORS["red"]
            rows.append(html.Tr([
                html.Td(time_str, style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                html.Td(tr["symbol"], style={"color": COLORS["accent"]}),
                html.Td(side_label, style={"color": side_color}),
                html.Td(f"{tr['qty']:.0f}", style={"color": COLORS["text"]}),
                html.Td(f"${pnl:+,.0f}", style={"color": pnl_color}),
                html.Td((tr["exit_reason"] or "")[:20], style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
            ], style={"borderBottom": f"1px solid {COLORS['border']}"}))

        header = html.Thead(html.Tr([
            html.Th(h, style={"color": COLORS["muted"], "padding": "6px", "fontSize": "0.8rem"})
            for h in [t('common.time'), t('common.symbol'), t('common.side'), t('common.quantity'), t('portfolio.pnl'), t('common.reason')]
        ]))
        return html.Table([header, html.Tbody(rows)], style={
            "width": "100%", "color": COLORS["text"], "fontSize": "0.9rem",
        })
    except Exception:
        return html.Div(t('trading.no_trades_today'),
                        style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"})


def _get_sell_options() -> list[dict]:
    """Build dropdown options from current portfolio positions."""
    options = []
    try:
        router = None
        try:
            from src.broker.registry import get_router
            router = get_router()
        except Exception:
            pass
        if router is None:
            return options
        positions = router.get_positions()
        for pos in sorted(positions, key=lambda p: getattr(p, "symbol", "")):
            sym = getattr(pos, "symbol", "")
            side = getattr(pos, "side", "long")
            qty = getattr(pos, "qty", 0)
            pnl_pct = getattr(pos, "unrealized_pnl_pct", 0) or 0
            sign = "+" if pnl_pct >= 0 else ""
            side_tag = " [SHORT]" if side == "short" else ""
            label = f"{sym}{side_tag}  \u2014  {qty:.0f} shares  ({sign}{pnl_pct*100:.1f}%)"
            options.append({"label": label, "value": sym})
    except Exception:
        pass
    return options


def trading_layout() -> html.Div:
    sell_options = _get_sell_options()

    return html.Div([
        # Refresh disabled during form use — manual refresh via trade submission
        dcc.Store(id="trading-refresh-trigger", data=0),

        html.H2(t('trading.title'), style={"color": COLORS["text"]}, className="mb-4"),

        dbc.Row([
            # Ordre Panel (venstre)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H5(t('trading.new_order'), className="mb-0"),
                        style={"backgroundColor": COLORS["card"]},
                    ),
                    dbc.CardBody([
                        # Asset class filter
                        dbc.Label(t('common.type'), style={"color": COLORS["muted"]}),
                        dbc.RadioItems(
                            id="trade-asset-class",
                            options=[
                                {"label": "Stock", "value": "stock"},
                                {"label": "Bond", "value": "bond"},
                                {"label": "Crypto", "value": "crypto"},
                                {"label": "Commodity", "value": "commodity"},
                            ],
                            value="stock",
                            inline=True,
                            className="mb-3",
                            style={"color": COLORS["text"]},
                        ),

                        # Symbol search / portfolio dropdown
                        dbc.Label(t('common.symbol'), style={"color": COLORS["muted"]}),
                        # Dropdown for BUY — populated by asset class callback
                        dcc.Dropdown(
                            id="trade-symbol",
                            placeholder=t('trading.search_symbol_placeholder'),
                            options=[],
                            searchable=True,
                            style={
                                "backgroundColor": COLORS["bg"],
                                "color": COLORS["text"],
                            },
                            className="mb-2",
                        ),
                        # Dropdown for SELL — pre-populated with portfolio positions
                        dcc.Dropdown(
                            id="trade-symbol-sell",
                            placeholder="Select position to sell...",
                            options=sell_options,
                            searchable=True,
                            style={
                                "backgroundColor": COLORS["bg"],
                                "color": COLORS["text"],
                                "display": "none",
                            },
                            className="mb-2",
                        ),

                        # Routing info
                        html.Div(id="trade-routing-info", className="mb-3"),

                        # Side toggle
                        dbc.Label(t('common.side'), style={"color": COLORS["muted"]}),
                        dbc.RadioItems(
                            id="trade-side",
                            options=[
                                {"label": t('common.buy'), "value": "buy"},
                                {"label": t('common.sell'), "value": "sell"},
                            ],
                            value="buy",
                            inline=True,
                            className="mb-3",
                            style={"color": COLORS["text"]},
                        ),

                        # Order type
                        dbc.Label(t('common.type'), style={"color": COLORS["muted"]}),
                        dbc.Select(
                            id="trade-order-type",
                            options=[
                                {"label": t('common.market'), "value": "market"},
                                {"label": t('common.limit'), "value": "limit"},
                            ],
                            value="market",
                            style={
                                "backgroundColor": COLORS["bg"],
                                "color": COLORS["text"],
                                "border": f"1px solid {COLORS['border']}",
                            },
                            className="mb-3",
                        ),

                        # Quantity
                        dbc.Label(t('common.quantity'), style={"color": COLORS["muted"]}),
                        dbc.Input(
                            id="trade-qty",
                            type="number",
                            placeholder=t('common.quantity'),
                            min=0,
                            step=1,
                            style={
                                "backgroundColor": COLORS["bg"],
                                "color": COLORS["text"],
                                "border": f"1px solid {COLORS['border']}",
                            },
                            className="mb-3",
                        ),

                        # Limit price (conditional)
                        html.Div([
                            dbc.Label(t('trading.limit_price'), style={"color": COLORS["muted"]}),
                            dbc.Input(
                                id="trade-limit-price",
                                type="number",
                                placeholder=t('trading.limit_price_placeholder'),
                                step=0.01,
                                style={
                                    "backgroundColor": COLORS["bg"],
                                    "color": COLORS["text"],
                                    "border": f"1px solid {COLORS['border']}",
                                },
                            ),
                        ], id="trade-limit-div", className="mb-3", style={"display": "none"}),

                        # Price / Cash / Cost info
                        html.Div(id="trade-price-info", className="mb-3"),

                        # Preview
                        html.Div(id="trade-preview", className="mb-3"),

                        # Confirm button
                        dbc.Button(
                            t('trading.place_order'),
                            id="trade-submit-btn",
                            color="success",
                            size="lg",
                            className="w-100",
                            disabled=True,
                        ),

                        # Result
                        html.Div(id="trade-result", className="mt-3"),

                    ], style={"backgroundColor": COLORS["card"]}),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], width=4),

            # Open Orders + Recent Trades (højre)
            dbc.Col([
                # Open Orders
                dbc.Card([
                    dbc.CardHeader(
                        html.H5(t('trading.open_orders'), className="mb-0"),
                        style={"backgroundColor": COLORS["card"]},
                    ),
                    dbc.CardBody(
                        html.Div(
                            t('trading.no_open_orders'),
                            id="trading-open-orders",
                            style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"},
                        ),
                        style={"backgroundColor": COLORS["card"], "minHeight": "60px"},
                    ),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
                   className="mb-4"),

                # Recent Trades — pre-rendered with fixed height to prevent layout shift
                dbc.Card([
                    dbc.CardHeader(
                        html.H5(t('trading.recent_trades'), className="mb-0"),
                        style={"backgroundColor": COLORS["card"]},
                    ),
                    dbc.CardBody(
                        html.Div(_build_recent_trades(), id="trading-recent-trades"),
                        style={"backgroundColor": COLORS["card"],
                               "minHeight": "400px", "overflowY": "auto", "maxHeight": "500px"},
                    ),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], width=8),
        ]),

        # ── Quick Trade: Buy/Sell Top 10 ──────────────────────────
        html.Hr(style={"borderColor": COLORS["border"], "margin": "24px 0"}),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-graph-up-arrow me-2 text-success"),
                            t('trading.buy_top_10'),
                        ], className="text-light mb-2"),
                        html.P(t('trading.buy_top_10_desc'),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mb-3"),
                        dbc.Button(
                            [html.I(className="bi bi-cart-plus me-2"), t('trading.buy_top_10')],
                            id="quick-buy-top10-btn",
                            color="success",
                            outline=True,
                            className="w-100",
                        ),
                    ], style={"backgroundColor": COLORS["card"]}),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-graph-down-arrow me-2 text-danger"),
                            t('trading.sell_top_10'),
                        ], className="text-light mb-2"),
                        html.P(t('trading.sell_top_10_desc'),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mb-3"),
                        dbc.Button(
                            [html.I(className="bi bi-cart-dash me-2"), t('trading.sell_top_10')],
                            id="quick-sell-top10-btn",
                            color="danger",
                            outline=True,
                            className="w-100",
                        ),
                    ], style={"backgroundColor": COLORS["card"]}),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], width=6),
        ], className="mb-3"),

        # Confirmation modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id="quick-trade-modal-title"), close_button=True),
            dbc.ModalBody(id="quick-trade-modal-body"),
            dbc.ModalFooter([
                dbc.Button(t('common.cancel'), id="quick-trade-cancel-btn",
                           color="secondary", className="me-2"),
                dbc.Button(id="quick-trade-confirm-btn", color="success"),
            ]),
        ], id="quick-trade-modal", is_open=False, centered=True, size="lg"),

        # Hidden stores + result
        dcc.Store(id="quick-trade-action-store", data=None),
        dcc.Store(id="quick-trade-prices-store", data=None),
        dcc.Store(id="quick-trade-cash-store", data=None),
        html.Div(id="quick-trade-result", className="mt-3"),

    ], style={"padding": "20px", "backgroundColor": COLORS["bg"]})


# ── Callbacks ───────────────────────────────────────────────

def register_trading_callbacks(app: object) -> None:
    """Registrér trading callbacks."""

    # ── Populate buy symbol dropdown from asset class + open markets ──
    @app.callback(
        Output("trade-symbol", "options"),
        Input("trade-asset-class", "value"),
        prevent_initial_call=False,
    )
    def _update_symbol_options(asset_class):
        from src.ops.market_calendar import MarketCalendar, MARKET_SYMBOLS
        from src.data.universe import ETFS_BONDS, ETFS_COMMODITIES, CRYPTO_TOP_20

        # Get currently open markets
        try:
            cal = MarketCalendar()
            open_markets = cal.get_open_markets()
        except Exception:
            open_markets = ["crypto", "us_stocks", "etfs"]

        # Collect symbols for the selected asset class from open markets
        symbols = []
        if asset_class == "stock":
            stock_markets = {"us_stocks", "eu_nordic", "london", "japan",
                             "hong_kong", "australia", "new_zealand", "india"}
            for m in open_markets:
                if m in stock_markets:
                    symbols.extend(MARKET_SYMBOLS.get(m, []))
            # Also add ETFs that aren't bonds/commodities
            if "etfs" in open_markets or "us_stocks" in open_markets:
                _skip = set(ETFS_BONDS) | set(ETFS_COMMODITIES)
                for s in MARKET_SYMBOLS.get("etfs", []):
                    if s not in _skip and s not in symbols:
                        symbols.append(s)
        elif asset_class == "bond":
            # Bonds are US-traded ETFs — available when US market is open
            if any(m in open_markets for m in ("us_stocks", "etfs")):
                symbols = list(ETFS_BONDS)
            else:
                symbols = list(ETFS_BONDS)  # show anyway, order may fail
        elif asset_class == "crypto":
            symbols = list(CRYPTO_TOP_20)  # crypto always open
        elif asset_class == "commodity":
            # Commodity ETFs + futures
            if any(m in open_markets for m in ("us_stocks", "etfs", "chicago")):
                symbols = list(ETFS_COMMODITIES)
                symbols.extend(s for s in MARKET_SYMBOLS.get("chicago", [])
                               if s.endswith("=F") and not s.startswith("ES")
                               and not s.startswith("NQ") and not s.startswith("YM")
                               and not s.startswith("RTY") and not s.startswith("Z")
                               and s not in symbols)
            else:
                symbols = list(ETFS_COMMODITIES)

        # Deduplicate and sort
        seen = set()
        unique = []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        unique.sort()

        options = [{"label": s, "value": s} for s in unique]
        return options

    # ── Show price, cash, and order cost when symbol/qty changes ──
    @app.callback(
        Output("trade-price-info", "children"),
        Input("trade-symbol", "value"),
        Input("trade-qty", "value"),
        Input("trade-side", "value"),
        prevent_initial_call=True,
    )
    def _update_price_info(symbol, qty, side):
        if not symbol:
            return ""
        qty = max(int(qty or 0), 0)

        # Fetch price
        price_usd = 0.0
        try:
            import yfinance as _yf_ti
            tk = _yf_ti.Ticker(symbol)
            price_usd = getattr(tk.fast_info, "last_price", 0) or 0
        except Exception:
            pass

        # FX rate
        usd_dkk = 6.90
        try:
            import yfinance as _yf_fx
            fx = _yf_fx.Ticker("DKK=X")
            r = getattr(fx.fast_info, "last_price", None)
            if r and r > 0:
                usd_dkk = r
        except Exception:
            pass

        price_dkk = price_usd * usd_dkk

        # Cash
        cash_dkk = 0.0
        try:
            from src.broker.paper_broker import PaperBroker
            pb = PaperBroker()
            cash_dkk = pb.get_account().cash * usd_dkk
        except Exception:
            pass

        price_rounded = round(price_dkk, 2)
        total_cost = round(price_rounded * qty, 2)
        cash_after = cash_dkk - total_cost if side == "buy" else cash_dkk + total_cost
        cost_color = COLORS["green"] if cash_after >= 0 else COLORS["red"]

        rows = [
            html.Div([
                html.Span(f"{t('common.price')}: ", style={"color": COLORS["muted"]}),
                html.Span(f"{price_rounded:,.2f} kr", style={"color": COLORS["text"], "fontWeight": "600"}),
            ]),
        ]
        if qty > 0:
            rows.append(html.Div([
                html.Span("Total: ", style={"color": COLORS["muted"]}),
                html.Span(f"{total_cost:,.2f} kr", style={"color": COLORS["accent"], "fontWeight": "600"}),
                html.Span(f"  ({qty} \u00d7 {price_rounded:,.2f})",
                           style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
            ]))
        rows.append(html.Div([
            html.Span(f"{t('portfolio.cash')}: ", style={"color": COLORS["muted"]}),
            html.Span(f"{cash_dkk:,.0f} kr", style={"color": COLORS["text"]}),
            html.Span(f" \u2192 {cash_after:,.0f} kr",
                       style={"color": cost_color, "fontWeight": "600"}) if qty > 0 else None,
        ]))

        return html.Div(rows, style={
            "backgroundColor": COLORS["bg"], "border": f"1px solid {COLORS['border']}",
            "borderRadius": "4px", "padding": "10px", "fontSize": "0.9rem",
        })

    # ── Toggle symbol input (buy) vs dropdown (sell) ─────────
    app.clientside_callback(
        """
        function(side) {
            var buyStyle = {
                'backgroundColor': '#0f1117', 'color': '#e2e8f0',
                'display': side === 'buy' ? 'block' : 'none'
            };
            var sellStyle = {
                'backgroundColor': '#0f1117', 'color': '#e2e8f0',
                'display': side === 'sell' ? 'block' : 'none'
            };
            return [buyStyle, sellStyle];
        }
        """,
        [
            Output("trade-symbol", "style"),
            Output("trade-symbol-sell", "style"),
        ],
        Input("trade-side", "value"),
    )

    # ── Sync sell dropdown selection to the main symbol state ──
    @app.callback(
        Output("trade-symbol", "value"),
        Input("trade-symbol-sell", "value"),
        State("trade-side", "value"),
        prevent_initial_call=True,
    )
    def sync_sell_to_symbol(sell_symbol, side):
        if side == "sell" and sell_symbol:
            return sell_symbol
        return dash.no_update

    # ── Auto-fill quantity from position when selling ─────────
    @app.callback(
        Output("trade-qty", "value"),
        Input("trade-symbol-sell", "value"),
        State("trade-side", "value"),
        prevent_initial_call=True,
    )
    def autofill_sell_qty(sell_symbol, side):
        if side != "sell" or not sell_symbol:
            return dash.no_update
        try:
            router = None
            try:
                from src.broker.registry import get_router
                router = get_router()
            except Exception:
                pass
            if router is None:
                return dash.no_update
            for pos in router.get_positions():
                if getattr(pos, "symbol", "") == sell_symbol:
                    return int(getattr(pos, "qty", 0))
        except Exception:
            pass
        return dash.no_update

    @app.callback(
        Output("trade-routing-info", "children"),
        Input("trade-symbol", "value"),
    )
    def update_routing_info(symbol: str | None) -> html.Div:
        if not symbol or len(symbol) < 2:
            return html.Div()

        try:
            from src.broker.broker_router import detect_exchange, detect_asset_type
            exchange = detect_exchange(symbol)
            asset_type = detect_asset_type(symbol)

            return html.Span(
                f"{t('common.exchange')}: {exchange or 'auto'} · {t('common.type')}: {asset_type or 'auto'}",
                style={"color": COLORS["muted"], "fontSize": "0.85rem"},
            )
        except Exception:
            return html.Div(
                html.Span(t('trading.enter_symbol_routing'),
                          style={"color": COLORS["muted"], "fontSize": "0.85rem"})
            )

    @app.callback(
        Output("trade-limit-div", "style"),
        Input("trade-order-type", "value"),
    )
    def toggle_limit_price(order_type: str) -> dict:
        if order_type == "limit":
            return {"display": "block"}
        return {"display": "none"}

    # Preview + button state — runs in browser, no server round-trip
    # Inject translated strings into JS at registration time
    _buy_label = t('common.buy')
    _sell_label = t('common.sell')
    _market_label = t('common.market')
    _limit_label = t('common.limit')
    _type_label = t('common.type')
    app.clientside_callback(
        f"""
        function(buySymbol, sellSymbol, side, qty, orderType, limitPrice) {{
            var previewEl = document.getElementById('trade-preview');
            var btnEl = document.getElementById('trade-submit-btn');
            if (!previewEl || !btnEl) return [window.dash_clientside.no_update, window.dash_clientside.no_update];

            var symbol = (side === 'sell') ? sellSymbol : buySymbol;
            if (!symbol || !qty || qty <= 0) {{
                previewEl.innerHTML = '';
                btnEl.disabled = true;
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }}

            var sideColor = side === 'buy' ? '#2ed573' : '#ff4757';
            var sideText = side === 'buy' ? '{_buy_label}' : '{_sell_label}';
            var typeText = orderType === 'limit' ? '{_limit_label}' : '{_market_label}';
            if (orderType === 'limit' && limitPrice) typeText += ' @ ' + limitPrice;

            previewEl.innerHTML =
                '<div style="background:#0f1117;border:1px solid #2d3748;border-radius:4px;padding:12px">' +
                '<div><strong style="color:' + sideColor + '">' + sideText + ' </strong>' +
                '<span style="color:#e2e8f0">' + Math.round(qty) + ' \u00d7 ' + symbol.toUpperCase() + '</span></div>' +
                '<div style="color:#64748b;font-size:0.85rem">{_type_label}: ' + typeText + '</div>' +
                '</div>';

            var canSubmit = !!(symbol && qty && qty > 0);
            if (orderType === 'limit' && !limitPrice) canSubmit = false;
            btnEl.disabled = !canSubmit;

            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }}
        """,
        [
            Output("trade-preview", "className"),   # dummy
            Output("trade-submit-btn", "className"), # dummy
        ],
        [
            Input("trade-symbol", "value"),
            Input("trade-symbol-sell", "value"),
            Input("trade-side", "value"),
            Input("trade-qty", "value"),
            Input("trade-order-type", "value"),
            Input("trade-limit-price", "value"),
        ],
    )

    @app.callback(
        Output("trade-result", "children"),
        Input("trade-submit-btn", "n_clicks"),
        [
            State("trade-symbol", "value"),
            State("trade-symbol-sell", "value"),
            State("trade-side", "value"),
            State("trade-qty", "value"),
            State("trade-order-type", "value"),
            State("trade-limit-price", "value"),
        ],
        prevent_initial_call=True,
    )
    def submit_trade(_clicks, buy_symbol, sell_symbol, side, qty, order_type, limit_price):
        symbol = sell_symbol if side == "sell" else buy_symbol
        if not symbol or not qty or qty <= 0:
            return no_update

        try:
            from src.broker.base_broker import OrderType
            router = None
            try:
                from src.broker.registry import get_router
                router = get_router()
            except Exception:
                pass
            if router is None:
                raise RuntimeError("No broker available — trader may not be running")

            # Convert string to OrderType enum
            ot = OrderType.LIMIT if order_type == "limit" else OrderType.MARKET

            sym = symbol.upper()
            lp = float(limit_price) if limit_price and order_type == "limit" else None

            if side == "buy":
                order = router.buy(symbol=sym, qty=float(qty), order_type=ot, limit_price=lp)
            else:
                # Check if this is a short position — closing a short requires BUY (cover)
                is_short = False
                try:
                    for pos in router.get_positions():
                        if getattr(pos, "symbol", "") == sym and getattr(pos, "side", "") == "short":
                            is_short = True
                            break
                except Exception:
                    pass

                if is_short:
                    # Cover the short by buying
                    order = router.buy(symbol=sym, qty=float(qty), order_type=ot, limit_price=lp)
                else:
                    order = router.sell(symbol=sym, qty=float(qty), order_type=ot, limit_price=lp)

            return dbc.Alert([
                html.Strong(f"{t('trading.order_placed')} "),
                html.Span(f"{t('common.buy') if side == 'buy' else t('common.sell')} {qty} × {symbol.upper()} ({order_type})"),
                html.Div(f"{t('trading.order_id')}: {getattr(order, 'order_id', 'N/A')}", style={"fontSize": "0.85rem"}),
            ], color="success", duration=15000)

        except Exception as e:
            logger.error(f"Ordre fejl: {e}")
            return dbc.Alert(f"{t('trading.order_failed')}: {e}", color="danger", duration=15000)

    # After a successful trade, navigate to refresh the page
    # so the pre-rendered trades table picks up the new trade.
    # This is a single full navigation, not a periodic blink.
    app.clientside_callback(
        """
        function(resultChildren) {
            if (!resultChildren) return window.dash_clientside.no_update;
            // Wait 2 seconds then soft-reload the trading page
            setTimeout(function() {
                if (window.location.pathname === '/trading') {
                    window.location.reload();
                }
            }, 2000);
            return window.dash_clientside.no_update;
        }
        """,
        Output("trading-refresh-trigger", "data"),
        Input("trade-result", "children"),
    )

    # ── Quick Trade: Buy/Sell Top 10 callbacks ────────────────

    def _get_top_recommendations(side: str, limit: int = 10):
        """Get top buy or sell candidates from cached scanner data."""
        try:
            from src.dashboard.app import _cache
            result = _cache.get("scanner_result")
            if not result:
                return []
            picks = result.top_buys if side == "buy" else result.top_sells
            return picks[:limit]
        except Exception:
            return []

    @app.callback(
        Output("quick-trade-modal", "is_open"),
        Output("quick-trade-modal-title", "children"),
        Output("quick-trade-modal-body", "children"),
        Output("quick-trade-action-store", "data"),
        Output("quick-trade-confirm-btn", "children"),
        Output("quick-trade-confirm-btn", "color"),
        Output("quick-trade-prices-store", "data"),
        Output("quick-trade-cash-store", "data"),
        [
            Input("quick-buy-top10-btn", "n_clicks"),
            Input("quick-sell-top10-btn", "n_clicks"),
            Input("quick-trade-cancel-btn", "n_clicks"),
            Input("quick-trade-confirm-btn", "n_clicks"),
        ],
        State("quick-trade-action-store", "data"),
        prevent_initial_call=True,
    )
    def handle_quick_trade(buy_c, sell_c, cancel_c, confirm_c, pending):
        _nu = no_update
        ctx = dash.callback_context
        if not ctx.triggered:
            return _nu, _nu, _nu, _nu, _nu, _nu, _nu, _nu
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "quick-trade-cancel-btn":
            return False, "", "", None, _nu, _nu, None, None

        if trigger == "quick-trade-confirm-btn" and pending:
            return False, "", "", pending, _nu, _nu, _nu, _nu

        # Buy or Sell button clicked — show confirmation
        if trigger == "quick-buy-top10-btn":
            side = "buy"
            picks = _get_top_recommendations("buy")
            title = t('trading.confirm_buy_title')
            msg_tpl = t('trading.confirm_buy_msg')
            btn_label = t('trading.buy_top_10')
            btn_color = "success"
        elif trigger == "quick-sell-top10-btn":
            side = "sell"
            picks = _get_top_recommendations("sell")
            title = t('trading.confirm_sell_title')
            msg_tpl = t('trading.confirm_sell_msg')
            btn_label = t('trading.sell_top_10')
            btn_color = "danger"
        else:
            return no_update, no_update, no_update, no_update, no_update, no_update

        if not picks:
            return True, title, html.P(t('trading.no_recommendations'), className="text-warning"), None, btn_label, btn_color, None, None

        msg = msg_tpl.replace("{count}", str(len(picks)))

        # Build position list for confirmation
        from config.settings import settings
        pos_pct = settings.risk.max_position_pct

        # Get cash and FX rate for buy side
        cash_dkk = 0.0
        usd_dkk = 6.90
        if side == "buy":
            try:
                from src.broker.paper_broker import PaperBroker
                pb = PaperBroker()
                cash_dkk = pb.get_account().cash
            except Exception:
                pass
            try:
                import yfinance as _yf
                _fx = _yf.Ticker("DKK=X")
                _r = getattr(_fx.fast_info, "last_price", None)
                if _r and _r > 0:
                    usd_dkk = _r
            except Exception:
                pass
            cash_dkk *= usd_dkk

        # Fetch prices and build rows
        rows = []
        total_commit = 0.0
        buy_prices_dkk = []
        for idx, a in enumerate(picks):
            reason = a.reasons[0] if a.reasons else ""
            price_dkk = 0.0
            if side == "buy":
                try:
                    import yfinance as _yf2
                    _tk = _yf2.Ticker(a.symbol)
                    price_dkk = (getattr(_tk.fast_info, "last_price", 0) or 0) * usd_dkk
                except Exception:
                    pass
                buy_prices_dkk.append(round(price_dkk, 2))
                total_commit += price_dkk

            row_cells = [
                html.Td(a.symbol, style={"color": COLORS["accent"], "fontWeight": "bold"}),
                html.Td(f"{a.score:.0f}", style={"color": COLORS["text"]}),
                html.Td(f"{a.change_pct:+.1f}%",
                         style={"color": COLORS["green"] if a.change_pct > 0 else COLORS["red"]}),
                html.Td(reason, style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
            ]
            if side == "buy":
                row_cells.extend([
                    html.Td(f"{price_dkk:,.0f} kr", style={"color": COLORS["text"]}),
                    html.Td(
                        dcc.Input(
                            id={"type": "qt-buy-qty", "index": idx},
                            type="number", min=0, step=1, value=1,
                            style={
                                "width": "60px", "backgroundColor": "#0f1117",
                                "color": "#e2e8f0", "border": "1px solid #2d3748",
                                "borderRadius": "4px", "textAlign": "center",
                                "fontSize": "0.85rem",
                            },
                            debounce=True,
                        ),
                    ),
                    html.Td(
                        html.Span(f"{price_dkk:,.0f} kr",
                                  id={"type": "qt-buy-line", "index": idx},
                                  style={"color": COLORS["accent"]}),
                    ),
                ])
            rows.append(html.Tr(row_cells))

        headers = [
            html.Th(t('common.symbol'), className="text-light"),
            html.Th(t('trading.score'), className="text-light"),
            html.Th(t('common.change'), className="text-light"),
            html.Th(t('common.reason'), className="text-light"),
        ]
        if side == "buy":
            headers.extend([
                html.Th(t('common.price'), className="text-light"),
                html.Th(t('common.quantity'), className="text-light"),
                html.Th("Total", className="text-light"),
            ])

        table = dbc.Table([
            html.Thead(html.Tr(headers)),
            html.Tbody(rows),
        ], bordered=True, hover=True, size="sm", className="table-dark")

        body_children = [
            html.P(msg, className="text-warning fw-bold", style={"fontSize": "1.1rem"}),
            html.P(f"{t('risk.max_position')}: {pos_pct:.0%}",
                   className="text-muted small"),
            table,
        ]

        if side == "buy":
            cash_ok = cash_dkk >= total_commit
            cash_color = COLORS["green"] if cash_ok else COLORS["red"]
            body_children.extend([
                html.Hr(style={"borderColor": COLORS["border"]}),
                dbc.Row([
                    dbc.Col([
                        html.Span(t('portfolio.cash') + ": ", style={"color": COLORS["muted"]}),
                        html.Span(f"{cash_dkk:,.0f} kr",
                                  style={"color": COLORS["text"], "fontWeight": "600"}),
                    ], width="auto"),
                    dbc.Col([
                        html.Span("Total: ", style={"color": COLORS["muted"]}),
                        html.Span(f"{total_commit:,.0f} kr",
                                  id="qt-buy-total",
                                  style={"color": cash_color, "fontWeight": "600"}),
                    ], width="auto"),
                ], justify="between"),
            ])
            body_children.append(
                dbc.Alert(t('trading.insufficient_cash'),
                          color="danger", className="mt-2 mb-0 py-2",
                          id="qt-buy-cash-warn",
                          style={"display": "block" if not cash_ok else "none"}),
            )

        body = html.Div(body_children)

        action = {"side": side, "symbols": [a.symbol for a in picks]}
        prices_data = buy_prices_dkk if side == "buy" else None
        cash_data = round(cash_dkk, 2) if side == "buy" else None
        return True, title, body, action, btn_label, btn_color, prices_data, cash_data

    # ── Recalculate buy totals when qty changes ──
    @app.callback(
        Output({"type": "qt-buy-line", "index": dash.ALL}, "children"),
        Output("qt-buy-total", "children"),
        Output("qt-buy-total", "style"),
        Output("qt-buy-cash-warn", "style"),
        Input({"type": "qt-buy-qty", "index": dash.ALL}, "value"),
        State("quick-trade-prices-store", "data"),
        State("quick-trade-cash-store", "data"),
        prevent_initial_call=True,
    )
    def _recalc_buy_totals(qtys, prices, cash_dkk):
        if not prices or not qtys:
            raise dash.exceptions.PreventUpdate
        lines = []
        grand = 0.0
        for q, p in zip(qtys, prices):
            qty = max(int(q or 0), 0)
            total = qty * p
            grand += total
            lines.append(f"{total:,.0f} kr")
        cash_ok = (cash_dkk or 0) >= grand
        color = COLORS["green"] if cash_ok else COLORS["red"]
        warn_style = {"display": "none"} if cash_ok else {"display": "block"}
        return lines, f"{grand:,.0f} kr", {"color": color, "fontWeight": "600"}, warn_style

    @app.callback(
        Output("quick-trade-result", "children"),
        Input("quick-trade-confirm-btn", "n_clicks"),
        State("quick-trade-action-store", "data"),
        State({"type": "qt-buy-qty", "index": dash.ALL}, "value"),
        prevent_initial_call=True,
    )
    def execute_quick_trade(n_clicks, action, buy_qtys):
        if not n_clicks or not action:
            return no_update

        side = action.get("side")
        symbols = action.get("symbols", [])
        if not symbols:
            return no_update

        try:
            from src.broker.registry import get_router
            from src.broker.base_broker import OrderType
            from config.settings import settings

            router = get_router()
            if not router:
                return dbc.Alert(t('trading.no_broker_connected'), color="warning")

            acc = None
            try:
                from src.broker.paper_broker import PaperBroker
                pb = PaperBroker()
                acc = pb.get_account()
            except Exception:
                pass

            results = []
            done = 0

            # Get FX rate for DKK display
            _usd_dkk = 6.90
            try:
                import yfinance as _yf_exec
                _fx = _yf_exec.Ticker("DKK=X")
                _r = getattr(_fx.fast_info, "last_price", None)
                if _r and _r > 0:
                    _usd_dkk = _r
            except Exception:
                pass

            if side == "buy":
                # Use quantities from the input fields
                for i, sym in enumerate(symbols):
                    qty = max(int((buy_qtys[i] if buy_qtys and i < len(buy_qtys) else 1) or 0), 0)
                    if qty == 0:
                        continue
                    try:
                        from src.data.market_data import MarketDataFetcher
                        mdf = MarketDataFetcher()
                        df = mdf.get_historical(sym, interval="1d", lookback_days=5)
                        if df.empty:
                            results.append(html.Li(f"\u2717 {sym}: no price data",
                                                   style={"color": COLORS["red"]}))
                            continue
                        price = float(df["Close"].iloc[-1])
                        price_dkk = price * _usd_dkk
                        order = router.buy(symbol=sym, qty=qty, order_type=OrderType.MARKET)
                        results.append(html.Li(
                            f"\u2713 {t('common.buy')} {qty} \u00d7 {sym} @ {price_dkk:,.0f} kr",
                            style={"color": COLORS["text"]}))
                        done += 1
                    except Exception as exc:
                        results.append(html.Li(f"\u2717 {sym}: {exc}",
                                               style={"color": COLORS["red"]}))

            elif side == "sell":
                # Sell positions we hold
                positions = list(router.get_positions())
                pos_map = {getattr(p, "symbol", ""): p for p in positions}

                for sym in symbols:
                    pos = pos_map.get(sym)
                    if not pos:
                        results.append(html.Li(f"\u2014 {sym}: no position held",
                                               style={"color": COLORS["muted"]}))
                        continue
                    try:
                        qty = getattr(pos, "qty", 0)
                        if qty <= 0:
                            continue
                        router.sell(symbol=sym, qty=qty, order_type=OrderType.MARKET)
                        results.append(html.Li(
                            f"\u2713 {t('common.sell')} {qty:.2f} \u00d7 {sym}",
                            style={"color": COLORS["text"]}))
                        done += 1
                    except Exception as exc:
                        results.append(html.Li(f"\u2717 {sym}: {exc}",
                                               style={"color": COLORS["red"]}))

            msg = t('trading.orders_done').replace("{count}", str(done))
            return dbc.Alert([
                html.Strong(msg, style={"color": COLORS["text"]}),
                html.Ul(results, className="list-unstyled mt-2 mb-0"),
            ], color="dark", style={"border": f"1px solid {COLORS['border']}",
                                    "backgroundColor": COLORS["card"]},
               dismissable=True)

        except Exception as exc:
            logger.error(f"Quick trade failed: {exc}")
            return dbc.Alert(f"{t('trading.orders_failed')}: {exc}", color="danger", dismissable=True)
