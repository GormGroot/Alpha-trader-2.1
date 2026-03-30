#!/usr/bin/env python3
"""Generate PDF changelog of all modifications to Alpha Trading Platform since GitHub download."""

from fpdf import FPDF
from datetime import datetime


class ChangelogPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 160, 120)
        self.cell(0, 8, "Alpha Trading Platform - Changelog", align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 160, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 40)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 160, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 70)
        self.ln(2)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def bullet(self, text, indent=5):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4.5, " " * indent + "- " + text)

    def file_entry(self, path, lines, desc):
        self.set_font("Courier", "", 7.5)
        self.set_text_color(0, 100, 80)
        self.cell(90, 4.5, path, new_x="RIGHT")
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(100, 100, 100)
        self.cell(18, 4.5, f"{lines} lines", new_x="RIGHT")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.5, desc, new_x="LMARGIN", new_y="NEXT")

    def table_row(self, cols, widths, bold=False, bg=False):
        if bg:
            self.set_fill_color(240, 248, 245)
        self.set_font("Helvetica", "B" if bold else "", 8)
        self.set_text_color(30, 30, 40)
        for i, (col, w) in enumerate(zip(cols, widths)):
            self.cell(w, 5.5, col, border=0, fill=bg, align="L" if i == 0 else "C")
        self.ln()

    def stat_box(self, label, value):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(40, 5, label)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(0, 130, 100)
        self.cell(40, 5, str(value), new_x="LMARGIN", new_y="NEXT")


def build_pdf():
    pdf = ChangelogPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── TITLE PAGE ──
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(0, 160, 120)
    pdf.cell(0, 15, "Alpha Trading Platform", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Changelog & Modification Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_draw_color(0, 160, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    info = [
        ("Platform:", "Rock 5B (RK3588, aarch64, Debian 12)"),
        ("Original source:", "GitHub download, v1.0.0 (March 19, 2026)"),
        ("Development period:", "March 22 - March 28, 2026 (9 days)"),
        ("Total Python files:", "108 files, ~58,000 lines"),
        ("New files added:", "38 files (30 Python + 8 language)"),
        ("Core files modified:", "7 files (with .bak backups)"),
        ("Custom code added:", "~15,000 lines across all additions"),
        ("Report date:", datetime.now().strftime("%B %d, %Y")),
    ]
    for label, value in info:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 6, label)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    # ── PHASE 1 ──
    pdf.add_page()
    pdf.section_title("Phase 1: Hardware Optimization & Market Operations (March 22)")

    pdf.subsection("NPU Accelerator - RK3588 Neural Processing Unit Integration")
    pdf.file_entry("src/ops/npu_accelerator.py", 708, "NEW - RKNN2 NPU acceleration for ML inference")
    pdf.file_entry("setup_npu.py", 176, "NEW - NPU setup and validation script")
    pdf.body_text(
        "Added hardware-specific optimization for the Rock 5B's RK3588 NPU (Neural Processing Unit). "
        "The npu_accelerator module provides RKNN2 model conversion and inference for the ML trading "
        "strategies, offloading gradient boosting and ensemble predictions from the CPU to the 6 TOPS NPU. "
        "Includes model quantization (INT8/FP16), batch inference pipeline, fallback to CPU when NPU is "
        "unavailable, and performance benchmarking utilities."
    )

    pdf.subsection("24/7 Market Operations Framework")
    pdf.file_entry("src/ops/daily_scheduler.py", 635, "NEW - CET-based daily task scheduler")
    pdf.file_entry("src/ops/market_calendar.py", 437, "NEW - Global market hours and holidays")
    pdf.file_entry("src/ops/market_handoff.py", 183, "NEW - Cross-market session handoff")
    pdf.file_entry("src/sentiment/sentiment_analyzer.py", 405, "NEW - Multi-source sentiment scoring")
    pdf.file_entry("config/default_config.yaml", 486, "NEW - Comprehensive YAML configuration")
    pdf.body_text(
        "Built a complete 24/7 operations framework with CET timezone scheduling. The DailyScheduler "
        "runs 6 scheduled tasks: morning check (07:30), EU open (09:00), US open (15:30), EU close "
        "(17:30), US close (22:00), and maintenance (23:00). Includes weekend/holiday detection for "
        "DK + US markets using Easter-based calculations. MarketCalendar provides real-time open/close "
        "status for 14 global exchanges. MarketHandoff manages session transitions between European "
        "and US trading hours with position size adjustments."
    )

    # ── PHASE 2 ──
    pdf.add_page()
    pdf.section_title("Phase 2: Core Trading Engine Upgrades (March 23-24)")

    pdf.subsection("Portfolio Persistence - SQLite Database Layer")
    pdf.file_entry("src/risk/portfolio_tracker.py", 568, "MODIFIED (+234 lines, was 334)")
    pdf.body_text(
        "Major upgrade: added complete SQLite persistence layer (PortfolioDB class). All positions, "
        "closed trades, equity history, and cash balance now survive process restarts. Previously, "
        "restarting the trader reset everything to the initial $100K capital. New features:\n"
        "  - Automatic DB schema creation (4 tables: portfolio_state, open_positions, closed_trades, equity_history)\n"
        "  - Load-from-DB on startup instead of fresh initialization\n"
        "  - Atomic state saves on every trade and price update\n"
        "  - Short position support with proper P&L tracking\n"
        "  - Peak price tracking per position for trailing stops"
    )

    pdf.subsection("Multi-Broker Aggregated Portfolio")
    pdf.file_entry("src/broker/aggregated_portfolio.py", 538, "NEW - Unified cross-broker portfolio view")
    pdf.body_text(
        "New module that aggregates positions across all connected brokers (Alpaca, Saxo, IBKR, Nordnet, "
        "Paper) into a single view. Features: DKK currency conversion with yfinance FX rates (fallback "
        "to static rates), per-broker/per-currency/per-asset-type breakdowns, combined trade history "
        "for tax reporting, and automatic currency detection based on exchange suffix."
    )

    pdf.subsection("Bug Fix: Duplicate ContinuousLearner Instance")
    pdf.file_entry("main.py", 574, "MODIFIED - Fixed memory leak")
    pdf.file_entry("patch_main.py", 140, "NEW - Patch script for main.py")
    pdf.body_text(
        "Critical bug fix: main.py was creating a standalone ContinuousLearner instance at startup, "
        "while AutoTrader.__init__ also created its own instance. This resulted in two competing "
        "learner threads analyzing the same trade database simultaneously, doubling memory usage and "
        "causing SQLite lock contention. Fix: removed the standalone instance from main.py, relying "
        "solely on AutoTrader's internal learner. The patch_main.py script automates this fix."
    )

    # ── PHASE 3 ──
    pdf.add_page()
    pdf.section_title("Phase 3: Dashboard & Internationalization (March 26)")

    pdf.subsection("Dashboard Page System - Multi-Broker Trading UI")
    pdf.file_entry("src/dashboard/pages/portfolio.py", 823, "NEW - Real-time portfolio dashboard")
    pdf.file_entry("src/dashboard/pages/market_explorer.py", 557, "NEW - Global market overview")
    pdf.file_entry("src/dashboard/pages/tax_center.py", 453, "NEW - Danish corporate tax dashboard")
    pdf.file_entry("src/dashboard/pages/broker_status.py", 224, "NEW - Broker connection monitor")
    pdf.file_entry("src/dashboard/pages/trading.py", 484, "NEW (Phase 4) - Order execution UI")
    pdf.body_text(
        "Completely new multi-page dashboard system replacing the monolithic single-page design. "
        "Each page is a self-contained module with its own layout function and Dash callbacks:\n"
        "  - Portfolio: live positions table, equity curve, asset allocation pie charts, P&L breakdown\n"
        "  - Market Explorer: 30s auto-refresh, global market heatmap, sector rotation tracking\n"
        "  - Tax Center: Danish corporate tax (27%/42% tiers), mark-to-market, loss harvesting suggestions\n"
        "  - Broker Status: 15s heartbeat checks, connection latency graphs, order queue depth\n"
        "  - Trading: live order entry with risk pre-checks, position sizing calculator"
    )

    pdf.subsection("Internationalization (i18n) - 8 Languages")
    pdf.file_entry("src/dashboard/i18n.py", 116, "NEW - Translation lookup engine")
    pdf.file_entry("lang/languages.json", 12, "NEW - Language registry")
    pdf.body_text(
        "Full internationalization system with 8 languages: English, Dansk (Danish), Deutsch, "
        "Espanol, Francais, Portugues, Vlaams (Flemish). Each language file (~530 lines) covers "
        "all dashboard labels, tooltips, strategy names, risk metrics, and tax terminology. "
        "Language selection persists in browser via dcc.Store."
    )
    langs = ["eng.lan / english.lan (553 lines)", "dansk.lan (553 lines)",
             "deutsch.lan (530 lines)", "espanol.lan (530 lines)",
             "francais.lan (530 lines)", "portugues.lan (530 lines)", "vlaams.lan (530 lines)"]
    for l in langs:
        pdf.bullet(l)

    pdf.subsection("Operations: Backup System")
    pdf.file_entry("src/ops/backup.py", 470, "NEW - Automated backup manager")
    pdf.body_text(
        "Full backup system with daily automated config archival (tar.gz), database backup, "
        "retention policy management, and disk space monitoring. Integrated into DailyScheduler's "
        "23:00 maintenance window. Backup history stored in ~/backups/config/."
    )

    pdf.subsection("Broker Registry")
    pdf.file_entry("src/broker/registry.py", 40, "NEW - Global AutoTrader accessor")
    pdf.body_text(
        "Simple registry module providing get_auto_trader()/set_auto_trader() functions, allowing "
        "dashboard settings pages to access and modify the running AutoTrader instance (toggle "
        "crypto trading, pattern strategy, risk parameters) without circular imports."
    )

    # ── PHASE 4 ──
    pdf.add_page()
    pdf.section_title("Phase 4: Core System Overhaul (March 27) - Busiest Day")

    pdf.subsection("AutoTrader - Major Feature Expansion")
    pdf.file_entry("src/trader/auto_trader.py", 1081, "MODIFIED (+317 lines, was 764)")
    pdf.body_text(
        "Largest single modification. Key additions:\n"
        "  - Pattern strategy toggle (enable/disable from dashboard)\n"
        "  - Crypto trading toggle (filter *-USD symbols dynamically)\n"
        "  - Per-exchange stop-loss system (config/exchange_stop_loss.json) with 16 exchange suffixes\n"
        "  - ContinuousLearner integration (moved from main.py, single instance)\n"
        "  - Persistent risk parameter loading from JSON configs at startup\n"
        "  - Adaptive thresholds adjusted by feedback loop every 20 scans\n"
        "  - Memory management: gc.collect() every 100 scans, explicit DataFrame cleanup\n"
        "  - Database pruning: scans kept 7 days, trades kept 30 days"
    )

    pdf.subsection("Signal Engine - Strategy Orchestration")
    pdf.file_entry("src/strategy/signal_engine.py", 431, "NEW - Parallel strategy execution")
    pdf.body_text(
        "New strategy orchestration layer that runs all strategies in parallel via ThreadPoolExecutor "
        "and aggregates signals with configurable consensus thresholds. Replaces the previous sequential "
        "strategy evaluation. Features: signal deduplication, confidence-weighted aggregation, "
        "SQLite signal history with configurable retention, and per-strategy performance tracking."
    )

    pdf.subsection("Pattern Strategy - Technical Pattern Recognition")
    pdf.file_entry("src/strategy/pattern_strategy.py", 209, "NEW - Background pattern scanner")
    pdf.body_text(
        "New strategy module that runs a background thread scanning for technical chart patterns "
        "(head-and-shoulders, double tops/bottoms, triangles, wedges, channels). Uses TTL-based "
        "caching (600s) with automatic eviction. Integrates with the existing patterns.py detection "
        "library and feeds signals into SignalEngine."
    )

    pdf.subsection("Continuous Learner - Adaptive ML Feedback Loop")
    pdf.file_entry("src/learning/continuous_learner.py", 1093, "NEW - Largest new module")
    pdf.body_text(
        "Comprehensive adaptive learning system (1,093 lines) that runs every 5 minutes:\n"
        "  - Analyzes recent trade outcomes and correlates with strategy signals\n"
        "  - Detects strategy drift using windowed performance metrics (deque, maxlen=50)\n"
        "  - Tracks per-model performance with hard cap at 50 models\n"
        "  - Adjusts AutoTrader confidence thresholds based on recent win rates\n"
        "  - SQLite persistence with 90-day trade retention, 30-day metric retention\n"
        "  - Automatic database pruning every 50 cycles\n"
        "  - Daemon thread with proper start/stop lifecycle"
    )

    pdf.subsection("Market Data & Broker Infrastructure")
    pdf.file_entry("src/data/market_data.py", 395, "NEW - Enhanced data fetcher")
    pdf.file_entry("src/broker/connection_manager.py", 350, "NEW - Broker health monitor")
    pdf.file_entry("src/broker/paper_broker.py", 339, "NEW - Paper trading simulator")
    pdf.body_text(
        "MarketDataFetcher: reusable HTTP session, explicit yfinance cache clearing after each fetch "
        "(_DFS.clear(), _ERRORS.clear()) to prevent memory accumulation. ConnectionManager: background "
        "heartbeat thread with sliding-window response time tracking (last 100 per broker). "
        "PaperBroker: full order lifecycle (market + limit), short selling support, "
        "pending order processing with price monitoring."
    )

    pdf.subsection("Risk Manager - Persistent Configuration")
    pdf.file_entry("src/risk/risk_manager.py", 366, "MODIFIED (+35 lines, was 331)")
    pdf.body_text(
        "Added JSON-based persistent configuration loading for three risk parameters:\n"
        "  - config/max_positions.json: maximum open positions override\n"
        "  - config/global_stop_loss.json: global stop-loss percentage\n"
        "  - config/risk_sizing.json: maximum position size percentage\n"
        "Also added exit bypass logic: selling an existing position is approved without risk checks "
        "(previously, exits could be blocked by the same limits that govern entries)."
    )

    pdf.subsection("Performance Monitoring & Reporting")
    pdf.file_entry("src/monitoring/performance_tracker.py", 587, "MODIFIED (+260 lines)")
    pdf.file_entry("src/dashboard/pages/performance_report.py", 1030, "NEW - PDF report generator")
    pdf.body_text(
        "PerformanceTracker expanded with per-strategy trade tracking, daily snapshots, benchmark "
        "comparison (S&P 500), and bounded storage (max 50,000 trades, 10,000 snapshots with "
        "automatic pruning). New performance_report.py generates comprehensive PDF reports with "
        "portfolio charts, strategy comparison tables, risk metrics, and trade logs."
    )

    pdf.subsection("Dashboard App - Massive Expansion")
    pdf.file_entry("src/dashboard/app.py", 6642, "MODIFIED (from ~2,000 to 6,642 lines)")
    pdf.body_text(
        "The dashboard grew from a basic 4-page template to a comprehensive 20-page trading terminal:\n"
        "  - Original pages expanded: Overblik, Aktieanalyse, Strategier, Risiko\n"
        "  - New analysis pages: Skat, Markedsoverblik, Sentiment, Kalender, Regime, Stress Test\n"
        "  - New data pages: System Health, Smart Money, Options Flow, Alt Data, Teknisk Analyse, Krypto\n"
        "  - Advanced indicators: Ichimoku Cloud, Fibonacci, Keltner/Donchian Channels, Volume Profile\n"
        "  - Auto-refresh intervals: 15s (broker status), 30s (portfolio, markets), 60s (main, tax)\n"
        "  - Dark theme with accent color scheme, responsive Bootstrap layout"
    )

    pdf.subsection("Runtime Configuration Files")
    pdf.file_entry("config/max_positions.json", 2, "NEW - Max open positions setting")
    pdf.file_entry("config/global_stop_loss.json", 2, "NEW - Global stop-loss percentage")
    pdf.file_entry("config/exchange_stop_loss.json", 5, "NEW - Per-exchange stop-loss overrides")
    pdf.file_entry("config/risk_sizing.json", 3, "NEW - Position sizing override")
    pdf.body_text(
        "Four new JSON config files allow the dashboard settings page to persist risk parameter "
        "changes across restarts. AutoTrader and RiskManager load these at startup, providing "
        "a UI-driven configuration layer on top of the Pydantic settings."
    )

    # ── PHASE 5 ──
    pdf.add_page()
    pdf.section_title("Phase 5: Diagnostics (March 28)")
    pdf.file_entry("diagnostics/capture_dump.sh", 90, "NEW - System diagnostic capture script")
    pdf.file_entry("diagnostics/dump_20260328_041110.json", 260, "NEW - Pre-crash system state dump")
    pdf.body_text(
        "Diagnostic tools added after investigating a system crash (March 28, 04:16 UTC). "
        "The crash was traced to Chromium's GPU process failing when rendering the Plotly dashboard "
        "on the RK3588's Mali GPU (DMA-BUF export failures, stack smashing in GPU subprocess). "
        "Not caused by the trading platform itself."
    )

    # ── BUG FIXES SUMMARY ──
    pdf.add_page()
    pdf.section_title("Bug Fixes Summary")

    bugs = [
        ("Memory Leak: Duplicate ContinuousLearner",
         "main.py + auto_trader.py",
         "main.py created a standalone ContinuousLearner, while AutoTrader also created one internally. "
         "Two competing threads analyzing the same DB, doubling memory and causing SQLite lock contention. "
         "Fixed by removing the standalone instance from main.py."),
        ("Exit Orders Blocked by Risk Manager",
         "src/risk/risk_manager.py",
         "Selling an existing position was subject to the same risk checks as opening new positions. "
         "This could block legitimate exits (e.g., stop-loss triggered but max daily loss already hit). "
         "Fixed with exit bypass logic that auto-approves sells of existing positions."),
        ("Portfolio Reset on Restart",
         "src/risk/portfolio_tracker.py",
         "Restarting the trading process reset all positions to initial capital ($100K), losing "
         "all trade history and open positions. Fixed by adding SQLite persistence layer that "
         "saves and restores complete portfolio state."),
        ("yfinance Memory Accumulation",
         "src/data/market_data.py + src/dashboard/app.py",
         "yfinance's internal _DFS and _ERRORS dictionaries accumulated cached responses "
         "indefinitely during long-running sessions. Fixed by explicitly clearing these caches "
         "after each data fetch and on dashboard cache eviction."),
        ("Dashboard Cache Unbounded Growth",
         "src/dashboard/app.py",
         "The global stock data cache grew without limit as users browsed different symbols. "
         "Fixed with TTL-based eviction (1200s), max size cap (20 entries), and gc.collect() "
         "on eviction."),
        ("AutoTrader Memory Pressure",
         "src/trader/auto_trader.py",
         "Large DataFrames from market data fetches were retained in scope across scan cycles. "
         "Fixed with explicit `del data; gc.collect()` after each scan, plus periodic cleanup "
         "every 100 scans including database pruning."),
        ("Stale _last_trade Entries",
         "src/trader/auto_trader.py",
         "The _last_trade dict (tracking cooldown per symbol) grew unbounded as symbols rotated "
         "in/out of the watchlist. Fixed with cleanup every 100 scans, removing entries older "
         "than 24 hours."),
    ]

    for title, files, desc in bugs:
        pdf.subsection(title)
        pdf.set_font("Courier", "", 7.5)
        pdf.set_text_color(0, 100, 80)
        pdf.cell(0, 4, f"  Files: {files}", new_x="LMARGIN", new_y="NEXT")
        pdf.body_text(desc)

    # ── FILE INVENTORY ──
    pdf.add_page()
    pdf.section_title("Complete File Inventory - All Changes")

    pdf.subsection("New Files (37 total)")
    new_files = [
        ("Mar 22", "src/ops/npu_accelerator.py", "708", "RK3588 NPU acceleration"),
        ("Mar 22", "setup_npu.py", "176", "NPU setup script"),
        ("Mar 22", "src/ops/daily_scheduler.py", "635", "24/7 task scheduler"),
        ("Mar 22", "src/ops/market_calendar.py", "437", "Global market hours"),
        ("Mar 22", "src/ops/market_handoff.py", "183", "Session handoff"),
        ("Mar 22", "src/sentiment/sentiment_analyzer.py", "405", "Sentiment scoring"),
        ("Mar 22", "config/default_config.yaml", "486", "Full configuration"),
        ("Mar 23", "patch_main.py", "140", "Main.py patch script"),
        ("Mar 24", "src/broker/aggregated_portfolio.py", "538", "Multi-broker aggregation"),
        ("Mar 26", "src/dashboard/pages/portfolio.py", "823", "Portfolio dashboard"),
        ("Mar 26", "src/dashboard/pages/market_explorer.py", "557", "Market explorer"),
        ("Mar 26", "src/dashboard/pages/tax_center.py", "453", "Tax center"),
        ("Mar 26", "src/dashboard/pages/broker_status.py", "224", "Broker status"),
        ("Mar 26", "src/dashboard/i18n.py", "116", "Translation engine"),
        ("Mar 26", "src/ops/backup.py", "470", "Backup manager"),
        ("Mar 26", "src/broker/registry.py", "40", "AutoTrader registry"),
        ("Mar 26", "lang/languages.json", "12", "Language config"),
        ("Mar 26", "lang/deutsch.lan", "530", "German translations"),
        ("Mar 26", "lang/espanol.lan", "530", "Spanish translations"),
        ("Mar 26", "lang/francais.lan", "530", "French translations"),
        ("Mar 26", "lang/portugues.lan", "530", "Portuguese translations"),
        ("Mar 26", "lang/vlaams.lan", "530", "Flemish translations"),
        ("Mar 27", "src/trader/auto_trader.py", "--", "Modified (see below)"),
        ("Mar 27", "src/strategy/signal_engine.py", "431", "Signal orchestration"),
        ("Mar 27", "src/strategy/pattern_strategy.py", "209", "Pattern scanner"),
        ("Mar 27", "src/learning/continuous_learner.py", "1093", "Adaptive ML feedback"),
        ("Mar 27", "src/data/market_data.py", "395", "Enhanced data fetcher"),
        ("Mar 27", "src/broker/connection_manager.py", "350", "Connection health"),
        ("Mar 27", "src/broker/paper_broker.py", "339", "Paper trading sim"),
        ("Mar 27", "src/dashboard/pages/trading.py", "484", "Trading UI"),
        ("Mar 27", "src/dashboard/pages/performance_report.py", "1030", "PDF report gen"),
        ("Mar 27", "lang/dansk.lan", "553", "Danish translations"),
        ("Mar 27", "lang/eng.lan", "553", "English translations"),
        ("Mar 27", "config/max_positions.json", "2", "Max positions config"),
        ("Mar 27", "config/global_stop_loss.json", "2", "Global stop-loss config"),
        ("Mar 27", "config/exchange_stop_loss.json", "5", "Exchange stop-loss config"),
        ("Mar 27", "config/risk_sizing.json", "3", "Position sizing config"),
    ]

    widths = [18, 85, 15, 62]
    pdf.table_row(["Date", "File", "Lines", "Description"], widths, bold=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    for i, (date, path, lines, desc) in enumerate(new_files):
        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.subsection("New Files (continued)")
            pdf.table_row(["Date", "File", "Lines", "Description"], widths, bold=True)
        pdf.table_row([date, path, lines, desc], widths, bg=(i % 2 == 0))

    pdf.ln(5)
    pdf.subsection("Modified Files (4 total, backups preserved)")
    mods = [
        ["auto_trader.py", "764", "1081", "+317", "Pattern strategy, crypto toggle, exchange stop-loss, feedback loop"],
        ["portfolio_tracker.py", "334", "568", "+234", "SQLite persistence, short support, peak tracking"],
        ["risk_manager.py", "331", "366", "+35", "Persistent config loading, exit bypass"],
        ["main.py", "576", "574", "-2", "Removed duplicate ContinuousLearner, added registry"],
    ]
    mwidths = [38, 15, 15, 12, 100]
    pdf.table_row(["File", "Before", "After", "Delta", "Key Changes"], mwidths, bold=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    for i, row in enumerate(mods):
        pdf.table_row(row, mwidths, bg=(i % 2 == 0))

    # ── ARCHITECTURE ──
    pdf.add_page()
    pdf.section_title("Architecture Impact")
    pdf.body_text(
        "The modifications transform the Alpha Trading Platform from a single-broker research tool "
        "into a production-grade multi-broker trading system optimized for the Rock 5B hardware. "
        "The key architectural changes are:"
    )

    changes = [
        "Stateless to Stateful: SQLite persistence across all major subsystems (portfolio, signals, "
        "learning, trades, scans). The trader survives restarts without data loss.",

        "Single-broker to Multi-broker: BrokerRouter + AggregatedPortfolio + ConnectionManager "
        "enable simultaneous trading across Alpaca, Saxo, IBKR, Nordnet, and Paper brokers.",

        "Static to Adaptive: ContinuousLearner creates a feedback loop where trade outcomes "
        "adjust strategy confidence thresholds every 5 minutes.",

        "Basic to Production Dashboard: 20-page Dash/Plotly UI with auto-refresh intervals, "
        "live trading controls, 8-language support, and PDF report generation.",

        "CPU-only to NPU-accelerated: RKNN2 integration offloads ML inference to the RK3588's "
        "6 TOPS neural processing unit.",

        "Manual to Automated Operations: 24/7 scheduler with market-aware task execution, "
        "automated backups, log archival, and cross-market session handoffs.",

        "Memory-unsafe to Memory-managed: Explicit cache eviction, gc.collect() cycles, "
        "database pruning, yfinance cache clearing, and bounded data structures throughout.",
    ]
    for c in changes:
        pdf.bullet(c, indent=5)
        pdf.ln(1)

    # ── PHASE 6 ──
    pdf.add_page()
    pdf.section_title("Phase 6: Critical Integration Fixes & Feature Wiring (March 28 PM)")
    pdf.body_text(
        "Major session focused on discovering and fixing disconnected subsystems. "
        "Multiple modules were fully implemented but silently failing due to wrong class names "
        "in import statements, or never wired into the main trading engine. All fixes verified "
        "against actual class definitions in the source modules."
    )

    pdf.subsection("1. Portfolio Dashboard - Short Position Column")
    pdf.file_entry("src/dashboard/pages/portfolio.py", 3, "MODIFIED - Added 'Side' column")
    pdf.body_text(
        "Added a 'Side' column to the positions table showing Long (green) or Short (red) based "
        "on whether qty < 0. Also added explanatory subtitles to the 'Total Portefolje' and 'Cash' "
        "KPI cards clarifying the difference (equity = cash + positions incl. unrealized P&L)."
    )

    pdf.subsection("2. IBKR Data Feed Settings")
    pdf.file_entry("src/dashboard/app.py", 80, "MODIFIED - New IBKR settings card")
    pdf.file_entry("config/ibkr_datafeed.json", 5, "NEW - IBKR connection config")
    pdf.body_text(
        "Added IBKR Data Feed card to the Settings page with toggle, host/port/client_id inputs, "
        "and save button. Persists to config/ibkr_datafeed.json. On enable, tests connection to "
        "TWS/IB Gateway (creates temporary asyncio event loop for worker thread compatibility). "
        "Shows port guide: 4001=live, 4002=paper (Gateway) / 7496=live, 7497=paper (TWS)."
    )

    pdf.subsection("3. Advanced Feedback Loop")
    pdf.file_entry("src/trader/auto_trader.py", 120, "MODIFIED - _apply_advanced_feedback()")
    pdf.file_entry("src/dashboard/app.py", 50, "MODIFIED - Toggle in settings")
    pdf.file_entry("config/advanced_feedback.json", 2, "NEW - Enable/disable config")
    pdf.body_text(
        "New _apply_advanced_feedback() method auto-applies the 6 recommendation types from "
        "the performance report that were previously informational only:\n"
        "  1. Per-exchange stop-loss tightening when avg P&L < -2% (relaxes on recovery > +3%)\n"
        "  2. Drawdown-based exposure reduction: >5% = 75%, >10% = 50% (restores at <2%)\n"
        "  3. Sharpe-based global stop-loss: <0.5 tightens by 1%, >1.5 relaxes\n"
        "  4. Benchmark underperformance: trailing SPY >2% reduces position sizing by 20%\n"
        "  5. Critical drawdown alert logging for manual review\n"
        "  6. All adjustments persisted to config JSON and applied to running risk managers\n"
        "Dashboard toggle allows user to enable/disable. Runs every 20 scans (~20 min)."
    )

    pdf.subsection("4. Smart Money / Insider Trading - Bug Fix & Integration")
    pdf.file_entry("src/trader/intelligence/alpha_score.py", 15, "MODIFIED - Fixed insider scoring")
    pdf.file_entry("src/trader/auto_trader.py", 12, "MODIFIED - Confidence adjustment")
    pdf.body_text(
        "BUG FIX: alpha_score.py line 701 called insider.get_recent() which does not exist on "
        "InsiderTracker. Silently failed (except Exception: pass), so insider data NEVER contributed "
        "to alpha scores. Fixed: now calls get_insider_sentiment(symbol, lookback_days=90) and maps "
        "the -100/+100 sentiment score to the 10-90 alpha score range. Cluster buying (+10) and "
        "C-suite buying (+5) boost applied.\n\n"
        "NEW: Wired get_confidence_adjustment() into auto_trader trade loop. Each signal now gets "
        "-15 to +15 confidence adjustment from smart money data. InsiderTracker lazily instantiated "
        "and cached on AutoTrader instance."
    )

    pdf.subsection("5. Options Flow - Bug Fix & Integration")
    pdf.file_entry("src/trader/intelligence/alpha_score.py", 12, "MODIFIED - Fixed options scoring")
    pdf.file_entry("src/trader/auto_trader.py", 12, "MODIFIED - Confidence adjustment")
    pdf.body_text(
        "BUG FIX: alpha_score.py imported OptionsFlowAnalyzer which does not exist. Actual class: "
        "OptionsFlowTracker. Also called options.analyze() which does not exist. Actual method: "
        "get_put_call_ratio(). Silently failed, so options flow NEVER contributed to alpha scores.\n\n"
        "Fixed: uses OptionsFlowTracker.get_put_call_ratio() for alpha scoring (put/call ratio "
        "interpretation). NEW: Wired get_confidence_adjustment() into trade loop (-10 to +10)."
    )

    pdf.subsection("6. Alternative Data (Google Trends) - Bug Fix & Integration")
    pdf.file_entry("src/trader/intelligence/alpha_score.py", 12, "MODIFIED - Fixed alt data scoring")
    pdf.file_entry("src/trader/auto_trader.py", 12, "MODIFIED - Confidence adjustment")
    pdf.body_text(
        "BUG FIX: alpha_score.py imported AlternativeDataFetcher which does not exist. Actual class: "
        "AlternativeDataTracker. Also treated get_google_trends() result as DataFrame - it returns "
        "GoogleTrendsResult with a .score property. Silently failed.\n\n"
        "Fixed: uses AlternativeDataTracker.get_google_trends() with proper .score/.trend_direction "
        "access. NEW: Wired get_confidence_adjustment() into trade loop (-10 to +10)."
    )

    pdf.subsection("7. Market Regime - Full Integration into Trading Engine")
    pdf.file_entry("src/trader/auto_trader.py", 50, "MODIFIED - Regime detection in scan loop")
    pdf.file_entry("src/trader/intelligence/news_pipeline.py", 12, "MODIFIED - Fixed regime detection")
    pdf.body_text(
        "RegimeDetector and AdaptiveStrategy were fully implemented (1,095 lines) but NEVER called "
        "from auto_trader.py. The entire regime system was dead code for trading.\n\n"
        "Fixed: Added step 1c in scan_and_trade() between data fetch and signal generation:\n"
        "  - Detects regime from SPY data (falls back to any 50+ bar symbol)\n"
        "  - Applies max_exposure cap: CRASH=10%, BEAR=30%, SIDEWAYS=50%, BULL=100%\n"
        "  - Applies stop-loss multiplier: CRASH=0.4x, BEAR=0.6x, BULL=1.5x\n"
        "  - BLOCKS all BUY signals in CRASH regime (only exits and shorts)\n"
        "  - Feeds regime into DynamicRiskManager for parameter adaptation\n"
        "  - Logs regime changes with confidence and score\n\n"
        "Also fixed news_pipeline.py: was importing RegimeDetector but ignoring it, hardcoding "
        "regime from sentiment score. Now actually runs detector on 6 months of SPY data."
    )

    pdf.subsection("8. Sentiment Event Detection - Bug Fix")
    pdf.file_entry("src/trader/intelligence/alpha_score.py", 3, "MODIFIED - Fixed method call")
    pdf.file_entry("src/trader/intelligence/news_pipeline.py", 3, "MODIFIED - Fixed method call")
    pdf.body_text(
        "BUG FIX: Both files called EventDetector.detect_events(articles) passing a list of "
        "NewsArticle objects. But detect_events() expects a plain text string and returns "
        "list[tuple], not list[DetectedEvent]. The code then accessed .sentiment, .impact, "
        ".event_type attributes that don't exist on tuples. Silently failed.\n\n"
        "Fixed: both now call detect_from_articles(articles) which accepts list[NewsArticle] "
        "and returns list[DetectedEvent] with proper attributes."
    )

    pdf.subsection("9. Risk Management - Major Reconnection")
    pdf.file_entry("main.py", 15, "MODIFIED - DynamicRiskManager instantiation")
    pdf.file_entry("src/trader/auto_trader.py", 60, "MODIFIED - Portfolio sync + circuit breakers")
    pdf.body_text(
        "CRITICAL: DynamicRiskManager (with circuit breakers, regime-adaptive risk parameters) was "
        "fully implemented but NEVER instantiated. Circuit breaker thresholds (3% daily, 7% weekly, "
        "15% drawdown) were completely dead code. Also, PortfolioTracker was disconnected - trades "
        "executed via router but never synced to the risk manager's portfolio, so P&L, drawdown, "
        "and daily loss calculations were always zero.\n\n"
        "Fixes applied:\n"
        "  1. main.py: DynamicRiskManager created alongside RiskManager, stored as _dynamic_risk_manager\n"
        "  2. auto_trader: Circuit breaker check added before every scan - halts on threshold breach\n"
        "  3. auto_trader: Position prices synced to risk manager portfolio every scan cycle\n"
        "  4. auto_trader: Broker positions synced to risk tracker (fills gaps)\n"
        "  5. auto_trader: Trade executions now call portfolio.open_position() / close_position()\n"
        "  6. auto_trader: Regime result fed into DynamicRiskManager.update_regime()"
    )

    pdf.subsection("10. ML Strategies - Wired into Signal Engine")
    pdf.file_entry("src/trader/auto_trader.py", 15, "MODIFIED - _build_strategies()")
    pdf.body_text(
        "MLStrategy (HistGradientBoosting) and EnsembleMLStrategy (RF + XGBoost + LogReg voting) "
        "were fully implemented but only used by AlphaScoreEngine for supplementary scoring. "
        "They were NOT included in _build_strategies() and therefore never generated trading signals.\n\n"
        "Fixed: Both now added to the signal engine with graceful fallback if dependencies missing:\n"
        "  - MLStrategy: weight 0.25\n"
        "  - EnsembleMLStrategy: weight 0.35\n"
        "Combined with existing RSI (0.30), SMA (0.30), Combined (0.40), and optional Pattern (0.20), "
        "the signal engine now runs up to 6 strategies per scan cycle."
    )

    # ── PHASE 6 BUG FIXES SUMMARY TABLE ──
    pdf.add_page()
    pdf.section_title("Phase 6: Bug Fix Impact Summary")
    pdf.body_text(
        "All bugs followed the same pattern: modules fully implemented with correct logic, but "
        "integration code used wrong class names or method signatures. Python's dynamic typing "
        "combined with except Exception: pass meant these failures were completely silent."
    )

    phase6_bugs = [
        ("Wrong Class", "OptionsFlowAnalyzer", "OptionsFlowTracker", "alpha_score.py"),
        ("Wrong Class", "AlternativeDataFetcher", "AlternativeDataTracker", "alpha_score.py"),
        ("Wrong Method", "insider.get_recent()", "get_insider_sentiment()", "alpha_score.py"),
        ("Wrong Method", "options.analyze()", "get_put_call_ratio()", "alpha_score.py"),
        ("Wrong Method", "detect_events(articles)", "detect_from_articles(articles)", "alpha_score.py, news_pipeline.py"),
        ("Wrong Type", "GoogleTrendsResult as DataFrame", "Use .score property", "alpha_score.py"),
        ("Dead Code", "DynamicRiskManager", "Never instantiated", "main.py"),
        ("Dead Code", "Circuit breakers", "Never checked", "auto_trader.py"),
        ("Disconnected", "PortfolioTracker", "Never synced with trades", "auto_trader.py"),
        ("Disconnected", "RegimeDetector", "Never called from trading loop", "auto_trader.py"),
        ("Disconnected", "MLStrategy", "Not in signal engine", "auto_trader.py"),
        ("Disconnected", "EnsembleMLStrategy", "Not in signal engine", "auto_trader.py"),
        ("Disconnected", "Smart money confidence", "Never applied to signals", "auto_trader.py"),
        ("Disconnected", "Options flow confidence", "Never applied to signals", "auto_trader.py"),
        ("Disconnected", "Alt data confidence", "Never applied to signals", "auto_trader.py"),
        ("Fake Data", "news_pipeline regime", "Hardcoded strings from sentiment", "news_pipeline.py"),
    ]

    bwidths = [22, 50, 52, 56]
    pdf.table_row(["Type", "What Was Broken", "Fix Applied", "File(s)"], bwidths, bold=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    for i, row in enumerate(phase6_bugs):
        if pdf.get_y() > 265:
            pdf.add_page()
            pdf.table_row(["Type", "What Was Broken", "Fix Applied", "File(s)"], bwidths, bold=True)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.table_row(list(row), bwidths, bg=(i % 2 == 0))

    pdf.ln(6)
    pdf.subsection("Confidence Adjustment Stack (NEW)")
    pdf.body_text(
        "Each trade signal now passes through four data sources that adjust confidence before execution:\n"
        "  - Smart Money (insider trading):  -15 to +15 points\n"
        "  - Options Flow (put/call, UOA):   -10 to +10 points\n"
        "  - Alternative Data (Google Trends, patents, GitHub): -10 to +10 points\n"
        "  - Combined maximum adjustment:    -35 to +35 points\n\n"
        "A strong insider cluster buying + bullish options flow + rising Google Trends can boost a "
        "borderline 45% confidence signal to 80%, while heavy insider selling + bearish options can "
        "kill a 60% signal entirely."
    )

    pdf.subsection("Active Strategy Weights (Updated)")
    pdf.body_text(
        "The signal engine now runs up to 6 strategies per scan:\n"
        "  - RSI Strategy:            0.30 weight\n"
        "  - SMA Crossover:           0.30 weight\n"
        "  - Combined (RSI+SMA):      0.40 weight\n"
        "  - Pattern (if enabled):    0.20 weight\n"
        "  - MLStrategy (NEW):        0.25 weight\n"
        "  - EnsembleMLStrategy (NEW): 0.35 weight"
    )

    # ── PHASE 7 ──
    pdf.add_page()
    pdf.section_title("Phase 7: NPU/GPU Data Processor & Historical Pipeline (March 28)")
    pdf.body_text(
        "Major addition: a hardware-accelerated data processing engine that transforms raw historical "
        "OHLCV data into a pre-computed 'processed data block' - ready-to-use ML features, trained "
        "models, and cached predictions stored in SQLite. Eliminates all computation at trade time. "
        "The daily downloader now automatically triggers the processor after each data refresh so "
        "the trader always starts with zero-latency signal generation."
    )

    pdf.subsection("1. Data Processor Engine - NPU + RTX GPU Acceleration")
    pdf.file_entry("src/ops/data_processor.py", 870, "NEW - Core processing engine")
    pdf.body_text(
        "New module that reads raw OHLCV from historical_master.db and produces a complete processed "
        "data block in data_cache/processed_data.db. Three main components:\n\n"
        "  Hardware Detection (HardwareProfile):\n"
        "    - Auto-detects CUDA GPU (RTX), RK3588 NPU, or falls back to CPU\n"
        "    - Reports device name, memory, core count\n\n"
        "  Feature Engine (FeatureEngine):\n"
        "    - Computes all 22 ML features (16 base + 6 ensemble) for every symbol/date\n"
        "    - GPU path: stacks hundreds of symbols into CUDA tensors, computes rolling SMA, RSI,\n"
        "      MACD, Bollinger, stochastic, ATR, returns, volatility in parallel across all symbols\n"
        "    - CPU path: numpy-vectorized single-symbol processing with identical output\n"
        "    - Features: RSI, MACD (line/signal/hist), SMA pct (20/50/200), SMA cross, Bollinger\n"
        "      position/width, volume ratio, OBV slope, returns (1d/5d/20d), volatility 20d,\n"
        "      regime score, ROC 10, stochastic K/D, ATR pct, volatility ratio\n\n"
        "  Model Trainer (ModelTrainer):\n"
        "    - Trains 4 models on the feature set with time-based train/test split (6-month holdout)\n"
        "    - ml_strategy: HistGradientBoostingClassifier (16 features, early stopping)\n"
        "    - ensemble_rf: RandomForestClassifier (22 features, n_jobs=-1 for all cores)\n"
        "    - ensemble_xgb: XGBoost (22 features, uses CUDA GPU when available)\n"
        "    - ensemble_lr: LogisticRegression with StandardScaler (22 features)\n"
        "    - Auto-exports all models to ONNX for NPU conversion via ModelExporter\n"
        "    - Saves training metrics (accuracy, AUC-ROC, duration, device) to model_state table\n\n"
        "  Prediction Cache:\n"
        "    - Runs all models against latest features for every symbol\n"
        "    - Caches ml_signal, ensemble_signal, probabilities, confidence, agreement count\n"
        "    - Tries NPU inference first (via NPUManager), falls back to in-memory sklearn\n"
        "    - Predictions table indexed by (symbol, date) for instant lookup"
    )

    pdf.subsection("2. Two Processing Modes")
    pdf.body_text(
        "  run() - Full rebuild:\n"
        "    - Loads ALL symbols from historical_master.db\n"
        "    - Computes features in GPU/CPU batches (200 symbols per GPU batch, 50 per CPU batch)\n"
        "    - Retrains all 4 ML models on complete dataset\n"
        "    - Generates predictions for every symbol\n"
        "    - GC between phases to manage Rock 5B's 8GB RAM\n\n"
        "  run_incremental() - Fast daily update:\n"
        "    - Compares historical_master.db dates vs processed_data.db dates\n"
        "    - Only processes symbols with new data since last run\n"
        "    - Reuses existing trained models (no retraining)\n"
        "    - Generates fresh predictions for updated symbols\n"
        "    - Designed for post-download latency: minutes, not hours"
    )

    pdf.subsection("3. Processed Data Schema (processed_data.db)")
    pdf.body_text(
        "Four SQLite tables:\n"
        "  - processed_features: 22 ML feature columns per symbol/date (indexed)\n"
        "  - predictions: ml_signal, ensemble_signal, probabilities, confidence, device used\n"
        "  - model_state: training metrics per model (accuracy, AUC, device, duration)\n"
        "  - process_log: run history (mode, symbols, features, predictions, timing)\n\n"
        "Public read API (zero computation):\n"
        "  - get_features(symbol, days) -> DataFrame with 22 columns\n"
        "  - get_prediction(symbol) -> dict with signals and confidence\n"
        "  - get_all_predictions() -> DataFrame sorted by confidence\n"
        "  - get_status() -> dict with DB stats, model info, last run details"
    )

    pdf.subsection("4. Historical Downloader Integration")
    pdf.file_entry("src/data/historical_downloader.py", 30, "MODIFIED - Processor auto-launch")
    pdf.body_text(
        "run_daily_update() now accepts run_processor=True (default) parameter. After downloading "
        "fresh OHLCV data for all ~700 symbols, it automatically launches "
        "DataProcessor.run_incremental() to rebuild the processed data block. The processor runs "
        "in the same process (no subprocess overhead), catches all exceptions gracefully, and "
        "reports results back through the stats dict.\n\n"
        "New _run_data_processor() method handles the integration with ImportError fallback "
        "if the data_processor module isn't available.\n\n"
        "CLI updated: --no-processor flag skips processing. --initial now triggers full "
        "processor run (with model training) after the initial 25-year download completes."
    )

    pdf.subsection("5. Daily Scheduler Integration")
    pdf.file_entry("src/ops/daily_scheduler.py", 40, "MODIFIED - Processor task added")
    pdf.body_text(
        "Two integration points:\n\n"
        "  23:00 CET - Maintenance task (existing):\n"
        "    - HistoricalDownloader.run_daily_update(run_processor=True) now passes through\n"
        "      to DataProcessor.run_incremental() automatically\n"
        "    - Captures processor results (symbols, features, predictions, device, duration)\n"
        "    - Results available in task details for monitoring\n\n"
        "  23:30 CET - NEW data_processor_retrain task:\n"
        "    - Runs every Sunday (weekly full retrain)\n"
        "    - Calls DataProcessor.run(retrain=True) for complete rebuild\n"
        "    - Retrains all 4 models on latest data, regenerates all predictions\n"
        "    - Skips on non-Sunday (logs 'not_sunday' and exits)\n"
        "    - 30-minute timeout, LOW priority, runs on all days (not market-day dependent)"
    )

    pdf.subsection("6. NPU Manager - Processed Data Block Integration")
    pdf.file_entry("src/ops/npu_accelerator.py", 80, "MODIFIED - Cache-first prediction path")
    pdf.body_text(
        "Three new methods on NPUManager for consuming the processed data block:\n\n"
        "  predict_ml_fast(model_name, symbol):\n"
        "    - Cache-first prediction: checks processed_data.db first (instant lookup)\n"
        "    - On cache miss: loads cached features, runs live NPU/CPU inference\n"
        "    - Returns {signal, probability, confidence, source, device}\n"
        "    - source='cache' (instant) or source='live_inference' (NPU/CPU)\n\n"
        "  get_cached_prediction(symbol) / get_cached_features(symbol, days):\n"
        "    - Direct read from processed_data.db via DataProcessor singleton\n"
        "    - Zero computation, zero model loading\n\n"
        "  status() and print_status() updated:\n"
        "    - Now includes processed data block info (feature count, prediction count, DB size)\n"
        "    - Shows trained model metrics (accuracy, AUC, device) from model_state table\n"
        "    - Unified view of NPU hardware + data processor state"
    )

    pdf.subsection("Data Flow Summary")
    pdf.body_text(
        "  23:00 CET  HistoricalDownloader.run_daily_update()\n"
        "               -> downloads fresh OHLCV for ~700 symbols (15-20 min)\n"
        "               -> triggers DataProcessor.run_incremental()\n"
        "                   -> computes 22 features per symbol (GPU or CPU)\n"
        "                   -> runs predictions through trained models (NPU or CPU)\n"
        "                   -> stores everything in processed_data.db\n\n"
        "  23:30 CET  (Sundays) DataProcessor.run() full retrain\n"
        "               -> retrains all 4 ML models on complete dataset\n"
        "               -> exports ONNX for NPU conversion\n"
        "               -> regenerates all predictions\n\n"
        "  Trading hours:  NPUManager.predict_ml_fast()\n"
        "               -> instant cache lookup from processed_data.db\n"
        "               -> zero-latency signals ready at market open"
    )

    # ── PHASE 7 FILE INVENTORY ADDITION ──
    pdf.add_page()
    pdf.section_title("Phase 7: File Inventory")

    pdf.subsection("New Files")
    p7_new = [
        ("Mar 28", "src/ops/data_processor.py", "870", "NPU/GPU data processor engine"),
    ]
    widths7 = [18, 85, 15, 62]
    pdf.table_row(["Date", "File", "Lines", "Description"], widths7, bold=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    for i, (date, path, lines, desc) in enumerate(p7_new):
        pdf.table_row([date, path, lines, desc], widths7, bg=(i % 2 == 0))

    pdf.ln(5)
    pdf.subsection("Modified Files")
    p7_mods = [
        ["historical_downloader.py", "713", "780", "+67", "Auto-launch processor after download, --no-processor flag"],
        ["daily_scheduler.py", "651", "700", "+49", "data_processor_retrain task, processor result capture"],
        ["npu_accelerator.py", "709", "789", "+80", "Cache-first predict_ml_fast(), status integration"],
    ]
    mwidths7 = [38, 15, 15, 12, 100]
    pdf.table_row(["File", "Before", "After", "Delta", "Key Changes"], mwidths7, bold=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    for i, row in enumerate(p7_mods):
        pdf.table_row(row, mwidths7, bg=(i % 2 == 0))

    # ── PHASE 7 BUG FIXES ──
    pdf.ln(6)
    pdf.subsection("Phase 7: Bugs Fixed / Issues Addressed")

    p7_bugs = [
        ("Zero-Latency Gap at Market Open",
         "src/ops/data_processor.py, src/data/historical_downloader.py",
         "Previously, the trader had to compute ML features and load models at startup, causing "
         "minutes of delay before the first signal. The data processor pre-computes everything "
         "overnight so the trader starts with instant predictions from processed_data.db."),
        ("Historical Data Never Used for ML Training",
         "src/ops/data_processor.py",
         "The 25-year historical_master.db existed but was never consumed by the ML training "
         "pipeline. Models trained only on recent DataPipeline data (~1 year). The data processor "
         "now trains on the full historical dataset with proper time-based splits."),
        ("NPU Models Never Retrained",
         "src/ops/daily_scheduler.py",
         "ONNX/RKNN models were exported once during initial setup but never updated as new data "
         "accumulated. Weekly Sunday retrain task now rebuilds and re-exports models automatically."),
        ("Daily Download Without Processing",
         "src/data/historical_downloader.py",
         "run_daily_update() downloaded fresh data every night but never triggered feature "
         "recomputation or prediction updates. The processed data block could go stale indefinitely. "
         "Now auto-launches incremental processor after every successful download."),
        ("GPU Never Utilized for Trading Workloads",
         "src/ops/data_processor.py",
         "The --mode research flag claimed GPU acceleration but was a stub that just launched the "
         "dashboard. The FeatureEngine GPU path now batch-processes hundreds of symbols on CUDA "
         "tensors. XGBoost training also uses CUDA when available (device='cuda', tree_method='hist')."),
    ]

    for title, files, desc in p7_bugs:
        pdf.subsection(title)
        pdf.set_font("Courier", "", 7.5)
        pdf.set_text_color(0, 100, 80)
        pdf.cell(0, 4, f"  Files: {files}", new_x="LMARGIN", new_y="NEXT")
        pdf.body_text(desc)

    # ── GLOBAL EXCHANGE COVERAGE ──
    pdf.add_page()
    pdf.section_title("Global Stock Exchange Coverage & Signal Intelligence")

    pdf.body_text(
        "The Alpha Trading Platform operates as a 24/7 global trading system spanning 11 stock exchanges "
        "across all major time zones. The MarketCalendar determines which markets are open at any moment "
        "(CET timezone), and the AutoTrader scans only symbols from currently-open markets every 1-5 minutes. "
        "As one market closes, the MarketHandoff engine transfers session context (risk-on/risk-off posture, "
        "position size multipliers) to the next opening market."
    )

    pdf.subsection("11 Global Exchanges")

    widths_ex = [45, 28, 15, 30, 72]
    pdf.table_row(["Exchange", "Region", "Sym", "Hours (CET)", "Key Instruments"], widths_ex, bold=True, bg=True)
    exchanges = [
        ("Crypto (24/7)", "Global", "9", "Always", "BTC, ETH, SOL, XRP, BNB, ADA, AVAX, DOT, MATIC"),
        ("NZX (New Zealand)", "APAC", "10", "22:00-03:00", "AIR.NZ, SPK.NZ, FPH.NZ, MEL.NZ"),
        ("ASX (Australia)", "APAC", "20", "01:00-07:00", "BHP, CBA, CSL, NAB, WBC, RIO, MQG"),
        ("TSE (Japan)", "APAC", "20", "01:00-07:30", "Toyota, Sony, SoftBank, Mitsubishi UFJ"),
        ("HKEX (Hong Kong)", "APAC", "20", "02:00-08:00", "Tencent, Alibaba, AIA, Meituan, JD"),
        ("NSE (India)", "APAC", "20", "04:45-11:15", "Reliance, TCS, HDFC, Infosys, ICICI"),
        ("EU + Nordic", "Europe", "50", "09:00-17:30", "NOVO-B, ASML, SAP, LVMH, Siemens, Volvo"),
        ("LSE (London)", "Europe", "10", "09:00-17:30", "Shell, AstraZeneca, HSBC, BP, GSK"),
        ("NYSE/NASDAQ (US)", "Americas", "49", "15:30-22:00", "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA"),
        ("CME/CBOT (Chicago)", "Americas", "19", "15:30-22:00", "Gold, Silver, Oil, S&P E-mini, 10Y"),
        ("Global ETFs", "Multi", "29", "15:30-22:00", "SPY, QQQ, IWM, VTI, GLD, XLK, VWO"),
    ]
    for row in exchanges:
        pdf.table_row(list(row), widths_ex)

    pdf.ln(2)
    pdf.body_text(
        "Total: 256 unique instruments tracked in real-time. The historical downloader fetches daily OHLCV "
        "data for approximately 700 symbols (including alternative universes and watchlist candidates) every "
        "night at 23:00 CET, storing 25 years of history in SQLite. The data processor then computes 22 ML "
        "features per symbol and generates fresh predictions via 4 trained models (HistGradientBoosting, "
        "RandomForest, XGBoost, LogisticRegression)."
    )

    pdf.subsection("News & Sentiment Intelligence Pipeline")
    pdf.body_text(
        "Every trading scan ingests live news from 4 sources and applies multi-layer sentiment analysis "
        "to adjust signal confidence before execution:"
    )
    pdf.bullet("Finnhub API -- company news, earnings calendar, economic events (credibility: 0.80)")
    pdf.bullet("Alpha Vantage -- market sentiment API with symbol-specific scoring (credibility: 0.75)")
    pdf.bullet("RSS Feeds -- Reuters (0.95), CNBC (0.80), Yahoo Finance (0.70), MarketWatch (0.75)")
    pdf.bullet("FinBERT NLP -- 3-layer sentiment: NPU-accelerated (~5ms) -> CPU FinBERT (~200ms) -> keyword fallback")
    pdf.ln(2)
    pdf.body_text(
        "Sentiment scores are weighted by source credibility and article age (24-hour half-life decay). "
        "The news pipeline detects 9 cross-impact themes that propagate signals across related stocks:"
    )
    pdf.bullet("Oil Price -> XOM, CVX, EQNR.OL, BP.L, TTE.PA, SHEL.L")
    pdf.bullet("Interest Rates -> JPM, BAC, GS, MS, C, WFC (financials)")
    pdf.bullet("Semiconductors -> ASML.AS, NVDA, AMD, TSM, INTC, QCOM")
    pdf.bullet("AI Spending -> NVDA, MSFT, GOOGL, AMZN, META, AMD, CRM")
    pdf.bullet("GLP-1 Obesity Drugs -> NOVO-B.CO, NVO, LLY, AMGN, VKTX")
    pdf.bullet("China Policy -> BABA, JD, PDD, NIO, XPEV, LI")
    pdf.bullet("USD Strength -> GLD, SLV, EEM, FXE, UUP")
    pdf.bullet("ECB Policy -> DANSKE.CO, SAN.MC, BNP.PA, DBK.DE, ING.AS")
    pdf.bullet("Energy Transition -> ENPH, SEDG, FSLR, NEE, ORSTED.CO, VESTAS.CO")

    pdf.subsection("Signal Flow: From 700 Symbols to Trade Execution")
    pdf.body_text(
        "The complete signal chain processes data through 7 stages before any trade is executed:\n\n"
        "1. MARKET CALENDAR: Checks CET time, determines which of 11 markets are open, returns only "
        "tradeable symbols (e.g. 15:30 CET = US pre-market -> 78 US symbols + ETFs).\n\n"
        "2. DATA FETCH: Gets OHLCV + 30 technical indicators for open symbols. Updates all position "
        "prices. Historical downloader provides 700+ symbol coverage overnight.\n\n"
        "3. REGIME DETECTION: Analyzes SPY data to detect Bull/Bear/Sideways/Crash. Adjusts max exposure "
        "(CRASH=10%, BEAR=30%, SIDEWAYS=50%, BULL=100%), stop-loss multiplier, and blocks new buys in CRASH.\n\n"
        "4. SIGNAL ENGINE: Runs 6 strategies in parallel via ThreadPoolExecutor:\n"
        "   - RSI Strategy (0.30 weight), SMA Crossover (0.30), Combined (0.40)\n"
        "   - Pattern Strategy (0.20), ML Strategy (0.25), Ensemble ML (0.35)\n"
        "   Returns BUY/SELL/HOLD with confidence 0-100% per symbol.\n\n"
        "5. ALPHA SCORE: 6-dimensional scoring (technical 25%, sentiment 20%, ML 20%, macro 10%, "
        "alternative data 15%, seasonality 10%) produces a 0-100 score per symbol.\n\n"
        "6. CONFIDENCE ADJUSTMENT: Smart money (insider trading: -15 to +15), options flow (-10 to +10), "
        "and alternative data (Google Trends: -10 to +10) adjust raw confidence by up to +/-35 points. "
        "A borderline 45% signal can be boosted to 80% by strong insider buying + bullish options flow.\n\n"
        "7. RISK FILTER & EXECUTION: Positions must pass max position size (10%), max exposure (80-100%), "
        "max correlation (0.85), max daily loss (15%), and circuit breaker checks. Max 8 new positions "
        "per scan, 30 concurrent positions total. Exchange-specific stop-losses (e.g. crypto 0.5%, "
        "US stocks 4%, Denmark 3%) protect against loss."
    )

    pdf.subsection("Impact: Why Global Coverage Matters")
    pdf.body_text(
        "Trading 256 symbols across 11 exchanges with 700+ symbol intelligence creates compound advantages:\n\n"
        "- CONTINUOUS ALPHA: When US markets close at 22:00 CET, Asian markets open at 22:00-01:00. "
        "The trader never sleeps -- it captures overnight moves in Asia that foreshadow European opens.\n\n"
        "- CROSS-IMPACT SIGNALS: A Reuters article about oil prices at 14:00 CET immediately impacts "
        "scoring for Shell (LSE), Equinor (Oslo), TotalEnergies (Paris), and Exxon (NYSE) -- even before "
        "some of those markets open. The news pipeline pre-positions sentiment so signals fire instantly "
        "at market open.\n\n"
        "- REGIME DIVERSIFICATION: A US bear market (SPY down) triggers defensive posture globally, but "
        "the trader can still find bullish signals in uncorrelated markets (Nordic pharma, Asian tech, "
        "gold futures). The 256-symbol universe provides escape routes that a single-market trader lacks.\n\n"
        "- ML FEATURE RICHNESS: The 22 ML features computed nightly for 700+ symbols create a training "
        "dataset of ~15,000 symbol-dates per year. Models trained on this breadth generalize better than "
        "models trained on a narrow US-only universe. XGBoost and RandomForest ensemble voting across "
        "this dataset produces more robust buy/sell predictions.\n\n"
        "- TAX OPTIMIZATION: With positions across multiple currencies (USD, DKK, EUR, GBP, HKD, AUD), "
        "the Danish mark-to-market tax engine can identify loss-harvesting opportunities in one currency "
        "while maintaining exposure in another -- reducing the 22-42% corporate tax burden."
    )

    # ── PHASE 8 ──
    pdf.add_page()
    pdf.section_title("Phase 8: i18n Completion, Tax Center Fix & Backtest Re-enable (March 28 PM)")

    pdf.subsection("Portfolio Table Headers Frozen in Danish -- Bug Fix")
    pdf.file_entry("src/dashboard/pages/portfolio.py", "~20", "MODIFIED - Dynamic translation headers")
    pdf.body_text(
        "BUG: The position table column headers (Symbol, Antal, Gns.Kost, Kurs, Vaerdi, Skat*) were "
        "evaluated at module import time via a module-level constant _POS_TABLE_HEADERS = [t('..'), ...]. "
        "Since default language is Danish at import, all headers were frozen in Danish regardless of "
        "language selection. Fix: replaced with a _pos_table_headers() function that evaluates t() calls "
        "at render time. Also replaced 6 hardcoded strings in the portfolio page with t() calls:\n"
        "  - 'Download Performance Report (PDF)' -> t('portfolio.download_report')\n"
        "  - 'Kontant + positioner inkl. urealiseret P&L' -> t('portfolio.total_desc')\n"
        "  - 'trading' / 'closed' count labels -> t('portfolio.trading_count/closed_count')\n"
        "  - 'Show all' button -> t('portfolio.show_all')\n"
        "  - 'Short' / 'Long' -> t('common.short/long')"
    )

    pdf.subsection("Portfolio Trading Count Mismatch -- Bug Fix")
    pdf.file_entry("src/dashboard/pages/portfolio.py", "~10", "MODIFIED - Synchronized count and rows")
    pdf.body_text(
        "BUG: The header showed '4 trading, 0 closed' but only 1 row was visible. The count was computed "
        "with a fresh call to _is_symbol_trading() while row rendering used a separate call inside "
        "_pos_to_dict(). Market open/close status could change between the two calls, causing a mismatch "
        "where the count said 'trading' but the row got display:none. Fix: pre-compute _pos_to_dict() "
        "once in portfolio_layout() and derive both the count and the table from the same data."
    )

    pdf.subsection("Tax Center Empty Data -- Critical Bug Fix")
    pdf.file_entry("src/dashboard/pages/tax_center.py", "~80", "MODIFIED - Wired live position data")
    pdf.body_text(
        "CRITICAL BUG: The tax center page passed an empty list [] to all tax calculation methods:\n"
        "  - calc.ytd_estimated_tax([])  -- always returned 0 tax\n"
        "  - calc.suggest_tax_optimization([])  -- never found optimization opportunities\n"
        "  - mtm.get_current_valuations()  -- method does not exist, silently failed\n\n"
        "Fix: added full position fetching from PaperBroker (same pattern as portfolio page) with "
        "USD/DKK FX conversion. Positions are now passed to ytd_estimated_tax(position_dicts) and "
        "suggest_tax_optimization(position_dicts). The non-existent get_current_valuations() was "
        "replaced with the correct calculate_ytd(year, position_dicts) method from MarkToMarketEngine. "
        "Tax center now shows actual estimated tax on unrealized gains."
    )

    pdf.subsection("Backtests Re-enabled with RAM Optimization")
    pdf.file_entry("src/dashboard/app.py", "~40", "MODIFIED - RAM-optimized backtest runner")
    pdf.file_entry("src/backtest/backtester.py", "~10", "MODIFIED - GC and memory cleanup")
    pdf.body_text(
        "The _run_backtests() function was disabled with 'return {}' because the original configuration "
        "(82 symbols x 2 years x 50+ indicator columns x 3 strategies) caused ~1-2GB peak RAM from "
        "DataFrame slicing (41,000+ copies per backtest). The overview and risk pages showed only a "
        "'Backtests disabled (RAM constraint)' warning.\n\n"
        "Fix: re-enabled with three RAM optimizations:\n"
        "  1. Reduced from 82 to 15 most liquid symbols (AAPL, MSFT, GOOGL, etc. + NOVO-B.CO, BTC-USD)\n"
        "  2. Reduced date range from 2 years to 1 year (2025-03-01 to 2026-03-01)\n"
        "  3. Added gc.collect() between strategy runs to free intermediate allocations\n"
        "  4. Added gc.collect() after data fetch phase in backtester\n"
        "Estimated peak RAM: ~150-200 MB (down from ~1.5 GB). The overview, risk, and tax analysis "
        "pages now show actual backtest-derived metrics (Sharpe, Sortino, drawdown, equity curves)."
    )

    pdf.subsection("Complete i18n: 168 Missing Keys Added to 5 Languages")
    pdf.file_entry("lang/deutsch.lan", "~650", "MODIFIED - Added 168 missing keys")
    pdf.file_entry("lang/francais.lan", "~650", "MODIFIED - Added 168 missing keys")
    pdf.file_entry("lang/espanol.lan", "~650", "MODIFIED - Added 168 missing keys")
    pdf.file_entry("lang/portugues.lan", "~650", "MODIFIED - Added 168 missing keys")
    pdf.file_entry("lang/vlaams.lan", "~650", "MODIFIED - Added 168 missing keys")
    pdf.body_text(
        "Cross-language comparison revealed that dansk.lan and eng.lan had 646 keys each, but all 5 "
        "other language files (DE, FR, ES, PT, VL) were missing 168 keys -- the exact same set. Missing "
        "sections: allocation (7), analysis (12), calendar (15), economy (18), health (33), regime (16), "
        "risk (1), settings (24), tax (42). All 168 keys were translated and added. Every language file "
        "now has 646 keys with 0 missing."
    )

    pdf.subsection("Chart Labels Translated -- 14 New Keys")
    pdf.file_entry("src/dashboard/app.py", "~12", "MODIFIED - Replaced hardcoded chart text with t()")
    pdf.file_entry("lang/*.lan (all 7)", "--", "MODIFIED - Added 'charts' section")
    pdf.body_text(
        "Multiple chart axis labels and legend entries were hardcoded in mixed Danish/English:\n"
        "  - 'Profit/Tab ($)', 'Antal handler', 'Vigtighed' (Danish)\n"
        "  - 'Portfolio Return', 'Composite Score', 'Portfolio Value ($)' (English)\n"
        "  - 'Med risikostyring' / 'Uden risikostyring' (Danish)\n"
        "All 12 hardcoded strings replaced with t('charts.*') calls. New 'charts' section added to "
        "all 7 language files with 14 translation keys covering chart axis labels, legend entries, "
        "and Monte Carlo simulation labels."
    )

    pdf.subsection("Backtest Disabled Message Translated")
    pdf.file_entry("src/dashboard/app.py", "~6", "MODIFIED - Used t() for fallback messages")
    pdf.body_text(
        "The overview and tax analysis pages had hardcoded English fallback messages when backtests "
        "were disabled. Replaced with t('risk.backtests_disabled') which is now translated in all "
        "7 languages. The 'Go to Tax Center' link text also uses t('nav.tax_center') instead of "
        "hardcoded English."
    )

    # Phase 8 file inventory
    pdf.subsection("Phase 8: File Inventory")

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 30, 40)
    pdf.cell(0, 5, "Modified Files", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    widths = [75, 20, 90]
    pdf.table_row(["File", "Delta", "Key Changes"], widths, bold=True, bg=True)
    p8_files = [
        ("src/dashboard/pages/portfolio.py", "+30", "Dynamic headers, count sync, i18n strings"),
        ("src/dashboard/pages/tax_center.py", "+80", "Live positions, MTM fix, FX conversion"),
        ("src/dashboard/app.py", "+50", "Backtests re-enabled, chart i18n, fallback i18n"),
        ("src/backtest/backtester.py", "+5", "GC cleanup after fetch and simulation"),
        ("lang/deutsch.lan", "+168", "Full translation parity (646 keys)"),
        ("lang/francais.lan", "+168", "Full translation parity (646 keys)"),
        ("lang/espanol.lan", "+168", "Full translation parity (646 keys)"),
        ("lang/portugues.lan", "+168", "Full translation parity (646 keys)"),
        ("lang/vlaams.lan", "+168", "Full translation parity (646 keys)"),
        ("lang/dansk.lan", "+19", "New portfolio/common/chart keys"),
        ("lang/eng.lan", "+19", "New portfolio/common/chart keys"),
    ]
    for f, delta, desc in p8_files:
        pdf.table_row([f, delta, desc], widths)

    # Phase 8 bug summary
    pdf.ln(4)
    pdf.subsection("Phase 8: Bug Fix Summary")
    widths_bug = [40, 55, 90]
    pdf.table_row(["Type", "What Was Broken", "Fix Applied"], widths_bug, bold=True, bg=True)
    p8_bugs = [
        ("Import-time eval", "_POS_TABLE_HEADERS frozen in DA", "Changed to function, eval at render"),
        ("Timing mismatch", "Trading count != visible rows", "Pre-compute once, share data"),
        ("Empty data", "ytd_estimated_tax([])", "Fetch live positions from PaperBroker"),
        ("Missing method", "mtm.get_current_valuations()", "Use mtm.calculate_ytd(year, pos)"),
        ("RAM bloat", "Backtests disabled (82sym x 2yr)", "15 symbols, 1 year, GC between runs"),
        ("Missing i18n", "168 keys missing x 5 languages", "All 7 files at 646 keys parity"),
        ("Hardcoded text", "Chart labels in mixed DA/EN", "14 new charts.* translation keys"),
    ]
    for typ, broken, fix in p8_bugs:
        pdf.table_row([typ, broken, fix], widths_bug)

    # ══════════════════════════════════════════════════════════════
    # PHASE 9 — 2026-03-29: Dashboard UX Overhaul, Rebalancer, Multi-Asset Trading
    # ══════════════════════════════════════════════════════════════
    pdf.section_title("Phase 9: Dashboard UX Overhaul, Rebalancer & Multi-Asset Trading (2026-03-29)")
    pdf.body_text(
        "Major feature release spanning the entire dashboard. Added portfolio rebalancing engine with "
        "5-class asset allocation (stocks, bonds, commodities, crypto, cash), multi-asset sell pages, "
        "exchange-level fund limits, report downloads, and dozens of UX fixes across all pages."
    )

    # ── Overview page fixes ──
    pdf.subsection("Overview Page (Overblik)")
    pdf.bullet("Portfolio vs S&P 500 chart: fixed x-axis mismatch (portfolio had integer index, SPY had datetime)")
    pdf.bullet("Daily returns chart: rewrote from pct_change() to trade-based P&L aggregation by exit date")
    pdf.bullet("Background data preload: backtests + benchmark fetched in daemon thread so GUI never freezes")
    pdf.bullet("Loading spinner shown while background data loads")
    pdf.bullet("Currency: replaced hardcoded $ with format_value() for DKK display throughout")
    pdf.bullet("All chart titles translated via t() across all 7 languages")

    # ── Portfolio page ──
    pdf.subsection("Portfolio Page")
    pdf.bullet("Added 'Type' column: Stock (blue), Crypto (purple), Bond (green), Commodity (orange)")
    pdf.bullet("Per Asset Type pie chart now includes all 4 types + Cash (was only Stocks/Crypto)")
    pdf.bullet("Positions table: vertical scrollbar (maxHeight 500px) for accessing all positions")
    pdf.bullet("Removed [:50] position limit - all positions now shown")
    pdf.bullet("Tax calculation fix: was computing 22% on USD amounts treated as DKK (297K tax on 100K profit)")
    pdf.bullet("Fixed by converting unrealized_pnl from USD to DKK before tax calc")
    pdf.bullet("'Vis alle' button: fixed double-click bug (was comparing Danish text to English 'Show all')")
    pdf.bullet("JS row rebuild updated with asset_class + exchange columns")

    # ── Trading page ──
    pdf.subsection("Trading Page (Handel)")
    pdf.bullet("New Order: added asset class radio (Stock/Bond/Crypto/Commodity)")
    pdf.bullet("Symbol input changed from free-text to searchable dropdown filtered by asset class + open markets")
    pdf.bullet("Price info box: shows current price (DKK), total cost, cash available, cash after order")
    pdf.bullet("Price calculation: fixed rounding (474 x 4 showed 1894 instead of 1896)")
    pdf.bullet("Buy Top 10 confirmation: qty inputs per stock, live total recalc, cash check, insufficient cash warning")
    pdf.bullet("Execute: changed from position-sizing (hundreds of shares) to user-specified qty (default 1)")
    pdf.bullet("Sell: fixed broken crypto/position sell - PaperBroker fallback added to _get_sell_options() and submit_trade()")
    pdf.bullet("Sell: preview and submit now read from correct dropdown (buy vs sell) based on side")
    pdf.bullet("Result alerts: fixed green-on-green unreadable text - now dark card background with proper text colors")

    # ── Sell Positions page ──
    pdf.subsection("Sell Positions Page (Saelg)")
    pdf.bullet("Added 'Sell All Bonds' and 'Sell All Commodities' cards (was only All/Crypto/Stocks)")
    pdf.bullet("Position classification uses _asset_class() for bonds/commodities detection")
    pdf.bullet("Fixed phantom modal popup: sell crypto triggered sell stocks modal due to component re-render")
    pdf.bullet("Guard added: check n_clicks > 0 to ignore spurious fires from _build_sell_cards() refresh")
    pdf.bullet("Sell confirmation: added gain/loss (DKK + %) next to each symbol")
    pdf.bullet("Cards auto-refresh after sell completes (execute_sell outputs to sell-cards-container)")
    pdf.bullet("Result alerts: fixed green-on-green - now dark card style")

    # ── Market Overview page ──
    pdf.subsection("Market Overview Page (Markedsoverblik)")
    pdf.bullet("Allocation donut: 5 editable % inputs (stocks/bonds/commodities/crypto/cash)")
    pdf.bullet("Donut updates live as inputs change, total validation (must = 100%)")
    pdf.bullet("Apply button triggers rebalancing engine:")
    pdf.bullet("  1. Classifies all positions into 5 asset classes", indent=10)
    pdf.bullet("  2. Computes current vs target allocation drift per class", indent=10)
    pdf.bullet("  3. Sells overweight classes (worst-scored positions first)", indent=10)
    pdf.bullet("  4. Buys underweight classes (top-scored candidates, with bond/commodity ETF fallbacks)", indent=10)
    pdf.bullet("Allocation persisted to config/allocation.json - survives page re-renders")
    pdf.bullet("Auto-refresh skip added for /marked to prevent input reset")
    pdf.bullet("Exchange Limits card: 11 exchanges with % limit fields, saved to config/exchange_limits.json")
    pdf.bullet("Removed qty/total columns from top 10 buy candidates table (simplified)")

    # ── Reports page ──
    pdf.subsection("Reports Page")
    pdf.bullet("Portfolio PDF: existing performance report download")
    pdf.bullet("Tax CSV: new - exports all closed trades (date, symbol, P&L, commission)")
    pdf.bullet("Settings JSON: new - exports all config/*.json and config/*.yaml as single file")

    # ── Global UX ──
    pdf.subsection("Global UX Improvements")
    pdf.bullet("Danish forced as default language on startup (set_language('da'))")
    pdf.bullet("All dropdown text changed to dark gray (#2d3748) on light background (#e2e8f0)")
    pdf.bullet("All input/select elements styled consistently via global CSS")
    pdf.bullet("Placeholder text: medium gray (#718096)")

    # ── Translations ──
    pdf.subsection("Translation Keys Added (all 7 languages)")
    pdf.bullet("charts.daily_returns_pct, charts.daily_pnl_pct, charts.portfolio_vs_sp500")
    pdf.bullet("portfolio.trading_only (for Vis alle toggle)")
    pdf.bullet("common.apply")
    pdf.bullet("trading.insufficient_cash")
    pdf.bullet("sell.sell_bonds, sell.sell_bonds_desc, sell.sell_commodities, sell.sell_commodities_desc")

    # Phase 9 file inventory
    pdf.subsection("Phase 9: File Inventory")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 30, 40)
    pdf.cell(0, 5, "Modified Files", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    widths = [75, 20, 90]
    pdf.table_row(["File", "Delta", "Key Changes"], widths, bold=True, bg=True)
    p9_files = [
        ("src/dashboard/app.py", "+600", "Rebalancer, allocation, exchange limits, sell cards, reports, CSS"),
        ("src/dashboard/pages/portfolio.py", "+80", "Asset class column, tax fix, scroll, pie chart"),
        ("src/dashboard/pages/trading.py", "+200", "Asset dropdown, price info, sell fix, qty inputs"),
        ("src/backtest/backtester.py", "+10", "Date-indexed equity curve, return all_dates"),
        ("lang/dansk.lan", "+25", "New sell/trading/charts/common keys"),
        ("lang/eng.lan", "+25", "New sell/trading/charts/common keys"),
        ("lang/francais.lan", "+20", "Bonds/commodities/apply/insufficient_cash"),
        ("lang/deutsch.lan", "+20", "Bonds/commodities/apply/insufficient_cash"),
        ("lang/espanol.lan", "+20", "Bonds/commodities/apply/insufficient_cash"),
        ("lang/portugues.lan", "+20", "Bonds/commodities/apply/insufficient_cash"),
        ("lang/vlaams.lan", "+20", "Bonds/commodities/apply/insufficient_cash"),
        ("generate_changelog_pdf.py", "+130", "Phase 9 changelog entry"),
    ]
    for f, delta, desc in p9_files:
        pdf.table_row([f, delta, desc], widths)

    # New files
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 30, 40)
    pdf.cell(0, 5, "New Config Files", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.table_row(["File", "", "Purpose"], widths, bold=True, bg=True)
    pdf.table_row(["config/allocation.json", "NEW", "Persisted asset allocation percentages"], widths)
    pdf.table_row(["config/exchange_limits.json", "NEW", "Per-exchange fund limit percentages"], widths)

    # Phase 9 bug summary
    pdf.ln(4)
    pdf.subsection("Phase 9: Bug Fix Summary")
    widths_bug = [40, 55, 90]
    pdf.table_row(["Type", "What Was Broken", "Fix Applied"], widths_bug, bold=True, bg=True)
    p9_bugs = [
        ("X-axis mismatch", "Portfolio vs SPY misaligned", "Date index on equity curve + x= param"),
        ("Empty chart", "Daglig afkast showed no data", "Trade P&L by exit date instead of pct_change"),
        ("GUI freeze", "Backtests block page load", "Daemon thread preload + loading spinner"),
        ("Tax 3x error", "297K tax on 100K profit", "Convert USD->DKK before 22% calc"),
        ("Double-click", "Vis alle needed 2 clicks", "n_clicks parity instead of text compare"),
        ("Phantom modal", "Sell crypto opened stocks modal", "Guard n_clicks > 0 on re-render"),
        ("Green on green", "Success alerts unreadable", "Dark card bg + explicit text colors"),
        ("Sell broken", "Crypto sell no confirmation", "PaperBroker fallback + read sell dropdown"),
        ("Price rounding", "474x4 = 1894 not 1896", "round(price, 2) before multiply"),
        ("Dropdown text", "White text on white dropdown", "Global CSS dark gray + light bg"),
        ("Alloc reset", "Donut reset on Apply click", "Persist to JSON + skip auto-refresh"),
    ]
    for typ, broken, fix in p9_bugs:
        pdf.table_row([typ, broken, fix], widths_bug)

    # Save
    output_path = "/home/rock/reports/alpha_trading_changelog.pdf"
    pdf.output(output_path)
    print(f"PDF saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
