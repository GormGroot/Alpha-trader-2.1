# Alpha Vision Trading Platform - Master Development Plan

**Dato:** 17. marts 2026
**Udarbejdet for:** Ole, Founder - Alpha Vision Sports Tech
**Formål:** Markedsledende tradingplatform med europæisk handel og kommercialisering

---

## 1. NUVÆRENDE PLATFORM STATUS

### Hvad er allerede bygget (styrker)

Alpha Vision har allerede en imponerende teknologisk foundation:

**Kerne-infrastruktur:** Python/FastAPI backend, Dash/Plotly dashboard, SQLite caching, 29 unit tests, Pydantic config system.

**Handelsstrategier (6+):** SMA Crossover, RSI, ML (HistGradientBoosting), Ensemble ML (3-model voting med Random Forest + XGBoost + Logistic Regression), Combined Strategy, Signal Engine med parallel execution.

**Risk Management:** Pre/post-trade checks, stop-loss/take-profit/trailing stop, correlation monitor, dynamic risk adjustment, volatility scaling, max drawdown protection.

**Data & Analyse:** 30+ tekniske indikatorer, FinBERT sentiment, makro-data via FRED, on-chain krypto, options flow, alternativ data (Google Trends, GitHub, patents), RSS news feeds.

**Markeder (data-niveau):** US stocks, Nordic (OMX C25, Stockholm, Oslo), Europe (STOXX 50, FTSE 100, DAX 40), Asia, crypto, forex, commodities, ETFs.

**Ekstra:** Dansk skatteberegning (FIFO, 27%/42%), email-notifikationer, audit log, health monitoring, backtesting engine med stress test.

### Kritiske mangler for markedslederskab

1. **Ingen betalings-/abonnementssystem** - kan ikke sælges
2. **Kun Alpaca broker** - begrænset til US markeder for reel handel
3. **Ingen europæisk broker-integration** - kan se data men ikke handle
4. **Ingen brugerregistrering/auth** - single-user system
5. **Primitive UI** - Dash er funktionel men ikke consumer-grade
6. **Ingen mobil app**
7. **Ingen social/community features**
8. **Ingen real-time streaming** - polling-baseret
9. **Ingen white-label mulighed**

---

## 2. MARKEDSLEDENDE FEATURES - KOMPLET ROADMAP

### FASE 1: Europæisk Handel & Fundament (Måned 1-3)

#### 1.1 Multi-Broker Integration

**Problem:** Alpaca understøtter kun US markeder. For europæiske aktier skal vi integrere en europæisk broker.

**Løsning: Interactive Brokers (IBKR) som primær europæisk broker**

Hvorfor IBKR:
- Dækker 150+ markeder i 33 lande inkl. alle europæiske børser
- Robust API (TWS API + Client Portal API)
- Lave kommissioner (0.05% for europæiske aktier, min. €1.25)
- Paper trading built-in
- Reguleret i EU (MiFID II compliant)
- Python SDK: `ib_insync` (async-venlig)

**Sekundær option: Saxo Bank OpenAPI**
- Stærk i Norden/Europa
- White-label API tilgængelig
- Reguleret i Danmark (Finanstilsynet)
- Mere kompleks onboarding men bedre for nordiske kunder

**Arkitektur:**
```
BrokerRouter (ny komponent)
├── AlpacaBroker    → US aktier, crypto
├── IBKRBroker      → Europæiske aktier, futures, options
├── SaxoBroker      → Nordiske aktier (optional)
└── PaperBroker     → Test af alle markeder
```

#### 1.2 Europæisk Markedsdata

**Udfordringer:**
- Europæiske børser har delayed data (15-20 min) medmindre man betaler
- Forskellige valutaer (EUR, GBP, SEK, NOK, DKK, CHF)
- Forskellige åbningstider per børs
- ISIN-baseret identifikation vs. US ticker-system

**Løsning:**
- yfinance virker allerede for europæiske tickers (.PA, .DE, .AS, .MI, .L, .CO, .ST, .OL)
- Tilføj real-time feeds via IBKR markedsdata (kræver abonnement, ~€10-30/børs/måned)
- Implementer multi-currency portfolio tracking med real-time FX
- Byg børs-kalender for alle europæiske markeder

#### 1.3 Multi-Currency Support

- Real-time valutakurser via ECB/IBKR
- Portfolio-værdi i brugerens basisvaluta
- P&L beregning i både lokal og basisvaluta
- Automatisk FX-hedging mulighed
- Udvidet skatteberegning per land (ikke kun DK)

---

### FASE 2: Moderne Platform & Brugeroplevelse (Måned 3-6)

#### 2.1 Ny Frontend (React/Next.js)

Dash/Plotly er excellent til prototyping men ikke til et salgbart produkt. Migration til:

- **Next.js 15** med App Router
- **TailwindCSS** + **shadcn/ui** for professionelt design
- **TradingView Lightweight Charts** (gratis, open-source) til charting
- **WebSocket** real-time data streaming
- **Responsive design** (desktop + tablet + mobil)

**Sider:**
1. Dashboard (portfolio overview, P&L, watchlists)
2. Market Explorer (screener, heatmaps, sektorer)
3. Trading (ordreindgang, positioner, ordrehistorik)
4. Strategies (strategy builder, backtest, performance)
5. Signals (AI-genererede handelssignaler med confidence scores)
6. Risk Center (drawdown, exposure, correlation matrix)
7. News & Sentiment (aggregeret nyhedsfeed med sentiment scores)
8. Settings (broker connections, notifications, preferences)
9. Admin (kun for platform-ejere)

#### 2.2 Brugerregistrering & Authentication

- **Auth0** eller **Clerk** for managed auth
- Email/password + Google/Apple SSO
- 2FA (TOTP + SMS)
- Role-based access (Free, Premium, Pro, Enterprise)
- API key management for avancerede brugere
- GDPR-compliant data handling

#### 2.3 Real-Time Data Streaming

- **WebSocket server** (FastAPI WebSocket eller Socket.IO)
- Live prisdata, ordrestatus, P&L updates
- Push notifications (browser + email + mobil)
- Server-Sent Events som fallback

---

### FASE 3: Avancerede Features for Markedslederskab (Måned 6-9)

#### 3.1 No-Code Strategy Builder

Den feature der differentierer mest fra konkurrenterne:

- **Visual strategy builder** med drag-and-drop
- Kombiner indikatorer, betingelser og actions grafisk
- Backtest direkte fra builder
- Deling af strategier med community
- Template-bibliotek med populære strategier

#### 3.2 AI Trading Copilot

- **Natural language strategi-input:** "Køb når RSI < 30 og MACD krydser op, sælg efter 5% profit"
- **AI-genererede markedsanalyser** per aktie
- **Portfolio optimization** med ML (Markowitz + Black-Litterman)
- **Anomaly alerts** - AI opdager usædvanlige mønstre
- **Earnings prediction** baseret på alternativ data

#### 3.3 Social & Community

- **Strategy leaderboard** - top performers synlige for alle
- **Copy trading** - følg og kopier andres strategier
- **Community chat** per aktie/sektor
- **Strategy marketplace** - sælg/køb strategier
- **Public portfolios** med opt-in transparency

#### 3.4 Avanceret Ordretyper

- **Bracket orders** (entry + stop-loss + take-profit i én)
- **OCO orders** (One-Cancels-Other)
- **Trailing stop** med percentage og ATR-baseret
- **Time-in-force:** GTC, GTD, IOC, FOK
- **Conditional orders** (if price X then execute Y)
- **Smart order routing** (bedste pris på tværs af børser)

#### 3.5 Options Trading

- Options chain viewer
- Greeks beregning (Delta, Gamma, Theta, Vega)
- P&L diagram for komplekse strategier
- Implied volatility surface
- Options screener (unusual activity)

---

### FASE 4: Monetisering & Salg (Måned 9-12)

#### 4.1 Subscription Tiers

| Feature | Free | Premium (€29/mdr) | Pro (€79/mdr) | Enterprise (Custom) |
|---------|------|-------------------|----------------|---------------------|
| Paper Trading | Ja | Ja | Ja | Ja |
| US Aktier (delayed) | Ja | Real-time | Real-time | Real-time |
| EU Aktier | Nej | 5 markeder | Alle | Alle |
| Strategier | 2 | 10 | Unlimited | Unlimited |
| Backtesting | 1 år | 5 år | 20 år | Custom |
| AI Signals | 3/dag | 20/dag | Unlimited | Unlimited |
| Copy Trading | Nej | 3 traders | Unlimited | Unlimited |
| API Access | Nej | Read-only | Full | Full + webhooks |
| Priority Support | Nej | Email | Email + Chat | Dedicated |
| White-label | Nej | Nej | Nej | Ja |
| Strategy Marketplace | Browse | Buy | Buy + Sell | Custom |

#### 4.2 Betalingsintegration

- **Stripe** som primær payment processor
- Subscription billing med Stripe Billing
- Metered billing for API usage
- Gratis prøveperiode (14 dage Premium)
- Årlige rabatter (2 måneder gratis)
- Refund policy

#### 4.3 Yderligere Revenue Streams

1. **Markedsdata-videresalg** - bundled data subscriptions
2. **Strategy Marketplace cut** - 30% af strategi-salg
3. **Affiliate program** - provisioner fra broker-referrals
4. **Enterprise licensing** - white-label til hedge funds/wealth managers
5. **Premium data add-ons** - options flow, insider tracking, alternativ data
6. **Educational content** - kurser i algo trading (subscription add-on)

---

## 3. EUROPÆISK HANDEL - TEKNISK LØSNING

### 3.1 Interactive Brokers Integration (Anbefalet)

**Teknisk arkitektur:**

```
Bruger → Alpha Vision API → BrokerRouter
                                ├── IBKR Gateway (TWS API via ib_insync)
                                │   ├── Europæiske børser
                                │   │   ├── Xetra (Tyskland)
                                │   │   ├── Euronext (FR, NL, BE, PT, IR)
                                │   │   ├── LSE (London)
                                │   │   ├── BME (Spanien)
                                │   │   ├── Borsa Italiana
                                │   │   ├── SIX (Schweiz)
                                │   │   ├── OMX (Norden)
                                │   │   └── Wiener Börse
                                │   ├── FX Trading
                                │   └── Options/Futures
                                └── Alpaca (US markeder)
```

**Implementeringsflow:**

1. Bruger opretter IBKR-konto (regulatory krav - vi kan ikke oprette for dem)
2. Bruger genererer API credentials i IBKR
3. Alpha Vision gemmer encrypted credentials
4. Vi forbinder via IB Gateway (headless) eller Client Portal API
5. Orders routes automatisk til korrekt børs baseret på instrument

**Dependencies:**
- `ib_insync` Python library
- IB Gateway eller TWS (kan køre i Docker)
- Market data subscriptions per børs

### 3.2 Regulatoriske Krav

**MiFID II Compliance:**
- Best execution policy (dokumenter ordreudførelse)
- Cost & charges disclosure
- Suitability assessment for komplekse produkter
- Transaction reporting
- Record keeping (5 år)

**GDPR:**
- Data Processing Agreement med brokers
- Right to erasure
- Data portability
- Cookie consent
- Privacy policy

**Licenskrav:**
- Alpha Vision som teknologileverandør behøver IKKE broker-licens
- Vi faciliterer adgang, vi håndterer ikke kunders penge
- Introducing Broker agreement med IBKR er en mulighed for provision
- Alternativt: ren SaaS model uden direkte broker-tilknytning

### 3.3 Børs-specifikke Krav

| Børs | Åbningstid (CET) | Valuta | Ticker Format | Clearing |
|------|-------------------|--------|---------------|----------|
| Xetra | 09:00-17:30 | EUR | ISIN/WKN | Clearstream |
| Euronext | 09:00-17:30 | EUR | ISIN | LCH |
| LSE | 08:00-16:30 | GBP | TIDM/ISIN | LCH |
| OMX Nordic | 09:00-17:30 | DKK/SEK/NOK | ISIN | Euroclear |
| SIX | 09:00-17:30 | CHF | ISIN | SIX SIS |
| BME | 09:00-17:30 | EUR | ISIN | Iberclear |
| Borsa Italiana | 09:00-17:30 | EUR | ISIN | Monte Titoli |

---

## 4. KOMMERCIALISERING - HVORDAN SÆLGER VI PLATFORMEN

### 4.1 Go-To-Market Strategi

**Target Segments:**

1. **Retail Algo Traders** (primær) - Teknisk kyndige investorer der vil automatisere
2. **Active Day Traders** - Ønsker bedre tools end deres broker tilbyder
3. **Quant-nysgerrige** - Vil lære ML/AI trading uden at kode
4. **Small Hedge Funds** - 1-5 personer der mangler infrastruktur
5. **Financial Advisors** - Vil tilbyde algo trading til deres kunder

**Positionering:**
"Alpha Vision - Den intelligente tradingplatform for europæiske investorer. AI-drevne strategier, multi-market handel, og no-code automation."

### 4.2 Pricing Strategy

**Freemium → Premium conversion funnel:**

1. **Gratis tier** med paper trading og basale features → tiltrækker brugere
2. **14-dages Premium trial** med fuld funktionalitet → viser værdi
3. **Conversion trigger**: Når bruger ser profitabel backtest → "Upgrade to trade live"
4. **Annual discount**: €29 × 10 måneder = €290/år (spar €58)

**Benchmark mod konkurrenter:**
- TradingView Premium: $14.95-$59.95/mdr
- QuantConnect: $8-$72/mdr
- NinjaTrader: $99/mdr eller $1,499 lifetime
- MetaTrader: Gratis (broker-subsideret)
- Alpha Vision Premium: €29/mdr - konkurrencedygtigt for EU markedet

### 4.3 Distribution Channels

1. **Website + SEO** - Content marketing om algo trading, europæiske markeder
2. **YouTube** - Strategy tutorials, backtesting demos, platform walkthroughs
3. **Reddit/Discord** - r/algotrading, r/europeanfinance, Discord communities
4. **Twitter/X** - Daglige signaler, market commentary
5. **Product Hunt** launch
6. **Broker partnerships** - IBKR Introducing Broker program
7. **Fintech events** - Nordic Fintech Week, Money2020 Europe
8. **Affiliate program** - Eksisterende finance YouTubers/bloggers

### 4.4 Infrastruktur for Salg

**Tech Stack for Kommercialisering:**
- **Landing page**: Next.js + Vercel
- **Payment**: Stripe Billing
- **Email marketing**: Resend eller SendGrid
- **Analytics**: PostHog (open-source, GDPR-venlig)
- **Customer support**: Intercom eller Crisp
- **Documentation**: Docusaurus
- **Status page**: Betteruptime

---

## 5. UDVIKLINGSPLAN - SPRINT BREAKDOWN

### Sprint 1-2 (Uge 1-4): Foundation

**Mål:** Multi-broker arkitektur + IBKR integration

- [ ] Refactor BrokerRouter abstraction layer
- [ ] Implementer IBKRBroker klasse med ib_insync
- [ ] Multi-currency portfolio tracking
- [ ] Europæisk børs-kalender
- [ ] Unit tests for alle nye broker-metoder
- [ ] Docker setup for IB Gateway

### Sprint 3-4 (Uge 5-8): Auth & Database

**Mål:** Brugerregistrering + PostgreSQL migration

- [ ] PostgreSQL migration fra SQLite
- [ ] User model med auth (Auth0/Clerk integration)
- [ ] Subscription tiers i database
- [ ] API authentication (JWT + API keys)
- [ ] Rate limiting per tier
- [ ] GDPR data management

### Sprint 5-8 (Uge 9-16): Ny Frontend

**Mål:** React/Next.js dashboard

- [ ] Next.js projekt setup med TailwindCSS
- [ ] TradingView Lightweight Charts integration
- [ ] WebSocket real-time data
- [ ] Alle 9 sider (Dashboard, Trading, Strategies, etc.)
- [ ] Responsive design
- [ ] Dark/light theme

### Sprint 9-10 (Uge 17-20): Betalingssystem

**Mål:** Stripe integration + subscription management

- [ ] Stripe Billing integration
- [ ] Subscription checkout flow
- [ ] Feature gating per tier
- [ ] Invoice generation
- [ ] Trial period logic
- [ ] Webhook handling for payment events

### Sprint 11-14 (Uge 21-28): Avancerede Features

**Mål:** Strategy builder + AI Copilot

- [ ] Visual strategy builder (React Flow)
- [ ] AI trading copilot (LLM-baseret)
- [ ] Copy trading system
- [ ] Strategy marketplace
- [ ] Advanced order types
- [ ] Push notifications

### Sprint 15-16 (Uge 29-32): Launch Prep

**Mål:** Polish, security audit, launch

- [ ] Security audit (OWASP top 10)
- [ ] Load testing
- [ ] Landing page + marketing site
- [ ] Documentation
- [ ] Beta test med 50-100 brugere
- [ ] Product Hunt launch
- [ ] App Store + Google Play (React Native wrapper)

---

## 6. UDVIKLINGSPROMPTS

Herunder er præcise prompts til at bygge hver komponent. Brug dem med Claude Code eller tilsvarende AI kodnings-assistent.

---

### PROMPT 1: BrokerRouter & Multi-Broker Arkitektur

```
Du skal refaktorere broker-laget i Alpha Vision trading platformen.

NUVÆRENDE TILSTAND:
- src/broker/base_broker.py - AbstractBaseBroker med buy(), sell(), get_positions(), get_account(), get_order_status(), cancel_order()
- src/broker/alpaca_broker.py - Alpaca integration (kun US markeder)
- src/broker/paper_broker.py - Paper trading simulator
- src/broker/models.py - Order, OrderStatus, AccountInfo, exceptions

OPGAVE:
1. Opret src/broker/broker_router.py - en BrokerRouter klasse der:
   - Tager en dict af broker-instanser keyed på region/type: {"us": AlpacaBroker, "eu": IBKRBroker, "paper": PaperBroker}
   - Router orders til korrekt broker baseret på symbolets børs/region
   - Har en symbol-til-børs mapping (f.eks. AAPL → us, NOVO-B.CO → eu_nordic, ASML.AS → eu_nl)
   - Aggregerer positioner og konto-info fra alle brokers
   - Beregner samlet portfolio-værdi i brugerens basisvaluta med real-time FX
   - Implementerer same interface som BaseBroker så den er drop-in replacement

2. Opret src/broker/ibkr_broker.py - Interactive Brokers integration:
   - Brug ib_insync library (pip install ib_insync)
   - Connect til IB Gateway eller TWS via localhost:4001 (live) eller 4002 (paper)
   - Implementer alle BaseBroker metoder
   - Håndter IBKR-specifikke ordretyper (LMT, MKT, STP, STP_LMT, TRAIL)
   - Map IBKR contract types til vores Order model
   - Håndter europæiske tickers via ISIN eller local ticker + exchange
   - Implementer connection health check og auto-reconnect
   - Rate limiting (max 50 messages/sec til IBKR)
   - Error handling for alle IBKR-specifikke fejl

3. Opret src/broker/exchange_calendar.py:
   - Trading hours for alle europæiske børser (Xetra, Euronext, LSE, OMX, SIX, BME, Borsa Italiana)
   - Helligdage per børs per år
   - is_market_open(exchange, datetime) → bool
   - next_market_open(exchange, datetime) → datetime
   - Timezone-aware (CET, GMT, etc.)

4. Opret src/broker/currency_manager.py:
   - Real-time FX rates via ECB eller IBKR
   - convert(amount, from_currency, to_currency) → float
   - Cache rates med 5-min TTL
   - Historiske FX rates for P&L beregning
   - Understøt: EUR, USD, GBP, DKK, SEK, NOK, CHF, PLN, CZK, HUF

5. Opdater tests/:
   - test_broker_router.py med mocked brokers
   - test_ibkr_broker.py med mocked IB connection
   - test_exchange_calendar.py for alle børser
   - test_currency_manager.py

KRAV:
- Async-compatible (ib_insync er async)
- Graceful degradation: hvis IBKR er nede, vis fejl men lad Alpaca fortsætte
- Logging med loguru
- Type hints overalt
- Docstrings på alle public metoder
```

---

### PROMPT 2: PostgreSQL Migration & User Management

```
Migrer Alpha Vision fra SQLite til PostgreSQL og tilføj bruger-management.

NUVÆRENDE TILSTAND:
- SQLite bruges til: market data cache, indicator snapshots, signal store, news cache, audit log
- Ingen bruger-model, single-user system
- Config i config/settings.py (Pydantic)

OPGAVE:

1. Database Setup:
   - Opret docker-compose.yml med PostgreSQL 16 + Redis (for caching/sessions)
   - Opret src/database/ mappe med:
     - models.py (SQLAlchemy ORM models)
     - migrations/ (Alembic for schema migrations)
     - session.py (async session factory med asyncpg)
     - repositories/ (repository pattern for data access)

2. Data Models (SQLAlchemy):
   - User: id, email, name, hashed_password, subscription_tier, created_at, updated_at, is_active, settings_json
   - Subscription: id, user_id, tier (free/premium/pro/enterprise), stripe_customer_id, stripe_subscription_id, status, current_period_start, current_period_end
   - BrokerConnection: id, user_id, broker_type (alpaca/ibkr/saxo), encrypted_credentials, is_active, last_connected
   - Portfolio: id, user_id, name, base_currency, broker_connection_id
   - Position: id, portfolio_id, symbol, exchange, quantity, avg_cost, currency, opened_at
   - Trade: id, portfolio_id, symbol, side, quantity, price, commission, currency, executed_at, strategy_name
   - Strategy: id, user_id, name, type, config_json, is_active, created_at
   - Signal: id, strategy_id, symbol, signal_type, confidence, created_at
   - Watchlist: id, user_id, name, symbols_json
   - AuditLog: id, user_id, action, details_json, ip_address, created_at

3. Auth Integration (Auth0 eller Clerk):
   - src/auth/auth_provider.py - Abstract auth interface
   - src/auth/clerk_auth.py - Clerk SDK integration
   - JWT validation middleware for FastAPI
   - Role-based access control decorator: @require_tier("premium")
   - API key authentication for programmatic access

4. API Endpoints (FastAPI):
   - POST /auth/register
   - POST /auth/login
   - GET /auth/me
   - GET /portfolio - brugerens porteføljer
   - POST /orders - placer ordre (verificerer tier-adgang)
   - GET /signals - hent signaler (begrænset per tier)
   - GET /strategies - brugerens strategier
   - POST /strategies - opret ny strategi
   - PUT /subscription - opgrader/nedgrader

5. Rate Limiting:
   - Free: 100 requests/time
   - Premium: 1000 requests/time
   - Pro: 10.000 requests/time
   - Enterprise: Unlimited
   - Implementer med Redis + sliding window

KRAV:
- Async overalt (asyncpg, async SQLAlchemy)
- Alembic migrations for alle schema ændringer
- Encrypted credentials (Fernet encryption for broker API keys)
- GDPR: data export endpoint, account deletion med cascade
- Connection pooling (min 5, max 20 connections)
- Health check endpoint for database
```

---

### PROMPT 3: Next.js Frontend

```
Byg en moderne frontend til Alpha Vision trading platformen.

TECH STACK:
- Next.js 15 med App Router
- TypeScript
- TailwindCSS + shadcn/ui komponenter
- TradingView Lightweight Charts (npm: lightweight-charts)
- Zustand for state management
- React Query (TanStack Query) for server state
- Socket.IO client for real-time data
- Clerk for authentication UI

SIDER OG LAYOUTS:

1. Layout:
   - Sidebar navigation (collapsible)
   - Top bar med: søgefelt (symbol search), notifikationer, bruger-menu, subscription badge
   - Dark theme som default, light theme option
   - Farver: #0f1117 (bg), #00d4aa (accent/grøn), #ff4757 (rød), #1a1d29 (cards)

2. Dashboard (/dashboard):
   - Portfolio summary card: total value, day P&L (% og absolut), all-time return
   - Holdings tabel: symbol, shares, avg cost, current price, P&L, weight%
   - Portfolio performance chart (vs S&P 500 benchmark)
   - Watchlist widget med live priser
   - Recent trades liste
   - AI Signals widget (top 3 signals med confidence bars)
   - Market overview: indices (S&P 500, STOXX 50, OMX C25), crypto, forex

3. Trading (/trading):
   - Ordre-panel: symbol input, buy/sell toggle, order type (market/limit/stop), quantity, price
   - Live ordrebog (bid/ask) når tilgængelig
   - Open orders liste med cancel-knap
   - Position details med P&L chart
   - Quick-trade buttons (1-click buy/sell for watchlist)

4. Market Explorer (/markets):
   - Markedsoversigt med heatmap (sektor-baseret, størrelse = market cap, farve = day change)
   - Screener: filtrer på pris, volume, RSI, market cap, sektor, land, exchange
   - Sektor-performance chart
   - Top gainers/losers tabel
   - Economic calendar widget

5. Strategies (/strategies):
   - Mine strategier: liste med performance metrics
   - Strategy detail: config, backtest resultater, live performance, P&L curve
   - Strategy Builder: visual flow editor (brug React Flow library)
   - Template library: færdige strategier man kan klone og tilpasse
   - Backtest runner med parameter inputs og resultat-visning

6. Signals (/signals):
   - Signal feed: kronologisk liste af AI-genererede signaler
   - Per signal: symbol, retning, confidence %, strategi-kilde, tidspunkt
   - Filtrering: per strategi, per marked, per confidence level
   - Signal performance: hit rate, avg return, Sharpe

7. Risk Center (/risk):
   - Portfolio drawdown chart
   - Exposure breakdown: per sektor, per land, per valuta
   - Correlation matrix heatmap
   - VaR (Value at Risk) estimation
   - Risk limits status (grøn/gul/rød indicators)

8. News & Sentiment (/news):
   - Aggregeret nyhedsfeed med sentiment score per artikel
   - Sentiment gauge per symbol (bullish/neutral/bearish)
   - Earnings calendar
   - Economic calendar med impact ratings

9. Settings (/settings):
   - Profil: navn, email, password change
   - Broker Connections: tilføj/fjern Alpaca, IBKR credentials
   - Notifications: email preferences, push notification settings
   - Subscription: nuværende plan, upgrade/downgrade, billing history
   - API Keys: generer/revoke API keys
   - Data & Privacy: export data, delete account

KOMPONENTER:
- PriceChart: TradingView Lightweight Charts wrapper med candlestick, line, area modes
- IndicatorOverlay: SMA, EMA, Bollinger bands som overlays
- OrderForm: smart form med validation, preview, confirmation
- PositionCard: kompakt position visning med mini-chart
- SignalBadge: farvekodede signal indicators
- PerformanceMetrics: Sharpe, drawdown, win rate i grid layout
- CurrencySelector: dropdown med flag-ikoner
- ExchangeBadge: viser hvilken børs et symbol handles på

API INTEGRATION:
- Alle endpoints via React Query hooks: usePortfolio(), useSignals(), useStrategies(), etc.
- WebSocket connection for real-time: priser, ordrestatus, P&L updates
- Optimistic updates for ordre-placering
- Error boundaries med retry logic
- Loading skeletons for alle data-fetching states

KRAV:
- Server-side rendering for SEO (landing pages)
- Client-side for app pages (after auth)
- Responsive: desktop (1200px+), tablet (768px), mobil (375px)
- Lighthouse score > 90
- Accessibility: WCAG 2.1 AA
- i18n ready (dansk, engelsk, svensk, norsk som start)
```

---

### PROMPT 4: Stripe Betalingsintegration

```
Implementer Stripe betalingssystem i Alpha Vision.

BACKEND (FastAPI):

1. src/payments/stripe_service.py:
   - Stripe SDK integration (pip install stripe)
   - create_customer(user) → stripe_customer_id
   - create_subscription(user, tier) → subscription
   - update_subscription(user, new_tier) → subscription (proration)
   - cancel_subscription(user) → cancellation at period end
   - get_billing_portal_url(user) → Stripe Customer Portal URL
   - handle_webhook(payload, signature) → process event

2. Stripe Products/Prices:
   - Product: "Alpha Vision Premium" → Price: €29/mdr, €290/år
   - Product: "Alpha Vision Pro" → Price: €79/mdr, €790/år
   - Product: "Alpha Vision Enterprise" → Custom pricing
   - Metered: "API Calls" → €0.001 per request over limit

3. Webhook Handler (POST /webhooks/stripe):
   - customer.subscription.created → activate tier
   - customer.subscription.updated → update tier
   - customer.subscription.deleted → downgrade to free
   - invoice.payment_succeeded → log payment
   - invoice.payment_failed → send warning email, 3-day grace period
   - customer.subscription.trial_will_end → send conversion email

4. Feature Gating Middleware:
   - @require_tier(minimum_tier="premium")
   - Check user's active subscription before allowing access
   - Graceful degradation: show upgrade prompt, don't error
   - Cache subscription status in Redis (5 min TTL)

5. Trial Logic:
   - 14-dage gratis Premium trial ved registrering
   - Kræver kreditkort (reducerer churn)
   - Email sequence: dag 1, dag 7, dag 12, dag 14
   - Auto-convert til betalt eller downgrade til free

FRONTEND (Next.js):

6. Pricing Page (/pricing):
   - 3-kolonne pricing table (Premium, Pro, Enterprise)
   - Feature comparison matrix
   - FAQ sektion
   - "Start Free Trial" CTA
   - Currency selector (EUR, DKK, SEK, NOK, GBP)

7. Checkout Flow:
   - Stripe Checkout (hosted) for simpel implementering
   - Eller Stripe Elements for embedded checkout
   - Success/cancel redirect pages
   - Subscription management via Stripe Customer Portal

8. In-App Upgrade Prompts:
   - Når bruger rammer feature limit → modal med upgrade option
   - Upgrade banner i sidebar for free users
   - "Unlock this feature" overlay på premium content

KRAV:
- PSD2/SCA compliant (Strong Customer Authentication for EU)
- Webhooks med signature verification
- Idempotency keys på alle Stripe API calls
- Test mode med Stripe test keys under development
- Subscription proration ved mid-cycle tier changes
- Tax handling via Stripe Tax (EU VAT auto-beregning)
```

---

### PROMPT 5: AI Trading Copilot

```
Byg en AI Trading Copilot til Alpha Vision platformen.

ARKITEKTUR:
- Backend: FastAPI endpoint der kalder Claude API (eller OpenAI)
- Frontend: Chat-lignende interface integreret i platformen
- Context: Brugerens portfolio, signaler, markedsdata som kontekst

KOMPONENTER:

1. src/ai/copilot.py - Core Copilot Engine:
   - Chat-baseret interface med konversationshistorik
   - System prompt der inkluderer:
     - Brugerens aktuelle positioner og P&L
     - Dagens top signaler
     - Relevante nyheder og sentiment scores
     - Markedsregime (bull/bear/sideways)
     - Brugerens risk profil og tier
   - Funktioner:
     a. analyze_stock(symbol) → Komplet analyse med technicals, sentiment, fair value estimate
     b. explain_signal(signal_id) → Forklaring af hvorfor et signal blev genereret
     c. suggest_portfolio_rebalance() → Forslag til portfolio ændringer med begrundelse
     d. natural_language_strategy(text) → Konverter "køb NOVO når RSI < 30" til strategy config
     e. market_summary() → Daglig markedsoversigt med key takeaways
     f. risk_assessment() → Vurder porteføljens risiko og foreslå hedging

2. src/ai/strategy_generator.py - NL til Strategy:
   - Parse naturligt sprog til strategi-parametre
   - Eksempler:
     - "Køb europæiske tech-aktier når de falder 10% fra 52-ugers high"
     - "Mean reversion strategi på danske aktier med 2% stop loss"
     - "Momentum strategi der handler de 5 stærkeste STOXX 50 aktier månedligt"
   - Output: Strategy config JSON der kan bruges i backtester
   - Validering: Check at alle parametre er gyldige før execution

3. src/ai/market_report.py - Automatiske Rapporter:
   - Daglig morgenrapport (kl 08:00 CET):
     - Overnight US market (S&P 500, Nasdaq close)
     - Asian market performance
     - European pre-market indicators
     - Key economic data i dag
     - Top AI signals for dagen
   - Ugentlig portefølje-rapport:
     - Ugens P&L breakdown
     - Bedste/værste positioner
     - Strategi-performance sammenligning
     - Foreslåede justeringer
   - Månedlig performance review:
     - Returns vs benchmarks
     - Risk metrics evolution
     - Strategy attribution analysis

4. Frontend Integration:
   - Floating chat button (bottom-right)
   - Slide-out chat panel
   - Quick actions: "Analyser [symbol]", "Forklar signal", "Markedsoversigt"
   - Markdown rendering i chat (charts, tabeller, kode)
   - Context-aware: Hvis bruger ser en aktie, copilot ved hvilken

KRAV:
- Streaming responses (SSE) for bedre UX
- Rate limiting: Free (5/dag), Premium (50/dag), Pro (unlimited)
- Conversation history gemt i database
- Ingen finansiel rådgivning disclaimer i alle responses
- Latency < 3 sekunder for standard queries
- Fallback til regelbaseret analyse hvis LLM er nede
```

---

### PROMPT 6: Visual Strategy Builder

```
Byg en visual, no-code strategy builder til Alpha Vision.

TECH STACK:
- React Flow (@xyflow/react) for node-baseret editor
- Zustand for strategy state management
- shadcn/ui for input forms

ARKITEKTUR:

Node Types:

1. DATA NODES (blå):
   - PriceData: {symbol, timeframe, lookback}
   - Indicator: {type: SMA/EMA/RSI/MACD/BB/etc, params}
   - Volume: {symbol, timeframe}
   - Sentiment: {symbol, source: news/social}
   - MacroData: {indicator: GDP/CPI/employment/etc}

2. CONDITION NODES (gul):
   - Comparison: {operator: >, <, ==, crosses_above, crosses_below}
   - Threshold: {value, type: absolute/percentage}
   - TimeFilter: {days: [Mon-Fri], hours: [09:00-17:00]}
   - CombineConditions: {logic: AND/OR/NOT}

3. ACTION NODES (grøn):
   - Buy: {order_type, quantity_mode: fixed/percentage/risk_based}
   - Sell: {order_type, quantity_mode: all/partial/percentage}
   - StopLoss: {type: fixed/trailing/ATR, value}
   - TakeProfit: {type: fixed/percentage, value}
   - Alert: {channels: email/push/sms}

4. MANAGEMENT NODES (lilla):
   - PositionSizing: {method: fixed/kelly/volatility_scaled, max_pct}
   - RiskLimit: {max_drawdown, max_positions, max_daily_loss}
   - Rebalance: {frequency: daily/weekly/monthly, method}

FLOW:
DataNode → Indicator → Condition → Action
Eksempel: PriceData(AAPL) → RSI(14) → CrossesBelow(30) → Buy(market, 2% of portfolio) + StopLoss(trailing, 3%)

FEATURES:
- Drag-and-drop fra toolbox sidebar
- Connections via drag between node ports
- Real-time validation (rød kant hvis ugyldig konfiguration)
- Mini-backtest: kør hurtigt backtest direkte i builder
- Templates: "Momentum", "Mean Reversion", "Breakout", "Pairs Trading"
- Export som JSON config der kan bruges i backtester
- Import: Konverter eksisterende strategy configs til visual
- Undo/redo (Ctrl+Z/Y)
- Save/load strategier
- Share strategi som link

BACKEND:
- POST /strategies/validate - valider strategy graph
- POST /strategies/compile - konverter graph til executable strategy
- POST /strategies/backtest - kør backtest på kompileret strategi
- StrategyCompiler klasse der konverterer node-graph til Python strategy

UI DETALJER:
- Toolbox sidebar (venstre): kategoriserede nodes man kan drag'e
- Canvas (center): React Flow editor
- Properties panel (højre): konfigurer selected node
- Bottom bar: validation status, "Run Backtest" knap, performance preview
- Dark theme matchende resten af platformen
```

---

### PROMPT 7: Copy Trading & Social Features

```
Implementer copy trading og sociale features i Alpha Vision.

BACKEND:

1. src/social/copy_trading.py:
   - TraderProfile: user_id, display_name, bio, track_record, risk_score, followers_count
   - CopyRelation: follower_id, leader_id, allocation (% af portfolio), max_position_size, active
   - Når leader placer en ordre:
     a. Event emittet til alle followers
     b. For hver follower: beregn proportional størrelse baseret på allocation
     c. Respekter followers risk limits
     d. Placer ordre via followers broker connection
     e. Log copy trade i audit log
   - Follower controls:
     - Pause/resume kopiering
     - Max tab per trade (€)
     - Max dagligt tab
     - Blacklist specifikke symboler
     - Delay (kopiér efter X minutter)

2. src/social/leaderboard.py:
   - Ranking baseret på: return %, Sharpe ratio, max drawdown, consistency
   - Tidshorisonter: 1M, 3M, 6M, 1Y, All-time
   - Filtrer per: strategi-type, marked, risk-niveau
   - Anti-gaming: minimum 3 måneders track record, minimum 20 trades
   - Verified performance (kun live trades, ikke paper)

3. src/social/strategy_marketplace.py:
   - StrategyListing: strategy_id, seller_id, name, description, price, backtest_results
   - Purchase flow: buyer betaler → strategi config kopieres → buyer kan tilpasse
   - Revenue split: 70% til seller, 30% til Alpha Vision
   - Rating system: 1-5 stjerner + text reviews
   - Refund policy: 7 dage hvis strategi performer under backtest results

4. Database Models:
   - TraderProfile (extension af User)
   - CopyRelation (many-to-many)
   - StrategyListing
   - StrategyPurchase
   - Rating
   - SocialFeed (aktivitets-stream)

5. API Endpoints:
   - GET /leaderboard?period=3m&sort=sharpe
   - POST /copy/{leader_id} - start kopiering
   - DELETE /copy/{leader_id} - stop kopiering
   - GET /marketplace - browse strategier
   - POST /marketplace/list - sælg en strategi
   - POST /marketplace/{listing_id}/purchase

FRONTEND:
   - Leaderboard side med trader cards (avatar, stats, follow button)
   - Trader profile page (performance charts, strategy info, followers)
   - Copy settings modal (allocation, limits)
   - Marketplace browse med filtre og søgning
   - Strategy detail page med backtest results og reviews
   - Social feed: "Ole købte NOVO-B", "Strategy X genererede 5% i dag"

KRAV:
- Real-time copy execution (< 1 sekund delay)
- Proportional sizing med rounding til hele aktier
- Audit trail for alle copy trades
- Compliance disclaimer: "Historiske resultater garanterer ikke fremtidige"
- Privacy: Traders kan vælge at skjule specifikke positioner
```

---

### PROMPT 8: Landing Page & Marketing Site

```
Byg en konverterings-optimeret landing page for Alpha Vision.

TECH STACK:
- Next.js 15 (App Router)
- TailwindCSS + Framer Motion for animationer
- Vercel deployment
- Posthog analytics

SIDER:

1. Landing Page (/):
   HERO SECTION:
   - Headline: "Trade Smarter Across European & US Markets"
   - Subheadline: "AI-powered trading signals, automated strategies, and multi-market access. Built for European investors."
   - CTA: "Start Free Trial" (grøn knap) + "Watch Demo" (ghost knap)
   - Hero visual: animeret dashboard screenshot/mockup

   SOCIAL PROOF:
   - "Trusted by X traders across Y countries"
   - Logos af partnere/medier (når tilgængeligt)

   FEATURES SECTION:
   - 6 feature cards med ikoner:
     1. Multi-Market Access (EU + US + Crypto)
     2. AI Trading Signals
     3. No-Code Strategy Builder
     4. Real-Time Risk Management
     5. Copy Top Traders
     6. Danish Tax Reports

   HOW IT WORKS:
   - 3-step illustration: Connect Broker → Configure Strategy → Trade Automatically

   PERFORMANCE SECTION:
   - Backtested strategy performance charts (med disclaimer)
   - Key metrics: avg return, Sharpe ratio, max drawdown

   PRICING SECTION:
   - 3-tier pricing cards (Premium, Pro, Enterprise)
   - Feature comparison toggle
   - Annual/monthly switch med savings badge

   TESTIMONIALS:
   - Trader testimonials med foto, navn, returns (med consent)

   FAQ:
   - Top 10 spørgsmål (accordion)

   FOOTER:
   - Links: Product, Pricing, Blog, Docs, API, Status
   - Legal: Privacy Policy, Terms of Service, Cookie Policy
   - Social: Twitter, Discord, LinkedIn, YouTube
   - Company: About, Contact, Careers

2. Pricing (/pricing):
   - Detaljeret feature comparison tabel
   - FAQ om billing
   - Enterprise kontakt-formular

3. Blog (/blog):
   - MDX-baserede blog posts
   - Kategorier: Trading Strategies, Market Analysis, Platform Updates, Education
   - SEO optimeret med structured data

4. Documentation (/docs):
   - Docusaurus-style docs
   - Getting Started guide
   - API Reference
   - Strategy Builder tutorial
   - FAQ

SEO:
- Meta tags, Open Graph, Twitter Cards
- Structured data (Organization, Product, FAQ)
- Sitemap.xml
- robots.txt
- Target keywords: "algo trading platform europe", "automated trading software", "AI trading signals", "copy trading platform", "European stock trading API"

CONVERSION OPTIMIZATION:
- Exit-intent popup med special offer
- Sticky CTA bar on scroll
- A/B test headline variants
- Analytics events: page_view, cta_click, trial_start, subscription_purchase
```

---

### PROMPT 9: Security & Compliance

```
Implementer security hardening og compliance for Alpha Vision.

OPGAVER:

1. API Security:
   - HTTPS everywhere (TLS 1.3)
   - CORS policy (kun tillad kendte origins)
   - Helmet-lignende security headers
   - Request signing for broker API calls
   - SQL injection prevention (parameterized queries via SQLAlchemy)
   - XSS prevention (content-type headers, CSP)
   - CSRF tokens for state-changing requests
   - Rate limiting med Redis (sliding window)

2. Data Security:
   - Fernet encryption for stored broker credentials
   - At-rest encryption for database (PostgreSQL TDE)
   - Secure key management (AWS KMS eller HashiCorp Vault)
   - Password hashing med bcrypt (via Auth0/Clerk)
   - API keys hashed med SHA-256 (kun vis én gang ved creation)
   - PII anonymisering i logs

3. GDPR Compliance:
   - src/compliance/gdpr.py:
     - export_user_data(user_id) → JSON/CSV med alle brugerdata
     - delete_user_data(user_id) → cascade delete med audit trail
     - anonymize_user(user_id) → erstatte PII med anonyme værdier
   - Cookie consent banner (PostHog har built-in)
   - Privacy Policy tekst (GDPR-compliant)
   - Data Processing Agreement template
   - Data breach notification procedure
   - DPO kontaktinfo

4. MiFID II Compliance (for europæisk handel):
   - Best Execution logging: timestamp, venue, price, slippage
   - Cost & charges disclosure per trade
   - Suitability questionnaire for nye brugere
   - Risk warnings på alle trading-relaterede sider
   - Transaction reporting (hvis krævet som Introducing Broker)
   - Record retention: 5 år for alle trades og kommunikation

5. Monitoring & Alerting:
   - Sentry for error tracking
   - Uptime monitoring (bedre uptime eller similar)
   - Anomaly detection på login attempts (brute force protection)
   - IP-baseret rate limiting for auth endpoints
   - Automated security scanning (dependency audit)

6. Infrastructure:
   - Docker + Docker Compose for alle services
   - Environment-baserede secrets (aldrig i kode)
   - Separate development, staging, production environments
   - Database backups (dagligt, 30 dages retention)
   - Disaster recovery plan

KRAV:
- OWASP Top 10 compliance
- Penetration test før launch
- Security audit checklist
- Incident response procedure
```

---

### PROMPT 10: DevOps & Deployment

```
Opsæt production-grade infrastruktur for Alpha Vision.

ARKITEKTUR:

Cloud Provider: Hetzner (EU-baseret, GDPR-venlig, cost-effective) eller AWS EU (Frankfurt)

Services:
- API Server: FastAPI på Docker, 2+ instanser bag load balancer
- Frontend: Next.js på Vercel (eller self-hosted med Node.js)
- Database: PostgreSQL 16 (managed eller Docker med persistant volumes)
- Cache: Redis 7 (sessions, rate limiting, real-time data)
- Message Queue: Redis Streams eller RabbitMQ (for async task processing)
- IB Gateway: Dockerized, 1 per aktiv bruger-session (eller shared pool)
- Monitoring: Grafana + Prometheus

docker-compose.yml:
- api: FastAPI app
- frontend: Next.js (eller Vercel deployment)
- postgres: PostgreSQL 16 med named volume
- redis: Redis 7
- ib-gateway: Interactive Brokers Gateway
- worker: Celery/ARQ worker for async tasks (backtests, rapporter)
- nginx: Reverse proxy + SSL termination
- prometheus: Metrics collection
- grafana: Dashboard & alerting

CI/CD (GitHub Actions):
- On push to main:
  1. Run tests (pytest)
  2. Run linting (ruff, mypy)
  3. Build Docker images
  4. Push to container registry
  5. Deploy to staging
  6. Run integration tests
  7. Manual approval for production deploy
  8. Deploy to production
  9. Health check
  10. Rollback if health check fails

Environment Config:
- .env.development
- .env.staging
- .env.production
- Secrets i GitHub Secrets / Vault

Scaling Plan:
- 0-100 brugere: Single server (Hetzner CX42, €18/mdr)
- 100-1000: 2 API servers + managed Postgres (€100/mdr)
- 1000-10.000: Kubernetes cluster, read replicas, CDN (€500/mdr)
- 10.000+: Multi-region, auto-scaling (€2000+/mdr)

KRAV:
- Zero-downtime deployments (rolling updates)
- Automated database backups (daily)
- SSL certificates (Let's Encrypt)
- Health check endpoints for alle services
- Structured logging (JSON format) → Grafana Loki
- Alert rules: API latency > 500ms, error rate > 1%, disk > 80%
```

---

## 7. BUDGET ESTIMAT

### Udviklingsomkostninger (med AI-assisteret udvikling)

| Fase | Varighed | Estimat (1-2 udviklere + AI) |
|------|----------|------------------------------|
| Fase 1: EU Handel + Broker | 3 måneder | €15.000-25.000 |
| Fase 2: Frontend + Auth + Payments | 3 måneder | €20.000-35.000 |
| Fase 3: Avancerede Features | 3 måneder | €15.000-25.000 |
| Fase 4: Launch + Marketing | 3 måneder | €10.000-20.000 |
| **Total** | **12 måneder** | **€60.000-105.000** |

### Løbende Driftsomkostninger (per måned)

| Post | Kostnad |
|------|---------|
| Hosting (Hetzner/AWS) | €50-500 |
| PostgreSQL managed | €20-100 |
| Stripe fees (2.9% + €0.25) | Variabel |
| IBKR markedsdata | €30-100 per bruger |
| Auth0/Clerk | €0-300 |
| Email (SendGrid) | €0-50 |
| Monitoring (Sentry) | €0-50 |
| LLM API (Claude/OpenAI) | €50-500 |
| **Total (100 brugere)** | **~€500-1.500/mdr** |

### Break-Even Analyse

Med €29/mdr Premium tier:
- 100 betalende brugere = €2.900/mdr revenue
- 50 brugere dækker løbende drift
- Break-even på udvikling: ~2-3 år med 100 brugere
- Med Pro tier (€79/mdr) og Enterprise: Hurtigere break-even

---

## 8. PRIORITERET HANDLINGSPLAN

### Nu (Denne uge)
1. Beslut broker-partner (IBKR anbefales)
2. Opret IBKR Institutional/Broker account
3. Start PROMPT 1 (Broker Router)

### Måned 1
4. Færdiggør multi-broker arkitektur
5. Start PROMPT 2 (Database + Auth)
6. Test europæisk handel i paper mode

### Måned 2-3
7. Start PROMPT 3 (Frontend)
8. Start PROMPT 4 (Stripe)
9. Alpha version klar til intern test

### Måned 4-6
10. Start PROMPT 5-6 (AI Copilot + Strategy Builder)
11. Closed beta med 20-50 brugere
12. Iterér baseret på feedback

### Måned 7-9
13. Start PROMPT 7 (Social/Copy Trading)
14. Start PROMPT 8 (Landing Page)
15. Start PROMPT 9 (Security)
16. Open beta

### Måned 10-12
17. Start PROMPT 10 (DevOps)
18. Product Hunt launch
19. Marketing push
20. First paying customers

---

*Alpha Vision - Fra sports tech til fintech. Markedsledende AI-drevet trading for Europa.*
