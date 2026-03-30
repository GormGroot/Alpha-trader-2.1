# Alpha Vision — Vision & Kerneprincipper

## To Produkter, Én Kodebase, Hver Sin Database

Alpha Vision er **ét Git repository** der kører i **to modes** på **to maskiner**.
Al kode deles via Git. Hver maskine har sin egen database og config.

```
┌─────────────────────────────────────────────────────────────┐
│                     GIT REPOSITORY                          │
│                 alpha-trading-platform/                      │
│                                                             │
│  src/                                                       │
│  ├── data/          ← Delt: markedsdata, indikatorer       │
│  ├── broker/        ← Delt: alle broker-integrationer       │
│  ├── strategy/      ← Delt: strategier, signal engine       │
│  ├── risk/          ← Delt: risk management                 │
│  ├── tax/           ← Delt: skatteberegning (privat+firma)  │
│  ├── sentiment/     ← Delt: nyheder, FinBERT               │
│  ├── backtest/      ← Delt: backtesting engine              │
│  ├── monitoring/    ← Delt: health, audit, alerts           │
│  ├── notifications/ ← Delt: email rapporter                 │
│  ├── dashboard/     ← Delt: Dash UI (viser mode-relevante  │
│  │                     sider via feature flags)              │
│  ├── research/      ← KUN Research: ML, LLM, learning      │
│  ├── trader/        ← KUN Trader: execution, skatteopt.     │
│  └── main.py        ← Entry point: --mode trader|research   │
│                                                             │
│  config/                                                    │
│  ├── settings.py    ← Delt config-loader                    │
│  ├── trader_config.yaml    ← Oles config                   │
│  └── research_config.yaml  ← Gorms config                  │
└─────────────────────────────────────────────────────────────┘

         │ git pull                          │ git pull
         ▼                                   ▼

┌─────────────────────┐          ┌──────────────────────────┐
│   OLES MASKINE      │          │   GORMS MASKINE          │
│   Alpha Trader      │          │   Alpha Research         │
│                     │          │                          │
│   Normal PC         │          │   10x GTX 2060           │
│   Ingen GPU krav    │          │   64 GB RAM              │
│                     │          │                          │
│   python main.py    │          │   python main.py         │
│     --mode trader   │          │     --mode research      │
│                     │          │                          │
│   ┌───────────────┐ │          │   ┌────────────────────┐ │
│   │ PostgreSQL    │ │          │   │ PostgreSQL +       │ │
│   │ (Oles data)   │ │          │   │ TimescaleDB        │ │
│   │               │ │          │   │ (Gorms data)       │ │
│   │ • Positioner  │ │          │   │                    │ │
│   │ • Handler     │ │          │   │ • Tick data        │ │
│   │ • Skat        │ │          │   │ • ML modeller      │ │
│   │ • Broker cred.│ │          │   │ • Research journal │ │
│   │ • P&L historik│ │          │   │ • Training metrics │ │
│   └───────────────┘ │          │   │ • Knowledge base   │ │
│                     │          │   └────────────────────┘ │
│   Brokers:          │          │                          │
│   • Alpaca (US)     │          │   GPUer:                 │
│   • Saxo (EU ETF)   │          │   • FinBERT (GPU 0)     │
│   • IBKR (EU/Global)│          │   • Mistral LLM (GPU 1) │
│   • Nordnet (Norden) │         │   • Deep Learning (2-7) │
│                     │          │   • Anomaly Det. (8)    │
│   Features:         │          │   • Jupyter (9)         │
│   • Live handel     │          │                          │
│   • Risk management │          │   Features:              │
│   • Selskabsskat    │          │   • Continuous learning  │
│   • Daglige rapporter│         │   • Market narratives    │
│   • 4-broker routing│          │   • RL portfolio mgmt    │
│                     │          │   • Anomaly discovery    │
└─────────────────────┘          └──────────────────────────┘

         │                                   │
         └──────────── FASE 2 ───────────────┘
                  Signal Bridge
            (Research sender signaler
             til Trader via delt fil
             eller simpel API)
```

### ALPHA RESEARCH (Gorms maskine)
Lokal research-platform med 10x GPU. Formål: kontinuerligt øge forståelsen af markederne
gennem deep learning, reinforcement learning, lokal LLM-analyse og autonom strategi-udvikling.
Koden udvikles sammen med Gorm (software ingeniør).

### ALPHA TRADER (Oles maskine)
Handelsværktøj til Oles firma (Alpha Vision). Formål: handle bredt på tværs af alle markeder
og instrumenter (aktier, ETF, råstoffer, crypto, forex, options, futures) via fire brokers
(Alpaca, Saxo Bank, Interactive Brokers, Nordnet). Kører på normal PC uden GPU-krav.
Fokus på execution, risk management og dansk skatteoptimering af firmaets skattetilgodehavende.

---

## Sådan fungerer det i praksis

### Én kodebase, to modes

Al kode lever i samme Git repo. Begge maskiner kloner det samme repository.
Forskellen er **hvilken mode** platformen startes i:

```bash
# På Oles maskine:
python -m src.main --mode trader

# På Gorms maskine:
python -m src.main --mode research
```

Mode bestemmer:
1. **Hvilken config der loades** — `trader_config.yaml` eller `research_config.yaml`
2. **Hvilken database der forbindes til** — hver maskine har sin egen PostgreSQL
3. **Hvilke features der aktiveres** — Trader starter ikke ML pipeline, Research starter ikke skatteberegning
4. **Hvilke dashboard-sider der vises** — Trader ser Tax Center, Research ser DL Lab

### Hver sin database

Databaserne er **helt adskilte**. Ingen deling, ingen synkronisering (endnu).

**Oles database (PostgreSQL):**
- Positioner fra alle 4 brokers
- Handelshistorik (alle trades, alle brokers)
- Skatteberegninger og tilgodehavende
- Broker credentials (encrypted)
- Portfolio snapshots
- Signaler fra eksisterende strategier (SMA, RSI, ML ensemble på CPU)
- Daglige P&L og rapporter

**Gorms database (PostgreSQL + TimescaleDB):**
- Historisk markedsdata (daglig, minut, tick)
- ML model checkpoints og training metrics
- Research journal og market narratives
- Knowledge base (akkumulerede markedsindsigter)
- RL agent performance
- Anomaly detection logs
- Sentiment og nyhedsanalyse historik

### Delt kode via Git

Når Gorm forbedrer en strategi eller fikser en bug:
```bash
# Gorm committer
git add src/strategy/sma_crossover.py
git commit -m "Improve SMA confidence calculation"
git push

# Ole puller ændringen
git pull
# → Oles Trader-version har nu den forbedrede strategi
```

Når Ole tilføjer en ny broker:
```bash
# Ole committer
git add src/broker/saxo_broker.py
git commit -m "Add Saxo Bank integration"
git push

# Gorm puller (kan bruge Saxo til paper trading i Research)
git pull
```

### Signal Bridge (Fase 2 — ikke fra start)

Når Research-maskinen er moden nok, kan den sende signaler til Trader:

```
Gorms maskine                              Oles maskine
┌──────────────┐    signals.json /     ┌──────────────────┐
│ TFT model    │    API / database     │ BrokerRouter     │
│ predicts     │ ──────────────────►   │ modtager signal  │
│ "ASML +3%    │                       │ → risk check     │
│  confidence  │                       │ → placer ordre   │
│  78%"        │                       │ → log til skat   │
└──────────────┘                       └──────────────────┘
```

Implementeringsmuligheder (vælges senere):
- **Delt netværksfil** — Research skriver `signals.json`, Trader poller den
- **Simpel HTTP API** — Research POST'er signaler til Traders endpoint
- **Delt PostgreSQL tabel** — Begge forbinder til en fælles "signals" database
- **Redis pub/sub** — Real-time signal streaming

Vi bygger dette NÅR Research-maskinens modeller har bevist sig i paper trading.

---

## Kerneprincipper

Disse principper gælder for AL kode i projektet. Referér til dem i docstrings og kommentarer.

### Princip 1: Data Først
> Indsaml, berig og forstå så meget markedsdata som muligt.
> Enhver beslutning skal være datadrevet. Ingen gætteri.

Gælder: Alle data-moduler, pipeline, indicators, sentiment.
Konsekvens: Vi cacher aggressivt, vi smider aldrig data væk, vi logger alt.

### Princip 2: Kontinuerlig Læring (RESEARCH)
> Modeller forbedrer sig automatisk over tid uden manuel intervention.
> Platformen bliver klogere hver uge.

Gælder: ML pipeline, training scheduler, walk-forward, drift detection.
Konsekvens: Alle modeller har retrain-schedule. Performance trackes dagligt.

### Princip 3: Dyb Forståelse
> Ikke bare signaler — forstå HVORFOR markeder bevæger sig.
> Kausalitet over korrelation.

Gælder: Narrative engine, causal graph, research journal, knowledge base.
Konsekvens: Enhver trade skal have en forklaring. Modeller skal være fortolkbare.

### Princip 4: Sikkerhed ved Autonomi
> Jo mere autonomt systemet handler, jo stærkere skal safety-mekanismerne være.
> Circuit breakers, drawdown limits og graceful degradation er ikke optional.

Gælder: Risk manager, portfolio tracker, autonomy levels, health monitor.
Konsekvens: Ingen ordre uden risk check. Ingen model-ændring uden validation.
Systemet skal overleve strømafbrydelse, GPU-fejl og internet-udfald.

### Princip 5: Informationsfordel (TRADER)
> Slå markedet ved at forstå nyheder hurtigere, kombinere flere datakilder,
> og handle uden følelser. Alpha Score kombinerer alt til ét signal.
> Fire brokers (Alpaca, Saxo, IBKR, Nordnet) giver adgang til hele verden.

Gælder: Alpha Score Engine, news pipeline, Claude API analyse, BrokerRouter,
morning/evening briefings, alert system, theme tracker.
Konsekvens: Enhver handel er baseret på kvantificeret intelligence, ikke mavefornemmelse.
Multi-currency P&L i DKK. Alle handler logges for skat.

### Princip 6: Skattebevidsthed (TRADER)
> Hver handel har skattemæssige konsekvenser. Platformen skal kende dem.
> Firmaets skattetilgodehavende skal udnyttes intelligent.

Gælder: Tax calculator, transaction log, trade decisions, P&L reporting.
Konsekvens: FIFO-beregning i real-time. Skattemæssig påvirkning vist før execution.
Realisering af tab/gevinst timed til optimal skatteudnyttelse.

### Princip 7: Robusthed over Kompleksitet
> Simpel kode der virker slår smart kode der fejler.
> Graceful degradation over graceful complexity.

Gælder: Alt. Hele kodebasen.
Konsekvens: Fallbacks overalt. Logging overalt. Tests for alt kritisk.
Hvis en broker er nede, handler de andre videre. Hvis en model fejler, brug fallback.

---

## Mappestruktur

```
alpha-trading-platform/
├── src/
│   ├── data/               # DELT: Markedsdata, indikatorer, pipeline, universe
│   │   ├── market_data.py        # ← Eksisterer (yfinance + SQLite cache)
│   │   ├── market_data_v2.py     # → NY: PostgreSQL/TimescaleDB version
│   │   ├── indicators.py         # ← Eksisterer (30+ indikatorer)
│   │   ├── pipeline.py           # ← Eksisterer (data scheduler)
│   │   ├── universe.py           # ← Eksisterer (alle markedsuniverser)
│   │   ├── alternative_data.py   # ← Eksisterer
│   │   ├── macro_indicators.py   # ← Eksisterer
│   │   ├── onchain.py            # ← Eksisterer
│   │   ├── options_flow.py       # ← Eksisterer
│   │   └── insider_tracking.py   # ← Eksisterer
│   │
│   ├── broker/             # DELT: Alle broker-integrationer
│   │   ├── base_broker.py        # ← Eksisterer (abstract interface)
│   │   ├── alpaca_broker.py      # ← Eksisterer (US handel)
│   │   ├── paper_broker.py       # ← Eksisterer (paper trading)
│   │   ├── models.py             # ← Eksisterer (Order, AccountInfo, etc.)
│   │   ├── ibkr_broker.py        # → NY: Interactive Brokers
│   │   ├── saxo_broker.py        # → NY: Saxo Bank
│   │   ├── saxo_auth.py          # → NY: Saxo OAuth2 handler
│   │   ├── nordnet_broker.py     # → NY: Nordnet (uofficiel API)
│   │   ├── nordnet_auth.py       # → NY: Nordnet session handler
│   │   ├── broker_router.py      # → NY: Route ordrer til korrekt broker
│   │   ├── exchange_calendar.py  # → NY: Åbningstider alle børser
│   │   └── currency_manager.py   # → NY: FX rates og conversion
│   │
│   ├── strategy/           # DELT: Alle strategier
│   │   ├── base_strategy.py      # ← Eksisterer
│   │   ├── sma_crossover.py      # ← Eksisterer
│   │   ├── rsi_strategy.py       # ← Eksisterer
│   │   ├── ml_strategy.py        # ← Eksisterer (CPU-baseret)
│   │   ├── ensemble_ml_strategy.py # ← Eksisterer (CPU-baseret)
│   │   ├── combined_strategy.py  # ← Eksisterer
│   │   ├── signal_engine.py      # ← Eksisterer
│   │   ├── market_scanner.py     # ← Eksisterer
│   │   ├── patterns.py           # ← Eksisterer
│   │   └── regime.py             # ← Eksisterer
│   │
│   ├── risk/               # DELT: Risk management
│   │   ├── risk_manager.py       # ← Eksisterer
│   │   ├── portfolio_tracker.py  # ← Eksisterer
│   │   ├── correlation_monitor.py # ← Eksisterer
│   │   ├── dynamic_risk.py       # ← Eksisterer
│   │   └── volatility_scaling.py # ← Eksisterer
│   │
│   ├── tax/                # DELT: Skatteberegning
│   │   ├── tax_calculator.py     # ← Eksisterer (privat 27%/42%)
│   │   ├── corporate_tax.py      # → NY: Selskabsskat 22% lagerbeskatning
│   │   ├── tax_credit_tracker.py # → NY: Skattetilgodehavende tracking
│   │   ├── mark_to_market.py     # → NY: Lagerbeskatning engine
│   │   ├── dividend_tracker.py   # → NY: Udbytte + kildeskat
│   │   ├── currency_pnl.py       # → NY: Valutakurs P&L
│   │   ├── tax_reports.py        # → NY: Årsrapporter til revisor
│   │   ├── tax_report.py         # ← Eksisterer
│   │   ├── tax_advisor.py        # ← Eksisterer
│   │   ├── transaction_log.py    # ← Eksisterer
│   │   └── currency.py           # ← Eksisterer
│   │
│   ├── sentiment/          # DELT: Nyheder og sentiment
│   │   ├── news_fetcher.py       # ← Eksisterer
│   │   ├── sentiment_analyzer.py # ← Eksisterer (FinBERT)
│   │   ├── event_detector.py     # ← Eksisterer
│   │   ├── earnings_tracker.py   # ← Eksisterer
│   │   └── macro_calendar.py     # ← Eksisterer
│   │
│   ├── backtest/           # DELT: Backtesting
│   │   ├── backtester.py         # ← Eksisterer
│   │   ├── comparison.py         # ← Eksisterer
│   │   └── stress_test.py        # ← Eksisterer
│   │
│   ├── monitoring/         # DELT: Health & audit
│   │   ├── health_monitor.py     # ← Eksisterer
│   │   ├── performance_tracker.py # ← Eksisterer
│   │   ├── audit_log.py          # ← Eksisterer
│   │   └── anomaly_detector.py   # ← Eksisterer
│   │
│   ├── notifications/      # DELT: Email alerts
│   │   ├── notifier.py           # ← Eksisterer
│   │   └── trading_notifier.py   # ← Eksisterer
│   │
│   ├── dashboard/          # DELT: Web UI (mode-aware)
│   │   └── app.py                # ← Eksisterer (udvides med feature flags)
│   │
│   ├── research/           # KUN RESEARCH MODE (Gorms maskine)
│   │   ├── ml/                   # → NY: GPU deep learning pipeline
│   │   │   ├── gpu_manager.py
│   │   │   ├── tft_model.py
│   │   │   ├── lstm_ensemble.py
│   │   │   ├── rl_agent.py
│   │   │   ├── trading_env.py
│   │   │   ├── anomaly_detector.py
│   │   │   ├── feature_engine.py
│   │   │   ├── model_registry.py
│   │   │   └── training_pipeline.py
│   │   ├── intelligence/         # → NY: LLM + markedsforståelse
│   │   │   ├── local_llm.py
│   │   │   ├── news_understanding.py
│   │   │   ├── narrative_engine.py
│   │   │   ├── regime_detector.py
│   │   │   ├── research_journal.py
│   │   │   └── causal_graph.py
│   │   └── learning/            # → NY: Continuous learning
│   │       ├── performance_tracker.py
│   │       ├── drift_detector.py
│   │       ├── walk_forward.py
│   │       ├── trade_analyzer.py
│   │       ├── strategy_allocator.py
│   │       ├── knowledge_base.py
│   │       └── meta_report.py
│   │
│   ├── trader/             # KUN TRADER MODE (Oles maskine)
│   │   ├── intelligence/        # → NY: Markedsanalyse & Alpha Score
│   │   │   ├── alpha_score.py         # Samlet 0-100 score per aktie
│   │   │   ├── news_pipeline.py       # Nyheds-aggregering + cross-impact
│   │   │   ├── morning_briefing.py    # AI morgen-briefing (Claude API)
│   │   │   ├── evening_analysis.py    # AI aften-analyse (Claude API)
│   │   │   ├── llm_client.py          # Claude API wrapper
│   │   │   ├── token_tracker.py       # API cost tracking
│   │   │   ├── alert_system.py        # Real-time alerts
│   │   │   ├── watchlist.py           # Dynamisk watchlist
│   │   │   └── theme_tracker.py       # Investerings-tema tracking
│   │   ├── execution/           # → NY: Multi-broker ordrehåndtering
│   │   │   ├── order_manager.py
│   │   │   ├── aggregated_portfolio.py
│   │   │   └── connection_manager.py
│   │   ├── tax_optimizer/       # → NY: Skatteoptimering
│   │   │   └── optimizer.py
│   │   └── reporting/           # → NY: Rapporter til revisor
│   │       ├── daily_report.py
│   │       └── annual_report.py
│   │
│   ├── bridge/             # SIGNAL BRIDGE (Fase 2)
│   │   ├── signal_publisher.py  # → Research sender signaler
│   │   └── signal_consumer.py   # → Trader modtager signaler
│   │
│   └── main.py             # Entry point: --mode trader|research
│
├── config/
│   ├── settings.py               # ← Eksisterer (udvides med mode)
│   ├── default_config.yaml       # ← Eksisterer
│   ├── trader_config.yaml        # → NY: Oles broker credentials, skat
│   └── research_config.yaml      # → NY: Gorms GPU allocation, ML params
│
├── docker/
│   ├── docker-compose.trader.yaml   # → NY: PostgreSQL + Redis
│   └── docker-compose.research.yaml # → NY: PostgreSQL + TimescaleDB + Redis + IB Gateway
│
├── VISION.md               # ← Du er her
├── ALPHA_TRADER_PLAN.md     # Oles handelsplatform plan + prompts
├── ALPHA_VISION_LOCAL_MASTERPLAN.md  # Gorms research platform plan + prompts
│
└── tests/                  # ← 29 eksisterende tests + nye
```

**← Eksisterer** = Allerede bygget og fungerer. Rør det ikke medmindre der er en god grund.
**→ NY** = Skal bygges. Referér til prompts i de respektive plan-dokumenter.

---

## main.py — Mode-baseret Entry Point

```python
"""
Alpha Vision Trading Platform

Kør i trader-mode (Oles maskine):
    python -m src.main --mode trader

Kør i research-mode (Gorms maskine):
    python -m src.main --mode research

Se VISION.md for arkitektur og kerneprincipper.
"""

# Princip 7: Robusthed over Kompleksitet
# main.py loader KUN de moduler der er relevante for den valgte mode.
# Trader-mode importerer aldrig research/ml/*.
# Research-mode importerer aldrig trader/tax_optimizer/*.
# Dette holder startup hurtigt og undgår unødvendige dependencies.
```

---

## Config Eksempler

### trader_config.yaml (Oles maskine)
```yaml
mode: trader

database:
  host: localhost
  port: 5432
  name: alpha_trader
  user: alpha
  password: ${DB_PASSWORD}  # Fra .env

brokers:
  alpaca:
    enabled: true
    base_url: https://paper-api.alpaca.markets  # Skift til live når klar
    api_key: ${ALPACA_API_KEY}
    api_secret: ${ALPACA_API_SECRET}
  saxo:
    enabled: true
    environment: sim  # Skift til live når klar
    app_key: ${SAXO_APP_KEY}
    app_secret: ${SAXO_APP_SECRET}
  ibkr:
    enabled: true
    host: 127.0.0.1
    port: 4002  # Paper: 4002, Live: 4001
    client_id: 1
  nordnet:
    enabled: true
    username: ${NORDNET_USERNAME}
    password: ${NORDNET_PASSWORD}

tax:
  type: corporate  # 'corporate' (selskab) eller 'personal' (privat)
  rate: 0.22
  tax_credit_initial: 0  # Sæt til faktisk beløb i DKK
  year_end: "12-31"
  base_currency: DKK

risk:
  max_position_pct: 5
  max_daily_loss_pct: 3
  max_drawdown_pct: 15
  stop_loss_pct: 5
  max_open_positions: 30

dashboard:
  pages:
    - overview
    - trading
    - markets
    - strategies
    - risk
    - tax_center      # Kun trader mode
    - broker_status   # Kun trader mode

notifications:
  morning_report: "07:30"
  evening_report: "22:30"
  email: ${ALERT_EMAIL}
```

### research_config.yaml (Gorms maskine)
```yaml
mode: research

database:
  host: localhost
  port: 5432
  name: alpha_research
  user: alpha
  password: ${DB_PASSWORD}
  timescaledb: true  # Aktivér TimescaleDB extensions

gpu:
  allocation:
    finbert: 0
    local_llm: 1
    training: [2, 3]
    backtesting: [4, 5]
    reinforcement_learning: [6, 7]
    anomaly_detection: 8
    jupyter: 9

ml:
  retrain_schedule:
    lstm_ensemble: "weekly"
    tft_model: "monthly"
    rl_agent: "quarterly"
  walk_forward:
    train_window: "3Y"
    validation_window: "3M"
    test_window: "1M"

local_llm:
  model: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
  variant: "Q4_K_M"
  gpu_device: 1
  context_length: 4096

brokers:
  alpaca:
    enabled: true
    base_url: https://paper-api.alpaca.markets  # Altid paper for research
    api_key: ${ALPACA_API_KEY}
    api_secret: ${ALPACA_API_SECRET}
  ibkr:
    enabled: true
    host: 127.0.0.1
    port: 4002  # Altid paper for research
    client_id: 1

dashboard:
  pages:
    - overview
    - stock_analysis
    - strategies
    - risk
    - research_lab     # Kun research mode
    - understanding    # Kun research mode
    - dl_lab           # Kun research mode
    - autonomy_control # Kun research mode

notifications:
  daily_narrative: "22:00"
  weekly_meta_report: "sunday 20:00"
  email: ${ALERT_EMAIL}
```

---

## Hvem bygger hvad

| Komponent | Ansvarlig | Prioritet |
|-----------|-----------|-----------|
| broker/ (IBKR, Saxo, Nordnet, Router) | Ole | Kritisk — handle ASAP |
| tax/ (selskabsskat, lagerbeskatning) | Ole | Høj |
| trader/intelligence/* (Alpha Score, Claude API) | Ole | Kritisk — kerne edge |
| trader/execution/* (multi-broker) | Ole | Kritisk |
| trader/reporting/* | Ole | Høj |
| main.py --mode flag | Ole + Gorm | Høj (gør tidligt) |
| config/ (trader + research yaml) | Ole + Gorm | Høj |
| research/ml/* (GPU pipeline) | Gorm | Høj |
| research/intelligence/* (LLM) | Gorm | Medium |
| research/learning/* (continuous) | Gorm | Medium |
| data/ (TimescaleDB migration) | Gorm | Høj |
| dashboard/ (mode-aware sider) | Ole + Gorm | Medium |
| bridge/ (signal sharing) | Ole + Gorm | Fase 2 |
| strategy/* (forbedringer) | Begge | Løbende |
| tests/* | Begge | Løbende |

---

## Første skridt

### Ole (Trader):
1. Tilføj `--mode` flag til `main.py`
2. Opret `trader_config.yaml` med broker credentials + ANTHROPIC_API_KEY
3. Byg Intelligence Engine + Claude API (PROMPT T0 + T0.5) — din kerne-edge
4. Start med Saxo Bank integration (PROMPT T1)
5. Byg BrokerRouter (PROMPT T4)
6. Byg selskabsskat (PROMPT T5)

### Gorm (Research):
1. Pull repo, opret `research_config.yaml`
2. Installér CUDA + PyTorch på GPU-maskinen
3. Start med TimescaleDB migration (PROMPT 1)
4. Byg GPU ML pipeline (PROMPT 2)
5. Byg lokal LLM (PROMPT 3)

### Sammen:
1. Enig om `--mode` convention og config format
2. Sæt Git workflow op (feature branches, PR reviews)
3. Planlæg signal bridge når Research har proven modeller
