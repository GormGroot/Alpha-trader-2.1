# Alpha Vision - Lokal Research & Trading Platform

**Dato:** 17. marts 2026
**Scope:** Privat, lokal platform til databehandling, markedsforståelse og autonom handel
**Hardware:** Windows PC med 10x Nvidia GTX 2060 (60GB samlet VRAM)

---

## NY VISION

Alpha Vision er ikke et SaaS-produkt. Det er et **privat forsknings- og handelsværktøj** der kører lokalt og har ét formål: kontinuerligt at øge din forståelse af markederne og handle intelligent baseret på den forståelse.

Tænk på det som dit personlige Bloomberg Terminal + Renæssance Technologies research lab — men drevet af moderne AI og 10 GPUer.

**Kerneprincipper:**
1. **Data først** — Indsaml, berig og forstå så meget markedsdata som muligt
2. **Kontinuerlig læring** — Modeller der automatisk forbedrer sig over tid
3. **Dyb forståelse** — Ikke bare signaler, men forståelse af *hvorfor* markeder bevæger sig
4. **Autonom handling** — Når forståelsen er tilstrækkelig, handler platformen selv
5. **Europæisk fokus** — Fuld dækning af europæiske markeder med lokal data

---

## 1. HVAD DER ÆNDRER SIG (vs. SaaS-planen)

### Fjernes helt
- Brugerregistrering, auth, multi-tenant
- Stripe betalingssystem, subscription tiers
- Landing page, marketing site
- Copy trading, social features, marketplace
- Rate limiting, API keys for eksterne brugere
- GDPR compliance (ingen brugerdata)
- White-label, mobile app

### Bevares og styrkes
- Multi-broker integration (Alpaca + IBKR) — du vil handle europæisk
- Risk management — endnu vigtigere når den kører autonomt
- Alle data-kilder — udvides massivt
- Backtesting — bliver GPU-accelereret
- Dashboard — Dash er fint til personligt brug, opgraderes løbende
- Email notifikationer — kritisk for autonom drift
- Dansk skat — stadig relevant

### Nyt og centralt
- **GPU-accelereret deep learning pipeline** (10x GTX 2060)
- **Markedsforståelses-engine** — forstå *hvorfor*, ikke bare *hvad*
- **Massiv dataindsamling og -lagring** — years of tick data
- **Continuous learning** — modeller der retrainer automatisk
- **Research notebook integration** — Jupyter for eksperimentering
- **LLM-drevet markedsanalyse** — lokal LLM til nyhedsforståelse
- **Regime detection** — forstå markedsfaser og tilpas strategier

---

## 2. ARKITEKTUR FOR LOKAL PLATFORM

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALPHA VISION LOCAL                            │
│                    Windows 10/11 + WSL2                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ DATA LAYER   │  │ INTELLIGENCE │  │ EXECUTION LAYER       │  │
│  │              │  │ LAYER        │  │                       │  │
│  │ Market Data  │  │ Deep Learning│  │ BrokerRouter          │  │
│  │ News/RSS     │  │ Pipeline     │  │  ├─ Alpaca (US)       │  │
│  │ Sentiment    │  │ (10x GPU)    │  │  ├─ IBKR (EU)        │  │
│  │ Alt Data     │  │              │  │  └─ Paper (Test)      │  │
│  │ Macro/FRED   │  │ Market       │  │                       │  │
│  │ On-chain     │  │ Understanding│  │ Risk Manager          │  │
│  │ Options Flow │  │ Engine       │  │ Portfolio Tracker      │  │
│  │ Insider      │  │              │  │ Order Manager          │  │
│  │ Earnings     │  │ Continuous   │  │                       │  │
│  │              │  │ Learning     │  │ Tax Calculator         │  │
│  └──────┬───────┘  │              │  └───────────┬───────────┘  │
│         │          │ Local LLM    │              │              │
│         │          │ (Mistral/    │              │              │
│         │          │  Llama)      │              │              │
│         │          └──────┬───────┘              │              │
│         │                 │                      │              │
│  ┌──────▼─────────────────▼──────────────────────▼───────────┐  │
│  │                    DATA STORE                              │  │
│  │  PostgreSQL (structured) + TimescaleDB (time-series)       │  │
│  │  + Redis (cache) + Parquet files (historical bulk)         │  │
│  └────────────────────────┬──────────────────────────────────┘  │
│                           │                                     │
│  ┌────────────────────────▼──────────────────────────────────┐  │
│  │                    INTERFACE LAYER                          │  │
│  │  Dash Dashboard │ Jupyter Notebooks │ CLI │ Email Alerts   │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. GPU-STRATEGI: 10x GTX 2060

Hver GTX 2060 har 6GB VRAM = 60GB samlet. Her er hvordan vi udnytter dem:

### GPU Allocation Plan

| GPU | Opgave | VRAM Forbrug |
|-----|--------|--------------|
| GPU 0 | FinBERT Sentiment (real-time) | ~2 GB |
| GPU 1 | Lokal LLM (Mistral 7B quantized) til nyhedsanalyse | ~5 GB |
| GPU 2-3 | Deep Learning Model Training (LSTM, Transformer) | 2x 6 GB |
| GPU 4-5 | Parallel Backtesting (GPU-accelereret) | 2x 6 GB |
| GPU 6-7 | Reinforcement Learning Agent (portfolio optimization) | 2x 6 GB |
| GPU 8 | Feature Engineering & Anomaly Detection | ~4 GB |
| GPU 9 | Reserve / Eksperimentering via Jupyter | 6 GB |

### PyTorch Multi-GPU Setup

```python
# Eksempel: Distribueret training på tværs af GPUer
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

# Specifik GPU allocation
sentiment_device = torch.device("cuda:0")
llm_device = torch.device("cuda:1")
training_devices = [2, 3]  # DataParallel
backtest_devices = [4, 5]
rl_devices = [6, 7]

# Model training på multiple GPUer
model = TransformerPricePredictor()
model = DataParallel(model, device_ids=training_devices)
```

---

## 4. FEATURE ROADMAP — NYT SCOPE

### FASE 1: Datainfrastruktur & EU Handel (Måned 1-2)

#### 1.1 TimescaleDB for Tidsseriedata

SQLite er for langsomt til den datamængde vi skal håndtere. Migration til:

- **PostgreSQL + TimescaleDB extension** — optimeret til tidsseriedata
- **Hypertables** for OHLCV data med automatisk partitionering
- **Continuous aggregates** — pre-beregnede minutdata → timedata → dagdata
- **Compression** — 10-20x komprimering af historisk data
- **Retention policies** — tick data: 1 år, minutdata: 5 år, dagdata: forever

Estimeret datamængde:
- 5.000 symboler × daglig data × 20 år ≈ 5 GB (ukomprimeret)
- 5.000 symboler × minutdata × 5 år ≈ 500 GB (ukomprimeret) → ~50 GB komprimeret
- Tick data (hvis tilgængeligt) ≈ 1-5 TB → ~100-500 GB komprimeret

**Diskanbefaling: 2 TB NVMe SSD** (hurtig adgang) + **4-8 TB HDD** (arkiv)

#### 1.2 Multi-Broker (Alpaca + IBKR)

Samme som SaaS-planen men simplificeret:
- BrokerRouter med Alpaca (US) + IBKR (EU)
- Ingen multi-tenant — direkte credentials i .env
- IB Gateway i Docker via WSL2
- Europæisk markedsdata via IBKR subscriptions

#### 1.3 Massiv Dataindsamling

Udvid data-universet markant:

**Nye datakilder:**
- **Tick data** — IBKR historical ticks for alle handlede symboler
- **Order book snapshots** — Level 2 data (bid/ask depth)
- **Corporate actions** — splits, dividender, mergers (automatisk justering)
- **Insider transactions** — SEC Form 4 / EU MAR meldinger
- **Short interest data** — daglig short float per aktie
- **Fund flows** — ETF inflows/outflows som sentiment proxy
- **Credit spreads** — investment grade vs high yield som risiko-indikator
- **Volatilitetsindekser** — VIX, VSTOXX, VIX term structure
- **Cross-asset correlationer** — aktier vs obligationer vs commodities vs crypto
- **Sæsonmønstre** — "Sell in May", Santa Rally, earnings seasonality

---

### FASE 2: Deep Learning Pipeline (Måned 2-4)

#### 2.1 Transformer-baseret Prisforudsigelse

Den eksisterende ML-pipeline (sklearn + XGBoost) er god men begrænset. Med 10 GPUer kan vi køre langt mere sofistikerede modeller:

**Temporal Fusion Transformer (TFT):**
- State-of-the-art for tidsserie-forecasting
- Håndterer multiple input-typer: statiske (sektor, land), kendte fremtidige (earnings dato), observerede (pris, volume)
- Attention mechanism viser *hvilke* features driver forudsigelsen → forståelse
- Multi-horizon forecasting: 1 dag, 1 uge, 1 måned samtidig

**LSTM Ensemble:**
- 5-10 LSTM modeller med forskellige arkitekturer
- Diversity through: forskellige lookback windows, feature subsets, dropout rates
- Voting eller stacking for final prediction
- Automatisk retraining ugentligt

**Features til Deep Learning:**
- Rå prisdata (OHLCV) normaliseret
- Alle 30+ tekniske indikatorer
- Sentiment scores (FinBERT)
- Makro-indikatorer (FRED)
- Options implied volatility
- Cross-asset returns (korrelationer)
- Calendar features (day of week, month, earnings proximity)
- Market regime (bull/bear/sideways)

#### 2.2 Reinforcement Learning Portfolio Manager

I stedet for regelbaserede strategier: en RL-agent der lærer at optimere porteføljen:

**Deep Reinforcement Learning:**
- **Algoritme:** PPO (Proximal Policy Optimization) eller SAC (Soft Actor-Critic)
- **State:** Porteføljens nuværende positioner, markedsdata, indikatorer, sentiment
- **Actions:** Køb/sælg/hold for hvert symbol, position sizing
- **Reward:** Risk-adjusted return (Sharpe ratio) med drawdown penalty
- **Training:** Simuleret handel over historisk data med transaktionsomkostninger
- **Multi-agent:** Separate agenter for forskellige markedsregimer

**Fordele vs. regelbaseret:**
- Opdager komplekse, ikke-lineære mønstre
- Tilpasser sig automatisk til nye markedsforhold
- Ingen manuelle parametre at tune (agenten finder dem selv)
- Kan lære timing, sizing og risikostyring simultant

#### 2.3 Anomaly Detection & Alpha Discovery

Brug GPU-power til at finde nye trading-muligheder:

- **Autoencoder** for prisadfærd — find aktier der opfører sig "unormalt"
- **Clustering** af pris-mønstre — find gentagne mønstre der leder til profit
- **Feature importance** — hvilke datapunkter forudsiger mest?
- **Regime change detection** — Hidden Markov Model til markedsfaser
- **Cross-asset anomalier** — når korrelationer bryder sammen = mulighed

---

### FASE 3: Markedsforståelses-Engine (Måned 4-6)

#### 3.1 Lokal LLM til Nyhedsanalyse

Kør en lokal LLM (ingen API-omkostninger, ingen databegrænsninger):

**Model:** Mistral 7B Instruct (quantized til 4-bit, ~5GB VRAM på GPU 1)

**Anvendelser:**
- **Nyhedsopsummering:** Konverter 50 artikler til 5 key takeaways
- **Event extraction:** "NOVO NORDISK announced Q4 earnings beat, revenue up 32%"
- **Kausalitetsanalyse:** "Olieprisen steg pga. OPEC production cut → energiaktier up"
- **Risiko-vurdering:** "Fed signalerer hawkish → obligationer ned → tech rotation?"
- **Earnings analyse:** Parse earnings calls og ekstraher guidance, tone, nøgletal

**Pipeline:**
```
News RSS → FinBERT (sentiment score, GPU 0)
         → Mistral (understanding & context, GPU 1)
         → Structured event → Database
         → Impact assessment → Strategi-justering
```

#### 3.2 Market Narrative Engine

Bygger en løbende "forståelse" af markedet:

- **Daglig market narrative:** Hvad skete der og hvorfor?
- **Regime identification:** Er vi i risk-on, risk-off, rotation, panic?
- **Theme tracking:** Hvilke temaer driver markedet? (AI, GLP-1, energy transition, etc.)
- **Divergence detection:** Når pris og fundamentals divergerer → mulighed
- **Macro-micro linking:** Hvordan påvirker makro-data specifikke sektorer/aktier?

Alt gemmes i databasen med timestamps → du kan "spole tilbage" og se hvad platformen tænkte på en given dag.

#### 3.3 Research Journal (Automatisk)

Platformen skriver automatisk en daglig research log:

```
=== 17. marts 2026 ===

MARKEDSREGIME: Risk-on, moderat bullish
DOMINERENDE TEMA: AI infrastructure spending accelererer

OBSERVATIONER:
- STOXX 50 +0.8%, drevet af tech og healthcare
- NOVO-B.CO +2.3% efter analyst upgrade (Barclays → Overweight)
- VIX faldt til 14.2 — complacency rising
- EUR/USD 1.0845, stabil trods ECB tale

MODEL PERFORMANCE:
- Transformer ensemble: 62% accuracy (7-dag forecast)
- RL agent: +0.3% i dag, +4.2% MTD, Sharpe 1.8
- Sentiment model: Korrekt bullish call på ASML

POSITIONER:
- Åbnede: ASML.AS (long), SAP.DE (long)
- Lukkede: EQNR.OL (+3.1% profit)
- Stop-loss hit: Ingen

LÆRING:
- FinBERT overvurderer sentiment på earnings-dage → needs calibration
- LSTM model 3 performer bedst i trending markets → increase weight i bull regime
- Cross-asset signal: credit spreads tightening bekræfter risk-on
```

---

### FASE 4: Continuous Learning & Adaptation (Måned 6-8)

#### 4.1 Walk-Forward Optimization

I stedet for statisk backtesting → rolling retrain:

- **Daglig retraining** af ML-modeller med nyeste data
- **Walk-forward:** Træn på 3 år → test på 1 måned → rul fremad
- **Parameter sweep** på GPU cluster: test tusindvis af parameter-kombinationer parallelt
- **Model selection:** Automatisk vælg den bedste model-kombination per markedsregime
- **Concept drift detection:** Alarm hvis model performance falder under threshold

#### 4.2 Meta-Learning

Platformen lærer af sine egne fejl:

- **Trade journal analyse:** Kategoriser vundne/tabte trades, find mønstre
- **Feature attribution:** Hvilke signals var vigtigst for gode trades?
- **Regime adaptation:** Hvilke strategier virker i hvilke markedsfaser?
- **Drawdown autopsy:** Hvad gik galt under tabsperioder?
- **Confidence calibration:** Er 80% confidence virkelig 80% win rate?

#### 4.3 Adaptive Strategy Weights

I stedet for faste vægte mellem strategier:

```
Markedsregime → Strategy Selector
├── Trending Bull:  60% Momentum, 20% ML, 10% Mean Reversion, 10% RL
├── Trending Bear:  10% Momentum, 30% ML, 40% Hedging, 20% RL
├── Sideways:       20% Momentum, 20% ML, 40% Mean Reversion, 20% RL
├── High Volatility:30% Momentum, 20% ML, 10% Mean Reversion, 40% RL
└── Regime Change:  Reduce all positions, increase cash
```

Vægtene justeres automatisk baseret på rolling performance.

---

### FASE 5: Avanceret Autonom Handel (Måned 8-12)

#### 5.1 Fuld Autonomi med Safety Net

Gradvis overgang fra semi-autonom til fuld autonomi:

**Level 1 (Nuværende):** Platformen genererer signaler, du godkender
**Level 2:** Platformen handler automatisk inden for stramme limits (2% max position, 5% max dag-tab)
**Level 3:** Platformen justerer sine egne risk-limits baseret på regime og confidence
**Level 4:** Fuld autonomi med emergency stop og daily email report

**Safety Mechanisms:**
- **Circuit breaker:** Stop al handel hvis dagligt tab > X%
- **Drawdown protection:** Reducer position sizes progressivt ved drawdown
- **Correlation alert:** Stop hvis alle positioner bevæger sig samme retning
- **Liquidity check:** Handl aldrig i illiqvide markeder
- **News pause:** Stop handel 30 min omkring major economic releases
- **Manual override:** Altid muligt at stoppe via CLI eller email-kommando

#### 5.2 Multi-Timeframe Analyse

Kombiner analyse på tværs af tidshorisonter:

- **Månedlig:** Makro-regime, sektor-rotation, thematic trends
- **Ugentlig:** Momentum, relative strength, fund flows
- **Daglig:** Tekniske signaler, sentiment, earnings
- **Intraday:** Entry/exit timing, volume profil, order flow

**Confluence scoring:** Når alle timeframes er enige = høj confidence

#### 5.3 Europæisk Specialisering

Fokusér på europæiske edge-cases:

- **OMX C25 expertise** — deep model trænet specifikt på danske aktier
- **Nordisk arbitrage** — pris-forskelle mellem nordiske børser
- **European earnings calendar** — timing af europæisk earnings season
- **ECB policy trading** — reaction patterns til ECB beslutninger
- **Cross-listing opportunities** — aktier handlet på multiple europæiske børser
- **Dansk aktiesparekonto optimering** — special skattemæssig strategi

---

## 5. HARDWARE SPECIFIKATION

### Anbefalet Setup

| Komponent | Spec | Formål |
|-----------|------|--------|
| CPU | AMD Ryzen 9 7950X (16-core) eller Intel i9-14900K | Data processing, backtesting |
| RAM | 64 GB DDR5 | Store datasets in memory |
| GPU | 10x Nvidia GTX 2060 6GB | ML training, inference, RL |
| OS Drive | 1 TB NVMe SSD | Windows + WSL2 + Docker |
| Data Drive | 2 TB NVMe SSD | TimescaleDB + aktiv data |
| Archive | 4-8 TB HDD | Historisk tick data, backups |
| PSU | 1600W+ (10 GPUer × 160W TDP) | Strøm til alle GPUer |
| Køling | God airflow / dedikeret rum | 10 GPUer genererer varme |
| Network | Stabil internet (fiber anbefales) | Kontinuerlig datafeed |
| UPS | 1000VA+ | Beskyt mod strømafbrydelser |

### Strømforbrug
- 10x GTX 2060: ~1.600W under fuld belastning
- CPU + RAM + drives: ~200W
- **Total: ~1.800W** under peak, ~800W idle
- Årligt strømforbrug: ~7.000-15.000 kWh ≈ 7.000-15.000 DKK/år

### Software Stack

```
Windows 11 Pro
├── WSL2 (Ubuntu 22.04)
│   ├── Docker Desktop (GPU passthrough)
│   │   ├── PostgreSQL 16 + TimescaleDB
│   │   ├── Redis 7
│   │   ├── IB Gateway
│   │   └── Grafana + Prometheus
│   ├── Python 3.11+ (conda environment)
│   │   ├── PyTorch 2.x (CUDA 12)
│   │   ├── transformers (FinBERT)
│   │   ├── llama-cpp-python (lokal LLM)
│   │   ├── stable-baselines3 (RL)
│   │   ├── ib_insync
│   │   ├── FastAPI + uvicorn
│   │   └── Alle eksisterende dependencies
│   └── JupyterLab (research notebooks)
└── Nvidia Driver 550+ (CUDA 12.x)
```

---

## 6. DISK SPACE BREAKDOWN

| Data Type | Størrelse | Vækst/år | Opbevaring |
|-----------|----------|----------|------------|
| Platform kode | 500 MB | Minimal | NVMe SSD |
| Python env + models | 15 GB | 2 GB | NVMe SSD |
| Docker images | 10 GB | 2 GB | NVMe SSD |
| TimescaleDB (daglig data, 20 år) | 10 GB | 500 MB | NVMe SSD |
| TimescaleDB (minutdata, 5 år) | 50-100 GB | 20 GB | NVMe SSD |
| TimescaleDB (tick data, 1 år) | 200-500 GB | 200 GB | NVMe SSD → HDD |
| Parquet archives (historisk) | 100-500 GB | 50 GB | HDD |
| ML model checkpoints | 20-50 GB | 10 GB | NVMe SSD |
| Jupyter notebooks + research | 5 GB | 2 GB | NVMe SSD |
| Logs + audit trail | 5-10 GB | 5 GB | HDD |
| Backups | 100-200 GB | 50 GB | HDD |
| **Total (år 1)** | **~500 GB - 1.5 TB** | | |
| **Total (år 3)** | **~1.5 - 4 TB** | | |

**Anbefaling: 2 TB NVMe + 8 TB HDD = tilstrækkeligt for 3-5 år**

---

## 7. UDVIKLINGSPROMPTS — LOKALT RESEARCH-FOKUS

---

### PROMPT 1: TimescaleDB Migration & Data Pipeline

```
Migrer Alpha Vision fra SQLite til PostgreSQL + TimescaleDB for tidsseriedata.

NUVÆRENDE TILSTAND:
- src/data/market_data.py bruger SQLite til OHLCV caching
- src/data/pipeline.py scheduler for data fetching
- src/data/indicators.py beregner 30+ indikatorer
- Data hentes via yfinance med rate limiting

OPGAVE:

1. Docker Setup:
   - docker-compose.yml med:
     - PostgreSQL 16 + TimescaleDB extension
     - Redis 7 (for real-time cache)
     - PgAdmin (for database administration)
   - Persistent volumes for data
   - GPU passthrough config for WSL2

2. Database Schema (src/database/):
   - models.py med SQLAlchemy + TimescaleDB:

   Hypertables (tidsserier):
   - ohlcv_daily: time, symbol, open, high, low, close, volume, adj_close
     → Hypertable partitioneret på time (chunk: 1 måned)
     → Continuous aggregate: ohlcv_weekly, ohlcv_monthly
   - ohlcv_minute: time, symbol, open, high, low, close, volume
     → Hypertable (chunk: 1 dag)
     → Compression policy: komprimér data ældre end 7 dage
     → Retention policy: slet data ældre end 5 år (arkivér til Parquet først)
   - ohlcv_tick: time, symbol, price, volume, bid, ask, bid_size, ask_size
     → Hypertable (chunk: 1 time)
     → Compression efter 1 dag
     → Retention: 1 år i DB, arkivér til Parquet

   Regular tables:
   - symbols: id, ticker, name, exchange, currency, sector, industry, country, isin, market_cap
   - exchanges: id, name, timezone, open_time, close_time, holidays_json
   - indicators_cache: time, symbol, indicator_name, value_json (JSONB)
   - sentiment_scores: time, symbol, source, score, headline, article_url
   - macro_data: time, indicator_name, value, source
   - signals: time, symbol, strategy, signal_type, confidence, metadata_json
   - trades: time, symbol, side, quantity, price, commission, currency, strategy, broker
   - portfolio_snapshots: time, positions_json, cash, total_value, currency
   - model_performance: time, model_name, metric_name, value
   - research_log: time, entry_text, market_regime, key_observations_json

3. Data Migration Script:
   - src/database/migrate_sqlite.py
   - Læs alle eksisterende SQLite data
   - Indsæt i TimescaleDB med korrekte timestamps
   - Verificér dataintegritet efter migration

4. Opdateret Market Data Manager:
   - src/data/market_data_v2.py
   - Brug asyncpg for async database operations
   - Batch inserts (1000 rows ad gangen)
   - Upsert logic (INSERT ON CONFLICT UPDATE)
   - Query helpers:
     - get_ohlcv(symbol, start, end, interval) → DataFrame
     - get_latest_price(symbol) → float
     - get_symbols_by_exchange(exchange) → list
     - get_indicator_history(symbol, indicator, period) → Series
   - Parquet export for archiving:
     - archive_to_parquet(table, before_date, output_dir)

5. Enhanced Data Pipeline:
   - src/data/pipeline_v2.py
   - Scheduler med:
     - Hvert minut: Fetch priser for aktive watchlist (via IBKR streaming)
     - Hvert 5. minut: Opdater indikatorer for aktive symboler
     - Hver time: Fetch nyheder og sentiment
     - Dagligt kl 18:00 CET: Fuld daglig data update for alle symboler
     - Ugentligt: Retrain ML modeller
     - Månedligt: Arkivér gammel data til Parquet, database maintenance
   - Dead letter queue for fejlede fetches
   - Dashboard metrics: datamængde, freshness, coverage

KRAV:
- Async overalt (asyncpg, ikke psycopg2)
- Connection pooling (min 5, max 20)
- Graceful shutdown (flush pending writes)
- Schema migrations med Alembic
- Index-strategi: compound index på (symbol, time) for alle hypertables
- Monitoring: query latency, disk usage, compression ratio
```

---

### PROMPT 2: GPU-Accelereret Deep Learning Pipeline

```
Byg en multi-GPU deep learning pipeline til Alpha Vision.

HARDWARE: 10x Nvidia GTX 2060 (6GB VRAM each), CUDA 12.x
FRAMEWORK: PyTorch 2.x

OPGAVE:

1. GPU Manager (src/ml/gpu_manager.py):
   - Discover alle tilgængelige GPUer via torch.cuda
   - Track VRAM usage per GPU
   - Allokér GPUer til tasks:
     GPU 0: FinBERT inference (always-on)
     GPU 1: Local LLM inference (always-on)
     GPU 2-3: Model training (DataParallel)
     GPU 4-5: Parallel backtesting
     GPU 6-7: RL agent training
     GPU 8: Feature engineering / anomaly detection
     GPU 9: Jupyter / eksperimentering
   - Dynamic reallocation: Hvis training er færdig, frigiv GPUer til backtest
   - Health monitoring: temperatur, utilization, memory per GPU
   - Graceful OOM handling: reduce batch size automatically

2. Temporal Fusion Transformer (src/ml/tft_model.py):
   - Implementer TFT fra pytorch-forecasting library ELLER custom implementation
   - Input features:
     Static: sector, country, market_cap_bucket
     Known future: day_of_week, month, earnings_date_distance, options_expiry
     Observed: OHLCV, RSI, MACD, BB, sentiment_score, volume_ratio, macro_indicators
   - Multi-horizon output: 1d, 5d, 20d return forecasts
   - Attention weights → "forklarbarhed" (hvilke features driver prediction)
   - Training config:
     - Lookback: 252 trading days (1 år)
     - Batch size: auto (fit i 6GB VRAM)
     - Learning rate: 1e-3 med cosine annealing
     - Early stopping: patience 10 epochs
     - Validation: 20% holdout
   - Train på GPU 2-3 med DataParallel
   - Save best checkpoint + training metrics

3. LSTM Ensemble (src/ml/lstm_ensemble.py):
   - 5 LSTM varianter:
     a. Vanilla LSTM (2 layers, 128 hidden)
     b. Bidirectional LSTM (2 layers, 64 hidden)
     c. LSTM + Attention (2 layers, 128 hidden, self-attention)
     d. GRU variant (2 layers, 128 hidden)
     e. Conv1D + LSTM hybrid (conv features → LSTM)
   - Diversity through:
     - Forskellige lookback windows: 20, 60, 120, 252 dage
     - Forskellige feature subsets (technical-only, sentiment-included, macro-included)
     - Forskellige dropout rates: 0.1, 0.2, 0.3
   - Ensemble methods:
     - Simple average
     - Weighted average (weights from validation performance)
     - Stacking (meta-learner på top)
   - Auto-retrain: Ugentligt med walk-forward

4. Reinforcement Learning Agent (src/ml/rl_agent.py):
   - Library: stable-baselines3 med PyTorch backend
   - Environment (src/ml/trading_env.py):
     - Gym-compatible environment
     - State: [positions, cash, prices, indicators, sentiment] (flattened)
     - Action space: Continuous [-1, 1] per symbol (-1 = full short, 0 = hold, 1 = full long)
     - Reward: daily_return - 0.5 * max_drawdown_penalty - transaction_costs
     - Episode: 252 trading days
   - Algorithm: PPO (Proximal Policy Optimization)
   - Training:
     - 1M timesteps initial training
     - Curriculum learning: start simpelt (1 aktie), gradvist flere
     - Multiple random seeds for robustness
   - Evaluation: Sharpe ratio, max drawdown, profit factor vs buy-and-hold
   - Train på GPU 6-7

5. Anomaly Detector (src/ml/anomaly_detector.py):
   - Variational Autoencoder (VAE) for prisadfærd:
     - Input: 20-dag prissekvens + volume + indicators
     - Latent space: 16 dimensions
     - Reconstruction error = anomaly score
     - High anomaly → potentiel trading opportunity
   - Isolation Forest (GPU-accelereret via cuML hvis muligt)
   - Run på GPU 8

6. Feature Engineering (src/ml/feature_engine.py):
   - GPU-accelereret feature beregning med cuDF/cuML (RAPIDS):
     - Rolling statistics (mean, std, skew, kurtosis)
     - Cross-correlation features
     - Fourier features (dominant frequencies)
     - Wavelet decomposition
     - PCA dimensionality reduction
   - Feature importance tracking over tid
   - Automatic feature selection (remove low-importance features)

7. Model Registry (src/ml/model_registry.py):
   - Track alle trænede modeller med metadata:
     - Model type, hyperparams, training date, dataset version
     - Validation metrics (accuracy, Sharpe, drawdown)
     - Feature importance rankings
     - Prediction confidence calibration
   - A/B testing: Kør nye modeller parallelt med eksisterende
   - Automatic promotion: Hvis ny model outperformer over 30 dage → promote
   - Rollback: Hvis promoted model underperformer → revert

8. Continuous Training Pipeline (src/ml/training_pipeline.py):
   - Scheduler:
     - Dagligt: Incremental update af alle modeller med ny data
     - Ugentligt: Full retrain af LSTM ensemble
     - Månedligt: Full retrain af TFT + hyperparameter search
     - Kvartalsvist: RL agent full retrain med ny data
   - Walk-forward validation:
     - Train: 3 år rolling window
     - Validate: 3 måneder
     - Test: 1 måned (aldrig brugt til training)
   - Hyperparameter optimization: Optuna med GPU-parallel trials
   - Early stopping + checkpointing for alle training runs

KRAV:
- Mixed precision training (fp16) for hurtigere training og lavere VRAM
- Gradient accumulation for effektiv batch size > VRAM limit
- Model checkpointing hvert 10. epoch
- TensorBoard logging for training visualization
- Reproducibility: Fixed seeds, logged hyperparams
- Fault tolerance: Resume training efter crash
- Memory-mapped datasets for data der er for stor til RAM
```

---

### PROMPT 3: Lokal LLM & Markedsforståelse

```
Implementer en lokal LLM-baseret markedsforståelses-engine.

HARDWARE: GPU 1 (GTX 2060, 6GB VRAM)
MODEL: Mistral 7B Instruct (4-bit quantized, ~5GB VRAM)

OPGAVE:

1. LLM Setup (src/intelligence/local_llm.py):
   - Brug llama-cpp-python med CUDA acceleration
   - Model: TheBloke/Mistral-7B-Instruct-v0.2-GGUF (Q4_K_M variant, ~4.4GB)
   - Konfiguration:
     - n_gpu_layers: -1 (alle layers på GPU)
     - n_ctx: 4096 (context window)
     - temperature: 0.3 (lav for faktuel analyse)
     - max_tokens: 1024
   - Async inference via thread pool
   - Request queue med prioritering (real-time > batch)
   - Fallback: Hvis GPU er optaget → CPU inference (langsomt men muligt)

2. News Understanding Pipeline (src/intelligence/news_understanding.py):
   - Input: Rå nyheder fra news_fetcher.py (headline + summary)
   - Processing pipeline:
     a. FinBERT (GPU 0): Sentiment score (-1 til +1)
     b. Mistral (GPU 1): Structured extraction:
        - Affected companies (tickers)
        - Event type (earnings, M&A, regulation, macro, product, legal)
        - Impact magnitude (low/medium/high)
        - Impact direction (positive/negative/neutral)
        - Time horizon (immediate/short-term/long-term)
        - Causal chain: "X happened → leads to Y → affects Z"
        - Related themes (AI, healthcare, energy, etc.)
     c. Store structured output i database

   Prompt template:
   ```
   Analyze this financial news article and extract structured information.

   HEADLINE: {headline}
   SUMMARY: {summary}
   DATE: {date}

   Respond in JSON format:
   {
     "affected_tickers": ["TICKER1", "TICKER2"],
     "event_type": "earnings|merger|regulation|macro|product|legal|other",
     "impact_magnitude": "low|medium|high",
     "impact_direction": "positive|negative|neutral",
     "time_horizon": "immediate|short_term|long_term",
     "causal_chain": "Event X → Impact Y → Effect on Z",
     "themes": ["theme1", "theme2"],
     "key_numbers": {"metric": value},
     "trading_implication": "Brief actionable insight"
   }
   ```

3. Market Narrative Engine (src/intelligence/narrative_engine.py):
   - Kør dagligt efter markedsluk (17:30 CET for EU, 22:00 CET for US)
   - Aggreger dagens:
     - Alle nyheds-events (structured)
     - Prisændringer per sektor/index
     - Indikator-ændringer (RSI levels, MACD crossovers)
     - Model predictions vs actual
     - Positionsændringer og P&L
   - Send til Mistral med prompt:
   ```
   You are a senior market analyst. Based on today's data, write a concise
   market narrative explaining what happened and why.

   TODAY'S DATA:
   {aggregated_data}

   Write in this format:
   REGIME: [Current market regime]
   NARRATIVE: [2-3 paragraphs explaining today's market action]
   KEY DRIVERS: [Top 3 factors moving markets]
   THEMES: [Active investment themes]
   RISKS: [Top risks to watch]
   OPPORTUNITIES: [Potential opportunities identified]
   LEARNING: [What our models got right/wrong and why]
   ```
   - Gem narrative i database med timestamp
   - Send som del af daily email report

4. Regime Detector (src/intelligence/regime_detector.py):
   - Kombiner multiple signals til regime classification:
     - VIX level og trend
     - Credit spreads (IG vs HY)
     - Breadth indicators (% aktier over 200 SMA)
     - Sector rotation pattern
     - Cross-asset correlations
     - Fund flow direction
     - Sentiment aggregate
   - Regimer: RISK_ON, RISK_OFF, ROTATION, PANIC, EUPHORIA, TRANSITION
   - Confidence score per regime
   - Transition detection: Alarm når regime skifter
   - Historisk regime log → korrelér med strategi-performance

5. Research Journal (src/intelligence/research_journal.py):
   - Automatisk daglig research entry genereret af Mistral
   - Inkluderer:
     - Markedsnarrative (fra narrative engine)
     - Model performance metrics
     - Alle åbnede/lukkede positioner med begrundelse
     - Anomalier opdaget af anomaly detector
     - Ny data observationer (earnings surprises, insider trades, etc.)
   - Searchable via full-text search i PostgreSQL
   - Ugentlig og månedlig summary (aggregeret af Mistral)
   - Format: Markdown gemt som tekst i database + .md fil export

6. Causal Graph (src/intelligence/causal_graph.py):
   - Byg en graf af kausaliteter over tid:
     - "ECB rate hike" → affects → "EUR/USD" → affects → "European exporters"
     - "Oil price spike" → affects → "Energy sector" → affects → "Transport costs"
   - NetworkX graph gemt i database
   - Mistral opdaterer grafen dagligt baseret på nye events
   - Query: "Hvad påvirker NOVO-B?" → Follow graph → Return causal chain
   - Visualisering i dashboard (D3.js eller Plotly network graph)

KRAV:
- Inference latency < 5 sekunder per request
- Batch processing for daglige analyser (ikke real-time)
- Structured JSON output parsing med validation
- Retry logic hvis LLM output er malformed
- Token usage tracking (for at optimere prompts)
- A/B test: Sammenlign Mistral output med FinBERT-only baseline
```

---

### PROMPT 4: IBKR Integration for Europæisk Handel

```
Implementer Interactive Brokers integration til europæisk aktiehandel.

NUVÆRENDE TILSTAND:
- src/broker/base_broker.py - AbstractBaseBroker interface
- src/broker/alpaca_broker.py - US handel via Alpaca
- src/broker/paper_broker.py - Paper trading
- src/broker/models.py - Order, AccountInfo, etc.

DETTE ER EN LOKAL SINGLE-USER PLATFORM.
Ingen auth, ingen multi-tenant. Credentials direkte i .env fil.

OPGAVE:

1. IB Gateway Docker Setup:
   - Dockerfile for IB Gateway (headless mode)
   - docker-compose service med:
     - Port 4001 (live) og 4002 (paper)
     - Volume for IB Gateway config
     - Auto-restart on crash
     - Health check
   - Setup script der konfigurerer IB Gateway med credentials fra .env:
     IBKR_USERNAME=xxx
     IBKR_PASSWORD=xxx
     IBKR_TRADING_MODE=paper  # eller live

2. IBKRBroker (src/broker/ibkr_broker.py):
   - Brug ib_insync library (pip install ib_insync)
   - Implementer BaseBroker interface:

   connect():
     - Connect til IB Gateway via localhost:4002 (paper default)
     - Client ID: 1 (single user)
     - Auto-reconnect med exponential backoff
     - Verify connection med account info request

   buy(symbol, quantity, order_type, limit_price):
     - Map symbol til IBKR Contract:
       - Europæiske aktier: Stock(symbol, exchange, currency)
       - Exchange mapping:
         .CO → "CSE" (Copenhagen)
         .ST → "SFB" (Stockholm)
         .OL → "OSE" (Oslo)
         .HE → "HEX" (Helsinki)
         .DE → "IBIS" (Xetra)
         .PA → "SBF" (Euronext Paris)
         .AS → "AEB" (Euronext Amsterdam)
         .MI → "BVME" (Milan)
         .L → "LSE" (London)
         .SW → "EBS" (SIX Swiss)
         .MC → "BM" (Madrid)
       - US aktier: Stock(symbol, "SMART", "USD")
     - Create IBKR Order objekt (MarketOrder, LimitOrder, StopOrder, TrailingStopOrder)
     - Place order via ib.placeOrder()
     - Return mapped Order objekt
     - Event handling: orderStatusEvent, fillEvent

   sell(symbol, quantity, order_type, limit_price):
     - Samme som buy men med "SELL" action
     - Support for selling fractional positions

   get_positions():
     - ib.positions() → map til vores Position model
     - Inkluder unrealized P&L
     - Konverter valuta til base currency (DKK)

   get_account():
     - ib.accountSummary() → AccountInfo
     - Map: NetLiquidation, TotalCashValue, BuyingPower
     - Multi-currency: vis per valuta + samlet i DKK

   get_order_status(order_id):
     - ib.openOrders() og ib.trades()
     - Map IBKR status → vores OrderStatus

   cancel_order(order_id):
     - ib.cancelOrder(order)

   get_historical_data(symbol, duration, bar_size):
     - ib.reqHistoricalData() for høj-kvalitets data
     - Duration: "1 Y", "6 M", "1 W", "1 D"
     - Bar size: "1 day", "1 hour", "5 mins", "1 min"
     - Return som DataFrame

   stream_market_data(symbols, callback):
     - ib.reqMktData() for live streaming
     - Callback med (symbol, price, volume, bid, ask)
     - Brugt til real-time dashboard opdateringer

3. BrokerRouter (src/broker/broker_router.py):
   - Tager config: {"us": AlpacaBroker, "eu": IBKRBroker, "paper": PaperBroker}
   - Route logic:
     - Tjek symbol suffix (.CO, .DE, etc.) → IBKR
     - Rene US tickers (AAPL, MSFT) → Alpaca
     - Crypto (BTC-USD etc.) → Alpaca
     - Configurable override i settings
   - Aggregeret portfolio:
     - get_all_positions() → kombineret fra alle brokers
     - get_total_value(base_currency="DKK") → samlet værdi med FX conversion
   - Health check: Status for hver broker connection

4. Exchange Calendar (src/broker/exchange_calendar.py):
   - Trading hours og helligdage for alle europæiske børser
   - Data source: exchange-calendars Python library (pip install exchange-calendars)
   - is_open(exchange, datetime) → bool
   - next_open(exchange) → datetime
   - Bruges af pipeline scheduler til at vide hvornår der skal fetches data

5. Currency Manager (src/broker/currency_manager.py):
   - Real-time FX via IBKR (ib.reqMktData for forex pairs)
   - Fallback: ECB reference rates (gratis API)
   - convert(amount, from_ccy, to_ccy) → float
   - Cache med 5-min TTL i Redis
   - Historisk FX for P&L beregning
   - Currencies: EUR, USD, GBP, DKK, SEK, NOK, CHF

6. Tests:
   - test_ibkr_broker.py (med mocked ib_insync)
   - test_broker_router.py
   - test_exchange_calendar.py
   - test_currency_manager.py

KRAV:
- Async-compatible (ib_insync supports asyncio)
- Rate limiting: Max 50 messages/sec til IBKR
- Error handling: Connection lost, order rejected, insufficient funds
- Logging med loguru: Alle ordrer, fills, errors
- Config i .env: IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_TRADING_MODE
```

---

### PROMPT 5: Continuous Learning & Meta-Analysis

```
Byg et continuous learning system der automatisk forbedrer Alpha Visions modeller.

FORMÅL: Platformen skal blive klogere over tid uden manuel intervention.

OPGAVE:

1. Performance Tracker (src/learning/performance_tracker.py):
   - Track for HVER model, HVER dag:
     - Predictions vs actual outcomes
     - Accuracy, precision, recall per signal type
     - P&L attribution (hvor meget tjente/tabte denne model)
     - Feature importance changes over tid
     - Confidence calibration (er 80% confidence = 80% win rate?)
   - Gem i model_performance tabel
   - Rolling metrics: 7d, 30d, 90d, 365d

2. Concept Drift Detector (src/learning/drift_detector.py):
   - Detect når markedsforhold ændrer sig og modeller degraderer:
   - Metoder:
     - Page-Hinkley test på prediction error
     - Population Stability Index (PSI) på feature distributions
     - ADWIN (Adaptive Windowing) på accuracy
   - Triggers:
     - DRIFT_WARNING: Accuracy faldet 5% over 30 dage → log
     - DRIFT_DETECTED: Accuracy faldet 10% → trigger retrain
     - REGIME_CHANGE: Fundamental shift → alarm + fuld model review
   - Actions:
     - Automatisk retrain med nyeste data
     - Forøg learning rate midlertidigt
     - Skift model weights i ensemble

3. Walk-Forward Optimizer (src/learning/walk_forward.py):
   - Continuous walk-forward validation:
     - Hvert weekend: Træn alle modeller med seneste data
     - Window: Ekspanderende (brug al tilgængelig data) eller rolling (3 år)
     - Validation: Seneste 3 måneder
     - Test: Næste uge (aldrig set af model)
   - Hyperparameter search:
     - Optuna med GPU-parallel trials (brug GPU 4-5 når ikke brugt til backtest)
     - Search spaces defineret per model
     - Pruning af dårlige trials (early stopping)
   - Model selection:
     - Kør alle modeller parallelt
     - Rank efter risk-adjusted return (Sharpe)
     - Automatic model promotion/demotion

4. Trade Analyzer (src/learning/trade_analyzer.py):
   - Post-mortem analyse af ALLE trades:
   - Per trade gemmes:
     - Entry/exit signal source (hvilken model/strategi)
     - Market conditions ved entry (regime, VIX, breadth)
     - Technical levels (RSI, MACD state, BB position)
     - Sentiment ved entry
     - Holding period
     - Max favorable excursion (MFE)
     - Max adverse excursion (MAE)
     - P&L og return
   - Pattern mining:
     - "Trades der fejler oftest har: RSI > 60, VIX rising, negative sentiment"
     - "Bedste trades har: Regime=RISK_ON, Multiple strategy agreement, Volume confirmation"
   - Use Mistral (GPU 1) til at generere naturlig-sprog forklaring af mønstre
   - Ugentlig trade review i research journal

5. Adaptive Strategy Allocator (src/learning/strategy_allocator.py):
   - Dynamisk justering af strategi-vægte baseret på performance:
   - Input: Rolling performance per strategi per regime
   - Output: Optimal vægt-allokering
   - Metoder:
     - Regime-conditional weights (lookup tabel der opdateres)
     - Online learning: Exponentially weighted moving average af returns
     - Meta-RL: En mini RL-agent der vælger strategy mix
   - Constraints:
     - Min 5% vægt per aktiv strategi (for diversity)
     - Max 50% vægt per strategi (ingen single-point failure)
     - Sum = 100%
   - Transition smoothing: Skift gradvist over 5 dage (ikke abrupt)

6. Knowledge Base (src/learning/knowledge_base.py):
   - Akkumuleret viden der persisterer:
   - Entries:
     - "NOVO-B.CO reagerer kraftigt på GLP-1 nyheder (avg +/- 3%)"
     - "ECB rate decisions påvirker EUR/DKK med 24h delay"
     - "OMX C25 har tendens til at følge S&P 500 med 1 dags lag"
     - "Earnings season Q4 (jan-feb) har højere volatilitet for nordiske aktier"
   - Auto-discovery: Mistral analyserer trade patterns og foreslår nye entries
   - Bruges af strategi-engine som ekstra input
   - Searchable med full-text search
   - Visualization: Knowledge graph i dashboard

7. Monthly Meta-Report (src/learning/meta_report.py):
   - Automatisk månedlig rapport:
     - Total P&L og return
     - Model performance ranking
     - Drift events og responses
     - Knowledge base nye entries
     - Strategy allocation changes
     - Hardware utilization (GPU, disk, network)
     - Forslag til forbedringer (genereret af Mistral)
   - Sendt via email + gemt som markdown i research_journal

KRAV:
- Alt kører automatisk via scheduler (ingen manuel intervention)
- Database-backed: Alt logges for fremtidig analyse
- Graceful degradation: Hvis en model fejler, brug fallback
- Alerts: Email ved drift detection, model promotion, regime change
- Reproducerbarhed: Alle training runs kan reproduceres fra logged config
```

---

### PROMPT 6: Opgraderet Dashboard for Research

```
Opgrader Dash dashboardet til et research-fokuseret interface.

NUVÆRENDE TILSTAND:
- src/dashboard/app.py med 4 sider (Overview, Stock, Strategies, Risk)
- Dash + Plotly + Bootstrap
- Port 8050, dark theme

BEHOLD Dash/Plotly (det er fint til personligt brug). Tilføj nye sider og widgets.

NYE SIDER:

1. Research Lab (/research):
   - Research Journal viewer:
     - Daglige entries med search og filtrering
     - Markdown rendering
     - Calendar view (klik på en dag → se den dags entry)
   - Model Performance Dashboard:
     - Accuracy over tid per model (line chart)
     - Confusion matrix per model
     - Feature importance bar chart (top 20 features)
     - Confidence calibration plot (predicted vs actual probability)
   - Knowledge Base Browser:
     - Searchable liste af akkumulerede insights
     - Graph visualization af causal connections
     - Timeline view: hvornår blev hver insight opdaget

2. Market Understanding (/understanding):
   - Regime Indicator:
     - Current regime med confidence (stort display)
     - Regime history chart (stacked area over tid)
     - Per-regime strategi performance
   - Market Narrative:
     - Dagens AI-genererede narrative
     - Key themes med styrke-indikator
     - News feed med sentiment scores og event tags
   - Cross-Asset Dashboard:
     - Correlation matrix heatmap (aktier, bonds, FX, commodities, crypto)
     - Correlation change detector (hvad divergerer?)
     - Sector rotation chart
     - Breadth indicators (advance/decline, % over 200 SMA)

3. Deep Learning Lab (/dl-lab):
   - Training Monitor:
     - Live training loss/accuracy kurver (via TensorBoard embed eller Plotly)
     - GPU utilization per GPU (temperatur, VRAM, compute %)
     - Training queue (planlagte og igangværende training jobs)
   - Model Comparison:
     - Side-by-side performance charts
     - A/B test results
     - Drift detection status per model
   - Backtest Results:
     - Equity curves for multiple strategier
     - Drawdown comparison
     - Trade distribution (histogram af returns)
     - Monte Carlo simulation (confidence intervals)

4. European Markets (/europe):
   - Heatmap: Alle europæiske markeder (farve = daglig ændring)
   - Per-børs oversigt: OMX C25, DAX, STOXX 50, FTSE 100
   - Currency tracker: EUR, GBP, SEK, NOK, DKK, CHF
   - ECB tracker: Næste møde, rate expectations, recent speeches
   - Nordic Focus:
     - C25 constituents med signaler
     - Danske aktier sentiment
     - Insider trades Danmark

5. Autonomy Control (/control):
   - Master switch: ON/OFF for autonom handel
   - Autonomy level selector (1-4)
   - Live positions med real-time P&L
   - Open orders med cancel-knapper
   - Risk dashboard:
     - Current drawdown vs limit (gauge chart)
     - Position exposure per sektor/land/valuta (pie charts)
     - Daily P&L curve
     - Circuit breaker status (armed/triggered)
   - Emergency stop button (stort og rødt)
   - Activity log: Alle handler, model decisions, alerts (real-time stream)

OPGRADEREDE EKSISTERENDE SIDER:

6. Overview (opdateret):
   - Multi-broker portfolio (Alpaca + IBKR combined)
   - Valuta-breakdown (DKK, USD, EUR, etc.)
   - Performance vs multiple benchmarks (S&P 500, STOXX 50, OMX C25)
   - AI Signal feed med confidence bars

7. Stock Analysis (opdateret):
   - Tilføj: ML prediction overlay (TFT forecast med confidence bands)
   - Tilføj: Sentiment timeline per aktie
   - Tilføj: Anomaly score indicator
   - Tilføj: RL agent's current view (buy/sell/hold probability)
   - Tilføj: Related causal events fra knowledge base

KRAV:
- Auto-refresh: Positions og P&L hvert 10 sekund
- Responsive callbacks (ingen blocking)
- Caching af dyre queries (Redis)
- Alle charts eksportérbare som PNG
- Keyboard shortcuts: Ctrl+1-9 for sider
```

---

### PROMPT 7: Automated Backup & Reliability

```
Implementer backup, recovery og reliability for en lokal platform
der kører autonomt og handler med rigtige penge.

DETTE ER KRITISK — platformen skal kunne overleve:
- Strømafbrydelse
- GPU/hardware fejl
- Internet udfald
- Windows opdatering der genstarter
- Database korruption

OPGAVE:

1. Backup System (src/ops/backup_manager.py):
   - Daglig backup kl 23:00:
     - PostgreSQL pg_dump (komprimeret)
     - Alle config filer
     - Model checkpoints (seneste per model)
     - Research journal export
   - Gem til:
     - Lokal: backup/ mappe på HDD
     - Ekstern: USB drive eller netværksdrev (konfigurerbar)
   - Retention: 30 daglige, 12 månedlige, 5 årlige backups
   - Verificér backup integritet (pg_restore --list)
   - Alert hvis backup fejler

2. Auto-Recovery (src/ops/recovery_manager.py):
   - Ved opstart: Tjek systemstatus
     - Er database tilgængelig?
     - Er alle GPU'er detected?
     - Er IB Gateway connected?
     - Er internet tilgængeligt?
   - Recovery scenarios:
     a. Clean shutdown → Normal start
     b. Crash recovery → Check open orders, sync positions med broker
     c. Database korruption → Restore fra seneste backup
     d. Internet udfald → Pause trading, continue data caching, retry connection
     e. GPU failure → Redistribute tasks to remaining GPUs
     f. IB Gateway disconnected → Reconnect med exponential backoff

3. Windows Service / Autostart:
   - Script der kører Alpha Vision som Windows service via NSSM eller Task Scheduler
   - Auto-start efter reboot
   - Auto-restart efter crash (max 3 attempts per time)
   - Startup sequence:
     1. Start Docker containers (PostgreSQL, Redis, IB Gateway)
     2. Wait for health checks
     3. Start data pipeline
     4. Start trading engine
     5. Start dashboard
     6. Send "System Online" email

4. Health Dashboard Widget:
   - I Dash dashboard (/control):
     - System uptime
     - Per-component status (grøn/gul/rød)
     - Disk space remaining
     - GPU temperatures og utilization
     - Last backup status og timestamp
     - Network connectivity status
     - Open positions sync status (lokal vs broker)

5. Graceful Degradation Plan:
   - Hvis internet nede:
     → Stop ny-ordre placering
     → Bevar eksisterende stop-loss/take-profit (de er hos broker)
     → Cache data lokalt, sync når internet vender tilbage
   - Hvis GPU fejler:
     → Omfordel tasks (sentiment → CPU fallback, models → reduced ensemble)
     → Alert via email
   - Hvis disk fuld:
     → Stop data indsamling
     → Arkiver + komprimér gamle data
     → Alert
   - Hvis broker disconnected:
     → Retry connection
     → Tighten risk limits midlertidigt
     → Alert

6. Monitoring & Alerting:
   - Prometheus metrics for alle services
   - Grafana dashboards for system health
   - Email alerts for:
     - Komponent nede > 5 minutter
     - Drawdown > threshold
     - Model drift detected
     - Backup failed
     - Disk > 80% fuld
     - GPU temp > 85°C

KRAV:
- Idempotent recovery (kan køres flere gange uden side effects)
- Atomic backup (consistent database snapshot)
- Max 5 minutter fra strømafbrydelse til fuld operation
- Log alt i audit_log for post-mortem analyse
```

---

## 8. PRIORITERET HANDLINGSPLAN

### Uge 1-2: Foundation
1. Installér WSL2 + Docker Desktop + CUDA drivers
2. Kør PROMPT 1 (TimescaleDB migration)
3. Verificér alle 10 GPUer er tilgængelige i PyTorch

### Uge 3-4: Europæisk Handel
4. Kør PROMPT 4 (IBKR integration)
5. Opret IBKR paper trading konto
6. Test handel med europæiske aktier i paper mode

### Uge 5-8: Deep Learning
7. Kør PROMPT 2 (GPU ML pipeline)
8. Træn første TFT og LSTM modeller
9. Baseline performance metrics

### Uge 9-12: Intelligence
10. Kør PROMPT 3 (Lokal LLM + markedsforståelse)
11. Daglige market narratives kørende
12. Research journal populeret

### Uge 13-16: Learning & Dashboard
13. Kør PROMPT 5 (Continuous learning)
14. Kør PROMPT 6 (Opgraderet dashboard)
15. Alt kørende i integration

### Uge 17-20: Reliability & Go-Live
16. Kør PROMPT 7 (Backup & reliability)
17. 4 ugers paper trading med fuld pipeline
18. Gradvis overgang til live trading (Level 1 → 2 → 3)

---

## 9. BUDGET (Lokal Platform)

| Post | Engangs | Løbende/år |
|------|---------|------------|
| Hardware (har allerede GPUer) | €0 | - |
| 2 TB NVMe SSD | €150-200 | - |
| 8 TB HDD | €150-200 | - |
| UPS 1000VA | €200-300 | - |
| IBKR konto (min. deposit) | €0 (ingen min.) | - |
| IBKR markedsdata (EU) | - | €300-600/år |
| Strøm (10 GPUer) | - | €7.000-15.000/år |
| Finnhub API (premium, optional) | - | €0-600/år |
| FRED API | - | €0 (gratis) |
| **Total** | **€500-700** | **€7.300-16.200/år** |

Ingen SaaS fees. Ingen subscription costs. Alt kører lokalt.
Den største løbende udgift er strøm til 10 GPUer.

**Tip:** Kør kun alle 10 GPUer under training. I normal drift bruges 2-3 GPUer (sentiment + LLM + inference), resten kan være idle → ~€3.000-5.000/år i strøm.

---

*Alpha Vision — Dit private market intelligence lab.*
