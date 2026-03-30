# Alpha Trader 2.1 — Komplet Code Review Rapport

**Dato:** 30. marts 2026
**Reviewer:** Claude (Opus 4.6)
**Repo:** github.com/GormGroot/Alpha-trader-2.1
**Commit:** da8b314 (main)

---

## Opsummering

| Svaerhedsgrad | Antal |
|---------------|-------|
| **KRITISK**   | 15    |
| **HOEJ**      | 24    |
| **MEDIUM**    | 37    |
| **LAV**       | 29    |
| **Total**     | **105** |

**Tests:** 226 fejlede / 1432 bestaaet / 12 warnings

---

## TOP 15 KRITISKE FUND (FIX FOERST)

### K-1. `nordnet_broker.py` linje 270 — sell() mangler `short` parameter
**Problem:** `sell()` mangler parameteren `short: bool = False` som `BaseBroker` definerer. Naar `BrokerRouter.sell()` kalder `broker.sell(..., short=True)`, crasher det med `TypeError`. Short-sell via Nordnet er brudt.
**Fix:** Tilfoej `short: bool = False` til signaturen.

### K-2. `paper_broker.py` linje 195 — NoneType crash i limit-ordre check
**Problem:** `price <= order.limit_price` crasher med `TypeError` naar `limit_price` er `None` (f.eks. market-ordre i pending-listen).
**Fix:** Tilfoej guard: `if order.limit_price is not None and price <= order.limit_price`.

### K-3. `paper_broker.py` linje 304-323 — Short-positioner beregner INGEN fees
**Problem:** `_fill_short_open()` og `_fill_short_close()` kalder ikke `_fee_calc.calculate()`. Short-positioner faar kunstigt bedre P&L.
**Fix:** Tilfoej fee-beregning som i `_fill_buy()`/`_fill_sell()`.

### K-4. `risk_manager.py` linje 160 — Exit-bypass omgaar ALLE risikotjek for shorts
**Problem:** `if side == "short" and symbol in self.portfolio.positions` returnerer `approved=True` UDEN at tjekke halted, max-drawdown, dagligt tab. Under et crash kan positioner stadig aabnes/lukkes.
**Fix:** Flyt dette tjek EFTER halted/drawdown-checks.

### K-5. `risk_manager.py` linje 107-128 — JSON config overskriver risk-parametre UDEN validering
**Problem:** Tre JSON-filer kan overskrive `max_position_pct`, `max_open_positions` og `stop_loss_pct` uden graensevalidering. En korrupt fil kan saette `max_position_pct = 1.0` (100% i een position). Alle exceptions svaelges med `except Exception: pass`.
**Fix:** Tilfoej validering: `0.001 <= max_position_pct <= 0.20`.

### K-6. `portfolio_tracker.py` linje 353 — Short-positioner tjekker IKKE margin
**Problem:** Ved short-aabning tilfojes `qty * price` til cash uden margin-reservation. Systemet kan bruge short-proceeds til nye positioner = kunstig koebekraft.
**Fix:** Tilfoej margin-reservation (f.eks. `cost * 1.5`).

### K-7. `connection_manager.py` linje 163-226 — Race condition i check_broker
**Problem:** `check_broker()` modificerer `health`-objektet UDEN lock, mens `_monitor_loop` bruger lock. Samtidige kald kan korrumpere `consecutive_failures`.
**Fix:** Tag `self._lock` i `check_broker()`.

### K-8. `alpaca_broker.py` linje 214 — cancel_order retry er brudt
**Problem:** `@_retry_on_transient()` + indre try/except der fanger `APIError`. Den indre handler sluger fejlen FOER retry-dekoratoren naar den. Transiente fejl retries ikke.
**Fix:** Fjern enten dekoratoren eller den indre try/except.

### K-9. `config/settings.py` linje 128 — Dead code: validering naas aldrig
**Problem:** `max_position_pct > 1.0` checket naas aldrig fordi et tidligere check (`stop_loss_pct > 0.10`) kaster fejl foerst. `max_position_pct` valideres dermed ALDRIG.
**Fix:** Flyt `max_position_pct`-check til separat blok.

### K-10. `config/default_config.yaml` — Risk-config modstrid: 8 positioner x 15% = 120%
**Problem:** `max_position_pct: 0.15` og `max_open_positions: 8` giver op til 120% eksponering, men `max_exposure_pct: 0.80`. Modstridende begransninger.
**Fix:** Saenk `max_position_pct` til 0.08-0.10 saa 8 x 10% = 80%.

### K-11. `docker-compose.yml` linje 19 — Hardkodet PostgreSQL password "changeme"
**Problem:** Default password bruges hvis `.env` mangler. Port 5432 er eksponeret.
**Fix:** Fjern default. Bind til localhost: `"127.0.0.1:5432:5432"`.

### K-12. `corporate_tax.py` linje 528 — Mutable state i skatteberegning
**Problem:** `self.tax_credit` aendres permanent i `calculate_annual_tax()`. Gentagne kald akkumulerer forkerte vaerdier.
**Fix:** Brug lokal variabel, returner ny credit-balance.

### K-13. `continuous_learner.py` linje 96 — Baggrundstraad uden shutdown
**Problem:** Traaden startes men der er ingen `stop()`/`join()` metode. Memory/resource leak ved programstop.
**Fix:** Implementer `stop()` med `self._running = False; self._thread.join(timeout=5)`.

### K-14. `auto_trader.py` linje 341 — Race condition i weekend mode
**Problem:** Koden tilgaar `self._portfolio` via baade dict-access og attribut-access. Hvis `_portfolio` aldrig saettes, crasher det med `AttributeError`.
**Fix:** Tilfoej konsistent tjek: `if hasattr(self, '_portfolio') and self._portfolio is not None`.

### K-15. Dobbelt konfiguration af risk-parametre
**Problem:** `max_position_pct` defineres i baade `risk_sizing.json` OG `default_config.yaml`. Uklart hvilken har prioritet.
**Fix:** Vaelg een autoritativ kilde. Dokumenter prioritetsraekkefoelge.

---

## HOEJE FUND (24 stk)

| # | Fil | Problem |
|---|-----|---------|
| H-1 | `order_manager.py:501` | SQLite ikke thread-safe — concurrent writes giver "database is locked" |
| H-2 | `ibkr_broker.py:496` | `submitted_at` bruger `datetime.now()` i stedet for faktisk tidsstempel |
| H-3 | `saxo_auth.py:212` | Base64 "encryption" fallback — tokens laeses paa 1 sekund |
| H-4 | `nordnet_auth.py:141` | Password sendt som form-data — kan logges af urllib3 debug |
| H-5 | `aggregated_portfolio.py:384` | Division by zero i `unrealized_pnl_pct` ved zero cost basis |
| H-6 | `portfolio_tracker.py:555` | Sharpe-ratio bruger population std (ddof=0) i stedet for sample std (ddof=1) |
| H-7 | `dynamic_risk.py:538` | `is_trading_allowed` property nulstiller circuit breaker som sideeffekt |
| H-8 | `signal_engine.py:289` | Prune sletter signalhistorik >7 dage ved HVERT scan |
| H-9 | `broker_router.py:433` | `get_account()` summerer USD + DKK + EUR uden FX-konvertering |
| H-10 | `auto_trader.py:603` | Permanent mode-skift til konservativ — ingen vej tilbage |
| H-11 | `auto_trader.py:225` | `position_size_pct` override overskriver base foer feedback-loop |
| H-12 | `corporate_tax.py:591` | Tab giver `tax_impact = 0` i stedet for negativt tilgodehavende |
| H-13 | `tax_calculator.py:205` | Udbytte beskattes med 27% (person) — uklart om selskab bruger 22% |
| H-14 | `notifier.py:64` | SMTP-credentials i plaintext — synlige i logs/tracebacks |
| H-15 | `continuous_learner.py:312` | `accuracy_7d` er reelt all-time accuracy — forkerte ensemble-vaegter |
| H-16 | `continuous_learner.py:452` | Drift-check nulstilles ved retrain — 20 trades "blindt" |
| H-17 | `default_config.yaml:374` | SL 1.5%/TP 2.5%/TS 1.2% er crypto-tunet — for stramt til aktier |
| H-18 | `docker-compose.yml:37` | Redis eksponeret paa 0.0.0.0:6379 uden password |
| H-19 | `Dockerfile:5` | Python 3.11 brugt, men platform er Python 3.14 |
| H-20 | `allocation.json` | Crypto 5% allokering vs. weekend 60% — modstridende |
| H-21 | `exchange_limits.json` | Boerser i config mangler i symbol-listen og vice versa |
| H-22 | `exchange_stop_loss.json` | Kun 4 boerser har stop-loss — resten faar 1.5% default |
| H-23 | `main.py:169` | Tilgaar privat `router._brokers` — fragilt |
| H-24 | `test_risk.py:592` | Test asserter `max_position_pct <= 0.05` men config siger 0.15 |

---

## MEDIUM FUND (37 stk)

| # | Fil | Problem |
|---|-----|---------|
| M-1 | `nordnet_broker.py:233` | Market-ordre simulering med +2% markup — dyr slippage |
| M-2 | `paper_broker.py:82` | Short-cover tjekker case-sensitive symbol match |
| M-3 | `ibkr_broker.py:191` | Exchange suffix-matching er greedy — korte suffixes matcher foerst |
| M-4 | `order_manager.py:382` | `assert` bruges til SQL-injection beskyttelse — fjernes med `-O` |
| M-5 | `broker_router.py:164` | `RoutingConfig()` allokeres ved HVERT kald |
| M-6 | `saxo_broker.py:196` | Rate-limit retry design er skoert |
| M-7 | `connection_manager.py:181` | avg_response_ms inkluderer off-by-one vaerdi |
| M-8 | `base_strategy.py:87` | Position sizing med confidence 10 giver 10% af max — for lavt |
| M-9 | `combined_strategy.py:35` | Division by zero hvis vaegter summerer til 0 |
| M-10 | `ensemble_ml_strategy.py:558` | `_majority_vote` er biased mod BUY-signaler |
| M-11 | `patterns.py:233` | Inkonsistent DataFrame copy-pattern |
| M-12 | `risk_manager.py:216` | Cash-check justerer silently ned uden warning |
| M-13 | `alpha_score.py:531` | Nye ML-objekter oprettes ved HVER scoring — langsomt |
| M-14 | `theme_tracker.py:186` | IndexError ved ETF hist med <5 bars |
| M-15 | `news_pipeline.py:482` | `sorted()` med `key=themes.get` uden default |
| M-16 | `corporate_tax.py:38` | Skattesats fra miljovariabel uden validering |
| M-17 | `currency.py:155` | ECB CSV parsing fejler stille ved API-aendring |
| M-18 | `mark_to_market.py:356` | Tab viser 0 skat i stedet for negativt tilgodehavende |
| M-19 | `notifier.py:85` | Ingen rate limiting paa email-afsendelse |
| M-20 | `fee_calculator.py:60` | Global mutable `_FEE_CONFIG` uden thread-safety |
| M-21 | `anomaly_detector.py:111` | Anomalier er kun in-memory — tabes ved genstart |
| M-22 | `health_monitor.py:144` | Health events er kun in-memory trods `_db_path` |
| M-23 | `social_sentiment.py:88` | Ticker-detektion matcher alle store 2-5 bogstavs ord |
| M-24 | `continuous_learner.py:456` | Crisis patterns er hardcoded |
| M-25 | `auto_trader.py:382` | `_dynamic_risk_manager` vs `_dynamic_risk` inkonsistens |
| M-26 | `auto_trader.py:829` | `gc.collect()` efter hvert scan — unodvendig CPU |
| M-27 | `settings.py:308` | `lru_cache` paa Settings uden reload-mekanisme |
| M-28 | `default_config.yaml:163` | Duplikat Airbus: `AIR.DE` + `AIR.PA` |
| M-29 | `default_config.yaml:106` | VIX (`^VIX`) kan ikke handles — er et indeks |
| M-30 | `requirements.txt` | Mangler `pyyaml` (kun i trader-requirements) |
| M-31 | `requirements.txt` | `pytest` dupliceret i begge requirements-filer |
| M-32 | `.env.example` | `IBKR_ENABLED` mangler |
| M-33 | `ibkr_datafeed.json` | Port 4002 vs. main.py default 7497 — inkonsistent |
| M-34 | `src/main.py:30` | `.env` indlaeses kun i src/main.py, ikke root main.py |
| M-35 | `conftest.py:18` | Skoert monkeypatch af `__defaults__` |
| M-36 | `indicators.py` | RSI-beregningsfejl kan give forkerte signaler |
| M-37 | `market_handoff.py` | Forkerte market-nogler (`"eu"` vs `"eu_nordic"`) — signaler matcher aldrig |

---

## TEST-RESULTATER

### 226 fejlede tests — rodsaarsager:

| Antal | Fejltype | Fix |
|-------|----------|-----|
| **148** | `FileExistsError: data_cache` | Brug `os.makedirs(..., exist_ok=True)` |
| **22** | `ModuleNotFoundError: sklearn` | Tilfoej `scikit-learn` til requirements.txt |
| **10** | `ImportError: pyarrow` | Tilfoej `pyarrow` til requirements.txt |
| **10** | MockRouter mangler `broker_override` param | Opdater mock-signaturer |
| **5** | Exchange-koder forkerte (AEB/SBF/HEX/EBS) | Ret exchange-mapping eller tests |
| **5** | Numpy array vs objekt-attribut | Fix ensemble return-type |
| **5** | API-signaturer ude af sync | Opdater test-konstruktorer |
| **4** | Afrundingsfejl i assertions | Juster tolerancer |
| **3** | Logik-fejl (scheduler, RSI, sizing) | Fix i kildekoden |
| **1** | `feedparser` mangler | Tilfoej til requirements.txt |
| **12** | RuntimeWarning: division by zero | Fix numpy-beregninger i market_scanner |

### Manglende dependencies:
```
scikit-learn    # ML-strategier (22 tests)
pyarrow         # Parquet-support (10 tests)
feedparser      # RSS news (1 test + import-fejl)
```

---

## ANBEFALINGER — PRIORITERET RAEKKEFOELGE

### Uge 1: Kritiske fixes (blokerer produktion)
1. Fix K-1 til K-8 (broker/risk kritiske bugs)
2. Tilfoej manglende dependencies til requirements.txt
3. Fix `data_cache` mkdir (148 tests)
4. Fix risk-config modstrid (K-10, K-15)

### Uge 2: Hoeje fixes (datakonsistens + sikkerhed)
5. Fix SQLite threading (H-1)
6. Fix FX-summering i broker_router (H-9)
7. Fix Sharpe-ratio ddof (H-6)
8. Fix circuit breaker sideeffekt (H-7)
9. Fix Docker sikkerhed (K-11, H-18)
10. Opdater test-signaturer til at matche kode

### Uge 3: Medium fixes (stabilitet)
11. Fix market handoff nogler (M-37)
12. Fix RSI-beregning (M-36)
13. Fix assert-baseret SQL-beskyttelse (M-4)
14. Implementer email rate limiting (M-19)

### Loeende: Lav-prioritet
- Type annotations, kommentarer, kosmetik

---

## HURTIG-FIX GUIDE (copy-paste)

### Fix 148 test-fejl paa 1 minut:
Find alle `os.mkdir('data_cache')` og erstat med:
```python
os.makedirs('data_cache', exist_ok=True)
```

### Fix 33 test-fejl (dependencies):
```bash
pip install scikit-learn pyarrow feedparser
# Tilfoej til requirements.txt:
echo "scikit-learn>=1.3.0" >> requirements.txt
echo "pyarrow>=14.0.0" >> requirements.txt
echo "feedparser>=6.0.0" >> requirements.txt
```

### Fix K-1 (Nordnet sell crash):
```python
# nordnet_broker.py linje 270 — aendr fra:
def sell(self, symbol, qty, order_type, limit_price) -> Order:
# til:
def sell(self, symbol, qty, order_type, limit_price, short: bool = False) -> Order:
```

---

*Rapport genereret af Claude Opus 4.6 — 30. marts 2026*
*Baseret paa gennemgang af 120+ filer, 30.000+ linjer kode*
