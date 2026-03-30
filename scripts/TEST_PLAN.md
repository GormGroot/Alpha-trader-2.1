# Alpha Trader — Paper Trading Test Plan

## Forudsætninger

```bash
# 1. Installer dependencies
pip install -r requirements.txt -r requirements-trader.txt

# 2. Konfigurér credentials
cp .env.example .env
# Udfyld ALPACA_API_KEY og ALPACA_SECRET_KEY med paper keys
# → Opret gratis på https://app.alpaca.markets/signup

# 3. (Valgfrit) Start PostgreSQL
docker-compose up -d postgres redis
```

---

## Dag 1: Smoke Test + Connection

### Morgen
```bash
# Kør smoke test FØRST — fanger import-fejl uden credentials
python scripts/smoke_test.py

# Derefter paper test med Alpaca
python scripts/paper_test.py
```

### Tjek:
- [ ] Alle 10 steps i paper_test.py er PASS
- [ ] Alpaca account viser korrekt equity
- [ ] BrokerRouter resolver AAPL → alpaca
- [ ] Exchange detection virker (.CO→CSE, .DE→XETRA, etc.)
- [ ] Paper ordre (BUY 1 AAPL) eksekveres
- [ ] OrderManager gemmer i SQLite

### Eftermiddag
```bash
# Start dashboard
python main.py --mode dashboard

# Åbn browser: http://localhost:8050
```

### Tjek:
- [ ] Dashboard loader uden fejl
- [ ] Sidebar viser TRADING-sektion med 5 nye sider
- [ ] /portfolio viser KPI'er
- [ ] /trading viser ordre-panel
- [ ] /tax viser skatteberegning
- [ ] /markets viser heatmap
- [ ] /status viser Alpaca som connected

---

## Dag 2: Trading Flow

### Morgen — Ordrer
```bash
# Start i trader mode (med scheduler)
python main.py --mode trader --paper
```

### Test via dashboard:
1. Gå til /trading
2. Indtast "AAPL" i søgefelt
3. Verificér routing info (→ Alpaca)
4. Placér BUY 2 AAPL (market)
5. Tjek at ordren dukker op i "Open Orders"
6. Vent på fill
7. Gå til /portfolio — se positionen
8. Placér SELL 1 AAPL
9. Tjek at P&L beregnes korrekt

### Eftermiddag — Skat
1. Gå til /tax
2. Verificér lagerbeskatning for AAPL-positionen
3. Tjek skattetilgodehavende gauge
4. Klik "Export" for årsrapport (CSV)

### Tjek:
- [ ] Ordre eksekveres korrekt via dashboard
- [ ] Positions opdateres i /portfolio
- [ ] Tax impact vises ved salg
- [ ] CSV-export virker

---

## Dag 3: Scheduler + Reports

### Morgen
```bash
# Start med scheduler
python main.py --mode trader --paper

# Scheduler kører automatisk:
# 07:30 CET — Morning check
# 09:00 CET — EU market open
```

### Manuel scheduler test:
```python
# I Python console:
from src.ops.daily_scheduler import DailyScheduler
s = DailyScheduler()
result = s.run_task_now("morning_check")
print(result.status, result.details)
```

### Email test (hvis SMTP konfigureret):
```python
from src.ops.email_reports import EmailReportRunner
runner = EmailReportRunner()
runner.send_morning_report()  # Sender til REPORT_EMAIL_TO
```

### Tjek:
- [ ] Scheduler starter og viser tidsplan i logs
- [ ] Morning check kører (verificér broker connections)
- [ ] Email rapport (hvis konfigureret) har korrekt data
- [ ] Sample rapport i data/sample_evening_report.html ser fin ud

---

## Dag 4: Stabilitet

### Kør platformen hele dagen
```bash
python main.py --mode trader --paper
```

### Observer:
- [ ] Ingen crashes over 8+ timer
- [ ] Dashboard auto-refresh virker (30s positioner, 15s broker status)
- [ ] Scheduler kører tasks på rette tidspunkter
- [ ] Logs i logs/trading.log er fornuftige
- [ ] Hukommelsesforbrug stiger ikke konstant (memory leak)

### Stress test:
```bash
# Kør 20 hurtige ordrer
python -c "
from src.broker.alpaca_broker import AlpacaBroker
import os
b = AlpacaBroker(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), 'https://paper-api.alpaca.markets')
b.connect()
for i in range(10):
    b.buy('AAPL', 1)
    b.sell('AAPL', 1)
print('20 ordrer placeret')
"
```

---

## Dag 5: Backup + Cleanup

### Test backup
```bash
python -c "
from src.ops.backup import BackupManager
bm = BackupManager()
result = bm.run_daily_backup()
print(f'Backup: {result.success}, {result.size_bytes/1024:.0f} KB')
print(f'Components: {result.components}')
print(bm.verify_latest())
"
```

### Tjek:
- [ ] SQLite backup virker (data/test_*.db filer)
- [ ] Config backup (config/ folder kopieret)
- [ ] PostgreSQL dump (kræver running PG)
- [ ] Verify checksum OK
- [ ] Cleanup sletter filer ældre end retention

---

## Næste Skridt (efter paper test)

Når alle 5 dage er bestået:

1. **Tilføj IBKR** — Start TWS paper, sæt IBKR_PORT=7497
2. **Tilføj Nordnet** — Brug rigtige credentials (demo-konto)
3. **Tilføj Saxo** — OpenAPI SIM environment
4. **Multi-broker test** — Verificér routing på tværs af alle 4
5. **Live mode** — Skift til `--mode trader` (uden --paper)
