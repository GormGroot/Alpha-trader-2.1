# Alpha Trading Platform

Automatiseret aktiehandelsplatform bygget i Python. Platformen henter markedsdata, kører handelsstrategier, backtester performance og viser resultater i et web-dashboard.

## Opsætning

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Udfyld med dine API-nøgler
```

## Projektstruktur

```
src/
├── data/       - Markedsdata-hentning (yfinance, API'er)
├── strategy/   - Handelsstrategier (SMA, RSI, custom)
├── broker/     - Mægler-integration (Alpaca API)
├── risk/       - Risikostyring og position sizing
├── backtest/   - Backtesting-motor
└── dashboard/  - Web-dashboard (Dash/Plotly)
tests/          - Unit tests
config/         - Konfigurationsfiler
```

## Kør dashboard

```bash
python -m src.dashboard.app
```
