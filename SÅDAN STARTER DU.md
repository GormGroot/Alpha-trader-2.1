# Alpha Trader — Kom i gang

## Den hurtige version

**Dobbeltklik på `START.command`** i Finder. Færdig.

Det installerer alt første gang (tager 1-2 min), og åbner derefter dashboardet i din browser automatisk.

---

## Hvad sker der?

Når du dobbeltklikker på `START.command`:

1. Den checker at Python 3 er installeret på din Mac
2. Første gang: opretter en "virtual environment" (en isoleret Python-installation, så det ikke roder med andre ting på din Mac)
3. Første gang: installerer alle de biblioteker platformen bruger
4. Starter platformen i **paper trading mode** (ingen rigtige penge)
5. Åbner `http://localhost:8050` i din browser — det er dashboardet

## Hvad ser du i dashboardet?

Sidebar til venstre med:

- **Portfolio** — dine positioner og samlet værdi
- **Trading** — ordrehistorik og nye ordrer
- **Skat** — selskabsskat, lagerbeskatning, DBO-rater
- **Markets** — heatmap, søg instrumenter, watchlist
- **Broker Status** — om Alpaca er forbundet

## Sådan stopper du

Tryk `Ctrl+C` i Terminal-vinduet, eller luk Terminal.

## Troubleshooting

**"Python 3 er ikke installeret"**
→ Åbn Terminal og kør: `brew install python3`
→ Har du ikke Homebrew? Gå til python.org/downloads

**"Permission denied" når du dobbeltklikker**
→ Højreklik → Åbn med → Terminal

**"Port 8050 already in use"**
→ Du kører det allerede, eller noget andet bruger porten
→ Luk det andet program, eller kør manuelt med en anden port

## Køre manuelt (for nørder)

```bash
cd alpha-trading-platform
source .venv/bin/activate
python main.py --mode trader --paper
```
