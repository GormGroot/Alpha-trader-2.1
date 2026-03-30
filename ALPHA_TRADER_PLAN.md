# Alpha Trader — Oles Handelsplatform

**Formål:** Handle bredt med firmaets kapital via fire brokers. Slå markedet med
intelligent analyse, nyheder og data. Udnyt skattetilgodehavende.
**Hardware:** Normal PC (en enkelt GPU er nice-to-have for FinBERT, men ikke påkrævet)
**Brokers:** Alpaca, Saxo Bank, Interactive Brokers, Nordnet

---

## 1. HVAD ALPHA TRADER ER

Et intelligent handelsværktøj der giver dig en informationsfordel over markedet.
Du åbner ét dashboard og ser ikke bare priser — du ser hvad der driver dem, hvad
der er ved at ske, og hvad du bør gøre ved det. Du handler amerikanske tech-aktier,
danske C25-aktier, europæiske ETF'er, guld, olie, bitcoin, EUR/USD — alt sammen med
AI-drevet analyse, fuld risk management og real-time skatteoverblik.

**Alpha Trader er bygget til at slå markedet ved at:**
- Forstå nyheder hurtigere og dybere end du kan manuelt
- Kombinere 30+ indikatorer, sentiment, makro og alternativ data i ét signal
- Reagere på events inden markedet har fordøjet dem
- Finde mønstre på tværs af markeder, sektorer og asset classes
- Køre strategier 24/7 uden følelser eller træthed

**Det er IKKE:**
- Den tunge GPU research-platform (det er Gorms maskine)
- Et SaaS-produkt
- En erstatning for din broker — det er et intelligent lag OVEN PÅ dine brokers

---

## 2. FIRE BROKERS — KOMPLET MARKEDSDÆKNING

### Broker-routing: Hvem handler hvad?

| Instrument | Primær Broker | Backup | Marked |
|------------|---------------|--------|--------|
| US Aktier (AAPL, MSFT, etc.) | Alpaca | IBKR | NYSE, NASDAQ |
| US ETF'er (SPY, QQQ, VTI) | Alpaca | IBKR | NYSE Arca |
| Crypto (BTC, ETH, SOL) | Alpaca | — | 24/7 |
| Danske aktier (NOVO, MAERSK) | Nordnet | Saxo | OMX C25 |
| Svenske aktier (VOLVO, ABB) | Nordnet | Saxo | OMX Stockholm |
| Norske aktier (EQNR, DNB) | Nordnet | Saxo | Oslo Børs |
| EU aktier (ASML, SAP, LVMH) | IBKR | Saxo | Xetra, Euronext, LSE |
| UK aktier (HSBC, BP, AZN) | IBKR | Saxo | LSE |
| EU ETF'er (IWDA, VWCE) | Saxo | IBKR | Xetra, Euronext |
| Obligationer / Bonds | Saxo | IBKR | Diverse |
| Råstoffer (Guld, Olie, etc.) | IBKR | Saxo | COMEX, NYMEX, ICE |
| Forex (EUR/USD, GBP/DKK) | IBKR | Saxo | Interbank |
| Futures | IBKR | — | CME, Eurex, ICE |
| Options (US) | IBKR | — | CBOE, ISE |
| Options (EU) | IBKR | — | Eurex |
| Fonde (danske investeringsforeninger) | Nordnet | Saxo | Fondsbørsen |

### Hvorfor fire brokers?

- **Alpaca:** Gratis US handel, god API, crypto, allerede integreret
- **Saxo Bank:** Dansk reguleret, bred EU-dækning, god til ETF'er og obligationer, dansk skat-integration
- **Interactive Brokers:** Billigst for aktiv handel, 150+ markeder, bedste API, options/futures
- **Nordnet:** Bedst for nordiske aktier, danske fonde, aktiesparekonto (privat), nem dansk skat

### Broker API'er

| Broker | API Type | Python Library | Auth |
|--------|----------|---------------|------|
| Alpaca | REST + WebSocket | `alpaca-trade-api` | API key + secret |
| Saxo Bank | REST (OpenAPI) | `saxo_openapi` eller raw `requests` | OAuth2 (24h token refresh) |
| Interactive Brokers | Socket (TWS API) | `ib_insync` | IB Gateway credentials |
| Nordnet | REST (unofficial) | `nordnet` eller raw `requests` | Session token (login) |

**OBS om Nordnet:** Nordnet har ingen officiel API for private kunder. Der findes
community-libraries (nordnet-python), men de kan bryde ved opdateringer.
Alternativ: Brug Saxo for nordiske aktier hvis Nordnet-API er ustabil.

---

## 3. SKATTEOPTIMERING

### Firmaets situation
- Selskab (ApS/A/S) med skattetilgodehavende
- Beskatning: Selskabsskat 22% på realiserede gevinster
- Lagerbeskatning på visse porteføljeaktier og investeringsselskaber
- Realisationsbeskatning på andre aktier

### Hvad platformen skal gøre

**Real-time skatteoverblik:**
- Urealiseret gevinst/tab per position
- Skattemæssig konsekvens af at sælge HVER position NU
- "Hvad sker der med min skat hvis jeg sælger X?" — simulering

**Skatteoptimeret handel:**
- **Tab-realisering:** Sælg positioner med tab for at udnytte skattetilgodehavende
- **Gevinst-timing:** Udskyd realisering af gevinst til næste regnskabsår
- **Wash sale awareness:** Advar hvis du køber samme aktie inden 30 dage efter tab-salg
- **FIFO tracking:** Automatisk FIFO per position per broker
- **Lagerbeskatning vs realisationsbeskatning:** Flag per instrument

**Skatterapporter:**
- Kvartalsvis skatteoversigt (estimeret skat)
- Årsrapport klar til revisor
- Per-broker transaktionslog (til afstemning med brokers årsopgørelse)
- Valutakurs-dokumentation for alle ikke-DKK handler

### Skatteberegning per instrument-type

| Type | Beskatning (selskab) | Kilde |
|------|---------------------|-------|
| Danske aktier | Lagerbeskatning (22%) | Aktieavancebeskatningsloven §9 |
| Udenlandske aktier | Lagerbeskatning (22%) | Aktieavancebeskatningsloven §9 |
| ETF'er (aktiebaserede) | Lagerbeskatning (22%) | ABL §19 |
| ETF'er (obligationsbaserede) | Lagerbeskatning (22%) | Kursgevinstloven |
| Obligationer | Lagerbeskatning (22%) | Kursgevinstloven |
| Crypto | Lagerbeskatning (22%) | Statsskatteloven §4-6 |
| Forex gevinster | Lagerbeskatning (22%) | Kursgevinstloven |
| Options/Futures | Lagerbeskatning (22%) | Kursgevinstloven §29-33 |
| Råvarer (fysisk) | Realisationsbeskatning (22%) | Statsskatteloven |
| Udbytter (DK) | 22% (modregnes i skat) | Selskabsskatteloven |
| Udbytter (udenlandske) | 22% - kildeskat-kredit | DBO-aftaler |

**VIGTIGT:** Selskaber er som udgangspunkt lagerbeskattet på ALLE finansielle aktiver.
Det betyder skat betales af urealiserede gevinster ved regnskabsårets udgang.
Skattetilgodehavende kan modregnes i disse gevinster.

---

## 4. FEATURES — PRIORITERET

### Kritisk (v1 — handle + analyse ASAP)

0. **Intelligence Engine (markedsanalyse)**
   - Real-time nyhedsaggregering fra alle kilder (RSS, Finnhub, Alpha Vantage)
   - FinBERT sentiment scoring på CPU (eller GPU hvis tilgængelig)
   - Event detection: earnings, M&A, FDA, CEO changes, insider trades
   - Macro dashboard: FRED data, ECB, Fed, employment, inflation
   - Morning briefing: AI-genereret markedsoversigt hver morgen
   - Multi-source signal scoring: kombiner technicals + sentiment + macro + alt data

1. **BrokerRouter med 4 brokers**
   - Alpaca (eksisterer), IBKR, Saxo, Nordnet
   - Automatisk routing baseret på instrument
   - Unified ordreindgang

2. **Samlet Portfolio View**
   - Alle positioner fra alle 4 brokers i ét view
   - Total værdi i DKK med real-time FX
   - P&L per position, per broker, totalt

3. **Ordrehåndtering**
   - Market orders, limit orders, stop orders
   - Order status tracking på tværs af brokers
   - Cancel/modify orders

4. **Basis Risk Management**
   - Max position size (% af portefølje)
   - Stop-loss per position
   - Total drawdown limit
   - Daily loss limit

5. **Transaktionslog til Skat**
   - Alle handler logges: dato, instrument, side, antal, pris, valuta, kurs, broker
   - FIFO-beregning per instrument
   - Eksport til CSV/Excel for revisor

### Høj prioritet (v2 — skatteoptimering + dybere analyse)

6. **Skatteberegner (selskab)**
   - Lagerbeskatning: beregn urealiseret gevinst/tab ved årsskifte
   - Skattetilgodehavende tracker
   - "Sælg for at realisere tab" — foreslå optimale salg
   - Simulering: "Hvad bliver min skat hvis markedet lukker her?"

7. **Multi-currency Management**
   - FX tracking for alle handler
   - Valutagevinst/-tab beregning (skattemæssigt relevant)
   - DKK som basisvaluta, alle beregninger konverteres

8. **Eksisterende Strategier**
   - SMA, RSI, ML ensemble (CPU-baseret, ingen GPU nødvendig)
   - Signal engine med buy/sell signaler
   - Backtest mod historisk data

### Høj prioritet (v2.5 — avanceret analyse)

9. **Claude/LLM API Markedsanalyse**
    - Brug Claude API (betalt, men ingen GPU nødvendig) til:
      - Daglig markedsnarrative: "Hvad skete og hvorfor?"
      - Earnings call analyse: Ekstraher guidance, tone, nøgletal
      - Event impact assessment: "Hvad betyder denne nyhed for min portefølje?"
      - Portefølje-rådgiver: "Baseret på nuværende data, hvad bør jeg gøre?"
    - FinBERT sentiment aggregering per aktie/sektor/marked
    - Anomaly detection: Usædvanlige pris/volume-mønstre
    - Cross-asset analyse: "Olie stiger → hvad påvirkes?"

10. **Alpha Score per Aktie**
    - Samlet score (0-100) der kombinerer ALLE datakilder:
      - Tekniske indikatorer (RSI, MACD, SMA, Bollinger) → 25%
      - Sentiment (FinBERT nyheder + social) → 20%
      - ML ensemble prediction (XGBoost, RF, LogReg) → 20%
      - Makro-regime tilpasning → 10%
      - Alternativ data (Google Trends, insider trades, options flow) → 15%
      - Sæsonmønstre + earnings proximity → 10%
    - Ranked watchlist: Top 20 aktier sorted by Alpha Score
    - Historical Alpha Score accuracy tracking

11. **Sektor Rotation & Theme Tracking**
    - Automatisk detektering af roterende kapital (tech → energy → healthcare)
    - Theme tracking: AI, GLP-1, energy transition, defense, reshoring
    - ETF flow analyse som proxy for institutional positioning
    - "Hvad er hot, hvad er ikke" dagligt overblik

### Medium prioritet (v3 — fuld automatisering)

12. **Automatisk Handel (semi-autonom)**
    - Strategier kan handle automatisk inden for risk limits
    - Alpha Score > 80 → auto-buy signal
    - Alpha Score < 20 → auto-sell signal
    - Daglig email rapport med alle beslutninger og begrundelser
    - Emergency stop

13. **Avanceret Dashboard**
    - Europæisk markeds-heatmap
    - Skatteoverview widget
    - Broker-status per connection
    - Alpha Score leaderboard
    - Sentiment timeline per aktie
    - News feed med real-time sentiment scoring
    - "Markedets puls" — aggregeret bull/bear gauge

---

## 5. UDVIKLINGSPROMPTS — ALPHA TRADER

---

### PROMPT T0: Intelligence Engine — Analyse & Nyheder

```
Byg Alpha Traders intelligence engine der giver informationsfordel over markedet.

KONTEKST:
- Lokal single-user platform, kører på normal PC (CPU-baseret, valgfri GPU for FinBERT)
- Eksisterende moduler der ALLEREDE virker og skal bruges:
  - src/sentiment/news_fetcher.py — Finnhub, Alpha Vantage, RSS feeds
  - src/sentiment/sentiment_analyzer.py — FinBERT + keyword fallback
  - src/sentiment/event_detector.py — FDA, earnings, M&A, exec changes
  - src/sentiment/earnings_tracker.py — Earnings surprise detection
  - src/sentiment/macro_calendar.py — Economic calendar
  - src/data/indicators.py — 30+ tekniske indikatorer
  - src/data/alternative_data.py — Google Trends, GitHub, patents, job postings
  - src/data/macro_indicators.py — FRED API, recession indicators
  - src/data/options_flow.py — Unusual options, put/call ratio, IV
  - src/data/onchain.py — Crypto fear/greed, DeFi TVL
  - src/strategy/signal_engine.py — Parallel strategy runner
  - src/strategy/regime.py — Market regime classification

ALT OVENFOR EKSISTERER. Byg OVEN PÅ det — importer og brug det.

OPGAVE:

1. Alpha Score Engine (src/trader/intelligence/alpha_score.py):
   - Beregn en samlet score (0-100) per aktie der kombinerer ALLE datakilder:

   class AlphaScoreEngine:
     def calculate_alpha_score(self, symbol: str) -> AlphaScore:
       # Hent alle datapunkter parallelt
       technicals = self._technical_score(symbol)    # 0-100, vægt 25%
       sentiment = self._sentiment_score(symbol)      # 0-100, vægt 20%
       ml_prediction = self._ml_score(symbol)         # 0-100, vægt 20%
       macro_alignment = self._macro_score(symbol)    # 0-100, vægt 10%
       alt_data = self._alternative_score(symbol)     # 0-100, vægt 15%
       seasonality = self._seasonality_score(symbol)  # 0-100, vægt 10%

       # Weighted combination
       total = (technicals * 0.25 + sentiment * 0.20 + ml_prediction * 0.20
                + macro_alignment * 0.10 + alt_data * 0.15 + seasonality * 0.10)

       return AlphaScore(
         symbol=symbol,
         total=total,
         breakdown={...},
         signal="STRONG_BUY" if total > 80 else "BUY" if total > 65
                 else "HOLD" if total > 35 else "SELL" if total > 20
                 else "STRONG_SELL",
         confidence=self._calculate_confidence(breakdown),
         explanation=self._generate_explanation(breakdown),
         timestamp=datetime.now()
       )

   Sub-score beregninger:
   a. _technical_score(): Brug indicators.py
      - RSI position (oversold=high score, overbought=low)
      - MACD signal (bullish cross=high)
      - Price vs SMA20/50/200 (over all=bullish)
      - Bollinger Band position
      - Volume confirmation
      - ADX trend strength

   b. _sentiment_score(): Brug sentiment_analyzer.py + news_fetcher.py
      - Aggregér FinBERT scores fra seneste 24h nyheder
      - Vægt efter kilde-troværdighed og nyhed (nyere = højere vægt)
      - Earnings sentiment boost/penalty
      - Social sentiment (hvis tilgængelig)

   c. _ml_score(): Brug eksisterende ml_strategy.py og ensemble_ml_strategy.py
      - Kør begge modeller
      - Konverter BUY/SELL/HOLD + confidence til 0-100

   d. _macro_score(): Brug macro_indicators.py + regime.py
      - Er markedsregimet bullish/bearish?
      - Passer denne aktie til regimet? (defensiv i bear, cyclical i bull)
      - Rentemiljø favorabelt for sektoren?
      - Recession sandsynlighed

   e. _alternative_score(): Brug alternative_data.py + options_flow.py
      - Google Trends (stigende interesse = positiv)
      - Options flow (unusual bullish activity = positiv)
      - Insider trading (net buying = positiv)
      - Put/Call ratio (extreme fear = contrarian positiv)

   f. _seasonality_score():
      - Historisk performance denne måned (10-års snit)
      - Earnings proximity (30 dage før = høj vol, reduce score certainty)
      - Day-of-week/month-of-year patterns
      - "Sell in May" og lignende kalendermønstre

2. News Intelligence Pipeline (src/trader/intelligence/news_pipeline.py):

   class NewsPipeline:
     def __init__(self):
       self.fetcher = NewsFetcher()        # Eksisterer
       self.analyzer = SentimentAnalyzer()  # Eksisterer
       self.detector = EventDetector()      # Eksisterer

     def run_pipeline(self, symbols: list[str]) -> IntelligenceReport:
       # 1. Fetch alt nyt (RSS, Finnhub, Alpha Vantage)
       articles = self.fetcher.fetch_all(symbols)

       # 2. Sentiment score per artikel
       scored = [self.analyzer.analyze(a) for a in articles]

       # 3. Event detection
       events = self.detector.detect(articles)

       # 4. Aggregér per symbol
       per_symbol = self._aggregate_by_symbol(scored, events)

       # 5. Cross-reference: påvirker en nyhed andre aktier?
       cross_impacts = self._detect_cross_impacts(events)
       # Eksempel: "OPEC cut production" → påvirker EQNR.OL, BP.L, XOM

       # 6. Prioritér: Hvad er VIGTIGST lige nu?
       prioritized = self._rank_by_impact(per_symbol, cross_impacts)

       return IntelligenceReport(
         timestamp=datetime.now(),
         top_events=prioritized[:10],
         per_symbol=per_symbol,
         cross_impacts=cross_impacts,
         market_mood=self._overall_mood(scored),
         regime=RegimeDetector().detect()  # Eksisterer
       )

   _detect_cross_impacts(events):
     # Simpel regelbaseret cross-impact mapping:
     cross_impact_rules = {
       "oil_price": ["XOM", "CVX", "EQNR.OL", "BP.L", "TTE.PA", "ENI.MI"],
       "interest_rate": ["BANKS", "REITS", "UTILITIES"],  # Sektor-påvirkning
       "usd_strength": ["EMERGING_MARKETS", "GOLD", "EXPORTERS"],
       "semiconductor": ["ASML.AS", "NVDA", "AMD", "TSM", "INTC"],
       "glp1_obesity": ["NOVO-B.CO", "LLY", "AMGN", "VKTX"],
       "ai_spending": ["NVDA", "MSFT", "GOOGL", "AMZN", "META"],
       "china_policy": ["BABA", "JD", "PDD", "NIO", "XPEV"],
       "ecb_policy": ["EUR_BANKS", "EU_BONDS", "EUR_FX"],
     }
     # Match event keywords til rules, return affected symbols

3. Morning Briefing Generator (src/trader/intelligence/morning_briefing.py):

   Kør dagligt kl 07:30 CET. Brug Claude API (Anthropic) til at generere:

   class MorningBriefing:
     def generate(self) -> str:
       # Saml data
       data = {
         "overnight_us": self._get_us_close(),     # S&P, Nasdaq, Dow
         "asia_session": self._get_asia_summary(),  # Nikkei, Hang Seng
         "eu_premarket": self._get_eu_futures(),     # STOXX futures
         "fx_moves": self._get_fx_overnight(),       # EUR/USD, GBP, DKK
         "crypto": self._get_crypto_24h(),           # BTC, ETH
         "commodities": self._get_commodity_moves(), # Oil, Gold
         "top_news": self._get_top_news(limit=10),   # Vigtigste nyheder
         "todays_calendar": self._get_economic_calendar(),  # ECB, Fed, earnings
         "portfolio_exposure": self._get_current_positions(),
         "top_alpha_scores": self._get_top_alpha_scores(n=10),
         "regime": RegimeDetector().detect(),
       }

       # Send til Claude API for intelligent sammenfatning
       prompt = f"""
       Du er Oles private markedsanalytiker. Skriv en kort, skarp morgen-briefing.
       Fokusér på hvad der ER RELEVANT for hans portefølje og hvad han bør handle på.
       Vær direkte — ingen fluff. Hvis der er en klar mulighed, sig det.
       Hvis der er en risiko for porteføljen, sig det.

       DATA:
       {json.dumps(data, indent=2)}

       Format:
       🔑 HOVEDPOINTER (3-5 bullets, det vigtigste)
       📊 MARKEDSOVERBLIK (kort)
       🎯 HANDLEMULIGHEDER (aktier med høj Alpha Score + begrundelse)
       ⚠️ RISICI (hvad kan gå galt i dag)
       📅 VIGTIGE EVENTS I DAG (earnings, macro releases)
       💼 PORTEFØLJENOTER (noget der påvirker dine positioner)
       """

       response = anthropic.messages.create(
         model="claude-sonnet-4-20250514",
         max_tokens=2000,
         messages=[{"role": "user", "content": prompt}]
       )
       return response.content[0].text

   - Gem briefing i database
   - Send som email
   - Vis i dashboard

4. Evening Analysis (src/trader/intelligence/evening_analysis.py):
   - Kør dagligt kl 22:00 CET efter US close
   - Samme tilgang som morning briefing men retrospektiv:
     - "Hvad skete i dag og hvorfor?"
     - "Hvordan performede porteføljen vs markedet?"
     - "Hvad sagde Alpha Score vs hvad der faktisk skete?" (calibration)
     - "Hvad bør du overveje til i morgen?"
     - "Skattemæssig status: estimeret lagerbeskatning opdateret"
   - Claude API til at generere intelligent analyse

5. Alert System (src/trader/intelligence/alert_system.py):
   - Real-time alerts (ikke kun daglige rapporter):
     - Alpha Score > 85 for en watchlist-aktie → "Stærkt købssignal for ASML"
     - Alpha Score dropper 30+ points → "Advarsel: NOVO sentiment kollapser"
     - Earnings surprise > 10% → "AAPL beat med 15%, aktien +4% after hours"
     - Insider køb > $1M → "CEO of X bought $2M in shares"
     - Unusual options activity → "Stor bullish bet på TSLA $300 calls"
     - Regime change → "Markedet skifter fra risk-on til risk-off"
     - Portefølje drawdown > 3% → "Din portefølje er faldet 3.5% i dag"
   - Send via email (prioriteret: CRITICAL, HIGH, MEDIUM, LOW)
   - Dashboard notification feed (real-time)

6. Watchlist Intelligence (src/trader/intelligence/watchlist.py):
   - Dynamisk watchlist baseret på Alpha Score:
     - Top 20 aktier fra hele universet, sorted by score
     - Opdateret dagligt
     - Inkluder: symbol, score, signal, top driver, sentiment mood
   - Bruger-definerede watchlists med automatisk Alpha Score
   - "Discovery mode": Scan 5000+ symboler, find nye muligheder
   - Sektor-watchlist: Top 3 per sektor

7. Theme Tracker (src/trader/intelligence/theme_tracker.py):
   - Track aktive investerings-temaer:
     themes = {
       "AI_Infrastructure": {
         "symbols": ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "ASML.AS"],
         "keywords": ["artificial intelligence", "GPU", "data center", "LLM"],
         "strength": 0-100 (baseret på nyhedsvolume + prismomentum)
       },
       "GLP1_Obesity": {
         "symbols": ["NOVO-B.CO", "LLY", "AMGN", "VKTX"],
         "keywords": ["GLP-1", "obesity", "weight loss", "semaglutide", "Wegovy"],
         "strength": 0-100
       },
       "European_Defense": {
         "symbols": ["RHM.DE", "BA.PA", "SAF.PA", "SAAB-B.ST"],
         "keywords": ["defense spending", "NATO", "military", "rearmament"],
         "strength": 0-100
       },
       # ... flere temaer
     }
   - Automatisk theme detection fra nyhedsflow
   - Theme momentum: Stiger/falder interessen?
   - Cross-theme korrelation
   - Claude API til at identificere nye temaer månedligt

CLAUDE API INTEGRATION:
- pip install anthropic
- API key i .env: ANTHROPIC_API_KEY=xxx
- Model: claude-sonnet-4-20250514 (god balance mellem pris og kvalitet)
- Estimeret forbrug: ~$30-50/måned (2 briefings/dag + alerts)
- Fallback: Hvis API er nede, brug template-baseret rapport (uden AI-tekst)

KRAV:
- News pipeline kører hvert 15. minut i markedsåbningstid
- Alpha Score opdateres dagligt for hele universet, hvert 15. min for watchlist
- Morning briefing klar kl 07:30 CET (før EU-marked åbner)
- Evening analysis klar kl 22:15 CET (efter US close)
- Alerts real-time med max 5 min delay
- Alle analyser gemmes i database for historisk tracking
- Alpha Score accuracy tracking: Var score 80+ faktisk = god investering?
- FinBERT kører på CPU (langsommere men virker, ~1 sek/artikel)
  Hvis GPU tilgængelig: Brug den → ~0.05 sek/artikel
```

---

### PROMPT T0.5: Claude API Integration & Markedsanalyse

```
Implementer Claude API integration til intelligent markedsanalyse.

KONTEKST:
- Alpha Trader bruger Claude API (Anthropic) til tekstanalyse
- IKKE en lokal LLM — vi bruger API'en for bedre kvalitet og ingen GPU-krav
- Bruges til: Morning briefing, evening analysis, earnings analyse, ad-hoc spørgsmål

OPGAVE:

1. Claude Client (src/trader/intelligence/llm_client.py):
   class ClaudeClient:
     def __init__(self):
       self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
       self.model = "claude-sonnet-4-20250514"
       self.token_tracker = TokenTracker()  # Track forbrug

     async def analyze(self, prompt: str, system: str = None,
                       max_tokens: int = 2000) -> str:
       """Send prompt til Claude og return response."""
       messages = [{"role": "user", "content": prompt}]
       response = self.client.messages.create(
         model=self.model,
         max_tokens=max_tokens,
         system=system or self._default_system_prompt(),
         messages=messages
       )
       self.token_tracker.log(
         input_tokens=response.usage.input_tokens,
         output_tokens=response.usage.output_tokens,
         cost=self._calculate_cost(response.usage)
       )
       return response.content[0].text

     def _default_system_prompt(self) -> str:
       return """Du er en erfaren markedsanalytiker der arbejder for Alpha Vision,
       et dansk investeringsselskab. Du analyserer globale markeder med fokus på
       europæiske og amerikanske aktier, ETF'er, råvarer, crypto og forex.

       Regler:
       - Vær direkte og konkret. Ingen generelle floskel.
       - Kvantificér når muligt (%, beløb, niveauer).
       - Referér til specifikke aktier, niveauer og events.
       - Advar eksplicit om risici.
       - Platformen handler for et dansk selskab — husk skattemæssige implikationer.
       - Svar på dansk medmindre data er på engelsk.
       - DISCLAIMER: Du giver analyse, ikke investeringsrådgivning."""

2. Analyse-funktioner:

   analyze_earnings(symbol, earnings_data) → str:
     """Analyser en earnings rapport: beat/miss, guidance, nøgletal, trading implikation."""

   analyze_event_impact(event, portfolio_positions) → str:
     """Vurdér en nyhedsevents påvirkning på porteføljen."""

   compare_stocks(symbols: list[str], criteria: str) → str:
     """Sammenlign aktier på specifikke kriterier."""

   portfolio_review(positions, market_data, alpha_scores) → str:
     """Fuld portefølje-gennemgang med anbefalinger."""

   sector_outlook(sector: str, macro_data) → str:
     """Sektorudsigt baseret på makro og nyheder."""

   ad_hoc_query(question: str, context: dict) → str:
     """Svar på ethvert markedsspørgsmål med fuld kontekst."""

3. Token & Cost Tracker (src/trader/intelligence/token_tracker.py):
   - Log hvert API call: timestamp, tokens in/out, estimeret cost
   - Dagligt budget: Max $5/dag (konfigurerbart)
   - Alert hvis budget nærmer sig
   - Månedlig cost rapport
   - Pricing (Claude Sonnet):
     - Input: $3/MTok
     - Output: $15/MTok
     - Estimat: 2 briefings + 5 alerts + ad-hoc = ~$1-2/dag

4. Dashboard Integration:
   - "Spørg Alpha" widget: Tekstfelt hvor du kan stille markedsspørgsmål
   - Morning briefing widget (vises fra 07:30)
   - Evening analysis widget (vises fra 22:15)
   - Klik på en aktie → "Analyser med AI" knap

KRAV:
- Async API calls (non-blocking)
- Retry med exponential backoff ved API fejl
- Cache: Gentag ikke samme analyse inden for 1 time
- Fallback: Template-baseret rapport hvis API er nede
- Cost cap: Automatisk stop ved dagligt budget-loft
```

---

### PROMPT T1: Saxo Bank Integration

```
Implementer Saxo Bank broker integration til Alpha Vision.

KONTEKST:
- Lokal single-user platform (ingen auth, ingen multi-tenant)
- Credentials i .env fil
- Eksisterende broker interface: src/broker/base_broker.py
- Saxo bruges primært til: EU ETF'er, obligationer, danske investeringsforeninger

SAXO OPENAPI:
- Base URL: https://gateway.saxobank.com/openapi/ (live)
- Simulation: https://gateway.saxobank.com/sim/openapi/
- Auth: OAuth2 med 24-timers token refresh
- Docs: https://www.developer.saxo/openapi/learn
- Rate limit: 120 requests/min per endpoint group

OPGAVE:

1. Saxo Auth Manager (src/broker/saxo_auth.py):
   - OAuth2 flow:
     a. Redirect bruger til Saxo login (én gang, manuelt)
     b. Modtag authorization code
     c. Exchange for access_token + refresh_token
     d. Auto-refresh access_token hver 20 min (expires efter 20 min)
     e. Refresh_token expires efter 24 timer → ny login nødvendig
   - Gem tokens encrypted i lokal fil (.saxo_tokens)
   - Token refresh scheduler (baggrunds-thread)
   - Alert når refresh_token nærmer sig udløb

2. SaxoBroker (src/broker/saxo_broker.py):
   Implementer BaseBroker interface:

   connect():
     - Load tokens fra .saxo_tokens
     - Verify token validity via /port/v1/accounts/me
     - Hent AccountKey og ClientKey
     - Return account info

   buy(symbol, quantity, order_type, limit_price):
     - Map symbol til Saxo instrument:
       - Brug /ref/v1/instruments?Keywords={symbol}&AssetTypes=Stock,Etf,Bond,FxSpot
       - Cache instrument lookup (instrument → Uic mapping)
     - POST /trade/v2/orders med:
       - AccountKey, Uic, AssetType, BuySell: "Buy"
       - OrderType: "Market" / "Limit" / "Stop"
       - Amount, OrderPrice (for limit)
       - OrderDuration: {DurationType: "DayOrder"} eller "GoodTillCancel"
     - Return mapped Order objekt

   sell(symbol, quantity, order_type, limit_price):
     - Samme som buy med BuySell: "Sell"

   get_positions():
     - GET /port/v1/positions/me
     - Map til vores Position model
     - Inkluder: Uic, Amount, AverageOpenPrice, CurrentPrice, ProfitLossOnTrade
     - Currency conversion til DKK

   get_account():
     - GET /port/v1/accounts/me
     - GET /port/v1/balances/me
     - Map: CashBalance, TotalValue, MarginAvailable, UnrealizedProfitLoss

   get_order_status(order_id):
     - GET /port/v1/orders/me
     - Map Saxo status → vores OrderStatus
     - Saxo statuses: Working, Filled, Cancelled, Rejected

   cancel_order(order_id):
     - DELETE /trade/v2/orders/{orderId}

   get_instruments(search_query, asset_types):
     - GET /ref/v1/instruments med filters
     - Return liste af tradeable instruments
     - Asset types: Stock, Etf, Bond, FxSpot, FxForwards, ContractFutures, CfdOnStock, etc.

3. Saxo-specifik funktionalitet:
   - Instrument search med fuzzy matching
   - Asset type mapping (Saxo bruger numeriske AssetType codes)
   - Currency handling (Saxo returnerer priser i instrumentets valuta)
   - Fee estimation via /ref/v1/instruments/details (Commission)
   - Corporate actions tracking

ENVIRONMENT VARIABLES:
   SAXO_APP_KEY=xxx (fra Saxo Developer Portal)
   SAXO_APP_SECRET=xxx
   SAXO_REDIRECT_URI=http://localhost:8080/callback
   SAXO_ENVIRONMENT=sim  # eller live
   SAXO_ACCOUNT_KEY=xxx (efter første login)

KRAV:
- Retry logic med exponential backoff (429 Too Many Requests)
- Rate limiting: Max 2 requests/sec (konservativt)
- Token encryption at rest
- Graceful handling af expired tokens (auto-refresh, alert hvis refresh fails)
- Logging af alle API calls med loguru
- Error mapping: Saxo error codes → vores exceptions
```

---

### PROMPT T2: Interactive Brokers Integration

```
Implementer Interactive Brokers integration til Alpha Vision.

Se PROMPT 4 i ALPHA_VISION_LOCAL_MASTERPLAN.md for fuld IBKR spec.

EKSTRA FOR TRADER-VERSION:
- Primær for: EU aktier, UK aktier, råstoffer, forex, futures, options
- Options chain data: reqSecDefOptParams() + reqMktData() for options
- Futures chain: ContFuture for continuous contracts
- Råvarer: Commodity futures (GC=Gold, CL=Oil, SI=Silver, HG=Copper, NG=NatGas)
- Forex: Forex pairs som Cash contracts (EUR.USD, GBP.DKK, etc.)

ENVIRONMENT VARIABLES:
   IBKR_HOST=127.0.0.1
   IBKR_PORT=4002 (paper) / 4001 (live)
   IBKR_CLIENT_ID=1
   IBKR_TRADING_MODE=paper
```

---

### PROMPT T3: Nordnet Integration

```
Implementer Nordnet broker integration til Alpha Vision.

KONTEKST:
- Nordnet har INGEN officiel API for private kunder/firmaer
- Community library: nordnet-next-api (uofficiel, kan bryde)
- Alternativ: Web scraping af Nordnet's interne API
- Bruges primært til: Danske aktier, svenske aktier, norske aktier, danske fonde

VIGTIG BEGRÆNSNING:
Nordnet integration er "best effort" og kan kræve vedligeholdelse
ved Nordnet platform-opdateringer. Saxo Bank er backup for nordiske aktier.

OPGAVE:

1. Nordnet Session Manager (src/broker/nordnet_auth.py):
   - Login via https://www.nordnet.dk/api/2/authentication/basic/login
   - Session cookie-baseret auth
   - Auto-relogin ved session expiry
   - Credentials i .env: NORDNET_USERNAME, NORDNET_PASSWORD

2. NordnetBroker (src/broker/nordnet_broker.py):
   Implementer BaseBroker interface med Nordnet's interne API:

   connect():
     - Login og gem session
     - Hent account ID via /api/2/accounts
     - Return account info

   get_positions():
     - GET /api/2/accounts/{accid}/positions
     - Map til vores Position model
     - Nordnet returnerer: instrument_id, qty, acq_price, market_value

   buy(symbol, quantity, order_type, limit_price):
     - Lookup instrument_id via /api/2/instruments?query={symbol}
     - POST /api/2/accounts/{accid}/orders med:
       - identifier: instrument_id
       - side: "BUY"
       - price: limit_price (eller 0 for market)
       - volume: quantity
       - order_type: "LIMIT" / "MARKET"
       - validity: {type: "DAY"} eller {type: "UNTIL_DATE", date: "YYYY-MM-DD"}

   sell/get_account/get_order_status/cancel_order:
     - Tilsvarende mappings

3. Nordnet-specifikke features:
   - Instrument search med Nordnet's søge-API
   - Fond-handel (danske investeringsforeninger)
   - Udbytte-tracking
   - Nordnet kursliste integration

4. Fallback Strategy:
   - Hvis Nordnet API fejler → log warning → route til Saxo
   - Marker Nordnet som "degraded" i health monitor
   - Periodic health check hvert 5. minut

KRAV:
- Defensive coding: Alle API responses validates (schema kan ændre sig)
- User-Agent header der ligner browser (undgå blocking)
- Rate limiting: Max 1 request/sec (aggressiv for at undgå ban)
- Session keepalive: Ping hvert 5. min
- ALLE data caches lokalt (reducer API calls)
```

---

### PROMPT T4: BrokerRouter — Samlet Ordrehåndtering

```
Byg BrokerRouter der samler alle fire brokers til ét unified interface.

OPGAVE:

1. BrokerRouter (src/broker/broker_router.py):

   class BrokerRouter(BaseBroker):
     brokers: dict[str, BaseBroker]  # "alpaca", "ibkr", "saxo", "nordnet"

   Routing Logic:
   - Instrument → Broker mapping defineret i config:
     routing_rules:
       # Nøjagtige matches
       "BTC-USD": "alpaca"
       "ETH-USD": "alpaca"

       # Exchange-baseret routing
       exchanges:
         "NYSE": "alpaca"
         "NASDAQ": "alpaca"
         "CSE": "nordnet"      # Copenhagen (fallback: saxo)
         "SFB": "nordnet"      # Stockholm (fallback: saxo)
         "OSE": "nordnet"      # Oslo (fallback: saxo)
         "HEX": "nordnet"      # Helsinki (fallback: saxo)
         "XETRA": "ibkr"      # Tyskland
         "SBF": "ibkr"        # Euronext Paris
         "AEB": "ibkr"        # Euronext Amsterdam
         "LSE": "ibkr"        # London
         "EBS": "ibkr"        # SIX Swiss
         "CME": "ibkr"        # Futures
         "CBOE": "ibkr"       # Options

       # Asset type fallback
       asset_types:
         "forex": "ibkr"
         "futures": "ibkr"
         "options": "ibkr"
         "commodity": "ibkr"
         "etf_eu": "saxo"
         "bond": "saxo"
         "fund_dk": "nordnet"
         "crypto": "alpaca"

       # Fallback chain
       fallback: ["ibkr", "saxo", "alpaca"]

   resolve_broker(symbol, asset_type=None) → (broker_name, broker_instance):
     1. Check exact match
     2. Detect exchange from symbol suffix (.CO, .ST, .DE, .L, etc.)
     3. Check asset_type routing
     4. Try fallback chain
     5. Raise RoutingError if no broker can handle

2. Aggregated Portfolio (src/broker/aggregated_portfolio.py):

   get_all_positions() → list[AggregatedPosition]:
     - Fetch positions from all connected brokers (parallel)
     - Merge by symbol (same stock on different brokers = combined)
     - Add broker source tag
     - Convert all to DKK

   get_total_value(base_currency="DKK") → PortfolioSummary:
     - Sum across all brokers
     - Breakdown: per broker, per asset type, per currency, per country
     - Include cash balances from all brokers
     - Real-time FX conversion

   get_combined_trades(start_date, end_date) → list[Trade]:
     - All trades from all brokers in one timeline
     - Sorted by timestamp
     - Used for tax calculation and P&L reporting

3. Connection Manager (src/broker/connection_manager.py):
   - Health check for each broker every 60 seconds
   - Status: CONNECTED, DEGRADED, DISCONNECTED
   - Auto-reconnect with exponential backoff
   - Alert when broker disconnects
   - Dashboard widget showing status per broker

4. Order Manager (src/broker/order_manager.py):
   - Unified order placement:
     place_order(symbol, side, quantity, order_type, limit_price, broker_override=None)
     → Resolve broker → Validate order → Execute → Log → Return status
   - Order tracking across brokers
   - Cancel order by unified order ID
   - Order history with full details

KRAV:
- Parallel broker queries (asyncio.gather for positions/accounts)
- Timeout per broker: 10 seconds (don't let one slow broker block all)
- Cache positions: refresh every 30 seconds, force-refresh on trade
- Transaction ID mapping: unified ID → broker-specific ID
- Config-driven routing (YAML, easily changeable)
```

---

### PROMPT T5: Selskabsskat-modul

```
Byg et skatteberegnings-modul for dansk selskab (ApS/A/S).

KONTEKST:
- Oles firma har et skattetilgodehavende der skal udnyttes
- Selskaber er LAGERBESKATTET på finansielle aktiver (22% selskabsskat)
- Lagerbeskatning = skat på urealiserede gevinster ved regnskabsårets udgang
- Skattetilgodehavende kan fremføres og modregnes i fremtidig skattepligtig indkomst

NUVÆRENDE TILSTAND:
- src/tax/tax_calculator.py — beregner privat aktieskat (27%/42% progression)
- Skal ERSTATTES med selskabsskat-beregning

OPGAVE:

1. Corporate Tax Calculator (src/tax/corporate_tax.py):

   class CorporateTaxCalculator:
     tax_rate = 0.22  # 22% selskabsskat
     tax_credit: float  # Aktuelt skattetilgodehavende

   calculate_unrealized_pnl(positions, year_end_prices) → dict:
     - For hver position:
       - Primo-værdi (værdi ved regnskabsårets start eller anskaffelsessum)
       - Ultimo-værdi (værdi ved regnskabsårets udgang)
       - Urealiseret gevinst/tab = ultimo - primo
     - Return per-position og total urealiseret P&L

   calculate_realized_pnl(trades, year) → dict:
     - Alle realiserede gevinster/tab i regnskabsåret
     - FIFO-baseret for aktier
     - Inkluder: valutakursgevinster/-tab

   calculate_annual_tax(year) → TaxResult:
     - Samlet skattepligtig indkomst = urealiseret + realiseret P&L + udbytter - omkostninger
     - Brutto skat = indkomst × 22%
     - Skattetilgodehavende modregning
     - Netto skat = max(0, brutto - tilgodehavende)
     - Resterende tilgodehavende

   simulate_sale(symbol, quantity) → TaxImpact:
     - "Hvad sker der skattemæssigt hvis jeg sælger X?"
     - Vis: realiseret gevinst/tab, skatteeffekt, påvirkning af tilgodehavende

   suggest_tax_optimization() → list[TaxSuggestion]:
     - Analyser porteføljen for skatteoptimerings-muligheder:
       a. Positioner med urealiseret tab → "Sælg for at realisere tab og reducer skattepligtig indkomst"
       b. Positioner med urealiseret gevinst → "Overvej timing — realisér i år med tilgodehavende"
       c. Wash sale warnings → "Vent 30 dage før genkøb af X"
       d. Dividende-timing → "Ex-dividend dato nærmer sig for X"
     - Sortér efter skattemæssig impact (DKK)

2. Tax Credit Tracker (src/tax/tax_credit_tracker.py):
   - Track skattetilgodehavende over tid:
     - Start-balance (input manuelt)
     - Tilgang (tab-år tilføjer)
     - Forbrug (gevinst-år modregner)
     - Aktuel balance
   - Visualisering: Tilgodehavende over tid (bar chart)
   - Projection: "Med nuværende urealiseret P&L, hvad bliver tilgodehavendet ved årsskifte?"
   - Alert: "Tilgodehavende er næsten opbrugt — overvej at realisere tab"

3. Lagerbeskatning Engine (src/tax/mark_to_market.py):
   - Ved regnskabsårets udgang:
     - Hent alle positioners markedsværdi
     - Beregn urealiseret gevinst/tab vs primo (eller anskaffelse)
     - Beregn lagerbeskatning
     - Generér rapport

   - Løbende tracking:
     - "Estimeret lagerbeskatning YTD" (baseret på nuværende priser)
     - Opdateres dagligt
     - Dashboard widget

4. Dividend Tracker (src/tax/dividend_tracker.py):
   - Log alle udbytter: dato, symbol, brutto, kildeskat, netto
   - Udenlandsk kildeskat-kredit beregning (DBO-aftaler)
   - Lande med DBO: USA (15%), Tyskland (26.375%), UK (0%), etc.
   - Reclaimable withholding tax tracking

5. Currency P&L (src/tax/currency_pnl.py):
   - Valutakursgevinst/-tab er skattemæssigt relevant for selskaber
   - Track: Hver gang du køber/sælger i fremmed valuta
   - Beregn: FX gevinst/tab ved realisering
   - Aggregér per valuta per år

6. Tax Reports (src/tax/tax_reports.py):
   - Årsrapport (Excel):
     - Ark 1: Overblik (total P&L, skat, tilgodehavende)
     - Ark 2: Alle handler (transaktionslog)
     - Ark 3: Lagerbeskatning per position
     - Ark 4: Udbytter og kildeskat
     - Ark 5: Valutakursgevinster
     - Ark 6: Skattetilgodehavende bevægelse
   - Kvartalsrapport (summary)
   - Eksport til CSV for revisor

7. Dashboard Widgets:
   - Skattetilgodehavende gauge (brugt vs. resterende)
   - Estimeret skat ved årsskifte (baseret på nuværende priser)
   - "Skatteoptimerings-muligheder" liste
   - Per-position skatteimpact

ENVIRONMENT VARIABLES:
   TAX_CREDIT_INITIAL=500000  # Skattetilgodehavende i DKK
   TAX_YEAR_END=12-31  # Regnskabsårets slutning
   COMPANY_TAX_RATE=0.22

KRAV:
- Alle beregninger i DKK
- FX rates fra ECB eller Nationalbanken (officielle kurser til skat)
- FIFO strict (SKAT kræver FIFO for aktier)
- Audit trail: Alle skatteberegninger logges med inputs
- Eksportérbare rapporter
- DISCLAIMER: "Denne beregning er vejledende. Konsulter din revisor."
```

---

### PROMPT T6: Samlet Dashboard for Trader

```
Opgrader Dash dashboardet til multi-broker trading med skat.

NUVÆRENDE TILSTAND:
- src/dashboard/app.py (4 sider)
- Dash + Plotly + Bootstrap, dark theme, port 8050

OPGAVE:

Tilføj/opgrader disse sider:

1. Portfolio Overview (/):
   - TOTAL portfolio værdi (alle 4 brokers samlet) i DKK
   - Day P&L, MTD, YTD med % og absolut
   - Breakdown pie charts:
     - Per broker (Alpaca, Saxo, IBKR, Nordnet)
     - Per asset type (aktier, ETF, crypto, forex, råvarer, bonds, fonde)
     - Per valuta (DKK, USD, EUR, GBP, SEK, NOK, CHF)
     - Per land/region (US, DK, EU, UK, Norden, Emerging)
     - Per sektor
   - Positions table:
     | Symbol | Broker | Qty | Avg Cost | Current | P&L | P&L% | Weight | Skat* |
     * Skat-kolonne viser estimeret skatteeffekt ved salg
   - Performance chart vs benchmarks (S&P 500, STOXX 50, OMX C25)
   - Cash balance per broker

2. Trading (/trading):
   - Smart ordre-panel:
     - Symbol søgefelt med autocomplete (søg på tværs af alle brokers)
     - Vis: Navn, børs, valuta, broker der vil håndtere ordren
     - Side: BUY/SELL toggle
     - Type: Market/Limit/Stop dropdown
     - Antal + beløb-calculator ("køb for 10.000 DKK")
     - FX preview: "5.000 DKK ≈ $472 @ 10.59"
     - Skatteimpact preview: "Denne handel realiserer +5.200 DKK gevinst → ~1.144 DKK skat"
     - CONFIRM knap med summary
   - Open orders tabel (alle brokers) med cancel-knapper
   - Recent trades feed

3. Tax Center (/tax):
   - Skattetilgodehavende:
     - Stort tal: "Resterende: XXX.XXX DKK"
     - Bar chart: Tilgang/forbrug over tid
     - Projection: "Ved årsskifte estimeret: XXX.XXX DKK"
   - Lagerbeskatning estimat:
     - Estimeret urealiseret P&L per position
     - Total estimeret lagerbeskatning ved årsskifte
     - "Worst case" / "Best case" scenarier
   - Skatteoptimering:
     - Liste af foreslåede handler med skattemæssig begrundelse
     - "Realisér tab i X for at spare Y DKK i skat"
     - Wash sale warnings
   - Udbytteoversigt:
     - Modtagne udbytter YTD
     - Kildeskat betalt vs. reclaimable
   - Export-knap: Download årsrapport som Excel

4. Market Explorer (/markets):
   - Heatmap: Europæiske + US markeder
   - Instrument-søgning på tværs af alle brokers
   - Watchlists med live priser
   - Nyheder med sentiment scores

5. Broker Status (/status):
   - Connection status per broker (grøn/gul/rød)
   - Last successful sync timestamp
   - Positions count per broker
   - Cash balance per broker
   - API rate limit status
   - "Reconnect" knap per broker

6. Strategies (/strategies):
   - Eksisterende strategier (SMA, RSI, ML)
   - Signaler med "Execute" knap (semi-autonom)
   - Backtest resultater
   - Strategi performance over tid

KRAV:
- Auto-refresh: Positioner hvert 30 sek, priser hvert 10 sek
- Responsive callbacks (ingen blocking)
- Error handling: Vis venligt hvis en broker er disconnected
- Alle beløb i DKK med original valuta i parentes
- Dark theme (eksisterende farver: #0f1117, #00d4aa, #ff4757)
```

---

### PROMPT T7: Backup & Daglig Drift

```
Implementer backup og automatisering for daglig drift af Alpha Trader.

OPGAVE:

1. Daily Routine Scheduler (src/ops/daily_scheduler.py):
   Automatisk daglig rutine:

   07:30 CET: Morgen-check
     - Verificér alle broker connections
     - Hent overnight US markedsdata
     - Beregn portfolio value
     - Send morgen-email: "Portfolio: XXX.XXX DKK (+0.5%)"

   09:00 CET: EU marked åbner
     - Start live data feed for europæiske positioner
     - Kør eksisterende strategier (SMA, RSI)
     - Generér signaler

   15:30 CET: US marked åbner
     - Start live data feed for US positioner
     - Opdater cross-market analyse

   17:30 CET: EU marked lukker
     - Snapshot af europæiske positioner
     - Opdater skatteberegning
     - Log daglige movements

   22:00 CET: US marked lukker
     - Dagligt portfolio snapshot
     - Beregn daglig P&L (alle markeder)
     - Kør skatteoptimering check
     - Send aften-rapport email:
       - Daglig P&L per broker, per asset type
       - Signaler genereret
       - Skatteoptimerings-forslag
       - Broker health status

   23:00 CET: Vedligeholdelse
     - Database backup
     - Arkivér gamle logs
     - Disk space check

2. Email Reports (src/ops/email_reports.py):
   - Morgenrapport (kort): Portfolio value + overnight changes
   - Aftenrapport (detaljeret): Full P&L + signaler + skat
   - Ugentlig rapport: Performance summary + skat YTD
   - Alarm emails: Drawdown > 5%, broker disconnected, etc.

3. Backup (src/ops/backup.py):
   - Daglig PostgreSQL dump
   - Config filer backup
   - Gem til separat drive/mappe
   - 30 dages retention
   - Verificér backup integritet

4. Windows Autostart:
   - Task Scheduler script der starter:
     1. Docker (PostgreSQL, Redis)
     2. Python platform
     3. Dashboard
   - Auto-restart ved crash
   - Logfil for startup/shutdown

KRAV:
- Alt kører uovervåget
- Graceful handling af weekender/helligdage (spring trading over)
- Timezone-aware (CET for EU, ET for US)
- Logging af al scheduler-aktivitet
```

---

## 6. PRIORITERET HANDLINGSPLAN

### Uge 1-2: Intelligence + Broker Setup (parallelt)
1. Kør PROMPT T0 (Intelligence Engine) — Alpha Score, news pipeline
2. Kør PROMPT T0.5 (Claude API) — morning briefing, evening analysis
3. Opret konti: Saxo (developer), IBKR (paper), Nordnet
4. Kør PROMPT T1 (Saxo integration)

→ Efter uge 2: Du har daglige AI-briefings og kan handle via Alpaca + Saxo

### Uge 3-4: Flere Brokers & Routing
5. Kør PROMPT T2 (IBKR integration)
6. Kør PROMPT T3 (Nordnet integration)
7. Kør PROMPT T4 (BrokerRouter)
8. Test: Kan du se positioner fra alle 4 brokers + Alpha Scores i ét view?

### Uge 5-6: Skat
9. Kør PROMPT T5 (Selskabsskat)
10. Input skattetilgodehavende
11. Test: Korrekt lagerbeskatning beregning + skatteimpact per trade

### Uge 7-8: Dashboard & Drift
12. Kør PROMPT T6 (Dashboard — nu med intelligence widgets)
13. Kør PROMPT T7 (Backup & drift)
14. 2 ugers test i paper mode med fuld intelligence pipeline

### Uge 9+: Go Live
15. Skift fra paper til live (én broker ad gangen)
16. Start med Alpaca (allerede testet)
17. Derefter IBKR → Saxo → Nordnet
18. Track Alpha Score accuracy fra dag 1

---

## 7. BUDGET

| Post | Engangs | Løbende/år |
|------|---------|------------|
| Saxo developer konto | €0 | €0 |
| IBKR konto | €0 | €0 |
| Nordnet konto | €0 | €0 |
| IBKR markedsdata (EU) | — | €300-600 |
| Saxo markedsdata | — | Inkluderet |
| Nordnet markedsdata | — | Inkluderet |
| **Claude API (Anthropic)** | — | **€360-600** (~$30-50/mdr) |
| Finnhub premium (optional) | — | €0-600 |
| Server/hosting | €0 (lokal) | €0 |
| Strøm (normal PC) | — | ~€200 |
| **Total** | **€0** | **€860-2.000/år** |

Claude API er den nye udgift — men den erstatter hvad du ellers ville bruge
timer på at læse nyheder og analysere manuelt. €30-50/mdr for en 24/7
markedsanalytiker der aldrig sover er et godt trade.

---

## 8. HVAD DER GØR ALPHA TRADER MARKEDSLEDENDE

Alpha Trader slår ikke markedet med én ting — den slår det med **kombinationen**:

1. **Hastighed:** News pipeline scanner alle kilder hvert 15. minut.
   Du ser events og sentiment FØR du ville opdage dem manuelt.

2. **Bredde:** Alpha Score kombinerer 6 dimensioner (technical, sentiment,
   ML, macro, alt data, seasonality). Ingen menneskelig analytiker
   kan holde styr på alt det simultant for 5.000 symboler.

3. **Konsistens:** Ingen følelser, ingen træthed, ingen confirmation bias.
   Samme analyse-ramme kl 07:30 som kl 22:00.

4. **Kontekst:** Claude API forstår sammenhænge.
   "OPEC skærer produktion → olieprisen stiger → Equinor stiger →
   men transport-aktier falder → og din portefølje har 15% i shipping."
   Den forbindelse tager tid at lave manuelt. Platformen gør det på sekunder.

5. **Læringsloop:** Alpha Score accuracy trackes. Over tid justeres vægte.
   Modellerne forbedres ugentligt. Platformen ved hvad der virker og
   hvad der ikke gør — for DIN specifikke investeringsstil.

6. **Skattebevidsthed:** Ingen anden platform fortæller dig skatteeffekten
   FØR du handler. "Denne trade realiserer 50.000 DKK gevinst → 11.000 DKK skat.
   Men du har 200.000 DKK i tilgodehavende, så netto effekt: 0 DKK."

---

*Alpha Trader — Analyse. Handle. Slå markedet. Skatteoptimeret.*
