#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Alpha Trader — Dobbeltklik for at starte
# ═══════════════════════════════════════════════════════════
#
#  Første gang: Installerer alt automatisk
#  Derefter:    Starter dashboard på http://localhost:8050
#

# Gå til projektmappen (samme mappe som dette script)
cd "$(dirname "$0")"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Alpha Trader — Starter op..."
echo "══════════════════════════════════════════════════"
echo ""

# ── 1. Check Python ──────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 er ikke installeret."
    echo ""
    echo "   Installer det her:"
    echo "   https://www.python.org/downloads/"
    echo ""
    echo "   Eller kør: brew install python3"
    echo ""
    read -p "Tryk Enter for at lukke..."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "✓ $PYTHON_VERSION fundet"

# ── 2. Opret virtual environment (kun første gang) ───────
if [ ! -d ".venv" ]; then
    echo ""
    echo "→ Første gang — opretter virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment oprettet"
fi

# Aktivér virtual environment
source .venv/bin/activate
echo "✓ Virtual environment aktiveret"

# ── 3. Installer dependencies (kun hvis nødvendigt) ──────
if [ ! -f ".venv/.deps_installed" ]; then
    echo ""
    echo "→ Installerer dependencies (tager 1-2 minutter første gang)..."
    echo ""
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    pip install -r requirements-trader.txt -q

    if [ $? -eq 0 ]; then
        touch .venv/.deps_installed
        echo ""
        echo "✓ Alle dependencies installeret"
    else
        echo ""
        echo "❌ Fejl under installation. Prøv manuelt:"
        echo "   pip install -r requirements.txt -r requirements-trader.txt"
        read -p "Tryk Enter for at lukke..."
        exit 1
    fi
fi

# ── 4. Check .env fil ───────────────────────────────────
if [ ! -f ".env" ]; then
    echo ""
    echo "❌ .env fil mangler!"
    echo "   Kopiér .env.example og udfyld dine API keys:"
    echo "   cp .env.example .env"
    read -p "Tryk Enter for at lukke..."
    exit 1
fi
echo "✓ .env konfiguration fundet"

# ── 5. Start! ───────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  ✓ Alt klar — starter Alpha Trader"
echo "══════════════════════════════════════════════════"
echo ""
echo "  Dashboard:  http://localhost:8050"
echo "  Stop:       Tryk Ctrl+C i dette vindue"
echo ""
echo "  Åbner browser om 3 sekunder..."
echo ""

# Åbn browser automatisk efter 3 sek
(sleep 3 && open "http://localhost:8050") &

# Start platformen
python3 main.py --mode trader --paper

# Hvis den stopper
echo ""
echo "Alpha Trader stoppet."
read -p "Tryk Enter for at lukke..."
