#!/usr/bin/env python3
"""
Patch main.py to run the scanner 24/7 using MarketCalendar.
Run once: python patch_main.py
"""
import re
from pathlib import Path

main_py = Path("main.py")
content = main_py.read_text()

OLD = '''def start_continuous_scanner(auto_trader, interval_minutes: int = 1) -> threading.Thread:
    """
    Kør AutoTrader scan hvert N. minut under markedstid.

    Markedstid (CET):
      08:00 - 22:30  (dækker pre-EU til post-US)

    Med 1-minut interval:
      = ~870 scans per handelsdag (near real-time)

    Data hentes som 1-minutters bars fra yfinance.
    """
    import threading
    from src.ops.daily_scheduler import is_market_day

    _stop = threading.Event()

    def _scanner_loop():
        while not _stop.is_set():
            now = datetime.now(TZ_CET)
            hour, minute = now.hour, now.minute
            today = now.date()

            # Markedstid: 08:00 - 22:30 CET på handelsdage
            in_market_hours = is_market_day(today) and (
                (hour == 8 and minute >= 0)
                or (9 <= hour <= 21)
                or (hour == 22 and minute <= 30)
            )

            # Crypto kører altid, men med 60 min interval uden for markedstid
            if in_market_hours:
                try:
                    logger.info(f"[scanner] ── Scan kl. {now:%H:%M} CET ──")
                    result = auto_trader.scan_and_trade()
                    trades = result.trades_executed
                    if trades > 0:
                        logger.info(f"[scanner] {trades} handler udført!")
                except Exception as e:
                    logger.error(f"[scanner] Scan fejl: {e}")

                # Vent 15 minutter (eller til stop)
                _stop.wait(timeout=interval_minutes * 60)

            else:
                # Uden for markedstid: sov 5 minutter og check igen
                _stop.wait(timeout=300)

    thread = threading.Thread(
        target=_scanner_loop,
        name="ContinuousScanner",
        daemon=True,
    )
    thread.start()
    return thread'''

NEW = '''def start_continuous_scanner(auto_trader, interval_minutes: int = 1) -> threading.Thread:
    """
    Kør AutoTrader scan hvert N. minut — 24/7 global market coverage.

    Bruger MarketCalendar til at bestemme hvilke markeder der er åbne.
    Scanner kører altid på handelsdage — MarketCalendar filtrerer symboler.

    Markedsdækning (CET):
      22:00 - 03:00  New Zealand
      01:00 - 07:30  Tokyo + Sydney
      02:00 - 08:00  Hong Kong
      04:45 - 11:15  Mumbai
      09:00 - 17:30  EU + Nordic + London
      10:00 - 15:30  US Pre-market
      15:30 - 22:00  US Regular
      22:00 - 02:00  US Post-market
      24/7           Crypto
    """
    import threading
    from src.ops.daily_scheduler import is_market_day

    _stop = threading.Event()

    def _scanner_loop():
        while not _stop.is_set():
            now = datetime.now(TZ_CET)
            today = now.date()

            # Scanner kører alle handelsdage — MarketCalendar håndterer hvilke symboler
            if is_market_day(today):
                try:
                    # Check hvilke markeder er åbne nu
                    open_markets = []
                    try:
                        from src.ops.market_calendar import get_calendar
                        open_markets = get_calendar().get_open_markets(now)
                    except Exception:
                        pass

                    logger.info(
                        f"[scanner] ── Scan kl. {now:%H:%M} CET ──"
                        f" Åbne markeder: {open_markets or ['alle']}"
                    )
                    result = auto_trader.scan_and_trade()
                    trades = result.trades_executed
                    if trades > 0:
                        logger.info(f"[scanner] {trades} handler udført!")
                except Exception as e:
                    logger.error(f"[scanner] Scan fejl: {e}")

                _stop.wait(timeout=interval_minutes * 60)

            else:
                # Weekend — kun crypto scanner hvert 5. minut
                try:
                    logger.debug(f"[scanner] Weekend — kun crypto scan {now:%H:%M}")
                    result = auto_trader.scan_and_trade()
                    if result.trades_executed > 0:
                        logger.info(f"[scanner] Crypto: {result.trades_executed} handler")
                except Exception as e:
                    logger.error(f"[scanner] Weekend scan fejl: {e}")
                _stop.wait(timeout=300)

    thread = threading.Thread(
        target=_scanner_loop,
        name="ContinuousScanner",
        daemon=True,
    )
    thread.start()
    return thread'''

if OLD in content:
    content = content.replace(OLD, NEW)
    # Also update the startup summary log line
    content = content.replace(
        'logger.info(f"  ✓ Scanner: hvert 1. minut, 08:00-22:30 CET (~870 scans/dag, LIVE)")',
        'logger.info(f"  ✓ Scanner: hvert 1. minut, 24/7 global market coverage (MarketCalendar)")'
    )
    main_py.write_text(content)
    print("✅ main.py patched successfully — scanner now runs 24/7")
else:
    print("❌ Could not find target function — may already be patched or different version")
    print("   Try manual edit of start_continuous_scanner() in main.py")
