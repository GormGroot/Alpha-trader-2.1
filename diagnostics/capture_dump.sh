#!/bin/bash
# Quick diagnostic dump — run anytime to capture system state
cd ~/tmp/Traide/Trade.1.0/alpha-trading-platform-main
source venv/bin/activate
python -c "
import json, sqlite3, psutil
from pathlib import Path
from datetime import datetime

diag = {'timestamp': datetime.now().isoformat(), 'errors': []}

# DBs
for db_name in ['paper_portfolio.db', 'auto_trader_log.db', 'signal_log.db', 'learning.db']:
    db_path = Path(f'data_cache/{db_name}')
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                tables = conn.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall()
                diag[db_name] = {t[0]: conn.execute(f'SELECT COUNT(*) FROM {t[0]}').fetchone()[0] for t in tables}
        except Exception as e:
            diag['errors'].append(f'{db_name}: {e}')

# Portfolio
try:
    from src.broker.paper_broker import PaperBroker
    pb = PaperBroker()
    acc = pb.get_account()
    positions = pb.get_positions()
    diag['portfolio'] = {
        'cash': round(acc.cash, 2), 'equity': round(acc.equity, 2),
        'num_positions': len(positions),
        'positions': [{'symbol': p.symbol, 'qty': p.qty, 'pnl': round(getattr(p, 'unrealized_pnl', 0) or 0, 2)} for p in positions],
    }
except Exception as e:
    diag['errors'].append(f'portfolio: {e}')

# Configs
for cfg in ['exchange_stop_loss.json', 'global_stop_loss.json', 'max_positions.json', 'risk_sizing.json']:
    p = Path(f'config/{cfg}')
    diag[f'config/{cfg}'] = json.loads(p.read_text()) if p.exists() else 'NOT FOUND'

# AutoTrader
try:
    from src.broker.registry import get_auto_trader
    trader = get_auto_trader()
    if trader:
        rm = getattr(trader, '_risk_manager', None)
        diag['auto_trader'] = {'running': True, 'scans': getattr(trader, '_total_scans', 0)}
        if rm: diag['risk_manager'] = {'sl': rm.stop_loss_pct, 'pos_pct': rm.max_position_pct, 'max_pos': rm.max_open_positions}
    else:
        diag['auto_trader'] = 'NOT RUNNING'
except Exception as e:
    diag['errors'].append(f'trader: {e}')

# Markets
try:
    from src.ops.market_calendar import MarketCalendar
    diag['open_markets'] = list(MarketCalendar(include_pre_market=True, include_post_market=True).get_open_markets())
except: pass

# Equity + trades summary
try:
    with sqlite3.connect('data_cache/paper_portfolio.db') as conn:
        eq = conn.execute('SELECT MIN(equity), MAX(equity), COUNT(*) FROM equity_history').fetchone()
        diag['equity_range'] = {'min': eq[0], 'max': eq[1], 'count': eq[2]}
        tr = conn.execute('SELECT COUNT(*), SUM(realized_pnl) FROM closed_trades').fetchone()
        diag['trades'] = {'count': tr[0], 'total_pnl': round(tr[1] or 0, 2)}
except: pass

# System
diag['system'] = {'cpu': psutil.cpu_percent(), 'mem_gb': round(psutil.virtual_memory().used/1e9, 1), 'mem_pct': psutil.virtual_memory().percent}

out = Path('diagnostics/dump_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
out.write_text(json.dumps(diag, indent=2, default=str))
print(f'Saved: {out}')
print(f'Errors: {len(diag[\"errors\"])}')
for e in diag['errors']: print(f'  > {e}')
p = diag.get('portfolio', {})
print(f'Portfolio: {p.get(\"num_positions\",\"?\")} pos, \${p.get(\"cash\",\"?\"):,} cash, \${p.get(\"equity\",\"?\"):,} equity')
print(f'RAM: {diag[\"system\"][\"mem_gb\"]}GB ({diag[\"system\"][\"mem_pct\"]}%)')
"
