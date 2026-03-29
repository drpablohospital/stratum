from pathlib import Path
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import plotly.graph_objects as go
from nicegui import app, ui

BASE_DIR = Path(__file__).resolve().parent
ASSET_BG = BASE_DIR / 'fondo.png'
AMBIENT_STATE_FILE = BASE_DIR / 'ambient_state.json'
LIVE_TERMINAL_FILE = BASE_DIR / 'live_terminal.txt'
APOLO_PROGRESS_FILE = BASE_DIR / 'optimizer_progress.txt'
APOLO_BEST_FILE = BASE_DIR / 'best_params.json'

STATE = {
    'last_terminal_line': '[idle] waiting for signal...',
    'headline': 'Awaiting signal',
    'strategy': 'config actual',
    'apolo_status': 'IDLE',
    'bot_status': 'LISTENING',
    'best_score': '—',
    'updated_at': '—',
    'terminal_lines': ['[idle] waiting for signal...'],
    'btc_price': '—',
    'btc_change_day': '—',
    'btc_day_range': '—',
    'btc_trend': 'Neutral',
}

hero_label = None
meta_label = None
strategy_label = None
apolo_label = None
bot_label = None
best_label = None
terminal_box = None
candle_box = None
signal_label = None
btc_price_label = None
btc_change_label = None
btc_range_label = None
btc_trend_label = None
clock_label = None


def safe_read_text(path: Path, fallback: str = '') -> str:
    try:
        if path.exists():
            return path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        pass
    return fallback


def load_shared_state() -> dict:
    try:
        if AMBIENT_STATE_FILE.exists():
            return json.loads(AMBIENT_STATE_FILE.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}


def tail_lines(path: Path, limit: int = 40) -> list[str]:
    text = safe_read_text(path)
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()][-limit:]


def detect_best_score() -> str:
    try:
        if not APOLO_BEST_FILE.exists():
            return '—'
        data = json.loads(APOLO_BEST_FILE.read_text(encoding='utf-8'))
        for key in ('score', 'best_score', 'objective', 'value'):
            if key in data:
                return str(data[key])
        return 'saved'
    except Exception:
        return '—'


def detect_apolo_status(shared: dict, last_line: str) -> str:
    if shared.get('apolo_running'):
        return 'ACTIVE'
    progress = safe_read_text(APOLO_PROGRESS_FILE).lower()
    if 'complet' in progress or 'finished' in progress:
        return 'DONE'
    if '[apolo]' in last_line.lower():
        return 'ACTIVE'
    return 'IDLE'


def detect_bot_status(shared: dict, last_line: str) -> str:
    if shared.get('bot_running'):
        return 'ACTIVE'
    low = last_line.lower()
    if '[error]' in low or 'traceback' in low:
        return 'ERROR'
    if '[backtest]' in low:
        return 'BACKTEST'
    if '[bot]' in low:
        return 'ACTIVE'
    return 'LISTENING'


def extract_headline(line: str) -> str:
    if not line:
        return 'Awaiting signal'
    cleaned = line.strip()
    for prefix in ('[news]', '[watch]', '[apolo]', '[bot]', '[backtest]', '[report]', '[config]'):
        if cleaned.lower().startswith(prefix):
            return cleaned[len(prefix):].strip() or 'Awaiting signal'
    return cleaned


def signal_mode(line: str) -> str:
    low = line.lower()
    if low.startswith('[news]'):
        return 'Market intelligence'
    if low.startswith('[apolo]'):
        return 'Optimization trace'
    if low.startswith('[bot]'):
        return 'Execution trace'
    if low.startswith('[backtest]'):
        return 'Backtest trace'
    if low.startswith('[error]'):
        return 'Critical state'
    return 'System listening'


def build_terminal_markup(lines: list[str]) -> str:
    shown = list(reversed(lines[-8:])) if lines else ['[idle] waiting for signal...']
    body = ''.join(f'<div class="ambient-log-line">{line}</div>' for line in shown)
    return f'''
    <div class="ambient-terminal-glass">
      <div class="ambient-terminal-head">
        <span class="ambient-dot coral"></span>
        <span class="ambient-dot peach"></span>
        <span class="ambient-dot cyan"></span>
        <span class="ambient-terminal-title">LIVE TRACE</span>
      </div>
      <div class="ambient-terminal-body">{body}</div>
    </div>
    '''


def fetch_binance_klines(symbol: str = 'BTCUSDT', interval: str = '15m', limit: int = 140) -> pd.DataFrame:
    params = urlencode({'symbol': symbol, 'interval': interval, 'limit': limit})
    url = f'https://api.binance.com/api/v3/klines?{params}'
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0 STRATUM/1.0'})
    with urlopen(req, timeout=12) as response:
        raw = json.loads(response.read().decode('utf-8'))

    cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ]
    df = pd.DataFrame(raw, columns=cols)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]


def update_market_stats(df: pd.DataFrame) -> None:
    if df.empty:
        return

    last_close = float(df['close'].iloc[-1])
    first_open = float(df['open'].iloc[0])
    day_change_pct = ((last_close - first_open) / first_open) * 100 if first_open else 0.0
    day_high = float(df['high'].max())
    day_low = float(df['low'].min())

    sma_fast = df['close'].tail(12).mean()
    sma_slow = df['close'].tail(36).mean()
    trend = 'Bullish drift' if sma_fast > sma_slow else 'Bearish drift' if sma_fast < sma_slow else 'Neutral'

    STATE['btc_price'] = f'{last_close:,.2f}'
    STATE['btc_change_day'] = f'{day_change_pct:+.2f}%'
    STATE['btc_day_range'] = f'{day_low:,.0f} — {day_high:,.0f}'
    STATE['btc_trend'] = trend


def build_candles() -> go.Figure:
    df = fetch_binance_klines(interval='15m', limit=140)
    update_market_stats(df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['open_time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='#88d6e2',
        decreasing_line_color='#e6a48d',
        increasing_fillcolor='rgba(136,214,226,.70)',
        decreasing_fillcolor='rgba(230,164,141,.62)',
        whiskerwidth=0.40,
        name='BTC 15m',
    ))

    # Suavemente añadimos una media corta para sugerir dirección
    ma = df['close'].rolling(12).mean()
    fig.add_trace(go.Scatter(
        x=df['open_time'],
        y=ma,
        mode='lines',
        line=dict(color='rgba(35,68,107,.55)', width=2),
        name='MA 12',
        hoverinfo='skip',
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            visible=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=False,
            visible=False,
        ),
    )
    return fig


def render_candles() -> None:
    if candle_box is None:
        return
    try:
        candle_box.clear()
        with candle_box:
            ui.plotly(build_candles()).classes('w-full h-64')
    except Exception:
        pass


def apply_state() -> None:
    if hero_label:
        hero_label.set_text(STATE['headline'])
    if signal_label:
        signal_label.set_text(signal_mode(STATE['last_terminal_line']))
    if strategy_label:
        strategy_label.set_text(f'STRATEGY · {STATE["strategy"]}')
    if meta_label:
        meta_label.set_text(f'Updated {STATE["updated_at"]} · DOGMA COIN SAPI DE CV')
    if apolo_label:
        apolo_label.set_text(f'APOLO {STATE["apolo_status"]}')
    if bot_label:
        bot_label.set_text(f'BOT {STATE["bot_status"]}')
    if best_label:
        best_label.set_text(f'BEST SCORE {STATE["best_score"]}')
    if btc_price_label:
        btc_price_label.set_text(STATE['btc_price'])
    if btc_change_label:
        btc_change_label.set_text(STATE['btc_change_day'])
    if btc_range_label:
        btc_range_label.set_text(STATE['btc_day_range'])
    if btc_trend_label:
        btc_trend_label.set_text(STATE['btc_trend'])
    if clock_label:
        clock_label.set_text(time.strftime('%H:%M:%S'))
    if terminal_box:
        terminal_box.set_content(build_terminal_markup(STATE['terminal_lines']))


def refresh_state() -> None:
    shared = load_shared_state()
    lines = shared.get('last_terminal_lines') or tail_lines(LIVE_TERMINAL_FILE, 40)
    last_line = shared.get('last_terminal_line') or (lines[-1] if lines else STATE['last_terminal_line'])

    STATE['last_terminal_line'] = last_line
    STATE['terminal_lines'] = lines[-40:] if lines else [STATE['last_terminal_line']]
    STATE['headline'] = extract_headline(last_line)
    STATE['strategy'] = shared.get('strategy', STATE['strategy'])
    STATE['apolo_status'] = detect_apolo_status(shared, last_line)
    STATE['bot_status'] = detect_bot_status(shared, last_line)
    STATE['best_score'] = detect_best_score()
    STATE['updated_at'] = shared.get('updated_at', time.strftime('%H:%M:%S'))
    apply_state()


def add_styles() -> None:
    bg_url = '/assets/bg' if ASSET_BG.exists() else ''
    ui.add_head_html(f'''
    <style>
      body, .nicegui-content {{
        margin: 0;
        background:
          linear-gradient(180deg, rgba(247,243,236,.84), rgba(235,230,220,.90)),
          url('{bg_url}');
        background-size: cover;
        background-position: center;
        color: #122737;
        font-family: Inter, Segoe UI, Arial, sans-serif;
      }}

      .ambient-shell {{
        min-height: 100vh;
        padding: 18px;
      }}

      .ambient-surface {{
        min-height: calc(100vh - 36px);
        border-radius: 38px;
        background:
          linear-gradient(180deg, rgba(255,255,255,.20), rgba(255,255,255,.08)),
          linear-gradient(135deg, rgba(255,255,255,.14), transparent 36%);
        border: 1px solid rgba(255,255,255,.24);
        box-shadow: 0 28px 90px rgba(35,68,107,.10), inset 0 1px 0 rgba(255,255,255,.34);
        backdrop-filter: blur(22px);
        overflow: hidden;
        position: relative;
      }}

      .ambient-surface::before {{
        content: '';
        position: absolute;
        inset: 0;
        background:
          radial-gradient(circle at 16% 18%, rgba(168,221,229,.16), transparent 22%),
          radial-gradient(circle at 86% 14%, rgba(233,138,107,.10), transparent 18%),
          radial-gradient(circle at 74% 80%, rgba(217,221,242,.14), transparent 18%);
        pointer-events: none;
      }}

      .ambient-grid {{
        position: relative;
        z-index: 1;
        min-height: calc(100vh - 36px);
        display: grid;
        grid-template-columns: 1.08fr .92fr;
      }}

      .ambient-left {{
        padding: 34px 36px 28px;
        display: grid;
        grid-template-rows: auto auto 1fr auto;
        gap: 20px;
      }}

      .ambient-right {{
        padding: 24px 28px 24px 8px;
        display: grid;
        align-items: stretch;
      }}

      .brand-row {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
      }}

      .brand-wrap {{
        display: grid;
        gap: 6px;
      }}

      .brand-title {{
        font-size: 32px;
        font-weight: 800;
        letter-spacing: .16em;
        color: #143148;
      }}

      .brand-sub {{
        font-size: 11px;
        letter-spacing: .18em;
        text-transform: uppercase;
        color: #5c7785;
        font-family: Fira Code, Consolas, monospace;
      }}

      .top-right-wrap {{
        display: grid;
        gap: 10px;
        justify-items: end;
      }}

      .strategy-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 999px;
        background: linear-gradient(180deg, rgba(255,255,255,.42), rgba(255,255,255,.22));
        border: 1px solid rgba(255,255,255,.42);
        color: #22436c;
        font-size: 12px;
        font-weight: 700;
        backdrop-filter: blur(14px);
        box-shadow: 0 10px 28px rgba(35,68,107,.06);
      }}

      .ambient-nav-btn {{
        border-radius: 999px;
        padding: 10px 14px;
        font-size: 12px;
        font-weight: 700;
        font-family: Fira Code, Consolas, monospace;
        letter-spacing: .05em;
        background: linear-gradient(180deg, rgba(255,255,255,.38), rgba(255,255,255,.18));
        color: #173149;
        border: 1px solid rgba(255,255,255,.34);
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 18px rgba(35,68,107,.06);
      }}

      .eyebrow {{
        color: #597382;
        text-transform: uppercase;
        letter-spacing: .20em;
        font-size: 11px;
        font-family: Fira Code, Consolas, monospace;
      }}

      .hero-block {{
        display: grid;
        gap: 12px;
        align-content: start;
      }}

      .hero-text {{
        font-size: clamp(32px, 5vw, 72px);
        line-height: 1.0;
        letter-spacing: -.05em;
        font-weight: 800;
        max-width: 12ch;
        text-wrap: balance;
        background: linear-gradient(135deg, #163149 0%, #2f6ea2 38%, #d18468 72%, #e6a48d 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
      }}

      .signal-label {{
        color: #587385;
        font-size: 12px;
        font-family: Fira Code, Consolas, monospace;
      }}

      .meta-row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
      }}

      .soft-chip {{
        display: inline-flex;
        align-items: center;
        padding: 10px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .06em;
        font-family: Fira Code, Consolas, monospace;
        background: linear-gradient(180deg, rgba(255,255,255,.28), rgba(255,255,255,.12));
        border: 1px solid rgba(255,255,255,.24);
        color: #28465e;
        backdrop-filter: blur(14px);
      }}

      .meta-line {{
        color: #5f7583;
        font-size: 12px;
        font-family: Fira Code, Consolas, monospace;
      }}

      .clock-line {{
        color: #2b5570;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: .08em;
        font-family: Fira Code, Consolas, monospace;
      }}

      .float-panel {{
        height: 100%;
        border-radius: 34px;
        background: linear-gradient(180deg, rgba(255,255,255,.24), rgba(255,255,255,.10));
        border: 1px solid rgba(255,255,255,.24);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.34), 0 18px 50px rgba(35,68,107,.08);
        backdrop-filter: blur(18px);
        padding: 16px;
        display: grid;
        grid-template-rows: auto auto auto 1fr;
        gap: 14px;
      }}

      .panel-kicker {{
        color: #5a7382;
        text-transform: uppercase;
        letter-spacing: .18em;
        font-size: 11px;
        font-family: Fira Code, Consolas, monospace;
      }}

      .market-stats {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
      }}

      .stat-card {{
        border-radius: 22px;
        padding: 12px 14px;
        background: linear-gradient(180deg, rgba(255,255,255,.28), rgba(255,255,255,.12));
        border: 1px solid rgba(255,255,255,.24);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.26);
      }}

      .stat-k {{
        color: #5f7686;
        font-size: 10px;
        letter-spacing: .16em;
        text-transform: uppercase;
        font-family: Fira Code, Consolas, monospace;
      }}

      .stat-v {{
        margin-top: 6px;
        color: #173149;
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -.03em;
      }}

      .candle-glass {{
        border-radius: 26px;
        padding: 10px 10px 2px;
        background: linear-gradient(180deg, rgba(255,255,255,.24), rgba(255,255,255,.06));
        border: 1px solid rgba(255,255,255,.22);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.26);
      }}

      .ambient-terminal-glass {{
        border-radius: 24px;
        overflow: hidden;
        background: linear-gradient(180deg, rgba(15,24,34,.96), rgba(10,18,27,.96));
        border: 1px solid rgba(110,203,232,.12);
        box-shadow: 0 18px 46px rgba(12,22,34,.18);
      }}

      .ambient-terminal-head {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-bottom: 1px solid rgba(110,203,232,.10);
        background: rgba(255,255,255,.02);
      }}

      .ambient-dot {{
        width: 9px;
        height: 9px;
        border-radius: 999px;
        display: inline-block;
      }}

      .ambient-dot.coral {{ background: #e98a6b; }}
      .ambient-dot.peach {{ background: #edb08a; }}
      .ambient-dot.cyan {{ background: #6ecbe8; }}

      .ambient-terminal-title {{
        margin-left: 8px;
        color: #8fb3c1;
        font-size: 11px;
        letter-spacing: .16em;
        font-family: Fira Code, Consolas, monospace;
      }}

      .ambient-terminal-body {{
        padding: 14px;
        display: grid;
        gap: 4px;
      }}

      .ambient-log-line {{
        color: #dff6ff;
        font-size: 12px;
        line-height: 1.55;
        font-family: Fira Code, Consolas, monospace;
        padding: 3px 0;
        border-bottom: 1px solid rgba(255,255,255,.03);
      }}

      @media (max-width: 1200px) {{
        .ambient-grid {{ grid-template-columns: 1fr; }}
        .ambient-left {{ padding-bottom: 10px; }}
        .ambient-right {{ padding: 0 22px 22px; }}
        .hero-text {{ max-width: 100%; }}
        .market-stats {{ grid-template-columns: 1fr 1fr; }}
      }}
    </style>
    ''')


@app.get('/assets/bg')
def get_bg_asset():
    return ui.file_response(ASSET_BG) if ASSET_BG.exists() else None


@ui.page('/')
def ambient_page() -> None:
    global hero_label, meta_label, strategy_label, apolo_label, bot_label, best_label
    global terminal_box, candle_box, signal_label
    global btc_price_label, btc_change_label, btc_range_label, btc_trend_label, clock_label

    add_styles()

    with ui.column().classes('ambient-shell w-full'):
        with ui.element('div').classes('ambient-surface'):
            with ui.element('div').classes('ambient-grid'):

                with ui.element('div').classes('ambient-left'):
                    with ui.element('div').classes('brand-row'):
                        with ui.element('div').classes('brand-wrap'):
                            ui.label('STRATUM').classes('brand-title')
                            ui.label('by DOGMA · ambient decision surface').classes('brand-sub')

                        with ui.element('div').classes('top-right-wrap'):
                            strategy_label = ui.label('STRATEGY · config actual').classes('strategy-pill')
                            ui.button('Abrir STRATUM GUI', on_click=lambda: ui.navigate.to('http://127.0.0.1:8081')).classes('ambient-nav-btn')

                    with ui.element('div').classes('hero-block'):
                        ui.label('Latest signal').classes('eyebrow')
                        hero_label = ui.label('Awaiting signal').classes('hero-text')
                        signal_label = ui.label('System listening').classes('signal-label')
                        clock_label = ui.label(time.strftime('%H:%M:%S')).classes('clock-line')

                    with ui.element('div').classes('meta-row'):
                        apolo_label = ui.label('APOLO IDLE').classes('soft-chip')
                        bot_label = ui.label('BOT LISTENING').classes('soft-chip')
                        best_label = ui.label('BEST SCORE —').classes('soft-chip')

                    meta_label = ui.label('Updated —').classes('meta-line')

                with ui.element('div').classes('ambient-right'):
                    with ui.element('div').classes('float-panel'):
                        ui.label('BTC market pulse · 15m').classes('panel-kicker')

                        with ui.element('div').classes('market-stats'):
                            with ui.element('div').classes('stat-card'):
                                ui.label('Last price').classes('stat-k')
                                btc_price_label = ui.label('—').classes('stat-v')

                            with ui.element('div').classes('stat-card'):
                                ui.label('Day change').classes('stat-k')
                                btc_change_label = ui.label('—').classes('stat-v')

                            with ui.element('div').classes('stat-card'):
                                ui.label('Day range').classes('stat-k')
                                btc_range_label = ui.label('—').classes('stat-v')

                            with ui.element('div').classes('stat-card'):
                                ui.label('Trend').classes('stat-k')
                                btc_trend_label = ui.label('Neutral').classes('stat-v')

                        with ui.element('div').classes('candle-glass'):
                            candle_box = ui.column().classes('w-full')

                        terminal_box = ui.html(build_terminal_markup(STATE['terminal_lines'])).classes('w-full')

    refresh_state()
    render_candles()
    ui.timer(2.0, refresh_state)
    ui.timer(30.0, render_candles)

ui.run(title='STRATUM Idle', favicon='◉', port=8082, reload=False)
