from __future__ import annotations

import copy
import html
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nicegui import ui, app

BASE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BASE_DIR / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)

PARAMS_FILE = CONFIGS_DIR / "params_bt.json"
DEFAULT_CONFIG_FILE = CONFIGS_DIR / "config.json"

APOLO_PROGRESS_FILE = BASE_DIR / "optimizer_progress.txt"
APOLO_BEST_FILE = BASE_DIR / "best_params.json"
APOLO_RESULTS_FILE = BASE_DIR / "optimizer_results.csv"
AMBIENT_STATE_FILE = BASE_DIR / "ambient_state.json"
LIVE_TERMINAL_FILE = BASE_DIR / "live_terminal.txt"

BOT_SCRIPT = BASE_DIR / "stratum.py"

import traceback

run_backtest = None
import_errors = []

for module_name in ("stratum_bt", "argentum_bt"):
    try:
        mod = __import__(module_name, fromlist=["run_backtest"])
        run_backtest = getattr(mod, "run_backtest")
        print(f"[IMPORT] Backtester cargado desde {module_name}.py")
        break
    except Exception as e:
        import_errors.append((module_name, repr(e), traceback.format_exc()))

if run_backtest is None:
    print("Error: no se pudo importar ningún backtester.")
    for module_name, err, tb in import_errors:
        print(f"\n--- Error importando {module_name}.py ---")
        print(err)
        print(tb)
    sys.exit(1)

DEFAULT_CONFIG: dict[str, Any] = {
    "symbol": "BTC/USDT:USDT",
    "market_symbol": "BTCUSDT",
    "poll_seconds": 10,
    "status_heartbeat_seconds": 60,
    "tf_signal": "15m",
    "tf_confirm": "5m",
    "tf_entry": "5m",
    "tf_trend": "1h",
    "live_chart_tf": "15m",
    "mode": "feature_prob",  # feature_prob | legacy_classic
    "use_1h_trend_filter": True,
    "feature_prob": {
        "ret_1_weight": 0.10,
        "ret_3_weight": 0.12,
        "ret_12_weight": 0.14,
        "range_pos_weight": 0.10,
        "vol_ratio_weight": 0.08,
        "realized_vol_weight": 0.08,
        "slope_weight": 0.12,
        "trend_weight": 0.16,
        "breakout_weight": 0.10,
        "long_threshold": 0.58,
        "short_threshold": 0.58,
        "persistence_bars": 2,
    },
    "risk_engine": {
        "sl_vol_mult": 1.20,
        "tp_vol_mult": 2.80,
        "breakeven_R": 1.0,
        "trail_start_R": 1.5,
        "trail_vol_mult": 1.0,
        "risk_per_trade_pct": 0.50,
        "max_hold_minutes": 360,
        "cooldown_minutes": 180,
    },
    "execution": {
        "fees_enabled": True,
        "fee_rate_roundtrip": 0.001,
        "slippage_roundtrip_pct": 0.0001,
        "label_exits": True,
    },
    "legacy": {
        "ema_fast_trend": 50,
        "ema_slow_trend": 200,
        "ema_fast_signal": 50,
        "ema_slow_signal": 200,
        "donchian_window_signal": 96,
        "rsi_len_signal": 14,
        "rsi_long": 58,
        "rsi_short": 42,
        "atr_len_signal": 14,
        "atr_ma_window_signal": 96,
        "atr_min_mult": 1.25,
        "breakout_buffer_atr": 0.55,
        "sl_atr_mult": 1.10,
        "tp_atr_mult": 3.20,
        "breakeven_R": 1.0,
        "trail_start_R": 1.5,
        "trail_atr_mult": 1.0,
    },
}

config: dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)

output_log: list[str] = []
stats_data: dict[str, Any] | None = None
trades_data: pd.DataFrame | None = None
raw_report_text = ""
current_strategy_name = "default_feature"
show_backtest_overlay = False

terminal_html = None
stats_card = None
raw_report_box = None
plot_eq = None
plot_pie = None
plot_hist = None
plot_scatter = None
plot_bar = None
header_status = None
market_watch_label = None
news_source_label = None
strategy_name_label = None
live_chart_box = None
apolo_status_badge = None
apolo_progress_label = None
apolo_best_box = None
apolo_meta_label = None

param_controls: dict[str, Any] = {}
param_value_chips: dict[str, Any] = {}

pending_logs: queue.Queue[str] = queue.Queue()
pending_results: queue.Queue[tuple[str, Any, Any]] = queue.Queue()

last_log_time = time.time()
last_ui_idle_emit = 0.0
idle_message_index = 0

apolo_process: subprocess.Popen | None = None
apolo_started_at: float | None = None
apolo_last_output_at: float | None = None
last_apolo_line: str | None = None

trading_process: subprocess.Popen | None = None
trading_started_at: float | None = None
trading_last_output_at: float | None = None
last_trading_line: str | None = None

news_items: list[dict[str, str]] = []
news_last_fetch_at: float | None = None
news_last_emit_at = 0.0
news_cursor = 0
news_refresh_in_progress = False

last_chart_refresh_at = 0.0
chart_refresh_in_progress = False

PALETTE = {
    "bg_main": "#F5F1E8",
    "bg_soft": "#EEE8DC",
    "panel": "#FBF8F2",
    "panel_alt": "#FFFDF9",
    "text_main": "#111111",
    "text_soft": "#4A4A4A",
    "text_muted": "#7A746A",
    "border": "#D8D1C5",
    "border_strong": "#C8BFAF",
    "primary_blue": "#2373B8",
    "primary_blue_dark": "#23446B",
    "accent_cyan": "#A8DDE5",
    "accent_coral": "#E98A6B",
    "accent_peach": "#EDB08A",
    "accent_pink": "#E7C7D6",
    "accent_lavender": "#D9DDF2",
    "terminal_bg": "#101923",
    "terminal_panel": "#162331",
    "terminal_text": "#DFF6FF",
    "terminal_muted": "#8FB3C1",
    "terminal_accent": "#6ECBE8",
}

IDLE_MARKET_LINES = [
    "[watch] XAU/USD en observación. Oro, dólar y liquidez global bajo radar.",
    "[watch] petróleo, shipping y riesgo geopolítico: monitor pasivo en espera de evento.",
    "[watch] flujo macro: bonos, USD index y divisas emergentes sin señal crítica por ahora.",
    "[watch] innovación, chips, IA y energía: terminal en modo escucha estratégica.",
    "[watch] volatilidad latente. esperando siguiente impulso útil para trading o investigación.",
    "[watch] defensivos, commodities y guerra comercial: sin evento nuevo, vigilancia activa.",
]

FALLBACK_WATCH_MESSAGES = [
    "Market intelligence feed · gold, dollar, crude, FX and geopolitics",
    "Listening mode · commodities, yields, inflation, shipping and breakthroughs",
    "Standby signal · XAU, DXY, energy, majors FX and risk catalysts",
]

NEWS_QUERIES = [
    "gold OR dollar OR oil OR forex OR war OR inflation OR central bank",
    "commodities OR dollar index OR treasury yields OR opec OR energy",
    "artificial intelligence OR semiconductor OR defense OR shipping OR sanctions",
]

PARAM_SECTIONS = [
    {
        "title": "Mercado y ejecución",
        "icon": "tune",
        "items": [
            dict(path=("symbol",), label="Símbolo", typ="str", help="Par de trading en formato exchange."),
            dict(path=("market_symbol",), label="Market symbol", typ="str", help="Símbolo usado por el bot para órdenes."),
            dict(path=("mode",), label="Motor", typ="str", choices=["feature_prob", "legacy_classic"], help="Elige motor moderno probabilístico o compatibilidad clásica."),
            dict(path=("tf_signal",), label="TF señal", typ="str", choices=["5m", "15m", "30m", "1h"], help="Marco principal donde vive la estructura de señal."),
            dict(path=("tf_confirm",), label="TF confirmación", typ="str", choices=["1m", "3m", "5m", "15m"], help="Marco usado para confirmar entradas o timing fino."),
            dict(path=("tf_entry",), label="TF entrada", typ="str", choices=["1m", "3m", "5m", "15m"], help="Marco final de ejecución si el motor lo usa."),
            dict(path=("tf_trend",), label="TF tendencia", typ="str", choices=["15m", "1h", "4h"], help="Marco superior del filtro de tendencia."),
            dict(path=("live_chart_tf",), label="TF gráfico live", typ="str", choices=["1m", "5m", "15m", "1h"], help="Marco visual de la gráfica principal."),
            dict(path=("poll_seconds",), label="Poll", typ="int", min_val=2, max_val=60, step=1, unit="s", help="Intervalo de sondeo del bot."),
            dict(path=("status_heartbeat_seconds",), label="Heartbeat", typ="int", min_val=10, max_val=300, step=5, unit="s", help="Frecuencia de estado del bot."),
            dict(path=("use_1h_trend_filter",), label="Usar filtro de tendencia", typ="bool", help="Aplica filtro superior cuando el motor lo utilice."),
        ],
    },
    {
        "title": "Feature engine",
        "icon": "auto_graph",
        "items": [
            dict(path=("feature_prob", "ret_1_weight"), label="Peso ret 1", typ="float", min_val=0.0, max_val=1.0, step=0.01, help="Peso del retorno de 1 barra."),
            dict(path=("feature_prob", "ret_3_weight"), label="Peso ret 3", typ="float", min_val=0.0, max_val=1.0, step=0.01, help="Peso del retorno de 3 barras."),
            dict(path=("feature_prob", "ret_12_weight"), label="Peso ret 12", typ="float", min_val=0.0, max_val=1.0, step=0.01, help="Peso del retorno de 12 barras."),
            dict(path=("feature_prob", "range_pos_weight"), label="Peso rango", typ="float", min_val=0.0, max_val=1.0, step=0.01, help="Peso de la posición del close dentro del rango."),
            dict(path=("feature_prob", "vol_ratio_weight"), label="Peso vol ratio", typ="float", min_val=0.0, max_val=1.0, step=0.01, help="Peso del volumen relativo."),
            dict(path=("feature_prob", "realized_vol_weight"), label="Peso realized vol", typ="float", min_val=0.0, max_val=1.0, step=0.01, help="Peso de volatilidad realizada."),
            dict(path=("feature_prob", "slope_weight"), label="Peso slope", typ="float", min_val=0.0, max_val=1.0, step=0.01, help="Peso de la pendiente de la estructura."),
            dict(path=("feature_prob", "trend_weight"), label="Peso tendencia", typ="float", min_val=0.0, max_val=1.0, step=0.01, help="Peso del filtro de tendencia."),
            dict(path=("feature_prob", "breakout_weight"), label="Peso breakout", typ="float", min_val=0.0, max_val=1.0, step=0.01, help="Peso de breakout estructural."),
            dict(path=("feature_prob", "long_threshold"), label="Threshold long", typ="float", min_val=0.50, max_val=0.90, step=0.01, help="Umbral de probabilidad para largos."),
            dict(path=("feature_prob", "short_threshold"), label="Threshold short", typ="float", min_val=0.50, max_val=0.90, step=0.01, help="Umbral de probabilidad para cortos."),
            dict(path=("feature_prob", "persistence_bars"), label="Persistencia", typ="int", min_val=1, max_val=6, step=1, help="Barras consecutivas requeridas para confirmar señal."),
        ],
    },
    {
        "title": "Risk engine",
        "icon": "shield",
        "items": [
            dict(path=("risk_engine", "sl_vol_mult"), label="SL vol mult", typ="float", min_val=0.5, max_val=4.0, step=0.05, help="Stop loss en múltiplos de volatilidad."),
            dict(path=("risk_engine", "tp_vol_mult"), label="TP vol mult", typ="float", min_val=1.0, max_val=8.0, step=0.05, help="Take profit en múltiplos de volatilidad."),
            dict(path=("risk_engine", "breakeven_R"), label="Breakeven R", typ="float", min_val=0.5, max_val=4.0, step=0.1, help="Mover a breakeven a partir de este R."),
            dict(path=("risk_engine", "trail_start_R"), label="Trail start", typ="float", min_val=0.5, max_val=6.0, step=0.1, help="Activar trailing desde este R."),
            dict(path=("risk_engine", "trail_vol_mult"), label="Trail vol mult", typ="float", min_val=0.5, max_val=4.0, step=0.05, help="Distancia de trailing por volatilidad."),
            dict(path=("risk_engine", "risk_per_trade_pct"), label="Riesgo por trade", typ="float", min_val=0.1, max_val=2.0, step=0.05, unit="%", help="Capital arriesgado por operación."),
            dict(path=("risk_engine", "max_hold_minutes"), label="Max hold", typ="int", min_val=30, max_val=1440, step=30, unit="min", help="Tiempo máximo de permanencia."),
            dict(path=("risk_engine", "cooldown_minutes"), label="Cooldown", typ="int", min_val=0, max_val=720, step=15, unit="min", help="Espera entre operaciones."),
        ],
    },
    {
        "title": "Ejecución y costes",
        "icon": "bolt",
        "items": [
            dict(path=("execution", "fees_enabled"), label="Usar fees", typ="bool", help="Incluye comisión en resultados."),
            dict(path=("execution", "fee_rate_roundtrip"), label="Fee roundtrip", typ="float", min_val=0.0001, max_val=0.01, step=0.0001, help="Comisión ida y vuelta."),
            dict(path=("execution", "slippage_roundtrip_pct"), label="Slippage", typ="float", min_val=0.0, max_val=0.01, step=0.0001, help="Deslizamiento estimado."),
            dict(path=("execution", "label_exits"), label="Etiquetar salidas", typ="bool", help="Clasifica cierres como SL, BE, trail, etc."),
        ],
    },
    {
        "title": "Legacy mode / compatibilidad",
        "icon": "history",
        "items": [
            dict(path=("legacy", "ema_fast_trend"), label="EMA rápida tendencia", typ="int", min_val=5, max_val=200, step=1, help="Compatibilidad con filtro clásico."),
            dict(path=("legacy", "ema_slow_trend"), label="EMA lenta tendencia", typ="int", min_val=20, max_val=400, step=1, help="Compatibilidad con filtro clásico."),
            dict(path=("legacy", "ema_fast_signal"), label="EMA rápida señal", typ="int", min_val=5, max_val=200, step=1, help="Compatibilidad con señal clásica."),
            dict(path=("legacy", "ema_slow_signal"), label="EMA lenta señal", typ="int", min_val=20, max_val=400, step=1, help="Compatibilidad con señal clásica."),
            dict(path=("legacy", "donchian_window_signal"), label="Donchian señal", typ="int", min_val=10, max_val=240, step=1, help="Compatibilidad con breakout clásico."),
            dict(path=("legacy", "rsi_len_signal"), label="RSI length", typ="int", min_val=5, max_val=30, step=1, help="Compatibilidad con RSI clásico."),
            dict(path=("legacy", "rsi_long"), label="RSI long", typ="float", min_val=40, max_val=80, step=0.5, help="Umbral para largos."),
            dict(path=("legacy", "rsi_short"), label="RSI short", typ="float", min_val=20, max_val=60, step=0.5, help="Umbral para cortos."),
            dict(path=("legacy", "atr_len_signal"), label="ATR length", typ="int", min_val=5, max_val=50, step=1, help="Compatibilidad con ATR clásico."),
            dict(path=("legacy", "atr_ma_window_signal"), label="ATR MA", typ="int", min_val=10, max_val=240, step=1, help="Promedio ATR."),
            dict(path=("legacy", "atr_min_mult"), label="ATR mínimo", typ="float", min_val=0.5, max_val=2.5, step=0.05, help="Filtro mínimo ATR."),
            dict(path=("legacy", "breakout_buffer_atr"), label="Buffer breakout", typ="float", min_val=0.1, max_val=1.2, step=0.05, help="Confirmación de breakout."),
            dict(path=("legacy", "sl_atr_mult"), label="SL ATR", typ="float", min_val=0.4, max_val=3.0, step=0.05, help="Stop loss clásico."),
            dict(path=("legacy", "tp_atr_mult"), label="TP ATR", typ="float", min_val=1.0, max_val=6.0, step=0.05, help="Take profit clásico."),
            dict(path=("legacy", "breakeven_R"), label="Breakeven R", typ="float", min_val=0.5, max_val=3.0, step=0.1, help="Mover a break-even."),
            dict(path=("legacy", "trail_start_R"), label="Trail start", typ="float", min_val=0.5, max_val=5.0, step=0.1, help="Activa trailing clásico."),
            dict(path=("legacy", "trail_atr_mult"), label="Trail ATR", typ="float", min_val=0.5, max_val=2.5, step=0.05, help="Distancia de trailing clásico."),
        ],
    },
]


def deep_merge_dicts(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_path_value(path: tuple[str, ...], default: Any = None) -> Any:
    obj: Any = config
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def set_path_value(path: tuple[str, ...], value: Any) -> None:
    obj = config
    for key in path[:-1]:
        if key not in obj or not isinstance(obj[key], dict):
            obj[key] = {}
        obj = obj[key]
    obj[path[-1]] = value


def path_key(path: tuple[str, ...]) -> str:
    return ".".join(path)


def format_value(value: Any, typ: str, unit: str | None = None) -> str:
    if typ == "bool":
        return "ON" if bool(value) else "OFF"
    if typ == "int":
        try:
            base = f"{int(float(value))}"
        except Exception:
            base = str(value)
        return f"{base}{unit or ''}"
    if typ == "float":
        try:
            base = f"{float(value):.2f}"
        except Exception:
            base = str(value)
        return f"{base}{unit or ''}"
    return str(value)


def refresh_parameter_controls() -> None:
    for section in PARAM_SECTIONS:
        for item in section["items"]:
            p = tuple(item["path"])
            key = path_key(p)
            typ = item["typ"]
            value = get_path_value(p)
            unit = item.get("unit")
            ctrl = param_controls.get(key)
            chip = param_value_chips.get(key)

            if ctrl is not None:
                try:
                    if typ == "int":
                        ctrl.set_value(int(value))
                    elif typ == "float":
                        ctrl.set_value(float(value))
                    elif typ == "bool":
                        ctrl.set_value(bool(value))
                    else:
                        ctrl.set_value(value)
                except Exception:
                    pass

            if chip is not None:
                chip.set_text(format_value(value, typ, unit))


def normalize_stream_line(raw_line: str) -> list[str]:
    if not raw_line:
        return []
    text = raw_line.replace("\r", "\n")
    return [chunk.strip() for chunk in text.split("\n") if chunk.strip()]


def queue_log(msg: str) -> None:
    pending_logs.put(str(msg))


def build_runtime_config() -> dict[str, Any]:
    runtime = copy.deepcopy(config)

    runtime["signal_engine"] = runtime.get("mode", "feature_prob")

    legacy = runtime.get("legacy", {})
    risk = runtime.get("risk_engine", {})
    execution = runtime.get("execution", {})

    # aliases legacy / compatibilidad con backtesters anteriores
    runtime["ema_fast_trend"] = legacy.get("ema_fast_trend")
    runtime["ema_slow_trend"] = legacy.get("ema_slow_trend")
    runtime["ema_fast_signal"] = legacy.get("ema_fast_signal")
    runtime["ema_slow_signal"] = legacy.get("ema_slow_signal")
    runtime["donchian_window_signal"] = legacy.get("donchian_window_signal")
    runtime["rsi_len_signal"] = legacy.get("rsi_len_signal")
    runtime["rsi_long"] = legacy.get("rsi_long")
    runtime["rsi_short"] = legacy.get("rsi_short")
    runtime["atr_len_signal"] = legacy.get("atr_len_signal")
    runtime["atr_ma_window_signal"] = legacy.get("atr_ma_window_signal")
    runtime["atr_min_mult"] = legacy.get("atr_min_mult")
    runtime["breakout_buffer_atr"] = legacy.get("breakout_buffer_atr")
    runtime["sl_atr_mult"] = legacy.get("sl_atr_mult")
    runtime["tp_atr_mult"] = legacy.get("tp_atr_mult")
    runtime["trail_atr_mult"] = legacy.get("trail_atr_mult")

    runtime["breakeven_R"] = risk.get("breakeven_R")
    runtime["trail_start_R"] = risk.get("trail_start_R")
    runtime["max_hold_minutes"] = risk.get("max_hold_minutes")
    runtime["cooldown_minutes"] = risk.get("cooldown_minutes")
    runtime["risk_per_trade_pct"] = risk.get("risk_per_trade_pct")

    runtime["fees_enabled"] = execution.get("fees_enabled")
    runtime["fee_rate_roundtrip"] = execution.get("fee_rate_roundtrip")
    runtime["slippage_roundtrip_pct"] = execution.get("slippage_roundtrip_pct")
    runtime["label_exits"] = execution.get("label_exits")

    return runtime


def write_ambient_state() -> None:
    try:
        payload = {
            "last_terminal_line": output_log[-1] if output_log else "[idle] esperando eventos...",
            "last_terminal_lines": output_log[-40:],
            "updated_at": time.strftime("%H:%M:%S"),
            "strategy": current_strategy_name,
            "apolo_running": bool(apolo_process and apolo_process.poll() is None),
            "bot_running": bool(trading_process and trading_process.poll() is None),
            "best_params_exists": APOLO_BEST_FILE.exists(),
            "progress_exists": APOLO_PROGRESS_FILE.exists(),
        }
        with open(AMBIENT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(LIVE_TERMINAL_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(output_log[-200:]))
    except Exception:
        pass


def log_kind(line: str) -> str:
    s = str(line).lower()
    if s.startswith("[news]"):
        return "news"
    if s.startswith("[apolo]"):
        return "apolo"
    if s.startswith("[backtest]"):
        return "backtest"
    if s.startswith("[bot]"):
        return "bot"
    if s.startswith("[config]"):
        return "config"
    if s.startswith("[report]"):
        return "report"
    if s.startswith("[error]"):
        return "error"
    if s.startswith("[watch]"):
        return "watch"
    return "plain"


def render_log_line(line: str) -> str:
    return f'<div class="log-line log-{log_kind(line)}">{html.escape(str(line))}</div>'


def terminal_markup() -> str:
    lines = output_log[-500:] if output_log else ["[idle] esperando eventos..."]
    lines = list(reversed(lines))
    body = "".join(render_log_line(line) for line in lines)
    return (
        '<div class="retro-terminal-shell">'
        '  <div class="retro-terminal-toolbar">'
        '    <span class="dot dot-coral"></span>'
        '    <span class="dot dot-peach"></span>'
        '    <span class="dot dot-cyan"></span>'
        '    <span class="terminal-title">STRATUM · LIVE CONSOLE · BACKTEST / BOT / APOLO / AUDIT</span>'
        '    <span class="terminal-pulse"></span>'
        "  </div>"
        '  <div class="retro-terminal-body">'
        f'    <div class="terminal-pre">{body}</div>'
        '    <div class="terminal-status-row">'
        '      <div class="terminal-progress"><span></span></div>'
        '      <div class="terminal-cursor">_</div>'
        "    </div>"
        "  </div>"
        "</div>"
    )


def render_terminal() -> None:
    if terminal_html is not None:
        terminal_html.set_content(terminal_markup())


def append_log(msg: str) -> None:
    global last_log_time
    output_log.append(str(msg))
    last_log_time = time.time()
    render_terminal()
    write_ambient_state()


def export_terminal() -> None:
    out = BASE_DIR / "terminal_export.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(output_log))
    ui.notify("Terminal exportada", type="positive")
    append_log(f"[report] terminal exportada a {out.name}")


def build_raw_report() -> str:
    lines: list[str] = []
    lines.append("=== STRATUM RAW BACKTEST REPORT ===")
    lines.append(f"timestamp_local={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"current_strategy={current_strategy_name}")
    lines.append("")
    lines.append("--- CONFIG_UI ---")
    lines.append(json.dumps(config, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("--- CONFIG_RUNTIME ---")
    lines.append(json.dumps(build_runtime_config(), indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("--- STATS_DATA ---")
    if stats_data is not None:
        try:
            lines.append(json.dumps(stats_data, indent=2, ensure_ascii=False, default=str))
        except Exception:
            lines.append(str(stats_data))
    else:
        lines.append("null")
    lines.append("")
    lines.append("--- METRICS ---")
    lines.append(f"trades={safe_stat('trades', 0)}")
    lines.append(f"winrate_pct={float(safe_stat('winrate', 0)):.4f}")
    lines.append(f"total_pnl_pct={float(safe_stat('total_pnl', 0)):.4f}")
    lines.append(f"avg_pnl_pct={float(safe_stat('avg_pnl', 0)):.6f}")
    lines.append(f"max_dd_pct={float(safe_stat('max_dd', 0)):.4f}")
    lines.append(f"trades_per_day={float(safe_stat('tpd', 0)):.4f}")
    lines.append(f"projected_month_pct={float(safe_stat('proj_month', 0)):.4f}")
    lines.append("")
    lines.append("--- EXIT_COUNTS ---")
    try:
        lines.append(json.dumps(safe_stat("exit_counts", {}), indent=2, ensure_ascii=False, default=str))
    except Exception:
        lines.append(str(safe_stat("exit_counts", {})))
    lines.append("")

    if trades_data is not None and hasattr(trades_data, "empty") and not trades_data.empty:
        lines.append("--- TRADES_COLUMNS ---")
        try:
            lines.append(json.dumps({c: str(t) for c, t in trades_data.dtypes.items()}, indent=2, ensure_ascii=False))
        except Exception as e:
            lines.append(f"error_columns={e}")
        lines.append("")
        lines.append("--- TRADES_HEAD_20_CSV ---")
        try:
            lines.append(trades_data.head(20).to_csv(index=False))
        except Exception as e:
            lines.append(f"error_head={e}")
        lines.append("")
        lines.append("--- TRADES_TAIL_20_CSV ---")
        try:
            lines.append(trades_data.tail(20).to_csv(index=False))
        except Exception as e:
            lines.append(f"error_tail={e}")
        lines.append("")
    else:
        lines.append("--- TRADES ---")
        lines.append("No trades_data available")
        lines.append("")

    lines.append("--- ML_HINT ---")
    lines.append("Objetivo: usar este reporte para proponer mejoras al config JSON maximizando retorno, consistencia, estabilidad, frecuencia útil, distribución de exits y reducción de drawdown.")
    return "\n".join(lines)


def update_raw_report_box() -> None:
    if raw_report_box is not None:
        raw_report_box.set_value(raw_report_text)


def export_raw_report() -> None:
    if not raw_report_text.strip():
        ui.notify("No hay reporte bruto todavía", type="warning")
        return
    out = BASE_DIR / "raw_backtest_report.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(raw_report_text)
    ui.notify("Reporte bruto exportado", type="positive")
    append_log(f"[report] {out.name} exportado")


def available_strategy_names() -> list[str]:
    names = []
    for p in sorted(CONFIGS_DIR.glob("*.json")):
        if p.name in {"params_bt.json", "config.json"}:
            continue
        names.append(p.stem)
    return names[:50]


def update_strategy_label() -> None:
    if strategy_name_label is not None:
        strategy_name_label.set_text(f"Estrategia activa · {current_strategy_name}")


async def prompt_load_config() -> None:
    global current_strategy_name, config
    dialog = ui.dialog()
    suggestions = available_strategy_names()
    default_name = suggestions[0] if suggestions else "mi_estrategia"

    with dialog, ui.card().classes("panel-card").style("width: 440px"):
        ui.label("Cargar estrategia").classes("section-title")
        ui.label("Escribe el nombre del archivo sin .json").classes("section-help")
        name_input = ui.input(label="Nombre de estrategia", value=default_name, placeholder="mi_estrategia").props("outlined")
        if suggestions:
            with ui.row().classes("w-full gap-2 mt-2 flex-wrap"):
                for s in suggestions[:8]:
                    ui.button(s, on_click=lambda _, v=s: name_input.set_value(v)).classes("btn-soft")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancelar", on_click=lambda: dialog.submit(False)).classes("btn-soft")
            ui.button("Cargar", on_click=lambda: dialog.submit(True)).classes("btn-primary")

    ok = await dialog
    if not ok:
        return

    name = str(name_input.value or "").strip().replace(".json", "")
    if not name:
        ui.notify("Escribe un nombre de estrategia", type="warning")
        return

    path = CONFIGS_DIR / f"{name}.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        config = deep_merge_dicts(DEFAULT_CONFIG, loaded)

        current_strategy_name = path.stem
        update_strategy_label()
        refresh_parameter_controls()
        refresh_live_chart(force=True)

        ui.notify(f"Estrategia cargada: {path.name}", type="positive")
        append_log(f"[config] estrategia cargada desde {path.name}")
    except Exception as e:
        ui.notify(f"Error cargando estrategia: {e}", type="negative")
        append_log(f"[error] no se pudo cargar {path.name}: {e}")


async def prompt_save_strategy() -> None:
    global current_strategy_name
    dialog = ui.dialog()
    default_name = current_strategy_name if current_strategy_name != "config actual" else "mi_estrategia"

    with dialog, ui.card().classes("panel-card").style("width: 440px"):
        ui.label("Guardar estrategia").classes("section-title")
        ui.label("Se guardará la configuración actual como archivo JSON").classes("section-help")
        name_input = ui.input(label="Nombre de estrategia", value=default_name, placeholder="mi_estrategia").props("outlined")
        with ui.row().classes("w-full justify-end gap-2 mt-4"):
            ui.button("Cancelar", on_click=lambda: dialog.submit(False)).classes("btn-soft")
            ui.button("Guardar", on_click=lambda: dialog.submit(True)).classes("btn-primary")

    ok = await dialog
    if not ok:
        return

    name = str(name_input.value or "").strip().replace(".json", "")
    if not name:
        ui.notify("Escribe un nombre de estrategia", type="warning")
        return

    path = CONFIGS_DIR / f"{name}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        with open(PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(build_runtime_config(), f, indent=2, ensure_ascii=False)
        current_strategy_name = path.stem
        update_strategy_label()
        ui.notify(f"Estrategia guardada: {path.name}", type="positive")
        append_log(f"[config] estrategia guardada como {path.name}")
    except Exception as e:
        ui.notify(f"Error guardando estrategia: {e}", type="negative")
        append_log(f"[error] no se pudo guardar {path.name}: {e}")


# ======================== Kucoin API (reemplaza a Binance y Bybit) ========================
import os
import requests
from requests.auth import HTTPProxyAuth

def fetch_kucoin_klines(symbol: str = "BTC-USDT", interval: str = "15m", limit: int = 180) -> pd.DataFrame:
    """
    Obtiene velas de KuCoin usando proxy si está configurado en variables de entorno.
    """
    # Mapeo de intervalos
    interval_map = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "2h": "2hour",
        "4h": "4hour",
        "6h": "6hour",
        "12h": "12hour",
        "1d": "1day",
        "1w": "1week",
    }
    kucoin_interval = interval_map.get(interval, "15min")
    if symbol == "BTCUSDT":
        symbol = "BTC-USDT"
    url = f"https://api.kucoin.com/api/v1/market/candles?type={kucoin_interval}&symbol={symbol}&limit={limit}"

    # Leer configuración del proxy desde variables de entorno
    proxy_url = os.environ.get("PROXY_URL")
    proxy_user = os.environ.get("PROXY_USER")
    proxy_pass = os.environ.get("PROXY_PASS")

    proxies = None
    auth = None
    if proxy_url and proxy_user and proxy_pass:
        proxies = {"http": proxy_url, "https": proxy_url}
        auth = HTTPProxyAuth(proxy_user, proxy_pass)
        print(f"[proxy] Usando proxy: {proxy_url}")  # Opcional, para logs
    else:
        print("[proxy] No se encontraron variables de proxy, conectando directamente")

    if proxies:
        response = requests.get(url, proxies=proxies, auth=auth, timeout=12)
    else:
        response = requests.get(url, timeout=12)

    response.raise_for_status()
    data = response.json()

    if data.get("code") != "200000":
        raise Exception(f"KuCoin API error: {data.get('msg')}")

    klines = data["data"]
    df = pd.DataFrame(klines, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["time"].astype(int), unit="ms")
    df = df.sort_values("open_time")
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def fetch_kucoin_klines_idle(symbol: str = "BTC-USDT", interval: str = "15m", limit: int = 140) -> pd.DataFrame:
    """Versión idle, igual pero con proxy."""
    return fetch_kucoin_klines(symbol, interval, limit)

def infer_trade_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    time_candidates = ["entry_time", "time", "timestamp", "ts", "datetime"]
    price_candidates = ["entry_price", "entry_px", "price", "px", "close"]
    side_candidates = ["side", "direction", "signal", "trade_side"]
    time_col = next((c for c in time_candidates if c in df.columns), None)
    price_col = next((c for c in price_candidates if c in df.columns), None)
    side_col = next((c for c in side_candidates if c in df.columns), None)
    return time_col, price_col, side_col


def build_live_candles_figure() -> go.Figure:
    live_tf = config.get("live_chart_tf", config.get("tf_signal", "15m"))
    df = fetch_kucoin_klines(
        symbol=config.get("market_symbol", "BTC-USDT"),
        interval=live_tf,
        limit=180,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=f"BTCUSDT {live_tf}",
            increasing_line_color=PALETTE["primary_blue"],
            decreasing_line_color=PALETTE["accent_coral"],
            increasing_fillcolor="rgba(35,115,184,.65)",
            decreasing_fillcolor="rgba(233,138,107,.65)",
        )
    )

    if show_backtest_overlay and trades_data is not None and hasattr(trades_data, "empty") and not trades_data.empty:
        time_col, price_col, side_col = infer_trade_columns(trades_data)
        if time_col and price_col:
            overlay = trades_data.copy()
            overlay[time_col] = pd.to_datetime(overlay[time_col], errors="coerce")
            overlay[price_col] = pd.to_numeric(overlay[price_col], errors="coerce")
            overlay = overlay.dropna(subset=[time_col, price_col]).tail(200)

            if not overlay.empty:
                marker_colors = []
                marker_symbols = []
                for _, row in overlay.iterrows():
                    side_val = str(row.get(side_col, "")).lower() if side_col else ""
                    if "short" in side_val or "sell" in side_val:
                        marker_colors.append(PALETTE["accent_coral"])
                        marker_symbols.append("triangle-down")
                    else:
                        marker_colors.append(PALETTE["accent_cyan"])
                        marker_symbols.append("triangle-up")

                fig.add_trace(
                    go.Scatter(
                        x=overlay[time_col],
                        y=overlay[price_col],
                        mode="markers",
                        name="Trades backtest",
                        marker=dict(size=10, color=marker_colors, symbol=marker_symbols, line=dict(width=1, color="rgba(17,17,17,.45)")),
                        hovertemplate="Trade %{x}<br>Px %{y}<extra></extra>",
                    )
                )

    fig.update_layout(
        title=f"BTCUSDT · live view {live_tf}",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=14, r=14, t=42, b=14),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def refresh_live_chart(force: bool = False) -> None:
    global last_chart_refresh_at, chart_refresh_in_progress
    if live_chart_box is None:
        return
    now = time.time()
    if chart_refresh_in_progress:
        return
    if not force and now - last_chart_refresh_at < 45:
        return

    chart_refresh_in_progress = True
    try:
        fig = build_live_candles_figure()
        live_chart_box.clear()
        with live_chart_box:
            ui.plotly(fig).classes("w-full h-80 chart-shell")
        last_chart_refresh_at = now
    except Exception as e:
        append_log(f"[error] live chart: {e}")
    finally:
        chart_refresh_in_progress = False


def reset_live_chart() -> None:
    global show_backtest_overlay
    show_backtest_overlay = False
    refresh_live_chart(force=True)
    append_log("[config] gráfico live reseteado al mercado actual")


def copy_terminal_to_clipboard() -> None:
    ui.run_javascript(
        'navigator.clipboard.writeText(Array.from(document.querySelectorAll(".terminal-pre .log-line")).map(x => x.innerText).join("\\n") || "")'
    )


def copy_raw_report_to_clipboard() -> None:
    ui.run_javascript('navigator.clipboard.writeText(document.querySelector(".raw-report-box textarea")?.value || "")')


def read_trading_pipe(pipe) -> None:
    global trading_last_output_at, last_trading_line
    for raw_line in iter(pipe.readline, ""):
        for line in normalize_stream_line(raw_line):
            if line == last_trading_line:
                continue
            last_trading_line = line
            trading_last_output_at = time.time()
            queue_log(f"[bot] {line}")
    pipe.close()


def launch_bot_embedded(testnet: bool) -> None:
    global trading_process, trading_started_at, trading_last_output_at, last_trading_line

    if trading_process and trading_process.poll() is None:
        ui.notify("El bot de trading ya está corriendo", type="warning")
        append_log("[bot] intento de inicio ignorado: ya estaba en ejecución")
        return

    if not BOT_SCRIPT.exists():
        ui.notify(f"No se encontró {BOT_SCRIPT.name}", type="negative")
        append_log(f"[error] no se encontró {BOT_SCRIPT.name}")
        return

    runtime_cfg = build_runtime_config()

    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(runtime_cfg, f, indent=2, ensure_ascii=False)

    env = os.environ.copy()
    env["TESTNET"] = str(testnet).lower()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [sys.executable, BOT_SCRIPT.name]
    try:
        trading_process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        trading_started_at = time.time()
        trading_last_output_at = trading_started_at
        last_trading_line = None
        threading.Thread(target=read_trading_pipe, args=(trading_process.stdout,), daemon=True).start()

        target = "TESTNET" if testnet else "LIVE"
        ui.notify(f"Bot lanzado en {target}", type="positive")
        append_log(f"[bot] usando python: {sys.executable}")
        append_log(f"[bot] iniciado en {target} con configuración actual de la interfaz")
    except Exception as e:
        ui.notify(f"Error iniciando bot: {e}", type="negative")
        append_log(f"[error] launch_bot: {e}")


async def prompt_launch_bot() -> None:
    dialog = ui.dialog()
    mode = {"value": "testnet"}

    with dialog, ui.card().classes("panel-card").style("width: 360px"):
        ui.label("Iniciar bot de trading").classes("section-title")
        ui.label("Elige el modo de ejecución").classes("section-help")
        with ui.row().classes("w-full gap-2 mt-4"):
            ui.button("Testnet", on_click=lambda: (mode.__setitem__("value", "testnet"), dialog.submit(True))).classes("btn-primary")
            ui.button("Live", on_click=lambda: (mode.__setitem__("value", "live"), dialog.submit(True))).classes("btn-warm")
        ui.button("Cancelar", on_click=lambda: dialog.submit(False)).classes("btn-soft mt-4")

    ok = await dialog
    if ok:
        launch_bot_embedded(testnet=(mode["value"] == "testnet"))


def run_backtest_with_logs(cfg: dict[str, Any], log_callback):
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            try:
                result = run_backtest(cfg, None, log_callback)
            except TypeError:
                result = run_backtest(cfg, None)

        text = buffer.getvalue().strip()
        if text:
            for line in text.splitlines():
                log_callback(line)

        return result
    except Exception as e:
        text = buffer.getvalue().strip()
        if text:
            for line in text.splitlines():
                log_callback(line)
        if "418" in str(e) or "I'm a teapot" in str(e):
            log_callback("[error] Binance rechazó la descarga histórica (418). Revisa rango de fechas, endpoint o rate limit en el backtester.")
        log_callback(f"[error] backtest: {e}")
        return None, None


def start_backtest() -> None:
    if header_status is not None:
        header_status.set_text("BACKTEST RUNNING")
    queue_log("[backtest] iniciando ejecución...")

    def run() -> None:
        runtime_cfg = build_runtime_config()
        result = run_backtest_with_logs(runtime_cfg, queue_log)
        if isinstance(result, tuple) and len(result) == 2:
            trades, stats = result
        else:
            trades, stats = None, None
        pending_results.put(("backtest", trades, stats))

    threading.Thread(target=run, daemon=True).start()


def safe_stat(key: str, default: Any = 0) -> Any:
    if not stats_data:
        return default
    return stats_data.get(key, default)


def stat_chip(title: str, value: str, tone: str = "neutral") -> None:
    tone_map = {
        "neutral": "background: rgba(255,255,255,.72); border: 1px solid var(--border);",
        "primary": "background: linear-gradient(135deg, rgba(35,115,184,.12), rgba(168,221,229,.32)); border: 1px solid rgba(35,115,184,.25);",
        "warm": "background: linear-gradient(135deg, rgba(233,138,107,.15), rgba(237,176,138,.22)); border: 1px solid rgba(233,138,107,.20);",
        "soft": "background: linear-gradient(135deg, rgba(217,221,242,.30), rgba(231,199,214,.24)); border: 1px solid rgba(48,78,115,.16);",
    }
    with ui.card().classes("stat-chip").style(tone_map[tone]):
        ui.label(title).classes("stat-chip-title")
        ui.label(value).classes("stat-chip-value")


def update_results() -> None:
    if not stats_data or stats_card is None:
        return

    stats_card.clear()
    with stats_card:
        with ui.grid(columns=4).classes("w-full gap-4 xl:grid-cols-4 lg:grid-cols-2 md:grid-cols-2 sm:grid-cols-1"):
            stat_chip("Trades", str(safe_stat("trades")), "primary")
            stat_chip("Winrate", f"{float(safe_stat('winrate', 0)):.2f}%", "soft")
            stat_chip("PnL total", f"{float(safe_stat('total_pnl', 0)):.2f}%", "warm")
            stat_chip("PnL promedio", f"{float(safe_stat('avg_pnl', 0)):.3f}%", "neutral")
            stat_chip("Max DD", f"{float(safe_stat('max_dd', 0)):.2f}%", "warm")
            stat_chip("Trades/día", f"{float(safe_stat('tpd', 0)):.2f}", "neutral")
            stat_chip("Proyección mensual", f"{float(safe_stat('proj_month', 0)):.2f}%", "primary")
            stat_chip("Fees", "ON" if config.get("execution", {}).get("fees_enabled") else "OFF", "soft")

    if trades_data is None or not hasattr(trades_data, "empty") or trades_data.empty:
        return

    data = trades_data.copy()
    equity = data["pnl_equity_pct"].cumsum() if "pnl_equity_pct" in data.columns else pd.Series(dtype=float)

    plot_eq.clear()
    with plot_eq:
        if not equity.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=list(range(len(equity))),
                    y=equity,
                    mode="lines",
                    line=dict(color=PALETTE["primary_blue"], width=3),
                    fill="tozeroy",
                    fillcolor="rgba(35,115,184,0.10)",
                )
            )
            fig_eq.update_layout(
                title="Curva de equity",
                xaxis_title="Trade #",
                yaxis_title="Ganancia acumulada %",
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            ui.plotly(fig_eq).classes("w-full h-96 chart-shell")
        else:
            ui.label("No hay columna pnl_equity_pct para curva de equity.").classes("section-help")

    exit_counts = safe_stat("exit_counts", {})
    plot_pie.clear()
    with plot_pie:
        if exit_counts:
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=list(exit_counts.keys()),
                        values=list(exit_counts.values()),
                        hole=0.45,
                        marker=dict(
                            colors=[
                                PALETTE["accent_cyan"],
                                PALETTE["accent_coral"],
                                PALETTE["accent_lavender"],
                                PALETTE["accent_peach"],
                                PALETTE["primary_blue"],
                            ]
                        ),
                    )
                ]
            )
            fig_pie.update_layout(title="Motivos de salida", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            ui.plotly(fig_pie).classes("w-full h-96 chart-shell")
        else:
            ui.label("Sin motivos de salida disponibles aún.").classes("section-help")

    plot_hist.clear()
    with plot_hist:
        if "pnl_equity_pct" in data.columns:
            fig_hist = px.histogram(data, x="pnl_equity_pct", nbins=30, title="Distribución de PnL por trade", color_discrete_sequence=[PALETTE["primary_blue"]])
            fig_hist.update_layout(bargap=0.10, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            ui.plotly(fig_hist).classes("w-full h-96 chart-shell")
        else:
            ui.label("No hay columna pnl_equity_pct para histograma.").classes("section-help")

    plot_scatter.clear()
    with plot_scatter:
        if "R" in data.columns and "max_R" in data.columns:
            fig_scatter = px.scatter(
                data,
                x="R",
                y="max_R",
                title="R vs Max_R alcanzado",
                labels={"R": "R", "max_R": "Max R"},
                color="exit_reason" if "exit_reason" in data.columns else None,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            ui.plotly(fig_scatter).classes("w-full h-96 chart-shell")
        else:
            ui.label("No hay columnas R / max_R en este resultado.").classes("section-help")

    plot_bar.clear()
    with plot_bar:
        if "entry_time" in data.columns and "pnl_equity_pct" in data.columns:
            data["date"] = pd.to_datetime(data["entry_time"], errors="coerce").dt.date
            daily_pnl = data.groupby("date")["pnl_equity_pct"].sum().reset_index()
            fig_bar = px.bar(daily_pnl, x="date", y="pnl_equity_pct", title="PnL diario", color_discrete_sequence=[PALETTE["accent_coral"]])
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            ui.plotly(fig_bar).classes("w-full h-96 chart-shell")
        else:
            ui.label("No hay columnas suficientes para gráfico diario.").classes("section-help")

    refresh_live_chart(force=True)


def fetch_live_macro_news() -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    seen: set[str] = set()
    headers = {"User-Agent": "Mozilla/5.0 STRATUM/1.0"}

    for query in NEWS_QUERIES:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=12) as response:
            raw = response.read()
        root = ET.fromstring(raw)

        for item in root.findall("./channel/item")[:5]:
            title = (item.findtext("title") or "").strip()
            if not title or title in seen:
                continue
            seen.add(title)
            pub_date = (item.findtext("pubDate") or "").strip()
            timestamp = ""
            if pub_date:
                try:
                    timestamp = parsedate_to_datetime(pub_date).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    timestamp = pub_date

            collected.append({"title": title, "timestamp": timestamp, "source": "Google News RSS"})
            if len(collected) >= 12:
                return collected

    return collected


def refresh_live_macro_news(background: bool = True) -> None:
    global news_refresh_in_progress
    if news_refresh_in_progress:
        return
    news_refresh_in_progress = True

    def worker() -> None:
        global news_items, news_last_fetch_at, news_refresh_in_progress
        try:
            items = fetch_live_macro_news()
            if items:
                news_items = items
                news_last_fetch_at = time.time()
                pending_logs.put(f"[news] feed actualizado con {len(items)} titulares macro / energía / geopolítica")
            else:
                pending_logs.put("[news] no llegaron titulares nuevos; manteniendo fallback visual")
        except Exception as e:
            pending_logs.put(f"[news] feed no disponible: {e}")
        finally:
            news_refresh_in_progress = False

    if background:
        threading.Thread(target=worker, daemon=True).start()
    else:
        worker()


def emit_live_news_to_terminal(now: float) -> None:
    global news_cursor, news_last_emit_at
    if not news_items or now - news_last_emit_at < 25:
        return
    item = news_items[news_cursor % len(news_items)]
    news_cursor += 1
    news_last_emit_at = now
    stamp = f" · {item['timestamp']}" if item.get("timestamp") else ""
    pending_logs.put(f"[news] {item['title']}{stamp}")


def read_apolo_pipe(pipe) -> None:
    global apolo_last_output_at, last_apolo_line
    for raw_line in iter(pipe.readline, ""):
        for line in normalize_stream_line(raw_line):
            if line == last_apolo_line:
                continue
            last_apolo_line = line
            apolo_last_output_at = time.time()
            queue_log(f"[apolo] {line}")
    pipe.close()


def start_apolo(resume: bool = True) -> None:
    global apolo_process, apolo_started_at, apolo_last_output_at, last_apolo_line

    if apolo_process and apolo_process.poll() is None:
        ui.notify("Apolo ya está corriendo", type="warning")
        append_log("[apolo] intento de inicio ignorado: ya estaba en ejecución")
        return

    apolo_path = BASE_DIR / "apolo.py"
    if not apolo_path.exists():
        ui.notify("No se encontró apolo.py", type="negative")
        append_log("[error] apolo.py no encontrado")
        return

    cmd = [sys.executable, "apolo.py", "--trials", "500", "--jobs", "4"]
    if resume:
        cmd.append("--resume")

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        apolo_process = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        apolo_started_at = time.time()
        apolo_last_output_at = apolo_started_at
        last_apolo_line = None
        threading.Thread(target=read_apolo_pipe, args=(apolo_process.stdout,), daemon=True).start()

        ui.notify("Apolo iniciado", type="positive")
        append_log(f"[apolo] usando python: {sys.executable}")
        append_log(f"[apolo] cwd: {BASE_DIR}")
        append_log(f"[apolo] iniciado con comando: {' '.join(cmd)}")
        refresh_apolo_status(force=True)
    except Exception as e:
        ui.notify(f"No se pudo iniciar Apolo: {e}", type="negative")
        append_log(f"[error] start_apolo: {e}")


def stop_apolo() -> None:
    global apolo_process
    if not apolo_process or apolo_process.poll() is not None:
        ui.notify("Apolo no está activo", type="warning")
        append_log("[apolo] stop ignorado: no había proceso activo")
        return

    try:
        apolo_process.terminate()
        append_log("[apolo] terminate enviado")
        ui.notify("Señal de parada enviada a Apolo", type="warning")
    except Exception as e:
        ui.notify(f"Error deteniendo Apolo: {e}", type="negative")
        append_log(f"[error] stop_apolo: {e}")


def connect_apolo_monitor() -> None:
    refresh_apolo_status(force=True)
    append_log("[apolo] monitor de progreso sincronizado con archivos locales")
    ui.notify("Monitor de Apolo actualizado", type="info")


def render_best_params(data: dict[str, Any] | None) -> str:
    if not data:
        return f'<pre class="mini-pre">Sin {APOLO_BEST_FILE.name} todavía.</pre>'
    return f'<pre class="mini-pre">{html.escape(json.dumps(data, indent=2, ensure_ascii=False))}</pre>'


def refresh_apolo_status(force: bool = False) -> None:
    proc_running = apolo_process is not None and apolo_process.poll() is None

    if apolo_status_badge is not None:
        apolo_status_badge.set_text(f"Apolo {'RUNNING' if proc_running else 'IDLE'}")

    progress_text = f"Sin {APOLO_PROGRESS_FILE.name} todavía."
    if APOLO_PROGRESS_FILE.exists():
        try:
            progress_text = APOLO_PROGRESS_FILE.read_text(encoding="utf-8").strip() or progress_text
        except Exception as e:
            progress_text = f"No se pudo leer {APOLO_PROGRESS_FILE.name}: {e}"
    if apolo_progress_label is not None:
        apolo_progress_label.set_text(progress_text)

    best_data = None
    if APOLO_BEST_FILE.exists():
        try:
            best_data = json.loads(APOLO_BEST_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            best_data = {"error": f"No se pudo leer {APOLO_BEST_FILE.name}: {e}"}
    if apolo_best_box is not None:
        apolo_best_box.set_content(render_best_params(best_data))

    meta_parts = [f"Base dir: {BASE_DIR.name}"]
    if apolo_started_at:
        meta_parts.append(f"Inicio gui: {time.strftime('%H:%M:%S', time.localtime(apolo_started_at))}")
    if apolo_last_output_at:
        meta_parts.append(f"Última salida: {time.strftime('%H:%M:%S', time.localtime(apolo_last_output_at))}")
    if APOLO_RESULTS_FILE.exists():
        try:
            rows = len(pd.read_csv(APOLO_RESULTS_FILE))
            meta_parts.append(f"Intentos guardados: {rows}")
        except Exception:
            meta_parts.append("Intentos guardados: archivo no legible")
    else:
        meta_parts.append(f"Sin {APOLO_RESULTS_FILE.name}")

    if apolo_meta_label is not None:
        apolo_meta_label.set_text(" · ".join(meta_parts))


def process_ui_events() -> None:
    global stats_data, trades_data, raw_report_text, idle_message_index, last_ui_idle_emit, show_backtest_overlay

    while not pending_logs.empty():
        append_log(pending_logs.get())

    while not pending_results.empty():
        kind, trades, stats = pending_results.get()
        if kind == "backtest":
            if stats is not None:
                stats_data = stats
                trades_data = trades
                show_backtest_overlay = True
                append_log("[backtest] ejecución finalizada")
                if header_status is not None:
                    header_status.set_text("BACKTEST READY")
                update_results()
                raw_report_text = build_raw_report()
                update_raw_report_box()
                append_log("[report] reporte bruto actualizado")
            else:
                append_log("[backtest] sin resultados")
                if header_status is not None:
                    header_status.set_text("BACKTEST ERROR")

    now = time.time()

    if market_watch_label is not None:
        if news_items:
            item = news_items[news_cursor % len(news_items)]
            market_watch_label.set_text(item["title"])
            if news_source_label is not None:
                suffix = f" · {item['timestamp']}" if item.get("timestamp") else ""
                news_source_label.set_text(f"{item.get('source', 'Live RSS')}{suffix}")
        else:
            market_watch_label.set_text(FALLBACK_WATCH_MESSAGES[int(now // 5) % len(FALLBACK_WATCH_MESSAGES)])
            if news_source_label is not None:
                news_source_label.set_text("Fallback visual · esperando feed real")

    if news_last_fetch_at is None or now - news_last_fetch_at > 600:
        refresh_live_macro_news(background=True)

    emit_live_news_to_terminal(now)

    if not news_items and now - last_log_time > 8 and now - last_ui_idle_emit > 8:
        queue_log(IDLE_MARKET_LINES[idle_message_index % len(IDLE_MARKET_LINES)])
        idle_message_index += 1
        last_ui_idle_emit = now

    if live_chart_box is not None and now - last_chart_refresh_at > 60:
        refresh_live_chart(force=False)

    refresh_apolo_status()
    write_ambient_state()


def add_styles() -> None:
    ui.add_head_html(
        f"""
        <style>
            :root {{
                --bg-main: {PALETTE["bg_main"]};
                --bg-soft: {PALETTE["bg_soft"]};
                --panel: {PALETTE["panel"]};
                --panel-alt: {PALETTE["panel_alt"]};
                --text-main: {PALETTE["text_main"]};
                --text-soft: {PALETTE["text_soft"]};
                --text-muted: {PALETTE["text_muted"]};
                --border: {PALETTE["border"]};
                --border-strong: {PALETTE["border_strong"]};
                --primary: {PALETTE["primary_blue"]};
                --primary-dark: {PALETTE["primary_blue_dark"]};
                --accent-cyan: {PALETTE["accent_cyan"]};
                --accent-coral: {PALETTE["accent_coral"]};
                --accent-pink: {PALETTE["accent_pink"]};
                --accent-peach: {PALETTE["accent_peach"]};
                --accent-lavender: {PALETTE["accent_lavender"]};
                --terminal-bg: {PALETTE["terminal_bg"]};
                --terminal-panel: {PALETTE["terminal_panel"]};
                --terminal-text: {PALETTE["terminal_text"]};
                --terminal-muted: {PALETTE["terminal_muted"]};
                --terminal-accent: {PALETTE["terminal_accent"]};
            }}

            body, .nicegui-content {{
                background:
                    radial-gradient(circle at 10% 10%, rgba(168,221,229,.42), transparent 22%),
                    radial-gradient(circle at 92% 18%, rgba(233,138,107,.18), transparent 22%),
                    radial-gradient(circle at 80% 82%, rgba(217,221,242,.35), transparent 20%),
                    linear-gradient(180deg, #f8f5ee 0%, var(--bg-main) 100%);
                color: var(--text-main);
                font-family: Inter, Segoe UI, Arial, sans-serif;
            }}

            .page-shell {{
                max-width: 1780px;
                margin: 0 auto;
                padding: 10px 14px 18px;
            }}

            .main-grid {{
                display: grid;
                grid-template-columns: minmax(520px, 1fr) minmax(560px, 1.2fr);
                gap: 16px;
                align-items: start;
            }}

            .left-sticky {{
                position: sticky;
                top: 8px;
            }}

            .hero-wrap {{
                background: linear-gradient(135deg, rgba(255,255,255,.72), rgba(255,253,249,.90));
                border: 1px solid rgba(200,191,175,.52);
                box-shadow: 0 12px 30px rgba(35,68,107,.06);
                border-radius: 22px;
                padding: 12px 16px;
                backdrop-filter: blur(10px);
            }}

            .hero-title {{
                font-size: 22px;
                font-weight: 800;
                letter-spacing: .14em;
                color: #143148;
            }}

            .hero-subtitle {{
                color: var(--text-soft);
                font-size: 12px;
                line-height: 1.35;
                max-width: 760px;
            }}

            .hero-badge {{
                padding: 7px 12px;
                border-radius: 999px;
                font-size: 11px;
                font-weight: 700;
                border: 1px solid rgba(35,115,184,.18);
                color: var(--primary-dark);
                background: linear-gradient(135deg, rgba(168,221,229,.30), rgba(255,255,255,.78));
            }}

            .strategy-chip {{
                margin-top: 8px;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 9px 14px;
                border-radius: 999px;
                background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(248,246,241,.88));
                border: 1px solid rgba(214,208,198,.86);
                color: #23446b;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: .02em;
                box-shadow: 0 8px 20px rgba(35,68,107,.05);
            }}

            .chart-header-row {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 8px;
                flex-wrap: wrap;
            }}

            .ticker-band {{
                margin-top: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
                flex-wrap: wrap;
                background: rgba(255,255,255,.58);
                border: 1px solid rgba(216,209,197,.72);
                border-radius: 999px;
                padding: 6px 12px;
            }}

            .ticker-kicker {{
                font-size: 11px;
                font-weight: 700;
                letter-spacing: .12em;
                text-transform: uppercase;
                color: var(--primary-dark);
                padding-right: 12px;
                border-right: 1px solid rgba(216,209,197,.88);
            }}

            .ticker-text {{
                color: var(--text-soft);
                font-size: 12px;
                flex: 1;
            }}

            .ticker-meta {{
                color: var(--text-muted);
                font-size: 11px;
                white-space: nowrap;
            }}

            .header-actions {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 10px;
            }}

            .panel-card, .q-card {{
                background: linear-gradient(180deg, rgba(255,255,255,.86), rgba(251,248,242,.94));
                border-radius: 22px;
                border: 1px solid rgba(200,191,175,.52);
                box-shadow: 0 14px 36px rgba(35,68,107,.07);
                padding: 16px;
            }}

            .section-title {{
                font-size: 16px;
                font-weight: 700;
                letter-spacing: -.01em;
            }}

            .section-help {{
                color: var(--text-soft);
                font-size: 12px;
                line-height: 1.45;
            }}

            .param-stack {{
                display: flex;
                flex-direction: column;
                gap: 12px;
            }}

            .param-block {{
                background: rgba(255,255,255,.52);
                border: 1px solid rgba(216,209,197,.72);
                border-radius: 18px;
                padding: 12px 12px 10px;
            }}

            .param-head {{
                display: flex;
                justify-content: space-between;
                align-items: start;
                gap: 12px;
                margin-bottom: 8px;
            }}

            .param-label {{
                font-size: 13px;
                font-weight: 700;
                color: var(--text-main);
            }}

            .param-help {{
                color: var(--text-muted);
                font-size: 11px;
                line-height: 1.35;
            }}

            .param-value-chip, .status-chip {{
                padding: 6px 10px;
                border-radius: 999px;
                font-size: 11px;
                font-weight: 700;
                background: linear-gradient(135deg, rgba(35,115,184,.10), rgba(168,221,229,.24));
                color: var(--primary-dark);
                border: 1px solid rgba(35,115,184,.16);
                min-width: 84px;
                text-align: center;
            }}

            .param-input, .q-field__control {{
                border-radius: 16px !important;
                background: rgba(255,255,255,.90) !important;
            }}

            .q-field--outlined .q-field__control:before, .q-field--outlined .q-field__control:after {{
                border-color: rgba(200,191,175,.90) !important;
            }}

            .q-slider {{ color: var(--primary); }}
            .q-slider__track {{ color: var(--primary); }}
            .q-slider__track-container {{ opacity: 1; }}
            .q-slider__track-markers {{ color: rgba(200,191,175,.75) !important; }}
            .q-slider__pin-text {{ font-weight: 700; }}
            .q-toggle__thumb:after {{ background: var(--primary) !important; }}
            .q-toggle__track {{ background: rgba(200,191,175,.92) !important; }}

            .q-btn {{
                border-radius: 12px;
                text-transform: none;
                font-weight: 700;
                letter-spacing: .02em;
                padding: 9px 14px;
                font-size: 12px;
                font-family: 'Fira Code', Consolas, monospace;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.08), 0 8px 20px rgba(11,25,38,.10);
            }}

            .btn-primary {{
                background: linear-gradient(180deg, rgba(30,47,66,.92), rgba(18,30,42,.98));
                color: #e9f8ff;
                border: 1px solid rgba(110,203,232,.22);
            }}

            .btn-soft {{
                background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(250,248,244,.96));
                color: #183247;
                border: 1px solid rgba(210,205,195,.95);
                box-shadow: 0 8px 18px rgba(35,68,107,.05);
            }}

            .btn-warm {{
                background: linear-gradient(180deg, rgba(230,128,96,.98), rgba(184,74,49,.98));
                color: #fff8f4;
                border: 1px solid rgba(163,71,49,.34);
                box-shadow: 0 14px 28px rgba(184,74,49,.22);
            }}

            .chart-shell {{
                background: rgba(255,255,255,.72);
                border-radius: 18px;
                padding: 8px;
            }}

            .stat-chip {{
                border-radius: 20px;
                padding: 16px;
                min-height: 100px;
                box-shadow: 0 8px 20px rgba(35,68,107,.04);
            }}

            .stat-chip-title {{
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: .08em;
                color: var(--text-muted);
                font-weight: 700;
            }}

            .stat-chip-value {{
                margin-top: 8px;
                font-size: 24px;
                font-weight: 700;
                color: var(--text-main);
                letter-spacing: -.03em;
            }}

            .retro-terminal-shell {{
                background: linear-gradient(180deg, rgba(15,23,32,.98), rgba(22,35,49,.98));
                border-radius: 24px;
                overflow: hidden;
                border: 1px solid rgba(110,203,232,.18);
                box-shadow: 0 18px 50px rgba(11,25,38,.28);
            }}

            .retro-terminal-toolbar {{
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 10px 14px;
                border-bottom: 1px solid rgba(110,203,232,.12);
                background: rgba(255,255,255,.02);
            }}

            .dot {{
                width: 10px;
                height: 10px;
                border-radius: 999px;
                display: inline-block;
            }}

            .dot-coral {{ background: var(--accent-coral); }}
            .dot-peach {{ background: var(--accent-peach); }}
            .dot-cyan {{ background: var(--terminal-accent); }}

            .terminal-title {{
                margin-left: 10px;
                font-family: 'Fira Code', Consolas, monospace;
                font-size: 12px;
                letter-spacing: .08em;
                color: var(--terminal-muted);
            }}

            .terminal-pulse {{
                margin-left: auto;
                width: 9px;
                height: 9px;
                border-radius: 999px;
                background: var(--terminal-accent);
                box-shadow: 0 0 14px rgba(110,203,232,.65);
                animation: pulse-dot 1.5s infinite ease-in-out;
            }}

            .retro-terminal-body {{
                padding: 14px;
            }}

            .terminal-pre {{
                margin: 0;
                min-height: calc(100vh - 106px);
                max-height: calc(100vh - 106px);
                overflow-y: auto;
                color: var(--terminal-text);
                background: transparent;
                font-size: 12px;
                line-height: 1.55;
                font-family: 'Fira Code', Consolas, monospace;
                display: flex;
                flex-direction: column;
                gap: 0;
            }}

            .log-line {{
                padding: 3px 0;
                border-bottom: 1px solid rgba(255,255,255,.03);
            }}

            .log-news {{ color: #8fe7ff; }}
            .log-apolo {{ color: #ffd39f; }}
            .log-backtest {{ color: #b8f1c8; }}
            .log-bot {{ color: #f7c2d8; }}
            .log-config {{ color: #d9ddf2; }}
            .log-report {{ color: #f6e29a; }}
            .log-error {{ color: #ff9b9b; font-weight: 700; }}
            .log-watch {{ color: #8fb3c1; }}
            .log-plain {{ color: var(--terminal-text); }}

            .terminal-actions {{
                display: flex;
                justify-content: flex-end;
                gap: 10px;
                margin-top: 10px;
                flex-wrap: wrap;
            }}

            .terminal-status-row {{
                display: flex;
                align-items: center;
                gap: 16px;
                margin-top: 14px;
            }}

            .terminal-progress {{
                height: 8px;
                border-radius: 999px;
                background: rgba(255,255,255,.07);
                flex: 1;
                overflow: hidden;
            }}

            .terminal-progress span {{
                display: block;
                width: 38%;
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, rgba(110,203,232,.18), var(--terminal-accent), rgba(110,203,232,.18));
                animation: terminal-flow 2.4s infinite linear;
            }}

            .terminal-cursor {{
                color: var(--terminal-accent);
                font-family: 'Fira Code', Consolas, monospace;
                animation: blink 1.0s steps(1) infinite;
            }}

            .glass-tabs {{
                background: linear-gradient(180deg, rgba(16,25,35,.96), rgba(22,35,49,.96));
                border: 1px solid rgba(110,203,232,.14);
                border-radius: 16px;
                padding: 6px;
            }}

            .glass-tabs .q-tab {{
                border-radius: 12px;
                margin-right: 6px;
                padding: 8px 14px;
                min-height: 38px;
                color: var(--terminal-muted) !important;
                background: rgba(255,255,255,.02);
                font-family: 'Fira Code', Consolas, monospace;
            }}

            .glass-tabs .q-tab--active {{
                background: linear-gradient(180deg, rgba(32,49,67,.98), rgba(16,25,35,.98));
                color: var(--terminal-text) !important;
                border: 1px solid rgba(110,203,232,.18);
            }}

            .apolo-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
            }}

            .mini-pre {{
                margin: 0;
                white-space: pre-wrap;
                font-family: 'Fira Code', Consolas, monospace;
                font-size: 12px;
                line-height: 1.55;
                color: var(--text-main);
            }}

            .raw-report-box textarea {{
                font-family: Consolas, monospace !important;
                font-size: 12px !important;
                min-height: 340px !important;
            }}

            @keyframes pulse-dot {{
                0%,100% {{ transform: scale(1); opacity: .75; }}
                50% {{ transform: scale(1.25); opacity: 1; }}
            }}

            @keyframes blink {{
                0%,49% {{ opacity: 1; }}
                50%,100% {{ opacity: 0; }}
            }}

            @keyframes terminal-flow {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(260%); }}
            }}

            @media (max-width: 1280px) {{
                .main-grid {{ grid-template-columns: 1fr; }}
                .left-sticky {{ position: static; }}
                .terminal-pre {{ min-height: 420px; max-height: 420px; }}
                .apolo-grid {{ grid-template-columns: 1fr; }}
                .ticker-meta {{ width: 100%; }}
            }}
        </style>
        """
    )


# ======================== IDLE PAGE (INTEGRADA) ========================
IDLE_ASSET_BG = BASE_DIR / "cryp_co_012.png"

@ui.page("/idle")
def idle_page() -> None:
    """Página STRATUM Idle (monitoreo pasivo) integrada en la misma aplicación."""
    # Copiar estado compartido desde la GUI principal
    idle_state = {
        'last_terminal_line': '[idle] waiting for signal...',
        'headline': 'Awaiting signal',
        'strategy': current_strategy_name,
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

    # Usar la misma función de KuCoin para la página idle
    def fetch_kucoin_klines_idle(symbol: str = 'BTC-USDT', interval: str = '15m', limit: int = 140) -> pd.DataFrame:
        return fetch_kucoin_klines(symbol, interval, limit)

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
        idle_state['btc_price'] = f'{last_close:,.2f}'
        idle_state['btc_change_day'] = f'{day_change_pct:+.2f}%'
        idle_state['btc_day_range'] = f'{day_low:,.0f} — {day_high:,.0f}'
        idle_state['btc_trend'] = trend

    def build_candles_idle() -> go.Figure:
        df = fetch_kucoin_klines_idle(interval='15m', limit=140)
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
            xaxis=dict(showgrid=False, visible=False, rangeslider=dict(visible=False)),
            yaxis=dict(showgrid=False, visible=False),
        )
        return fig

    def render_candles_idle(candle_box) -> None:
        if candle_box is None:
            return
        try:
            candle_box.clear()
            with candle_box:
                ui.plotly(build_candles_idle()).classes('w-full h-64')
        except Exception:
            pass

    def refresh_idle_state(hero_label, signal_label, strategy_label, meta_label,
                           apolo_label, bot_label, best_label,
                           btc_price_label, btc_change_label, btc_range_label, btc_trend_label,
                           clock_label, terminal_box, candle_box):
        shared = load_shared_state()
        lines = shared.get('last_terminal_lines') or tail_lines(LIVE_TERMINAL_FILE, 40)
        last_line = shared.get('last_terminal_line') or (lines[-1] if lines else idle_state['last_terminal_line'])
        idle_state['last_terminal_line'] = last_line
        idle_state['terminal_lines'] = lines[-40:] if lines else [idle_state['last_terminal_line']]
        idle_state['headline'] = extract_headline(last_line)
        idle_state['strategy'] = shared.get('strategy', idle_state['strategy'])
        idle_state['apolo_status'] = detect_apolo_status(shared, last_line)
        idle_state['bot_status'] = detect_bot_status(shared, last_line)
        idle_state['best_score'] = detect_best_score()
        idle_state['updated_at'] = shared.get('updated_at', time.strftime('%H:%M:%S'))

        # Actualizar UI
        if hero_label:
            hero_label.set_text(idle_state['headline'])
        if signal_label:
            signal_label.set_text(signal_mode(idle_state['last_terminal_line']))
        if strategy_label:
            strategy_label.set_text(f'STRATEGY · {idle_state["strategy"]}')
        if meta_label:
            meta_label.set_text(f'Updated {idle_state["updated_at"]} · DOGMA COIN SAPI DE CV')
        if apolo_label:
            apolo_label.set_text(f'APOLO {idle_state["apolo_status"]}')
        if bot_label:
            bot_label.set_text(f'BOT {idle_state["bot_status"]}')
        if best_label:
            best_label.set_text(f'BEST SCORE {idle_state["best_score"]}')
        if btc_price_label:
            btc_price_label.set_text(idle_state['btc_price'])
        if btc_change_label:
            btc_change_label.set_text(idle_state['btc_change_day'])
        if btc_range_label:
            btc_range_label.set_text(idle_state['btc_day_range'])
        if btc_trend_label:
            btc_trend_label.set_text(idle_state['btc_trend'])
        if clock_label:
            clock_label.set_text(time.strftime('%H:%M:%S'))
        if terminal_box:
            terminal_box.set_content(build_terminal_markup(idle_state['terminal_lines']))

    # Construir la interfaz Idle
    add_styles()  # Reutilizar estilos de la GUI principal

    with ui.column().classes('ambient-shell w-full'):
        with ui.element('div').classes('ambient-surface'):
            with ui.element('div').classes('ambient-grid'):
                with ui.element('div').classes('ambient-left'):
                    with ui.element('div').classes('brand-row'):
                        with ui.element('div').classes('brand-wrap'):
                            ui.label('STRATUM').classes('brand-title')
                            ui.label('by DOGMA · ambient decision surface').classes('brand-sub')
                        with ui.element('div').classes('top-right-wrap'):
                            strategy_label = ui.label(f'STRATEGY · {idle_state["strategy"]}').classes('strategy-pill')
                            ui.button('Abrir STRATUM GUI', on_click=lambda: ui.navigate.to('/platform')).classes('ambient-nav-btn')
                    with ui.element('div').classes('hero-block'):
                        ui.label('Latest signal').classes('eyebrow')
                        hero_label = ui.label(idle_state['headline']).classes('hero-text')
                        signal_label = ui.label(signal_mode(idle_state['last_terminal_line'])).classes('signal-label')
                        clock_label = ui.label(time.strftime('%H:%M:%S')).classes('clock-line')
                    with ui.element('div').classes('meta-row'):
                        apolo_label = ui.label(f'APOLO {idle_state["apolo_status"]}').classes('soft-chip')
                        bot_label = ui.label(f'BOT {idle_state["bot_status"]}').classes('soft-chip')
                        best_label = ui.label(f'BEST SCORE {idle_state["best_score"]}').classes('soft-chip')
                    meta_label = ui.label(f'Updated {idle_state["updated_at"]} · DOGMA COIN SAPI DE CV').classes('meta-line')
                with ui.element('div').classes('ambient-right'):
                    with ui.element('div').classes('float-panel'):
                        ui.label('BTC market pulse · 15m').classes('panel-kicker')
                        with ui.element('div').classes('market-stats'):
                            with ui.element('div').classes('stat-card'):
                                ui.label('Last price').classes('stat-k')
                                btc_price_label = ui.label(idle_state['btc_price']).classes('stat-v')
                            with ui.element('div').classes('stat-card'):
                                ui.label('Day change').classes('stat-k')
                                btc_change_label = ui.label(idle_state['btc_change_day']).classes('stat-v')
                            with ui.element('div').classes('stat-card'):
                                ui.label('Day range').classes('stat-k')
                                btc_range_label = ui.label(idle_state['btc_day_range']).classes('stat-v')
                            with ui.element('div').classes('stat-card'):
                                ui.label('Trend').classes('stat-k')
                                btc_trend_label = ui.label(idle_state['btc_trend']).classes('stat-v')
                        with ui.element('div').classes('candle-glass'):
                            candle_box = ui.column().classes('w-full')
                        terminal_box = ui.html(build_terminal_markup(idle_state['terminal_lines'])).classes('w-full')

    # Timer de refresco
    def idle_refresh_loop():
        refresh_idle_state(hero_label, signal_label, strategy_label, meta_label,
                           apolo_label, bot_label, best_label,
                           btc_price_label, btc_change_label, btc_range_label, btc_trend_label,
                           clock_label, terminal_box, candle_box)
        render_candles_idle(candle_box)

    ui.timer(2.0, idle_refresh_loop)
    ui.timer(30.0, lambda: render_candles_idle(candle_box))
    idle_refresh_loop()  # inicial


# ======================== DASHBOARD PRINCIPAL (con enlace a idle) ========================
def build_dashboard() -> None:
    """Construye la interfaz principal de STRATUM (sin autenticación)."""
    global terminal_html, stats_card, raw_report_box, plot_eq, plot_pie, plot_hist, plot_scatter, plot_bar, live_chart_box
    global header_status, market_watch_label, news_source_label, strategy_name_label
    global apolo_status_badge, apolo_progress_label, apolo_best_box, apolo_meta_label

    add_styles()

    with ui.column().classes("page-shell w-full"):
        with ui.element("div").classes("main-grid w-full"):
            with ui.column().classes("w-full left-sticky"):
                with ui.card().classes("panel-card w-full"):
                    terminal_html = ui.html(terminal_markup()).classes("w-full")
                    with ui.row().classes("terminal-actions w-full"):
                        ui.button("Copiar", on_click=copy_terminal_to_clipboard).classes("btn-soft")
                        ui.button("Exportar TXT", on_click=export_terminal).classes("btn-soft")
                        ui.button("Limpiar", on_click=lambda: (output_log.clear(), render_terminal(), write_ambient_state())).classes("btn-soft")

            with ui.column().classes("w-full gap-4"):
                with ui.card().classes("hero-wrap w-full"):
                    with ui.row().classes("w-full items-center justify-between gap-4 no-wrap"):
                        with ui.column().classes("gap-1"):
                            ui.label("STRATUM").classes("hero-title")
                            ui.label("by DOGMA · superficie de decisión para backtesting, ejecución, optimización y lectura contextual del mercado.").classes("hero-subtitle")
                        with ui.column().classes("items-end gap-1"):
                            header_status = ui.label("READY").classes("hero-badge")
                            ui.label("DOGMA COIN SAPI DE CV · execution surface · market intelligence").classes("section-help text-right")

                    strategy_name_label = ui.label("").classes("strategy-chip")
                    update_strategy_label()

                    with ui.row().classes("chart-header-row w-full mt-3"):
                        with ui.column().classes("gap-1"):
                            ui.label("BTCUSDT · vista live dinámica").classes("section-title")
                            ui.label("La gráfica toma el timeframe live configurado y puede superponer las operaciones del último backtest.").classes("section-help")
                        with ui.row().classes("gap-2 flex-wrap"):
                            ui.button("Refrescar gráfico", on_click=lambda: refresh_live_chart(True)).classes("btn-soft")
                            ui.button("Reset live", on_click=reset_live_chart).classes("btn-soft")

                    live_chart_box = ui.column().classes("w-full")

                    with ui.row().classes("ticker-band w-full mt-3"):
                        ui.label("MARKET INTELLIGENCE").classes("ticker-kicker")
                        market_watch_label = ui.label(FALLBACK_WATCH_MESSAGES[0]).classes("ticker-text")
                        news_source_label = ui.label("Inicializando feed…").classes("ticker-meta")

                    with ui.element("div").classes("header-actions w-full"):
                        ui.button("Correr backtest", on_click=start_backtest).classes("btn-primary")
                        ui.button("Cargar estrategia", on_click=prompt_load_config).classes("btn-soft")
                        ui.button("Guardar estrategia", on_click=prompt_save_strategy).classes("btn-soft")
                        ui.button("Iniciar bot de trading", on_click=prompt_launch_bot).classes("btn-warm")
                        ui.button("Vista IDLE", on_click=lambda: ui.navigate.to("/idle")).classes("btn-soft")
                        # Botón redundante eliminado para evitar confusión

                with ui.tabs().classes("glass-tabs w-full") as tabs:
                    tab_params = ui.tab("Parámetros")
                    tab_apolo = ui.tab("Apolo")
                    tab_stats = ui.tab("Resumen")
                    tab_charts = ui.tab("Gráficos")

                with ui.tab_panels(tabs, value=tab_params).classes("w-full bg-transparent"):
                    with ui.tab_panel(tab_params):
                        with ui.card().classes("panel-card w-full"):
                            ui.label("Strategy model").classes("section-title")
                            ui.label("Feature engine moderno + legacy mode de compatibilidad.").classes("section-help mb-4")

                            with ui.column().classes("param-stack w-full"):
                                for section in PARAM_SECTIONS:
                                    expansion_open = section["title"] == "Mercado y ejecución"
                                    with ui.expansion(section["title"], icon=section["icon"], value=expansion_open).classes("w-full"):
                                        for item in section["items"]:
                                            typ = item["typ"]
                                            p = tuple(item["path"])
                                            current = get_path_value(p)
                                            key = path_key(p)

                                            with ui.element("div").classes("param-block"):
                                                with ui.row().classes("param-head w-full items-start justify-between no-wrap"):
                                                    with ui.column().classes("gap-1"):
                                                        ui.label(item["label"]).classes("param-label")
                                                        if item.get("help"):
                                                            ui.label(item["help"]).classes("param-help")

                                                    value_chip = ui.label(format_value(current, typ, item.get("unit"))).classes("param-value-chip")
                                                    param_value_chips[key] = value_chip

                                                if typ == "str" and item.get("choices"):
                                                    ctrl = ui.toggle(item["choices"], value=current)
                                                    ctrl.on_value_change(lambda e, p_=p, t=typ, u=item.get("unit"), chip=value_chip: (set_path_value(p_, e.value), chip.set_text(format_value(e.value, t, u))))
                                                elif typ == "str":
                                                    ctrl = ui.input(value=str(current or ""), placeholder=item.get("help", ""))
                                                    ctrl.props("outlined")
                                                    ctrl.on_value_change(lambda e, p_=p, t=typ, u=item.get("unit"), chip=value_chip: (set_path_value(p_, str(e.value)), chip.set_text(format_value(e.value, t, u))))
                                                elif typ == "bool":
                                                    ctrl = ui.switch(value=bool(current))
                                                    ctrl.on_value_change(lambda e, p_=p, t=typ, u=item.get("unit"), chip=value_chip: (set_path_value(p_, bool(e.value)), chip.set_text(format_value(e.value, t, u))))
                                                elif typ == "int":
                                                    ctrl = ui.slider(min=item["min_val"], max=item["max_val"], step=item["step"], value=int(current)).props("label label-always snap")
                                                    ctrl.on_value_change(lambda e, p_=p, t=typ, u=item.get("unit"), chip=value_chip: (set_path_value(p_, int(e.value)), chip.set_text(format_value(e.value, t, u))))
                                                else:
                                                    ctrl = ui.slider(min=item["min_val"], max=item["max_val"], step=item["step"], value=float(current)).props("label label-always")
                                                    ctrl.on_value_change(lambda e, p_=p, t=typ, u=item.get("unit"), chip=value_chip: (set_path_value(p_, float(e.value)), chip.set_text(format_value(e.value, t, u))))

                                                ctrl.classes("param-input w-full")
                                                param_controls[key] = ctrl

                    with ui.tab_panel(tab_apolo):
                        with ui.card().classes("panel-card w-full"):
                            with ui.row().classes("w-full items-center justify-between"):
                                with ui.column().classes("gap-1"):
                                    ui.label("Optimization engine · APOLO").classes("section-title")
                                    ui.label("Monitor de búsqueda de parámetros. Lee progreso, mejores hallazgos y actividad reciente del optimizador.").classes("section-help")
                                apolo_status_badge = ui.label("Apolo IDLE").classes("status-chip")

                            with ui.row().classes("w-full gap-3 mt-4 flex-wrap"):
                                ui.button("Iniciar + reanudar", on_click=lambda: start_apolo(True)).classes("btn-primary")
                                ui.button("Iniciar limpio", on_click=lambda: start_apolo(False)).classes("btn-soft")
                                ui.button("Actualizar estado", on_click=connect_apolo_monitor).classes("btn-soft")
                                ui.button("Detener", on_click=stop_apolo).classes("btn-warm")

                            with ui.element("div").classes("apolo-grid w-full mt-5"):
                                with ui.card().classes("panel-card w-full"):
                                    ui.label("Progreso actual").classes("section-title")
                                    apolo_progress_label = ui.label(f"Sin {APOLO_PROGRESS_FILE.name} todavía.").classes("section-help")
                                    apolo_meta_label = ui.label("Apolo aún no ha generado metadatos locales.").classes("section-help mt-4")

                                with ui.card().classes("panel-card w-full"):
                                    ui.label("Mejores parámetros encontrados").classes("section-title")
                                    apolo_best_box = ui.html(render_best_params(None)).classes("w-full")

                    with ui.tab_panel(tab_stats):
                        with ui.card().classes("panel-card w-full"):
                            ui.label("Estadísticas del backtest").classes("section-title")
                            ui.label("Aquí se condensan los resultados operativos más útiles para iterar parámetros con rapidez.").classes("section-help mb-4")
                            stats_card = ui.column().classes("w-full gap-4")

                            ui.separator().classes("my-5")

                            ui.label("Reporte bruto para análisis / ML").classes("section-title")
                            ui.label("Texto crudo con config, métricas, muestras de trades y contexto útil para alimentar modelos que sugieran mejores parámetros.").classes("section-help mb-3")

                            with ui.row().classes("terminal-actions w-full"):
                                ui.button("Copiar reporte", on_click=copy_raw_report_to_clipboard).classes("btn-soft")
                                ui.button("Exportar reporte TXT", on_click=export_raw_report).classes("btn-soft")

                            raw_report_box = ui.textarea(label="Raw report", value="Aún no hay reporte. Ejecuta un backtest.").props("filled autogrow").classes("w-full raw-report-box")

                    with ui.tab_panel(tab_charts):
                        with ui.card().classes("panel-card w-full"):
                            ui.label("Exploración visual").classes("section-title")
                            ui.label("Gráficos orientados a análisis operativo, edge y diagnóstico de riesgo.").classes("section-help mb-4")

                            with ui.tabs().classes("glass-tabs w-full") as chart_tabs:
                                tab_eq = ui.tab("Equity")
                                tab_pie = ui.tab("Salidas")
                                tab_hist = ui.tab("Distribución")
                                tab_scatter = ui.tab("R vs MaxR")
                                tab_bar = ui.tab("PnL diario")

                            with ui.tab_panels(chart_tabs, value=tab_eq).classes("w-full bg-transparent"):
                                with ui.tab_panel(tab_eq):
                                    plot_eq = ui.column().classes("w-full")
                                with ui.tab_panel(tab_pie):
                                    plot_pie = ui.column().classes("w-full")
                                with ui.tab_panel(tab_hist):
                                    plot_hist = ui.column().classes("w-full")
                                with ui.tab_panel(tab_scatter):
                                    plot_scatter = ui.column().classes("w-full")
                                with ui.tab_panel(tab_bar):
                                    plot_bar = ui.column().classes("w-full")

    ui.timer(0.35, process_ui_events)

    append_log("[ui] STRATUM cargado correctamente")
    refresh_apolo_status(force=True)
    refresh_live_macro_news(background=True)
    refresh_live_chart(force=True)
    write_ambient_state()


# ======================== DOGMA LANDING PAGE ========================
def render_dogma_landing() -> None:
    add_styles()

    ui.add_head_html("""
    <style>
        .landing-shell {
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }

        .landing-shell::before {
            content: '';
            position: fixed;
            inset: 0;
            pointer-events: none;
            background:
                radial-gradient(circle at 12% 14%, rgba(168,221,229,.42), transparent 20%),
                radial-gradient(circle at 86% 18%, rgba(233,138,107,.20), transparent 22%),
                radial-gradient(circle at 72% 82%, rgba(217,221,242,.34), transparent 20%),
                radial-gradient(circle at 35% 88%, rgba(231,199,214,.22), transparent 22%);
            z-index: 0;
        }

        .landing-wrap {
            position: relative;
            z-index: 1;
            max-width: 1480px;
            margin: 0 auto;
            padding: 24px 18px 40px;
        }

        .landing-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 16px 0 20px;
        }

        .dogma-mark {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .dogma-name {
            font-size: 30px;
            line-height: 1;
            font-weight: 900;
            letter-spacing: .22em;
            color: #15324b;
        }

        .dogma-sub {
            font-size: 10px;
            letter-spacing: .36em;
            text-transform: uppercase;
            color: #6a7682;
        }

        .landing-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .landing-hero {
            display: grid;
            grid-template-columns: 1.05fr .95fr;
            gap: 18px;
            align-items: stretch;
            margin-top: 10px;
        }

        .landing-card {
            border-radius: 28px;
            border: 1px solid rgba(200,191,175,.56);
            background: linear-gradient(180deg, rgba(255,255,255,.82), rgba(251,248,242,.92));
            box-shadow: 0 18px 40px rgba(35,68,107,.08);
            backdrop-filter: blur(12px);
        }

        .landing-copy {
            padding: 34px 30px 30px;
        }

        .landing-kicker {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,.72);
            border: 1px solid rgba(216,209,197,.82);
            color: #33526e;
            font-size: 12px;
            font-weight: 700;
            box-shadow: 0 8px 20px rgba(35,68,107,.05);
        }

        .landing-kicker::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #2373B8;
            box-shadow: 0 0 12px rgba(35,115,184,.45);
        }

        .landing-title {
            margin-top: 18px;
            font-size: 68px;
            line-height: .92;
            letter-spacing: -.055em;
            font-weight: 900;
            color: #143148;
            max-width: 820px;
        }

        .landing-title .gradient {
            display: block;
            background: linear-gradient(90deg, #2373B8 0%, #6ECBE8 42%, #E98A6B 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .landing-body {
            margin-top: 20px;
            max-width: 760px;
            font-size: 18px;
            line-height: 1.7;
            color: #4b5c6a;
        }

        .landing-cta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 24px;
        }

        .landing-terminal {
            padding: 18px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 100%;
        }

        .landing-terminal-shell {
            background: linear-gradient(180deg, rgba(15,23,32,.98), rgba(22,35,49,.98));
            border-radius: 24px;
            overflow: hidden;
            border: 1px solid rgba(110,203,232,.18);
            box-shadow: 0 18px 50px rgba(11,25,38,.24);
        }

        .landing-terminal-head {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-bottom: 1px solid rgba(255,255,255,.08);
            background: rgba(255,255,255,.03);
        }

        .landing-dot {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
        }

        .landing-dot.coral { background: #E98A6B; }
        .landing-dot.peach { background: #EDB08A; }
        .landing-dot.cyan  { background: #6ECBE8; }

        .landing-terminal-title {
            margin-left: 8px;
            color: #8FB3C1;
            font-family: 'Fira Code', Consolas, monospace;
            font-size: 12px;
            letter-spacing: .16em;
        }

        .landing-terminal-body {
            padding: 18px 18px 20px;
            color: #DFF6FF;
            font-family: 'Fira Code', Consolas, monospace;
            font-size: 12px;
            line-height: 1.7;
        }

        .landing-progress {
            margin-top: 16px;
            height: 8px;
            border-radius: 999px;
            background: rgba(255,255,255,.08);
            overflow: hidden;
        }

        .landing-progress span {
            display: block;
            width: 46%;
            height: 100%;
            background: linear-gradient(90deg, rgba(110,203,232,.12), #6ECBE8, #E98A6B);
            animation: terminal-flow 2.6s linear infinite;
        }

        .landing-bars {
            display: grid;
            grid-template-columns: repeat(12, minmax(0, 1fr));
            gap: 6px;
            align-items: end;
            height: 118px;
            margin-top: 20px;
        }

        .landing-bars span {
            border-radius: 999px 999px 0 0;
            background: linear-gradient(180deg, rgba(204,238,245,.88), rgba(35,115,184,.92), rgba(233,138,107,.88));
            opacity: .92;
        }

        .landing-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 18px;
        }

        .landing-info-card {
            padding: 22px 22px 20px;
        }

        .landing-section-kicker {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: .28em;
            color: #6e7b87;
            font-weight: 700;
        }

        .landing-section-title {
            margin-top: 10px;
            font-size: 34px;
            line-height: 1.02;
            letter-spacing: -.04em;
            font-weight: 800;
            color: #143148;
        }

        .landing-paragraph {
            margin-top: 16px;
            font-size: 15px;
            line-height: 1.75;
            color: #556574;
        }

        .landing-list {
            display: grid;
            gap: 10px;
            margin-top: 18px;
        }

        .landing-list-item {
            border-radius: 18px;
            background: rgba(255,255,255,.70);
            border: 1px solid rgba(255,255,255,.94);
            padding: 14px 14px 12px;
        }

        .landing-list-title {
            color: #17314a;
            font-size: 15px;
            font-weight: 700;
        }

        .landing-list-text {
            margin-top: 4px;
            color: #607181;
            font-size: 13px;
            line-height: 1.6;
        }

        .landing-dark-band {
            margin-top: 18px;
            border-radius: 28px;
            overflow: hidden;
            background:
                radial-gradient(circle at 80% 24%, rgba(233,138,107,.24), transparent 20%),
                radial-gradient(circle at 24% 18%, rgba(35,115,184,.36), transparent 24%),
                linear-gradient(180deg, #13202d 0%, #0f1b27 100%);
            border: 1px solid rgba(200,191,175,.16);
            box-shadow: 0 18px 46px rgba(35,68,107,.12);
        }

        .landing-dark-grid {
            display: grid;
            grid-template-columns: 1fr .9fr;
        }

        .landing-dark-copy {
            padding: 32px 28px;
            color: #E6F6FF;
        }

        .landing-dark-title {
            margin-top: 12px;
            font-size: 38px;
            line-height: 1.02;
            letter-spacing: -.04em;
            font-weight: 800;
        }

        .landing-dark-body {
            margin-top: 18px;
            color: #b8d4e2;
            font-size: 15px;
            line-height: 1.75;
            max-width: 760px;
        }

        .landing-dark-right {
            padding: 28px;
            display: flex;
            align-items: center;
        }

        .landing-core-box {
            width: 100%;
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,.10);
            background: rgba(255,255,255,.05);
            backdrop-filter: blur(10px);
            padding: 20px;
        }

        .landing-core-item {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 12px;
            color: #E6F6FF;
            font-size: 14px;
        }

        .landing-core-index {
            width: 38px;
            height: 38px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,.08);
            border: 1px solid rgba(255,255,255,.10);
            color: #6ECBE8;
            font-weight: 800;
        }

        .landing-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            padding: 26px 2px 6px;
            color: #5f7080;
            font-size: 14px;
        }

        .landing-footer strong {
            color: #17314a;
        }

        @media (max-width: 1100px) {
            .landing-hero,
            .landing-grid,
            .landing-dark-grid {
                grid-template-columns: 1fr;
            }

            .landing-title {
                font-size: 50px;
            }

            .landing-nav {
                flex-direction: column;
                align-items: flex-start;
            }
        }

        @media (max-width: 720px) {
            .landing-title {
                font-size: 40px;
            }

            .landing-body {
                font-size: 16px;
            }

            .landing-wrap {
                padding: 16px 12px 28px;
            }
        }
    </style>
    """)

    with ui.column().classes('landing-shell w-full'):
        with ui.column().classes('landing-wrap w-full'):
            with ui.row().classes('landing-nav w-full'):
                with ui.column().classes('dogma-mark'):
                    ui.label('DOGMA').classes('dogma-name')
                    ui.label('digital finance tools · crypto infrastructure · portfolio intelligence').classes('dogma-sub')

                with ui.row().classes('landing-actions'):
                    ui.button('Más información', on_click=lambda: ui.run_javascript("document.getElementById('about-section')?.scrollIntoView({behavior:'smooth'})")).classes('btn-soft')
                    ui.button('Disclaimer', on_click=lambda: ui.run_javascript("document.getElementById('disclaimer-section')?.scrollIntoView({behavior:'smooth'})")).classes('btn-soft')
                    ui.button('Login / plataforma', on_click=lambda: ui.navigate.to('/login')).classes('btn-primary')

            with ui.element('div').classes('landing-hero w-full'):
                with ui.element('div').classes('landing-card landing-copy'):
                    ui.label('Finetech indie para infraestructura, estrategia y operación digital').classes('landing-kicker')
                    ui.html("""
                        <div class="landing-title">
                            Herramientas para
                            <span class="gradient">finanzas digitales</span>
                            y sistemas de decisión.
                        </div>
                    """)
                    ui.label(
                        'DOGMA desarrolla superficies de análisis, operación, visualización de portafolios y herramientas para activos digitales con una estética limpia, inmersiva y orientada a decisión.'
                    ).classes('landing-body')

                    with ui.row().classes('landing-cta-row'):
                        ui.button('Entrar a DOGMA Tools', on_click=lambda: ui.navigate.to('/login')).classes('btn-warm')
                        ui.button('Quiénes somos', on_click=lambda: ui.run_javascript("document.getElementById('about-section')?.scrollIntoView({behavior:'smooth'})")).classes('btn-soft')
                        ui.button('Avisos y alcance', on_click=lambda: ui.run_javascript("document.getElementById('disclaimer-section')?.scrollIntoView({behavior:'smooth'})")).classes('btn-soft')

                with ui.element('div').classes('landing-card landing-terminal'):
                    ui.html("""
                        <div class="landing-terminal-shell">
                            <div class="landing-terminal-head">
                                <span class="landing-dot coral"></span>
                                <span class="landing-dot peach"></span>
                                <span class="landing-dot cyan"></span>
                                <span class="landing-terminal-title">DOGMA TERMINAL</span>
                            </div>
                            <div class="landing-terminal-body">
                                <div>[live] digital finance surface online</div>
                                <div>[tools] portfolio intelligence ready</div>
                                <div>[signals] systematic research layer initialized</div>
                                <div>[status] awaiting operator intent...</div>
                                <div class="landing-progress"><span></span></div>
                                <div class="landing-bars">
                                    <span style="height:38%"></span>
                                    <span style="height:54%"></span>
                                    <span style="height:48%"></span>
                                    <span style="height:62%"></span>
                                    <span style="height:46%"></span>
                                    <span style="height:69%"></span>
                                    <span style="height:73%"></span>
                                    <span style="height:66%"></span>
                                    <span style="height:59%"></span>
                                    <span style="height:76%"></span>
                                    <span style="height:68%"></span>
                                    <span style="height:71%"></span>
                                </div>
                            </div>
                        </div>
                    """)

            with ui.element('div').classes('landing-grid w-full'):
                with ui.element('div').props('id=about-section').classes('landing-card landing-info-card'):
                    ui.label('Quiénes somos').classes('landing-section-kicker')
                    ui.label('DOGMA es una firma independiente orientada a tecnología financiera y activos digitales.').classes('landing-section-title')
                    ui.label(
                        'Construimos herramientas para análisis, operación, visualización de portafolios y superficies de decisión para usuarios y equipos que necesitan navegar mercados digitales con más claridad, control y diseño.'
                    ).classes('landing-paragraph')
                    ui.label(
                        'Nuestro enfoque combina software, investigación, automatización, interfaces especializadas y una visión modular para productos de crypto, administración de fondos y ecosistemas de finanzas digitales.'
                    ).classes('landing-paragraph')

                with ui.element('div').classes('landing-card landing-info-card'):
                    ui.label('Qué hacemos').classes('landing-section-kicker')
                    with ui.element('div').classes('landing-list'):
                        items = [
                            ('DOGMA Tools', 'Plataforma de herramientas para investigación, operación y superficies de análisis.'),
                            ('Crypto & digital finance', 'Capas de trabajo orientadas a activos digitales, infraestructura cuantitativa y flujos de mercado.'),
                            ('Portafolios y manejo de fondos', 'Interfaces y sistemas de apoyo para monitoreo, análisis y organización de estrategias.'),
                            ('Arquitectura modular', 'Landing, login, productos, dashboards y capas de riesgo diseñadas para crecer por etapas.'),
                        ]
                        for title, text in items:
                            with ui.element('div').classes('landing-list-item'):
                                ui.label(title).classes('landing-list-title')
                                ui.label(text).classes('landing-list-text')

            with ui.element('div').classes('landing-dark-band w-full'):
                with ui.element('div').classes('landing-dark-grid'):
                    with ui.element('div').classes('landing-dark-copy'):
                        ui.label('Surface').classes('landing-section-kicker')
                        ui.label('Una capa de acceso para herramientas cuantitativas, operación y exploración financiera.').classes('landing-dark-title')
                        ui.label(
                            'El landing está pensado como umbral de entrada: identidad, claridad institucional básica, acceso a la plataforma y futuras rutas para productos, onboarding y documentación.'
                        ).classes('landing-dark-body')
                        with ui.row().classes('landing-cta-row'):
                            ui.button('Continuar a la plataforma', on_click=lambda: ui.navigate.to('/login')).classes('btn-soft')
                            ui.button('Ver disclaimer', on_click=lambda: ui.run_javascript("document.getElementById('disclaimer-section')?.scrollIntoView({behavior:'smooth'})")).classes('btn-soft')

                    with ui.element('div').classes('landing-dark-right'):
                        with ui.element('div').classes('landing-core-box'):
                            ui.label('Core vectors').classes('landing-section-kicker')
                            core = [
                                'Research interfaces',
                                'Systematic execution surfaces',
                                'Digital finance tooling',
                                'Portfolio visualization layers',
                                'Crypto-native product architecture',
                            ]
                            for i, item in enumerate(core, start=1):
                                with ui.element('div').classes('landing-core-item'):
                                    ui.label(str(i)).classes('landing-core-index')
                                    ui.label(item)

            with ui.element('div').props('id=disclaimer-section').classes('landing-card landing-info-card w-full mt-4'):
                ui.label('Disclaimer').classes('landing-section-kicker')
                ui.label('Información general, acceso a herramientas y contexto institucional.').classes('landing-section-title')
                ui.label(
                    'El contenido del sitio es de carácter informativo y de acceso a herramientas. No constituye una invitación pública general a invertir ni sustituye asesoría legal, fiscal, contable o financiera personalizada.'
                ).classes('landing-paragraph')
                ui.label(
                    'Cualquier uso de herramientas, interfaces, señales, simulaciones, análisis o visualizaciones debe entenderse dentro del marco de evaluación propia del usuario y de su perfil de riesgo.'
                ).classes('landing-paragraph')
                ui.label(
                    'Algunas funciones del ecosistema DOGMA pueden evolucionar por etapas: acceso, login, módulos internos, documentación ampliada, rutas de onboarding y productos complementarios.'
                ).classes('landing-paragraph')

            with ui.row().classes('landing-footer w-full'):
                ui.label('Fintech independiente para herramientas de finanzas digitales, crypto y capas de portafolio.')
                with ui.row().classes('landing-actions'):
                    ui.button('Ir al login', on_click=lambda: ui.navigate.to('/login')).classes('btn-soft')
                    ui.button('Ir a la plataforma', on_click=lambda: ui.navigate.to('/login')).classes('btn-primary')


# ======================== LOGIN Y ARRANQUE ========================
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "cambia_esto_en_produccion")

@ui.page("/")
def landing_page() -> None:
    """Landing público de DOGMA."""
    render_dogma_landing()

@ui.page("/login")
def login_page() -> None:
    """Página de inicio de sesión."""
    def try_login():
        if username.value == "admin" and password.value == ADMIN_PASSWORD:
            app.storage.user.update({'authenticated': True, 'username': username.value})
            ui.navigate.to('/platform')
        else:
            ui.notify('Credenciales incorrectas', type='negative')

    ui.page_title("DOGMA / STRATUM - Login")
    with ui.card().classes('absolute-center panel-card').style('min-width: 340px'):
        ui.label('DOGMA / STRATUM').classes('text-h5 text-center')
        ui.label('Acceso a la plataforma').classes('section-help text-center')
        username = ui.input('Usuario', value='admin').props('outlined')
        password = ui.input('Contraseña', password=True, password_toggle_button=True).props('outlined')
        with ui.row().classes('w-full gap-2'):
            ui.button('Ingresar', on_click=try_login).classes('btn-primary w-full')
        ui.button('Volver al landing', on_click=lambda: ui.navigate.to('/')).classes('btn-soft w-full mt-2')

@ui.page("/platform")
def protected_page() -> None:
    """Página principal protegida por autenticación."""
    if not app.storage.user.get('authenticated', False):
        ui.navigate.to('/login')
        return
    build_dashboard()

# Inicializar sesión
def init_user_session():
    if 'authenticated' not in app.storage.user:
        app.storage.user.update({'authenticated': False})

app.on_connect(init_user_session)

# Obtener puerto desde variable de entorno de Render
port = int(os.environ.get('PORT', 8083))

ui.run(
    title="DOGMA / STRATUM",
    favicon="◉",
    host="0.0.0.0",
    port=port,
    storage_secret=os.environ.get("STORAGE_SECRET", "default_secret_change_in_production"),
    reload=False
)
