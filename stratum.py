# ===============================
# STRATUM TRADER BOT (vNext) - BYBIT VERSION
# Feature-based probabilistic engine + legacy fallback
# ===============================

from __future__ import annotations
import sys
import os
import json
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import ccxt
from dotenv import load_dotenv

# ===============================
# UTF-8 FIX (IMPORTANTE WINDOWS)
# ===============================
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ===============================
# PATHS
# ===============================
BASE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BASE_DIR / "configs"
RUNTIME_DIR = BASE_DIR / "runtime"

CONFIGS_DIR.mkdir(exist_ok=True)
RUNTIME_DIR.mkdir(exist_ok=True)

CONFIG_FILE = CONFIGS_DIR / "config.json"
PARAMS_FILE = CONFIGS_DIR / "params_bt.json"

STATE_FILE = RUNTIME_DIR / "state.json"
TRADES_FILE = RUNTIME_DIR / "trades.csv"

# ===============================
# ORACLE / TELEGRAM (OPCIONAL)
# ===============================
SEND_TELEGRAM = None
try:
    from oracle import send_telegram as SEND_TELEGRAM
except Exception:
    SEND_TELEGRAM = None

# ===============================
# UTILIDADES
# ===============================
def now_ts():
    return int(datetime.now(timezone.utc).timestamp())

def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def load_json(path, default=None):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def log(msg):
    print(msg, flush=True)

def notify(msg: str):
    log(msg)
    if SEND_TELEGRAM:
        try:
            SEND_TELEGRAM(msg)
        except Exception as e:
            log(f"[WARN] telegram failed: {e}")

# ===============================
# CONFIG
# ===============================
def load_config():
    if PARAMS_FILE.exists():
        return load_json(PARAMS_FILE)
    if CONFIG_FILE.exists():
        return load_json(CONFIG_FILE)
    raise RuntimeError("No config found in configs/")

# ===============================
# EXCHANGE (BYBIT)
# ===============================
def get_exchange(testnet=True):
    load_dotenv()

    apiKey = os.getenv("BYBIT_API_KEY")
    secret = os.getenv("BYBIT_SECRET")

    if not apiKey or not secret:
        log("[ERROR] BYBIT_API_KEY or BYBIT_SECRET not set in environment")
        raise RuntimeError("Missing Bybit API credentials")

    # Configuración para Bybit
    exchange = ccxt.bybit({
        "apiKey": apiKey,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",   # "spot" o "future". Por defecto usamos futuros.
            "adjustForTimeDifference": True,
        }
    })

    if testnet:
        # Bybit testnet requiere una URL específica
        exchange.set_sandbox_mode(True)
        # Opcional: cambiar a la URL de testnet de Bybit
        # exchange.urls['api'] = 'https://api-testnet.bybit.com'
        # Pero set_sandbox_mode(True) ya debería hacerlo.

    return exchange

# ===============================
# DATA
# ===============================
def fetch_ohlc(ex, symbol, tf, limit=500):
    ohlc = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlc, columns=["time","open","high","low","close","vol"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

# ===============================
# FEATURE ENGINE (sin cambios)
# ===============================
def realized_vol(series, window=20):
    rets = np.log(series / series.shift(1))
    return rets.rolling(window).std() * np.sqrt(window)

def slope(series, window=8):
    out = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series)+1):
        y = series.iloc[i-window:i].values
        x = np.arange(len(y))
        out.iloc[i-1] = np.polyfit(x, y, 1)[0]
    return out

def build_features(df):
    f = df.copy()
    f["ret1"] = f["close"].pct_change(1)
    f["ret3"] = f["close"].pct_change(3)
    f["ret12"] = f["close"].pct_change(12)
    f["range_pos"] = (f["close"] - f["low"]) / (f["high"] - f["low"] + 1e-9)
    f["vol_ratio"] = f["vol"] / (f["vol"].rolling(20).mean() + 1e-9)
    f["rv"] = realized_vol(f["close"])
    f["rv_ratio"] = f["rv"] / (f["rv"].rolling(20).mean() + 1e-9)
    f["ma_fast"] = f["close"].rolling(12).mean()
    f["ma_slow"] = f["close"].rolling(36).mean()
    f["slope"] = slope(f["ma_fast"].bfill())
    f["breakout_up"] = f["close"] > f["high"].rolling(20).max().shift(1)
    f["breakout_dn"] = f["close"] < f["low"].rolling(20).min().shift(1)
    return f

# ===============================
# LEGACY ENGINE (sin cambios)
# ===============================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)

def legacy_signal(df, cfg):
    lg = cfg.get("legacy", {})
    work = df.copy()
    work["ema_fast"] = ema(work["close"], int(lg.get("ema_fast_signal", 50)))
    work["ema_slow"] = ema(work["close"], int(lg.get("ema_slow_signal", 200)))
    work["rsi"] = rsi(work["close"], int(lg.get("rsi_len_signal", 14)))
    row = work.iloc[-1]
    if row["ema_fast"] > row["ema_slow"] and row["rsi"] >= float(lg.get("rsi_long", 58)):
        return "buy", row
    if row["ema_fast"] < row["ema_slow"] and row["rsi"] <= float(lg.get("rsi_short", 42)):
        return "sell", row
    return None, row

# ===============================
# FEATURE SIGNAL ENGINE (sin cambios)
# ===============================
def feature_signal(row, cfg):
    fp = cfg["feature_prob"]
    score_long = 0.0
    score_short = 0.0
    if row["ret1"] > 0:
        score_long += fp["ret_1_weight"]
    else:
        score_short += fp["ret_1_weight"]
    if row["ret3"] > 0:
        score_long += fp["ret_3_weight"]
    else:
        score_short += fp["ret_3_weight"]
    if row["ret12"] > 0:
        score_long += fp["ret_12_weight"]
    else:
        score_short += fp["ret_12_weight"]
    if row["range_pos"] > 0.65:
        score_long += fp["range_pos_weight"]
    if row["range_pos"] < 0.35:
        score_short += fp["range_pos_weight"]
    if row["vol_ratio"] > 1.2:
        score_long += fp["vol_ratio_weight"] * 0.5
        score_short += fp["vol_ratio_weight"] * 0.5
    if row["rv_ratio"] > 1.05:
        score_long += fp["realized_vol_weight"] * 0.5
        score_short += fp["realized_vol_weight"] * 0.5
    if row["slope"] > 0:
        score_long += fp["slope_weight"]
    else:
        score_short += fp["slope_weight"]
    if row["breakout_up"]:
        score_long += fp["breakout_weight"]
    if row["breakout_dn"]:
        score_short += fp["breakout_weight"]
    total = max(score_long + score_short, 1e-9)
    return score_long / total, score_short / total

# ===============================
# POSITION (adaptado para Bybit)
# ===============================
def get_position(ex, symbol):
    try:
        # Bybit devuelve posiciones en fetch_positions()
        positions = ex.fetch_positions([symbol])
        for p in positions:
            contracts = float(p.get("contracts", 0) or 0)
            if contracts != 0:
                return p
    except Exception as e:
        log(f"[WARN] fetch_positions failed: {e}")
    return None

# ===============================
# ORDER (sin cambios)
# ===============================
def place_order(ex, symbol, side, size):
    try:
        return ex.create_market_order(symbol, side, size)
    except Exception as e:
        log(f"[ERR] order: {e}")
        return None

# ===============================
# STATE INIT (sin cambios)
# ===============================
def make_default_state(cfg):
    return {
        "cooldown_until": 0,
        "last_heartbeat_ts": 0,
        "last_signal": None,
        "last_entry_side": None,
        "last_entry_price": None,
        "boot_time": utc_now_str(),
        "strategy_snapshot": {
            "symbol": cfg.get("symbol"),
            "mode": cfg.get("mode"),
            "tf_signal": cfg.get("tf_signal"),
            "tf_confirm": cfg.get("tf_confirm"),
            "tf_entry": cfg.get("tf_entry"),
            "tf_trend": cfg.get("tf_trend"),
        },
    }

# ===============================
# MAIN LOOP
# ===============================
def main():
    cfg = load_config()
    testnet = os.getenv("TESTNET", "true").lower() == "true"

    ex = get_exchange(testnet)
    symbol = cfg["symbol"]

    state = load_json(STATE_FILE, None)
    if state is None:
        state = make_default_state(cfg)
        save_json(STATE_FILE, state)

    heartbeat_seconds = int(cfg.get("status_heartbeat_seconds", 60))
    poll_seconds = int(cfg.get("poll_seconds", 10))
    mode = cfg.get("mode", "feature_prob")

    notify(
        f"[BOT] iniciado | env={'TESTNET' if testnet else 'LIVE'} | exchange=BYBIT | "
        f"symbol={symbol} | mode={mode} | tf_signal={cfg.get('tf_signal')} | "
        f"tf_confirm={cfg.get('tf_confirm')} | tf_trend={cfg.get('tf_trend')}"
    )

    notify(f"[BOT] config source: {PARAMS_FILE if PARAMS_FILE.exists() else CONFIG_FILE}")

    while True:
        try:
            now = now_ts()

            # heartbeat
            if now - int(state.get("last_heartbeat_ts", 0)) >= heartbeat_seconds:
                pos = get_position(ex, symbol)
                pos_txt = "OPEN" if pos else "FLAT"
                cd_until = int(state.get("cooldown_until", 0))
                cd_left = max(0, cd_until - now)

                hb = (
                    f"[BOT][HEARTBEAT] {utc_now_str()} | "
                    f"mode={cfg.get('mode')} | symbol={symbol} | tf={cfg.get('tf_signal')} | "
                    f"status={pos_txt} | cooldown_left={cd_left}s"
                )
                log(hb)
                state["last_heartbeat_ts"] = now
                save_json(STATE_FILE, state)

            # si hay posición abierta, reportar y esperar
            pos = get_position(ex, symbol)
            if pos:
                log("[BOT] posición abierta detectada; gestión avanzada aún no implementada")
                time.sleep(poll_seconds)
                continue

            # cooldown
            if now < int(state.get("cooldown_until", 0)):
                time.sleep(poll_seconds)
                continue

            # motor
            if mode == "legacy_classic":
                df = fetch_ohlc(ex, symbol, cfg["tf_signal"], 500)
                side, row = legacy_signal(df, cfg)
                if side is None:
                    log(
                        f"[BOT][LEGACY] sin entrada | "
                        f"close={float(row['close']):.2f} | "
                        f"ema_fast={float(row['ema_fast']):.2f} | "
                        f"ema_slow={float(row['ema_slow']):.2f} | "
                        f"rsi={float(row['rsi']):.2f}"
                    )
                    time.sleep(poll_seconds)
                    continue
                price = float(row["close"])
                log(f"[BOT][LEGACY] señal {side} @ {price:.2f}")
            else:
                df = fetch_ohlc(ex, symbol, cfg["tf_signal"], 500)
                feat = build_features(df)
                row = feat.iloc[-1]
                prob_long, prob_short = feature_signal(row, cfg)
                log(
                    f"[BOT][FEATURE] close={float(row['close']):.2f} | "
                    f"prob_long={prob_long:.3f} | prob_short={prob_short:.3f} | "
                    f"rv={float(row['rv']) if pd.notna(row['rv']) else 0:.6f} | "
                    f"vol_ratio={float(row['vol_ratio']) if pd.notna(row['vol_ratio']) else 0:.3f}"
                )
                if prob_long >= cfg["feature_prob"]["long_threshold"]:
                    side = "buy"
                elif prob_short >= cfg["feature_prob"]["short_threshold"]:
                    side = "sell"
                else:
                    time.sleep(poll_seconds)
                    continue
                price = float(row["close"])
                log(f"[BOT][FEATURE] señal {side} @ {price:.2f}")

            # sizing simple
            size = 0.001   # Ajusta según tu capital y el par (Bybit puede requerir cantidad en BTC o USDT)

            order = place_order(ex, symbol, side, size)
            if order is None:
                log("[BOT] orden fallida")
                time.sleep(poll_seconds)
                continue

            state["last_entry_side"] = side
            state["last_entry_price"] = price
            state["last_signal"] = utc_now_str()
            state["cooldown_until"] = now_ts() + int(cfg["risk_engine"]["cooldown_minutes"]) * 60
            save_json(STATE_FILE, state)

            notify(
                f"[BOT] ORDEN ejecutada | side={side} | px={price:.2f} | "
                f"size={size} | cooldown={cfg['risk_engine']['cooldown_minutes']}m"
            )

            try:
                row_to_save = pd.DataFrame([{
                    "timestamp": utc_now_str(),
                    "symbol": symbol,
                    "mode": mode,
                    "side": side,
                    "price": price,
                    "size": size,
                    "env": "TESTNET" if testnet else "LIVE",
                }])
                if TRADES_FILE.exists():
                    row_to_save.to_csv(TRADES_FILE, mode="a", header=False, index=False)
                else:
                    row_to_save.to_csv(TRADES_FILE, index=False)
            except Exception as e:
                log(f"[WARN] no se pudo escribir trades.csv: {e}")

            time.sleep(poll_seconds)

        except Exception as e:
            log(f"[ERR] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
