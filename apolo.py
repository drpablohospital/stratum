#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import time
import math
import copy
import random
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# =========================================================
# Import del backtester
# =========================================================
run_backtest = None
for module_name in ("stratum_bt", "argentum_bt"):
    try:
        mod = __import__(module_name, fromlist=["run_backtest"])
        run_backtest = getattr(mod, "run_backtest")
        break
    except Exception:
        continue

if run_backtest is None:
    print("[APOLO][ERROR] No se encuentra stratum_bt.py ni argentum_bt.py")
    sys.exit(1)

# =========================================================
# Surrogate ML opcional
# =========================================================
SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestRegressor = None

# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = BASE_DIR / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)

DATA_CACHE_DIR = BASE_DIR / "data_cache"
DATA_CACHE_DIR.mkdir(exist_ok=True)

BEST_PARAMS_FILE = BASE_DIR / "best_params.json"
RESULTS_FILE = BASE_DIR / "optimizer_results.csv"
RESULTS_JSONL = BASE_DIR / "optimizer_results.jsonl"
PROGRESS_FILE = BASE_DIR / "optimizer_progress.txt"

BINANCE_FAPI_BASE = "https://fapi.binance.com"

# =========================================================
# Config base compatible con trader / GUI actual
# =========================================================
DEFAULT_CONFIG: dict[str, Any] = {
    "symbol": "BTC/USDT:USDT",
    "market_symbol": "BTCUSDT",
    "days": 60,
    "poll_seconds": 10,
    "status_heartbeat_seconds": 60,
    "tf_signal": "15m",
    "tf_confirm": "5m",
    "tf_entry": "5m",
    "tf_trend": "1h",
    "live_chart_tf": "15m",
    "mode": "feature_prob",
    "signal_engine": "feature_prob",
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

    # Hybrid mode parameters
    "hybrid": {
        "ml_weight": 0.6,
        "long_threshold": 0.58,
        "short_threshold": 0.58,
        "use_ml_model": True,
        "ml_model_path": "btc_ml_model.pkl",
    },
}

SEARCH_SPACE_FEATURE = {
    ("tf_signal",): ["5m", "15m", "30m", "1h"],
    ("tf_confirm",): ["1m", "3m", "5m", "15m"],
    ("tf_entry",): ["1m", "3m", "5m", "15m"],
    ("tf_trend",): ["15m", "1h", "4h"],
    ("live_chart_tf",): ["5m", "15m", "1h"],
    ("use_1h_trend_filter",): [True, False],

    ("feature_prob", "ret_1_weight"): (0.02, 0.25, "float"),
    ("feature_prob", "ret_3_weight"): (0.04, 0.30, "float"),
    ("feature_prob", "ret_12_weight"): (0.04, 0.35, "float"),
    ("feature_prob", "range_pos_weight"): (0.02, 0.25, "float"),
    ("feature_prob", "vol_ratio_weight"): (0.02, 0.25, "float"),
    ("feature_prob", "realized_vol_weight"): (0.02, 0.25, "float"),
    ("feature_prob", "slope_weight"): (0.02, 0.30, "float"),
    ("feature_prob", "trend_weight"): (0.04, 0.35, "float"),
    ("feature_prob", "breakout_weight"): (0.02, 0.25, "float"),
    ("feature_prob", "long_threshold"): (0.53, 0.72, "float"),
    ("feature_prob", "short_threshold"): (0.53, 0.72, "float"),
    ("feature_prob", "persistence_bars"): (1, 4, "int"),

    ("risk_engine", "sl_vol_mult"): (0.8, 2.5, "float"),
    ("risk_engine", "tp_vol_mult"): (1.4, 5.5, "float"),
    ("risk_engine", "breakeven_R"): (0.8, 2.5, "float"),
    ("risk_engine", "trail_start_R"): (1.0, 4.0, "float"),
    ("risk_engine", "trail_vol_mult"): (0.5, 2.5, "float"),
    ("risk_engine", "risk_per_trade_pct"): (0.10, 1.00, "float"),
    ("risk_engine", "max_hold_minutes"): (60, 720, "int"),
    ("risk_engine", "cooldown_minutes"): (15, 360, "int"),
}

SEARCH_SPACE_LEGACY = {
    ("tf_signal",): ["5m", "15m", "30m"],
    ("tf_confirm",): ["1m", "3m", "5m", "15m"],
    ("tf_entry",): ["1m", "3m", "5m", "15m"],
    ("tf_trend",): ["15m", "1h", "4h"],
    ("live_chart_tf",): ["5m", "15m", "1h"],
    ("use_1h_trend_filter",): [True, False],

    ("legacy", "ema_fast_trend"): (20, 80, "int"),
    ("legacy", "ema_slow_trend"): (100, 300, "int"),
    ("legacy", "ema_fast_signal"): (10, 80, "int"),
    ("legacy", "ema_slow_signal"): (40, 250, "int"),
    ("legacy", "donchian_window_signal"): (20, 150, "int"),
    ("legacy", "rsi_len_signal"): (7, 30, "int"),
    ("legacy", "rsi_long"): (50, 75, "float"),
    ("legacy", "rsi_short"): (25, 50, "float"),
    ("legacy", "atr_len_signal"): (7, 30, "int"),
    ("legacy", "atr_ma_window_signal"): (20, 150, "int"),
    ("legacy", "atr_min_mult"): (0.6, 2.0, "float"),
    ("legacy", "breakout_buffer_atr"): (0.15, 1.0, "float"),
    ("legacy", "sl_atr_mult"): (0.8, 3.5, "float"),
    ("legacy", "tp_atr_mult"): (1.5, 6.0, "float"),
    ("legacy", "breakeven_R"): (0.5, 3.0, "float"),
    ("legacy", "trail_start_R"): (0.8, 4.5, "float"),
    ("legacy", "trail_atr_mult"): (0.4, 2.5, "float"),

    ("risk_engine", "risk_per_trade_pct"): (0.10, 1.00, "float"),
    ("risk_engine", "max_hold_minutes"): (30, 720, "int"),
    ("risk_engine", "cooldown_minutes"): (5, 360, "int"),
}

# New search space for hybrid mode (combines heuristic and ML)
SEARCH_SPACE_HYBRID = {
    ("tf_signal",): ["5m", "15m", "30m", "1h"],
    ("tf_confirm",): ["1m", "3m", "5m", "15m"],
    ("tf_entry",): ["1m", "3m", "5m", "15m"],
    ("tf_trend",): ["15m", "1h", "4h"],
    ("live_chart_tf",): ["5m", "15m", "1h"],
    ("use_1h_trend_filter",): [True, False],

    ("hybrid", "ml_weight"): (0.0, 1.0, "float"),
    ("hybrid", "long_threshold"): (0.53, 0.72, "float"),
    ("hybrid", "short_threshold"): (0.53, 0.72, "float"),
    ("hybrid", "use_ml_model"): [True],  # always true for this mode, but keep for consistency

    # Heuristic features (feature_prob) - we also optimize them for hybrid
    ("feature_prob", "ret_1_weight"): (0.02, 0.25, "float"),
    ("feature_prob", "ret_3_weight"): (0.04, 0.30, "float"),
    ("feature_prob", "ret_12_weight"): (0.04, 0.35, "float"),
    ("feature_prob", "range_pos_weight"): (0.02, 0.25, "float"),
    ("feature_prob", "vol_ratio_weight"): (0.02, 0.25, "float"),
    ("feature_prob", "realized_vol_weight"): (0.02, 0.25, "float"),
    ("feature_prob", "slope_weight"): (0.02, 0.30, "float"),
    ("feature_prob", "trend_weight"): (0.04, 0.35, "float"),
    ("feature_prob", "breakout_weight"): (0.02, 0.25, "float"),

    ("risk_engine", "sl_vol_mult"): (0.8, 2.5, "float"),
    ("risk_engine", "tp_vol_mult"): (1.4, 5.5, "float"),
    ("risk_engine", "breakeven_R"): (0.8, 2.5, "float"),
    ("risk_engine", "trail_start_R"): (1.0, 4.0, "float"),
    ("risk_engine", "trail_vol_mult"): (0.5, 2.5, "float"),
    ("risk_engine", "risk_per_trade_pct"): (0.10, 1.00, "float"),
    ("risk_engine", "max_hold_minutes"): (60, 720, "int"),
    ("risk_engine", "cooldown_minutes"): (15, 360, "int"),
}

TIMEFRAME_ORDER = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}

# =========================================================
# Helpers
# =========================================================
def deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get_path_value(d: dict[str, Any], path: tuple[str, ...], default=None):
    obj = d
    for k in path:
        if not isinstance(obj, dict) or k not in obj:
            return default
        obj = obj[k]
    return obj


def set_path_value(d: dict[str, Any], path: tuple[str, ...], value: Any):
    obj = d
    for k in path[:-1]:
        if k not in obj or not isinstance(obj[k], dict):
            obj[k] = {}
        obj = obj[k]
    obj[path[-1]] = value


def flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in cfg.items():
        nk = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_config(v, nk + "__"))
        else:
            out[nk] = v
    return out


def build_runtime_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["signal_engine"] = out.get("mode", "feature_prob")

    legacy = out.get("legacy", {})
    risk = out.get("risk_engine", {})
    execution = out.get("execution", {})

    out["ema_fast_trend"] = legacy.get("ema_fast_trend")
    out["ema_slow_trend"] = legacy.get("ema_slow_trend")
    out["ema_fast_signal"] = legacy.get("ema_fast_signal")
    out["ema_slow_signal"] = legacy.get("ema_slow_signal")
    out["donchian_window_signal"] = legacy.get("donchian_window_signal")
    out["rsi_len_signal"] = legacy.get("rsi_len_signal")
    out["rsi_long"] = legacy.get("rsi_long")
    out["rsi_short"] = legacy.get("rsi_short")
    out["atr_len_signal"] = legacy.get("atr_len_signal")
    out["atr_ma_window_signal"] = legacy.get("atr_ma_window_signal")
    out["atr_min_mult"] = legacy.get("atr_min_mult")
    out["breakout_buffer_atr"] = legacy.get("breakout_buffer_atr")
    out["sl_atr_mult"] = legacy.get("sl_atr_mult")
    out["tp_atr_mult"] = legacy.get("tp_atr_mult")
    out["trail_atr_mult"] = legacy.get("trail_atr_mult")

    out["breakeven_R"] = risk.get("breakeven_R")
    out["trail_start_R"] = risk.get("trail_start_R")
    out["max_hold_minutes"] = risk.get("max_hold_minutes")
    out["cooldown_minutes"] = risk.get("cooldown_minutes")
    out["risk_per_trade_pct"] = risk.get("risk_per_trade_pct")

    out["fees_enabled"] = execution.get("fees_enabled")
    out["fee_rate_roundtrip"] = execution.get("fee_rate_roundtrip")
    out["slippage_roundtrip_pct"] = execution.get("slippage_roundtrip_pct")
    out["label_exits"] = execution.get("label_exits")
    return out


# =========================================================
# Dataset cache auto-download
# =========================================================
def fetch_klines_binance(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1500) -> pd.DataFrame:
    rows = []
    cur = start_ms
    headers = {"User-Agent": "STRATUM-APOLO/1.0"}

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit,
        }
        r = requests.get(f"{BINANCE_FAPI_BASE}/fapi/v1/klines", params=params, timeout=30, headers=headers)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        last_open = int(data[-1][0])
        if len(data) < limit:
            break
        cur = last_open + 1
        time.sleep(0.08)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "close_time"])

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["timestamp", "open", "high", "low", "close", "volume", "close_time"]]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def dataset_cache_path(symbol: str, interval: str, days: int) -> Path:
    return DATA_CACHE_DIR / f"{symbol.lower()}_{interval}_{days}d.csv"


def ensure_dataset(symbol: str = "BTCUSDT", interval: str = "5m", days: int = 180, max_age_hours: int = 6) -> Path:
    path = dataset_cache_path(symbol, interval, days)

    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        if age_hours <= max_age_hours:
            print(f"[APOLO] Dataset cache reutilizado: {path.name}")
            return path

    print(f"[APOLO] Descargando dataset {symbol} {interval} ({days}d)...")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(days * 24 * 3600 * 1000)

    df = fetch_klines_binance(symbol=symbol, interval=interval, start_ms=start_ms, end_ms=end_ms)
    if df.empty:
        raise RuntimeError(f"No se pudo descargar dataset {symbol} {interval}")

    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[APOLO] Dataset guardado en {path.name} ({len(df)} filas)")
    return path


def warmup_datasets():
    # Pre-carga suave para evitar errores de scripts viejos que esperen CSV local
    targets = [
        ("BTCUSDT", "5m", 180),
        ("BTCUSDT", "15m", 240),
        ("BTCUSDT", "1h", 365),
    ]
    for sym, tf, days in targets:
        try:
            ensure_dataset(sym, tf, days)
        except Exception as e:
            print(f"[APOLO][WARN] No se pudo cachear {sym} {tf}: {e}")


# =========================================================
# Aleatorización / mutación
# =========================================================
def sanitize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(cfg)

    # Sanitize feature_prob thresholds
    fp = c.get("feature_prob", {})
    if "long_threshold" in fp:
        fp["long_threshold"] = float(np.clip(fp["long_threshold"], 0.50, 0.90))
    if "short_threshold" in fp:
        fp["short_threshold"] = float(np.clip(fp["short_threshold"], 0.50, 0.90))
    if "persistence_bars" in fp:
        fp["persistence_bars"] = int(np.clip(int(round(fp["persistence_bars"])), 1, 6))

    # Sanitize hybrid thresholds
    hy = c.get("hybrid", {})
    if "long_threshold" in hy:
        hy["long_threshold"] = float(np.clip(hy["long_threshold"], 0.50, 0.90))
    if "short_threshold" in hy:
        hy["short_threshold"] = float(np.clip(hy["short_threshold"], 0.50, 0.90))
    if "ml_weight" in hy:
        hy["ml_weight"] = float(np.clip(hy["ml_weight"], 0.0, 1.0))

    sig = c.get("tf_signal", "15m")
    confirm = c.get("tf_confirm", "5m")
    entry = c.get("tf_entry", "5m")
    trend = c.get("tf_trend", "1h")

    if TIMEFRAME_ORDER.get(confirm, 5) > TIMEFRAME_ORDER.get(sig, 15):
        c["tf_confirm"] = sig
    if TIMEFRAME_ORDER.get(entry, 5) > TIMEFRAME_ORDER.get(sig, 15):
        c["tf_entry"] = c["tf_confirm"]
    if TIMEFRAME_ORDER.get(trend, 60) < TIMEFRAME_ORDER.get(sig, 15):
        c["tf_trend"] = "1h"

    lg = c.get("legacy", {})
    if lg:
        if lg.get("ema_fast_signal", 50) >= lg.get("ema_slow_signal", 200):
            a = int(lg["ema_fast_signal"])
            b = int(lg["ema_slow_signal"])
            lg["ema_fast_signal"], lg["ema_slow_signal"] = min(a, b - 1), max(a + 1, b)
        if lg.get("ema_fast_trend", 50) >= lg.get("ema_slow_trend", 200):
            a = int(lg["ema_fast_trend"])
            b = int(lg["ema_slow_trend"])
            lg["ema_fast_trend"], lg["ema_slow_trend"] = min(a, b - 1), max(a + 1, b)
        if lg.get("rsi_long", 58) <= lg.get("rsi_short", 42):
            hi = max(float(lg["rsi_long"]), float(lg["rsi_short"]) + 1.0)
            lo = min(float(lg["rsi_short"]), float(lg["rsi_long"]) - 1.0)
            lg["rsi_long"] = hi
            lg["rsi_short"] = lo

    return c


def random_value(spec):
    if isinstance(spec, list):
        return random.choice(spec)
    lo, hi, typ = spec
    if typ == "int":
        return random.randint(int(lo), int(hi))
    return random.uniform(float(lo), float(hi))


def random_candidate(mode: str | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if mode is None:
        # 30% hybrid, 35% feature_prob, 35% legacy
        r = random.random()
        if r < 0.3:
            chosen_mode = "hybrid"
        elif r < 0.65:
            chosen_mode = "feature_prob"
        else:
            chosen_mode = "legacy_classic"
    else:
        chosen_mode = mode
    cfg["mode"] = chosen_mode
    cfg["signal_engine"] = chosen_mode

    if chosen_mode == "hybrid":
        search_space = SEARCH_SPACE_HYBRID
    elif chosen_mode == "feature_prob":
        search_space = SEARCH_SPACE_FEATURE
    else:
        search_space = SEARCH_SPACE_LEGACY

    for path, spec in search_space.items():
        set_path_value(cfg, path, random_value(spec))

    # Ensure hybrid mode has ML model path
    if chosen_mode == "hybrid":
        cfg["hybrid"]["ml_model_path"] = str(BASE_DIR / "btc_ml_model.pkl")
        cfg["hybrid"]["use_ml_model"] = True

    return sanitize_config(cfg)


def mutate_value(base_value, spec, strength: float = 0.20):
    if isinstance(spec, list):
        if random.random() < 0.70:
            return base_value
        return random.choice(spec)

    lo, hi, typ = spec
    if typ == "int":
        span = max(1, int(round((hi - lo) * strength)))
        v = int(base_value) + random.randint(-span, span)
        return int(np.clip(v, lo, hi))
    else:
        span = (hi - lo) * strength
        v = float(base_value) + random.uniform(-span, span)
        return float(np.clip(v, lo, hi))


def mutate_candidate(cfg: dict[str, Any], strength: float = 0.20, flip_mode_prob: float = 0.05) -> dict[str, Any]:
    c = copy.deepcopy(cfg)

    if random.random() < flip_mode_prob:
        # Flip mode between the three
        modes = ["feature_prob", "legacy_classic", "hybrid"]
        current = c.get("mode")
        new_mode = random.choice([m for m in modes if m != current])
        c["mode"] = new_mode
        c["signal_engine"] = new_mode

    mode = c.get("mode", "feature_prob")
    if mode == "hybrid":
        search_space = SEARCH_SPACE_HYBRID
    elif mode == "feature_prob":
        search_space = SEARCH_SPACE_FEATURE
    else:
        search_space = SEARCH_SPACE_LEGACY

    for path, spec in search_space.items():
        if random.random() < 0.65:
            current = get_path_value(c, path)
            set_path_value(c, path, mutate_value(current, spec, strength))

    # Ensure hybrid mode has ML model path
    if c.get("mode") == "hybrid":
        if "hybrid" not in c:
            c["hybrid"] = {}
        c["hybrid"]["ml_model_path"] = str(BASE_DIR / "btc_ml_model.pkl")
        c["hybrid"]["use_ml_model"] = True

    return sanitize_config(c)


# =========================================================
# Vectorización para surrogate model
# =========================================================
SURROGATE_FEATURES = [
    ("mode",),
    ("tf_signal",),
    ("tf_confirm",),
    ("tf_entry",),
    ("tf_trend",),
    ("live_chart_tf",),
    ("use_1h_trend_filter",),

    ("feature_prob", "ret_1_weight"),
    ("feature_prob", "ret_3_weight"),
    ("feature_prob", "ret_12_weight"),
    ("feature_prob", "range_pos_weight"),
    ("feature_prob", "vol_ratio_weight"),
    ("feature_prob", "realized_vol_weight"),
    ("feature_prob", "slope_weight"),
    ("feature_prob", "trend_weight"),
    ("feature_prob", "breakout_weight"),
    ("feature_prob", "long_threshold"),
    ("feature_prob", "short_threshold"),
    ("feature_prob", "persistence_bars"),

    ("risk_engine", "sl_vol_mult"),
    ("risk_engine", "tp_vol_mult"),
    ("risk_engine", "breakeven_R"),
    ("risk_engine", "trail_start_R"),
    ("risk_engine", "trail_vol_mult"),
    ("risk_engine", "risk_per_trade_pct"),
    ("risk_engine", "max_hold_minutes"),
    ("risk_engine", "cooldown_minutes"),

    ("legacy", "ema_fast_trend"),
    ("legacy", "ema_slow_trend"),
    ("legacy", "ema_fast_signal"),
    ("legacy", "ema_slow_signal"),
    ("legacy", "donchian_window_signal"),
    ("legacy", "rsi_len_signal"),
    ("legacy", "rsi_long"),
    ("legacy", "rsi_short"),
    ("legacy", "atr_len_signal"),
    ("legacy", "atr_ma_window_signal"),
    ("legacy", "atr_min_mult"),
    ("legacy", "breakout_buffer_atr"),
    ("legacy", "sl_atr_mult"),
    ("legacy", "tp_atr_mult"),
    ("legacy", "breakeven_R"),
    ("legacy", "trail_start_R"),
    ("legacy", "trail_atr_mult"),

    # Hybrid features
    ("hybrid", "ml_weight"),
    ("hybrid", "long_threshold"),
    ("hybrid", "short_threshold"),
]


def encode_tf(v: str) -> float:
    return float(TIMEFRAME_ORDER.get(v, 0))


def candidate_to_vector(cfg: dict[str, Any]) -> list[float]:
    vec = []
    for path in SURROGATE_FEATURES:
        v = get_path_value(cfg, path)
        if path == ("mode",):
            # Encode mode: 0=feature_prob, 1=legacy, 2=hybrid
            if v == "feature_prob":
                vec.append(0.0)
            elif v == "legacy_classic":
                vec.append(1.0)
            else:
                vec.append(2.0)
        elif isinstance(v, bool):
            vec.append(1.0 if v else 0.0)
        elif isinstance(v, str):
            vec.append(encode_tf(v))
        elif v is None:
            vec.append(0.0)
        else:
            vec.append(float(v))
    return vec


# =========================================================
# Scoring
# =========================================================
def compute_score(trades_df: pd.DataFrame, stats: dict[str, Any], cfg: dict) -> float:
    if trades_df is None or trades_df.empty:
        return -1e9

    gains = trades_df[trades_df["pnl_equity_pct"] > 0]["pnl_equity_pct"].sum()
    losses = abs(trades_df[trades_df["pnl_equity_pct"] < 0]["pnl_equity_pct"].sum())
    profit_factor = gains / losses if losses > 0 else 10.0

    total_pnl = float(stats.get("total_pnl", 0.0))
    winrate = float(stats.get("winrate", 0.0))
    max_dd = abs(float(stats.get("max_dd", 0.0)))
    trades = int(stats.get("trades", 0))
    tpd = float(stats.get("tpd", 0.0))
    proj_month = float(stats.get("proj_month", 0.0))

    trade_penalty = 0.0
    if trades < 15:
        trade_penalty = 5.0 + (15 - trades) * 0.4

    overtrade_penalty = max(0.0, tpd - 8.0) * 1.25
    dd_penalty = max_dd * 0.7

    score = (
        total_pnl * 1.0
        + min(profit_factor, 6.0) * 6.0
        + winrate * 0.12
        + proj_month * 0.12
        - dd_penalty
        - trade_penalty
        - overtrade_penalty
    )

    # Small bonus for using ML model
    if cfg.get("mode") == "hybrid":
        score += 0.5

    return float(score)


# =========================================================
# Evaluación
# =========================================================
def evaluate_candidate(candidate_cfg: dict[str, Any]) -> dict[str, Any] | None:
    cfg = build_runtime_config(candidate_cfg)
    # Ensure the model path is set for hybrid mode
    if cfg.get("mode") == "hybrid":
        cfg["ml_model_path"] = candidate_cfg.get("hybrid", {}).get("ml_model_path", str(BASE_DIR / "btc_ml_model.pkl"))
    try:
        trades_df, stats = run_backtest(cfg)
        if trades_df is None or trades_df.empty:
            return None

        gains = trades_df[trades_df["pnl_equity_pct"] > 0]["pnl_equity_pct"].sum()
        losses = abs(trades_df[trades_df["pnl_equity_pct"] < 0]["pnl_equity_pct"].sum())
        profit_factor = gains / losses if losses > 0 else 10.0

        total_pnl = float(stats.get("total_pnl", 0.0))
        winrate = float(stats.get("winrate", 0.0))
        max_dd = float(stats.get("max_dd", 0.0))
        trades = int(stats.get("trades", 0))
        tpd = float(stats.get("tpd", 0.0))
        proj_month = float(stats.get("proj_month", 0.0))
        exit_counts = stats.get("exit_counts", {})

        score = compute_score(trades_df, stats, candidate_cfg)

        return {
            "config": candidate_cfg,
            "mode": candidate_cfg.get("mode"),
            "trades": trades,
            "winrate": winrate,
            "total_pnl": total_pnl,
            "profit_factor": float(profit_factor),
            "score": score,
            "max_dd": max_dd,
            "tpd": tpd,
            "proj_month": proj_month,
            "exit_counts": exit_counts,
        }
    except Exception as e:
        print(f"[APOLO][ERROR] Evaluando config: {e}")
        return None


# =========================================================
# Persistencia / GUI bridge
# =========================================================
def write_progress(text: str):
    PROGRESS_FILE.write_text(text, encoding="utf-8")


def save_best(best_row: dict[str, Any]):
    payload = copy.deepcopy(best_row["config"])
    payload["_meta"] = {
        "score": best_row["score"],
        "trades": best_row["trades"],
        "winrate": best_row["winrate"],
        "total_pnl": best_row["total_pnl"],
        "profit_factor": best_row["profit_factor"],
        "max_dd": best_row["max_dd"],
        "tpd": best_row["tpd"],
        "proj_month": best_row["proj_month"],
        "mode": best_row["mode"],
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def result_to_flat_row(r: dict[str, Any]) -> dict[str, Any]:
    row = {
        "mode": r["mode"],
        "trades": r["trades"],
        "winrate": r["winrate"],
        "total_pnl": r["total_pnl"],
        "profit_factor": r["profit_factor"],
        "score": r["score"],
        "max_dd": r["max_dd"],
        "tpd": r["tpd"],
        "proj_month": r["proj_month"],
    }
    row.update(flatten_config(r["config"], prefix="cfg__"))
    return row


def append_jsonl(path: Path, obj: dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_history_jsonl(path: Path) -> list[dict[str, Any]]:
    history = []
    if not path.exists():
        return history
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                history.append(json.loads(line))
            except Exception:
                continue
    return history


def save_results_csv(history: list[dict[str, Any]]):
    if not history:
        return
    rows = [result_to_flat_row(r) for r in history]
    pd.DataFrame(rows).to_csv(RESULTS_FILE, index=False)


# =========================================================
# Surrogate
# =========================================================
def fit_surrogate(history: list[dict[str, Any]]):
    if not SKLEARN_AVAILABLE or len(history) < 20:
        return None

    X = np.array([candidate_to_vector(r["config"]) for r in history], dtype=float)
    y = np.array([float(r["score"]) for r in history], dtype=float)

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X, y)
    return model


def propose_batch(history: list[dict[str, Any]], batch_size: int, explore_ratio: float = 0.45) -> list[dict[str, Any]]:
    if not history:
        return [random_candidate() for _ in range(batch_size)]

    best = max(history, key=lambda r: r["score"])
    proposals: list[dict[str, Any]] = []

    n_mutate = max(1, int(batch_size * 0.35))
    for _ in range(n_mutate):
        proposals.append(mutate_candidate(best["config"], strength=random.uniform(0.08, 0.22), flip_mode_prob=0.02))

    n_explore = max(1, int(batch_size * explore_ratio))
    for _ in range(n_explore):
        proposals.append(random_candidate())

    remaining = batch_size - len(proposals)
    if remaining > 0:
        model = fit_surrogate(history)
        if model is not None:
            pool = [random_candidate() for _ in range(max(remaining * 15, 80))]
            X = np.array([candidate_to_vector(c) for c in pool], dtype=float)
            preds = model.predict(X)
            ranked = sorted(zip(pool, preds), key=lambda z: z[1], reverse=True)
            for cand, _ in ranked[:remaining]:
                proposals.append(cand)
        else:
            for _ in range(remaining):
                proposals.append(mutate_candidate(best["config"], strength=random.uniform(0.12, 0.30), flip_mode_prob=0.04))

    return proposals[:batch_size]


# =========================================================
# Worker
# =========================================================
def worker(candidate_cfg: dict[str, Any]):
    return evaluate_candidate(candidate_cfg)


# =========================================================
# Main optimizer
# =========================================================
def run_optimizer(
    num_trials: int = 300,
    n_jobs: int = 4,
    resume: bool = False,
    batch_size: int = 8,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    warmup_datasets()

    history: list[dict[str, Any]] = []
    best_row = None

    if resume and RESULTS_JSONL.exists():
        history = load_history_jsonl(RESULTS_JSONL)
        if history:
            best_row = max(history, key=lambda r: r["score"])
            print(f"[APOLO] Reanudando desde {len(history)} resultados previos")
            print(f"[APOLO] Mejor score previo: {best_row['score']:.4f}")

    done = len(history)
    if best_row is None and history:
        best_row = max(history, key=lambda r: r["score"])

    while done < num_trials:
        remaining = num_trials - done
        current_batch_size = min(batch_size, remaining)
        proposals = propose_batch(history, current_batch_size)

        print(f"[APOLO] Lote {done + 1}-{done + current_batch_size}/{num_trials} | batch={current_batch_size} | mode_bias={best_row['mode'] if best_row else 'mixed'}")

        if n_jobs > 1 and current_batch_size > 1:
            with mp.Pool(processes=min(n_jobs, current_batch_size)) as pool:
                results = pool.map(worker, proposals)
        else:
            results = [worker(c) for c in proposals]

        for r in results:
            done += 1
            if r is None:
                progress_text = f"{done}/{num_trials} evaluados | Mejor score: {best_row['score']:.4f}" if best_row else f"{done}/{num_trials} evaluados | Sin mejor aún"
                write_progress(progress_text)
                print(f"[APOLO] {done}/{num_trials} -> resultado vacío")
                continue

            history.append(r)
            append_jsonl(RESULTS_JSONL, r)

            is_new_best = best_row is None or r["score"] > best_row["score"]
            if is_new_best:
                best_row = r
                save_best(best_row)
                print(
                    f"[APOLO][BEST] score={best_row['score']:.4f} | "
                    f"mode={best_row['mode']} | pnl={best_row['total_pnl']:.2f} | "
                    f"pf={best_row['profit_factor']:.2f} | dd={best_row['max_dd']:.2f} | "
                    f"trades={best_row['trades']}"
                )

            progress_text = (
                f"{done}/{num_trials} probados | Mejor score: {best_row['score']:.4f}\n"
                f"Best mode: {best_row['mode']} | PnL: {best_row['total_pnl']:.2f} | PF: {best_row['profit_factor']:.2f} | DD: {best_row['max_dd']:.2f}\n"
                f"Último score: {r['score']:.4f} | trades={r['trades']} | wr={r['winrate']:.2f}"
            )
            write_progress(progress_text)

            print(
                f"[APOLO] {done}/{num_trials} | "
                f"mode={r['mode']} | score={r['score']:.4f} | "
                f"pnl={r['total_pnl']:.2f} | pf={r['profit_factor']:.2f} | "
                f"dd={r['max_dd']:.2f} | trades={r['trades']}"
            )

        save_results_csv(history)

    if best_row is not None:
        save_best(best_row)
        save_results_csv(history)
        print("[APOLO] Optimización completada.")
        print(f"[APOLO] Mejor score: {best_row['score']:.4f}")
        print(f"[APOLO] Mejor modo: {best_row['mode']}")
        print("[APOLO] Mejores parámetros guardados en best_params.json")
        write_progress(
            f"FINALIZADO | {len(history)}/{num_trials} evaluados | Mejor score: {best_row['score']:.4f}\n"
            f"Mode: {best_row['mode']} | PnL: {best_row['total_pnl']:.2f} | PF: {best_row['profit_factor']:.2f} | DD: {best_row['max_dd']:.2f}"
        )
    else:
        print("[APOLO] Optimización completada sin resultados válidos.")
        write_progress("FINALIZADO | Sin resultados válidos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300, help="Número total de evaluaciones")
    parser.add_argument("--jobs", type=int, default=4, help="Procesos paralelos")
    parser.add_argument("--resume", action="store_true", help="Reanudar desde resultados existentes")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamaño de lote por iteración")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    print(f"[APOLO] Inicio | sklearn={'ON' if SKLEARN_AVAILABLE else 'OFF'} | jobs={args.jobs} | trials={args.trials} | batch={args.batch_size}")
    run_optimizer(
        num_trials=args.trials,
        n_jobs=args.jobs,
        resume=args.resume,
        batch_size=args.batch_size,
        seed=args.seed,
    )
