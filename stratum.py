"""
STRATUM TRADER BOT (vNext) - BYBIT VERSION
Full position management + ML / Hybrid mode using pre-trained XGBoost model.
Configurable via dashboard (config.json / params_bt.json)
"""

from __future__ import annotations

import sys
import os
import json
import time
import pickle
import copy
import traceback
from pathlib import Path
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ===============================
# UTF-8 FIX (WINDOWS)
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

ML_MODEL_FILE = BASE_DIR / "btc_ml_model.pkl"
ML_SCALER_FILE = BASE_DIR / "scaler.pkl"
ML_FEATURES_FILE = BASE_DIR / "feature_columns.json"
ML_MAPPING_FILE = BASE_DIR / "label_mapping.json"
ML_METADATA_FILE = BASE_DIR / "ml_metadata.json"


# ===============================
# ORACLE / TELEGRAM
# ===============================
SEND_TELEGRAM = None
FORMAT_SIGNAL = None
FORMAT_ENTRY = None
FORMAT_CLOSE = None
FORMAT_ERROR = None
FORMAT_STARTUP = None

try:
    from oracle import (
        send_telegram as SEND_TELEGRAM,
        format_signal as FORMAT_SIGNAL,
        format_entry as FORMAT_ENTRY,
        format_close as FORMAT_CLOSE,
        format_error as FORMAT_ERROR,
        format_startup as FORMAT_STARTUP,
    )
    print("[ORACLE] Telegram/Oracle importado correctamente", flush=True)
except Exception as e:
    print(f"[ORACLE] Oracle no disponible: {e}", flush=True)
    SEND_TELEGRAM = None
    FORMAT_SIGNAL = None
    FORMAT_ENTRY = None
    FORMAT_CLOSE = None
    FORMAT_ERROR = None
    FORMAT_STARTUP = None


# ===============================
# UTILITIES
# ===============================
def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_json(path, default=None):
    path = Path(path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def log(msg: str) -> None:
    print(msg, flush=True)


def notify(msg: str) -> None:
    log(msg)
    if SEND_TELEGRAM:
        try:
            SEND_TELEGRAM(msg)
        except Exception as e:
            log(f"[WARN] telegram failed: {e}")


def fmt(x, n=2):
    try:
        return round(float(x), n)
    except Exception:
        return x


def path_exists_str(path: Path) -> str:
    return "OK" if Path(path).exists() else "MISSING"


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def load_ml_metadata() -> dict:
    if ML_METADATA_FILE.exists():
        try:
            return json.loads(ML_METADATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def expected_trade_pnl_pct(risk_pct: float, tp_mult: float, sl_mult: float, fee_rt: float = 0.0, slip_rt: float = 0.0) -> float:
    """
    Estimación simple del PnL % sobre equity si el trade alcanza TP completo,
    asumiendo sizing por riesgo y restando fees/slippage como aproximación.
    """
    if sl_mult <= 0:
        return 0.0
    gross = risk_pct * (tp_mult / sl_mult)
    # aproximación sencilla de costes sobre equity
    cost_adj = risk_pct * (fee_rt + slip_rt) * 2.0
    return gross - cost_adj


def ensure_trades_csv() -> None:
    if TRADES_FILE.exists():
        return
    df = pd.DataFrame(columns=[
        "timestamp",
        "symbol",
        "mode",
        "side",
        "price",
        "size",
        "sl",
        "tp",
        "env",
    ])
    df.to_csv(TRADES_FILE, index=False)


def append_trade_row(row: dict) -> None:
    ensure_trades_csv()
    row_df = pd.DataFrame([row])
    if TRADES_FILE.exists() and TRADES_FILE.stat().st_size > 0:
        row_df.to_csv(TRADES_FILE, mode="a", header=False, index=False)
    else:
        row_df.to_csv(TRADES_FILE, index=False)


def build_plain_startup_message(cfg: dict, equity: float, model_status: str, config_source: str) -> str:
    risk = cfg.get("risk_engine", {})
    execution = cfg.get("execution", {})
    mode = cfg.get("mode", "feature_prob")
    tp_mult = safe_float(risk.get("tp_vol_mult", 2.8))
    sl_mult = safe_float(risk.get("sl_vol_mult", 1.2))
    risk_pct = safe_float(risk.get("risk_per_trade_pct", 0.5))
    fee_rt = safe_float(execution.get("fee_rate_roundtrip", 0.001))
    slip_rt = safe_float(execution.get("slippage_roundtrip_pct", 0.0001))
    exp_pnl = expected_trade_pnl_pct(risk_pct, tp_mult, sl_mult, fee_rt, slip_rt)
    meta = load_ml_metadata()
    last_train = meta.get("last_train", "desconocida")
    acc = meta.get("test_accuracy", "N/A")

    return (
        "🚀 STRATUM iniciado\n"
        f"• Env: LIVE\n"
        f"• Exchange: BYBIT\n"
        f"• Symbol: {cfg.get('symbol')}\n"
        f"• Mode: {mode}\n"
        f"• tf_signal/tf_confirm/tf_trend: {cfg.get('tf_signal')} / {cfg.get('tf_confirm')} / {cfg.get('tf_trend')}\n"
        f"• Source config: {config_source}\n"
        f"• Equity inicial: {equity:.2f} USDT\n"
        f"• Modelo ML: {model_status}\n"
        f"• Último entrenamiento: {last_train}\n"
        f"• Accuracy metadata: {acc}\n"
        f"• Riesgo por trade: {risk_pct:.3f}%\n"
        f"• SLxATR / TPxATR: {sl_mult:.3f} / {tp_mult:.3f}\n"
        f"• PnL esperado aprox al TP: {exp_pnl:.3f}% equity\n"
        f"• Trades CSV: {TRADES_FILE.name}\n"
    )


def build_plain_entry_message(symbol: str, side: str, price: float, qty: float, sl: float, tp: float, rr: float, mode: str) -> str:
    return (
        f"🟢 ENTRY {side.upper()} {symbol}\n"
        f"• Modo: {mode}\n"
        f"• Precio: {price:.2f}\n"
        f"• Size: {qty:.6f}\n"
        f"• SL: {sl:.2f}\n"
        f"• TP: {tp:.2f}\n"
        f"• R/R: {rr:.2f}"
    )


def build_plain_close_message(symbol: str, side: str, entry: float, mark: float, pnl: float, exit_reason: str) -> str:
    return (
        f"🔴 CLOSE {symbol}\n"
        f"• Side: {side}\n"
        f"• Entry: {entry:.2f}\n"
        f"• Exit/Mark: {mark:.2f}\n"
        f"• PnL: {pnl:.2f}%\n"
        f"• Reason: {exit_reason}"
    )


def build_plain_error_message(context: str, err: str) -> str:
    return f"⚠️ STRATUM ERROR\n• Contexto: {context}\n• Error: {err}"


def send_startup_message(cfg: dict, equity: float, model_status: str, config_source: str) -> None:
    msg = None
    if FORMAT_STARTUP:
        try:
            summary = build_plain_startup_message(cfg, equity, model_status, config_source)
            msg = FORMAT_STARTUP(summary)
        except Exception:
            msg = None
    if not msg:
        msg = build_plain_startup_message(cfg, equity, model_status, config_source)
    notify(msg)


def send_entry_message(symbol: str, side: str, price: float, qty: float, sl: float, tp: float, rr: float, mode: str, equity: float | None = None) -> None:
    msg = None
    if FORMAT_ENTRY:
        try:
            # compatibilidad con tu versión vieja
            msg = FORMAT_ENTRY(
                side=side.upper(),
                price=price,
                qty=qty,
                equity=equity if equity is not None else 0.0,
                trigger=price,
                sl_price=sl,
            )
        except Exception:
            msg = None
    if not msg:
        msg = build_plain_entry_message(symbol, side, price, qty, sl, tp, rr, mode)
    notify(msg)


def send_close_message(symbol: str, side: str, entry: float, mark: float, pnl: float, exit_reason: str, equity_before: float | None = None) -> None:
    msg = None
    if FORMAT_CLOSE:
        try:
            msg = FORMAT_CLOSE(side, entry, mark, pnl, exit_reason, equity_before if equity_before is not None else 0.0)
        except Exception:
            msg = None
    if not msg:
        msg = build_plain_close_message(symbol, side, entry, mark, pnl, exit_reason)
    notify(msg)


def send_error_message(context: str, err: str) -> None:
    msg = None
    if FORMAT_ERROR:
        try:
            msg = FORMAT_ERROR(err, context)
        except Exception:
            msg = None
    if not msg:
        msg = build_plain_error_message(context, err)
    notify(msg)


# ===============================
# CONFIG
# ===============================
def load_config() -> tuple[dict, str]:
    default_cfg = load_json(CONFIG_FILE, {})
    source_used = str(CONFIG_FILE.name)

    cfg = None
    if PARAMS_FILE.exists():
        try:
            cfg = load_json(PARAMS_FILE)
            source_used = str(PARAMS_FILE.name)
        except Exception as e:
            log(f"[ERROR] Failed to load {PARAMS_FILE}: {e}")

    if cfg is None:
        cfg = default_cfg
        source_used = str(CONFIG_FILE.name)

    if not isinstance(cfg, dict):
        log("[ERROR] Config inválida")
        return default_cfg, str(CONFIG_FILE.name)

    merged = copy.deepcopy(default_cfg)
    for key, value in cfg.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value

    return merged, source_used


# ===============================
# EXCHANGE (BYBIT / LIVE ONLY)
# ===============================
def get_exchange():
    load_dotenv()
    api_key = os.getenv("BYBIT_API_KEY")
    secret = os.getenv("BYBIT_SECRET")

    if not api_key or not secret:
        log("[ERROR] BYBIT_API_KEY or BYBIT_SECRET not set in environment")
        raise RuntimeError("Missing Bybit API credentials")

    exchange = ccxt.bybit({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,
            "recvWindow": 10000,
        },
    })
    return exchange


# ===============================
# DATA
# ===============================
def fetch_ohlc(ex, symbol, tf, limit=500):
    ohlc = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    df = pd.DataFrame(ohlc, columns=["time", "open", "high", "low", "close", "vol"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ===============================
# INDICATORS (for SL/TP & legacy)
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


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ===============================
# FEATURE ENGINE (heuristic)
# ===============================
def realized_vol(series, window=20):
    rets = np.log(series / series.shift(1))
    return rets.rolling(window).std() * np.sqrt(window)


def slope(series, window=8):
    out = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series) + 1):
        y = series.iloc[i - window:i].values
        x = np.arange(len(y))
        out.iloc[i - 1] = np.polyfit(x, y, 1)[0]
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
# LEGACY ENGINE
# ===============================
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
# ML MODEL & MULTI-TIMEFRAME FEATURES
# ===============================
ML_MODEL = None
SCALER = None
FEATURE_COLUMNS = None


def load_ml_artifacts():
    global ML_MODEL, SCALER, FEATURE_COLUMNS
    try:
        with open(ML_MODEL_FILE, "rb") as f:
            ML_MODEL = pickle.load(f)
        with open(ML_SCALER_FILE, "rb") as f:
            SCALER = pickle.load(f)
        with open(ML_FEATURES_FILE, "r", encoding="utf-8") as f:
            FEATURE_COLUMNS = json.load(f)

        if not isinstance(FEATURE_COLUMNS, list):
            raise TypeError(f"feature_columns.json debe contener una lista, llegó {type(FEATURE_COLUMNS)}")

        log(f"[ML] Modelo cargado correctamente | n_features={len(FEATURE_COLUMNS)}")
        return True
    except Exception as e:
        log(f"[ML] Error cargando modelo: {e}")
        return False


def compute_features_tf(df, tf_label):
    features = pd.DataFrame(index=df.index, dtype="float32")
    features["ret_1"] = df["close"].pct_change(1).astype("float32")
    features["ret_3"] = df["close"].pct_change(3).astype("float32")
    features["ret_12"] = df["close"].pct_change(12).astype("float32")

    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(14).mean()

    features["realized_vol"] = (atr_series / df["close"]).astype("float32")
    features["vol_ratio"] = (atr_series / atr_series.rolling(96).mean()).astype("float32")
    features["range_pos"] = ((df["high"] - df["low"]) / df["close"]).astype("float32")

    def slope_series(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    features["slope"] = df["close"].rolling(20).apply(slope_series, raw=True).astype("float32")
    features["slope"] = features["slope"] / df["close"].shift(1)
    features["ema_fast"] = df["close"].ewm(span=20).mean().astype("float32")
    features["ema_slow"] = df["close"].ewm(span=50).mean().astype("float32")
    features["trend"] = ((features["ema_fast"] - features["ema_slow"]) / df["close"]).astype("float32")

    donchian_high = df["high"].rolling(96).max()
    donchian_low = df["low"].rolling(96).min()
    features["breakout"] = np.where(
        df["close"] > donchian_high.shift(1), 1,
        np.where(df["close"] < donchian_low.shift(1), -1, 0)
    ).astype("int8")

    features = features.add_prefix(f"{tf_label}_")
    features.dropna(inplace=True)
    return features


def resample_to_base(features, base_freq):
    return features.resample(base_freq).ffill()


def get_multi_tf_features(ex, symbol, base_freq="1h"):
    timeframes = ["1m", "15m", "30m", "1h", "1d", "1w"]
    interval_data = {}

    for tf in timeframes:
        df = fetch_ohlc(ex, symbol, tf, limit=500)
        if not df.empty:
            interval_data[tf] = df

    combined = None
    for tf, df in interval_data.items():
        feats = compute_features_tf(df, tf)
        if feats.empty:
            continue
        feats = resample_to_base(feats, base_freq)
        if combined is None:
            combined = feats
        else:
            combined = combined.join(feats, how="inner")

    if combined is None or combined.empty:
        return None

    return combined.iloc[-1]


def ml_prediction(features_row):
    if ML_MODEL is None or SCALER is None or FEATURE_COLUMNS is None:
        return None
    try:
        if not isinstance(FEATURE_COLUMNS, list):
            raise TypeError(f"FEATURE_COLUMNS debe ser list, llegó {type(FEATURE_COLUMNS)}")

        if not isinstance(features_row, pd.Series):
            raise TypeError(f"features_row debe ser pd.Series, llegó {type(features_row)}")

        missing = [c for c in FEATURE_COLUMNS if c not in features_row.index]
        if missing:
            log(f"[ML] faltan features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            return None

        X = features_row[FEATURE_COLUMNS].values.reshape(1, -1)
        X_scaled = SCALER.transform(X)
        pred_class = ML_MODEL.predict(X_scaled)[0]
        mapping = {0: -1, 1: 0, 2: 1}
        return mapping[pred_class]
    except Exception as e:
        log(f"[ML] Predicción fallida: {e}")
        return None


# ===============================
# POSITION & ORDER MANAGEMENT
# ===============================
def get_position(ex, symbol):
    try:
        positions = ex.fetch_positions([symbol])
    except Exception:
        positions = ex.fetch_positions()

    for p in positions:
        if p.get("symbol") == symbol:
            contracts = float(p.get("contracts") or 0.0)
            if abs(contracts) < 1e-12:
                return None

            side = p.get("side")
            entry = float(p.get("entryPrice") or 0.0)
            mark = float(p.get("markPrice") or 0.0)

            if mark == 0.0:
                ticker = ex.fetch_ticker(symbol)
                mark = float(ticker["last"])

            side_norm = "LONG" if side and side.lower() == "long" else "SHORT"
            return {
                "contracts": contracts,
                "entry": entry,
                "mark": mark,
                "side": side_norm,
            }

    return None


def cancel_if_exists(ex, symbol, order_id):
    if not order_id:
        return None
    try:
        ex.cancel_order(order_id, symbol)
    except Exception:
        pass
    return None


def order_filled(ex, symbol, order_id):
    if not order_id:
        return False
    try:
        o = ex.fetch_order(order_id, symbol)
        return o.get("status") == "closed"
    except Exception:
        return False


def place_sl_tp_orders(ex, symbol, side, qty, sl_price, tp_price):
    close_side = "sell" if side == "buy" else "buy"

    sl = ex.create_order(
        symbol, "STOP_MARKET", close_side, qty, None,
        params={
            "stopPrice": float(f"{sl_price:.2f}"),
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
        }
    )
    tp = ex.create_order(
        symbol, "TAKE_PROFIT_MARKET", close_side, qty, None,
        params={
            "stopPrice": float(f"{tp_price:.2f}"),
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
        }
    )
    return sl["id"], tp["id"]


def replace_sl_order(ex, symbol, side, qty, new_sl_price, old_sl_id):
    cancel_if_exists(ex, symbol, old_sl_id)
    close_side = "sell" if side == "buy" else "buy"
    sl = ex.create_order(
        symbol, "STOP_MARKET", close_side, qty, None,
        params={
            "stopPrice": float(f"{new_sl_price:.2f}"),
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
        }
    )
    return sl["id"]


def sl_move_pct(entry_px, sl_px, side):
    if side == "buy":
        return max((entry_px - sl_px) / entry_px, 1e-9)
    else:
        return max((sl_px - entry_px) / entry_px, 1e-9)


def pnl_percent(entry, mark, side, lev):
    if entry <= 0:
        return 0.0
    if side == "LONG":
        return ((mark - entry) / entry) * lev * 100
    else:
        return ((entry - mark) / entry) * lev * 100


def compute_notional_usdt(equity_usdt, risk_pct, sl_move_pct_value):
    risk_usdt = equity_usdt * (risk_pct / 100.0)
    if sl_move_pct_value <= 1e-9:
        return 0.0
    return float(risk_usdt / sl_move_pct_value)


def market_open_with_minimums(ex, symbol, side, target_notional_usdt, equity_usdt, lev):
    ticker = ex.fetch_ticker(symbol)
    price = float(ticker["last"])
    market = ex.market(symbol)
    min_qty = float(market.get("limits", {}).get("amount", {}).get("min", 0.00001))
    min_notional = min_qty * price * 1.01
    max_notional = equity_usdt * lev * 0.98

    if min_notional > max_notional:
        needed_equity = min_notional / lev
        raise Exception(
            f"Notional mínimo ≈{min_notional:.2f} USDT excede máximo {max_notional:.2f}. "
            f"Necesitas equity ≥{needed_equity:.2f} USDT."
        )

    notional = max(min_notional, min(target_notional_usdt, max_notional))
    qty = notional / price
    qty = float(ex.amount_to_precision(symbol, qty))
    order = ex.create_order(symbol, "market", side, qty)
    return order, qty, notional, price


def get_usdt_equity(ex):
    bal = ex.fetch_balance()
    total = bal.get("total", {})
    return float(total.get("USDT", 0.0) or 0.0)


# ===============================
# STATE INIT
# ===============================
def make_default_state(cfg):
    return {
        "cooldown_until": 0,
        "last_heartbeat_ts": 0,
        "last_signal": None,
        "loss_streak": 0,
        "paused": False,
        "orders": {"sl_id": None, "tp_id": None},
        "trade": {
            "entry_time": None,
            "entry_px": None,
            "init_sl_px": None,
            "tp_px": None,
            "atr": None,
            "max_fav_R": -999,
            "last_sl_px": None,
            "side": None,
        },
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
    cfg, config_source = load_config()
    ex = get_exchange()
    symbol = cfg["symbol"]

    ensure_trades_csv()

    mode = cfg.get("mode", "feature_prob")
    model_status = "N/A"
    if mode in ("ml_model", "hybrid"):
        if load_ml_artifacts():
            model_status = "OK"
        else:
            log("[BOT] ML artifacts missing. Falling back to feature_prob.")
            mode = "feature_prob"
            cfg["mode"] = "feature_prob"
            model_status = "FALLBACK feature_prob"

    state = load_json(STATE_FILE, None)
    if state is None:
        state = make_default_state(cfg)
        save_json(STATE_FILE, state)

    risk = cfg.get("risk_engine", {})
    execution = cfg.get("execution", {})
    hybrid_cfg = cfg.get("hybrid", {})
    legacy_cfg = cfg.get("legacy", {})
    feature_cfg = cfg.get("feature_prob", {})

    sl_vol_mult = float(risk.get("sl_vol_mult", 1.2))
    tp_vol_mult = float(risk.get("tp_vol_mult", 2.8))
    breakeven_R = float(risk.get("breakeven_R", 1.0))
    trail_start_R = float(risk.get("trail_start_R", 1.5))
    trail_vol_mult = float(risk.get("trail_vol_mult", 1.0))
    risk_per_trade_pct = float(risk.get("risk_per_trade_pct", 0.5))
    max_hold_minutes = int(risk.get("max_hold_minutes", 360))
    cooldown_minutes = int(risk.get("cooldown_minutes", 180))

    heartbeat_seconds = int(cfg.get("status_heartbeat_seconds", 60))
    poll_seconds = int(cfg.get("poll_seconds", 10))
    leverage = float(cfg.get("leverage", 5))
    max_loss_streak = int(cfg.get("max_loss_streak", 5))

    fee_rt = float(execution.get("fee_rate_roundtrip", 0.001))
    slip_rt = float(execution.get("slippage_roundtrip_pct", 0.0001))

    # Startup sanity
    equity = get_usdt_equity(ex)

    log("[BOT] ==================================================")
    log(f"[BOT] iniciado | env=LIVE | exchange=BYBIT | symbol={symbol} | mode={mode} | tf_signal={cfg.get('tf_signal')} | tf_confirm={cfg.get('tf_confirm')} | tf_trend={cfg.get('tf_trend')}")
    log(f"[BOT] config source={config_source}")
    log(f"[BOT] leverage={leverage}x | risk_per_trade_pct={risk_per_trade_pct:.4f}% | cooldown={cooldown_minutes}m | max_hold={max_hold_minutes}m")
    log(f"[BOT] risk engine | SLxATR={sl_vol_mult:.4f} | TPxATR={tp_vol_mult:.4f} | BE_R={breakeven_R:.4f} | TRAIL_R={trail_start_R:.4f} | TRAIL_VOL={trail_vol_mult:.4f}")
    log(f"[BOT] execution | fees_enabled={execution.get('fees_enabled', True)} | fee_rt={fee_rt} | slip_rt={slip_rt}")
    log(f"[BOT] files | config={path_exists_str(CONFIG_FILE)} | params_bt={path_exists_str(PARAMS_FILE)} | state={path_exists_str(STATE_FILE)} | trades_csv={path_exists_str(TRADES_FILE)}")
    log(f"[BOT] ML files | model={path_exists_str(ML_MODEL_FILE)} | scaler={path_exists_str(ML_SCALER_FILE)} | features={path_exists_str(ML_FEATURES_FILE)} | metadata={path_exists_str(ML_METADATA_FILE)} | status={model_status}")
    log(f"[BOT] wallet | equity_usdt={equity:.2f}")
    log(f"[BOT] expected | rr={tp_vol_mult / max(sl_vol_mult, 1e-9):.4f} | est_tp_pnl_pct={expected_trade_pnl_pct(risk_per_trade_pct, tp_vol_mult, sl_vol_mult, fee_rt, slip_rt):.4f}%")
    if mode == "hybrid":
        log(f"[BOT] hybrid | ml_weight={float(hybrid_cfg.get('ml_weight', 0.6)):.4f} | long_th={float(hybrid_cfg.get('long_threshold', feature_cfg.get('long_threshold', 0.58))):.4f} | short_th={float(hybrid_cfg.get('short_threshold', feature_cfg.get('short_threshold', 0.58))):.4f}")
    elif mode == "feature_prob":
        log(f"[BOT] feature_prob | long_th={float(feature_cfg.get('long_threshold', 0.58)):.4f} | short_th={float(feature_cfg.get('short_threshold', 0.58)):.4f} | persistence={int(feature_cfg.get('persistence_bars', 2))}")
    elif mode == "legacy_classic":
        log(f"[BOT] legacy | ema_fast_signal={legacy_cfg.get('ema_fast_signal')} | ema_slow_signal={legacy_cfg.get('ema_slow_signal')} | rsi_len={legacy_cfg.get('rsi_len_signal')}")
    log("[BOT] ==================================================")

    send_startup_message(cfg, equity, model_status, config_source)

    last_signal_candle_ts = None

    while True:
        try:
            now = now_ts()

            if state.get("paused"):
                log("[BOT] PAUSED due to loss streak. Set paused=false to resume.")
                time.sleep(10)
                continue

            # heartbeat
            if now - int(state.get("last_heartbeat_ts", 0)) >= heartbeat_seconds:
                pos = get_position(ex, symbol)
                pos_txt = "OPEN" if pos else "FLAT"
                cd_until = int(state.get("cooldown_until", 0))
                cd_left = max(0, cd_until - now)
                loss_streak = int(state.get("loss_streak", 0))
                eq_now = get_usdt_equity(ex)
                hb = (
                    f"[BOT][HEARTBEAT] {utc_now_str()} | "
                    f"mode={cfg.get('mode')} | symbol={symbol} | tf={cfg.get('tf_signal')} | "
                    f"status={pos_txt} | cooldown_left={cd_left}s | loss_streak={loss_streak} | equity={eq_now:.2f} USDT"
                )
                log(hb)
                state["last_heartbeat_ts"] = now
                save_json(STATE_FILE, state)

            # 1) manage open position
            pos = get_position(ex, symbol)
            if pos:
                entry = float(pos["entry"])
                mark = float(pos["mark"])
                side_pos = pos["side"]
                qty = abs(pos["contracts"])

                tr = state.get("trade", {})
                init_sl_px = float(tr.get("init_sl_px") or 0.0)
                atr_val = float(tr.get("atr") or 0.0)
                max_fav_R = float(tr.get("max_fav_R") or -999)
                last_sl_px = float(tr.get("last_sl_px") or init_sl_px)

                if init_sl_px > 0 and entry > 0:
                    init_risk = abs(entry - init_sl_px)
                    if init_risk < 1e-9:
                        cur_R = 0.0
                    else:
                        if side_pos == "LONG":
                            cur_R = (mark - entry) / init_risk
                        else:
                            cur_R = (entry - mark) / init_risk
                    max_fav_R = max(max_fav_R, cur_R)
                    tr["max_fav_R"] = max_fav_R

                new_sl_px = last_sl_px
                if max_fav_R >= breakeven_R:
                    be_px = entry
                    if side_pos == "LONG":
                        new_sl_px = max(last_sl_px, be_px)
                    else:
                        new_sl_px = min(last_sl_px, be_px)

                if max_fav_R >= trail_start_R and atr_val > 0:
                    if side_pos == "LONG":
                        trail_px = mark - atr_val * trail_vol_mult
                        new_sl_px = max(new_sl_px, trail_px)
                    else:
                        trail_px = mark + atr_val * trail_vol_mult
                        new_sl_px = min(new_sl_px, trail_px)

                if abs(new_sl_px - last_sl_px) / max(1.0, entry) > 1e-4:
                    old_sl = state["orders"].get("sl_id")
                    side_entry = "buy" if side_pos == "LONG" else "sell"
                    new_sl_id = replace_sl_order(ex, symbol, side_entry, qty, new_sl_px, old_sl)
                    state["orders"]["sl_id"] = new_sl_id
                    tr["last_sl_px"] = new_sl_px
                    state["trade"] = tr
                    save_json(STATE_FILE, state)
                    log(f"[BOT] SL updated -> {fmt(new_sl_px, 2)} (R={fmt(cur_R, 2)})")

                tp_filled = order_filled(ex, symbol, state["orders"].get("tp_id"))
                sl_filled = order_filled(ex, symbol, state["orders"].get("sl_id"))
                pos_check = get_position(ex, symbol)

                if not pos_check:
                    pnl = pnl_percent(entry, mark, side_pos, leverage)
                    if tp_filled:
                        exit_reason = "TP"
                    elif sl_filled:
                        exit_reason = "SL"
                    else:
                        exit_reason = "MANUAL"

                    log(f"[BOT] POSITION CLOSED | {side_pos} | PnL={pnl:.2f}% | reason={exit_reason}")

                    equity_before = None
                    try:
                        # aproximación rápida
                        equity_before = get_usdt_equity(ex)
                    except Exception:
                        pass

                    send_close_message(symbol, side_pos, entry, mark, pnl, exit_reason, equity_before)

                    if exit_reason in ("SL",) and pnl < 0:
                        state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
                        if state["loss_streak"] >= max_loss_streak:
                            state["paused"] = True
                            notify(f"[BOT] PAUSED after {max_loss_streak} consecutive losses.")
                    else:
                        state["loss_streak"] = 0

                    state["cooldown_until"] = now + cooldown_minutes * 60
                    state["orders"] = {"sl_id": None, "tp_id": None}
                    state["trade"] = {
                        "entry_time": None,
                        "entry_px": None,
                        "init_sl_px": None,
                        "tp_px": None,
                        "atr": None,
                        "max_fav_R": -999,
                        "last_sl_px": None,
                        "side": None,
                    }
                    save_json(STATE_FILE, state)
                    time.sleep(poll_seconds)
                    continue

                entry_time_iso = tr.get("entry_time")
                if entry_time_iso:
                    entry_dt = pd.to_datetime(entry_time_iso, utc=True)
                    age_min = (pd.Timestamp.now(tz="UTC") - entry_dt).total_seconds() / 60.0
                    if age_min >= max_hold_minutes:
                        log(f"[BOT] Closing position due to max hold time ({max_hold_minutes} min)")
                        close_side = "sell" if side_pos == "LONG" else "buy"
                        ex.create_order(symbol, "market", close_side, qty, params={"reduceOnly": True})
                        time.sleep(2)
                        continue

                time.sleep(poll_seconds)
                continue

            # 2) no position: check cooldown
            if now < int(state.get("cooldown_until", 0)):
                time.sleep(poll_seconds)
                continue

            # 3) generate signal (only on new candle of tf_signal)
            tf_signal = cfg["tf_signal"]
            df = fetch_ohlc(ex, symbol, tf_signal, 500)
            if df.empty:
                time.sleep(poll_seconds)
                continue

            last_ts = int(df["time"].iloc[-1].timestamp())
            if last_signal_candle_ts is None:
                last_signal_candle_ts = last_ts
            is_new_candle = (last_ts != last_signal_candle_ts)
            if not is_new_candle:
                time.sleep(poll_seconds)
                continue
            last_signal_candle_ts = last_ts

            side = None
            price = None

            if mode == "legacy_classic":
                side, row = legacy_signal(df, cfg)
                if side is not None:
                    price = float(row["close"])
                    log(f"[BOT][LEGACY] signal {side} @ {price:.2f}")

            elif mode == "feature_prob":
                feat = build_features(df)
                row = feat.iloc[-1]
                prob_long, prob_short = feature_signal(row, cfg)
                log(
                    f"[BOT][FEATURE] close={row['close']:.2f} | "
                    f"prob_long={prob_long:.3f} | prob_short={prob_short:.3f} | "
                    f"rv={row['rv']:.6f} | vol_ratio={row['vol_ratio']:.3f}"
                )
                if prob_long >= cfg["feature_prob"]["long_threshold"]:
                    side = "buy"
                    price = float(row["close"])
                elif prob_short >= cfg["feature_prob"]["short_threshold"]:
                    side = "sell"
                    price = float(row["close"])

            elif mode == "ml_model":
                features_row = get_multi_tf_features(ex, symbol, base_freq="1h")
                if features_row is None:
                    log("[ML] No se pudieron obtener features multi-tf")
                    time.sleep(poll_seconds)
                    continue

                ml_pred = ml_prediction(features_row)
                if ml_pred is None:
                    log("[ML] Predicción fallida")
                    time.sleep(poll_seconds)
                    continue

                if ml_pred == 1:
                    side = "buy"
                elif ml_pred == -1:
                    side = "sell"

                if side is not None:
                    price = df["close"].iloc[-1]
                    log(f"[BOT][ML] signal {side} @ {price:.2f} (ml_pred={ml_pred})")

            elif mode == "hybrid":
                feat = build_features(df)
                row = feat.iloc[-1]
                prob_long, prob_short = feature_signal(row, cfg)

                features_row = get_multi_tf_features(ex, symbol, base_freq="1h")
                if features_row is None:
                    log("[ML] No se pudieron obtener features multi-tf para hybrid")
                    time.sleep(poll_seconds)
                    continue

                ml_pred = ml_prediction(features_row)
                if ml_pred is None:
                    log("[ML] Predicción fallida en hybrid")
                    time.sleep(poll_seconds)
                    continue

                ml_weight = float(hybrid_cfg.get("ml_weight", 0.6))
                heuristic_weight = 1.0 - ml_weight

                if ml_pred == 1:
                    ml_long = 1.0
                    ml_short = 0.0
                elif ml_pred == -1:
                    ml_long = 0.0
                    ml_short = 1.0
                else:
                    ml_long = 0.0
                    ml_short = 0.0

                combined_long = ml_weight * ml_long + heuristic_weight * prob_long
                combined_short = ml_weight * ml_short + heuristic_weight * prob_short
                long_th = hybrid_cfg.get("long_threshold", cfg["feature_prob"]["long_threshold"])
                short_th = hybrid_cfg.get("short_threshold", cfg["feature_prob"]["short_threshold"])

                log(
                    f"[BOT][HYBRID] close={row['close']:.2f} | "
                    f"ml_pred={ml_pred} | heur_long={prob_long:.3f} heur_short={prob_short:.3f} | "
                    f"combined_long={combined_long:.3f} combined_short={combined_short:.3f}"
                )

                if combined_long >= long_th:
                    side = "buy"
                    price = float(row["close"])
                elif combined_short >= short_th:
                    side = "sell"
                    price = float(row["close"])

            if side is None:
                time.sleep(poll_seconds)
                continue

            atr_val = atr(df[["open", "high", "low", "close"]], 14).iloc[-1]
            if atr_val <= 0:
                log("[BOT] ATR zero, skipping entry")
                time.sleep(poll_seconds)
                continue

            if side == "buy":
                sl_price = price - atr_val * sl_vol_mult
                tp_price = price + atr_val * tp_vol_mult
            else:
                sl_price = price + atr_val * sl_vol_mult
                tp_price = price - atr_val * tp_vol_mult

            sl_pct_move_value = sl_move_pct(price, sl_price, side)
            if sl_pct_move_value <= 0:
                log("[BOT] SL too close, skipping")
                time.sleep(poll_seconds)
                continue

            equity = get_usdt_equity(ex)
            target_notional = compute_notional_usdt(equity, risk_per_trade_pct, sl_pct_move_value)

            try:
                order, qty, used_notional, entry_price = market_open_with_minimums(
                    ex, symbol, side, target_notional, equity, leverage
                )
            except Exception as e:
                log(f"[BOT] Order failed: {e}")
                send_error_message("Order placement", str(e))
                time.sleep(poll_seconds)
                continue

            time.sleep(2)
            pos_new = get_position(ex, symbol)
            if not pos_new:
                log("[BOT] Position not found after entry")
                send_error_message("Post-entry position check", "Position not found after entry")
                time.sleep(poll_seconds)
                continue

            entry_px = float(pos_new["entry"])
            qty_pos = abs(pos_new["contracts"])

            sl_id, tp_id = place_sl_tp_orders(ex, symbol, side, qty_pos, sl_price, tp_price)

            state["orders"]["sl_id"] = sl_id
            state["orders"]["tp_id"] = tp_id
            state["trade"] = {
                "entry_time": utc_now_str(),
                "entry_px": entry_px,
                "init_sl_px": sl_price,
                "tp_px": tp_price,
                "atr": atr_val,
                "max_fav_R": -999,
                "last_sl_px": sl_price,
                "side": side,
            }
            state["last_entry_side"] = side
            state["last_entry_price"] = entry_px
            state["last_signal"] = utc_now_str()
            state["cooldown_until"] = now + cooldown_minutes * 60
            save_json(STATE_FILE, state)

            log(
                f"[BOT] ENTER {side.upper()} @ {entry_px:.2f} | "
                f"size={qty_pos:.6f} | notional={used_notional:.2f} | "
                f"SL={sl_price:.2f} TP={tp_price:.2f} | ATR={atr_val:.2f} | rr={tp_vol_mult / max(sl_vol_mult, 1e-9):.2f}"
            )

            send_entry_message(
                symbol=symbol,
                side=side,
                price=entry_px,
                qty=qty_pos,
                sl=sl_price,
                tp=tp_price,
                rr=(tp_vol_mult / max(sl_vol_mult, 1e-9)),
                mode=mode,
                equity=equity,
            )

            try:
                append_trade_row({
                    "timestamp": utc_now_str(),
                    "symbol": symbol,
                    "mode": mode,
                    "side": side,
                    "price": entry_px,
                    "size": qty_pos,
                    "sl": sl_price,
                    "tp": tp_price,
                    "env": "LIVE",
                })
                log(f"[BOT] trades.csv append OK -> {TRADES_FILE}")
            except Exception as e:
                log(f"[WARN] could not write trades.csv: {e}")
                send_error_message("Trade CSV append", str(e))

            time.sleep(poll_seconds)

        except Exception as e:
            log(f"[ERR] {e}")
            traceback.print_exc()
            send_error_message("Loop principal", str(e))
            time.sleep(5)


if __name__ == "__main__":
    main()
