#!/usr/bin/env python3
"""
STRATUM Backtester (BTC/USDT) - Hybrid / ML / Heuristic / Legacy
Optimized to backtest ONLY the last year and save results for the GUI.
"""

from __future__ import annotations

import json
import pickle
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Callable, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================
# Paths and config
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
CONFIGS_DIR = BASE_DIR / "configs"
RUNTIME_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)

PARAMS_FILE = CONFIGS_DIR / "params_bt.json"
BACKTEST_TRADES_FILE = RUNTIME_DIR / "backtest_trades.csv"
BACKTEST_STATS_FILE = RUNTIME_DIR / "backtest_stats.json"

# ML artifacts
ML_MODEL_FILE = BASE_DIR / "btc_ml_model.pkl"
SCALER_FILE = BASE_DIR / "scaler.pkl"
FEATURE_COLUMNS_FILE = BASE_DIR / "feature_columns.json"

# Supported TFs for feature construction
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "1d", "1w"]

# This value is only used as a preferred filename pattern.
# If it doesn't exist, the loader will fallback automatically.
YEARS = 3

# Backtest only the most recent year
BACKTEST_LOOKBACK_DAYS = 1000

# =========================================================
# Generic helpers
# =========================================================
def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def normalize_ohlcv(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()

    # If time is a column, promote it to index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time")

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()

    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    for key in ["open", "high", "low", "close"]:
        if key in cols_lower and cols_lower[key] != key:
            rename_map[cols_lower[key]] = key

    if "volume" in cols_lower:
        rename_map[cols_lower["volume"]] = "vol"
    elif "vol" in cols_lower and cols_lower["vol"] != "vol":
        rename_map[cols_lower["vol"]] = "vol"

    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "vol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{source_name} missing required columns: {missing}. Available: {list(df.columns)}"
        )

    # Keep only last year
    if len(df) > 0:
        cutoff = df.index.max() - pd.Timedelta(days=BACKTEST_LOOKBACK_DAYS)
        df = df.loc[df.index >= cutoff].copy()

    # Ensure numeric
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)

    return df.sort_index()


def find_best_available_parquet(tf: str) -> Path:
    preferred = BASE_DIR / f"btc_{tf}_{YEARS}y.parquet"
    if preferred.exists():
        return preferred

    candidates = sorted(BASE_DIR.glob(f"btc_{tf}_*y.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"Missing data file: {preferred} and no fallback matched btc_{tf}_*y.parquet"
        )

    fallback = candidates[-1]
    print(f"[BT] Requested {preferred.name} not found, using fallback {fallback.name}")
    return fallback


# =========================================================
# Data loading
# =========================================================
def load_btc_data(tf: str) -> pd.DataFrame:
    # Special support for 5m built from 1m if 5m parquet is missing
    if tf == "5m":
        preferred_5m = BASE_DIR / f"btc_5m_{YEARS}y.parquet"
        if preferred_5m.exists():
            df = pd.read_parquet(preferred_5m)
            return normalize_ohlcv(df, preferred_5m.name)

        one_min_file = find_best_available_parquet("1m")
        print(f"[BT] 5m parquet missing, building 5m from {one_min_file.name}...")
        df_1m = pd.read_parquet(one_min_file)
        df_1m = normalize_ohlcv(df_1m, one_min_file.name)

        df_5m = pd.DataFrame({
            "open": df_1m["open"].resample("5min").first(),
            "high": df_1m["high"].resample("5min").max(),
            "low": df_1m["low"].resample("5min").min(),
            "close": df_1m["close"].resample("5min").last(),
            "vol": df_1m["vol"].resample("5min").sum(),
        }).dropna()

        return df_5m.sort_index()

    parquet_file = find_best_available_parquet(tf)
    df = pd.read_parquet(parquet_file)
    return normalize_ohlcv(df, parquet_file.name)


def resample_to_base(df: pd.DataFrame, base_freq: str) -> pd.DataFrame:
    return df.resample(base_freq).ffill()


# =========================================================
# Multi-timeframe ML features
# =========================================================
def compute_features_tf(df: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    """
    Feature construction compatible with the ML model.
    """
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
        df["close"] > donchian_high.shift(1),
        1,
        np.where(df["close"] < donchian_low.shift(1), -1, 0),
    ).astype("int8")

    features = features.add_prefix(f"{tf_label}_")
    features = features.dropna()
    features.index = pd.to_datetime(features.index, errors="coerce")
    features = features[~features.index.isna()].sort_index()
    features = features[~features.index.duplicated(keep="last")]
    return features


def build_multi_tf_features(base_freq: str = "1h") -> pd.DataFrame:
    all_feats = None
    min_rows = 10   # Ignorar timeframes con muy pocas filas

    for tf in TIMEFRAMES:
        print(f"Loading {tf} data...")
        try:
            df = load_btc_data(tf)
            feats = compute_features_tf(df, tf)
            feats = resample_to_base(feats, base_freq)
            feats = feats[~feats.index.duplicated(keep="last")]

            if len(feats) < min_rows:
                print(f"[WARN] {tf} has only {len(feats)} rows, skipping")
                continue

            if all_feats is None:
                all_feats = feats
            else:
                # Outer join para no perder datos de otros timeframes
                all_feats = all_feats.join(feats, how="outer")
        except Exception as e:
            print(f"[WARN] Failed to process {tf}: {e}")
            continue

    if all_feats is None:
        return pd.DataFrame()

    all_feats = all_feats[~all_feats.index.duplicated(keep="last")]
    return all_feats


def align_features_to_signal(full_features: pd.DataFrame, df_signal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Alinea features multi-TF al timeframe de señal usando merge_asof.
    Evita errores de dtype object vs datetime64 y problemas con nombres de índice.
    """
    full_features = full_features.copy()
    df_signal = df_signal.copy()

    # Forzar índices datetime y limpios
    full_features.index = pd.to_datetime(full_features.index, errors="coerce")
    df_signal.index = pd.to_datetime(df_signal.index, errors="coerce")

    full_features = full_features[~full_features.index.isna()].sort_index()
    df_signal = df_signal[~df_signal.index.isna()].sort_index()

    full_features = full_features[~full_features.index.duplicated(keep="last")]
    df_signal = df_signal[~df_signal.index.duplicated(keep="last")]

    # Forzar nombre único del índice antes de reset_index
    full_features.index.name = "time"
    df_signal.index.name = "time"

    left = df_signal.reset_index()
    right = full_features.reset_index()

    # Blindaje extra: si por cualquier razón no quedó 'time', renombrar primera columna
    if "time" not in left.columns:
        left = left.rename(columns={left.columns[0]: "time"})
    if "time" not in right.columns:
        right = right.rename(columns={right.columns[0]: "time"})

    left["time"] = pd.to_datetime(left["time"], errors="coerce")
    right["time"] = pd.to_datetime(right["time"], errors="coerce")

    left = left.dropna(subset=["time"]).sort_values("time")
    right = right.dropna(subset=["time"]).sort_values("time")

    merged = pd.merge_asof(
        left,
        right,
        on="time",
        direction="backward",
    )

    merged = merged.dropna().set_index("time")

    feature_cols = list(full_features.columns)
    signal_cols = list(df_signal.columns)

    df_signal_aligned = merged[signal_cols].copy()
    full_features_aligned = merged[feature_cols].copy()

    return full_features_aligned, df_signal_aligned


# =========================================================
# Heuristic features and signals
# =========================================================
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


def build_heuristic_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()

    if "vol" not in f.columns:
        if "volume" in f.columns:
            f["vol"] = f["volume"]
        else:
            raise KeyError(
                f"build_heuristic_features expected 'vol' column. Available: {list(f.columns)}"
            )

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


def heuristic_signal(row: pd.Series, cfg: dict) -> Tuple[float, float]:
    fp = cfg.get("feature_prob", {})
    score_long = 0.0
    score_short = 0.0

    if row["ret1"] > 0:
        score_long += fp.get("ret_1_weight", 0.10)
    else:
        score_short += fp.get("ret_1_weight", 0.10)

    if row["ret3"] > 0:
        score_long += fp.get("ret_3_weight", 0.12)
    else:
        score_short += fp.get("ret_3_weight", 0.12)

    if row["ret12"] > 0:
        score_long += fp.get("ret_12_weight", 0.14)
    else:
        score_short += fp.get("ret_12_weight", 0.14)

    if row["range_pos"] > 0.65:
        score_long += fp.get("range_pos_weight", 0.10)
    if row["range_pos"] < 0.35:
        score_short += fp.get("range_pos_weight", 0.10)

    if row["vol_ratio"] > 1.2:
        score_long += fp.get("vol_ratio_weight", 0.08) * 0.5
        score_short += fp.get("vol_ratio_weight", 0.08) * 0.5

    if row["rv_ratio"] > 1.05:
        score_long += fp.get("realized_vol_weight", 0.08) * 0.5
        score_short += fp.get("realized_vol_weight", 0.08) * 0.5

    if row["slope"] > 0:
        score_long += fp.get("slope_weight", 0.12)
    else:
        score_short += fp.get("slope_weight", 0.12)

    if row["breakout_up"]:
        score_long += fp.get("breakout_weight", 0.10)
    if row["breakout_dn"]:
        score_short += fp.get("breakout_weight", 0.10)

    total = max(score_long + score_short, 1e-9)
    return score_long / total, score_short / total


# =========================================================
# Legacy indicators/signals
# =========================================================
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


def legacy_signal(df: pd.DataFrame, cfg: dict) -> Tuple[Optional[str], pd.Series]:
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


# =========================================================
# ML model
# =========================================================
ML_MODEL = None
SCALER = None
FEATURE_COLUMNS = None


def load_ml_model() -> bool:
    global ML_MODEL, SCALER, FEATURE_COLUMNS
    try:
        with open(ML_MODEL_FILE, "rb") as f:
            ML_MODEL = pickle.load(f)
        with open(SCALER_FILE, "rb") as f:
            SCALER = pickle.load(f)
        with open(FEATURE_COLUMNS_FILE, "r", encoding="utf-8") as f:
            FEATURE_COLUMNS = json.load(f)

        if not isinstance(FEATURE_COLUMNS, list):
            raise TypeError(f"feature_columns.json must contain a list, got {type(FEATURE_COLUMNS)}")

        print("[BT] ML model loaded successfully")
        return True
    except Exception as e:
        print(f"[BT] Error loading ML model: {e}")
        return False


def ml_signal(features_row: pd.Series) -> Optional[int]:
    if ML_MODEL is None or SCALER is None or FEATURE_COLUMNS is None:
        return None
    try:
        missing = [c for c in FEATURE_COLUMNS if c not in features_row.index]
        if missing:
            # Solo mostrar una vez para evitar spam
            print(f"[BT] Missing ML features: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            return None

        X = features_row[FEATURE_COLUMNS].values.reshape(1, -1)
        X_scaled = SCALER.transform(X)
        pred_class = ML_MODEL.predict(X_scaled)[0]
        mapping = {0: -1, 1: 0, 2: 1}
        return mapping[pred_class]
    except Exception as e:
        print(f"[BT] ML prediction error: {e}")
        return None


# =========================================================
# Risk management
# =========================================================
def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def compute_sl_tp(entry_price: float, atr_val: float, side: str, cfg: dict) -> Tuple[float, float]:
    risk = cfg.get("risk_engine", {})
    sl_mult = float(risk.get("sl_vol_mult", 1.2))
    tp_mult = float(risk.get("tp_vol_mult", 2.8))
    if side == "buy":
        sl = entry_price - atr_val * sl_mult
        tp = entry_price + atr_val * tp_mult
    else:
        sl = entry_price + atr_val * sl_mult
        tp = entry_price - atr_val * tp_mult
    return sl, tp


# =========================================================
# Main backtest
# =========================================================
def run_backtest(cfg: dict, progress_callback: Callable = None, **kwargs) -> Tuple[pd.DataFrame, dict]:
    mode = cfg.get("mode", "feature_prob")
    print(f"[BT] Starting backtest in mode: {mode}")
    print(f"[BT] Lookback: last {BACKTEST_LOOKBACK_DAYS} days")

    if mode in ("ml_model", "hybrid"):
        print("[BT] Building multi-timeframe features (this may take a while)...")
        try:
            full_features = build_multi_tf_features(base_freq="1h")
        except Exception as e:
            print(f"[BT] Error building multi-TF features: {e}")
            return None, {}

        tf_signal = cfg.get("tf_signal", "15m")
        df_signal = load_btc_data(tf_signal)

        if full_features.empty:
            print("[BT] full_features is empty after building, cannot proceed.")
            return pd.DataFrame(), {}
        if df_signal.empty:
            print("[BT] df_signal is empty after loading.")
            return pd.DataFrame(), {}

        print(f"[BT] full_features shape: {full_features.shape}, index range: {full_features.index.min()} to {full_features.index.max()}")
        print(f"[BT] df_signal shape: {df_signal.shape}, index range: {df_signal.index.min()} to {df_signal.index.max()}")

        full_features, df_signal = align_features_to_signal(full_features, df_signal)
        print(f"[BT] Aligned rows: features={len(full_features)} signal={len(df_signal)}")

        if full_features.empty or df_signal.empty:
            print("[BT] No overlapping rows after alignment")
            return pd.DataFrame(), {}
    else:
        tf_signal = cfg.get("tf_signal", "15m")
        df_signal = load_btc_data(tf_signal).sort_index()
        full_features = pd.DataFrame()  # placeholder

    trades = []
    position = None
    entry_price = 0.0
    entry_time = None
    entry_atr = 0.0
    entry_side = None
    sl_price = 0.0
    tp_price = 0.0
    max_fav_R = -999.0
    last_sl_price = 0.0

    risk = cfg.get("risk_engine", {})
    breakeven_R = float(risk.get("breakeven_R", 1.0))
    trail_start_R = float(risk.get("trail_start_R", 1.5))
    trail_vol_mult = float(risk.get("trail_vol_mult", 1.0))
    max_hold_minutes = int(risk.get("max_hold_minutes", 360))
    cooldown_minutes = int(risk.get("cooldown_minutes", 180))

    cooldown_end = None

    for idx in range(len(df_signal)):
        if idx < 100:   # saltar inicio para estabilizar indicadores
            continue

        row = df_signal.iloc[idx]
        current_time = row.name
        current_price = row["close"]

        if position is not None:
            age_min = (current_time - entry_time).total_seconds() / 60.0
            if age_min >= max_hold_minutes:
                pnl_pct = (
                    (current_price - entry_price) / entry_price * 100
                    if entry_side == "buy"
                    else (entry_price - current_price) / entry_price * 100
                )
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "side": entry_side,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "TIME",
                    "max_fav_R": max_fav_R,
                })
                position = None
                cooldown_end = current_time + timedelta(minutes=cooldown_minutes)
                continue

            init_risk = abs(entry_price - sl_price) if sl_price != 0 else 1e-9
            cur_R = (
                (current_price - entry_price) / init_risk
                if entry_side == "buy"
                else (entry_price - current_price) / init_risk
            )
            max_fav_R = max(max_fav_R, cur_R)

            new_sl = last_sl_price
            if max_fav_R >= breakeven_R:
                be_price = entry_price
                if entry_side == "buy":
                    new_sl = max(last_sl_price, be_price)
                else:
                    new_sl = min(last_sl_price, be_price)

            if max_fav_R >= trail_start_R and entry_atr > 0:
                if entry_side == "buy":
                    trail_price = current_price - entry_atr * trail_vol_mult
                    new_sl = max(new_sl, trail_price)
                else:
                    trail_price = current_price + entry_atr * trail_vol_mult
                    new_sl = min(new_sl, trail_price)

            if entry_side == "buy":
                if current_price <= new_sl:
                    exit_reason = "SL"
                    pnl_pct = (new_sl - entry_price) / entry_price * 100
                elif current_price >= tp_price:
                    exit_reason = "TP"
                    pnl_pct = (tp_price - entry_price) / entry_price * 100
                else:
                    exit_reason = None
            else:
                if current_price >= new_sl:
                    exit_reason = "SL"
                    pnl_pct = (entry_price - new_sl) / entry_price * 100
                elif current_price <= tp_price:
                    exit_reason = "TP"
                    pnl_pct = (entry_price - tp_price) / entry_price * 100
                else:
                    exit_reason = None

            if exit_reason:
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "side": entry_side,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "max_fav_R": max_fav_R,
                })
                position = None
                cooldown_end = current_time + timedelta(minutes=cooldown_minutes)
                continue

            if new_sl != last_sl_price:
                last_sl_price = new_sl

            continue

        if cooldown_end is not None and current_time < cooldown_end:
            continue

        side = None

        if mode == "legacy_classic":
            side, _ = legacy_signal(df_signal.iloc[:idx + 1], cfg)
            if side is None:
                continue

        elif mode == "feature_prob":
            df_heuristic = build_heuristic_features(df_signal.iloc[:idx + 1])
            row_heuristic = df_heuristic.iloc[-1]
            prob_long, prob_short = heuristic_signal(row_heuristic, cfg)

            if prob_long >= cfg.get("feature_prob", {}).get("long_threshold", 0.58):
                side = "buy"
            elif prob_short >= cfg.get("feature_prob", {}).get("short_threshold", 0.58):
                side = "sell"
            else:
                continue

        elif mode == "ml_model":
            if current_time not in full_features.index:
                continue

            ml_pred = ml_signal(full_features.loc[current_time])
            if ml_pred == 1:
                side = "buy"
            elif ml_pred == -1:
                side = "sell"
            else:
                continue

        elif mode == "hybrid":
            if current_time not in full_features.index:
                continue

            ml_pred = ml_signal(full_features.loc[current_time])
            if ml_pred is None:
                continue

            df_heuristic = build_heuristic_features(df_signal.iloc[:idx + 1])
            row_heuristic = df_heuristic.iloc[-1]
            prob_long, prob_short = heuristic_signal(row_heuristic, cfg)

            ml_weight = float(cfg.get("hybrid", {}).get("ml_weight", 0.6))
            heuristic_weight = 1.0 - ml_weight

            if ml_pred == 1:
                ml_long, ml_short = 1.0, 0.0
            elif ml_pred == -1:
                ml_long, ml_short = 0.0, 1.0
            else:
                ml_long, ml_short = 0.0, 0.0

            combined_long = ml_weight * ml_long + heuristic_weight * prob_long
            combined_short = ml_weight * ml_short + heuristic_weight * prob_short
            long_th = float(cfg.get("hybrid", {}).get("long_threshold", 0.58))
            short_th = float(cfg.get("hybrid", {}).get("short_threshold", 0.58))

            if combined_long >= long_th:
                side = "buy"
            elif combined_short >= short_th:
                side = "sell"
            else:
                continue

        else:
            continue

        atr_val = atr(df_signal.iloc[:idx + 1], 14).iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        sl, tp = compute_sl_tp(current_price, atr_val, side, cfg)

        position = True
        entry_side = side
        entry_price = current_price
        entry_time = current_time
        entry_atr = atr_val
        sl_price = sl
        tp_price = tp
        last_sl_price = sl
        max_fav_R = -999.0

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    if trades_df.empty:
        return trades_df, {}

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], errors="coerce")
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], errors="coerce")

    # Compatibility with GUI and Apolo
    trades_df["pnl_equity_pct"] = trades_df["pnl_pct"]

    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] < 0]
    winrate = len(wins) / total_trades * 100 if total_trades > 0 else 0.0
    total_pnl = trades_df["pnl_pct"].sum()
    avg_pnl = trades_df["pnl_pct"].mean()

    if len(losses) > 0 and abs(losses["pnl_pct"].sum()) > 0:
        profit_factor = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())
    else:
        profit_factor = 10.0

    max_dd = trades_df["pnl_pct"].cumsum().min() if total_trades > 0 else 0.0

    if total_trades > 1:
        days = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).total_seconds() / 86400.0
        tpd = total_trades / days if days > 0 else 0.0
    else:
        days = 1.0
        tpd = 0.0

    proj_month = total_pnl * 30 / max(1.0, days) if total_trades > 1 else total_pnl
    exit_counts = trades_df["exit_reason"].value_counts().to_dict()

    stats = {
        "trades": total_trades,
        "winrate": winrate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "tpd": tpd,
        "proj_month": proj_month,
        "exit_counts": exit_counts,
        "lookback_days": BACKTEST_LOOKBACK_DAYS,
    }

    return trades_df, stats


# =========================================================
# Save outputs for GUI
# =========================================================
def save_backtest_outputs(trades: Optional[pd.DataFrame], stats: dict) -> None:
    if trades is not None and not trades.empty:
        trades.to_csv(BACKTEST_TRADES_FILE, index=False)
    else:
        pd.DataFrame().to_csv(BACKTEST_TRADES_FILE, index=False)

    save_json(BACKTEST_STATS_FILE, stats or {})


# =========================================================
# Standalone entrypoint
# =========================================================
if __name__ == "__main__":
    # Cargar configuración desde archivo o crear por defecto
    cfg = None
    if PARAMS_FILE.exists():
        try:
            with open(PARAMS_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print("[BT] Config loaded from", PARAMS_FILE)
        except json.JSONDecodeError as e:
            print(f"[BT] Error: {PARAMS_FILE} is not valid JSON: {e}")
            print("[BT] Renaming corrupted file and creating default config...")
            # Respaldo del archivo corrupto
            backup = PARAMS_FILE.with_suffix(".json.bak")
            PARAMS_FILE.rename(backup)
            print(f"[BT] Corrupted file backed up to {backup}")
            cfg = None
    else:
        print("[BT] Config file not found, using default.")

    if cfg is None:
        # Configuración por defecto (la que proporcionaste)
        cfg = {
            "symbol": "BTC/USDT:USDT",
            "market_symbol": "BTCUSDT",
            "poll_seconds": 10,
            "status_heartbeat_seconds": 60,
            "tf_signal": "5m",
            "tf_confirm": "3m",
            "tf_entry": "1m",
            "tf_trend": "1h",
            "live_chart_tf": "5m",
            "mode": "hybrid",
            "use_1h_trend_filter": True,
            "feature_prob": {
                "ret_1_weight": 0.03141615189885727,
                "ret_3_weight": 0.18866015004301281,
                "ret_12_weight": 0.3003548965021518,
                "range_pos_weight": 0.055325684938848874,
                "vol_ratio_weight": 0.10299065188227281,
                "realized_vol_weight": 0.11834937021251411,
                "slope_weight": 0.10247384571054015,
                "trend_weight": 0.24522627006305897,
                "breakout_weight": 0.1580495714431668,
                "long_threshold": 0.5679382121445798,
                "short_threshold": 0.5348897816190203,
                "persistence_bars": 2
            },
            "hybrid": {
                "ml_weight": 0.6,
                "long_threshold": 0.5679382121445798,
                "short_threshold": 0.5348897816190203
            },
            "risk_engine": {
                "sl_vol_mult": 1.8849241204267702,
                "tp_vol_mult": 5.405936553596987,
                "breakeven_R": 1.605070106941517,
                "trail_start_R": 1.3465267712254836,
                "trail_vol_mult": 2.0697531461338303,
                "risk_per_trade_pct": 0.5238500076546306,
                "max_hold_minutes": 331,
                "cooldown_minutes": 61
            },
            "execution": {
                "fees_enabled": True,
                "fee_rate_roundtrip": 0.001,
                "slippage_roundtrip_pct": 0.0001,
                "label_exits": True
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
                "sl_atr_mult": 1.1,
                "tp_atr_mult": 3.2,
                "breakeven_R": 1.0,
                "trail_start_R": 1.5,
                "trail_atr_mult": 1.0
            },
            "days": 60,
            "signal_engine": "hybrid",
            "leverage": 5,
            "max_loss_streak": 5
        }
        # Opcional: guardar la configuración por defecto para futuras ejecuciones
        with open(PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[BT] Default config saved to {PARAMS_FILE}")

    # Cargar modelo ML (si existe)
    load_ml_model()

    trades, stats = run_backtest(cfg)
    save_backtest_outputs(trades, stats)

    print(f"[BT] Trades: {0 if trades is None else len(trades)}")
    print(f"[BT] Stats: {stats}")
