import time, json, pickle
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ======================== BYBIT API (reemplaza a Binance) ========================
def fetch_bybit_klines_range(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> pd.DataFrame:
    """
    Descarga velas históricas de Bybit (spot) en un rango de tiempo.
    Intervalos soportados: 1m,3m,5m,15m,30m,1h,2h,4h,6h,12h,1d,1w,1M.
    """
    # Mapeo de intervalos
    interval_map = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
        "1d": "D", "1w": "W", "1M": "M"
    }
    bybit_interval = interval_map.get(interval)
    if not bybit_interval:
        raise ValueError(f"Intervalo {interval} no soportado por Bybit")

    all_klines = []
    cursor = None

    while True:
        url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={bybit_interval}&limit={limit}"
        if start_ms:
            url += f"&start={start_ms}"
        if end_ms:
            url += f"&end={end_ms}"
        if cursor:
            url += f"&cursor={cursor}"

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode") != 0:
            raise Exception(f"Bybit API error: {data.get('retMsg')}")

        result = data["result"]
        klines = result.get("list", [])
        if not klines:
            break

        # Bybit devuelve descendente (más reciente primero), lo invertimos para orden ascendente
        all_klines.extend(reversed(klines))

        cursor = result.get("nextPageCursor")
        if not cursor:
            break

        # Pequeña pausa para evitar rate limit
        time.sleep(0.1)

    if not all_klines:
        return pd.DataFrame()

    # Convertir a DataFrame
    df = pd.DataFrame(all_klines, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    # Orden ascendente (por si acaso)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df[["open_time", "open", "high", "low", "close", "volume"]]

# Alias para mantener compatibilidad con el código existente
fetch_klines = fetch_bybit_klines_range

# ======================== ML artifacts (sin cambios) ========================
ML_ARTIFACTS = {
    'model': None,
    'scaler': None,
    'feature_columns': None,
    'inv_label_mapping': None,
    'loaded': False
}

def load_ml_artifacts(model_path='btc_ml_model.pkl', scaler_path='scaler.pkl',
                      features_path='feature_columns.json', mapping_path='label_mapping.json'):
    """Carga los artefactos ML si existen."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(features_path, 'r') as f:
            feature_columns = json.load(f)
        with open(mapping_path, 'r') as f:
            label_mapping = json.load(f)
        inv_label_mapping = {v: k for k, v in label_mapping.items()}
        ML_ARTIFACTS.update({
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'inv_label_mapping': inv_label_mapping,
            'loaded': True
        })
        print("Artefactos ML cargados correctamente.")
        return True
    except Exception as e:
        print(f"No se pudieron cargar los artefactos ML: {e}. Usando modo legacy.")
        return False

def compute_ml_features(df):
    """Calcula exactamente las mismas características que se usaron en el entrenamiento."""
    features = pd.DataFrame(index=df.index)
    # Retornos
    features['ret_1'] = df['close'].pct_change(1)
    features['ret_3'] = df['close'].pct_change(3)
    features['ret_12'] = df['close'].pct_change(12)
    # Volatilidad realizada (ATR normalizado)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    features['realized_vol'] = atr / df['close']
    features['vol_ratio'] = atr / atr.rolling(96).mean()
    # Rango de precio
    features['range_pos'] = (df['high'] - df['low']) / df['close']
    # Pendiente lineal (20 barras)
    slope = []
    for i in range(20, len(df)):
        x = np.arange(20)
        y = df['close'].iloc[i-20:i].values
        slope.append(np.polyfit(x, y, 1)[0])
    slope = pd.Series(slope, index=df.index[20:])
    features['slope'] = slope / df['close'].shift(1)
    # Tendencia EMAs
    features['ema_fast'] = df['close'].ewm(span=20).mean()
    features['ema_slow'] = df['close'].ewm(span=50).mean()
    features['trend'] = (features['ema_fast'] - features['ema_slow']) / df['close']
    # Señal de ruptura (Donchian)
    donchian_high = df['high'].rolling(96).max()
    donchian_low = df['low'].rolling(96).min()
    features['breakout'] = np.where(df['close'] > donchian_high.shift(1), 1,
                                    np.where(df['close'] < donchian_low.shift(1), -1, 0))
    features.dropna(inplace=True)
    return features

# ======================== Funciones de indicadores (sin cambios) ========================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50.0)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.copy().set_index("open_time")
    o = d["open"].resample(rule).first()
    h = d["high"].resample(rule).max()
    l = d["low"].resample(rule).min()
    c = d["close"].resample(rule).last()
    v = d["volume"].resample(rule).sum()
    out = pd.DataFrame({"open_time": o.index, "open": o.values, "high": h.values, "low": l.values, "close": c.values, "volume": v.values})
    out = out.dropna().reset_index(drop=True)
    return out

def fee_equity_pct_from_slmove(risk_pct: float, fee_roundtrip: float, sl_move_pct: float) -> float:
    if sl_move_pct <= 1e-12:
        return 0.0
    return risk_pct * (fee_roundtrip / sl_move_pct)

def atr_percentile(df, atr_col='atr', window=100):
    """Calcula el percentil del último ATR dentro de una ventana."""
    atr_series = df[atr_col].tail(window)
    if len(atr_series) == 0:
        return 50.0
    current = atr_series.iloc[-1]
    pct = (atr_series < current).sum() / len(atr_series) * 100
    return pct

def get_dynamic_sl_mult(pct, config):
    vf = config.get('volatility_filter', {})
    high_pct = vf.get('atr_percentile_high', 80)
    low_pct = vf.get('atr_percentile_low', 20)
    mult_high = vf.get('sl_mult_high', 2.0)
    mult_low = vf.get('sl_mult_low', 1.0)
    if pct >= high_pct:
        return mult_high
    elif pct <= low_pct:
        return mult_low
    else:
        return 1.0

def classify_stop_exit(side: str, entry_px: float, stop_px: float, label_exits: bool = True, eps: float = 1e-6) -> str:
    if not label_exits:
        return "SL"
    if side == "LONG":
        if stop_px < entry_px - eps:
            return "SL"
        if abs(stop_px - entry_px) <= eps:
            return "BE"
        return "TRAIL"
    else:
        if stop_px > entry_px + eps:
            return "SL"
        if abs(stop_px - entry_px) <= eps:
            return "BE"
        return "TRAIL"

def simulate_trade_1m(df1m, entry_i, side, cfg, atr_value, sl_mult, tp_mult, label_exits):
    lev = float(cfg.get("leverage", 5))
    entry = float(df1m.at[entry_i, "close"])
    sl_move_pct = atr_value / entry * sl_mult
    tp_move_pct = atr_value / entry * tp_mult
    if side == "LONG":
        sl_price = entry * (1 - sl_move_pct)
        tp_price = entry * (1 + tp_move_pct)
    else:
        sl_price = entry * (1 + sl_move_pct)
        tp_price = entry * (1 - tp_move_pct)

    be_R = cfg.get("breakeven_R", 1.0)
    trail_start_R = cfg.get("trail_start_R", 1.8)
    trail_atr_k = cfg.get("trail_atr_mult", 0.8)
    max_hold_min = cfg.get("max_hold_minutes", 30)
    max_hold_bars = max_hold_min

    init_risk = abs(entry - sl_price) / entry
    if init_risk <= 1e-12:
        init_risk = 1e-9

    max_R = 0.0
    current_sl = sl_price
    best_price = entry   # para long: precio máximo; para short: precio mínimo

    for j in range(entry_i + 1, min(entry_i + 1 + max_hold_bars, len(df1m))):
        high = df1m.at[j, "high"]
        low = df1m.at[j, "low"]
        close = df1m.at[j, "close"]

        # Actualizar mejor precio
        if side == "LONG":
            best_price = max(best_price, high)
            cur_R = (best_price - entry) / init_risk
        else:
            best_price = min(best_price, low)
            cur_R = (entry - best_price) / init_risk
        max_R = max(max_R, cur_R)

        # Breakeven (solo si se ha superado be_R)
        if max_R >= be_R:
            if side == "LONG":
                current_sl = max(current_sl, entry)
            else:
                current_sl = min(current_sl, entry)

        # Trailing basado en mejor precio
        if max_R >= trail_start_R:
            if side == "LONG":
                trail_px = best_price - atr_value * trail_atr_k
                current_sl = max(current_sl, trail_px)
            else:
                trail_px = best_price + atr_value * trail_atr_k
                current_sl = min(current_sl, trail_px)

        # Verificar si se toca stop o take
        if side == "LONG":
            if low <= current_sl:
                exit_px = current_sl
                exit_reason = classify_stop_exit(side, entry, exit_px, label_exits)
                return {"exit_i": j, "exit_reason": exit_reason, "exit_px": exit_px, "max_R": max_R}
            if high >= tp_price:
                exit_px = tp_price
                exit_reason = "TP"
                return {"exit_i": j, "exit_reason": exit_reason, "exit_px": exit_px, "max_R": max_R}
        else:
            if high >= current_sl:
                exit_px = current_sl
                exit_reason = classify_stop_exit(side, entry, exit_px, label_exits)
                return {"exit_i": j, "exit_reason": exit_reason, "exit_px": exit_px, "max_R": max_R}
            if low <= tp_price:
                exit_px = tp_price
                exit_reason = "TP"
                return {"exit_i": j, "exit_reason": exit_reason, "exit_px": exit_px, "max_R": max_R}

    # Timeout
    exit_i = min(entry_i + max_hold_bars, len(df1m)-1)
    exit_px = df1m.at[exit_i, "close"]
    exit_reason = "TIME"
    return {"exit_i": exit_i, "exit_reason": exit_reason, "exit_px": exit_px, "max_R": max_R}


def run_backtest(cfg, progress_callback=None, log_callback=None):
    """
    Ejecuta el backtest de STRATUM con la configuración dada.
    Retorna (trades_df, stats_dict)

    Parámetros opcionales:
        progress_callback: función que recibe (porcentaje, mensaje)
        log_callback: función que recibe (mensaje)
    """
    # --- Determinar modo ---
    ml_enabled = cfg.get('mode') == 'ml_model'
    if ml_enabled:
        # Intentar cargar artefactos ML
        loaded = load_ml_artifacts()
        if not loaded:
            ml_enabled = False
            print("Falló carga ML, usando modo legacy")
    if ml_enabled:
        print("Modo ML activado")
    else:
        print("Modo legacy (indicadores técnicos)")

    sym = cfg["market_symbol"]
    days = int(cfg.get("days", 60))
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000

    # Descargar datos 1m (usando Bybit)
    if log_callback:
        log_callback("Descargando datos de 1m desde Bybit...")
    print("Descargando datos de 1m desde Bybit...")
    try:
        df1m = fetch_klines(sym, "1m", start_ms, now_ms)
    except Exception as e:
        if log_callback:
            log_callback(f"Error descargando datos: {e}")
        return pd.DataFrame(), {"error": f"Error descargando datos: {e}"}

    if log_callback:
        log_callback(f"Descargadas {len(df1m)} velas de 1m")
    print(f"Velas 1m: {len(df1m)}")
    if len(df1m) < 1000:
        if log_callback:
            log_callback("Datos insuficientes (menos de 1000 velas de 1m)")
        return pd.DataFrame(), {"error": "Datos insuficientes (menos de 1000 velas de 1m)"}

    # Resamplear a 5m
    if log_callback:
        log_callback("Resampleando a 5m...")
    print("Resampleando a 5m...")
    df5 = resample_ohlcv(df1m, "5min")
    if log_callback:
        log_callback(f"Resampleado a {len(df5)} velas de 5m")
    print(f"Velas 5m: {len(df5)}")

    # Indicadores en 5m (necesarios para legacy y también algunos para ML)
    if log_callback:
        log_callback("Calculando indicadores base...")
    print("Calculando indicadores base...")
    df5["ema_fast"] = ema(df5["close"], int(cfg.get("ema_fast", 20)))
    df5["ema_slow"] = ema(df5["close"], int(cfg.get("ema_slow", 50)))
    df5["rsi"] = rsi(df5["close"], int(cfg.get("rsi_len", 14)))
    df5["atr"] = atr(df5[["open","high","low","close"]], int(cfg.get("atr_len", 14)))
    df5["atr_ma"] = df5["atr"].rolling(int(cfg.get("atr_ma_window", 48))).mean()
    w = int(cfg.get("donchian_window", 20))
    df5["hh"] = df5["high"].rolling(w).max()
    df5["ll"] = df5["low"].rolling(w).min()

    # Parámetros de configuración (legacy)
    rsi_long = float(cfg.get("rsi_long", 55))
    rsi_short = float(cfg.get("rsi_short", 45))
    atr_min_mult = float(cfg.get("atr_min_mult", 0.8))
    buf_k = float(cfg.get("breakout_buffer_atr", 0.55))
    base_sl_mult = float(cfg.get("sl_atr_mult", 1.5))
    base_tp_mult = float(cfg.get("tp_atr_mult", 2.5))
    entry_window_bars = int(cfg.get("entry_window_1m_bars", 5))
    max_hold_min = int(cfg.get("max_hold_minutes", 30))
    cooldown_min = int(cfg.get("cooldown_minutes", 5))
    risk_pct = float(cfg.get("risk_per_trade_pct", 0.3))
    fees_enabled = bool(cfg.get("fees_enabled", True))
    fee_rt = float(cfg.get("fee_rate_roundtrip", 0.001))
    slip_rt = float(cfg.get("slippage_roundtrip_pct", 0.0))
    label_exits = bool(cfg.get("label_exits", True))
    tft_enabled = bool(cfg.get("tit_for_tat", {}).get("enabled", False))
    tft_loss_thresh = cfg.get("tit_for_tat", {}).get("loss_streak_threshold", 3)
    tft_reduction = cfg.get("tit_for_tat", {}).get("risk_reduction_factor", 0.5)
    tft_boost = cfg.get("tit_for_tat", {}).get("win_streak_boost", 1.2)

    trades = []
    cooldown_until = None
    loss_streak = 0
    win_streak = 0
    current_risk_pct = risk_pct

    warmup = max(int(cfg.get("ema_slow", 50)), w, int(cfg.get("atr_len", 14)), int(cfg.get("atr_ma_window", 48))) + 10

    # Mapa de tiempos para 1m (para localizar inicio de ventana)
    t1m_to_i = {t: i for i, t in enumerate(df1m["open_time"].dt.floor("1min").tolist())}

    total_bars = len(df5) - warmup
    processed = 0

    if log_callback:
        log_callback(f"Iniciando simulación de {total_bars} velas de 5m...")
    print(f"Procesando {total_bars} velas...")

    for idx5 in range(warmup, len(df5)):
        row5 = df5.iloc[idx5]
        t5 = row5["open_time"].floor("5min")

        # Cooldown
        if cooldown_until is not None and t5 <= cooldown_until:
            processed += 1
            continue

        # --- Señal: ML o Legacy ---
        if ml_enabled:
            # Necesitamos histórico mínimo para características (al menos 100 barras)
            if idx5 < 100:
                processed += 1
                continue
            # Tomar df5 hasta el índice actual
            df_hist = df5.iloc[:idx5+1].copy()
            features = compute_ml_features(df_hist)
            if features.empty:
                processed += 1
                continue
            # Seleccionar solo las columnas esperadas
            try:
                X = features.iloc[-1:][ML_ARTIFACTS['feature_columns']]
            except KeyError as e:
                print(f"Columna faltante en features: {e}")
                processed += 1
                continue
            # Escalar y predecir
            X_scaled = ML_ARTIFACTS['scaler'].transform(X)
            pred_raw = ML_ARTIFACTS['model'].predict(X_scaled)[0]  # 0,1,2
            signal = ML_ARTIFACTS['inv_label_mapping'][pred_raw]   # -1,0,1
            want_long = (signal == 1)
            want_short = (signal == -1)
        else:
            # Modo legacy: indicadores técnicos
            atr_ok = row5['atr'] >= row5['atr_ma'] * atr_min_mult
            trend_long = row5['ema_fast'] > row5['ema_slow']
            trend_short = row5['ema_fast'] < row5['ema_slow']
            rsi_val = row5['rsi']
            want_long = atr_ok and trend_long and (rsi_val >= rsi_long)
            want_short = atr_ok and trend_short and (rsi_val <= rsi_short)

        if not (want_long or want_short):
            processed += 1
            continue

        # Donchian previo (necesario para determinar el trigger)
        hh_prev = df5["hh"].iloc[idx5-1]
        ll_prev = df5["ll"].iloc[idx5-1]
        atr_v = float(row5["atr"])
        buffer = atr_v * buf_k

        if want_long:
            side = "LONG"
            trigger = hh_prev + buffer
        else:
            side = "SHORT"
            trigger = ll_prev - buffer

        # Ventana de entrada: después del cierre de la vela de 5m
        start_dt = t5 + pd.Timedelta(minutes=5)  # cierre de la vela de 5m
        # Buscar primera vela de 1m con open_time >= start_dt
        start_i = None
        for i in range(len(df1m)):
            if df1m.at[i, "open_time"] >= start_dt:
                start_i = i
                break
        if start_i is None or start_i >= len(df1m)-entry_window_bars:
            processed += 1
            continue

        # Buscar trigger en velas de 1m
        entry_i = None
        for i in range(start_i, start_i + entry_window_bars):
            high = df1m.at[i, "high"]
            low = df1m.at[i, "low"]
            if side == "LONG" and high >= trigger:
                entry_i = i
                break
            elif side == "SHORT" and low <= trigger:
                entry_i = i
                break

        if entry_i is None:
            processed += 1
            continue

        # Calcular multiplicadores dinámicos de SL
        atr_series_upto = df5['atr'].iloc[:idx5+1]
        if len(atr_series_upto) >= 200:
            pct = (atr_series_upto.tail(200) < atr_v).mean() * 100
        else:
            pct = 50.0
        dyn_mult = get_dynamic_sl_mult(pct, cfg)
        sl_mult = base_sl_mult * dyn_mult
        tp_mult = base_tp_mult  # también podría ser dinámico, pero por simplicidad lo dejamos fijo

        # Simular entrada y gestión
        sim = simulate_trade_1m(df1m, entry_i, side, cfg, atr_v, sl_mult, tp_mult, label_exits)
        entry_price = df1m.at[entry_i, "close"]
        exit_price = sim["exit_px"]

        # Cálculo de R y pnl
        init_risk = abs(entry_price - (entry_price * (1 - atr_v/entry_price * sl_mult))) / entry_price
        if init_risk <= 1e-12:
            init_risk = 1e-9
        if side == "LONG":
            move = (exit_price - entry_price) / entry_price
        else:
            move = (entry_price - exit_price) / entry_price
        R = move / init_risk
        pnl_equity = R * current_risk_pct

        # Comisiones y slippage
        if fees_enabled:
            sl_move_pct = atr_v / entry_price * sl_mult
            pnl_equity -= fee_equity_pct_from_slmove(current_risk_pct, fee_rt, sl_move_pct)
        if slip_rt > 0:
            sl_move_pct = atr_v / entry_price * sl_mult
            pnl_equity -= fee_equity_pct_from_slmove(current_risk_pct, slip_rt, sl_move_pct)

        # Registrar trade
        trades.append({
            "entry_time": df1m.at[entry_i, "open_time"],
            "exit_time": df1m.at[sim["exit_i"], "open_time"],
            "side": side,
            "exit_reason": sim["exit_reason"],
            "R": R,
            "pnl_equity_pct": pnl_equity,
            "max_R": sim["max_R"],
            "sl_mult": sl_mult,
            "atr_pct": pct
        })

        # Tit for tat: actualizar rachas y ajustar riesgo
        if pnl_equity > 0:
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0
        if tft_enabled:
            if loss_streak >= tft_loss_thresh:
                current_risk_pct = risk_pct * tft_reduction
            elif win_streak >= 2:
                current_risk_pct = risk_pct * tft_boost
            else:
                current_risk_pct = risk_pct

        # Cooldown
        cooldown_until = trades[-1]["exit_time"] + pd.Timedelta(minutes=cooldown_min)

        processed += 1
        # Reportar progreso cada 1% o cada 100 velas
        if progress_callback and processed % max(1, total_bars // 100) == 0:
            pct_progress = processed / total_bars * 100
            progress_callback(pct_progress, f"Procesando vela {processed}/{total_bars}")

    # Análisis de resultados
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        stats = {"trades": 0, "error": "No trades generados"}
        if log_callback:
            log_callback("No se generaron trades.")
        return trades_df, stats

    total = len(trades_df)
    winrate = (trades_df["pnl_equity_pct"] > 0).mean() * 100.0
    total_pnl = trades_df["pnl_equity_pct"].sum()
    avg_pnl = trades_df["pnl_equity_pct"].mean()
    curve = trades_df["pnl_equity_pct"].cumsum()
    max_dd = (curve - curve.cummax()).min()
    span_days = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).total_seconds() / (24*3600)
    tpd = total / max(span_days, 1e-9)
    proj_month = 30.0 * tpd * avg_pnl

    stats = {
        "trades": total,
        "winrate": winrate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "max_dd": max_dd,
        "tpd": tpd,
        "proj_month": proj_month,
        "exit_counts": trades_df["exit_reason"].value_counts().to_dict()
    }

    if log_callback:
        log_callback(f"Backtest finalizado: {total} trades generados.")
    return trades_df, stats


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    trades_df, stats = run_backtest(cfg)

    if trades_df.empty:
        print(f"\nError: {stats.get('error', 'No se generaron trades.')}")
        return

    total = stats["trades"]
    winrate = stats["winrate"]
    total_pnl = stats["total_pnl"]
    avg_pnl = stats["avg_pnl"]
    max_dd = stats["max_dd"]
    tpd = stats["tpd"]
    proj_month = stats["proj_month"]

    print("\n=== STRATUM BACKTEST SUMMARY ===")
    print(f"Trades: {total} | Trades/day: {tpd:.2f} | Winrate: {winrate:.1f}%")
    print(f"Total PnL (equity%): {total_pnl:.2f}% | Avg/trade: {avg_pnl:.3f}%")
    print(f"Max DD approx (equity%): {max_dd:.2f}%")
    print("\nExit reasons:")
    for reason, count in stats["exit_counts"].items():
        print(f"  {reason}: {count}")
    print(f"\nProjected monthly (linear): {proj_month:.2f}% equity/month")

    trades_df.to_csv("stratum_trades.csv", index=False, encoding="utf-8")
    print("\nSaved: stratum_trades.csv")


if __name__ == "__main__":
    main()
