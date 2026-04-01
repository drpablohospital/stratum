"""
sa_trainer_enhanced.py - Entrenamiento multi-timeframe para BTCUSDT con GPU
Mejorado con datos externos y más timeframes
"""

import os
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# ======================================================
# CONFIGURACIÓN
# ======================================================

SYMBOL = "BTCUSDT"
BASE_TIMEFRAME = "1h"          # marco base para combinar características
HORIZON_HOURS = 12             # horizonte de predicción en horas
THRESHOLD_PCT = 0.75           # retorno mínimo para señal (%)
PERSISTENCE_BARS = 2           # persistencia (no usado en entrenamiento)

# Timeframes a utilizar
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "1d", "1w"]

# Años de histórico (desde 2017, fecha más antigua de Binance)
START_YEAR = 2017

# Modelo: 'xgboost' o 'random_forest'
MODEL_TYPE = "xgboost"

# Parámetros XGBoost con GPU (para versión >= 2.0)
XGB_GPU_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',
    'n_jobs': 1,
}

# Parámetros adicionales para XGBoost
# Importante:
# early_stopping_rounds va aquí en algunas versiones modernas de xgboost,
# no en model.fit(...)
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': 3,
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 50,
}

# Símbolos de Yahoo Finance para índices y activos relacionados
EXTERNAL_SYMBOLS = {
    'sp500': '^GSPC',
    'nasdaq': '^IXIC',
    'dow': '^DJI',
    'vix': '^VIX',
    'nvda': 'NVDA',
}

# ======================================================
# FUNCIONES DE DESCARGA (Binance)
# ======================================================

def fetch_binance_klines(symbol, interval, start_date, end_date, max_retries=3):
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    all_klines = []
    limit = 1000

    with tqdm(total=None, desc=f"Descargando {interval}", unit="batch") as pbar:
        while start_ts < end_ts:
            url = (
                f"https://api.binance.com/api/v3/klines"
                f"?symbol={symbol}&interval={interval}&startTime={start_ts}&limit={limit}"
            )

            for _ in range(max_retries):
                try:
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    print(f"Error en {interval}, start={start_ts}: {e}. Reintentando...")
                    time.sleep(2)
            else:
                print(f"Fallo después de {max_retries} reintentos en {interval}. Abortando descarga.")
                break

            if not data:
                break

            all_klines.extend(data)
            last_open = data[-1][0]
            start_ts = last_open + 1
            pbar.update(1)

            if len(data) < limit:
                break

            time.sleep(0.1)

    cols = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_vol', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ]

    df = pd.DataFrame(all_klines, columns=cols)

    if df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype('float32')

    return df


def load_or_download_data(interval, start_date, end_date):
    fname = f"btc_{interval}_{START_YEAR}_{end_date.year}y.parquet"

    if os.path.exists(fname):
        print(f"Cargando {fname}...")
        df = pd.read_parquet(fname)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

    print(f"Descargando {interval} ({start_date.date()} a {end_date.date()})...")
    df = fetch_binance_klines(SYMBOL, interval, start_date, end_date)

    if not df.empty:
        df.to_parquet(fname, compression='snappy')
        print(f"Guardado {fname}")

    return df


# ======================================================
# DATOS EXTERNOS (Yahoo Finance)
# ======================================================

def fetch_external_data(start_date, end_date):
    """Descarga datos de índices y activos relacionados desde Yahoo Finance."""
    data_dict = {}

    for name, ticker in EXTERNAL_SYMBOLS.items():
        print(f"Descargando {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                print(f"Advertencia: No se obtuvieron datos para {ticker}")
                continue

            cols = ['Close']
            if 'Volume' in df.columns:
                cols.append('Volume')

            df = df[cols].copy()

            if len(cols) > 1:
                df.columns = [f"{name}_close", f"{name}_volume"]
            else:
                df.columns = [f"{name}_close"]

            data_dict[name] = df

        except Exception as e:
            print(f"Error descargando {ticker}: {e}")

    return data_dict


def resample_external_to_base(external_data, base_freq):
    """Resamplea datos externos al timeframe base."""
    aligned = []

    for _, df in external_data.items():
        df_resampled = df.resample(base_freq).last()
        aligned.append(df_resampled)

    if aligned:
        return pd.concat(aligned, axis=1)

    return pd.DataFrame()


# ======================================================
# FEATURE ENGINEERING POR TIMEFRAME
# ======================================================

def compute_features_tf(df, tf_label):
    if df.empty or len(df) < 120:
        return pd.DataFrame()

    features = pd.DataFrame(index=df.index, dtype='float32')

    # Retornos
    features['ret_1'] = df['close'].pct_change(1).astype('float32')
    features['ret_3'] = df['close'].pct_change(3).astype('float32')
    features['ret_12'] = df['close'].pct_change(12).astype('float32')

    # ATR y volatilidad
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()
    features['realized_vol'] = (atr / df['close']).astype('float32')
    features['vol_ratio'] = (atr / atr.rolling(96).mean()).astype('float32')

    # Rango normalizado
    features['range_pos'] = ((df['high'] - df['low']) / df['close']).astype('float32')

    # Slope de media móvil
    def slope_series(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    features['slope'] = df['close'].rolling(20).apply(slope_series, raw=True).astype('float32')
    features['slope'] = (features['slope'] / df['close'].shift(1)).astype('float32')

    # EMA y tendencia
    features['ema_fast'] = df['close'].ewm(span=20, adjust=False).mean().astype('float32')
    features['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean().astype('float32')
    features['trend'] = ((features['ema_fast'] - features['ema_slow']) / df['close']).astype('float32')

    # Donchian
    donchian_high = df['high'].rolling(96).max()
    donchian_low = df['low'].rolling(96).min()
    features['breakout'] = np.where(
        df['close'] > donchian_high.shift(1), 1,
        np.where(df['close'] < donchian_low.shift(1), -1, 0)
    ).astype('int8')

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    features['rsi'] = (100 - 100 / (1 + rs)).fillna(50).astype('float32')

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    features['macd'] = (ema12 - ema26).astype('float32')
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean().astype('float32')
    features['macd_hist'] = (features['macd'] - features['macd_signal']).astype('float32')

    # Bandas de Bollinger
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    features['bb_upper'] = (sma20 + 2 * std20).astype('float32')
    features['bb_lower'] = (sma20 - 2 * std20).astype('float32')
    features['bb_width'] = ((features['bb_upper'] - features['bb_lower']) / sma20).astype('float32')

    features = features.add_prefix(f"{tf_label}_")
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)

    return features


# ======================================================
# COMBINACIÓN DE CARACTERÍSTICAS
# ======================================================

def resample_to_base(features, base_freq):
    return features.resample(base_freq).ffill()


def combine_features(interval_data, base_freq, external_features):
    combined = None

    for tf, df in interval_data.items():
        print(f"Calculando características para {tf}...")
        feats = compute_features_tf(df, tf)

        if feats.empty:
            continue

        feats = resample_to_base(feats, base_freq)

        if combined is None:
            combined = feats
        else:
            combined = combined.join(feats, how='outer')

    if external_features is not None and not external_features.empty:
        if combined is None:
            combined = external_features
        else:
            combined = combined.join(external_features, how='outer')

    if combined is None:
        return pd.DataFrame()

    combined = combined.sort_index()
    combined = combined.ffill()
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)

    return combined


# ======================================================
# ETIQUETAS
# ======================================================

def create_labels_base(df_base, horizon_hours, threshold_pct):
    freq_min = pd.to_timedelta(BASE_TIMEFRAME).total_seconds() / 60
    horizon_bars = int(horizon_hours * 60 / freq_min)

    forward_ret = df_base['close'].shift(-horizon_bars) / df_base['close'] - 1

    labels = np.zeros(len(df_base), dtype='int8')
    labels[forward_ret > threshold_pct / 100] = 1
    labels[forward_ret < -threshold_pct / 100] = -1

    return pd.Series(labels, index=df_base.index, name='label')


# ======================================================
# ENTRENAMIENTO
# ======================================================

def train_model(X_train, y_train, X_test, y_test, model_type='xgboost'):
    label_mapping = {-1: 0, 0: 1, 1: 2}
    y_train_mapped = y_train.map(label_mapping).values
    y_test_mapped = y_test.map(label_mapping).values

    if model_type == 'xgboost':
        import xgboost as xgb

        print(f"Versión de xgboost detectada: {xgb.__version__}")

        params = XGB_PARAMS.copy()

        if xgb.__version__ >= '2.0.0':
            params.update(XGB_GPU_PARAMS)
            print("Usando GPU con device='cuda'")
        else:
            params.update({
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': 0,
                'n_jobs': 1,
            })
            print("Usando GPU con método antiguo (gpu_hist)")

        model = xgb.XGBClassifier(**params)

    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Modelo {model_type} no soportado.")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'xgboost':
        # Importante:
        # En tu entorno, early_stopping_rounds NO debe pasarse a fit().
        # Ya va dentro de XGB_PARAMS.
        eval_set = [(X_test_scaled, y_test_mapped)]

        model.fit(
            X_train_scaled,
            y_train_mapped,
            eval_set=eval_set,
            verbose=False
        )
    else:
        model.fit(X_train_scaled, y_train_mapped)

    preds_mapped = model.predict(X_test_scaled)
    inv_map = {v: k for k, v in label_mapping.items()}
    preds = np.array([inv_map[p] for p in preds_mapped])

    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['SHORT', 'NEUTRAL', 'LONG']))

    return model, scaler, label_mapping


# ======================================================
# MAIN
# ======================================================

def main():
    print("=== Entrenamiento multi-timeframe para BTCUSDT con GPU (mejorado) ===\n")

    end_date = datetime.now()
    start_date = datetime(START_YEAR, 1, 1)

    # 1. Descargar datos BTC
    interval_data = {}
    for tf in TIMEFRAMES:
        print(f"\n--- {tf} ---")
        df = load_or_download_data(tf, start_date, end_date)
        if not df.empty:
            interval_data[tf] = df
        else:
            print(f"Advertencia: No se obtuvieron datos para {tf}")

    if not interval_data:
        print("No se descargaron datos de BTC. Abortando.")
        return

    # 2. Descargar datos externos
    print("\n--- Descargando datos externos ---")
    external_raw = fetch_external_data(start_date, end_date)
    external_features = resample_external_to_base(external_raw, BASE_TIMEFRAME)
    print(f"Características externas: {external_features.shape if external_features is not None else 'None'}")

    # 3. Combinar características
    print(f"\nCombinando características a frecuencia {BASE_TIMEFRAME}...")
    features = combine_features(interval_data, BASE_TIMEFRAME, external_features)

    if features.empty:
        print("No se pudieron construir características. Abortando.")
        return

    print(f"Características totales: {features.shape}")

    # 4. Datos base para etiquetas
    if BASE_TIMEFRAME not in interval_data:
        df_base = load_or_download_data(BASE_TIMEFRAME, start_date, end_date)
    else:
        df_base = interval_data[BASE_TIMEFRAME]

    if df_base.empty:
        print("No se obtuvieron datos base para crear etiquetas. Abortando.")
        return

    common_idx = features.index.intersection(df_base.index)
    features = features.loc[common_idx].copy()
    df_base = df_base.loc[common_idx].copy()

    print("Creando etiquetas...")
    labels = create_labels_base(df_base, HORIZON_HOURS, THRESHOLD_PCT)

    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx].copy()
    labels = labels.loc[common_idx].copy()

    # Quitar últimas filas con horizonte incompleto
    valid_mask = labels.notna()
    features = features.loc[valid_mask]
    labels = labels.loc[valid_mask]

    print(f"Características: {features.shape[1]}, muestras: {features.shape[0]}")
    print("Distribución de etiquetas:")
    print(labels.value_counts())

    if len(features) < 1000:
        print("Muy pocas muestras para entrenar con seguridad. Abortando.")
        return

    # 5. Split temporal 80/20
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = labels.iloc[:split_idx], labels.iloc[split_idx:]

    # 6. Entrenamiento
    print(f"\nEntrenando modelo {MODEL_TYPE}...")
    model, scaler, label_mapping = train_model(
        X_train, y_train, X_test, y_test, MODEL_TYPE
    )

    # 7. Guardar artefactos
    with open('btc_ml_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('label_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2)

    with open('feature_columns.json', 'w', encoding='utf-8') as f:
        json.dump(list(features.columns), f, indent=2)

    config = {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'timeframes': TIMEFRAMES,
        'base_timeframe': BASE_TIMEFRAME,
        'horizon_hours': HORIZON_HOURS,
        'threshold_pct': THRESHOLD_PCT,
        'model_type': MODEL_TYPE,
        'external_symbols': EXTERNAL_SYMBOLS,
    }

    with open('training_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print("\n✅ Modelo y artefactos guardados:")
    print("   btc_ml_model.pkl")
    print("   scaler.pkl")
    print("   label_mapping.json")
    print("   feature_columns.json")
    print("   training_config.json")
    print("\nAhora puedes usarlos en tu backtester (config.json con mode='ml_model').")


if __name__ == "__main__":
    main()
