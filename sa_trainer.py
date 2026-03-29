"""
train_multi_tf_gpu.py
Entrena modelo ML con datos de múltiples timeframes (1m, 15m, 30m, 1h, 1d, 1w)
usando 10 años de histórico. Combina características de todos los timeframes.
Aprovecha GPU (NVIDIA) para XGBoost.
Guarda artefactos compatibles con backtester.
"""

import os
import sys
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# ======================================================
# CONFIGURACIÓN
# ======================================================

SYMBOL = "BTCUSDT"
BASE_TIMEFRAME = "1h"          # marco base para combinar todas las características
HORIZON_HOURS = 12              # horizonte de predicción en horas (para etiquetar)
THRESHOLD_PCT = 0.75            # retorno mínimo para señal (porcentaje)
PERSISTENCE_BARS = 2            # persistencia de la señal (no usado en entrenamiento pero se guarda)

# Timeframes a descargar y usar (todos los que quieras)
TIMEFRAMES = ["1m", "15m", "30m", "1h", "1d", "1w"]

# Años de histórico
YEARS = 10

# Modelo a entrenar: 'xgboost' (recomendado con GPU) o 'random_forest'
MODEL_TYPE = "xgboost"

# Configuración de GPU para XGBoost (si está disponible)
XGB_GPU_PARAMS = {
    'tree_method': 'gpu_hist',      # usa GPU
    'predictor': 'gpu_predictor',   # predicción en GPU
    'gpu_id': 0,                    # primera GPU
    'n_jobs': 1,                    # evitar conflictos de hilos
}

# ======================================================
# FUNCIONES DE DESCARGA
# ======================================================

def fetch_binance_klines(symbol, interval, start_date, end_date, max_retries=3):
    """
    Descarga histórico completo de Binance para un intervalo.
    """
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    all_klines = []
    limit = 1000

    with tqdm(total=None, desc=f"Descargando {interval}", unit="batch") as pbar:
        while start_ts < end_ts:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_ts}&limit={limit}"
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

    # Convertir a DataFrame
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
    df = pd.DataFrame(all_klines, columns=cols)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    # Optimizar tipos para ahorrar memoria
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype('float32')
    return df

def load_or_download_data(interval, start_date, end_date):
    """
    Carga datos de un intervalo desde archivo CSV, o los descarga.
    """
    fname = f"btc_{interval}_{YEARS}y.parquet"  # usar parquet es más rápido y compacto
    if os.path.exists(fname):
        print(f"Cargando {fname}...")
        df = pd.read_parquet(fname)
        return df
    else:
        print(f"Descargando {interval} ({YEARS} años)...")
        df = fetch_binance_klines(SYMBOL, interval, start_date, end_date)
        if not df.empty:
            df.to_parquet(fname, compression='snappy')
            print(f"Guardado {fname}")
        return df

# ======================================================
# FUNCIONES DE FEATURE ENGINEERING (por intervalo)
# ======================================================

def compute_features_tf(df, tf_label):
    """
    Calcula características para un DataFrame en un timeframe específico.
    Las ventanas se expresan en número de barras, pero representan diferentes
    períodos de tiempo según el timeframe.
    """
    features = pd.DataFrame(index=df.index, dtype='float32')

    # Retornos (1, 3, 12 barras)
    features['ret_1'] = df['close'].pct_change(1).astype('float32')
    features['ret_3'] = df['close'].pct_change(3).astype('float32')
    features['ret_12'] = df['close'].pct_change(12).astype('float32')

    # Volatilidad realizada (ATR normalizado)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    features['realized_vol'] = (atr / df['close']).astype('float32')
    features['vol_ratio'] = (atr / atr.rolling(96).mean()).astype('float32')

    # Rango de precio
    features['range_pos'] = ((df['high'] - df['low']) / df['close']).astype('float32')

    # Pendiente lineal (20 barras) - optimizado con rolling apply
    def slope_series(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    # Usamos rolling con función, puede ser lento pero es aceptable
    features['slope'] = df['close'].rolling(20).apply(slope_series, raw=True).astype('float32')
    features['slope'] = features['slope'] / df['close'].shift(1)

    # Tendencia EMAs
    features['ema_fast'] = df['close'].ewm(span=20).mean().astype('float32')
    features['ema_slow'] = df['close'].ewm(span=50).mean().astype('float32')
    features['trend'] = ((features['ema_fast'] - features['ema_slow']) / df['close']).astype('float32')

    # Señal de ruptura Donchian
    donchian_high = df['high'].rolling(96).max()
    donchian_low = df['low'].rolling(96).min()
    features['breakout'] = np.where(df['close'] > donchian_high.shift(1), 1,
                                    np.where(df['close'] < donchian_low.shift(1), -1, 0)).astype('int8')

    # Agregar prefijo al nombre de cada columna según el timeframe
    features = features.add_prefix(f"{tf_label}_")
    features.dropna(inplace=True)
    return features

# ======================================================
# COMBINACIÓN DE CARACTERÍSTICAS
# ======================================================

def resample_to_base(features, base_freq):
    """
    Resamplea un DataFrame de características a la frecuencia base (forward fill).
    """
    return features.resample(base_freq).ffill()

def combine_features(interval_data, base_freq):
    """
    Recibe diccionario con DataFrames de cada intervalo (sin características aún),
    calcula características para cada uno, las resamplea a la frecuencia base
    y las concatena.
    """
    combined = None
    for tf, df in interval_data.items():
        print(f"Calculando características para {tf}...")
        feats = compute_features_tf(df, tf)
        if feats.empty:
            continue
        # Resamplear a base_freq
        feats = resample_to_base(feats, base_freq)
        if combined is None:
            combined = feats
        else:
            # Unir con inner join para mantener solo tiempos comunes
            combined = combined.join(feats, how='inner')
    return combined

# ======================================================
# ETIQUETAS
# ======================================================

def create_labels_base(df_base, horizon_hours, threshold_pct):
    """
    Crea etiquetas en la frecuencia base (BASE_TIMEFRAME) usando horizonte en horas.
    """
    # Convertir horizonte a número de barras en base_freq
    freq_min = pd.to_timedelta(BASE_TIMEFRAME).total_seconds() / 60
    horizon_bars = int(horizon_hours * 60 / freq_min)

    forward_ret = df_base['close'].shift(-horizon_bars) / df_base['close'] - 1
    labels = np.zeros(len(df_base), dtype='int8')
    labels[forward_ret > threshold_pct/100] = 1
    labels[forward_ret < -threshold_pct/100] = -1
    return pd.Series(labels, index=df_base.index, name='label')

# ======================================================
# ENTRENAMIENTO
# ======================================================

def train_model(X_train, y_train, X_test, y_test, model_type='xgboost'):
    # Mapear etiquetas -1,0,1 a 0,1,2
    label_mapping = {-1: 0, 0: 1, 1: 2}
    y_train_mapped = y_train.map(label_mapping).values
    y_test_mapped = y_test.map(label_mapping).values

    if model_type == 'xgboost':
        import xgboost as xgb
        # Parámetros base + GPU
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'verbosity': 1,
        }
        # Añadir parámetros GPU si están disponibles
        try:
            # Comprobar si GPU está disponible
            import xgboost as xgb_test
            if xgb_test.__version__ >= '1.0.0':
                params.update(XGB_GPU_PARAMS)
                print("Usando GPU para XGBoost")
            else:
                print("Versión de XGBoost antigua, usando CPU")
        except:
            print("GPU no disponible, usando CPU")

        model = xgb.XGBClassifier(**params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Modelo {model_type} no soportado.")

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar
    model.fit(X_train_scaled, y_train_mapped)

    # Evaluar
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
    print("=== Entrenamiento multi‑timeframe para BTCUSDT con GPU ===\n")
    # Fechas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=YEARS*365)

    # Descargar datos para cada intervalo
    interval_data = {}
    for tf in TIMEFRAMES:
        print(f"\n--- {tf} ---")
        df = load_or_download_data(tf, start_date, end_date)
        if not df.empty:
            interval_data[tf] = df
        else:
            print(f"Advertencia: No se obtuvieron datos para {tf}")

    if not interval_data:
        print("No se descargaron datos. Abortando.")
        return

    # Combinar características en frecuencia base
    print(f"\nCombinando características a frecuencia {BASE_TIMEFRAME}...")
    features = combine_features(interval_data, BASE_TIMEFRAME)
    print(f"Características totales: {features.shape}")

    # Obtener DataFrame base (para etiquetas)
    if BASE_TIMEFRAME not in interval_data:
        df_base = load_or_download_data(BASE_TIMEFRAME, start_date, end_date)
    else:
        df_base = interval_data[BASE_TIMEFRAME]

    # Alinear índices
    common_idx = features.index.intersection(df_base.index)
    features = features.loc[common_idx]
    df_base = df_base.loc[common_idx]

    # Crear etiquetas
    print("Creando etiquetas...")
    labels = create_labels_base(df_base, HORIZON_HOURS, THRESHOLD_PCT)

    # Alinear features y labels
    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    print(f"Características: {features.shape[1]}, muestras: {features.shape[0]}")
    print("Distribución de etiquetas:")
    print(labels.value_counts())

    # División temporal (80% entrenamiento, 20% prueba)
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = labels.iloc[:split_idx], labels.iloc[split_idx:]

    # Entrenar modelo
    print(f"\nEntrenando modelo {MODEL_TYPE}...")
    model, scaler, label_mapping = train_model(X_train, y_train, X_test, y_test, MODEL_TYPE)

    # Guardar artefactos
    with open('btc_ml_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)
    with open('feature_columns.json', 'w') as f:
        json.dump(list(features.columns), f)

    print("\n✅ Modelo y artefactos guardados: btc_ml_model.pkl, scaler.pkl, label_mapping.json, feature_columns.json")
    print("Ahora puedes usarlos en tu backtester (config.json con mode='ml_model')")

if __name__ == "__main__":
    main()
