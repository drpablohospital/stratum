# telegram_commands.py (versión final con /doom que muestra tabla de opciones)
import os
import json
import pandas as pd
import ccxt
import requests
import numpy as np
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# =========================
# Configuración
# =========================
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ALLOWED_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))

print(f"🤖 Bot de comandos Telegram iniciado...")
print(f"🔑 Token: {TELEGRAM_TOKEN[:5]}...{TELEGRAM_TOKEN[-5:] if TELEGRAM_TOKEN else 'NO CONFIGURADO'}")
print(f"📢 Chat ID permitido: {ALLOWED_CHAT_ID}")

# Eliminar webhook previo
try:
    resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook")
    print(f"✅ Webhook eliminado: {resp.json()}")
except Exception as e:
    print(f"⚠️ No se pudo eliminar webhook: {e}")

# =========================
# Funciones auxiliares (indicadores)
# =========================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi_series(series, length):
    delta = series.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def get_btc_volatility(days=365):
    """Obtiene la volatilidad diaria de BTC (desviación estándar de retornos logarítmicos)"""
    try:
        ex = ccxt.binanceusdm()
        ohlc = ex.fetch_ohlcv('BTC/USDT:USDT', '1d', limit=days)
        df = pd.DataFrame(ohlc, columns=['ts','open','high','low','close','vol'])
        df['close'] = df['close'].astype(float)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        vol = df['log_return'].std()
        return vol if not np.isnan(vol) else 0.02
    except Exception as e:
        print(f"Error calculando volatilidad: {e}")
        return 0.02

def liquidation_probability(stop_loss_pct, duration_days, daily_vol):
    """Calcula la probabilidad de que un movimiento adverso del stop_loss_pct ocurra en duration_days"""
    if duration_days <= 0:
        return 100.0
    period_vol = daily_vol * math.sqrt(duration_days)
    if period_vol <= 0:
        return 100.0
    threshold = (stop_loss_pct / 100.0) / period_vol
    prob = 0.5 * math.erfc(threshold / math.sqrt(2))
    return prob * 100

def find_max_duration(stop_pct, target_prob, daily_vol, max_days=30):
    """Encuentra la máxima duración (días) tal que la probabilidad de liquidación <= target_prob"""
    low, high = 0.01, max_days
    best = low
    for _ in range(50):
        mid = (low + high) / 2
        prob = liquidation_probability(stop_pct, mid, daily_vol)
        if prob <= target_prob:
            best = mid
            low = mid
        else:
            high = mid
    return best

def get_historical_stats():
    """Carga estadísticas desde pitonisa_trades.csv (o trades.csv) incluyendo duración media"""
    result = {"total": 0, "winrate": 0, "avg_pnl": 0, "max_dd": 0, "avg_duration_days": 0.25}
    try:
        df = None
        if os.path.exists("pitonisa_trades.csv"):
            df = pd.read_csv("pitonisa_trades.csv")
            if 'pnl_equity_pct' not in df.columns and 'pnl_pct' in df.columns:
                df['pnl_equity_pct'] = df['pnl_pct']
        elif os.path.exists("trades.csv"):
            df = pd.read_csv("trades.csv")
            if 'pnl_pct' in df.columns:
                df['pnl_equity_pct'] = df['pnl_pct']

        if df is not None:
            total_trades = len(df)
            win_trades = len(df[df['pnl_equity_pct'] > 0])
            winrate = win_trades / total_trades * 100 if total_trades > 0 else 0
            avg_pnl = df['pnl_equity_pct'].mean()
            curve = df['pnl_equity_pct'].cumsum()
            max_dd = (curve - curve.cummax()).min()

            avg_duration = 0.25
            if 'entry_time' in df.columns and 'exit_time' in df.columns:
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                df['exit_time'] = pd.to_datetime(df['exit_time'])
                durations = (df['exit_time'] - df['entry_time']).dt.total_seconds() / (24 * 3600)
                avg_duration = durations.mean()
            elif 'ts_utc' in df.columns and 'event' in df.columns:
                pass

            result.update({
                "total": total_trades,
                "winrate": winrate,
                "avg_pnl": avg_pnl,
                "max_dd": max_dd,
                "avg_duration_days": avg_duration if avg_duration > 0 else 0.25
            })
    except Exception as e:
        print(f"Error cargando estadísticas: {e}")
    return result

def get_current_indicators():
    """Obtiene indicadores actuales (ATR, RSI, tendencia) desde Binance"""
    ex = ccxt.binanceusdm()
    symbol = "BTC/USDT:USDT"
    ohlc = ex.fetch_ohlcv(symbol, '15m', limit=200)
    df = pd.DataFrame(ohlc, columns=['ts','open','high','low','close','vol'])
    for c in ['open','high','low','close']:
        df[c] = df[c].astype(float)

    ema_fast = 50
    ema_slow = 200
    rsi_len = 14
    atr_len = 14
    don_w = 20
    atr_min_mult = 1.25
    rsi_long = 58
    rsi_short = 42

    df["ema_fast"] = ema(df["close"], ema_fast)
    df["ema_slow"] = ema(df["close"], ema_slow)
    df["rsi"] = rsi_series(df["close"], rsi_len)
    df["atr"] = atr(df, atr_len)
    df["hh"] = df["high"].rolling(don_w).max()
    df["ll"] = df["low"].rolling(don_w).min()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    hh_prev = prev["hh"]
    ll_prev = prev["ll"]
    atr_val = last["atr"]
    rsi_val = last["rsi"]
    trend_long = last["ema_fast"] > last["ema_slow"]
    trend_short = last["ema_fast"] < last["ema_slow"]
    last_price = last["close"]

    atr_ma_series = df["atr"].rolling(96).mean()
    atr_ma = atr_ma_series.iloc[-1] if len(df) >= 96 else atr_val
    atr_ok = atr_val >= (atr_ma * atr_min_mult) if pd.notna(atr_ma) else True

    if trend_long and rsi_val >= rsi_long and atr_ok:
        signal = "LONG"
        trigger = hh_prev + (atr_val * 0.55)
        sl = trigger - atr_val * 1.1
        tp = trigger + atr_val * 3.2
    elif trend_short and rsi_val <= rsi_short and atr_ok:
        signal = "SHORT"
        trigger = ll_prev - (atr_val * 0.55)
        sl = trigger + atr_val * 1.1
        tp = trigger - atr_val * 3.2
    else:
        signal = "NEUTRO"
        trigger = sl = tp = 0.0

    return {
        "price": last_price,
        "atr": atr_val,
        "rsi": rsi_val,
        "trend": "ALCISTA" if trend_long else ("BAJISTA" if trend_short else "NEUTRA"),
        "signal": signal,
        "trigger": trigger,
        "sl": sl,
        "tp": tp,
        "atr_ok": atr_ok
    }

# =========================
# Manejadores de comandos
# =========================
async def echo_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"📩 Mensaje recibido de {update.effective_chat.id}: {update.message.text}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        await update.message.reply_text("No autorizado")
        return
    help_text = """
🤖 *AURUM - Estrategia Pitonisa*

*Comandos disponibles:*
/help - Muestra esta ayuda
/bet - Señal actual con probabilidad (modo casino)
/pro [rango] - Calcula SL/TP para un rango de precios (ej: /pro 75555-60000)
/doom [rango] [capital] [riesgo%] [tiempo_horas (opcional)] - Calcula apalancamiento máximo seguro (ej: /doom 60000-65000 100 5 24)
/about - Explicación básica de la estrategia
/advance - Detalles técnicos y rendimiento histórico
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        await update.message.reply_text("No autorizado")
        return
    text = """
🤖 *¿Qué es AURUM?*

AURUM es un ESTRATEGIA de trading automatizada para Bitcoin en Binance Futures. Opera con apalancamiento fijo (5x) y arriesga un pequeño porcentaje del capital por operación.

*🎯 Estrategia principal (Pitonisa)*
- Se basa en rupturas de máximos/mínimos de 20 velas (Donchian) en timeframe 15m.
- Filtros de tendencia (EMA 50/200 en 15m y 1h) para evitar contra-tendencia.
- Filtro de volatilidad (ATR) para operar solo cuando hay movimiento.
- Gestión de riesgo: stop loss inicial basado en ATR, breakeven al alcanzar 1R y trailing stop desde 1.5R.
- Toma de ganancias: TP fijo 3.2 veces el riesgo inicial.

*📊 Rendimiento histórico* (backtest)
(Consulta /advance para más detalles)
"""
    await update.message.reply_text(text, parse_mode='Markdown')

async def advance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        await update.message.reply_text("No autorizado")
        return
    stats = get_historical_stats()
    text = f"""
<b>📈 AURUM - Datos avanzados</b>

<b>📊 Backtest</b> (pitonisa_trades.csv)
• Trades totales: {stats['total']}
• Winrate: {stats['winrate']:.1f}%
• Promedio por trade: {stats['avg_pnl']:.3f}%
• Máximo drawdown: {stats['max_dd']:.2f}%
• Duración media: {stats['avg_duration_days']*24:.1f} horas

<b>⚙️ Parámetros clave</b>
• Apalancamiento: 5x
• Riesgo por trade: 1% del capital
• Timeframes: 15m (señal) y 5m (ejecución)
• Indicadores: EMA(50/200), RSI(14), ATR(14), Donchian(20)
• Distancias: SL = 1.1×ATR, TP = 3.2×ATR, buffer ruptura = 0.55×ATR
• Gestión: Breakeven en 1R, trailing desde 1.5R (1×ATR)

<b>📈 Proyección anual estimada</b> (lineal sobre backtest)
• Trades/día: ~{stats['total']/max(1, (datetime.now()-datetime(2023,1,1)).days)*30:.1f}
• PnL mensual proyectado: {stats['avg_pnl'] * (stats['total']/max(1, (datetime.now()-datetime(2023,1,1)).days))*30:.2f}%
"""
    await update.message.reply_text(text, parse_mode='HTML')

async def bet_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        await update.message.reply_text("No autorizado")
        return

    ind = get_current_indicators()
    stats = get_historical_stats()

    prob = stats['winrate']
    if ind['signal'] == "NEUTRO":
        prob = 0.0

    confidence = "BAJA"
    if ind['atr_ok'] and ind['signal'] != "NEUTRO":
        confidence = "MEDIA"
        if ind['rsi'] > 70 and ind['signal'] == "LONG":
            confidence = "ALTA (sobrecompra)"
        elif ind['rsi'] < 30 and ind['signal'] == "SHORT":
            confidence = "ALTA (sobreventa)"

    vol_text = "ALTA" if ind['atr'] > 300 else "MEDIA" if ind['atr'] > 200 else "BAJA"

    mensaje = f"""
🎲 *APUESTA AURUM* 🎲

📈 *Señal actual:* **{ind['signal']}**
📊 *Probabilidad (histórica):* {prob:.0f}%
🔥 *Confianza:* {confidence}

📉 *Indicadores:*
• Precio: ${ind['price']:,.0f}
• Volatilidad (ATR15): ${ind['atr']:.0f} ({vol_text})
• RSI(14): {ind['rsi']:.1f}
• Tendencia 15m: {ind['trend']}

{ "🎯 *Zona de ruptura:* " + (f"${ind['trigger']:,.0f}" if ind['trigger'] else "No aplica") if ind['signal'] != "NEUTRO" else "" }

*⏱️ Si entras ahora:*
• Usa stop loss según estrategia (consulta /pro)
• El tiempo promedio de una operación es ~3-6 horas
"""
    await update.message.reply_text(mensaje, parse_mode='Markdown')

async def pro_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        await update.message.reply_text("No autorizado")
        return

    args = context.args
    if len(args) != 1:
        await update.message.reply_text("Uso: /pro [rango]\nEjemplo: /pro 75555-60000")
        return

    rango = args[0]
    try:
        if '-' not in rango:
            raise ValueError
        high_str, low_str = rango.split('-')
        high = float(high_str.replace(',', ''))
        low = float(low_str.replace(',', ''))
    except:
        await update.message.reply_text("Formato inválido. Usa números separados por guión, ej: 75555-60000")
        return

    ind = get_current_indicators()
    atr = ind['atr']
    if atr <= 0:
        await update.message.reply_text("No se pudo obtener ATR. Reintenta más tarde.")
        return

    sl_k = 1.1
    tp_k = 3.2

    sl_short = high + atr * sl_k
    tp_short = high - atr * tp_k
    sl_long = low - atr * sl_k
    tp_long = low + atr * tp_k

    inv = 100
    lev = 5

    risk_short = abs(high - sl_short) / high * lev * 100
    reward_short = abs(high - tp_short) / high * lev * 100
    rr_short = reward_short / risk_short if risk_short > 0 else 0

    risk_long = abs(low - sl_long) / low * lev * 100
    reward_long = abs(low - tp_long) / low * lev * 100
    rr_long = reward_long / risk_long if risk_long > 0 else 0

    stats = get_historical_stats()
    duration_days = stats['avg_duration_days'] if stats['avg_duration_days'] > 0 else 0.25
    daily_vol = get_btc_volatility()

    short_sl_pct = abs(high - sl_short) / high * 100
    long_sl_pct = abs(low - sl_long) / low * 100

    prob_short = liquidation_probability(short_sl_pct, duration_days, daily_vol)
    prob_long = liquidation_probability(long_sl_pct, duration_days, daily_vol)

    mensaje = f"""
📊 *Rango ingresado:* ${high:,.0f} (SHORT) | ${low:,.0f} (LONG)

🔴 *SHORT* (entrada en ${high:,.0f})
• Stop Loss: ${sl_short:,.0f}
• Take Profit: ${tp_short:,.0f}
• Riesgo (5x, $100): {risk_short:.2f}%
• Ganancia potencial: {reward_short:.2f}%
• Ratio Riesgo/Beneficio: {rr_short:.2f}
• Probabilidad de liquidación: {prob_short:.2f}%

🟢 *LONG* (entrada en ${low:,.0f})
• Stop Loss: ${sl_long:,.0f}
• Take Profit: ${tp_long:,.0f}
• Riesgo (5x, $100): {risk_long:.2f}%
• Ganancia potencial: {reward_long:.2f}%
• Ratio Riesgo/Beneficio: {rr_long:.2f}
• Probabilidad de liquidación: {prob_long:.2f}%

*📈 Contexto actual*
• Volatilidad (ATR15): ${atr:.0f}
• Tendencia 15m: {ind['trend']}
• RSI(14): {ind['rsi']:.1f}
• Señal automática: {ind['signal']}
"""
    await update.message.reply_text(mensaje, parse_mode='Markdown')

# =========================
# Comando /doom mejorado con tabla de opciones
# =========================
async def doom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != ALLOWED_CHAT_ID:
        await update.message.reply_text("No autorizado")
        return

    args = context.args
    if len(args) < 3 or len(args) > 4:
        await update.message.reply_text(
            "Uso: /doom [rango] [capital_USDT] [riesgo_max%] [tiempo_horas (opcional)]\n"
            "Ejemplo: /doom 60000-65000 100 5 24\n"
            "Si no das tiempo, se muestra una tabla con opciones predefinidas."
        )
        return

    rango = args[0]
    try:
        capital = float(args[1])
        max_risk = float(args[2])
        duration_hours = float(args[3]) if len(args) == 4 else None
    except:
        await update.message.reply_text("Los parámetros deben ser números.")
        return

    # Parsear rango
    try:
        if '-' not in rango:
            raise ValueError
        low_str, high_str = rango.split('-')
        low = float(low_str.replace(',', ''))
        high = float(high_str.replace(',', ''))
        if low > high:
            low, high = high, low
        if high - low <= 0:
            raise ValueError
    except:
        await update.message.reply_text("Formato de rango inválido. Ej: 60000-65000")
        return

    # Datos necesarios
    daily_vol = get_btc_volatility()
    stats = get_historical_stats()
    avg_duration_hours = stats['avg_duration_days'] * 24

    favorable_move_pct = (high - low) / low * 100

    # Si se proporciona tiempo, calcular directamente
    if duration_hours is not None:
        duration_days = duration_hours / 24
        max_leverage = 1
        best_prob = 100.0
        for L in range(1, 101):
            liq_dist = 100 / L
            prob = liquidation_probability(liq_dist, duration_days, daily_vol)
            if prob <= max_risk:
                max_leverage = L
                best_prob = prob
            else:
                break

        if max_leverage == 1 and best_prob > max_risk:
            await update.message.reply_text(
                f"⚠️ Incluso con 1x la probabilidad de liquidación ({best_prob:.1f}%) "
                f"supera el riesgo máximo ({max_risk}%). Considera ampliar el rango o reducir el tiempo."
            )
            return

        liq_dist = 100 / max_leverage
        max_duration_days = find_max_duration(liq_dist, max_risk, daily_vol, max_days=30)
        max_duration_hours = max_duration_days * 24

        gain_pct = favorable_move_pct * max_leverage
        gain_usd = capital * (gain_pct / 100)

        if best_prob <= 5:
            risk_label = "🟢 BAJO"
        elif best_prob <= 10:
            risk_label = "🟡 MODERADO"
        elif best_prob <= 20:
            risk_label = "🟠 ALTO"
        else:
            risk_label = "🔴 MUY ALTO (¡DOOM!)"

        mensaje = f"""
💀 *DOOM - Calculadora de Apalancamiento* 💀

📊 *Rango:* ${low:,.0f} - ${high:,.0f}
💰 *Capital:* ${capital:.2f} USDT
🎯 *Riesgo máximo aceptado:* {max_risk}%
⏱️ *Tiempo de operación:* {duration_hours:.1f} horas

📐 *Movimiento favorable:* {favorable_move_pct:.2f}%
⚡ *Apalancamiento máximo seguro:* **{max_leverage}x**
📉 *Probabilidad de liquidación:* {best_prob:.2f}%
📈 *Riesgo:* {risk_label}

🔒 *Tiempo máximo seguro:* **{max_duration_hours:.1f} horas**
   (con este apalancamiento puedes mantener la posición hasta ese tiempo sin superar el riesgo)

🚀 *Si el precio recorre el rango a tu favor:*
• Ganancia potencial: {gain_pct:.2f}% sobre capital → **${gain_usd:,.2f}**
• ROI: {gain_pct:.2f}%

*Nota:* El cálculo se basa en la volatilidad histórica de BTC ({daily_vol*100:.1f}% diaria). El apalancamiento máximo evita la liquidación mientras el precio se mantenga dentro del rango. ¡Opera con cuidado!
"""
        await update.message.reply_text(mensaje, parse_mode='Markdown')
        return

    # Si no se especificó tiempo, mostrar tabla con opciones
    opciones = [
        ("1 hora", 1),
        ("6 horas", 6),
        ("1 día", 24),
        ("2 días", 48),
        ("1 semana", 168),
        (f"Media histórica ({avg_duration_hours:.1f} h)", avg_duration_hours)
    ]

    # Construir la tabla
    tabla = f"📊 *Opciones para {capital:.0f} USDT, riesgo ≤ {max_risk}%*\n\n"
    tabla += "⏱️ *Horizonte*      🔢 *Leverage*   📉 *Probabilidad*   💰 *Ganancia*\n"
    tabla += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    for nombre, horas in opciones:
        if horas <= 0:
            continue
        dur_days = horas / 24
        best = 1
        best_prob = 100.0
        for L in range(1, 101):
            liq = 100 / L
            prob = liquidation_probability(liq, dur_days, daily_vol)
            if prob <= max_risk:
                best = L
                best_prob = prob
            else:
                break
        if best == 1 and best_prob > max_risk:
            # Incluso 1x supera el riesgo
            ganancia_pct = favorable_move_pct * 1
            ganancia_usd = capital * (ganancia_pct / 100)
            tabla += f"{nombre:<12}   {best:>3}x        ⚠️ {best_prob:.1f}%   ${ganancia_usd:,.0f}\n"
        else:
            ganancia_pct = favorable_move_pct * best
            ganancia_usd = capital * (ganancia_pct / 100)
            tabla += f"{nombre:<12}   {best:>3}x        {best_prob:>5.1f}%   ${ganancia_usd:,.0f}\n"

    tabla += f"\n🎯 *Movimiento favorable:* {favorable_move_pct:.2f}%"
    tabla += f"\n📈 *Volatilidad diaria:* {daily_vol*100:.1f}%"
    tabla += f"\n\n_Elige el horizonte que mejor se adapte a tu estrategia y usa /doom [rango] [capital] [riesgo%] [horas]_"

    await update.message.reply_text(tabla, parse_mode='Markdown')

# =========================
# Main
# =========================
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_all))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(CommandHandler("advance", advance_command))
    app.add_handler(CommandHandler("bet", bet_command))
    app.add_handler(CommandHandler("pro", pro_command))
    app.add_handler(CommandHandler("doom", doom_command))

    print("🤖 Bot de comandos Telegram iniciado (modo completo). Esperando mensajes...")
    app.run_polling()

if __name__ == "__main__":
    main()
