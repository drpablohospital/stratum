# oracle.py
# Módulo de notificaciones para STRATUM Trading Bot
# Envía mensajes a Telegram sobre señales, entradas, salidas y heartbeats

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

print(f"[ORACLE] Token: {TELEGRAM_TOKEN[:5] if TELEGRAM_TOKEN else 'None'}...")
print(f"[ORACLE] Chat ID: {TELEGRAM_CHAT_ID}")

# ============================================
# FUNCIÓN PRINCIPAL DE ENVÍO
# ============================================
def send_telegram(message):
    """
    Envía un mensaje a Telegram
    Args:
        message (str): Mensaje a enviar (puede incluir HTML)
    Returns:
        bool: True si se envió correctamente, False si no
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram no configurado - salta mensaje")
        print(f"Mensaje no enviado:\n{message}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID.strip(),  # Importante: sin espacios
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("[OK] Telegram enviado")
            return True
        else:
            print(f"[ERR] Error Telegram {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"[ERR] Excepción enviando a Telegram: {e}")
        return False


# ============================================
# FORMATO DE MENSAJES
# ============================================

def format_signal(side, trigger, sl, tp, atr, note=""):
    """
    Formatea mensaje cuando se ARMA una señal
    """
    side_emoji = "🟢" if side == "LONG" else "🔴"

    return f"""
🚨 <b>🔮 SEÑAL STRATUM DETECTADA</b> 🚨

<b>{side_emoji} {side} LISTO PARA ROMPER</b>
━━━━━━━━━━━━━━━━━━
🎯 <b>Trigger:</b> ${trigger:,.2f}
🛡️ <b>Stop Loss:</b> ${sl:,.2f}
💰 <b>Take Profit:</b> ${tp:,.2f}
📊 <b>ATR(15):</b> ${atr:,.2f}

<i>⏰ {datetime.now().strftime('%H:%M:%S')} UTC</i>
{('📝 ' + note) if note else ''}
"""


def format_entry(side, price, qty, equity, trigger=None, sl_price=None):
    """
    Formatea mensaje cuando se ABRE una posición
    Incluye ejemplos con diferentes apalancamientos para $100
    """
    side_emoji = "🟢" if side == "LONG" else "🔴"

    # Calcular riesgo para ejemplos
    risk_dist = abs(price - sl_price) / price if sl_price and sl_price > 0 else 0.02

    # Ejemplos con $100 USD
    inv_base = 100
    examples = []
    for lev_ex in [5, 15, 30]:
        size = inv_base * lev_ex
        risk_usd = size * risk_dist
        examples.append(f"{lev_ex}x → ${size:,.0f} (riesgo ${risk_usd:.2f})")

    examples_text = "\n      ".join(examples)

    # Línea de trigger si aplica
    trigger_line = f"\n🎯 <b>Trigger:</b> ${trigger:,.2f}" if trigger else ""

    return f"""
✅ <b>POSICIÓN ABIERTA</b> ✅

<b>{side_emoji} {side} @ ${price:,.2f}</b>
━━━━━━━━━━━━━━━━━━{trigger_line}
📦 <b>Cantidad:</b> {qty:.6f} BTC
💰 <b>Equity usado:</b> ${equity:.2f}

<b>📈 EJEMPLO CON $100 USD:</b>
      {examples_text}

<i>⏰ {datetime.now().strftime('%H:%M:%S')} UTC</i>
"""


def format_close(side, entry, exit_px, pnl_pct, reason, equity_before=None):
    """
    Formatea mensaje cuando se CIERRA una posición
    """
    emoji = "🟢" if pnl_pct > 0 else "🔴"
    sign = "+" if pnl_pct > 0 else ""

    # Traducir razón a español amigable
    reason_translated = {
        "TP": "Take Profit 🎯",
        "SL": "Stop Loss 🛡️",
        "BE": "Breakeven ⚖️",
        "TRAIL": "Trailing Stop 📏",
        "TIME": "Tiempo máximo ⏱️",
        "UNKNOWN": "Desconocido ❓"
    }.get(reason, reason)

    # Calcular ejemplos con $100
    inv_base = 100
    examples = []
    for lev_ex in [5, 15, 30]:
        pnl_usd = inv_base * lev_ex * (pnl_pct / 100)
        examples.append(f"{lev_ex}x → ${pnl_usd:+.2f}")

    examples_text = "\n      ".join(examples)

    # Equity si se proporciona
    equity_line = f"\n💰 <b>Equity final:</b> ${equity_before * (1 + pnl_pct/100):.2f}" if equity_before else ""

    return f"""
{emoji} <b>POSICIÓN CERRADA</b> {emoji}

<b>{side} | {reason_translated}</b>
━━━━━━━━━━━━━━━━━━
📉 <b>Entrada:</b> ${entry:,.2f}
📈 <b>Salida:</b> ${exit_px:,.2f}
💵 <b>PnL:</b> {sign}{pnl_pct:.2f}%{equity_line}

<b>💰 IMPACTO EN $100 USD:</b>
      {examples_text}

<i>⏰ {datetime.now().strftime('%H:%M:%S')} UTC</i>
"""


def format_heartbeat(btc_price, daily_change, atr_value, trend_15m, trend_1h,
                    rsi_value, armed_status, cooldown_mins, loss_streak,
                    equity, reason_no_trade=""):
    """
    Formatea mensaje de heartbeat (cada 2 horas)
    Explica por qué NO hay trade (educativo)
    """

    # Estado del armado
    if armed_status:
        armed_text = "🔫 ARMAS ACTIVAS"
    else:
        armed_text = "💤 ESPERANDO"

    # Flechas de tendencia con emojis
    trend_15m_emoji = "🟢" if trend_15m == "LONG" else ("🔴" if trend_15m == "SHORT" else "⚪")
    trend_1h_emoji = "🟢" if trend_1h == "LONG" else ("🔴" if trend_1h == "SHORT" else "⚪")

    trend_15m_text = f"{trend_15m_emoji} {trend_15m}" if trend_15m != "NEUTRO" else "⚪ NEUTRO"
    trend_1h_text = f"{trend_1h_emoji} {trend_1h}" if trend_1h != "NEUTRO" else "⚪ NEUTRO"

    # Cambio diario
    change_emoji = "🟢" if daily_change > 0 else ("🔴" if daily_change < 0 else "⚪")

    # Cooldown
    cooldown_text = f"{cooldown_mins} min" if cooldown_mins > 0 else "0"

    # Formatear precio
    price_str = f"${btc_price:,.2f}"

    return f"""
<b>🕒 HEARTBEAT STRATUM</b> · <i>{datetime.now().strftime('%H:%M UTC')}</i>

<b>📊 MERCADO</b>
━━━━━━━━━━━━━━━━━━
{change_emoji} <b>BTC:</b> {price_str} ({daily_change:+.2f}% 24h)
📏 <b>ATR(15):</b> ${atr_value:.2f}

<b>📈 TENDENCIA</b>
━━━━━━━━━━━━━━━━━━
⏱️ 15m: {trend_15m_text}
⏰ 1h:   {trend_1h_text}
📐 RSI(14): {rsi_value:.1f}

<b>🤖 ESTADO BOT</b>
━━━━━━━━━━━━━━━━━━
{armed_text} · Cooldown: {cooldown_text}
📉 Racha pérdidas: {loss_streak}
💰 Equity: ${equity:.2f}

<b>🔍 ¿POR QUÉ NO ENTRA?</b>
━━━━━━━━━━━━━━━━━━
{reason_no_trade}

<i>⏳ Próximo heartbeat en 2h</i>
"""


def format_error(error_msg, context=""):
    """
    Formatea mensaje de error (para alertas)
    """
    return f"""
🚨 <b>⚠️ ALERTA - ERROR</b> 🚨

<b>Contexto:</b> {context}
<b>Error:</b> {error_msg[:200]}

<i>⏰ {datetime.now().strftime('%H:%M:%S')} UTC</i>
"""


def format_startup(config_summary):
    """
    Formatea mensaje cuando el bot inicia
    """
    return f"""
<b>🚀 STRATUM BOT INICIADO</b>

<b>⚙️ CONFIGURACIÓN</b>
━━━━━━━━━━━━━━━━━━
{config_summary}

<i>✅ Monitoreando mercado 24/7</i>
<i>⏰ {datetime.now().strftime('%H:%M:%S')} UTC</i>
"""


# ============================================
# FUNCIÓN DE PRUEBA
# ============================================
def test_telegram():
    """
    Función para probar que Telegram funciona
    Ejecutar: python -c "import oracle; oracle.test_telegram()"
    """
    print("🔍 Probando conexión con Telegram...")

    test_msg = f"""
<b>🧪 MENSAJE DE PRUEBA</b>

Si ves esto, Telegram está configurado correctamente ✅

<b>Configuración actual:</b>
• Token: {TELEGRAM_TOKEN[:10]}...{TELEGRAM_TOKEN[-5:] if TELEGRAM_TOKEN else 'NO CONFIGURADO'}
• Chat ID: {TELEGRAM_CHAT_ID}

<i>⏰ {datetime.now().strftime('%H:%M:%S')} UTC</i>
"""

    success = send_telegram(test_msg)
    if success:
        print("✅ Prueba exitosa - revisa Telegram")
    else:
        print("❌ Falló la prueba - revisa token y chat ID")

    return success


# ============================================
# AUTO-EJECUCIÓN SI SE LLAMA DIRECTAMENTE
# ============================================
if __name__ == "__main__":
    print("🔮 ORACLE - Módulo de notificaciones para STRATUM")
    print("Este archivo no debe ejecutarse directamente.")
    print("Para probar: from oracle import test_telegram; test_telegram()")
