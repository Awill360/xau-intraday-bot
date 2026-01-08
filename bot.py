# bot.py
import os
import requests
from datetime import datetime, timedelta, timezone

# =========================
# CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

TD_API_KEY   = os.getenv("TD_API_KEY")
TD_BASE_URL  = "https://api.twelvedata.com/time_series"
SYMBOL       = "XAU/USD"

# Timezone Europe/Paris (offset simple: 1 en hiver, 2 en été)
TIMEZONE_OFFSET = int(os.getenv("TIMEZONE_OFFSET", "1"))

# --- Données & indicateurs sur 5 minutes ---
LOOKBACK_BARS = 360      # buffer confortable (>= 240) pour EMA/RSI/ATR
EMA_FAST      = 20       # EMA20 (5 min)
EMA_SLOW      = 50       # EMA50 (5 min)
RSI_LEN       = 14
ATR_LEN       = 14
DONCHIAN_LEN  = 12       # 12 x 5 min = 1h (pack horaire)

# --- Décision & risque ---
CONFIDENCE_MAP     = {3:0.80, 2:0.60, 1:0.40, 0:0.00}
ATR_LOW_PERCENTILE = 30
SESSION_FILTER_ON  = True
SESSIONS_PARIS     = [(8,16), (14,22)]  # London & New York (Europe/Paris)

SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0


# =========================
# TELEGRAM
# =========================
def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Secrets Telegram manquants.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=15)
        if r.status_code != 200:
            print("Telegram non-200:", r.text)
    except Exception as e:
        print("Erreur envoi Telegram:", e)


# =========================
# DATA (Twelve Data 5 min)
# =========================
def get_gold_data_5m():
    """
    Récupère des bougies 5 min XAU/USD depuis Twelve Data.
    Retour: liste triée chronologiquement de dicts {time, open, high, low, close}
    """
    if not TD_API_KEY:
        raise RuntimeError("TD_API_KEY manquant.")
    params = {
        "symbol": SYMBOL,
        "interval": "5min",        # interval officiel Twelve Data
        "outputsize": LOOKBACK_BARS,
        "timezone": "UTC",
        "apikey": TD_API_KEY,
    }
    r = requests.get(TD_BASE_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "status" in data and data["status"] == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")
    if "values" not in data:
        raise RuntimeError("Réponse TwelveData inattendue.")
    # values -> du plus récent au plus ancien ; on inverse pour chronologie
    values = list(reversed(data["values"]))
    candles = []
    for v in values:
        ts = v["datetime"].replace(" ", "T") + "Z"
        candles.append({
            "time": ts,
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low":  float(v["low"]),
            "close":float(v["close"]),
        })
    if len(candles) < 120:
        raise RuntimeError(f"Trop peu de bougies 5 min: {len(candles)}")
    return candles


# =========================
# INDICATEURS (sur séries 5 min)
# =========================
def ema_series(values, span):
    if not values:
        return []
    k = 2 / (span + 1)
    out = []
    seed = sum(values[:span]) / span if len(values) >= span else values[0]
    prev = seed
    for v in values:
        prev = (v * k) + (prev * (1 - k))
        out.append(prev)
    return out

def rsi_series(closes, period=14):
    if len(closes) < period + 1:
        return [None]*len(closes)
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]

    def ema(seq, p):
        k = 2 / (p + 1)
        out, prev = [], (sum(seq[:p]) / p if len(seq) >= p else seq[0])
        for x in seq:
            prev = (x * k) + (prev * (1 - k))
            out.append(prev)
        return out

    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)

    rsi = [None] * (period)
    for i in range(period-1, len(avg_gain)):
        ag = avg_gain[i]
        al = avg_loss[i] if avg_loss[i] != 0 else 1e-12
        rs = ag / al
        rsi_val = 100 - (100 / (1 + rs))
        rsi.append(rsi_val)
    rsi = [None] + rsi
    if len(rsi) < len(closes):
        rsi += [None] * (len(closes) - len(rsi))
    return rsi

def atr_series(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return [None]*len(closes)
    tr_list = []
    for i in range(len(closes)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            pc = closes[i-1]
            tr = max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc))
        tr_list.append(tr)
    atr = []
    for i in range(len(tr_list)):
        if i < period:
            atr.append(None)
        else:
            atr.append(sum(tr_list[i-period+1:i+1]) / period)
    return atr

def percentile(values, p):
    vals = [v for v in values if v is not None]
    if not vals: return None
    vals.sort()
    k = int((p/100.0) * (len(vals)-1))
    return vals[k]


# =========================
# SCORING / DÉCISION
# =========================
def last_pack_indices(n, pack_len=12):  # 12 x 5 min = 1h
    return list(range(max(n - pack_len, 0), n))

def score_trend(close, ema20, ema50, ema50_prev):
    slope_ema50 = ema50 - (ema50_prev if ema50_prev is not None else ema50)
    if ema20 is None or ema50 is None: return 0
    if ema20 > ema50 and close > ema20 and slope_ema50 >= 0: return +1
    if ema20 < ema50 and close < ema20 and slope_ema50 <= 0: return -1
    return 0

def score_momentum(rsi_curr, rsi_prev):
    if rsi_curr is None or rsi_prev is None: return 0
    if rsi_curr > 55 and rsi_curr > rsi_prev: return +1
    if rsi_curr < 45 and rsi_curr < rsi_prev: return -1
    return 0

def score_vol_structure(close, donchian_high, donchian_low, atr_curr, atr_median20):
    if atr_curr is None or atr_median20 is None: return 0
    expanding = atr_curr >= atr_median20
    if expanding and donchian_high is not None and close >= donchian_high: return +1
    if expanding and donchian_low  is not None and close <= donchian_low:  return -1
    return 0

def confidence(num_signals): return CONFIDENCE_MAP.get(num_signals, 0.0)

def in_sessions_paris(dt_paris):
    h = dt_paris.hour
    for start_h, end_h in SESSIONS_PARIS:
        if start_h <= h < end_h:
            return True
    return False


# =========================
# ANALYSE (5 min) -> ALERTE HORAIRE (pack 12)
# =========================
def analyze_market_5m(candles):
    """
    candles: série 5 min triée; décision basée sur le pack des 12 dernières (1h).
    Retourne dict {decision, confidence, text} ou None.
    """
    if len(candles) < max(LOOKBACK_BARS, EMA_SLOW+RSI_LEN+ATR_LEN+5):
        return None

    closes = [c["close"] for c in candles]
    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]

    ema20_s = ema_series(closes, EMA_FAST)
    ema50_s = ema_series(closes, EMA_SLOW)
    rsi_s   = rsi_series(closes, RSI_LEN)
    atr_s   = atr_series(highs, lows, closes, ATR_LEN)

    idx_pack = last_pack_indices(len(candles), DONCHIAN_LEN)
    last_idx = idx_pack[-1]
    prev_idx = last_idx - 1 if last_idx - 1 >= 0 else last_idx

    donchian_high = max(highs[i] for i in idx_pack)
    donchian_low  = min(lows[i]  for i in idx_pack)

    start = max(last_idx - 19, 0)  # fenêtre ATR pour médiane
    atr_window   = [atr_s[i] for i in range(start, last_idx+1)]
    atr_median20 = percentile(atr_window, 50)

    close     = closes[last_idx]
    ema20     = ema20_s[last_idx]
    ema50     = ema50_s[last_idx]
    ema50_prev= ema50_s[prev_idx]
    rsi_curr  = rsi_s[last_idx]
    rsi_prev  = rsi_s[prev_idx]
    atr_curr  = atr_s[last_idx]

    s1 = score_trend(close, ema20, ema50, ema50_prev)
    s2 = score_momentum(rsi_curr, rsi_prev)
    s3 = score_vol_structure(close, donchian_high, donchian_low, atr_curr, atr_median20)
    S = s1 + s2 + s3
    num_signals = sum(int(x != 0) for x in [s1, s2, s3])

    # Filtre faible volatilité (ATR < p30 des 20 dernières barres)
    atr_vals_20 = [x for x in atr_window if x is not None]
    low_vol = False
    if atr_vals_20:
        atr_p30 = percentile(atr_vals_20, ATR_LOW_PERCENTILE)
        low_vol = (atr_curr is not None and atr_p30 is not None and atr_curr < atr_p30)

    decision = "HOLD"
    if not low_vol:
        if S >= 2: decision = "BUY"
        elif S <= -2: decision = "SELL"

    conf = confidence(num_signals)
    sl = SL_ATR_MULT * atr_curr if atr_curr is not None else None
    tp = TP_ATR_MULT * atr_curr if atr_curr is not None else None

    text = (
        f"[{SYMBOL} - Décision horaire (5m pack x12)]\n"
        f"Signal: {decision} | Confiance: {conf:.2f}\n"
        f"Scores → Trend:{s1} Momentum:{s2} Vol/Struct:{s3} (S={S})\n"
        f"Close: {close:.2f} | EMA20: {ema20:.2f} | EMA50: {ema50:.2f}\n"
        f"Donchian(12) High/Low: {donchian_high:.2f} / {donchian_low:.2f}\n"
        f"ATR14: {atr_curr:.4f} | Median20: {atr_median20:.4f}\n"
        f"SL≈ {sl:.4f} | TP≈ {tp:.4f}"
    )
    return {"decision": decision, "confidence": conf, "text": text}


# =========================
# MAIN (une exécution = une décision horaire)
# =========================
def main():
    try:
        candles_5m = get_gold_data_5m()

        # Filtre session Europe/Paris
        now_utc = datetime.now(timezone.utc)
        paris = now_utc + timedelta(hours=TIMEZONE_OFFSET)
        if SESSION_FILTER_ON and not in_sessions_paris(paris):
            print("Hors sessions actives → HOLD (pas d'alerte).")
            return

        res = analyze_market_5m(candles_5m)
        if res:
            stamp = now_utc.strftime("%Y-%m-%d %H:%M UTC")
            send_telegram_message(f"{stamp}\n\n{res['text']}")
            print(res['text'])
        else:
            print("Pas d'alerte.")
    except Exception as e:
        print("Erreur principale:", e)

if __name__ == "__main__":
    main()
