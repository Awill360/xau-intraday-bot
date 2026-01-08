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

# Timezone Europe/Paris (offset simple; mets 1 en hiver, 2 en été via secret)
TIMEZONE_OFFSET = int(os.getenv("TIMEZONE_OFFSET", "1"))

LOOKBACK_BARS = 180  # buffer confortable pour EMA/RSI/ATR

# Indicateurs
EMA_FAST = 20
EMA_SLOW = 50
RSI_LEN = 14
ATR_LEN = 14
DONCHIAN_LEN = 6  # 6 x 10 min = 1h

# Règles de décision
CONFIDENCE_MAP = {3:0.80, 2:0.60, 1:0.40, 0:0.00}
ATR_LOW_PERCENTILE = 30
SESSION_FILTER_ON = True
SESSIONS_PARIS = [(8,16), (14,22)]  # London & New York

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
# DATA (Twelve Data 10 min)
# =========================
def get_gold_data():
    if not TD_API_KEY:
        raise RuntimeError("TD_API_KEY manquant.")
    params = {
        "symbol": SYMBOL,
        "interval": "10min",
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
    values = list(reversed(data["values"]))  # chronologique
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
    if len(candles) < 60:
        raise RuntimeError(f"Trop peu de bougies: {len(candles)}")
    return candles


# =========================
# INDICATEURS
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
def last_pack_indices(n, pack_len=6):
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
# ANALYSE 10 MIN -> ALERTE HORAIRE (une exécution = une décision)
# =========================
def analyze_market_10m(candles):
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

    start = max(last_idx - 19, 0)
    atr_window = [atr_s[i] for i in range(start, last_idx+1)]
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

    atr_vals_20 = [x for x in atr_window if x is not None]
    low_vol = False
    if atr_vals_20:
        atr_p30 = percentile(atr_vals_20, 30)
        low_vol = (atr_curr is not None and atr_p30 is not None and atr_curr < atr_p30)

    decision = "HOLD"
    if not low_vol:
        if S >= 2: decision = "BUY"
        elif S <= -2: decision = "SELL"

    conf = confidence(num_signals)
    sl = SL_ATR_MULT * atr_curr if atr_curr is not None else None
    tp = TP_ATR_MULT * atr_curr if atr_curr is not None else None

    text = (
        f"[{SYMBOL} - Décision horaire]\n"
        f"Signal: {decision} | Confiance: {conf:.2f}\n"
        f"Scores → Trend:{s1} Momentum:{s2} Vol/Struct:{s3} (S={S})\n"
        f"Close: {close:.2f} | EMA20: {ema20:.2f} | EMA50: {ema50:.2f}\n"
        f"Donchian(6) High/Low: {donchian_high:.2f} / {donchian_low:.2f}\n"
        f"ATR14: {atr_curr:.4f} | Median20: {atr_median20:.4f}\n"
        f"SL≈ {sl:.4f} | TP≈ {tp:.4f}"
    )
    return {"decision": decision, "confidence": conf, "text": text}


def main():
    try:
        candles = get_gold_data()
        # Filtre session Europe/Paris
        now_utc = datetime.now(timezone.utc)
        paris = now_utc + timedelta(hours=TIMEZONE_OFFSET)
        if SESSION_FILTER_ON and not in_sessions_paris(paris):
            print("Hors sessions actives → HOLD (pas d'alerte).")
            return
        res = analyze_market_10m(candles)
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
