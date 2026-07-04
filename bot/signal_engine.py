"""
Multi-timeframe signal engine with 3 TP levels.
- 1h candles: trend direction and S/R zones
- 15m candles: entry timing, pattern confirmation
- TP1 = 3x risk, TP2 = 5x risk, TP3 = 7x risk
- SL placed beyond the invalidation swing point
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd

from bot.patterns import detect_engulfing, detect_pin_bar
from bot.indicators import add_indicators, find_support_resistance, nearest_level

import logging
logger = logging.getLogger(__name__)

_seen_signals: dict[str, datetime] = {}


@dataclass
class Signal:
    symbol: str
    exchange: str
    direction: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    take_profit3: float
    sl_pct: float
    tp1_pct: float
    tp2_pct: float
    tp3_pct: float
    leverage: int
    reasons: list[str]
    fired_at: datetime = field(default_factory=datetime.utcnow)


def _is_duplicate(symbol: str, direction: str, cooldown_minutes: int) -> bool:
    key = f"{symbol}_{direction}"
    last = _seen_signals.get(key)
    if last and datetime.utcnow() - last < timedelta(minutes=cooldown_minutes):
        return True
    _seen_signals[key] = datetime.utcnow()
    return False


def _calculate_volatility(df: pd.DataFrame) -> str:
    atr = df["atr"].iloc[-1] if "atr" in df.columns else None
    price = df["close"].iloc[-1]
    if not atr or price == 0 or pd.isna(atr):
        return "high"
    return "high" if (atr / price) * 100 >= 2.0 else "low"


def _calculate_leverage(confidence: float, volatility: str) -> int:
    if confidence >= 85:
        return 15 if volatility == "low" else 10
    elif confidence >= 80:
        return 10 if volatility == "low" else 7
    elif confidence >= 75:
        return 7 if volatility == "low" else 5
    else:
        return 5 if volatility == "low" else 3


def _get_1h_trend(df_1h: pd.DataFrame) -> str:
    df = add_indicators(df_1h)
    last = df.iloc[-1]
    price = last["close"]
    ma_fast = last.get("ma_fast")
    ma_mid = last.get("ma_mid")
    if not ma_fast or not ma_mid:
        return "sideways"
    if ma_fast > ma_mid and price > ma_mid:
        return "uptrend"
    elif ma_fast < ma_mid and price < ma_mid:
        return "downtrend"
    return "sideways"


def _get_1h_levels(df_1h: pd.DataFrame) -> tuple[list, list]:
    df = add_indicators(df_1h)
    return find_support_resistance(df, lookback=80)


def _calculate_sl(df_15m: pd.DataFrame, direction: str) -> float | None:
    """
    Places SL beyond the nearest swing point.
    No percentage cap — swing point determines natural invalidation.
    """
    if direction == "long":
        sl = df_15m["low"].tail(10).min() * 0.998
    else:
        sl = df_15m["high"].tail(10).max() * 1.002
    return round(sl, 6)


def _score_15m(
    df_15m: pd.DataFrame,
    allowed_direction: str,
    support_1h: list,
    resistance_1h: list,
) -> tuple[float, list[str]]:
    df = add_indicators(df_15m)
    last = df.iloc[-1]
    price = last["close"]
    score = 0
    max_score = 0
    reasons = []

    # --- 15m candle pattern (30 pts) ---
    max_score += 30
    engulf = detect_engulfing(df)
    pin = detect_pin_bar(df)
    pattern = engulf or pin
    pattern_name = "Engulfing" if engulf else "Pin bar"
    if pattern == "bullish" and allowed_direction == "long":
        score += 30
        reasons.append(f"{pattern_name} bullish reversal on 15m")
    elif pattern == "bearish" and allowed_direction == "short":
        score += 30
        reasons.append(f"{pattern_name} bearish reversal on 15m")

    # No pattern — not worth scoring further
    if score == 0:
        return 0, []

    # --- 1h S/R proximity (25 pts) ---
    max_score += 25
    near_support = nearest_level(price, support_1h, "below")
    near_resistance = nearest_level(price, resistance_1h, "above")
    if near_support and abs(price - near_support) / price * 100 <= 1.0:
        if allowed_direction == "long":
            score += 25
            reasons.append(f"1H support zone ~{near_support:.4f}")
    if near_resistance and abs(near_resistance - price) / price * 100 <= 1.0:
        if allowed_direction == "short":
            score += 25
            reasons.append(f"1H resistance zone ~{near_resistance:.4f}")

    # --- RSI confirmation (20 pts) ---
    max_score += 20
    rsi = last.get("rsi")
    if rsi is not None:
        if rsi <= 40 and allowed_direction == "long":
            score += 20
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi >= 60 and allowed_direction == "short":
            score += 20
            reasons.append(f"RSI overbought ({rsi:.1f})")
        if rsi >= 75 and allowed_direction == "long":
            return 0, []
        if rsi <= 25 and allowed_direction == "short":
            return 0, []

    # --- 15m MA structure (15 pts) ---
    max_score += 15
    ma_fast = last.get("ma_fast")
    ma_mid = last.get("ma_mid")
    if ma_fast and ma_mid:
        if ma_fast > ma_mid and price > ma_fast and allowed_direction == "long":
            score += 15
            reasons.append("15m uptrend structure confirmed")
        elif ma_fast < ma_mid and price < ma_fast and allowed_direction == "short":
            score += 15
            reasons.append("15m downtrend structure confirmed")

    # --- Volume confirmation (10 pts) ---
    max_score += 10
    vol_avg = last.get("vol_avg20")
    if vol_avg and last["volume"] > vol_avg * 1.3:
        score += 10
        reasons.append("Volume spike confirms move")

    if not reasons:
        return 0, []

    return round(score / max_score * 100, 1), reasons


def build_signal(
    symbol: str,
    exchange_id: str,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    min_risk_reward: float,
    cooldown_minutes: int = 30,
) -> Signal | None:

    if df_1h is None or df_15m is None:
        return None
    if len(df_1h) < 50 or len(df_15m) < 50:
        return None

    # Step 1 — 1H trend
    trend_1h = _get_1h_trend(df_1h)
    if trend_1h == "uptrend":
        directions_to_try = ["long"]
    elif trend_1h == "downtrend":
        directions_to_try = ["short"]
    else:
        directions_to_try = ["long", "short"]

    # Step 2 — 1H S/R levels
    support_1h, resistance_1h = _get_1h_levels(df_1h)

    df_15m_ind = add_indicators(df_15m)
    best_sig = None
    best_confidence = 0

    for direction in directions_to_try:
        # Step 3 — score 15m setup
        confidence, reasons = _score_15m(
            df_15m, direction, support_1h, resistance_1h
        )
        if confidence == 0 or not reasons:
            continue

        if confidence <= best_confidence:
            continue

        # Step 4 — duplicate check
        if _is_duplicate(symbol, direction, cooldown_minutes):
            continue

        price = df_15m_ind["close"].iloc[-1]

        # Step 5 — calculate SL from swing point
        stop_loss = _calculate_sl(df_15m_ind, direction)
        if stop_loss is None:
            continue

        risk = abs(price - stop_loss)
        if risk == 0:
            continue

        # Step 6 — TP1=3x, TP2=5x, TP3=7x risk
        if direction == "long":
            entry = price
            take_profit1 = round(entry + (risk * 3), 6)
            take_profit2 = round(entry + (risk * 5), 6)
            take_profit3 = round(entry + (risk * 7), 6)
        else:
            entry = price
            take_profit1 = round(entry - (risk * 3), 6)
            take_profit2 = round(entry - (risk * 5), 6)
            take_profit3 = round(entry - (risk * 7), 6)

        sl_pct = round(abs(entry - stop_loss) / entry * 100, 3)
        tp1_pct = round(abs(take_profit1 - entry) / entry * 100, 3)
        tp2_pct = round(abs(take_profit2 - entry) / entry * 100, 3)
        tp3_pct = round(abs(take_profit3 - entry) / entry * 100, 3)

        if trend_1h != "sideways":
            reasons.insert(0, f"1H {trend_1h} confirmed")

        volatility = _calculate_volatility(df_15m_ind)
        leverage = _calculate_leverage(confidence, volatility)

        best_confidence = confidence
        best_sig = Signal(
            symbol=symbol,
            exchange=exchange_id,
            direction=direction,
            confidence=confidence,
            entry=round(entry, 6),
            stop_loss=stop_loss,
            take_profit1=take_profit1,
            take_profit2=take_profit2,
            take_profit3=take_profit3,
            sl_pct=sl_pct,
            tp1_pct=tp1_pct,
            tp2_pct=tp2_pct,
            tp3_pct=tp3_pct,
            leverage=leverage,
            reasons=reasons,
        )

    return best_sig
