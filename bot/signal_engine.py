"""
Multi-timeframe signal engine with 3 TP levels.
- 1h candles: trend direction and S/R zones
- 5m candles: entry timing, pattern confirmation (faster signals)
- TP1 = 3x risk, TP2 = 5x risk, TP3 = 7x risk
- SL placed beyond last 5 candle swing point (tight)
- Higher leverage table for 5m trading
- Redis-backed duplicate cache survives Railway restarts
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from bot.patterns import detect_engulfing, detect_pin_bar
from bot.indicators import add_indicators, find_support_resistance, nearest_level
from bot.redis_client import redis_get, redis_set

logger = logging.getLogger(__name__)

COOLDOWN_KEY_PREFIX = "cooldown:"


def _is_duplicate(symbol: str, direction: str, cooldown_minutes: int) -> bool:
    key = f"{COOLDOWN_KEY_PREFIX}{symbol}_{direction}"
    existing = redis_get(key)
    if existing:
        return True
    redis_set(key, "1", ex=cooldown_minutes * 60)
    return False


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


def _calculate_volatility(df: pd.DataFrame) -> str:
    atr = df["atr"].iloc[-1] if "atr" in df.columns else None
    price = df["close"].iloc[-1]
    if not atr or price == 0 or pd.isna(atr):
        return "high"
    return "high" if (atr / price) * 100 >= 1.5 else "low"


def _calculate_leverage(confidence: float, volatility: str) -> int:
    if confidence >= 85:
        return 50 if volatility == "low" else 35
    elif confidence >= 80:
        return 35 if volatility == "low" else 25
    elif confidence >= 75:
        return 25 if volatility == "low" else 20
    elif confidence >= 70:
        return 20 if volatility == "low" else 15
    else:
        return 15 if volatility == "low" else 10


def _get_1h_trend(df_1h: pd.DataFrame) -> str:
    """Determines trend from 1h candles."""
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
    """Gets S/R levels from 1h candles for TP targets."""
    df = add_indicators(df_1h)
    return find_support_resistance(df, lookback=80)


def _calculate_sl(df_5m: pd.DataFrame, direction: str) -> float:
    """
    Places SL beyond last 5 candle swing point — tight for 5m trading.
    Long: below lowest low of last 5 candles.
    Short: above highest high of last 5 candles.
    """
    if direction == "long":
        return round(df_5m["low"].tail(5).min() * 0.998, 6)
    else:
        return round(df_5m["high"].tail(5).max() * 1.002, 6)


def _is_after_impulse(df_5m: pd.DataFrame, direction: str) -> bool:
    """
    Checks if price made a meaningful move before signal candle.
    Prevents signals in choppy sideways price action.
    """
    if "atr" not in df_5m.columns:
        return True
    atr = df_5m["atr"].iloc[-1]
    if pd.isna(atr) or atr == 0:
        return True
    recent = df_5m.tail(6).iloc[:-1]
    if direction == "short":
        move = recent["close"].iloc[0] - recent["close"].iloc[-1]
    else:
        move = recent["close"].iloc[-1] - recent["close"].iloc[0]
    return move >= atr * 0.4


def _candle_is_significant(df_5m: pd.DataFrame) -> bool:
    """
    Signal candle body must be at least 15% of recent ATR.
    Filters noise on 5m timeframe.
    """
    if "atr" not in df_5m.columns:
        return True
    last = df_5m.iloc[-1]
    atr = df_5m["atr"].iloc[-1]
    if pd.isna(atr) or atr == 0:
        return True
    body = abs(last["close"] - last["open"])
    return body >= atr * 0.15


def _score_5m(
    df_5m: pd.DataFrame,
    allowed_direction: str,
    support_1h: list,
    resistance_1h: list,
) -> tuple[float, list[str]]:
    """
    Scores the 5m candle setup in the allowed direction.
    Returns (confidence_0_to_100, reasons).
    """
    df = add_indicators(df_5m)
    last = df.iloc[-1]
    price = last["close"]
    score = 0
    max_score = 0
    reasons = []

    # Candle significance check
    if not _candle_is_significant(df):
        return 0, []

    # --- Candle pattern (30 pts) — required ---
    max_score += 30
    engulf = detect_engulfing(df)
    pin = detect_pin_bar(df)
    pattern = engulf or pin
    pattern_name = "Engulfing" if engulf else "Pin bar"

    if pattern == "bullish" and allowed_direction == "long":
        score += 30
        reasons.append(f"{pattern_name} bullish reversal on 5m")
    elif pattern == "bearish" and allowed_direction == "short":
        score += 30
        reasons.append(f"{pattern_name} bearish reversal on 5m")
    else:
        return 0, []

    # Impulse move check
    if not _is_after_impulse(df, allowed_direction):
        return 0, []

    # --- 1H S/R proximity (25 pts) ---
    max_score += 25
    near_support = nearest_level(price, support_1h, "below")
    near_resistance = nearest_level(price, resistance_1h, "above")

    if allowed_direction == "long" and near_support:
        if abs(price - near_support) / price * 100 <= 1.5:
            score += 25
            reasons.append(f"1H support zone ~{near_support:.4f}")

    if allowed_direction == "short" and near_resistance:
        if abs(near_resistance - price) / price * 100 <= 1.5:
            score += 25
            reasons.append(f"1H resistance zone ~{near_resistance:.4f}")

    # --- RSI (20 pts) with contradiction block ---
    max_score += 20
    rsi = last.get("rsi")
    if rsi is not None:
        if allowed_direction == "long":
            if rsi >= 78:
                return 0, []
            if rsi <= 45:
                score += 20
                reasons.append(f"RSI oversold ({rsi:.1f})")
        elif allowed_direction == "short":
            if rsi <= 22:
                return 0, []
            if rsi >= 55:
                score += 20
                reasons.append(f"RSI overbought ({rsi:.1f})")

    # --- 5m MA structure (15 pts) ---
    max_score += 15
    ma_fast = last.get("ma_fast")
    ma_mid = last.get("ma_mid")
    if ma_fast and ma_mid:
        if allowed_direction == "long" and ma_fast > ma_mid and price > ma_fast:
            score += 15
            reasons.append("5m uptrend structure confirmed")
        elif allowed_direction == "short" and ma_fast < ma_mid and price < ma_fast:
            score += 15
            reasons.append("5m downtrend structure confirmed")

    # --- Volume (10 pts) ---
    max_score += 10
    vol_avg = last.get("vol_avg20")
    if vol_avg and vol_avg > 0 and last["volume"] > vol_avg * 1.3:
        score += 10
        reasons.append("Volume spike confirms move")

    if not reasons:
        return 0, []

    return round(score / max_score * 100, 1), reasons


def build_signal(
    symbol: str,
    exchange_id: str,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    min_risk_reward: float,
    cooldown_minutes: int = 30,
) -> Signal | None:

    if df_1h is None or df_5m is None:
        return None
    if len(df_1h) < 50 or len(df_5m) < 50:
        return None

    # Step 1 — 1H trend direction
    trend_1h = _get_1h_trend(df_1h)
    if trend_1h == "uptrend":
        directions_to_try = ["long"]
    elif trend_1h == "downtrend":
        directions_to_try = ["short"]
    else:
        directions_to_try = ["long", "short"]

    # Step 2 — 1H S/R levels for TP targets
    support_1h, resistance_1h = _get_1h_levels(df_1h)

    # Step 3 — 5m indicators
    df_5m_ind = add_indicators(df_5m)
    best_sig = None
    best_confidence = 0

    for direction in directions_to_try:

        # Step 4 — score 5m setup
        confidence, reasons = _score_5m(
            df_5m, direction, support_1h, resistance_1h
        )
        if confidence == 0 or not reasons:
            continue

        if confidence <= best_confidence:
            continue

        # Step 5 — duplicate check
        if _is_duplicate(symbol, direction, cooldown_minutes):
            continue

        # Step 6 — tight SL from 5m swing
        price = df_5m_ind["close"].iloc[-1]
        stop_loss = _calculate_sl(df_5m_ind, direction)
        risk = abs(price - stop_loss)
        if risk == 0:
            continue

        # Step 7 — TP at 3x, 5x, 7x risk
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

        volatility = _calculate_volatility(df_5m_ind)
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
