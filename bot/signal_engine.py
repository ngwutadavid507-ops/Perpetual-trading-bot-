"""
Multi-timeframe signal engine:
- 1h candles: trend direction, S/R zones, TP targets
- 15m candles: entry timing, candle pattern, SL placement
- Signal only fires when both timeframes agree on direction
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd

from bot.patterns import detect_engulfing, detect_pin_bar
from bot.indicators import add_indicators, find_support_resistance, nearest_level

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
    risk_reward1: float
    risk_reward2: float
    sl_pct: float
    tp1_pct: float
    tp2_pct: float
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
    """
    Relaxed 1h trend detection — only requires fast MA
    above/below mid MA, not full three-MA stack.
    """
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
    """
    Gets support and resistance levels from 1h candles.
    These are stronger, further apart levels suitable for TP targets.
    """
    df = add_indicators(df_1h)
    return find_support_resistance(df, lookback=80)


def _score_15m(
    df_15m: pd.DataFrame,
    allowed_direction: str,
    support_1h: list,
    resistance_1h: list,
) -> tuple[float, list[str]]:
    """
    Scores the 15m candle setup in the allowed direction.
    Returns (score_0_to_100, reasons).
    """
    df = add_indicators(df_15m)
    last = df.iloc[-1]
    price = last["close"]
    score = 0
    max_score = 0
    reasons = []

    # --- 15m candle pattern (35 pts) ---
    max_score += 35
    engulf = detect_engulfing(df)
    pin = detect_pin_bar(df)
    pattern = engulf or pin
    pattern_name = "Engulfing" if engulf else "Pin bar"
    if pattern == "bullish" and allowed_direction == "long":
        score += 35
        reasons.append(f"{pattern_name} bullish reversal on 15m")
    elif pattern == "bearish" and allowed_direction == "short":
        score += 35
        reasons.append(f"{pattern_name} bearish reversal on 15m")

    # --- 1h S/R proximity (25 pts) ---
    max_score += 25
    near_support = nearest_level(price, support_1h, "below")
    near_resistance = nearest_level(price, resistance_1h, "above")
    if near_support and abs(price - near_support) / price * 100 <= 0.8:
        if allowed_direction == "long":
            score += 25
            reasons.append(f"1H support zone ~{near_support:.4f}")
    if near_resistance and abs(near_resistance - price) / price * 100 <= 0.8:
        if allowed_direction == "short":
            score += 25
            reasons.append(f"1H resistance zone ~{near_resistance:.4f}")

    # --- RSI on 15m (20 pts) with contradiction block ---
    max_score += 20
    rsi = last.get("rsi")
    if rsi is not None:
        if rsi <= 35 and allowed_direction == "long":
            score += 20
            reasons.append(f"RSI oversold on 15m ({rsi:.1f})")
        elif rsi >= 65 and allowed_direction == "short":
            score += 20
            reasons.append(f"RSI overbought on 15m ({rsi:.1f})")
        # Hard contradiction block
        if rsi >= 70 and allowed_direction == "long":
            return 0, []
        if rsi <= 25 and allowed_direction == "short":
            return 0, []

    # --- Volume confirmation (20 pts) ---
    max_score += 20
    vol_avg = last.get("vol_avg20")
    if vol_avg and last["volume"] > vol_avg * 1.3:
        score += 20
        reasons.append("Volume spike confirms 15m move")

    if score == 0 or not reasons:
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

    # Step 1 — get 1h trend (hard filter)
    trend_1h = _get_1h_trend(df_1h)
    if trend_1h == "sideways":
        return None

    allowed_direction = "long" if trend_1h == "uptrend" else "short"

    # Step 2 — get 1h S/R levels for TP targets
    support_1h, resistance_1h = _get_1h_levels(df_1h)

    # Step 3 — score 15m entry setup
    confidence, reasons = _score_15m(
        df_15m, allowed_direction, support_1h, resistance_1h
    )
    if confidence == 0 or not reasons:
        return None

    # Step 4 — duplicate check
    if _is_duplicate(symbol, allowed_direction, cooldown_minutes):
        return None

    # Step 5 — build entry/SL from 15m precision
    df_15m_ind = add_indicators(df_15m)
    price = df_15m_ind["close"].iloc[-1]
    recent_swing_low = df_15m_ind["low"].tail(10).min()
    recent_swing_high = df_15m_ind["high"].tail(10).max()

    if allowed_direction == "long":
        entry = price
        stop_loss = recent_swing_low * 0.998
        risk = entry - stop_loss
        if risk <= 0:
            return None
        take_profit1 = entry + risk
        # TP2 from 1h resistance — bigger target
        target_1h = nearest_level(price, resistance_1h, "above")
        take_profit2 = target_1h if target_1h else entry + (risk * 2.5)
    else:
        entry = price
        stop_loss = recent_swing_high * 1.002
        risk = stop_loss - entry
        if risk <= 0:
            return None
        take_profit1 = entry - risk
        # TP2 from 1h support — bigger target
        target_1h = nearest_level(price, support_1h, "below")
        take_profit2 = target_1h if target_1h else entry - (risk * 2.5)

    rr1 = round(abs(take_profit1 - entry) / risk, 2)
    rr2 = round(abs(take_profit2 - entry) / risk, 2)

    if rr2 < min_risk_reward:
        return None

    sl_pct = round(abs(entry - stop_loss) / entry * 100, 3)
    tp1_pct = round(abs(take_profit1 - entry) / entry * 100, 3)
    tp2_pct = round(abs(take_profit2 - entry) / entry * 100, 3)

    # Add trend reason
    reasons.insert(0, f"1H trend: {'uptrend' if allowed_direction == 'long' else 'downtrend'} confirmed")

    volatility = _calculate_volatility(df_15m_ind)
    leverage = _calculate_leverage(confidence, volatility)

    return Signal(
        symbol=symbol,
        exchange=exchange_id,
        direction=allowed_direction,
        confidence=confidence,
        entry=round(entry, 6),
        stop_loss=round(stop_loss, 6),
        take_profit1=round(take_profit1, 6),
        take_profit2=round(take_profit2, 6),
        risk_reward1=rr1,
        risk_reward2=rr2,
        sl_pct=sl_pct,
        tp1_pct=tp1_pct,
        tp2_pct=tp2_pct,
        leverage=leverage,
        reasons=reasons,
    )
