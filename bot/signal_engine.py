from dataclasses import dataclass
import pandas as pd

from bot.patterns import detect_engulfing, detect_pin_bar
from bot.indicators import add_indicators, find_support_resistance, nearest_level


@dataclass
class Signal:
    symbol: str
    exchange: str
    direction: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    leverage: int
    reasons: list[str]


def _calculate_volatility(df: pd.DataFrame) -> str:
    atr = df["atr"].iloc[-1] if "atr" in df.columns else None
    price = df["close"].iloc[-1]
    if not atr or price == 0:
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


def _score_and_direction(df, support_levels, resistance_levels):
    last = df.iloc[-1]
    price = last["close"]
    reasons = []
    long_score = 0
    short_score = 0
    max_score = 0

    # --- Candle pattern (30 pts) ---
    max_score += 30
    engulf = detect_engulfing(df)
    pin = detect_pin_bar(df)
    pattern = engulf or pin
    if pattern == "bullish":
        long_score += 30
        reasons.append(f"{'Engulfing' if engulf else 'Pin bar'} bullish reversal candle")
    elif pattern == "bearish":
        short_score += 30
        reasons.append(f"{'Engulfing' if engulf else 'Pin bar'} bearish reversal candle")

    # --- S/R proximity (25 pts) ---
    max_score += 25
    near_support = nearest_level(price, support_levels, "below")
    near_resistance = nearest_level(price, resistance_levels, "above")
    if near_support and abs(price - near_support) / price * 100 <= 0.5:
        long_score += 25
        reasons.append(f"Price at support zone ~{near_support:.4f}")
    if near_resistance and abs(near_resistance - price) / price * 100 <= 0.5:
        short_score += 25
        reasons.append(f"Price at resistance zone ~{near_resistance:.4f}")

    # --- RSI (15 pts) ---
    max_score += 15
    rsi = last.get("rsi")
    if rsi is not None:
        if rsi <= 35:
            long_score += 15
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi >= 65:
            short_score += 15
            reasons.append(f"RSI overbought ({rsi:.1f})")

    # --- MA trend as hard filter (15 pts) ---
    max_score += 15
    ma_fast = last.get("ma_fast")
    ma_mid = last.get("ma_mid")
    if ma_fast and ma_mid:
        uptrend = ma_fast > ma_mid and price > ma_fast
        downtrend = ma_fast < ma_mid and price < ma_fast
        if uptrend:
            long_score += 15
            short_score = 0
            reasons.append("Uptrend confirmed on MA structure")
        elif downtrend:
            short_score += 15
            long_score = 0
            reasons.append("Downtrend confirmed on MA structure")

    # --- Volume (15 pts) ---
    max_score += 15
    vol_avg = last.get("vol_avg20")
    if vol_avg and last["volume"] > vol_avg * 1.3:
        if long_score >= short_score:
            long_score += 15
        else:
            short_score += 15
        reasons.append("Volume spike confirms move")

    if long_score == 0 and short_score == 0:
        return None, 0, []

    if long_score >= short_score:
        return "long", round(long_score / max_score * 100, 1), reasons
    return "short", round(short_score / max_score * 100, 1), reasons


def build_signal(symbol: str, exchange_id: str, raw_df, min_risk_reward: float) -> Signal | None:
    if raw_df is None or len(raw_df) < 60:
        return None

    df = add_indicators(raw_df)
    support_levels, resistance_levels = find_support_resistance(df)

    direction, confidence, reasons = _score_and_direction(df, support_levels, resistance_levels)
    if direction is None:
        return None

    price = df["close"].iloc[-1]
    recent_swing_low = df["low"].tail(10).min()
    recent_swing_high = df["high"].tail(10).max()

    if direction == "long":
        entry = price
        stop_loss = recent_swing_low * 0.998
        target_level = nearest_level(price, resistance_levels, "above")
        take_profit = target_level if target_level else price + (price - stop_loss) * 2
    else:
        entry = price
        stop_loss = recent_swing_high * 1.002
        target_level = nearest_level(price, support_levels, "below")
        take_profit = target_level if target_level else price - (stop_loss - price) * 2

    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk == 0:
        return None
    rr = round(reward / risk, 2)

    if rr < min_risk_reward:
        return None

    volatility = _calculate_volatility(df)
    leverage = _calculate_leverage(confidence, volatility)

    return Signal(
        symbol=symbol,
        exchange=exchange_id,
        direction=direction,
        confidence=confidence,
        entry=round(entry, 6),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
        risk_reward=rr,
        leverage=leverage,
        reasons=reasons,
    )
