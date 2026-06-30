"""
Combines pattern detection + indicators into a scored signal with
entry zone, stop loss, and take profit — the framework from the lesson:
confluence of S/R level + candle confirmation + indicator agreement,
with SL beyond the invalidation point and TP at the next opposing level,
filtered by minimum risk:reward.
"""

from dataclasses import dataclass

from bot.patterns import detect_engulfing, detect_pin_bar
from bot.indicators import add_indicators, find_support_resistance, nearest_level


@dataclass
class Signal:
    symbol: str
    exchange: str
    direction: str          # "long" or "short"
    confidence: float       # 0-100
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reasons: list[str]


def _score_and_direction(df, support_levels, resistance_levels) -> tuple[str | None, float, list[str]]:
    """Returns (direction, confidence_0_100, reasons)."""
    last = df.iloc[-1]
    price = last["close"]
    reasons = []
    long_score = 0
    short_score = 0
    max_score = 0

    # --- Candle pattern confirmation (worth 30 pts) ---
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

    # --- Proximity to support/resistance (worth 25 pts) ---
    max_score += 25
    near_support = nearest_level(price, support_levels, "below")
    near_resistance = nearest_level(price, resistance_levels, "above")
    if near_support and abs(price - near_support) / price * 100 <= 0.5:
        long_score += 25
        reasons.append(f"Price at support zone ~{near_support:.4f}")
    if near_resistance and abs(near_resistance - price) / price * 100 <= 0.5:
        short_score += 25
        reasons.append(f"Price at resistance zone ~{near_resistance:.4f}")

    # --- RSI extreme (worth 15 pts) ---
    max_score += 15
    rsi = last.get("rsi")
    if rsi is not None:
        if rsi <= 35:
            long_score += 15
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi >= 65:
            short_score += 15
            reasons.append(f"RSI overbought ({rsi:.1f})")

    # --- MA trend agreement (worth 15 pts) ---
    max_score += 15
    if last.get("ma_fast") and last.get("ma_mid"):
        if last["ma_fast"] > last["ma_mid"] and price > last["ma_fast"]:
            long_score += 15
            reasons.append("Price above fast MA, fast MA above mid MA (uptrend structure)")
        elif last["ma_fast"] < last["ma_mid"] and price < last["ma_fast"]:
            short_score += 15
            reasons.append("Price below fast MA, fast MA below mid MA (downtrend structure)")

    # --- Volume confirmation (worth 15 pts) ---
    max_score += 15
    if last.get("vol_avg20") and last["volume"] > last["vol_avg20"] * 1.3:
        if long_score >= short_score:
            long_score += 15
        else:
            short_score += 15
        reasons.append("Volume spike confirms move (>1.3x 20-period average)")

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

    price = df.iloc[-1]["close"]
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

    return Signal(
        symbol=symbol,
        exchange=exchange_id,
        direction=direction,
        confidence=confidence,
        entry=round(entry, 6),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
        risk_reward=rr,
        reasons=reasons,
  )
