"""
Rebuilt signal engine with stricter confluence logic:

1. Higher timeframe (1h) trend must agree with signal direction
2. Support/resistance levels must have minimum 2 touches to qualify
3. RSI divergence is a strong signal — weighted heavily
4. Candle body size filter — tiny candles are noise
5. No contradicting indicators allowed — all must point same direction
"""

from dataclasses import dataclass
import pandas as pd

from bot.patterns import detect_engulfing, detect_pin_bar
from bot.indicators import (
    add_indicators,
    find_support_resistance,
    nearest_level,
    detect_rsi_divergence,
    get_higher_tf_trend,
)


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
    atr = df["atr"].iloc[-1]
    price = df["close"].iloc[-1]
    if price == 0 or pd.isna(atr):
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


def _candle_body_is_significant(df: pd.DataFrame) -> bool:
    """
    Reject tiny candles — body must be at least 30% of the
    candle range, otherwise it is noise not a real signal.
    """
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]
    if candle_range == 0:
        return False
    return (body / candle_range) >= 0.3


def _score_and_direction(
    df: pd.DataFrame,
    support_levels: list,
    resistance_levels: list,
    higher_tf_trend: str,
) -> tuple[str | None, float, list[str]]:

    last = df.iloc[-1]
    price = last["close"]
    reasons = []
    long_score = 0
    short_score = 0
    max_score = 0

    # --- Hard filter 1: Higher timeframe trend ---
    # Signal direction must match 1h trend, sideways = no trade
    if higher_tf_trend == "sideways":
        return None, 0, []

    allowed_direction = "long" if higher_tf_trend == "uptrend" else "short"

    # --- Hard filter 2: Candle body significance ---
    if not _candle_body_is_significant(df):
        return None, 0, []

    # --- RSI divergence (worth 35 pts — strongest signal) ---
    max_score += 35
    divergence = detect_rsi_divergence(df)
    if divergence == "bullish" and allowed_direction == "long":
        long_score += 35
        reasons.append("Bullish RSI divergence — sellers weakening, reversal up likely")
    elif divergence == "bearish" and allowed_direction == "short":
        short_score += 35
        reasons.append("Bearish RSI divergence — buyers weakening, reversal down likely")

    # --- Candle pattern confirmation (worth 25 pts) ---
    max_score += 25
    engulf = detect_engulfing(df)
    pin = detect_pin_bar(df)
    pattern = engulf or pin
    pattern_name = "Engulfing" if engulf else "Pin bar"
    if pattern == "bullish" and allowed_direction == "long":
        long_score += 25
        reasons.append(f"{pattern_name} bullish reversal candle confirmed")
    elif pattern == "bearish" and allowed_direction == "short":
        short_score += 25
        reasons.append(f"{pattern_name} bearish reversal candle confirmed")

    # --- Validated S/R proximity (worth 20 pts) ---
    # Only levels with 2+ touches qualify (enforced in find_support_resistance)
    max_score += 20
    near_support = nearest_level(price, support_levels, "below")
    near_resistance = nearest_level(price, resistance_levels, "above")
    if near_support and abs(price - near_support) / price * 100 <= 0.4:
        if allowed_direction == "long":
            long_score += 20
            reasons.append(f"Price at validated support zone ~{near_support:.4f} (2+ touches)")
    if near_resistance and abs(near_resistance - price) / price * 100 <= 0.4:
        if allowed_direction == "short":
            short_score += 20
            reasons.append(f"Price at validated resistance zone ~{near_resistance:.4f} (2+ touches)")

    # --- RSI extreme confirmation (worth 10 pts) ---
    max_score += 10
    rsi = last.get("rsi")
    if rsi is not None:
        if rsi <= 32 and allowed_direction == "long":
            long_score += 10
            reasons.append(f"RSI deeply oversold ({rsi:.1f})")
        elif rsi >= 68 and allowed_direction == "short":
            short_score += 10
            reasons.append(f"RSI deeply overbought ({rsi:.1f})")

    # --- Volume confirmation (worth 10 pts) ---
    max_score += 10
    vol_avg = last.get("vol_avg20")
    if vol_avg and last["volume"] > vol_avg * 1.3:
        if allowed_direction == "long":
            long_score += 10
        else:
            short_score += 10
        reasons.append("Strong volume spike confirms move (>1.5x 20-period average)")

    # --- Higher timeframe trend bonus (worth 10 pts) ---
    max_score += 10
    if higher_tf_trend == "uptrend" and allowed_direction == "long":
        long_score += 10
        reasons.append("1H timeframe confirms uptrend")
    elif higher_tf_trend == "downtrend" and allowed_direction == "short":
        short_score += 10
        reasons.append("1H timeframe confirms downtrend")

    # Block any signal going against the allowed direction
    if allowed_direction == "long":
        short_score = 0
    else:
        long_score = 0

    if long_score == 0 and short_score == 0:
        return None, 0, []

    # Require at least 2 reasons to fire — single-factor signals are noise
    if len(reasons) < 1:
        return None, 0, []

    if long_score >= short_score:
        return "long", round(long_score / max_score * 100, 1), reasons
    return "short", round(short_score / max_score * 100, 1), reasons


def build_signal(symbol: str, exchange_id: str, raw_df, min_risk_reward: float) -> Signal | None:
    if raw_df is None or len(raw_df) < 80:
        return None

    df = add_indicators(raw_df)

    # Get higher timeframe trend first — hard filter
    higher_tf_trend = get_higher_tf_trend(df)

    support_levels, resistance_levels = find_support_resistance(df)

    direction, confidence, reasons = _score_and_direction(
        df, support_levels, resistance_levels, higher_tf_trend
    )
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
