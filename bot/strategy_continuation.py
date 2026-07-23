"""
Triple Timeframe Trend Strategy.

Rule: ALL THREE timeframes must agree or no trade.

4H: Establishes dominant trend direction
1H: Must agree with 4H (unless 4H is weak/transitioning)
5m: Price at pullback level + entry confirmation candle

Produces 1-5 high quality signals per day.
"""

import pandas as pd
import logging

from bot.indicators import (
    add_indicators,
    find_support_resistance,
    nearest_level,
)

logger = logging.getLogger(__name__)


def _get_trend(df: pd.DataFrame) -> tuple[str, float]:
    """
    Gets trend direction and strength from any timeframe.
    Returns (direction, strength) where strength is 0.0 to 1.0.
    """
    ind = add_indicators(df)
    last = ind.iloc[-1]
    price = last["close"]
    ema_8 = last.get("ema_8")
    ema_21 = last.get("ema_21")
    ema_50 = last.get("ema_50")

    if not all([ema_8, ema_21, ema_50]):
        return "sideways", 0.0

    if price > ema_8 > ema_21 > ema_50:
        return "uptrend", 1.0
    if ema_8 > ema_21 > ema_50 and price > ema_21:
        return "uptrend", 0.8
    if ema_8 > ema_21 and price > ema_21:
        return "uptrend", 0.6

    if price < ema_8 < ema_21 < ema_50:
        return "downtrend", 1.0
    if ema_8 < ema_21 < ema_50 and price < ema_21:
        return "downtrend", 0.8
    if ema_8 < ema_21 and price < ema_21:
        return "downtrend", 0.6

    return "sideways", 0.0


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame | None:
    """Resamples 1H candles to 4H."""
    try:
        df = df_1h.copy().set_index("timestamp")
        df_4h = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna().reset_index()
        return df_4h if len(df_4h) >= 15 else None
    except Exception as e:
        logger.debug(f"4H resample failed: {e}")
        return None


def _is_at_pullback_level(
    df_5m: pd.DataFrame,
    direction: str,
    support_1h: list[float],
    resistance_1h: list[float],
) -> tuple[bool, str]:
    """
    Checks if price is at a valid pullback level.
    Tolerance widened to 1.5% for S/R and 1.0% for EMAs.
    """
    ind = add_indicators(df_5m)
    last = ind.iloc[-1]
    price = last["close"]
    ema_21 = last.get("ema_21")
    ema_50 = last.get("ema_50")

    if direction == "long":
        near_support = nearest_level(price, support_1h, "below")
        if near_support and abs(price - near_support) / price * 100 <= 1.5:
            return True, f"Pullback to 1H support ~{near_support:.4f}"
        if ema_21 and abs(price - ema_21) / price * 100 <= 1.0:
            return True, "Pullback to 21 EMA"
        if ema_50 and abs(price - ema_50) / price * 100 <= 1.0:
            return True, "Pullback to 50 EMA"
    else:
        near_resistance = nearest_level(price, resistance_1h, "above")
        if near_resistance and abs(near_resistance - price) / price * 100 <= 1.5:
            return True, f"Rally to 1H resistance ~{near_resistance:.4f}"
        if ema_21 and abs(price - ema_21) / price * 100 <= 1.0:
            return True, "Rally to 21 EMA"
        if ema_50 and abs(price - ema_50) / price * 100 <= 1.0:
            return True, "Rally to 50 EMA"

    return False, ""


def _get_entry_candle(
    df_5m: pd.DataFrame,
    direction: str,
) -> tuple[bool, str]:
    """
    Checks for entry confirmation candle on 5m.
    For longs: bullish candle or lower wick rejection.
    For shorts: bearish candle or upper wick rejection.
    """
    ind = add_indicators(df_5m)
    last = ind.iloc[-1]

    open_ = last["open"]
    close = last["close"]
    high = last["high"]
    low = last["low"]
    body = abs(close - open_)
    candle_range = high - low
    if candle_range == 0:
        return False, ""

    body_ratio = body / candle_range
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low

    if direction == "long":
        if close > open_ and body_ratio >= 0.25:
            return True, "Bullish entry candle confirms long"
        if lower_wick >= body * 1.5 and lower_wick > upper_wick:
            return True, "Bullish rejection — buyers stepping in"
    else:
        if close < open_ and body_ratio >= 0.25:
            return True, "Bearish entry candle confirms short"
        if upper_wick >= body * 1.5 and upper_wick > lower_wick:
            return True, "Bearish rejection — sellers stepping in"

    return False, ""


def _check_volume_and_rsi(
    df_5m: pd.DataFrame,
    direction: str,
) -> tuple[int, list[str]]:
    """Volume and RSI bonus points."""
    ind = add_indicators(df_5m)
    last = ind.iloc[-1]
    points = 0
    reasons = []

    rsi = last.get("rsi")
    vol_ratio = last.get("vol_ratio")

    if rsi is not None:
        if direction == "long" and 25 <= rsi <= 65:
            points += 1
            reasons.append(f"RSI healthy ({rsi:.1f})")
        elif direction == "short" and 35 <= rsi <= 75:
            points += 1
            reasons.append(f"RSI healthy ({rsi:.1f})")
        if direction == "long" and rsi > 78:
            points -= 2
        if direction == "short" and rsi < 22:
            points -= 2

    if vol_ratio is not None:
        if vol_ratio > 1.2:
            points += 1
            reasons.append(f"Volume confirming ({vol_ratio:.1f}x avg)")
        elif vol_ratio < 0.4:
            points -= 1

    return points, reasons


def score_continuation(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    support_1h: list[float],
    resistance_1h: list[float],
) -> tuple[str | None, float, list[str]]:
    """
    Triple timeframe trend strategy.
    Requirements:
    1. 4H trend clear
    2. 1H trend agrees (or 4H weak)
    3. Price at pullback level (S/R or EMA)
    Entry candle removed — trend + location is sufficient.
    """
    reasons = []

    # Step 1 — 4H trend
    df_4h = _resample_to_4h(df_1h)
    if df_4h is None:
        return None, 0, []

    trend_4h, strength_4h = _get_trend(df_4h)
    if trend_4h == "sideways":
        return None, 0, []

    direction = "long" if trend_4h == "uptrend" else "short"

    # Step 2 — 1H trend
    trend_1h, strength_1h = _get_trend(df_1h)
    if trend_1h == "sideways":
        return None, 0, []

    if trend_1h != trend_4h and strength_4h >= 0.8:
        return None, 0, []

    if trend_1h != trend_4h:
        direction = "long" if trend_1h == "uptrend" else "short"
        reasons.append(
            f"1H {trend_1h} leads "
            f"(strength {round(strength_1h * 100)}%)"
        )
    else:
        reasons.append(
            f"4H + 1H {trend_4h} confirmed "
            f"(4H={round(strength_4h * 100)}% | "
            f"1H={round(strength_1h * 100)}%)"
        )

    # Step 3 — pullback level
    at_level, level_reason = _is_at_pullback_level(
        df_5m, direction, support_1h, resistance_1h
    )
    if not at_level:
        return None, 0, []

    reasons.append(level_reason)

    # Bonus points
    vol_rsi_pts, vol_rsi_reasons = _check_volume_and_rsi(df_5m, direction)
    vol_rsi_bonus = max(0, min(10, vol_rsi_pts * 5))
    reasons.extend(vol_rsi_reasons)

    # Confidence
    base_confidence = 65.0
    tf_4h_bonus = round(strength_4h * 15, 1)
    tf_1h_bonus = round(strength_1h * 15, 1)

    confidence = min(100.0, round(
        base_confidence + tf_4h_bonus + tf_1h_bonus + vol_rsi_bonus, 1
    ))

    logger.info(
        f"Triple TF {direction}: conf={confidence}% "
        f"(base=65 + 4H={tf_4h_bonus} + 1H={tf_1h_bonus} "
        f"+ bonus={vol_rsi_bonus})"
    )

    return direction, confidence, reasons
