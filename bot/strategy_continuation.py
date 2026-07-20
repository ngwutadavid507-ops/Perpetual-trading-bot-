"""
Triple Timeframe Trend Strategy — the only strategy that matters.

Rule: ALL THREE timeframes must agree or no trade.

4H: Establishes the dominant trend direction
1H: Price pulling back to a key EMA or S/R level within that trend  
5m: Entry confirmation candle at the pullback level

This produces 1-3 signals per day maximum.
Every signal that fires has three independent timeframes confirming it.
No pattern chasing. No random entries. Only high probability setups.
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
    Returns (direction, strength) where direction is
    'uptrend', 'downtrend' or 'sideways'
    and strength is 0.0 to 1.0.
    """
    ind = add_indicators(df)
    last = ind.iloc[-1]
    price = last["close"]
    ema_8 = last.get("ema_8")
    ema_21 = last.get("ema_21")
    ema_50 = last.get("ema_50")

    if not all([ema_8, ema_21, ema_50]):
        return "sideways", 0.0

    # Perfect uptrend: price > ema8 > ema21 > ema50
    if price > ema_8 > ema_21 > ema_50:
        strength = 1.0
        return "uptrend", strength

    # Strong uptrend: ema8 > ema21 > ema50, price above ema21
    if ema_8 > ema_21 > ema_50 and price > ema_21:
        strength = 0.8
        return "uptrend", strength

    # Weak uptrend: ema8 > ema21, price above ema21
    if ema_8 > ema_21 and price > ema_21:
        strength = 0.6
        return "uptrend", strength

    # Perfect downtrend: price < ema8 < ema21 < ema50
    if price < ema_8 < ema_21 < ema_50:
        strength = 1.0
        return "downtrend", strength

    # Strong downtrend: ema8 < ema21 < ema50, price below ema21
    if ema_8 < ema_21 < ema_50 and price < ema_21:
        strength = 0.8
        return "downtrend", strength

    # Weak downtrend: ema8 < ema21, price below ema21
    if ema_8 < ema_21 and price < ema_21:
        strength = 0.6
        return "downtrend", strength

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
        return df_4h if len(df_4h) >= 20 else None
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
    Checks if price is at a valid pullback level on 1H S/R or EMA.
    Returns (is_at_level, reason_string).
    """
    ind = add_indicators(df_5m)
    last = ind.iloc[-1]
    price = last["close"]
    ema_21 = last.get("ema_21")
    ema_50 = last.get("ema_50")

    if direction == "long":
        # Check 1H support proximity
        near_support = nearest_level(price, support_1h, "below")
        if near_support and abs(price - near_support) / price * 100 <= 0.8:
            return True, f"Pullback to 1H support ~{near_support:.4f}"

        # Check EMA21 proximity
        if ema_21 and abs(price - ema_21) / price * 100 <= 0.5:
            return True, "Pullback to 21 EMA"

        # Check EMA50 proximity
        if ema_50 and abs(price - ema_50) / price * 100 <= 0.5:
            return True, "Pullback to 50 EMA"

    else:  # short
        # Check 1H resistance proximity
        near_resistance = nearest_level(price, resistance_1h, "above")
        if near_resistance and abs(near_resistance - price) / price * 100 <= 0.8:
            return True, f"Rally to 1H resistance ~{near_resistance:.4f}"

        # Check EMA21 proximity
        if ema_21 and abs(price - ema_21) / price * 100 <= 0.5:
            return True, "Rally to 21 EMA"

        # Check EMA50 proximity
        if ema_50 and abs(price - ema_50) / price * 100 <= 0.5:
            return True, "Rally to 50 EMA"

    return False, ""


def _get_entry_candle(
    df_5m: pd.DataFrame,
    direction: str,
) -> tuple[bool, str]:
    """
    Checks for a valid entry confirmation candle on 5m.
    For longs: bullish candle after pullback (close > open, lower wick)
    For shorts: bearish candle after rally (close < open, upper wick)
    Returns (confirmed, reason).
    """
    ind = add_indicators(df_5m)
    last = ind.iloc[-1]
    prev = ind.iloc[-2] if len(ind) >= 2 else None

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
        # Bullish candle: close > open, body at least 30% of range
        if close > open_ and body_ratio >= 0.3:
            return True, "Bullish entry candle confirms long"
        # Rejection candle: lower wick >= 2x body
        if lower_wick >= body * 2 and lower_wick > upper_wick:
            return True, "Bullish rejection candle — buyers stepping in"

    else:  # short
        # Bearish candle: close < open, body at least 30% of range
        if close < open_ and body_ratio >= 0.3:
            return True, "Bearish entry candle confirms short"
        # Rejection candle: upper wick >= 2x body
        if upper_wick >= body * 2 and upper_wick > lower_wick:
            return True, "Bearish rejection candle — sellers stepping in"

    return False, ""


def _check_volume_and_rsi(
    df_5m: pd.DataFrame,
    direction: str,
) -> tuple[int, list[str]]:
    """
    Checks volume and RSI to add confidence points.
    Returns (bonus_points, reasons).
    """
    ind = add_indicators(df_5m)
    last = ind.iloc[-1]
    points = 0
    reasons = []

    rsi = last.get("rsi")
    vol_ratio = last.get("vol_ratio")

    if rsi is not None:
        if direction == "long" and 30 <= rsi <= 60:
            points += 1
            reasons.append(f"RSI healthy ({rsi:.1f}) — room to move up")
        elif direction == "short" and 40 <= rsi <= 70:
            points += 1
            reasons.append(f"RSI healthy ({rsi:.1f}) — room to move down")
        # Contradictions reduce confidence
        if direction == "long" and rsi > 75:
            points -= 2
        if direction == "short" and rsi < 25:
            points -= 2

    if vol_ratio is not None:
        if vol_ratio > 1.2:
            points += 1
            reasons.append(f"Volume confirming move ({vol_ratio:.1f}x avg)")
        elif vol_ratio < 0.5:
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
    ALL THREE must confirm or returns (None, 0, []).

    Confidence scoring:
    - Base: 60% (for passing all 3 timeframe checks)
    - 4H trend strength bonus: up to +15%
    - 1H trend strength bonus: up to +15%
    - RSI/Volume bonus: up to +10%
    Total possible: 100%
    """
    reasons = []

    # ── Step 1: 4H Trend (hard requirement) ──────────────────────────────────
    df_4h = _resample_to_4h(df_1h)
    if df_4h is None:
        return None, 0, []

    trend_4h, strength_4h = _get_trend(df_4h)
    if trend_4h == "sideways":
        logger.debug("Triple TF: 4H sideways — blocked")
        return None, 0, []

    direction = "long" if trend_4h == "uptrend" else "short"

    # ── Step 2: 1H Trend must agree with 4H (hard requirement) ───────────────
    trend_1h, strength_1h = _get_trend(df_1h)
    if trend_1h == "sideways" or trend_1h != trend_4h:
        logger.debug(
            f"Triple TF: 1H={trend_1h} disagrees with 4H={trend_4h} — blocked"
        )
        return None, 0, []

    reasons.append(
        f"4H + 1H {trend_4h} aligned "
        f"(4H strength {round(strength_4h * 100)}% | "
        f"1H strength {round(strength_1h * 100)}%)"
    )

    # ── Step 3: Price at pullback level on 1H S/R or EMA ─────────────────────
    at_level, level_reason = _is_at_pullback_level(
        df_5m, direction, support_1h, resistance_1h
    )
    if not at_level:
        logger.debug("Triple TF: not at pullback level — blocked")
        return None, 0, []

    reasons.append(level_reason)

    # ── Step 4: 5m entry confirmation candle ─────────────────────────────────
    confirmed, candle_reason = _get_entry_candle(df_5m, direction)
    if not confirmed:
        logger.debug("Triple TF: no entry candle — blocked")
        return None, 0, []

    reasons.append(candle_reason)

    # ── All three confirmed — calculate confidence ────────────────────────────
    base_confidence = 60.0

    # 4H strength bonus (up to 15%)
    tf_4h_bonus = round(strength_4h * 15, 1)

    # 1H strength bonus (up to 15%)
    tf_1h_bonus = round(strength_1h * 15, 1)

    # RSI/Volume bonus (up to 10%)
    vol_rsi_pts, vol_rsi_reasons = _check_volume_and_rsi(df_5m, direction)
    vol_rsi_bonus = max(0, min(10, vol_rsi_pts * 5))
    reasons.extend(vol_rsi_reasons)

    confidence = round(
        base_confidence + tf_4h_bonus + tf_1h_bonus + vol_rsi_bonus,
        1
    )
    confidence = min(100.0, confidence)

    logger.info(
        f"Triple TF {direction}: conf={confidence}% "
        f"(base=60 + 4H={tf_4h_bonus} + 1H={tf_1h_bonus} "
        f"+ vol/rsi={vol_rsi_bonus})"
    )

    return direction, confidence, reasons
