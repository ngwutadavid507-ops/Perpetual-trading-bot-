"""
Candlestick pattern detection.
Detects both reversal and continuation patterns.
Each function returns 'bullish', 'bearish', or None.
"""

import pandas as pd


def _body(row) -> float:
    return abs(row["close"] - row["open"])


def _range(row) -> float:
    return row["high"] - row["low"]


def _upper_wick(row) -> float:
    return row["high"] - max(row["close"], row["open"])


def _lower_wick(row) -> float:
    return min(row["close"], row["open"]) - row["low"]


# ─── Reversal Patterns ────────────────────────────────────────────────────────

def detect_engulfing(df: pd.DataFrame) -> str | None:
    """
    Bearish engulfing: green candle followed by red candle
    whose body completely swallows the previous body.
    Bullish engulfing: red candle followed by green candle
    whose body completely swallows the previous body.
    """
    if len(df) < 2:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]

    prev_bullish = prev["close"] > prev["open"]
    curr_bullish = curr["close"] > curr["open"]

    if prev_bullish and not curr_bullish:
        if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
            return "bearish"

    if not prev_bullish and curr_bullish:
        if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
            return "bullish"

    return None


def detect_pin_bar(df: pd.DataFrame, wick_ratio: float = 2.0) -> str | None:
    """
    Bearish pin bar: long upper wick, small body near the low.
    Bullish pin bar: long lower wick, small body near the high.
    Wick must be at least wick_ratio times the body length.
    """
    if len(df) < 1:
        return None
    row = df.iloc[-1]
    body = _body(row)
    full_range = _range(row)
    if full_range == 0 or body == 0:
        return None

    upper = _upper_wick(row)
    lower = _lower_wick(row)

    if upper >= body * wick_ratio and upper > lower * 2:
        return "bearish"

    if lower >= body * wick_ratio and lower > upper * 2:
        return "bullish"

    return None


def detect_doji(df: pd.DataFrame, body_ratio: float = 0.1) -> bool:
    """
    Doji: body is less than body_ratio of the full candle range.
    Signals indecision — useful as a reversal warning.
    """
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    full_range = _range(row)
    if full_range == 0:
        return False
    return (_body(row) / full_range) <= body_ratio


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> str | None:
    """
    Bullish divergence: price making lower lows but RSI making higher lows.
    Bearish divergence: price making higher highs but RSI making lower highs.
    Returns 'bullish', 'bearish', or None.
    """
    if len(df) < lookback or "rsi" not in df.columns:
        return None

    window = df.tail(lookback).reset_index(drop=True)
    prices = window["close"]
    rsi = window["rsi"]

    if rsi.isnull().any():
        return None

    # Bullish divergence — swing lows
    swing_lows = [
        i for i in range(2, len(window) - 2)
        if window["low"].iloc[i] == window["low"].iloc[i - 2:i + 3].min()
    ]
    if len(swing_lows) >= 2:
        i1, i2 = swing_lows[-2], swing_lows[-1]
        if prices.iloc[i2] < prices.iloc[i1] and rsi.iloc[i2] > rsi.iloc[i1]:
            return "bullish"

    # Bearish divergence — swing highs
    swing_highs = [
        i for i in range(2, len(window) - 2)
        if window["high"].iloc[i] == window["high"].iloc[i - 2:i + 3].max()
    ]
    if len(swing_highs) >= 2:
        i1, i2 = swing_highs[-2], swing_highs[-1]
        if prices.iloc[i2] > prices.iloc[i1] and rsi.iloc[i2] < rsi.iloc[i1]:
            return "bearish"

    return None


# ─── Continuation Patterns ────────────────────────────────────────────────────

def detect_pullback_rejection(df: pd.DataFrame, direction: str) -> bool:
    """
    Detects a weak pullback rejection candle in a trending market.
    For longs: a small candle near support with a lower wick showing buyers stepped in.
    For shorts: a small candle near resistance with an upper wick showing sellers stepped in.
    Used for trend continuation entries.
    """
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    body = _body(row)
    full_range = _range(row)
    if full_range == 0:
        return False

    upper = _upper_wick(row)
    lower = _lower_wick(row)
    body_ratio = body / full_range

    if direction == "long":
        # Small body with lower wick — buyers rejecting the pullback
        return body_ratio <= 0.5 and lower >= body * 1.2

    if direction == "short":
        # Small body with upper wick — sellers rejecting the rally
        return body_ratio <= 0.5 and upper >= body * 1.2

    return False


def detect_momentum_candle(df: pd.DataFrame, direction: str) -> bool:
    """
    Strong momentum candle in trend direction.
    Body must be at least 60% of the full candle range.
    Used to confirm trend strength for continuation signals.
    """
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    body = _body(row)
    full_range = _range(row)
    if full_range == 0:
        return False

    is_bullish = row["close"] > row["open"]
    body_ratio = body / full_range

    if direction == "long" and is_bullish and body_ratio >= 0.6:
        return True
    if direction == "short" and not is_bullish and body_ratio >= 0.6:
        return True

    return False
