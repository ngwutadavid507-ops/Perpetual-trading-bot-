"""
Candlestick pattern detection on the most recently CLOSED candle.
Implements the patterns covered in the lesson: engulfing, pin bar, doji.
Each function returns "bullish", "bearish", or None.
"""

import pandas as pd


def _body(row):
    return abs(row["close"] - row["open"])


def _range(row):
    return row["high"] - row["low"]


def detect_engulfing(df: pd.DataFrame) -> str | None:
    if len(df) < 2:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]

    prev_bullish = prev["close"] > prev["open"]
    curr_bullish = curr["close"] > curr["open"]

    # Bearish engulfing: green candle followed by a red candle that engulfs its body
    if prev_bullish and not curr_bullish:
        if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
            return "bearish"

    # Bullish engulfing: red candle followed by a green candle that engulfs its body
    if not prev_bullish and curr_bullish:
        if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
            return "bullish"

    return None


def detect_pin_bar(df: pd.DataFrame, wick_to_body_ratio: float = 2.0) -> str | None:
    if len(df) < 1:
        return None
    row = df.iloc[-1]
    body = _body(row)
    full_range = _range(row)
    if full_range == 0 or body == 0:
        return None

    upper_wick = row["high"] - max(row["close"], row["open"])
    lower_wick = min(row["close"], row["open"]) - row["low"]

    # Bearish pin bar: long upper wick, small body near the low
    if upper_wick >= body * wick_to_body_ratio and upper_wick > lower_wick:
        return "bearish"

    # Bullish pin bar: long lower wick, small body near the high
    if lower_wick >= body * wick_to_body_ratio and lower_wick > upper_wick:
        return "bullish"

    return None


def detect_doji(df: pd.DataFrame, body_to_range_max: float = 0.1) -> bool:
    if len(df) < 1:
        return False
    row = df.iloc[-1]
    full_range = _range(row)
    if full_range == 0:
        return False
    return (_body(row) / full_range) <= body_to_range_max
