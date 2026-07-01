"""
Indicator calculations, support/resistance zone detection,
RSI divergence detection, and multi-timeframe trend check.
"""

import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma_fast"] = ta.sma(df["close"], length=7)
    df["ma_mid"] = ta.sma(df["close"], length=25)
    df["ma_slow"] = ta.sma(df["close"], length=50)
    df["rsi"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 2]

    df["vol_avg20"] = df["volume"].rolling(20).mean()

    # ATR for volatility measurement
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    return df


def find_support_resistance(df: pd.DataFrame, lookback: int = 60, tolerance_pct: float = 0.15, min_touches: int = 2):
    """
    Swing-high/swing-low based S/R with minimum touch validation.
    A level only qualifies if price has respected it at least min_touches times.
    Single-touch levels are filtered out as they are unreliable.
    """
    window = df.tail(lookback).reset_index(drop=True)
    highs, lows = [], []

    for i in range(2, len(window) - 2):
        seg = window.iloc[i - 2:i + 3]
        if window["high"].iloc[i] == seg["high"].max():
            highs.append(window["high"].iloc[i])
        if window["low"].iloc[i] == seg["low"].min():
            lows.append(window["low"].iloc[i])

    def cluster_and_validate(levels):
        if not levels:
            return []
        levels = sorted(levels)
        clustered = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clustered[-1][-1]) / clustered[-1][-1] * 100 <= tolerance_pct:
                clustered[-1].append(lvl)
            else:
                clustered.append([lvl])
        # Only keep levels touched at least min_touches times
        validated = [
            sum(c) / len(c) for c in clustered
            if len(c) >= min_touches
        ]
        return validated

    return cluster_and_validate(lows), cluster_and_validate(highs)


def nearest_level(price: float, levels: list[float], direction: str):
    if direction == "below":
        candidates = [l for l in levels if l < price]
        return max(candidates) if candidates else None
    else:
        candidates = [l for l in levels if l > price]
        return min(candidates) if candidates else None


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> str | None:
    """
    Detects bullish or bearish RSI divergence over the lookback window.

    Bullish divergence: price making lower lows but RSI making higher lows
    — shows sellers are weakening, reversal up likely.

    Bearish divergence: price making higher highs but RSI making lower highs
    — shows buyers are weakening, reversal down likely.

    Returns 'bullish', 'bearish', or None.
    """
    if len(df) < lookback:
        return None

    window = df.tail(lookback).reset_index(drop=True)
    prices = window["close"]
    rsi = window["rsi"]

    if rsi.isnull().any():
        return None

    # Find two most recent swing lows for bullish divergence
    swing_lows = [
        i for i in range(2, len(window) - 2)
        if window["low"].iloc[i] == window["low"].iloc[i-2:i+3].min()
    ]

    if len(swing_lows) >= 2:
        i1, i2 = swing_lows[-2], swing_lows[-1]
        price_lower_low = prices.iloc[i2] < prices.iloc[i1]
        rsi_higher_low = rsi.iloc[i2] > rsi.iloc[i1]
        if price_lower_low and rsi_higher_low:
            return "bullish"

    # Find two most recent swing highs for bearish divergence
    swing_highs = [
        i for i in range(2, len(window) - 2)
        if window["high"].iloc[i] == window["high"].iloc[i-2:i+3].max()
    ]

    if len(swing_highs) >= 2:
        i1, i2 = swing_highs[-2], swing_highs[-1]
        price_higher_high = prices.iloc[i2] > prices.iloc[i1]
        rsi_lower_high = rsi.iloc[i2] < rsi.iloc[i1]
        if price_higher_high and rsi_lower_high:
            return "bearish"

    return None


def get_higher_tf_trend(df: pd.DataFrame) -> str:
    """
    Simulates a higher timeframe trend by resampling the 15m candles
    into 1h candles and checking MA structure on those.
    Returns 'uptrend', 'downtrend', or 'sideways'.
    """
    df = df.copy()
    df = df.set_index("timestamp")

    # Resample to 1h
    h1 = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    if len(h1) < 26:
        return "sideways"

    h1["ma_fast"] = ta.sma(h1["close"], length=7)
    h1["ma_slow"] = ta.sma(h1["close"], length=25)

    if h1["ma_fast"].isnull().iloc[-1] or h1["ma_slow"].isnull().iloc[-1]:
        return "sideways"

    last = h1.iloc[-1]
    price = last["close"]

    if last["ma_fast"] > last["ma_slow"] and price > last["ma_fast"]:
        return "uptrend"
    elif last["ma_fast"] < last["ma_slow"] and price < last["ma_fast"]:
        return "downtrend"
    return "sideways"
