"""
Indicator calculations and support/resistance zone detection.
Uses pandas-ta for standard indicators.
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
    return df


def find_support_resistance(df: pd.DataFrame, lookback: int = 60, tolerance_pct: float = 0.15):
    """
    Simple swing-high/swing-low based S/R: finds local extremes over the
    lookback window and clusters nearby levels together. Returns
    (support_levels, resistance_levels) as sorted lists of price floats.
    """
    window = df.tail(lookback).reset_index(drop=True)
    highs, lows = [], []

    for i in range(2, len(window) - 2):
        seg = window.iloc[i - 2:i + 3]
        if window["high"].iloc[i] == seg["high"].max():
            highs.append(window["high"].iloc[i])
        if window["low"].iloc[i] == seg["low"].min():
            lows.append(window["low"].iloc[i])

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        clustered = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clustered[-1][-1]) / clustered[-1][-1] * 100 <= tolerance_pct:
                clustered[-1].append(lvl)
            else:
                clustered.append([lvl])
        return [sum(c) / len(c) for c in clustered]

    return cluster(lows), cluster(highs)


def nearest_level(price: float, levels: list[float], direction: str):
    """
    direction='below' finds nearest support under price.
    direction='above' finds nearest resistance over price.
    """
    if direction == "below":
        candidates = [l for l in levels if l < price]
        return max(candidates) if candidates else None
    else:
        candidates = [l for l in levels if l > price]
        return min(candidates) if candidates else None
