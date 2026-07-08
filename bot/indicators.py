"""
Technical indicator calculations and S/R zone detection.
Used by both continuation and reversal strategies.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds all technical indicators to the dataframe.
    Works on any timeframe.
    """
    df = df.copy()

    # Moving averages
    df["ema_8"] = ta.ema(df["close"], length=8)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ma_fast"] = ta.sma(df["close"], length=7)
    df["ma_mid"] = ta.sma(df["close"], length=25)
    df["ma_slow"] = ta.sma(df["close"], length=50)

    # Momentum
    df["rsi"] = ta.rsi(df["close"], length=14)

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 2]
        df["macd_hist"] = macd.iloc[:, 1]

    # Volatility
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Volume
    df["vol_avg20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_avg20"].replace(0, 1)

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["bb_upper"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 2]

    return df


def find_support_resistance(
    df: pd.DataFrame,
    lookback: int = 80,
    tolerance_pct: float = 0.15,
    min_touches: int = 2,
) -> tuple[list[float], list[float]]:
    """
    Swing-high/swing-low based S/R detection.
    Only returns levels touched at least min_touches times.
    Single-touch levels are filtered as unreliable.
    """
    window = df.tail(lookback).reset_index(drop=True)
    highs, lows = [], []

    for i in range(2, len(window) - 2):
        seg = window.iloc[i - 2:i + 3]
        if window["high"].iloc[i] == seg["high"].max():
            highs.append(window["high"].iloc[i])
        if window["low"].iloc[i] == seg["low"].min():
            lows.append(window["low"].iloc[i])

    def cluster_and_validate(levels: list) -> list[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clustered = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clustered[-1][-1]) / clustered[-1][-1] * 100 <= tolerance_pct:
                clustered[-1].append(lvl)
            else:
                clustered.append([lvl])
        return [
            sum(c) / len(c) for c in clustered
            if len(c) >= min_touches
        ]

    return cluster_and_validate(lows), cluster_and_validate(highs)


def nearest_level(
    price: float,
    levels: list[float],
    direction: str,
) -> float | None:
    """
    Finds the nearest S/R level in the given direction.
    direction='below' for support, 'above' for resistance.
    """
    if direction == "below":
        candidates = [l for l in levels if l < price]
        return max(candidates) if candidates else None
    else:
        candidates = [l for l in levels if l > price]
        return min(candidates) if candidates else None


def get_trend_strength(df_1h: pd.DataFrame) -> tuple[str, float]:
    """
    Determines trend direction and strength from 1h candles.
    Returns (direction, strength_0_to_1).
    direction: 'uptrend', 'downtrend', or 'sideways'
    strength: 0.0 (weak) to 1.0 (very strong)
    """
    df = add_indicators(df_1h)
    last = df.iloc[-1]
    price = last["close"]

    ema_8 = last.get("ema_8")
    ema_21 = last.get("ema_21")
    ema_50 = last.get("ema_50")

    if not all([ema_8, ema_21, ema_50]):
        return "sideways", 0.0

    score = 0
    max_score = 0

    # EMA stack alignment (most important)
    max_score += 3
    if ema_8 > ema_21 > ema_50:
        score += 3
        direction = "uptrend"
    elif ema_8 < ema_21 < ema_50:
        score += 3
        direction = "downtrend"
    elif ema_8 > ema_21:
        score += 1
        direction = "uptrend"
    elif ema_8 < ema_21:
        score += 1
        direction = "downtrend"
    else:
        direction = "sideways"

    if direction == "sideways":
        return "sideways", 0.0

    # Price position relative to EMAs
    max_score += 2
    if direction == "uptrend":
        if price > ema_8:
            score += 2
        elif price > ema_21:
            score += 1
    else:
        if price < ema_8:
            score += 2
        elif price < ema_21:
            score += 1

    # MACD histogram direction
    max_score += 1
    macd_hist = last.get("macd_hist")
    if macd_hist is not None:
        if direction == "uptrend" and macd_hist > 0:
            score += 1
        elif direction == "downtrend" and macd_hist < 0:
            score += 1

    # RSI alignment
    max_score += 1
    rsi = last.get("rsi")
    if rsi is not None:
        if direction == "uptrend" and rsi > 50:
            score += 1
        elif direction == "downtrend" and rsi < 50:
            score += 1

    strength = round(score / max_score, 2)
    return direction, strength


def is_near_ema(
    df: pd.DataFrame,
    ema_col: str = "ema_21",
    tolerance_pct: float = 0.3,
) -> bool:
    """
    Checks if the current price is within tolerance_pct of an EMA.
    Used to detect pullback-to-EMA setups for continuation trades.
    """
    if ema_col not in df.columns:
        return False
    last = df.iloc[-1]
    ema_val = last.get(ema_col)
    if not ema_val:
        return False
    price = last["close"]
    return abs(price - ema_val) / price * 100 <= tolerance_pct
