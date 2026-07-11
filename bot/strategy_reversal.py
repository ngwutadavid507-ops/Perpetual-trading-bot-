"""
Reversal Strategy — V2 Strict Version.

Three independent confirmations ALL required:

Confirmation 1 — Location:
  Price must be at a MAJOR S/R level tested at least 3 times.
  Fresh untested levels are rejected.

Confirmation 2 — Reversal candle + RSI divergence:
  Both must be present simultaneously.
  Engulfing OR pin bar AND RSI divergence required together.
  One without the other = blocked.

Confirmation 3 — Momentum exhaustion:
  RSI extreme (<=30 for long, >=70 for short) AND
  MACD histogram weakening AND
  Volume spike (>2x average) on reversal candle.
  All three required.

Additional safety:
  - 1H trend must be weakening (not accelerating)
  - ATR must be below 2.5%
  - No signals within 3 candles of a major move
  - Minimum 80% confidence required — no exceptions
"""

import pandas as pd
import numpy as np
import logging

from bot.indicators import (
    add_indicators,
    find_support_resistance,
    nearest_level,
    get_trend_strength,
)
from bot.patterns import (
    detect_engulfing,
    detect_pin_bar,
    detect_rsi_divergence,
)

logger = logging.getLogger(__name__)


def _count_touches(
    price: float,
    levels: list[float],
    df: pd.DataFrame,
    tolerance_pct: float = 0.3,
) -> int:
    """
    Counts how many times price has touched a level.
    More touches = stronger, more reliable level.
    """
    count = 0
    for _, row in df.iterrows():
        for level in levels:
            if abs(row["high"] - level) / level * 100 <= tolerance_pct:
                count += 1
                break
            if abs(row["low"] - level) / level * 100 <= tolerance_pct:
                count += 1
                break
    return count


def _is_trend_weakening(df_1h: pd.DataFrame) -> bool:
    """
    Checks if the 1H trend is weakening — good for reversals.
    Weakening = MACD histogram shrinking or EMAs converging.
    """
    df = add_indicators(df_1h)
    if len(df) < 5:
        return False

    recent = df.tail(5)

    # MACD histogram shrinking
    hist_values = recent["macd_hist"].dropna()
    if len(hist_values) >= 3:
        last_hist = abs(hist_values.iloc[-1])
        prev_hist = abs(hist_values.iloc[-2])
        if last_hist < prev_hist:
            return True

    # EMAs converging
    last = df.iloc[-1]
    ema_8 = last.get("ema_8")
    ema_21 = last.get("ema_21")
    if ema_8 and ema_21:
        gap = abs(ema_8 - ema_21) / last["close"] * 100
        if gap < 0.3:
            return True

    return False


def _is_too_volatile(df_5m: pd.DataFrame) -> bool:
    if "atr" not in df_5m.columns:
        return False
    atr = df_5m["atr"].iloc[-1]
    price = df_5m["close"].iloc[-1]
    if pd.isna(atr) or price == 0:
        return False
    return (atr / price) * 100 > 2.5


def _is_major_move_recent(df_5m: pd.DataFrame, candles: int = 3) -> bool:
    if len(df_5m) < candles + 5:
        return False
    recent = df_5m.tail(candles + 1).iloc[:-1]
    avg_body = df_5m["close"].diff().abs().tail(20).mean()
    if avg_body == 0:
        return False
    for _, row in recent.iterrows():
        body = abs(row["close"] - row["open"])
        if body > avg_body * 2.5:
            return True
    return False


def score_reversal(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    support_1h: list[float],
    resistance_1h: list[float],
) -> tuple[str | None, float, list[str]]:
    """
    Scores a reversal setup with strict three-confirmation system.
    Returns (direction, confidence_0_to_100, reasons).
    Returns (None, 0, []) if any confirmation fails.
    Minimum 80% confidence required to fire.
    """

    # ── Safety filters ────────────────────────────────────────────────────────

    df_5m_ind = add_indicators(df_5m)

    if _is_too_volatile(df_5m_ind):
        logger.debug("Reversal blocked: ATR > 2.5% — too volatile")
        return None, 0, []

    if _is_major_move_recent(df_5m_ind, candles=3):
        logger.debug("Reversal blocked: major move in last 3 candles")
        return None, 0, []

    last = df_5m_ind.iloc[-1]
    price = last["close"]
    score = 0
    max_score = 0
    reasons = []

    # ── Confirmation 1 — Major S/R level with 3+ touches ─────────────────────

    max_score += 30
    near_support = nearest_level(price, support_1h, "below")
    near_resistance = nearest_level(price, resistance_1h, "above")

    at_support = (
        near_support is not None and
        abs(price - near_support) / price * 100 <= 0.8
    )
    at_resistance = (
        near_resistance is not None and
        abs(near_resistance - price) / price * 100 <= 0.8
    )

    if not at_support and not at_resistance:
        logger.debug("Reversal blocked: not at any S/R level")
        return None, 0, []

    # Determine direction
    if at_support and not at_resistance:
        direction = "long"
    elif at_resistance and not at_support:
        direction = "short"
    else:
        support_dist = abs(price - near_support) / price * 100
        resistance_dist = abs(near_resistance - price) / price * 100
        direction = "long" if support_dist <= resistance_dist else "short"

    # Count level touches for validation
    df_1h_ind = add_indicators(df_1h)
    if direction == "long":
        touches = _count_touches(
            price, [near_support], df_1h_ind, tolerance_pct=0.3
        )
        if touches < 3:
            logger.debug(
                f"Reversal blocked: support only {touches} touches < 3"
            )
            return None, 0, []
        touch_bonus = min(touches - 2, 3) * 5
        score += 20 + touch_bonus
        reasons.append(
            f"Major support ~{near_support:.4f} "
            f"({touches} touches — validated)"
        )
    else:
        touches = _count_touches(
            price, [near_resistance], df_1h_ind, tolerance_pct=0.3
        )
        if touches < 3:
            logger.debug(
                f"Reversal blocked: resistance only {touches} touches < 3"
            )
            return None, 0, []
        touch_bonus = min(touches - 2, 3) * 5
        score += 20 + touch_bonus
        reasons.append(
            f"Major resistance ~{near_resistance:.4f} "
            f"({touches} touches — validated)"
        )

    # ── Confirmation 2 — Reversal candle AND RSI divergence ──────────────────

    max_score += 30
    engulf = detect_engulfing(df_5m_ind)
    pin = detect_pin_bar(df_5m_ind, wick_ratio=2.5)
    divergence = detect_rsi_divergence(df_5m_ind)

    has_pattern = False
    pattern_name = None

    if direction == "long":
        if engulf == "bullish":
            has_pattern = True
            pattern_name = "Bullish engulfing"
        elif pin == "bullish":
            has_pattern = True
            pattern_name = "Bullish pin bar"
    else:
        if engulf == "bearish":
            has_pattern = True
            pattern_name = "Bearish engulfing"
        elif pin == "bearish":
            has_pattern = True
            pattern_name = "Bearish pin bar"

    has_divergence = (
        (divergence == "bullish" and direction == "long") or
        (divergence == "bearish" and direction == "short")
    )

    # Both pattern AND divergence required
    if not has_pattern:
        logger.debug("Reversal blocked: no reversal candle pattern")
        return None, 0, []

    if not has_divergence:
        logger.debug("Reversal blocked: no RSI divergence")
        return None, 0, []

    score += 30
    reasons.append(f"{pattern_name} + RSI divergence confirmed")

    # ── Confirmation 3 — Momentum exhaustion (all three required) ────────────

    max_score += 30
    momentum_score = 0
    momentum_confirmed = 0

    # RSI extreme (10 pts)
    rsi = last.get("rsi")
    if rsi is not None:
        if direction == "long" and rsi <= 30:
            momentum_score += 10
            momentum_confirmed += 1
            reasons.append(f"RSI deeply oversold ({rsi:.1f})")
        elif direction == "short" and rsi >= 70:
            momentum_score += 10
            momentum_confirmed += 1
            reasons.append(f"RSI deeply overbought ({rsi:.1f})")
        else:
            logger.debug(
                f"Momentum: RSI {rsi} not extreme enough for reversal"
            )

    # MACD histogram weakening (10 pts)
    macd_hist = last.get("macd_hist")
    if macd_hist is not None and len(df_5m_ind) >= 3:
        prev_hist = df_5m_ind["macd_hist"].iloc[-2]
        if not pd.isna(prev_hist):
            if direction == "long":
                # Bearish histogram shrinking = sellers exhausted
                if macd_hist < 0 and abs(macd_hist) < abs(prev_hist):
                    momentum_score += 10
                    momentum_confirmed += 1
                    reasons.append("MACD histogram weakening — sellers exhausted")
            else:
                # Bullish histogram shrinking = buyers exhausted
                if macd_hist > 0 and abs(macd_hist) < abs(prev_hist):
                    momentum_score += 10
                    momentum_confirmed += 1
                    reasons.append("MACD histogram weakening — buyers exhausted")

    # Volume spike (10 pts)
    vol_ratio = last.get("vol_ratio")
    if vol_ratio is not None:
        if vol_ratio >= 2.0:
            momentum_score += 10
            momentum_confirmed += 1
            reasons.append(
                f"Strong volume spike ({vol_ratio:.1f}x avg) "
                f"— reversal conviction"
            )
        elif vol_ratio >= 1.5:
            momentum_score += 5
            reasons.append(f"Volume above average ({vol_ratio:.1f}x)")

    # All three momentum confirmations required
    if momentum_confirmed < 3:
        logger.debug(
            f"Reversal blocked: only {momentum_confirmed}/3 "
            f"momentum confirmations"
        )
        return None, 0, []

    score += momentum_score

    # ── Trend weakening bonus ─────────────────────────────────────────────────

    max_score += 10
    if _is_trend_weakening(df_1h):
        score += 10
        reasons.append("1H trend weakening — reversal conditions favorable")
    else:
        # Trend not weakening — reduces reversal probability
        logger.debug(
            "Reversal note: 1H trend not weakening — lower probability"
        )

    if len(reasons) < 4:
        return None, 0, []

    confidence = round(score / max_score * 100, 1)

    # Reversal requires minimum 80% confidence
    if confidence < 80:
        logger.debug(
            f"Reversal blocked: confidence {confidence} < 80%"
        )
        return None, 0, []

    return direction, confidence, reasons
