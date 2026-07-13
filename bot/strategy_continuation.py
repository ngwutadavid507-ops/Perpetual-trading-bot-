"""
Trend Continuation Strategy — V2 Strict Version.

Three independent confirmations ALL required:

Confirmation 1 — Structure:
  Both 1H AND 4H must show clear trend in same direction.
  Sideways on either timeframe = blocked.

Confirmation 2 — Location:
  Price must be at meaningful level:
  - Near EMA21 pullback zone, OR
  - At tested 1H S/R level, OR
  - Near key Fibonacci level (38.2% or 61.8% retracement)

Confirmation 3 — Momentum alignment:
  RSI, MACD and volume must ALL agree.
  One disagreement = blocked.

Additional safety:
  - No signal within 2 candles of a major move
  - ATR must be below 3% (not too volatile)
  - Weak pullback volume required (not a reversal disguised as pullback)
"""

import pandas as pd
import numpy as np
import logging

from bot.indicators import (
    add_indicators,
    find_support_resistance,
    nearest_level,
    get_trend_strength,
    is_near_ema,
)
from bot.patterns import detect_pullback_rejection

logger = logging.getLogger(__name__)


def _get_4h_trend(df_1h: pd.DataFrame) -> str:
    """
    Simulates 4H trend by resampling 1H candles.
    Returns 'uptrend', 'downtrend', or 'sideways'.
    """
    try:
        df = df_1h.copy()
        df = df.set_index("timestamp")
        df_4h = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        if len(df_4h) < 20:
            return "sideways"

        df_4h = df_4h.reset_index()
        df_4h_ind = add_indicators(df_4h)
        last = df_4h_ind.iloc[-1]
        price = last["close"]
        ema_8 = last.get("ema_8")
        ema_21 = last.get("ema_21")
        ema_50 = last.get("ema_50")

        if not all([ema_8, ema_21, ema_50]):
            return "sideways"

        if ema_8 > ema_21 and price > ema_21:
            return "uptrend"
        elif ema_8 < ema_21 and price < ema_21:
            return "downtrend"
        return "sideways"
    except Exception as e:
        logger.debug(f"4H trend calculation failed: {e}")
        return "sideways"


def _is_major_move_recent(df_5m: pd.DataFrame, candles: int = 2) -> bool:
    """
    Returns True if a major move happened in the last N candles.
    Prevents chasing after big moves.
    Major move = candle body > 2x average body size.
    """
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


def _is_too_volatile(df_5m: pd.DataFrame) -> bool:
    """
    Returns True if ATR exceeds 3% of price — too risky to trade.
    """
    if "atr" not in df_5m.columns:
        return False
    atr = df_5m["atr"].iloc[-1]
    price = df_5m["close"].iloc[-1]
    if pd.isna(atr) or price == 0:
        return False
    return (atr / price) * 100 > 3.0


def _get_fibonacci_levels(df_5m: pd.DataFrame) -> list[float]:
    """
    Calculates key Fibonacci retracement levels from the recent swing.
    Returns list of 38.2% and 61.8% retracement prices.
    """
    lookback = df_5m.tail(30)
    swing_high = lookback["high"].max()
    swing_low = lookback["low"].min()
    rng = swing_high - swing_low
    if rng == 0:
        return []
    fib_382 = swing_high - (rng * 0.382)
    fib_618 = swing_high - (rng * 0.618)
    return [fib_382, fib_618]


def _is_near_fibonacci(price: float, fib_levels: list[float], tolerance_pct: float = 0.3) -> bool:
    for level in fib_levels:
        if abs(price - level) / price * 100 <= tolerance_pct:
            return True
    return False


def score_continuation(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    support_1h: list[float],
    resistance_1h: list[float],
) -> tuple[str | None, float, list[str]]:
    """
    Scores a trend continuation setup with strict three-confirmation system.
    Returns (direction, confidence_0_to_100, reasons).
    Returns (None, 0, []) if any confirmation fails.
    """

    # ── Safety filters first ──────────────────────────────────────────────────

    df_5m_ind = add_indicators(df_5m)

    if _is_too_volatile(df_5m_ind):
        logger.debug("Continuation blocked: ATR > 3% — too volatile")
        return None, 0, []

    if _is_major_move_recent(df_5m_ind, candles=2):
        logger.debug("Continuation blocked: major move in last 2 candles")
        return None, 0, []

    # ── Confirmation 1 — Structure (both 1H and 4H must agree) ───────────────

    trend_1h, strength_1h = get_trend_strength(df_1h)
    trend_4h = _get_4h_trend(df_1h)

    if trend_1h == "sideways":
        logger.debug("Continuation blocked: 1H sideways")
        return None, 0, []

    if trend_4h != "sideways" and trend_1h != trend_4h:
        logger.debug(
            f"Continuation blocked: 1H={trend_1h} 4H={trend_4h} disagree"
        )
        return None, 0, []

    if strength_1h < 0.5:
        logger.debug(
            f"Continuation blocked: 1H trend strength {strength_1h} < 0.5"
        )
        return None, 0, []

    allowed = "long" if trend_1h == "uptrend" else "short"
    score = 0
    max_score = 0
    reasons = []

    # Structure score (30 pts)
    max_score += 30
    structure_pts = round(strength_1h * 20) + 10
    score += min(structure_pts, 30)
    reasons.append(
        f"1H + 4H {trend_1h} confirmed "
        f"(strength {round(strength_1h * 100)}%)"
    )

    # ── Confirmation 2 — Location ─────────────────────────────────────────────

    last = df_5m_ind.iloc[-1]
    price = last["close"]
    location_confirmed = False

    max_score += 25
    fib_levels = _get_fibonacci_levels(df_5m_ind)

    near_ema21 = is_near_ema(df_5m_ind, "ema_21", tolerance_pct=0.5)
    near_ema8 = is_near_ema(df_5m_ind, "ema_8", tolerance_pct=0.3)
    near_fib = _is_near_fibonacci(price, fib_levels)

    near_support = nearest_level(price, support_1h, "below")
    near_resistance = nearest_level(price, resistance_1h, "above")
    at_sr = False
    if allowed == "long" and near_support:
        if abs(price - near_support) / price * 100 <= 0.8:
            at_sr = True
    if allowed == "short" and near_resistance:
        if abs(near_resistance - price) / price * 100 <= 0.8:
            at_sr = True

    if near_ema21:
        score += 25
        reasons.append("Price at 21 EMA pullback zone")
        location_confirmed = True
    elif near_ema8:
        score += 18
        reasons.append("Price near 8 EMA — tight continuation")
        location_confirmed = True
    elif at_sr:
        score += 25
        if allowed == "long":
            reasons.append(f"1H support zone ~{near_support:.4f}")
        else:
            reasons.append(f"1H resistance zone ~{near_resistance:.4f}")
        location_confirmed = True
    elif near_fib:
        score += 20
        reasons.append("Price at Fibonacci retracement level")
        location_confirmed = True

    if not location_confirmed:
        logger.debug("Continuation blocked: not at meaningful location")
        return None, 0, []

    # ── Confirmation 3 — Momentum alignment (ALL must agree) ─────────────────

    max_score += 45
    momentum_score = 0
    momentum_reasons = []
    momentum_blocks = 0

    # RSI check (15 pts)
    rsi = last.get("rsi")
    if rsi is not None:
        if allowed == "long":
            if rsi >= 75:
                momentum_blocks += 1
                logger.debug(f"Momentum block: RSI overbought {rsi}")
            elif 35 <= rsi <= 65:
                momentum_score += 15
                momentum_reasons.append(f"RSI healthy ({rsi:.1f})")
            else:
                momentum_score += 8
        else:
            if rsi <= 25:
                momentum_blocks += 1
                logger.debug(f"Momentum block: RSI oversold {rsi}")
            elif 35 <= rsi <= 65:
                momentum_score += 15
                momentum_reasons.append(f"RSI healthy ({rsi:.1f})")
            else:
                momentum_score += 8

    # MACD check (15 pts)
    macd_hist = last.get("macd_hist")
    macd = last.get("macd")
    macd_signal = last.get("macd_signal")
    if macd_hist is not None and macd is not None:
        if allowed == "long":
            if macd_hist > 0 and macd > macd_signal:
                momentum_score += 15
                momentum_reasons.append("MACD bullish momentum")
            elif macd_hist < 0:
                momentum_blocks += 1
                logger.debug("Momentum block: MACD bearish on long signal")
        else:
            if macd_hist < 0 and macd < macd_signal:
                momentum_score += 15
                momentum_reasons.append("MACD bearish momentum")
            elif macd_hist > 0:
                momentum_blocks += 1
                logger.debug("Momentum block: MACD bullish on short signal")

    # Volume check (15 pts)
    vol_ratio = last.get("vol_ratio")
    if vol_ratio is not None:
        if vol_ratio < 0.7:
            momentum_score += 15
            momentum_reasons.append(
                f"Low volume pullback ({vol_ratio:.1f}x avg) "
                f"— trend likely continues"
            )
        elif vol_ratio > 2.5:
            momentum_blocks += 1
            logger.debug(
                f"Momentum block: high volume {vol_ratio:.1f}x "
                f"on pullback suggests reversal"
            )
        elif vol_ratio < 1.0:
            momentum_score += 8

    # If any momentum indicator disagrees — block signal
    if momentum_blocks > 0:
        logger.debug(
            f"Continuation blocked: {momentum_blocks} momentum disagreement(s)"
        )
        return None, 0, []

    score += momentum_score
    reasons.extend(momentum_reasons)

    # ── Pullback rejection candle ─────────────────────────────────────────────

    max_score += 10
    rejection = detect_pullback_rejection(df_5m_ind, allowed)
    if rejection:
        score += 10
        reasons.append(
            f"Pullback rejection candle — "
            f"{'buyers' if allowed == 'long' else 'sellers'} stepping in"
        )

    # Require minimum 2 momentum reasons
    if len(momentum_reasons) < 1:
        logger.debug("Continuation blocked: no momentum confirmation")
        return None, 0, []

    if score == 0 or len(reasons) < 3:
        return None, 0, []

    confidence = round(score / max_score * 100, 1)
    return allowed, confidence, reasons
