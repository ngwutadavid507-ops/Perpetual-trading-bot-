"""
Trend Continuation Strategy.

Entry logic:
- 1H trend clearly established (EMA stack confirmed, strength > 0.5)
- Price pulls back to EMA21 or key support/resistance on 5m
- Pullback candle shows rejection (small body, wick toward trend)
- RSI between 35-65 on 5m (not extended, room to move)
- Volume on pullback lower than average (weak pullback = strong continuation)
- Entry on confirmation candle

Higher win rate than reversals but smaller individual moves.
Minimum confidence: 65%
"""

import pandas as pd
import logging

from bot.indicators import (
    add_indicators,
    find_support_resistance,
    nearest_level,
    get_trend_strength,
    is_near_ema,
)
from bot.patterns import (
    detect_pullback_rejection,
    detect_momentum_candle,
)

logger = logging.getLogger(__name__)


def score_continuation(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    support_1h: list[float],
    resistance_1h: list[float],
) -> tuple[str | None, float, list[str]]:
    """
    Scores a trend continuation setup.
    Returns (direction, confidence_0_to_100, reasons).
    Returns (None, 0, []) if no valid setup found.
    """

    # Step 1 — establish 1H trend (hard requirement)
    trend_direction, trend_strength = get_trend_strength(df_1h)

    if trend_direction == "sideways":
        return None, 0, []

    if trend_strength < 0.4:
        # Trend too weak for continuation trade
        return None, 0, []

    allowed = "long" if trend_direction == "uptrend" else "short"

    df_5m_ind = add_indicators(df_5m)
    last = df_5m_ind.iloc[-1]
    price = last["close"]
    score = 0
    max_score = 0
    reasons = []

    # Step 2 — 1H trend strength bonus (20 pts)
    max_score += 20
    trend_pts = round(trend_strength * 20)
    score += trend_pts
    reasons.append(
        f"1H {trend_direction} confirmed "
        f"(strength {round(trend_strength * 100)}%)"
    )

    # Step 3 — price near EMA21 pullback zone (25 pts)
    max_score += 25
    near_ema21 = is_near_ema(df_5m_ind, "ema_21", tolerance_pct=0.5)
    near_ema8 = is_near_ema(df_5m_ind, "ema_8", tolerance_pct=0.3)

    if near_ema21:
        score += 25
        reasons.append("Price pulling back to 21 EMA — continuation zone")
    elif near_ema8:
        score += 15
        reasons.append("Price near 8 EMA — tight continuation setup")

    # Step 4 — S/R level confluence (20 pts)
    max_score += 20
    if allowed == "long":
        near_support = nearest_level(price, support_1h, "below")
        if near_support and abs(price - near_support) / price * 100 <= 1.0:
            score += 20
            reasons.append(f"1H support zone ~{near_support:.4f} confirms pullback")
    else:
        near_resistance = nearest_level(price, resistance_1h, "above")
        if near_resistance and abs(near_resistance - price) / price * 100 <= 1.0:
            score += 20
            reasons.append(f"1H resistance zone ~{near_resistance:.4f} confirms rally")

    # Step 5 — pullback rejection candle (15 pts)
    max_score += 15
    rejection = detect_pullback_rejection(df_5m_ind, allowed)
    if rejection:
        score += 15
        reasons.append(
            f"Pullback rejection candle — "
            f"{'buyers' if allowed == 'long' else 'sellers'} stepping in"
        )

    # Step 6 — RSI in healthy zone (10 pts)
    # Not overbought/oversold — room to continue
    max_score += 10
    rsi = last.get("rsi")
    if rsi is not None:
        if allowed == "long" and 35 <= rsi <= 60:
            score += 10
            reasons.append(f"RSI healthy ({rsi:.1f}) — room to move up")
        elif allowed == "short" and 40 <= rsi <= 65:
            score += 10
            reasons.append(f"RSI healthy ({rsi:.1f}) — room to move down")
        # Hard block — RSI too extreme for continuation
        if allowed == "long" and rsi >= 80:
            return None, 0, []
        if allowed == "short" and rsi <= 20:
            return None, 0, []

    # Step 7 — weak pullback volume (10 pts)
    # Lower volume on pullback = healthy retracement, not reversal
    max_score += 10
    vol_ratio = last.get("vol_ratio")
    if vol_ratio is not None:
        if vol_ratio < 0.8:
            score += 10
            reasons.append("Low volume pullback — trend likely to continue")
        elif vol_ratio > 2.0:
            # High volume on pullback could signal reversal, not continuation
            return None, 0, []

    if score == 0 or len(reasons) < 2:
        return None, 0, []

    confidence = round(score / max_score * 100, 1)
    return allowed, confidence, reasons
