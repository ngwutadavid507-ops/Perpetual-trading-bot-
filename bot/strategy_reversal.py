"""
Reversal Strategy.

Entry logic:
- Price at a major 1H or 4H S/R level tested at least 2 times
- Strong reversal candle (engulfing or pin bar with body >30% of range)
- RSI divergence present (price making new extreme but RSI not confirming)
- Volume spike on reversal candle (>1.8x average — real conviction)
- 1H trend weakening (EMAs converging or flattening)

Lower win rate than continuation but bigger moves.
Minimum confidence: 80% — stricter than continuation.
"""

import pandas as pd
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


def score_reversal(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    support_1h: list[float],
    resistance_1h: list[float],
) -> tuple[str | None, float, list[str]]:
    """
    Scores a reversal setup.
    Returns (direction, confidence_0_to_100, reasons).
    Returns (None, 0, []) if no valid setup found.
    Requires minimum 80% confidence — much stricter than continuation.
    """

    df_5m_ind = add_indicators(df_5m)
    last = df_5m_ind.iloc[-1]
    price = last["close"]
    score = 0
    max_score = 0
    reasons = []

    # Step 1 — determine attempted reversal direction from S/R
    # Long reversal: price at support trying to bounce up
    # Short reversal: price at resistance trying to reject down
    near_support = nearest_level(price, support_1h, "below")
    near_resistance = nearest_level(price, resistance_1h, "above")

    at_support = (
        near_support is not None and
        abs(price - near_support) / price * 100 <= 1.0
    )
    at_resistance = (
        near_resistance is not None and
        abs(near_resistance - price) / price * 100 <= 1.0
    )

    if not at_support and not at_resistance:
        # Not at any significant level — no reversal setup
        return None, 0, []

    # Determine direction based on which level we're at
    if at_support and not at_resistance:
        direction = "long"
    elif at_resistance and not at_support:
        direction = "short"
    else:
        # At both — pick the closer one
        support_dist = abs(price - near_support) / price * 100
        resistance_dist = abs(near_resistance - price) / price * 100
        direction = "long" if support_dist <= resistance_dist else "short"

    # Step 2 — S/R level strength (25 pts)
    max_score += 25
    if direction == "long":
        score += 25
        reasons.append(
            f"Price at validated 1H support ~{near_support:.4f}"
        )
    else:
        score += 25
        reasons.append(
            f"Price at validated 1H resistance ~{near_resistance:.4f}"
        )

    # Step 3 — strong reversal candle required (30 pts)
    max_score += 30
    engulf = detect_engulfing(df_5m_ind)
    pin = detect_pin_bar(df_5m_ind, wick_ratio=2.5)
    pattern_name = None

    if direction == "long":
        if engulf == "bullish":
            score += 30
            pattern_name = "Bullish engulfing"
        elif pin == "bullish":
            score += 25
            pattern_name = "Bullish pin bar"
    else:
        if engulf == "bearish":
            score += 30
            pattern_name = "Bearish engulfing"
        elif pin == "bearish":
            score += 25
            pattern_name = "Bearish pin bar"

    if pattern_name:
        reasons.append(f"{pattern_name} reversal candle confirmed")
    else:
        # No reversal candle — reversal not confirmed
        return None, 0, []

    # Step 4 — RSI divergence (20 pts — strong signal)
    max_score += 20
    divergence = detect_rsi_divergence(df_5m_ind)
    if divergence == "bullish" and direction == "long":
        score += 20
        reasons.append("Bullish RSI divergence — sellers weakening")
    elif divergence == "bearish" and direction == "short":
        score += 20
        reasons.append("Bearish RSI divergence — buyers weakening")

    # Step 5 — RSI extreme (10 pts)
    max_score += 10
    rsi = last.get("rsi")
    if rsi is not None:
        if direction == "long" and rsi <= 35:
            score += 10
            reasons.append(f"RSI deeply oversold ({rsi:.1f})")
        elif direction == "short" and rsi >= 65:
            score += 10
            reasons.append(f"RSI deeply overbought ({rsi:.1f})")
        # Hard contradiction block
        if direction == "long" and rsi >= 75:
            return None, 0, []
        if direction == "short" and rsi <= 25:
            return None, 0, []

    # Step 6 — volume conviction (15 pts)
    max_score += 15
    vol_ratio = last.get("vol_ratio")
    if vol_ratio is not None:
        if vol_ratio >= 1.8:
            score += 15
            reasons.append(
                f"Strong volume spike ({vol_ratio:.1f}x average) "
                f"confirms reversal conviction"
            )
        elif vol_ratio >= 1.3:
            score += 8
            reasons.append(f"Volume above average ({vol_ratio:.1f}x)")

    # Step 7 — trend weakening bonus (0 pts but validates setup)
    # If 1H trend is weakening it supports the reversal thesis
    trend_direction, trend_strength = get_trend_strength(df_1h)
    if trend_strength < 0.5:
        reasons.append(
            f"1H trend weakening (strength {round(trend_strength * 100)}%) "
            f"— reversal conditions favorable"
        )

    if score == 0 or len(reasons) < 3:
        return None, 0, []

    confidence = round(score / max_score * 100, 1)
    return direction, confidence, reasons
