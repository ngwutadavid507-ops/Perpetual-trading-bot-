"""
V2 Signal Engine.
Combines both strategies:
- Trend Continuation (min 65% confidence)
- Reversal (min 80% confidence)

Each signal includes:
- Strategy type (continuation/reversal)
- Live price validation before firing
- News sentiment check
- TP1=3x, TP2=5x, TP3=7x risk
- Dynamic leverage based on confidence and volatility
- Redis-backed duplicate prevention
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from bot.indicators import add_indicators, find_support_resistance
from bot.strategy_continuation import score_continuation
from bot.strategy_reversal import score_reversal
from bot.news_filter import should_block_signal, get_confidence_boost
from bot.redis_client import redis_get, redis_set
from config.settings import Config

logger = logging.getLogger(__name__)

COOLDOWN_KEY_PREFIX = "cooldown:v2:"


def _is_duplicate(symbol: str, direction: str) -> bool:
    key = f"{COOLDOWN_KEY_PREFIX}{symbol}_{direction}"
    existing = redis_get(key)
    if existing:
        return True
    redis_set(key, "1", ex=Config.SIGNAL_COOLDOWN_MINUTES * 60)
    return False


@dataclass
class Signal:
    symbol: str
    exchange: str
    direction: str
    strategy_type: str          # 'continuation' or 'reversal'
    confidence: float
    entry: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    take_profit3: float
    sl_pct: float
    tp1_pct: float
    tp2_pct: float
    tp3_pct: float
    leverage: int
    reasons: list[str]
    news_sentiment: str         # 'bullish', 'bearish', 'neutral'
    fired_at: datetime = field(default_factory=datetime.utcnow)


def _calculate_volatility(df: pd.DataFrame) -> str:
    atr = df["atr"].iloc[-1] if "atr" in df.columns else None
    price = df["close"].iloc[-1]
    if not atr or price == 0 or pd.isna(atr):
        return "high"
    return "high" if (atr / price) * 100 >= 1.5 else "low"


def _calculate_leverage(
    confidence: float,
    volatility: str,
    strategy_type: str,
) -> int:
    """
    Dynamic leverage based on confidence, volatility and strategy type.
    Reversals get slightly lower leverage due to lower win rate.
    """
    if strategy_type == "reversal":
        if confidence >= 90:
            return 35 if volatility == "low" else 25
        elif confidence >= 85:
            return 25 if volatility == "low" else 20
        elif confidence >= 80:
            return 20 if volatility == "low" else 15
        else:
            return 15 if volatility == "low" else 10
    else:
        # Continuation
        if confidence >= 85:
            return 50 if volatility == "low" else 35
        elif confidence >= 80:
            return 35 if volatility == "low" else 25
        elif confidence >= 75:
            return 25 if volatility == "low" else 20
        elif confidence >= 70:
            return 20 if volatility == "low" else 15
        else:
            return 15 if volatility == "low" else 10


def _calculate_sl(df_5m: pd.DataFrame, direction: str) -> float:
    """Tight SL beyond last 5 candle swing point."""
    if direction == "long":
        return round(df_5m["low"].tail(5).min() * 0.998, 6)
    else:
        return round(df_5m["high"].tail(5).max() * 1.002, 6)


def build_signal(
    symbol: str,
    exchange_id: str,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    live_price: float | None = None,
) -> Signal | None:
    """
    Main signal builder. Tries both strategies and returns
    the highest confidence qualifying signal, or None.
    """
    if df_1h is None or df_5m is None:
        return None
    if len(df_1h) < 50 or len(df_5m) < 50:
        return None

    # Get 1H S/R levels — shared by both strategies
    df_1h_ind = add_indicators(df_1h)
    support_1h, resistance_1h = find_support_resistance(
        df_1h_ind, lookback=80, min_touches=2
    )

    df_5m_ind = add_indicators(df_5m)
    current_price = live_price if live_price else df_5m_ind["close"].iloc[-1]

    best_signal = None
    best_confidence = 0

    # ── Try Continuation Strategy ─────────────────────────────────────────────
    cont_direction, cont_confidence, cont_reasons = score_continuation(
        df_1h, df_5m, support_1h, resistance_1h
    )

    if (cont_direction and
            cont_confidence >= Config.MIN_CONFIDENCE_CONTINUATION and
            cont_confidence > best_confidence):

        if not _is_duplicate(symbol, cont_direction):

            # Live price validation
            if live_price:
                candle_price = df_5m_ind["close"].iloc[-1]
                slippage = abs(live_price - candle_price) / candle_price * 100
                if slippage > Config.MAX_ENTRY_SLIPPAGE_PCT:
                    logger.info(
                        f"{symbol}: continuation signal discarded — "
                        f"slippage {slippage:.2f}% > "
                        f"{Config.MAX_ENTRY_SLIPPAGE_PCT}%"
                    )
                else:
                    # News sentiment check
                    base = symbol.split("/")[0]
                    blocked, block_reason = should_block_signal(
                        base, cont_direction, Config.NEWS_LOOKBACK_HOURS
                    )
                    if blocked:
                        logger.info(
                            f"{symbol}: continuation blocked by news — "
                            f"{block_reason}"
                        )
                    else:
                        news_boost = get_confidence_boost(
                            base, cont_direction, Config.NEWS_LOOKBACK_HOURS
                        )
                        cont_confidence = min(100.0, cont_confidence + news_boost)
                        if news_boost > 0:
                            cont_reasons.append(
                                f"News sentiment confirms {cont_direction} "
                                f"(+{news_boost}% confidence)"
                            )

                        best_confidence = cont_confidence
                        best_signal = (
                            cont_direction, cont_confidence,
                            cont_reasons, "continuation"
                        )
            else:
                base = symbol.split("/")[0]
                blocked, block_reason = should_block_signal(
                    base, cont_direction, Config.NEWS_LOOKBACK_HOURS
                )
                if not blocked:
                    news_boost = get_confidence_boost(
                        base, cont_direction, Config.NEWS_LOOKBACK_HOURS
                    )
                    cont_confidence = min(100.0, cont_confidence + news_boost)
                    best_confidence = cont_confidence
                    best_signal = (
                        cont_direction, cont_confidence,
                        cont_reasons, "continuation"
                    )

    # ── Try Reversal Strategy ─────────────────────────────────────────────────
    rev_direction, rev_confidence, rev_reasons = score_reversal(
        df_1h, df_5m, support_1h, resistance_1h
    )

    if (rev_direction and
            rev_confidence >= Config.MIN_CONFIDENCE_REVERSAL and
            rev_confidence > best_confidence):

        if not _is_duplicate(symbol, rev_direction):

            if live_price:
                candle_price = df_5m_ind["close"].iloc[-1]
                slippage = abs(live_price - candle_price) / candle_price * 100
                if slippage <= Config.MAX_ENTRY_SLIPPAGE_PCT:
                    base = symbol.split("/")[0]
                    blocked, block_reason = should_block_signal(
                        base, rev_direction, Config.NEWS_LOOKBACK_HOURS
                    )
                    if not blocked:
                        news_boost = get_confidence_boost(
                            base, rev_direction, Config.NEWS_LOOKBACK_HOURS
                        )
                        rev_confidence = min(100.0, rev_confidence + news_boost)
                        if news_boost > 0:
                            rev_reasons.append(
                                f"News sentiment confirms {rev_direction} "
                                f"(+{news_boost}% confidence)"
                            )
                        best_confidence = rev_confidence
                        best_signal = (
                            rev_direction, rev_confidence,
                            rev_reasons, "reversal"
                        )
            else:
                base = symbol.split("/")[0]
                blocked, _ = should_block_signal(
                    base, rev_direction, Config.NEWS_LOOKBACK_HOURS
                )
                if not blocked:
                    news_boost = get_confidence_boost(
                        base, rev_direction, Config.NEWS_LOOKBACK_HOURS
                    )
                    rev_confidence = min(100.0, rev_confidence + news_boost)
                    best_confidence = rev_confidence
                    best_signal = (
                        rev_direction, rev_confidence,
                        rev_reasons, "reversal"
                    )

    if not best_signal:
        return None

    direction, confidence, reasons, strategy_type = best_signal

    # Build entry, SL, TPs
    entry = current_price
    stop_loss = _calculate_sl(df_5m_ind, direction)
    risk = abs(entry - stop_loss)

    if risk == 0:
        return None

    if direction == "long":
        take_profit1 = round(entry + (risk * 3), 6)
        take_profit2 = round(entry + (risk * 5), 6)
        take_profit3 = round(entry + (risk * 7), 6)
    else:
        take_profit1 = round(entry - (risk * 3), 6)
        take_profit2 = round(entry - (risk * 5), 6)
        take_profit3 = round(entry - (risk * 7), 6)

    sl_pct = round(abs(entry - stop_loss) / entry * 100, 3)
    tp1_pct = round(abs(take_profit1 - entry) / entry * 100, 3)
    tp2_pct = round(abs(take_profit2 - entry) / entry * 100, 3)
    tp3_pct = round(abs(take_profit3 - entry) / entry * 100, 3)

    volatility = _calculate_volatility(df_5m_ind)
    leverage = _calculate_leverage(confidence, volatility, strategy_type)

    # Get news sentiment for signal display
    base = symbol.split("/")[0]
    from bot.news_filter import get_news_sentiment
    news_sentiment, _ = get_news_sentiment(base, Config.NEWS_LOOKBACK_HOURS)

    return Signal(
        symbol=symbol,
        exchange=exchange_id,
        direction=direction,
        strategy_type=strategy_type,
        confidence=confidence,
        entry=round(entry, 6),
        stop_loss=round(stop_loss, 6),
        take_profit1=take_profit1,
        take_profit2=take_profit2,
        take_profit3=take_profit3,
        sl_pct=sl_pct,
        tp1_pct=tp1_pct,
        tp2_pct=tp2_pct,
        tp3_pct=tp3_pct,
        leverage=leverage,
        reasons=reasons,
        news_sentiment=news_sentiment,
    )
