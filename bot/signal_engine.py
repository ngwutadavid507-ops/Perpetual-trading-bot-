"""
V2 Signal Engine — Triple Timeframe Only.

Only one strategy now: Triple Timeframe Trend.
4H + 1H + 5m must all agree.
Produces 1-5 signals per day of genuinely high quality.

Observation system:
- Confidence >= 85%: fire immediately
- Confidence 60-84%: observe for 10 minutes (price-based)
- Cancel only if price moves >1% against setup during observation

SL: 2x ATR from entry (adapts to volatility)
TP: 3x, 5x, 7x risk
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from bot.indicators import add_indicators, find_support_resistance
from bot.strategy_continuation import score_continuation
from bot.news_filter import should_block_signal, get_confidence_boost
from bot.observation import (
    should_fire_immediately,
    start_observation,
    check_observation,
)
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
    strategy_type: str
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
    news_sentiment: str
    observation_confirmed: bool = False
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
    observation_confirmed: bool,
) -> int:
    """
    Leverage based on confidence and volatility.
    Observation-confirmed signals get slightly higher leverage.
    """
    obs_bonus = 5 if observation_confirmed else 0

    if confidence >= 95:
        base = 50 if volatility == "low" else 35
    elif confidence >= 90:
        base = 35 if volatility == "low" else 25
    elif confidence >= 85:
        base = 25 if volatility == "low" else 20
    elif confidence >= 80:
        base = 20 if volatility == "low" else 15
    elif confidence >= 75:
        base = 15 if volatility == "low" else 10
    else:
        base = 10 if volatility == "low" else 7

    return min(50, base + obs_bonus)


def _calculate_sl_and_tps(
    direction: str,
    entry: float,
    df_5m: pd.DataFrame,
) -> tuple:
    """
    ATR-based SL — 2x ATR from entry or beyond swing point,
    whichever gives more room. Protects against normal volatility.
    All percentages from live entry price.
    """
    atr = df_5m["atr"].iloc[-1] if "atr" in df_5m.columns else None
    if atr is None or pd.isna(atr) or atr == 0:
        atr = entry * 0.008

    if direction == "long":
        atr_sl = entry - (atr * 2)
        swing_sl = df_5m["low"].tail(10).min() * 0.998
        stop_loss = round(min(atr_sl, swing_sl), 8)
        if stop_loss >= entry:
            stop_loss = round(entry - (atr * 2), 8)
    else:
        atr_sl = entry + (atr * 2)
        swing_sl = df_5m["high"].tail(10).max() * 1.002
        stop_loss = round(max(atr_sl, swing_sl), 8)
        if stop_loss <= entry:
            stop_loss = round(entry + (atr * 2), 8)

    risk = abs(entry - stop_loss)
    if risk == 0:
        return None, None, None, None, None, None, None, None

    if direction == "long":
        tp1 = round(entry + (risk * 3), 8)
        tp2 = round(entry + (risk * 5), 8)
        tp3 = round(entry + (risk * 7), 8)
    else:
        tp1 = round(entry - (risk * 3), 8)
        tp2 = round(entry - (risk * 5), 8)
        tp3 = round(entry - (risk * 7), 8)

    sl_pct = round(abs(entry - stop_loss) / entry * 100, 3)
    tp1_pct = round(abs(tp1 - entry) / entry * 100, 3)
    tp2_pct = round(abs(tp2 - entry) / entry * 100, 3)
    tp3_pct = round(abs(tp3 - entry) / entry * 100, 3)

    return stop_loss, tp1, tp2, tp3, sl_pct, tp1_pct, tp2_pct, tp3_pct


def build_signal(
    symbol: str,
    exchange_id: str,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    live_price: float | None = None,
) -> Signal | None:

    if df_1h is None or df_5m is None:
        return None
    if len(df_1h) < 60 or len(df_5m) < 50:
        return None

    # Get 1H S/R levels
    df_1h_ind = add_indicators(df_1h)
    support_1h, resistance_1h = find_support_resistance(
        df_1h_ind, lookback=80, min_touches=2
    )

    df_5m_ind = add_indicators(df_5m)
    entry = live_price if live_price else df_5m_ind["close"].iloc[-1]

    # Slippage check
    if live_price:
        candle_price = df_5m_ind["close"].iloc[-1]
        slippage = abs(live_price - candle_price) / candle_price * 100
        if slippage > Config.MAX_ENTRY_SLIPPAGE_PCT:
            logger.debug(
                f"{symbol}: slippage {slippage:.2f}% — discarded"
            )
            return None

    # Triple timeframe scoring — only strategy
    direction, confidence, reasons = score_continuation(
        df_1h, df_5m, support_1h, resistance_1h
    )

    if not direction or confidence == 0:
        return None

    if confidence < Config.MIN_CONFIDENCE_CONTINUATION:
        return None

    # Duplicate check
    if _is_duplicate(symbol, direction):
        return None

    # News check
    base = symbol.split("/")[0]
    blocked, block_reason = should_block_signal(
        base, direction, Config.NEWS_LOOKBACK_HOURS
    )
    if blocked:
        logger.info(f"{symbol}: news block — {block_reason}")
        return None

    news_boost = get_confidence_boost(
        base, direction, Config.NEWS_LOOKBACK_HOURS
    )
    if news_boost > 0:
        confidence = min(100.0, confidence + news_boost)
        reasons.append(f"News confirms {direction} (+{news_boost}%)")

    # ── Observation system ────────────────────────────────────────────────────

    observation_confirmed = False

    if should_fire_immediately(confidence):
        logger.info(
            f"{symbol}: conf={confidence}% — firing immediately"
        )
    else:
        obs_status, obs_setup = check_observation(symbol, entry)

        if obs_status == "none":
            start_observation(
                symbol, direction, "continuation",
                confidence, exchange_id, entry
            )
            return None

        elif obs_status == "watching":
            return None

        elif obs_status == "fire":
            observation_confirmed = True
            logger.info(
                f"{symbol}: 10min observation passed — firing"
            )

        elif obs_status == "cancelled":
            return None

    # ── Build levels from live price ──────────────────────────────────────────

    result = _calculate_sl_and_tps(direction, entry, df_5m_ind)
    if result[0] is None:
        return None

    (stop_loss, tp1, tp2, tp3,
     sl_pct, tp1_pct, tp2_pct, tp3_pct) = result

    volatility = _calculate_volatility(df_5m_ind)
    leverage = _calculate_leverage(
        confidence, volatility, observation_confirmed
    )

    from bot.news_filter import get_news_sentiment
    news_sentiment, _ = get_news_sentiment(base, Config.NEWS_LOOKBACK_HOURS)

    return Signal(
        symbol=symbol,
        exchange=exchange_id,
        direction=direction,
        strategy_type="continuation",
        confidence=confidence,
        entry=round(entry, 8),
        stop_loss=stop_loss,
        take_profit1=tp1,
        take_profit2=tp2,
        take_profit3=tp3,
        sl_pct=sl_pct,
        tp1_pct=tp1_pct,
        tp2_pct=tp2_pct,
        tp3_pct=tp3_pct,
        leverage=leverage,
        reasons=reasons,
        news_sentiment=news_sentiment,
        observation_confirmed=observation_confirmed,
    )
