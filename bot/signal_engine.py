"""
V2 Signal Engine with observation system.

Flow:
1. Run both strategies to detect setup
2. If confidence >= 85% → fire immediately (too strong to wait)
3. If confidence 65-84% → start 10-minute observation period
4. On next scan, re-check if setup still valid after observation
5. If still valid → fire signal
6. If broken → cancel silently

SL always calculated from live entry price.
All percentages calculated from live entry price.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from bot.indicators import add_indicators, find_support_resistance
from bot.strategy_continuation import score_continuation
from bot.strategy_reversal import score_reversal
from bot.news_filter import should_block_signal, get_confidence_boost
from bot.observation import (
    should_fire_immediately,
    start_observation,
    check_observation,
    cancel_observation,
)
from bot.redis_client import redis_get, redis_set
from config.settings import Config

logger = logging.getLogger(__name__)

COOLDOWN_KEY_PREFIX = "cooldown:v2:"
LAST_SIGNAL_KEY = "last_signal_sent:v2"
MIN_SIGNAL_GAP_MINUTES = 45


def _is_duplicate(symbol: str, direction: str) -> bool:
    key = f"{COOLDOWN_KEY_PREFIX}{symbol}_{direction}"
    existing = redis_get(key)
    if existing:
        return True
    redis_set(key, "1", ex=Config.SIGNAL_COOLDOWN_MINUTES * 60)
    return False


def _is_too_soon() -> bool:
    last_sent = redis_get(LAST_SIGNAL_KEY)
    if not last_sent:
        return False
    try:
        last_dt = datetime.fromisoformat(last_sent)
        from datetime import timedelta
        gap = datetime.utcnow() - last_dt
        return gap < timedelta(minutes=MIN_SIGNAL_GAP_MINUTES)
    except Exception:
        return False


def _record_signal_sent():
    redis_set(
        LAST_SIGNAL_KEY,
        datetime.utcnow().isoformat(),
        ex=MIN_SIGNAL_GAP_MINUTES * 60 * 2
    )


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
    strategy_type: str,
    observation_confirmed: bool,
) -> int:
    """
    Dynamic leverage. Observation-confirmed signals get
    slightly higher leverage — they held up under scrutiny.
    Reversals always get lower leverage than continuation.
    """
    obs_bonus = 1 if observation_confirmed else 0

    if strategy_type == "reversal":
        if confidence >= 95:
            return (25 + obs_bonus * 5) if volatility == "low" else 20
        elif confidence >= 90:
            return (20 + obs_bonus * 5) if volatility == "low" else 15
        elif confidence >= 85:
            return (15 + obs_bonus * 5) if volatility == "low" else 10
        elif confidence >= 80:
            return 15 if volatility == "low" else 10
        else:
            return 10 if volatility == "low" else 7
    else:
        # Continuation
        if confidence >= 95:
            return (50 + obs_bonus * 0) if volatility == "low" else 35
        elif confidence >= 90:
            return (35 + obs_bonus * 5) if volatility == "low" else 25
        elif confidence >= 85:
            return (25 + obs_bonus * 5) if volatility == "low" else 20
        elif confidence >= 80:
            return (20 + obs_bonus * 5) if volatility == "low" else 15
        elif confidence >= 75:
            return 20 if volatility == "low" else 15
        else:
            return 15 if volatility == "low" else 10


def _calculate_sl_and_tps(
    direction: str,
    entry: float,
    df_5m: pd.DataFrame,
) -> tuple:
    """
    Calculates SL and TPs from LIVE ENTRY PRICE.
    All levels and percentages use live entry as reference.
    """
    if direction == "long":
        swing_sl = df_5m["low"].tail(5).min() * 0.998
        if swing_sl >= entry:
            atr = df_5m["atr"].iloc[-1] if "atr" in df_5m.columns else None
            atr = atr if (atr and not pd.isna(atr)) else entry * 0.005
            swing_sl = entry - (atr * 1.5)
        stop_loss = round(swing_sl, 8)
    else:
        swing_sl = df_5m["high"].tail(5).max() * 1.002
        if swing_sl <= entry:
            atr = df_5m["atr"].iloc[-1] if "atr" in df_5m.columns else None
            atr = atr if (atr and not pd.isna(atr)) else entry * 0.005
            swing_sl = entry + (atr * 1.5)
        stop_loss = round(swing_sl, 8)

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


def _score_setup(
    symbol: str,
    exchange_id: str,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    support_1h: list,
    resistance_1h: list,
) -> tuple[str | None, float, list[str], str] | None:
    """
    Runs both strategies and returns the best qualifying setup.
    Returns (direction, confidence, reasons, strategy_type) or None.
    """
    best = None
    best_confidence = 0

    # Try continuation
    cont_dir, cont_conf, cont_reasons = score_continuation(
        df_1h, df_5m, support_1h, resistance_1h
    )
    if cont_dir:
        logger.info(
            f"{symbol}: continuation {cont_dir} conf={cont_conf} "
            f"reasons={len(cont_reasons)}"
        )
    if (cont_dir and
            cont_conf >= Config.MIN_CONFIDENCE_CONTINUATION and
            cont_conf > best_confidence):
        best_confidence = cont_conf
        best = (cont_dir, cont_conf, cont_reasons, "continuation")

    # Try reversal
    rev_dir, rev_conf, rev_reasons = score_reversal(
        df_1h, df_5m, support_1h, resistance_1h
    )
    if rev_dir:
        logger.info(
            f"{symbol}: reversal {rev_dir} conf={rev_conf} "
            f"reasons={len(rev_reasons)}"
        )
    if (rev_dir and
            rev_conf >= Config.MIN_CONFIDENCE_REVERSAL and
            rev_conf > best_confidence):
        best_confidence = rev_conf
        best = (rev_dir, rev_conf, rev_reasons, "reversal")

    if not best:
        logger.debug(f"{symbol}: no qualifying setup found")

    return best

def build_signal(
    symbol: str,
    exchange_id: str,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    live_price: float | None = None,
) -> Signal | None:
    """
    Main signal builder with observation system.

    Returns a Signal ready to send, or None if:
    - No setup found
    - Setup under observation (not ready yet)
    - Setup broke during observation
    - News blocks signal
    - Duplicate cooldown active
    """
    if df_1h is None or df_5m is None:
        return None
    if len(df_1h) < 50 or len(df_5m) < 50:
        return None

    # Get 1H S/R levels
    df_1h_ind = add_indicators(df_1h)
    support_1h, resistance_1h = find_support_resistance(
        df_1h_ind, lookback=80, min_touches=2
    )

    # Score the setup
    setup = _score_setup(
        symbol, exchange_id, df_1h, df_5m,
        support_1h, resistance_1h
    )

    if not setup:
        # Only cancel observation if we have no setup at all
        # Don't cancel just because score temporarily dropped
        # The observation system handles expiry internally
        return None

    direction, confidence, reasons, strategy_type = setup

    # Check for duplicate cooldown
    if _is_duplicate(symbol, direction):
        return None

    # Live entry price
    df_5m_ind = add_indicators(df_5m)
    entry = live_price if live_price else df_5m_ind["close"].iloc[-1]

    # Validate slippage
    if live_price:
        candle_price = df_5m_ind["close"].iloc[-1]
        slippage = abs(live_price - candle_price) / candle_price * 100
        if slippage > Config.MAX_ENTRY_SLIPPAGE_PCT:
            logger.info(
                f"{symbol}: discarded — slippage {slippage:.2f}% "
                f"> {Config.MAX_ENTRY_SLIPPAGE_PCT}%"
            )
            return None

    # News sentiment check
    base = symbol.split("/")[0]
    blocked, block_reason = should_block_signal(
        base, direction, Config.NEWS_LOOKBACK_HOURS
    )
    if blocked:
        logger.info(f"{symbol}: blocked by news — {block_reason}")
        cancel_observation(symbol)
        return None

    news_boost = get_confidence_boost(
        base, direction, Config.NEWS_LOOKBACK_HOURS
    )
    if news_boost > 0:
        confidence = min(100.0, confidence + news_boost)
        reasons.append(
            f"News confirms {direction} (+{news_boost}% confidence)"
        )

    # ── Observation system ────────────────────────────────────────────────────

    observation_confirmed = False

    if should_fire_immediately(confidence):
        logger.info(
            f"{symbol}: conf={confidence}% >= 85 — "
            f"firing immediately"
        )
        observation_confirmed = False

    else:
        obs_status, obs_setup = check_observation(
            symbol, direction, confidence
        )

        if obs_status == "new":
            start_observation(
                symbol, direction, strategy_type,
                confidence, exchange_id
            )
            return None

        elif obs_status == "watching":
            return None

        elif obs_status == "fire":
            observation_confirmed = True
            logger.info(
                f"{symbol}: observation confirmed — firing"
            )

        elif obs_status == "cancelled":
            return None

    # ── Build signal levels ───────────────────────────────────────────────────

    result = _calculate_sl_and_tps(direction, entry, df_5m_ind)
    if result[0] is None:
        return None

    (stop_loss, tp1, tp2, tp3,
     sl_pct, tp1_pct, tp2_pct, tp3_pct) = result

    volatility = _calculate_volatility(df_5m_ind)
    leverage = _calculate_leverage(
        confidence, volatility, strategy_type, observation_confirmed
    )

    from bot.news_filter import get_news_sentiment
    news_sentiment, _ = get_news_sentiment(base, Config.NEWS_LOOKBACK_HOURS)

    return Signal(
        symbol=symbol,
        exchange=exchange_id,
        direction=direction,
        strategy_type=strategy_type,
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


def check_signal_spacing() -> bool:
    return not _is_too_soon()


def record_signal_sent_time():
    _record_signal_sent()
