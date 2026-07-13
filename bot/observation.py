"""
Observation system — watches potential setups for 2 candle periods
before firing a signal. Prevents entries on fake breakouts.

Rules:
- Confidence < 85%: observe for 10 minutes (2 x 5m candles)
- Confidence >= 85%: fire immediately
- Cancel only if price moves >1% against setup direction during observation
- Time expiry fires the signal if setup direction still holds
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from bot.redis_client import redis_get, redis_set, redis_delete

logger = logging.getLogger(__name__)

OBSERVATION_KEY = "observation:v2"
OBSERVATION_MINUTES = 10
IMMEDIATE_FIRE_CONFIDENCE = 85.0
MAX_ADVERSE_MOVE_PCT = 1.0


@dataclass
class ObservedSetup:
    symbol: str
    direction: str
    strategy_type: str
    confidence: float
    exchange_id: str
    entry_price: float
    detected_at: datetime = field(default_factory=datetime.utcnow)
    checks_passed: int = 0


def _key(symbol: str) -> str:
    return f"{OBSERVATION_KEY}:{symbol}"


def _save(setup: ObservedSetup):
    data = {
        "symbol": setup.symbol,
        "direction": setup.direction,
        "strategy_type": setup.strategy_type,
        "confidence": setup.confidence,
        "exchange_id": setup.exchange_id,
        "entry_price": setup.entry_price,
        "detected_at": setup.detected_at.isoformat(),
        "checks_passed": setup.checks_passed,
    }
    redis_set(_key(setup.symbol), json.dumps(data), ex=3600)


def _load(symbol: str) -> ObservedSetup | None:
    raw = redis_get(_key(symbol))
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return ObservedSetup(
            symbol=d["symbol"],
            direction=d["direction"],
            strategy_type=d["strategy_type"],
            confidence=float(d["confidence"]),
            exchange_id=d["exchange_id"],
            entry_price=float(d.get("entry_price", 0)),
            detected_at=datetime.fromisoformat(d["detected_at"]),
            checks_passed=int(d.get("checks_passed", 0)),
        )
    except Exception as e:
        logger.error(f"[observation] load failed for {symbol}: {e}")
        return None


def _delete(symbol: str):
    redis_delete(_key(symbol))


def should_fire_immediately(confidence: float) -> bool:
    return confidence >= IMMEDIATE_FIRE_CONFIDENCE


def start_observation(
    symbol: str,
    direction: str,
    strategy_type: str,
    confidence: float,
    exchange_id: str,
    entry_price: float,
) -> bool:
    existing = _load(symbol)
    if existing:
        if existing.direction == direction:
            logger.debug(
                f"[observation] {symbol} already watching {direction}"
            )
            return False
        else:
            _delete(symbol)

    setup = ObservedSetup(
        symbol=symbol,
        direction=direction,
        strategy_type=strategy_type,
        confidence=confidence,
        exchange_id=exchange_id,
        entry_price=entry_price,
    )
    _save(setup)
    logger.info(
        f"[observation] watching {symbol} {direction} "
        f"conf={confidence} entry={entry_price} — "
        f"fires in {OBSERVATION_MINUTES}min if price holds"
    )
    return True


def check_observation(
    symbol: str,
    current_price: float,
) -> tuple[str, ObservedSetup | None]:
    """
    Checks observation status based on TIME and PRICE — not re-scoring.

    Returns:
        ('fire', setup) — ready to fire
        ('watching', setup) — still observing
        ('cancelled', None) — price moved too far against setup
        ('none', None) — no observation active
    """
    setup = _load(symbol)
    if not setup:
        return "none", None

    # Price-based cancellation only
    if current_price and setup.entry_price > 0:
        move_pct = (current_price - setup.entry_price) / setup.entry_price * 100
        if setup.direction == "long" and move_pct < -MAX_ADVERSE_MOVE_PCT:
            logger.info(
                f"[observation] {symbol} LONG cancelled — "
                f"price dropped {abs(move_pct):.2f}% during observation"
            )
            _delete(symbol)
            return "cancelled", None
        elif setup.direction == "short" and move_pct > MAX_ADVERSE_MOVE_PCT:
            logger.info(
                f"[observation] {symbol} SHORT cancelled — "
                f"price rose {move_pct:.2f}% during observation"
            )
            _delete(symbol)
            return "cancelled", None

    # Time-based firing
    age = datetime.utcnow() - setup.detected_at
    if age >= timedelta(minutes=OBSERVATION_MINUTES):
        setup.checks_passed += 1
        logger.info(
            f"[observation] {symbol} {setup.direction} "
            f"held for {OBSERVATION_MINUTES}min — firing"
        )
        _delete(symbol)
        return "fire", setup

    remaining = OBSERVATION_MINUTES - int(age.total_seconds() / 60)
    setup.checks_passed += 1
    _save(setup)
    logger.info(
        f"[observation] {symbol} {setup.direction} "
        f"still watching — {remaining}min remaining"
    )
    return "watching", setup


def cancel_observation(symbol: str):
    _delete(symbol)
