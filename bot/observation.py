"""
Observation system — watches potential setups for 2 candle periods
before firing a signal. Prevents entries on fake breakouts.

Rules:
- Confidence < 85%: observe for 2 x 5m candles (10 minutes)
- Confidence >= 85%: fire immediately — too strong to wait
- If setup breaks during observation: cancel silently
- All three confirmations must still pass after observation period
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from bot.redis_client import redis_get, redis_set, redis_delete

logger = logging.getLogger(__name__)

OBSERVATION_KEY = "observation:v2"
OBSERVATION_MINUTES = 10  # 2 x 5m candles
IMMEDIATE_FIRE_CONFIDENCE = 85.0


@dataclass
class ObservedSetup:
    symbol: str
    direction: str
    strategy_type: str
    confidence: float
    exchange_id: str
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
            detected_at=datetime.fromisoformat(d["detected_at"]),
            checks_passed=int(d.get("checks_passed", 0)),
        )
    except Exception as e:
        logger.error(f"[observation] load failed for {symbol}: {e}")
        return None


def _delete(symbol: str):
    redis_delete(_key(symbol))


def should_fire_immediately(confidence: float) -> bool:
    """High confidence signals skip observation — fire right away."""
    return confidence >= IMMEDIATE_FIRE_CONFIDENCE


def start_observation(
    symbol: str,
    direction: str,
    strategy_type: str,
    confidence: float,
    exchange_id: str,
) -> bool:
    """
    Starts watching a setup. Returns True if observation started,
    False if already being observed in same direction.
    """
    existing = _load(symbol)
    if existing:
        if existing.direction == direction:
            logger.debug(
                f"[observation] {symbol} already under observation "
                f"({existing.direction})"
            )
            return False
        else:
            # Different direction — cancel old, start new
            _delete(symbol)

    setup = ObservedSetup(
        symbol=symbol,
        direction=direction,
        strategy_type=strategy_type,
        confidence=confidence,
        exchange_id=exchange_id,
    )
    _save(setup)
    logger.info(
        f"[observation] started watching {symbol} {direction} "
        f"conf={confidence} — fire in {OBSERVATION_MINUTES} min if holds"
    )
    return True


def check_observation(
    symbol: str,
    direction: str,
    confidence: float,
) -> tuple[str, ObservedSetup | None]:
    """
    Checks the observation status for a symbol.

    Returns:
        ('fire', setup) — observation complete, signal can fire
        ('watching', setup) — still in observation period
        ('cancelled', None) — setup broke down, direction changed
        ('new', None) — no observation exists yet
    """
    setup = _load(symbol)

    if not setup:
        return "new", None

    # Direction changed — cancel observation
    if setup.direction != direction:
        logger.info(
            f"[observation] {symbol} direction changed "
            f"{setup.direction} → {direction} — cancelled"
        )
        _delete(symbol)
        return "cancelled", None

    # Check if observation period is complete
    age = datetime.utcnow() - setup.detected_at
    if age >= timedelta(minutes=OBSERVATION_MINUTES):
        # Update confidence with latest score
        setup.confidence = confidence
        setup.checks_passed += 1
        _save(setup)
        logger.info(
            f"[observation] {symbol} observation complete — "
            f"setup held for {OBSERVATION_MINUTES} min — ready to fire"
        )
        _delete(symbol)
        return "fire", setup

    # Still watching
    setup.checks_passed += 1
    setup.confidence = confidence  # Update with latest
    _save(setup)
    remaining = OBSERVATION_MINUTES - int(age.total_seconds() / 60)
    logger.info(
        f"[observation] {symbol} still watching — "
        f"{remaining} min remaining | checks={setup.checks_passed}"
    )
    return "watching", setup


def cancel_observation(symbol: str):
    """Cancel observation for a symbol — setup broke down."""
    _delete(symbol)
    logger.info(f"[observation] {symbol} cancelled")
