"""
Signal lifecycle tracker.
Tracks every signal through its full lifecycle:
DETECTED → VALIDATED → SENT → ACTIVE → CLOSED

Stores all state in Redis so it survives Railway restarts.
Records timestamps at each stage for execution quality reporting.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from bot.redis_client import redis_hset, redis_hget, redis_hgetall, redis_hdel
from config.settings import Config

logger = logging.getLogger(__name__)

LIFECYCLE_KEY = "signal_lifecycle:v2"
MAX_SIGNAL_AGE_HOURS = 24


class SignalState(str, Enum):
    DETECTED = "detected"
    VALIDATED = "validated"
    SENT = "sent"
    ACTIVE = "active"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    TP3_HIT = "tp3_hit"
    SL_HIT = "sl_hit"
    EXPIRED = "expired"
    CLOSED = "closed"


@dataclass
class SignalLifecycle:
    symbol: str
    direction: str
    strategy_type: str
    confidence: float
    entry: float
    live_entry: float
    stop_loss: float
    take_profit1: float
    take_profit2: float
    take_profit3: float
    leverage: int
    sl_pct: float
    tp1_pct: float
    tp2_pct: float
    tp3_pct: float
    exchange: str
    state: SignalState = SignalState.DETECTED
    detected_at: datetime = field(default_factory=datetime.utcnow)
    sent_at: datetime | None = None
    tp1_hit_at: datetime | None = None
    tp2_hit_at: datetime | None = None
    tp3_hit_at: datetime | None = None
    sl_hit_at: datetime | None = None
    closed_at: datetime | None = None
    entry_slippage_pct: float = 0.0
    total_pnl_r: float = 0.0
    grade: str = ""


def _to_dict(lc: SignalLifecycle) -> dict:
    return {
        "symbol": lc.symbol,
        "direction": lc.direction,
        "strategy_type": lc.strategy_type,
        "confidence": lc.confidence,
        "entry": lc.entry,
        "live_entry": lc.live_entry,
        "stop_loss": lc.stop_loss,
        "take_profit1": lc.take_profit1,
        "take_profit2": lc.take_profit2,
        "take_profit3": lc.take_profit3,
        "leverage": lc.leverage,
        "sl_pct": lc.sl_pct,
        "tp1_pct": lc.tp1_pct,
        "tp2_pct": lc.tp2_pct,
        "tp3_pct": lc.tp3_pct,
        "exchange": lc.exchange,
        "state": lc.state.value,
        "detected_at": lc.detected_at.isoformat(),
        "sent_at": lc.sent_at.isoformat() if lc.sent_at else None,
        "tp1_hit_at": lc.tp1_hit_at.isoformat() if lc.tp1_hit_at else None,
        "tp2_hit_at": lc.tp2_hit_at.isoformat() if lc.tp2_hit_at else None,
        "tp3_hit_at": lc.tp3_hit_at.isoformat() if lc.tp3_hit_at else None,
        "sl_hit_at": lc.sl_hit_at.isoformat() if lc.sl_hit_at else None,
        "closed_at": lc.closed_at.isoformat() if lc.closed_at else None,
        "entry_slippage_pct": lc.entry_slippage_pct,
        "total_pnl_r": lc.total_pnl_r,
        "grade": lc.grade,
    }


def _from_dict(d: dict) -> SignalLifecycle:
    def parse_dt(val):
        return datetime.fromisoformat(val) if val else None

    return SignalLifecycle(
        symbol=d["symbol"],
        direction=d["direction"],
        strategy_type=d.get("strategy_type", "unknown"),
        confidence=float(d["confidence"]),
        entry=float(d["entry"]),
        live_entry=float(d.get("live_entry", d["entry"])),
        stop_loss=float(d["stop_loss"]),
        take_profit1=float(d["take_profit1"]),
        take_profit2=float(d["take_profit2"]),
        take_profit3=float(d["take_profit3"]),
        leverage=int(d["leverage"]),
        sl_pct=float(d["sl_pct"]),
        tp1_pct=float(d["tp1_pct"]),
        tp2_pct=float(d["tp2_pct"]),
        tp3_pct=float(d["tp3_pct"]),
        exchange=d["exchange"],
        state=SignalState(d.get("state", "active")),
        detected_at=parse_dt(d.get("detected_at")) or datetime.utcnow(),
        sent_at=parse_dt(d.get("sent_at")),
        tp1_hit_at=parse_dt(d.get("tp1_hit_at")),
        tp2_hit_at=parse_dt(d.get("tp2_hit_at")),
        tp3_hit_at=parse_dt(d.get("tp3_hit_at")),
        sl_hit_at=parse_dt(d.get("sl_hit_at")),
        closed_at=parse_dt(d.get("closed_at")),
        entry_slippage_pct=float(d.get("entry_slippage_pct", 0.0)),
        total_pnl_r=float(d.get("total_pnl_r", 0.0)),
        grade=d.get("grade", ""),
    )


def _grade_signal(lc: SignalLifecycle) -> str:
    """
    Grades a closed signal based on outcome and timing.
    A: TP2 or TP3 hit
    B: TP1 hit
    C: SL hit after >30 minutes (decent entry, bad timing)
    D: SL hit within 30 minutes (bad entry or stale price)
    """
    if lc.state == SignalState.TP3_HIT:
        return "A+"
    if lc.state == SignalState.TP2_HIT:
        return "A"
    if lc.state == SignalState.TP1_HIT:
        return "B"
    if lc.state == SignalState.SL_HIT:
        if lc.sl_hit_at and lc.sent_at:
            duration = lc.sl_hit_at - lc.sent_at
            if duration.total_seconds() < 1800:  # 30 minutes
                return "D"
        return "C"
    return "C"


class LifecycleTracker:

    def save(self, lc: SignalLifecycle):
        key = lc.symbol
        redis_hset(LIFECYCLE_KEY, key, json.dumps(_to_dict(lc)))

    def load(self, symbol: str) -> SignalLifecycle | None:
        raw = redis_hget(LIFECYCLE_KEY, symbol)
        if not raw:
            return None
        try:
            if not raw.strip().startswith("{"):
                redis_hdel(LIFECYCLE_KEY, symbol)
                return None
            return _from_dict(json.loads(raw))
        except Exception as e:
            logger.error(f"[lifecycle] failed to load {symbol}: {e}")
            redis_hdel(LIFECYCLE_KEY, symbol)
            return None

    def delete(self, symbol: str):
        redis_hdel(LIFECYCLE_KEY, symbol)

    def is_active(self, symbol: str) -> bool:
        lc = self.load(symbol)
        if not lc:
            return False
        return lc.state in (
            SignalState.SENT,
            SignalState.ACTIVE,
            SignalState.TP1_HIT,
            SignalState.TP2_HIT,
        )

    def get_active_direction(self, symbol: str) -> str | None:
        lc = self.load(symbol)
        if not lc:
            return None
        if self.is_active(symbol):
            return lc.direction
        return None

    def get_all_active(self) -> list[SignalLifecycle]:
        all_data = redis_hgetall(LIFECYCLE_KEY)
        active = []
        for symbol, raw in all_data.items():
            try:
                if not raw.strip().startswith("{"):
                    redis_hdel(LIFECYCLE_KEY, symbol)
                    continue
                lc = _from_dict(json.loads(raw))
                if self.is_active(symbol):
                    active.append(lc)
            except Exception as e:
                logger.error(f"[lifecycle] parse error for {symbol}: {e}")
                redis_hdel(LIFECYCLE_KEY, symbol)
        return active

    def mark_sent(self, symbol: str, live_price: float):
        lc = self.load(symbol)
        if not lc:
            return
        lc.state = SignalState.ACTIVE
        lc.sent_at = datetime.utcnow()
        lc.live_entry = live_price
        lc.entry_slippage_pct = round(
            abs(live_price - lc.entry) / lc.entry * 100, 3
        )
        self.save(lc)
        logger.info(
            f"[lifecycle] {symbol} SENT — "
            f"slippage {lc.entry_slippage_pct}%"
        )

    def mark_tp1(self, symbol: str, pnl_r: float):
        lc = self.load(symbol)
        if not lc:
            return
        lc.state = SignalState.TP1_HIT
        lc.tp1_hit_at = datetime.utcnow()
        lc.total_pnl_r = round(lc.total_pnl_r + pnl_r, 2)
        self.save(lc)
        logger.info(f"[lifecycle] {symbol} TP1 +{pnl_r}R")

    def mark_tp2(self, symbol: str, pnl_r: float):
        lc = self.load(symbol)
        if not lc:
            return
        lc.state = SignalState.TP2_HIT
        lc.tp2_hit_at = datetime.utcnow()
        lc.total_pnl_r = round(lc.total_pnl_r + pnl_r, 2)
        self.save(lc)
        logger.info(f"[lifecycle] {symbol} TP2 +{pnl_r}R")

    def mark_tp3(self, symbol: str, pnl_r: float):
        lc = self.load(symbol)
        if not lc:
            return
        lc.state = SignalState.TP3_HIT
        lc.tp3_hit_at = datetime.utcnow()
        lc.total_pnl_r = round(lc.total_pnl_r + pnl_r, 2)
        lc.closed_at = datetime.utcnow()
        lc.grade = _grade_signal(lc)
        self.save(lc)
        logger.info(f"[lifecycle] {symbol} TP3 +{pnl_r}R grade={lc.grade}")

    def mark_sl(self, symbol: str, pnl_r: float):
        lc = self.load(symbol)
        if not lc:
            return
        lc.state = SignalState.SL_HIT
        lc.sl_hit_at = datetime.utcnow()
        lc.closed_at = datetime.utcnow()
        lc.total_pnl_r = round(lc.total_pnl_r + pnl_r, 2)
        lc.grade = _grade_signal(lc)
        self.save(lc)
        logger.info(f"[lifecycle] {symbol} SL {pnl_r}R grade={lc.grade}")

    def mark_expired(self, symbol: str):
        lc = self.load(symbol)
        if not lc:
            return
        lc.state = SignalState.EXPIRED
        lc.closed_at = datetime.utcnow()
        lc.grade = "C"
        self.save(lc)

    async def check_all(self, get_price_fn, bot_token: str, chat_id: str):
        """
        Checks all active signals against current prices.
        Sends TP/SL notifications and updates lifecycle state.
        """
        from bot.notifier import send_result
        from bot.summary import record_result

        active = self.get_all_active()
        if not active:
            logger.debug("[lifecycle] no active signals to check")
            return

        logger.info(f"[lifecycle] checking {len(active)} active signals")

        for lc in active:
            # Check expiry
            age = datetime.utcnow() - lc.detected_at
            if age > timedelta(hours=MAX_SIGNAL_AGE_HOURS):
                await send_result(
                    bot_token, chat_id,
                    lc.symbol, lc.direction, "expired", 0.0,
                    lc.leverage
                )
                self.mark_expired(lc.symbol)
                continue

            price = get_price_fn(lc.symbol, lc.exchange)
            if price is None:
                continue

            risk = abs(lc.entry - lc.stop_loss)
            if risk == 0:
                continue

            logger.info(
                f"[lifecycle] {lc.symbol} {lc.direction} "
                f"entry={lc.entry} now={price} "
                f"sl={lc.stop_loss} tp1={lc.take_profit1}"
            )

            if lc.direction == "long":
                if price <= lc.stop_loss:
                    pnl_r = round((price - lc.entry) / risk, 2)
                    roi_pct = round(pnl_r * lc.sl_pct * lc.leverage, 2)
                    await send_result(
                        bot_token, chat_id,
                        lc.symbol, lc.direction, "sl",
                        pnl_r, lc.leverage, roi_pct
                    )
                    record_result(lc.symbol, lc.direction, "sl", pnl_r)
                    self.mark_sl(lc.symbol, pnl_r)
                    continue

                if lc.state == SignalState.ACTIVE and price >= lc.take_profit1:
                    pnl_r = round((lc.take_profit1 - lc.entry) / risk, 2)
                    roi_pct = round(pnl_r * lc.sl_pct * lc.leverage, 2)
                    await send_result(
                        bot_token, chat_id,
                        lc.symbol, lc.direction, "tp1",
                        pnl_r, lc.leverage, roi_pct
                    )
                    record_result(lc.symbol, lc.direction, "tp1", pnl_r)
                    self.mark_tp1(lc.symbol, pnl_r)

                elif lc.state == SignalState.TP1_HIT and price >= lc.take_profit2:
                    pnl_r = round((lc.take_profit2 - lc.entry) / risk, 2)
                    roi_pct = round(pnl_r * lc.sl_pct * lc.leverage, 2)
                    await send_result(
                        bot_token, chat_id,
                        lc.symbol, lc.direction, "tp2",
                        pnl_r, lc.leverage, roi_pct
                    )
                    record_result(lc.symbol, lc.direction, "tp2", pnl_r)
                    self.mark_tp2(lc.symbol, pnl_r)

                elif lc.state == SignalState.TP2_HIT and price >= lc.take_profit3:
                    pnl_r = round((lc.take_profit3 - lc.entry) / risk, 2)
                    roi_pct = round(pnl_r * lc.sl_pct * lc.leverage, 2)
                    await send_result(
                        bot_token, chat_id,
                        lc.symbol, lc.direction, "tp3",
                        pnl_r, lc.leverage, roi_pct
                    )
                    record_result(lc.symbol, lc.direction, "tp3", pnl_r)
                    self.mark_tp3(lc.symbol, pnl_r)

            else:  # short
                if price >= lc.stop_loss:
                    pnl_r = round((lc.entry - price) / risk, 2)
                    roi_pct = round(pnl_r * lc.sl_pct * lc.leverage, 2)
                    await send_result(
                        bot_token, chat_id,
                        lc.symbol, lc.direction, "sl",
                        pnl_r, lc.leverage, roi_pct
                    )
                    record_result(lc.symbol, lc.direction, "sl", pnl_r)
                    self.mark_sl(lc.symbol, pnl_r)
                    continue

                if lc.state == SignalState.ACTIVE and price <= lc.take_profit1:
                    pnl_r = round((lc.entry - lc.take_profit1) / risk, 2)
                    roi_pct = round(pnl_r * lc.sl_pct * lc.leverage, 2)
                    await send_result(
                        bot_token, chat_id,
                        lc.symbol, lc.direction, "tp1",
                        pnl_r, lc.leverage, roi_pct
                    )
                    record_result(lc.symbol, lc.direction, "tp1", pnl_r)
                    self.mark_tp1(lc.symbol, pnl_r)

                elif lc.state == SignalState.TP1_HIT and price <= lc.take_profit2:
                    pnl_r = round((lc.entry - lc.take_profit2) / risk, 2)
                    roi_pct = round(pnl_r * lc.sl_pct * lc.leverage, 2)
                    await send_result(
                        bot_token, chat_id,
                        lc.symbol, lc.direction, "tp2",
                        pnl_r, lc.leverage, roi_pct
                    )
                    record_result(lc.symbol, lc.direction, "tp2", pnl_r)
                    self.mark_tp2(lc.symbol, pnl_r)

                elif lc.state == SignalState.TP2_HIT and price <= lc.take_profit3:
                    pnl_r = round((lc.entry - lc.take_profit3) / risk, 2)
                    roi_pct = round(pnl_r * lc.sl_pct * lc.leverage, 2)
                    await send_result(
                        bot_token, chat_id,
                        lc.symbol, lc.direction, "tp3",
                        pnl_r, lc.leverage, roi_pct
                    )
                    record_result(lc.symbol, lc.direction, "tp3", pnl_r)
                    self.mark_tp3(lc.symbol, pnl_r)
