"""
Signal result tracker — Redis backed, survives Railway restarts.
Tracks open signals, sends TP/SL notifications,
records results for daily summary, handles reversals.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from bot.signal_engine import Signal
from bot.notifier import send_result, send_reversal_alert
from bot.summary import record_result
from bot.redis_client import redis_hset, redis_hget, redis_hgetall, redis_hdel, redis_set, redis_get

logger = logging.getLogger(__name__)

MAX_SIGNAL_AGE_HOURS = 24
MIN_REVERSAL_CONFIDENCE = 80.0
TRACKER_KEY = "open_signals"


def _sig_to_dict(sig: Signal) -> dict:
    return {
        "symbol": sig.symbol,
        "exchange": sig.exchange,
        "direction": sig.direction,
        "confidence": sig.confidence,
        "entry": sig.entry,
        "stop_loss": sig.stop_loss,
        "take_profit1": sig.take_profit1,
        "take_profit2": sig.take_profit2,
        "take_profit3": sig.take_profit3,
        "sl_pct": sig.sl_pct,
        "tp1_pct": sig.tp1_pct,
        "tp2_pct": sig.tp2_pct,
        "tp3_pct": sig.tp3_pct,
        "leverage": sig.leverage,
        "reasons": sig.reasons,
        "fired_at": sig.fired_at.isoformat(),
    }


def _dict_to_sig(d: dict) -> Signal:
    return Signal(
        symbol=d["symbol"],
        exchange=d["exchange"],
        direction=d["direction"],
        confidence=float(d["confidence"]),
        entry=float(d["entry"]),
        stop_loss=float(d["stop_loss"]),
        take_profit1=float(d["take_profit1"]),
        take_profit2=float(d["take_profit2"]),
        take_profit3=float(d["take_profit3"]),
        sl_pct=float(d["sl_pct"]),
        tp1_pct=float(d["tp1_pct"]),
        tp2_pct=float(d["tp2_pct"]),
        tp3_pct=float(d["tp3_pct"]),
        leverage=int(d["leverage"]),
        reasons=d["reasons"],
        fired_at=datetime.fromisoformat(d["fired_at"]),
    )


@dataclass
class TrackedSignal:
    signal: Signal
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    opened_at: datetime = field(default_factory=datetime.utcnow)


def _load_tracked(symbol: str) -> TrackedSignal | None:
    raw = redis_hget(TRACKER_KEY, symbol)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        sig = _dict_to_sig(data["signal"])
        return TrackedSignal(
            signal=sig,
            tp1_hit=data.get("tp1_hit", False),
            tp2_hit=data.get("tp2_hit", False),
            tp3_hit=data.get("tp3_hit", False),
            opened_at=datetime.fromisoformat(data.get("opened_at", datetime.utcnow().isoformat())),
        )
    except Exception as e:
        logger.error(f"[tracker] failed to load {symbol}: {e}")
        return None


def _save_tracked(symbol: str, tracked: TrackedSignal):
    data = {
        "signal": _sig_to_dict(tracked.signal),
        "tp1_hit": tracked.tp1_hit,
        "tp2_hit": tracked.tp2_hit,
        "tp3_hit": tracked.tp3_hit,
        "opened_at": tracked.opened_at.isoformat(),
    }
    redis_hset(TRACKER_KEY, symbol, json.dumps(data))


def _delete_tracked(symbol: str):
    redis_hdel(TRACKER_KEY, symbol)


def _get_all_symbols() -> list[str]:
    data = redis_hgetall(TRACKER_KEY)
    return list(data.keys())


class SignalTracker:

    def add(self, sig: Signal):
        tracked = TrackedSignal(signal=sig)
        _save_tracked(sig.symbol, tracked)
        logger.info(f"[tracker] tracking {sig.symbol} {sig.direction}")

    def is_symbol_active(self, symbol: str) -> bool:
        return redis_hget(TRACKER_KEY, symbol) is not None

    def get_active_trade(self, symbol: str) -> TrackedSignal | None:
        return _load_tracked(symbol)

    def get_active_symbols(self) -> set[str]:
        return set(_get_all_symbols())

    def close(self, symbol: str):
        _delete_tracked(symbol)
        logger.info(f"[tracker] closed {symbol}")

    async def handle_reversal(
        self,
        new_sig: Signal,
        bot_token: str,
        chat_id: str,
    ) -> bool:
        tracked = _load_tracked(new_sig.symbol)
        if not tracked:
            return False

        active = tracked.signal
        if active.direction == new_sig.direction:
            return False

        if new_sig.confidence < MIN_REVERSAL_CONFIDENCE:
            return False

        await send_reversal_alert(
            bot_token, chat_id,
            active_signal=active,
            new_signal=new_sig,
        )
        self.close(new_sig.symbol)
        logger.info(
            f"[tracker] reversal: {new_sig.symbol} "
            f"{active.direction} → {new_sig.direction}"
        )
        return True

    async def check_all(self, get_price_fn, bot_token: str, chat_id: str):
        symbols = _get_all_symbols()
        if not symbols:
            logger.debug("[tracker] no open signals to check")
            return

        logger.info(f"[tracker] checking {len(symbols)} open signals")

        for symbol in symbols:
            tracked = _load_tracked(symbol)
            if not tracked:
                continue

            sig = tracked.signal

            age = datetime.utcnow() - tracked.opened_at
            if age > timedelta(hours=MAX_SIGNAL_AGE_HOURS):
                await send_result(
                    bot_token, chat_id,
                    sig.symbol, sig.direction, "expired", 0.0
                )
                record_result(sig.symbol, sig.direction, "expired", 0.0)
                self.close(symbol)
                continue

            price = get_price_fn(sig.symbol, sig.exchange)
            if price is None:
                logger.debug(f"[tracker] no price for {symbol}")
                continue

            risk = abs(sig.entry - sig.stop_loss)
            if risk == 0:
                continue

            logger.info(
                f"[tracker] {symbol} {sig.direction} "
                f"entry={sig.entry} now={price} "
                f"sl={sig.stop_loss} tp1={sig.take_profit1}"
            )

            changed = False

            if sig.direction == "long":
                if price <= sig.stop_loss:
                    pnl_r = round((price - sig.entry) / risk, 2)
                    await send_result(bot_token, chat_id, sig.symbol, sig.direction, "sl", pnl_r)
                    record_result(sig.symbol, sig.direction, "sl", pnl_r)
                    self.close(symbol)
                    logger.info(f"[tracker] SL: {symbol} {pnl_r}R")
                    continue

                if not tracked.tp1_hit and price >= sig.take_profit1:
                    tracked.tp1_hit = True
                    pnl_r = round((sig.take_profit1 - sig.entry) / risk, 2)
                    await send_result(bot_token, chat_id, sig.symbol, sig.direction, "tp1", pnl_r)
                    record_result(sig.symbol, sig.direction, "tp1", pnl_r)
                    logger.info(f"[tracker] TP1: {symbol} +{pnl_r}R")
                    changed = True

                if tracked.tp1_hit and not tracked.tp2_hit and price >= sig.take_profit2:
                    tracked.tp2_hit = True
                    pnl_r = round((sig.take_profit2 - sig.entry) / risk, 2)
                    await send_result(bot_token, chat_id, sig.symbol, sig.direction, "tp2", pnl_r)
                    record_result(sig.symbol, sig.direction, "tp2", pnl_r)
                    logger.info(f"[tracker] TP2: {symbol} +{pnl_r}R")
                    changed = True

                if tracked.tp2_hit and not tracked.tp3_hit and price >= sig.take_profit3:
                    tracked.tp3_hit = True
                    pnl_r = round((sig.take_profit3 - sig.entry) / risk, 2)
                    await send_result(bot_token, chat_id, sig.symbol, sig.direction, "tp3", pnl_r)
                    record_result(sig.symbol, sig.direction, "tp3", pnl_r)
                    self.close(symbol)
                    logger.info(f"[tracker] TP3: {symbol} +{pnl_r}R")
                    continue

            else:
                if price >= sig.stop_loss:
                    pnl_r = round((sig.entry - price) / risk, 2)
                    await send_result(bot_token, chat_id, sig.symbol, sig.direction, "sl", pnl_r)
                    record_result(sig.symbol, sig.direction, "sl", pnl_r)
                    self.close(symbol)
                    logger.info(f"[tracker] SL: {symbol} {pnl_r}R")
                    continue

                if not tracked.tp1_hit and price <= sig.take_profit1:
                    tracked.tp1_hit = True
                    pnl_r = round((sig.entry - sig.take_profit1) / risk, 2)
                    await send_result(bot_token, chat_id, sig.symbol, sig.direction, "tp1", pnl_r)
                    record_result(sig.symbol, sig.direction, "tp1", pnl_r)
                    logger.info(f"[tracker] TP1: {symbol} +{pnl_r}R")
                    changed = True

                if tracked.tp1_hit and not tracked.tp2_hit and price <= sig.take_profit2:
                    tracked.tp2_hit = True
                    pnl_r = round((sig.entry - sig.take_profit2) / risk, 2)
                    await send_result(bot_token, chat_id, sig.symbol, sig.direction, "tp2", pnl_r)
                    record_result(sig.symbol, sig.direction, "tp2", pnl_r)
                    logger.info(f"[tracker] TP2: {symbol} +{pnl_r}R")
                    changed = True

                if tracked.tp2_hit and not tracked.tp3_hit and price <= sig.take_profit3:
                    tracked.tp3_hit = True
                    pnl_r = round((sig.entry - sig.take_profit3) / risk, 2)
                    await send_result(bot_token, chat_id, sig.symbol, sig.direction, "tp3", pnl_r)
                    record_result(sig.symbol, sig.direction, "tp3", pnl_r)
                    self.close(symbol)
                    logger.info(f"[tracker] TP3: {symbol} +{pnl_r}R")
                    continue

            # Save updated state if TP flags changed
            if changed:
                _save_tracked(symbol, tracked)
