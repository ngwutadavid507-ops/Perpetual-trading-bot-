"""
Signal result tracker and active position manager.
Tracks open signals, sends TP/SL notifications,
records results for daily summary, handles reversals.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from bot.signal_engine import Signal
from bot.notifier import send_result, send_reversal_alert
from bot.summary import record_result

logger = logging.getLogger(__name__)

MAX_SIGNAL_AGE_HOURS = 24
MIN_REVERSAL_CONFIDENCE = 80.0


@dataclass
class TrackedSignal:
    signal: Signal
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    opened_at: datetime = field(default_factory=datetime.utcnow)


class SignalTracker:
    def __init__(self):
        self._open: dict[str, TrackedSignal] = {}

    def add(self, sig: Signal):
        key = sig.symbol
        self._open[key] = TrackedSignal(signal=sig)
        logger.info(f"[tracker] tracking {key} {sig.direction}")

    def is_symbol_active(self, symbol: str) -> bool:
        return symbol in self._open

    def get_active_trade(self, symbol: str):
        return self._open.get(symbol)

    def get_active_symbols(self) -> set[str]:
        return set(self._open.keys())

    def close(self, symbol: str):
        self._open.pop(symbol, None)
        logger.info(f"[tracker] closed {symbol}")

    async def handle_reversal(
        self,
        new_sig: Signal,
        bot_token: str,
        chat_id: str,
    ) -> bool:
        tracked = self._open.get(new_sig.symbol)
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
        closed_keys = []

        for symbol, tracked in list(self._open.items()):
            sig = tracked.signal

            age = datetime.utcnow() - tracked.opened_at
            if age > timedelta(hours=MAX_SIGNAL_AGE_HOURS):
                await send_result(
                    bot_token, chat_id,
                    sig.symbol, sig.direction, "expired", 0.0
                )
                record_result(sig.symbol, sig.direction, "expired", 0.0)
                closed_keys.append(symbol)
                continue

            price = get_price_fn(sig.symbol, sig.exchange)
            if price is None:
                continue

            risk = abs(sig.entry - sig.stop_loss)
            if risk == 0:
                continue

            logger.debug(
                f"[tracker] {symbol} {sig.direction} "
                f"entry={sig.entry} now={price} "
                f"sl={sig.stop_loss} tp1={sig.take_profit1}"
            )

            if sig.direction == "long":
                if price <= sig.stop_loss:
                    pnl_r = round((price - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "sl", pnl_r
                    )
                    record_result(sig.symbol, sig.direction, "sl", pnl_r)
                    closed_keys.append(symbol)
                    logger.info(f"[tracker] SL: {symbol} {pnl_r}R")
                    continue

                if not tracked.tp1_hit and price >= sig.take_profit1:
                    tracked.tp1_hit = True
                    pnl_r = round((sig.take_profit1 - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp1", pnl_r
                    )
                    record_result(sig.symbol, sig.direction, "tp1", pnl_r)
                    logger.info(f"[tracker] TP1: {symbol} +{pnl_r}R")

                if tracked.tp1_hit and not tracked.tp2_hit and price >= sig.take_profit2:
                    tracked.tp2_hit = True
                    pnl_r = round((sig.take_profit2 - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp2", pnl_r
                    )
                    record_result(sig.symbol, sig.direction, "tp2", pnl_r)
                    logger.info(f"[tracker] TP2: {symbol} +{pnl_r}R")

                if tracked.tp2_hit and not tracked.tp3_hit and price >= sig.take_profit3:
                    tracked.tp3_hit = True
                    pnl_r = round((sig.take_profit3 - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp3", pnl_r
                    )
                    record_result(sig.symbol, sig.direction, "tp3", pnl_r)
                    closed_keys.append(symbol)
                    logger.info(f"[tracker] TP3: {symbol} +{pnl_r}R")

            else:
                if price >= sig.stop_loss:
                    pnl_r = round((sig.entry - price) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "sl", pnl_r
                    )
                    record_result(sig.symbol, sig.direction, "sl", pnl_r)
                    closed_keys.append(symbol)
                    logger.info(f"[tracker] SL: {symbol} {pnl_r}R")
                    continue

                if not tracked.tp1_hit and price <= sig.take_profit1:
                    tracked.tp1_hit = True
                    pnl_r = round((sig.entry - sig.take_profit1) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp1", pnl_r
                    )
                    record_result(sig.symbol, sig.direction, "tp1", pnl_r)
                    logger.info(f"[tracker] TP1: {symbol} +{pnl_r}R")

                if tracked.tp1_hit and not tracked.tp2_hit and price <= sig.take_profit2:
                    tracked.tp2_hit = True
                    pnl_r = round((sig.entry - sig.take_profit2) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp2", pnl_r
                    )
                    record_result(sig.symbol, sig.direction, "tp2", pnl_r)
                    logger.info(f"[tracker] TP2: {symbol} +{pnl_r}R")

                if tracked.tp2_hit and not tracked.tp3_hit and price <= sig.take_profit3:
                    tracked.tp3_hit = True
                    pnl_r = round((sig.entry - sig.take_profit3) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp3", pnl_r
                    )
                    record_result(sig.symbol, sig.direction, "tp3", pnl_r)
                    closed_keys.append(symbol)
                    logger.info(f"[tracker] TP3: {symbol} +{pnl_r}R")

        for key in closed_keys:
            self._open.pop(key, None)
