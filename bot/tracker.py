"""
Signal result tracker. Monitors every signal fired by the bot,
checks price every scan cycle, and sends a result notification
when TP1, TP2, or SL is hit. Also flags signals that have been
open too long without hitting either target as expired.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from bot.signal_engine import Signal
from bot.notifier import send_result

logger = logging.getLogger(__name__)

# Maximum time to track a signal before marking expired (hours)
MAX_SIGNAL_AGE_HOURS = 24


@dataclass
class TrackedSignal:
    signal: Signal
    tp1_hit: bool = False
    tp2_hit: bool = False
    sl_hit: bool = False
    expired: bool = False
    opened_at: datetime = field(default_factory=datetime.utcnow)


class SignalTracker:
    def __init__(self):
        self._open: dict[str, TrackedSignal] = {}

    def add(self, sig: Signal):
        key = f"{sig.symbol}_{sig.direction}_{sig.exchange}"
        self._open[key] = TrackedSignal(signal=sig)
        logger.info(f"Tracking signal: {key}")

    async def check_all(self, get_price_fn, bot_token: str, chat_id: str):
        """
        Call this every scan cycle. get_price_fn is a callable that
        takes a symbol and exchange_id and returns the current price or None.
        """
        closed_keys = []

        for key, tracked in self._open.items():
            sig = tracked.signal

            # Expire old signals
            age = datetime.utcnow() - tracked.opened_at
            if age > timedelta(hours=MAX_SIGNAL_AGE_HOURS):
                tracked.expired = True
                await send_result(
                    bot_token, chat_id,
                    sig.symbol, sig.direction,
                    "expired", 0.0
                )
                closed_keys.append(key)
                continue

            # Get current price
            price = get_price_fn(sig.symbol, sig.exchange)
            if price is None:
                continue

            risk = abs(sig.entry - sig.stop_loss)
            if risk == 0:
                continue

            if sig.direction == "long":
                # Check SL
                if price <= sig.stop_loss:
                    pnl_r = round((price - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction,
                        "sl", pnl_r
                    )
                    closed_keys.append(key)
                    continue

                # Check TP1
                if not tracked.tp1_hit and price >= sig.take_profit1:
                    tracked.tp1_hit = True
                    pnl_r = round((sig.take_profit1 - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction,
                        "tp1", pnl_r
                    )

                # Check TP2
                if tracked.tp1_hit and not tracked.tp2_hit and price >= sig.take_profit2:
                    tracked.tp2_hit = True
                    pnl_r = round((sig.take_profit2 - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction,
                        "tp2", pnl_r
                    )
                    closed_keys.append(key)

            else:  # short
                # Check SL
                if price >= sig.stop_loss:
                    pnl_r = round((sig.entry - price) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction,
                        "sl", pnl_r
                    )
                    closed_keys.append(key)
                    continue

                # Check TP1
                if not tracked.tp1_hit and price <= sig.take_profit1:
                    tracked.tp1_hit = True
                    pnl_r = round((sig.entry - sig.take_profit1) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction,
                        "tp1", pnl_r
                    )

                # Check TP2
                if tracked.tp1_hit and not tracked.tp2_hit and price <= sig.take_profit2:
                    tracked.tp2_hit = True
                    pnl_r = round((sig.entry - sig.take_profit2) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction,
                        "tp2", pnl_r
                    )
                    closed_keys.append(key)

        # Remove closed signals
        for key in closed_keys:
            self._open.pop(key, None)
            logger.info(f"Signal closed: {key}")
