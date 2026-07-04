import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from bot.signal_engine import Signal
from bot.notifier import send_result

logger = logging.getLogger(__name__)

MAX_SIGNAL_AGE_HOURS = 24


@dataclass
class TrackedSignal:
    signal: Signal
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
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
        closed_keys = []

        for key, tracked in self._open.items():
            sig = tracked.signal

            # Expire old signals
            age = datetime.utcnow() - tracked.opened_at
            if age > timedelta(hours=MAX_SIGNAL_AGE_HOURS):
                await send_result(
                    bot_token, chat_id,
                    sig.symbol, sig.direction,
                    "expired", 0.0
                )
                closed_keys.append(key)
                continue

            price = get_price_fn(sig.symbol, sig.exchange)
            if price is None:
                continue

            risk = abs(sig.entry - sig.stop_loss)
            if risk == 0:
                continue

            if sig.direction == "long":
                # SL check
                if price <= sig.stop_loss:
                    pnl_r = round((price - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "sl", pnl_r
                    )
                    closed_keys.append(key)
                    continue

                # TP1
                if not tracked.tp1_hit and price >= sig.take_profit1:
                    tracked.tp1_hit = True
                    pnl_r = round((sig.take_profit1 - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp1", pnl_r
                    )

                # TP2
                if tracked.tp1_hit and not tracked.tp2_hit and price >= sig.take_profit2:
                    tracked.tp2_hit = True
                    pnl_r = round((sig.take_profit2 - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp2", pnl_r
                    )

                # TP3
                if tracked.tp2_hit and not tracked.tp3_hit and price >= sig.take_profit3:
                    tracked.tp3_hit = True
                    pnl_r = round((sig.take_profit3 - sig.entry) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp3", pnl_r
                    )
                    closed_keys.append(key)

            else:  # short
                # SL check
                if price >= sig.stop_loss:
                    pnl_r = round((sig.entry - price) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "sl", pnl_r
                    )
                    closed_keys.append(key)
                    continue

                # TP1
                if not tracked.tp1_hit and price <= sig.take_profit1:
                    tracked.tp1_hit = True
                    pnl_r = round((sig.entry - sig.take_profit1) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp1", pnl_r
                    )

                # TP2
                if tracked.tp1_hit and not tracked.tp2_hit and price <= sig.take_profit2:
                    tracked.tp2_hit = True
                    pnl_r = round((sig.entry - sig.take_profit2) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp2", pnl_r
                    )

                # TP3
                if tracked.tp2_hit and not tracked.tp3_hit and price <= sig.take_profit3:
                    tracked.tp3_hit = True
                    pnl_r = round((sig.entry - sig.take_profit3) / risk, 2)
                    await send_result(
                        bot_token, chat_id,
                        sig.symbol, sig.direction, "tp3", pnl_r
                    )
                    closed_keys.append(key)

        for key in closed_keys:
            self._open.pop(key, None)
            logger.info(f"Signal closed: {key}")
