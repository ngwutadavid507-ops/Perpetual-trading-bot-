"""
Phoenix Signal Bot V2 — Main Entry Point

Key improvements in this version:
- Signal spacing: minimum 45 minutes between consecutive signals
- SL calculated from live entry price (fixes -2.74R anomaly)
- Signals sorted by confidence — best signal fires first
- After each signal bot waits 45 min before next regardless of scan
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from config.settings import Config
from bot.exchanges import (
    build_exchange,
    get_liquid_perp_symbols,
    fetch_dual_timeframe,
    fetch_live_price,
)
from bot.signal_engine import (
    build_signal,
    Signal,
    check_signal_spacing,
    record_signal_sent_time,
)
from bot.lifecycle import LifecycleTracker, SignalLifecycle, SignalState
from bot.notifier import send_signal, send_reversal_alert
from bot.summary import record_signal_sent, record_result, format_daily_summary
from bot.redis_client import redis_get, redis_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scanner")

lifecycle = LifecycleTracker()
_exchange_cache: dict = {}


def get_current_price(symbol: str, exchange: str) -> float | None:
    try:
        ex = _exchange_cache.get(exchange)
        if not ex:
            for ex_obj in _exchange_cache.values():
                try:
                    price = fetch_live_price(ex_obj, symbol)
                    if price:
                        return price
                except Exception:
                    continue
        else:
            return fetch_live_price(ex, symbol)
    except Exception as e:
        logger.debug(f"Price check failed for {symbol}: {e}")
    return None


def get_base_symbol(symbol: str) -> str:
    return symbol.split("/")[0]


def get_daily_count() -> int:
    key = f"daily_count:v2:{datetime.now(timezone.utc).date()}"
    val = redis_get(key)
    return int(val) if val else 0


def increment_daily_count():
    from bot.redis_client import redis_incr, redis_expire
    key = f"daily_count:v2:{datetime.now(timezone.utc).date()}"
    count = redis_incr(key)
    if count == 1:
        redis_expire(key, 48 * 3600)


def remaining_today() -> int:
    return max(0, Config.MAX_SIGNALS_PER_DAY - get_daily_count())


async def send_daily_summary():
    from telegram import Bot
    from telegram.constants import ParseMode
    bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
    try:
        text = format_daily_summary()
        await bot.send_message(
            chat_id=Config.TELEGRAM_CHAT_ID,
            text=text,
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info("Daily summary sent")
    except Exception as e:
        logger.error(f"Daily summary failed: {e}")


async def check_and_send_summary():
    now = datetime.now(timezone.utc)
    today = str(now.date())
    if now.hour == 0 and now.minute < 10:
        sent_key = f"summary_sent:v2:{today}"
        if not redis_get(sent_key):
            redis_set(sent_key, "1", ex=86400)
            await send_daily_summary()


async def scan_all_exchanges():
    daily_remaining = remaining_today()
    if daily_remaining <= 0:
        logger.info(
            f"Daily limit reached "
            f"({Config.MAX_SIGNALS_PER_DAY}) — skipping scan"
        )
        return

    # Check signal spacing — don't send if too soon after last signal
    if not check_signal_spacing():
        logger.info(
            "Signal spacing enforced — "
            "waiting 45 min since last signal"
        )
        return

    # {base_symbol: {exchange_id: (Signal, df_5m, live_price)}}
    exchange_signals: dict[str, dict[str, tuple]] = defaultdict(dict)

    for exchange_id in Config.EXCHANGES:
        exchange = build_exchange(exchange_id)
        _exchange_cache[exchange_id] = exchange

        symbols = get_liquid_perp_symbols(
            exchange, Config.MIN_24H_VOLUME_USDT
        )
        logger.info(f"[{exchange_id}] scanning {len(symbols)} liquid symbols")

        skipped = 0
        for symbol in symbols:
            try:
                df_1h, df_5m = fetch_dual_timeframe(exchange, symbol)
                if df_1h is None or df_5m is None:
                    skipped += 1
                    continue

                live_price = fetch_live_price(exchange, symbol)

                sig = build_signal(
                    symbol,
                    exchange_id,
                    df_1h,
                    df_5m,
                    live_price=live_price,
                )
                if sig is None:
                    continue

                base = get_base_symbol(symbol)
                exchange_signals[base][exchange_id] = (sig, df_5m, live_price)
                logger.info(
                    f"[{exchange_id}] candidate: {base} "
                    f"{sig.direction} {sig.strategy_type} "
                    f"conf={sig.confidence}"
                )

            except Exception as e:
                logger.error(f"[{exchange_id}] error on {symbol}: {e}")
                continue

        logger.info(f"[{exchange_id}] scan complete — {skipped} skipped")

    num_exchanges = len(Config.EXCHANGES)
    confirmed_signals: list[tuple] = []
    reversal_candidates: list[tuple] = []

    for base, ex_sigs in exchange_signals.items():

        directions = [sig.direction for sig, _, _ in ex_sigs.values()]
        if len(set(directions)) > 1:
            logger.debug(f"{base}: exchanges disagree — skipped")
            continue

        best_sig, best_df, best_price = max(
            ex_sigs.values(),
            key=lambda x: x[0].confidence
        )

        num_agreeing = len(ex_sigs)

        if num_agreeing == num_exchanges:
            best_sig.confidence = min(
                100.0, round(best_sig.confidence * 1.1, 1)
            )
            best_sig.exchange = "confirmed"
            logger.info(
                f"{base}: cross-exchange confirmed — "
                f"conf boosted to {best_sig.confidence}%"
            )
        elif best_sig.confidence >= Config.SINGLE_EXCHANGE_MIN_CONFIDENCE:
            best_sig.exchange = "confirmed"
            logger.info(
                f"{base}: single exchange "
                f"conf={best_sig.confidence} — accepted"
            )
        else:
            logger.debug(
                f"{base}: conf={best_sig.confidence} "
                f"< {Config.SINGLE_EXCHANGE_MIN_CONFIDENCE} — skipped"
            )
            continue

        full_symbol = best_sig.symbol
        active_direction = lifecycle.get_active_direction(full_symbol)

        if active_direction is not None:
            if active_direction != best_sig.direction:
                if best_sig.confidence >= 80:
                    reversal_candidates.append(
                        (best_sig, best_df, best_price)
                    )
                    logger.info(
                        f"Reversal: {base} "
                        f"{active_direction} → {best_sig.direction} "
                        f"conf={best_sig.confidence}"
                    )
                else:
                    logger.info(
                        f"{base}: opposing conf={best_sig.confidence} "
                        f"too low — ignored"
                    )
            else:
                logger.info(f"{base}: same direction active — skipped")
            continue

        confirmed_signals.append((best_sig, best_df, best_price))

    logger.info(
        f"Confirmed: {len(confirmed_signals)} | "
        f"Reversals: {len(reversal_candidates)} | "
        f"Daily: {get_daily_count()}/{Config.MAX_SIGNALS_PER_DAY}"
    )

    # Send reversal alerts first
    for sig, df_5m, live_price in reversal_candidates:
        active_lc = lifecycle.load(sig.symbol)
        if active_lc:
            await send_reversal_alert(
                Config.TELEGRAM_BOT_TOKEN,
                Config.TELEGRAM_CHAT_ID,
                active_symbol=sig.symbol,
                active_direction=active_lc.direction,
                new_sig=sig,
                df=df_5m,
            )
            lifecycle.delete(sig.symbol)
            await asyncio.sleep(4)

    # Sort by confidence — best signal first
    confirmed_signals.sort(
        key=lambda x: x[0].confidence, reverse=True
    )

    # Send ONLY the single best signal this scan cycle
    # Bot enforces 45 min gap before next signal via Redis
    for sig, df_5m, live_price in confirmed_signals:
        if remaining_today() <= 0:
            break

        # Re-check spacing — another scan might have sent in parallel
        if not check_signal_spacing():
            logger.info("Signal spacing: skipping — too soon after last")
            break

        current_price = live_price or get_current_price(
            sig.symbol, sig.exchange
        )

        await send_signal(
            Config.TELEGRAM_BOT_TOKEN,
            Config.TELEGRAM_CHAT_ID,
            sig,
            df=df_5m,
        )

        # Record signal sent time — enforces 45 min gap
        record_signal_sent_time()

        lc = SignalLifecycle(
            symbol=sig.symbol,
            direction=sig.direction,
            strategy_type=sig.strategy_type,
            confidence=sig.confidence,
            entry=sig.entry,
            live_entry=current_price or sig.entry,
            stop_loss=sig.stop_loss,
            take_profit1=sig.take_profit1,
            take_profit2=sig.take_profit2,
            take_profit3=sig.take_profit3,
            leverage=sig.leverage,
            sl_pct=sig.sl_pct,
            tp1_pct=sig.tp1_pct,
            tp2_pct=sig.tp2_pct,
            tp3_pct=sig.tp3_pct,
            exchange=sig.exchange,
        )
        lifecycle.save(lc)
        lifecycle.mark_sent(sig.symbol, current_price or sig.entry)

        record_signal_sent(
            sig.symbol,
            sig.direction,
            sig.confidence,
            sig.strategy_type,
            sig.leverage,
        )

        increment_daily_count()

        logger.info(
            f"SIGNAL SENT: {sig.symbol} {sig.direction} "
            f"{sig.strategy_type} conf={sig.confidence} | "
            f"Daily: {get_daily_count()}/{Config.MAX_SIGNALS_PER_DAY} | "
            f"Next signal in 45 min"
        )

        # Only send ONE signal per scan — spacing handles the rest
        break


async def run_once():
    Config.validate()
    await check_and_send_summary()
    await scan_all_exchanges()
    await lifecycle.check_all(
        get_current_price,
        Config.TELEGRAM_BOT_TOKEN,
        Config.TELEGRAM_CHAT_ID,
    )


async def run_forever():
    while True:
        try:
            await run_once()
        except Exception as e:
            logger.exception(f"Scan cycle failed: {e}")
        logger.info(
            f"Sleeping {Config.SCAN_INTERVAL_SECONDS}s until next scan"
        )
        await asyncio.sleep(Config.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(run_forever())
