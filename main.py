"""
Main scanner with:
- Single exchange signals fire if confidence >= 75%
- Cross-exchange confirmation boosts confidence by 10%
- Reversal alerts for opposing signals on active trades
- AI filter selects best 1-2 signals per cycle
- Daily limit: max 15 signals per day
- Active trade protection per symbol
- Daily summary sent at midnight UTC
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from config.settings import Config
from bot.exchanges import build_exchange, get_liquid_perp_symbols, fetch_dual_timeframe
from bot.signal_engine import build_signal, Signal
from bot.notifier import send_signal
from bot.tracker import SignalTracker
from bot.ai_filter import ai_select_signals, get_daily_count, remaining_today, MAX_SIGNALS_PER_DAY
from bot.summary import record_signal_sent, record_result, format_daily_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scanner")

tracker = SignalTracker()
_exchange_cache: dict = {}
_last_summary_date: str = ""


def required_confidence(symbol: str) -> float:
    return Config.MIN_CONFIDENCE_DEFAULT


def get_current_price(symbol: str, exchange: str) -> float | None:
    try:
        ex = _exchange_cache.get(exchange)
        if not ex:
            for ex_obj in _exchange_cache.values():
                try:
                    ticker = ex_obj.fetch_ticker(symbol)
                    price = ticker.get("last")
                    if price:
                        return price
                except Exception:
                    continue
        else:
            ticker = ex.fetch_ticker(symbol)
            return ticker.get("last")
    except Exception as e:
        logger.debug(f"Price check failed for {symbol}: {e}")
    return None


def get_base_symbol(symbol: str) -> str:
    return symbol.split("/")[0]


async def send_daily_summary():
    """Sends the daily performance summary to Telegram."""
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
        logger.error(f"Failed to send daily summary: {e}")


async def check_and_send_summary():
    """Sends summary at midnight UTC once per day."""
    global _last_summary_date
    now = datetime.now(timezone.utc)
    today = str(now.date())

    # Send at midnight — between 00:00 and 00:10 UTC
    if now.hour == 0 and now.minute < 10 and _last_summary_date != today:
        _last_summary_date = today
        await send_daily_summary()


async def scan_all_exchanges():
    daily_remaining = remaining_today()
    if daily_remaining <= 0:
        logger.info(f"Daily limit reached ({MAX_SIGNALS_PER_DAY}) — skipping scan")
        return

    # {base_symbol: {exchange_id: (Signal, df_15m)}}
    exchange_signals: dict[str, dict[str, tuple]] = defaultdict(dict)

    for exchange_id in Config.EXCHANGES:
        exchange = build_exchange(exchange_id)
        _exchange_cache[exchange_id] = exchange

        symbols = get_liquid_perp_symbols(exchange, Config.MIN_24H_VOLUME_USDT)
        logger.info(f"[{exchange_id}] scanning {len(symbols)} liquid symbols")

        skipped = 0
        for symbol in symbols:
            try:
                df_1h, df_15m = fetch_dual_timeframe(exchange, symbol)
                if df_1h is None or df_15m is None:
                    skipped += 1
                    continue

                sig = build_signal(
                    symbol,
                    exchange_id,
                    df_1h,
                    df_15m,
                    Config.MIN_RISK_REWARD,
                    Config.SIGNAL_COOLDOWN_MINUTES,
                )
                if sig is None:
                    continue

                if sig.confidence < required_confidence(symbol):
                    continue

                base = get_base_symbol(symbol)
                exchange_signals[base][exchange_id] = (sig, df_15m)
                logger.info(
                    f"[{exchange_id}] candidate: {base} "
                    f"{sig.direction} conf={sig.confidence}"
                )

            except Exception as e:
                logger.error(f"[{exchange_id}] error on {symbol}: {e}")
                continue

        logger.info(f"[{exchange_id}] scan complete — {skipped} skipped")

    num_exchanges = len(Config.EXCHANGES)
    confirmed_signals: list[tuple] = []
    reversal_signals: list[tuple] = []

    for base, ex_sigs in exchange_signals.items():

        # Check direction agreement
        directions = [sig.direction for sig, _ in ex_sigs.values()]
        if len(set(directions)) > 1:
            logger.debug(f"{base}: exchanges disagree on direction — skipped")
            continue

        # Pick highest confidence signal
        best_sig, best_df = max(
            ex_sigs.values(),
            key=lambda x: x[0].confidence
        )

        num_agreeing = len(ex_sigs)

        if num_agreeing == num_exchanges:
            # All exchanges agree — boost confidence 10%
            best_sig.confidence = min(100.0, round(best_sig.confidence * 1.1, 1))
            best_sig.exchange = "confirmed"
            logger.info(
                f"{base}: cross-exchange confirmed — "
                f"confidence boosted to {best_sig.confidence}%"
            )
        elif best_sig.confidence >= 65:
            # Single exchange high confidence — accept
            best_sig.exchange = "confirmed"
            logger.info(
                f"{base}: single exchange conf={best_sig.confidence} — accepted"
            )
        else:
            logger.debug(
                f"{base}: single exchange conf={best_sig.confidence} "
                f"< 75 — skipped"
            )
            continue

        # Check active trade on this symbol
        full_symbol = best_sig.symbol
        if tracker.is_symbol_active(full_symbol):
            active = tracker.get_active_trade(full_symbol)
            if active and active.signal.direction != best_sig.direction:
                if best_sig.confidence >= 80:
                    reversal_signals.append((best_sig, best_df))
                    logger.info(
                        f"Reversal candidate: {base} "
                        f"{active.signal.direction} → {best_sig.direction} "
                        f"conf={best_sig.confidence}"
                    )
                else:
                    logger.info(
                        f"{base}: opposing conf={best_sig.confidence} "
                        f"too low for reversal — ignored"
                    )
            else:
                logger.info(f"{base}: same direction already active — skipped")
            continue

        confirmed_signals.append((best_sig, best_df))

    logger.info(
        f"Confirmed signals: {len(confirmed_signals)} | "
        f"Reversal alerts: {len(reversal_signals)} | "
        f"Daily: {get_daily_count()}/{MAX_SIGNALS_PER_DAY}"
    )

    # Send reversal alerts first
    for sig, df_15m in reversal_signals:
        active_tracked = tracker.get_active_trade(sig.symbol)
        if active_tracked:
            was_sent = await tracker.handle_reversal(
                sig,
                Config.TELEGRAM_BOT_TOKEN,
                Config.TELEGRAM_CHAT_ID,
            )
            if was_sent:
                await asyncio.sleep(4)

    # AI filter selects best signals
    if confirmed_signals:
        selected = ai_select_signals(confirmed_signals)
        for sig, df_15m in selected:
            await send_signal(
                Config.TELEGRAM_BOT_TOKEN,
                Config.TELEGRAM_CHAT_ID,
                sig,
                df=df_15m,
            )
            tracker.add(sig)
            record_signal_sent(sig.symbol, sig.direction, sig.confidence)
            logger.info(
                f"SIGNAL SENT: {sig.symbol} {sig.direction} "
                f"conf={sig.confidence} | "
                f"Daily: {get_daily_count()}/{MAX_SIGNALS_PER_DAY}"
            )
            await asyncio.sleep(4)


async def run_once():
    Config.validate()
    await check_and_send_summary()
    await scan_all_exchanges()
    await tracker.check_all(
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
        logger.info(f"Sleeping {Config.SCAN_INTERVAL_SECONDS}s until next scan")
        await asyncio.sleep(Config.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(run_forever())
