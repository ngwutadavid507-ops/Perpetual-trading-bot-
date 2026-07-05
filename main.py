import asyncio
import logging

from config.settings import Config
from bot.exchanges import build_exchange, get_liquid_perp_symbols, fetch_dual_timeframe
from bot.signal_engine import build_signal
from bot.notifier import send_signal
from bot.tracker import SignalTracker
from bot.ai_filter import ai_select_signals, get_daily_count, remaining_today, MAX_SIGNALS_PER_DAY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scanner")

tracker = SignalTracker()
_exchange_cache: dict = {}


def required_confidence(symbol: str) -> float:
    return Config.MIN_CONFIDENCE_DEFAULT


def get_current_price(symbol: str, exchange_id: str) -> float | None:
    try:
        exchange = _exchange_cache.get(exchange_id)
        if not exchange:
            return None
        ticker = exchange.fetch_ticker(symbol)
        return ticker.get("last")
    except Exception as e:
        logger.debug(f"Price check failed for {symbol} on {exchange_id}: {e}")
        return None


async def scan_all_exchanges():
    """
    Scans all exchanges, collects all qualifying signals,
    passes them to Claude AI filter which selects the best
    1-2 per cycle while enforcing the daily limit of 15.
    """
    daily_remaining = remaining_today()
    if daily_remaining <= 0:
        logger.info(
            f"Daily signal limit reached "
            f"({MAX_SIGNALS_PER_DAY}/{MAX_SIGNALS_PER_DAY}) — skipping scan"
        )
        return

    all_candidates = []

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

                threshold = required_confidence(symbol)
                if sig.confidence < threshold:
                    continue

                all_candidates.append((sig, df_15m))
                logger.info(
                    f"[{exchange_id}] candidate: {symbol} "
                    f"{sig.direction} conf={sig.confidence}"
                )

            except Exception as e:
                logger.error(f"[{exchange_id}] error processing {symbol}: {e}")
                continue

        logger.info(
            f"[{exchange_id}] scan complete — "
            f"{skipped} symbols skipped (no data)"
        )

    logger.info(
        f"Total candidates this cycle: {len(all_candidates)} | "
        f"Daily: {get_daily_count()}/{MAX_SIGNALS_PER_DAY}"
    )

    if not all_candidates:
        logger.info("No qualifying signals this cycle")
        return

    # AI filter selects the best signals and enforces limits
    selected = ai_select_signals(all_candidates)

    for sig, df_15m in selected:
        await send_signal(
            Config.TELEGRAM_BOT_TOKEN,
            Config.TELEGRAM_CHAT_ID,
            sig,
            df=df_15m,
        )
        tracker.add(sig)
        logger.info(
            f"SIGNAL SENT: {sig.symbol} {sig.direction} "
            f"conf={sig.confidence} | "
            f"Daily: {get_daily_count()}/{MAX_SIGNALS_PER_DAY}"
        )
        await asyncio.sleep(4)


async def run_once():
    Config.validate()
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
