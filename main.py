import asyncio
import logging
from datetime import datetime, date

from config.settings import Config
from bot.exchanges import build_exchange, get_liquid_perp_symbols, fetch_dual_timeframe
from bot.signal_engine import build_signal, Signal
from bot.notifier import send_signal
from bot.tracker import SignalTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scanner")

STRICT_BASE_CURRENCIES = {"BTC"}
tracker = SignalTracker()
_exchange_cache: dict = {}

# Daily signal counter
_daily_count: dict[str, int] = {"date": str(date.today()), "count": 0}
MAX_SIGNALS_PER_DAY = 15
MAX_SIGNALS_PER_SCAN = 2


def _get_daily_count() -> int:
    today = str(date.today())
    if _daily_count["date"] != today:
        _daily_count["date"] = today
        _daily_count["count"] = 0
    return _daily_count["count"]


def _increment_daily_count():
    today = str(date.today())
    if _daily_count["date"] != today:
        _daily_count["date"] = today
        _daily_count["count"] = 0
    _daily_count["count"] += 1


def required_confidence(symbol: str) -> float:
    base = symbol.split("/")[0]
    if base in STRICT_BASE_CURRENCIES:
        return Config.MIN_CONFIDENCE_BTC
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
    sorts by confidence, and sends only the top MAX_SIGNALS_PER_SCAN.
    """
    if _get_daily_count() >= MAX_SIGNALS_PER_DAY:
        logger.info(f"Daily signal limit ({MAX_SIGNALS_PER_DAY}) reached — skipping scan")
        return

    all_candidates: list[tuple[Signal, object]] = []

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

    if not all_candidates:
        logger.info("No qualifying signals this cycle")
        return

    # Sort by confidence descending — best signals first
    all_candidates.sort(key=lambda x: x[0].confidence, reverse=True)

    # How many can we still send today
    remaining_today = MAX_SIGNALS_PER_DAY - _get_daily_count()
    to_send = all_candidates[:min(MAX_SIGNALS_PER_SCAN, remaining_today)]

    logger.info(
        f"Sending {len(to_send)} signal(s) this cycle "
        f"({_get_daily_count()}/{MAX_SIGNALS_PER_DAY} used today)"
    )

    for sig, df_15m in to_send:
        await send_signal(
            Config.TELEGRAM_BOT_TOKEN,
            Config.TELEGRAM_CHAT_ID,
            sig,
            df=df_15m,
        )
        tracker.add(sig)
        _increment_daily_count()
        logger.info(
            f"SIGNAL SENT: {sig.symbol} {sig.direction} "
            f"conf={sig.confidence} "
            f"({_get_daily_count()}/{MAX_SIGNALS_PER_DAY} today)"
        )
        # Delay between signals to avoid Telegram flood control
        await asyncio.sleep(5)


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
