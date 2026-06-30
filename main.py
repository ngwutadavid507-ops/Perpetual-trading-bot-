"""
Main entry point. Scans all configured exchanges and their liquid USDT
perpetual symbols on a schedule, scores each for a signal, and pushes
qualifying signals to Telegram. BTC requires a stricter confidence bar
than other symbols, as configured in .env.
"""

import asyncio
import logging

from config.settings import Config
from bot.exchanges import build_exchange, get_liquid_perp_symbols, fetch_ohlcv_df
from bot.signal_engine import build_signal
from bot.notifier import send_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scanner")

# Base currencies that require a stricter confidence threshold
STRICT_BASE_CURRENCIES = {"BTC"}


def required_confidence(symbol: str) -> float:
    base = symbol.split("/")[0]
    if base in STRICT_BASE_CURRENCIES:
        return Config.MIN_CONFIDENCE_BTC
    return Config.MIN_CONFIDENCE_DEFAULT


async def scan_exchange(exchange_id: str):
    exchange = build_exchange(exchange_id)
    symbols = get_liquid_perp_symbols(exchange, Config.MIN_24H_VOLUME_USDT)
    logger.info(f"[{exchange_id}] scanning {len(symbols)} liquid symbols")

    fired = 0
    for symbol in symbols:
        df = fetch_ohlcv_df(exchange, symbol, Config.TIMEFRAME)
        if df is None:
            continue

        sig = build_signal(symbol, exchange_id, df, Config.MIN_RISK_REWARD)
        if sig is None:
            continue

        threshold = required_confidence(symbol)
        if sig.confidence < threshold:
            continue

        await send_signal(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID, sig)
        logger.info(f"[{exchange_id}] SIGNAL FIRED: {symbol} {sig.direction} conf={sig.confidence}")
        fired += 1

    logger.info(f"[{exchange_id}] scan complete, {fired} signals fired")


async def run_once():
    Config.validate()
    tasks = [scan_exchange(ex) for ex in Config.EXCHANGES]
    await asyncio.gather(*tasks, return_exceptions=True)


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
