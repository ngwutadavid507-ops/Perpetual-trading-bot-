"""
Main scanner with:
- Cross-exchange confirmation: signal only fires when OKX AND BingX agree
- Reversal alerts: opposing high-confidence signal on active trade
- AI filter: selects best 1-2 signals per cycle
- Daily limit: max 15 signals per day
- Active trade protection: no new signal on symbol already in trade
"""

import asyncio
import logging
from collections import defaultdict

from config.settings import Config
from bot.exchanges import build_exchange, get_liquid_perp_symbols, fetch_dual_timeframe
from bot.signal_engine import build_signal, Signal
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


def get_current_price(symbol: str, exchange: str) -> float | None:
    try:
        ex = _exchange_cache.get(exchange)
        if not ex:
            return None
        ticker = ex.fetch_ticker(symbol)
        return ticker.get("last")
    except Exception as e:
        logger.debug(f"Price check failed for {symbol}: {e}")
        return None


def get_base_symbol(symbol: str) -> str:
    """Extracts base symbol e.g. ETH from ETH/USDT:USDT"""
    return symbol.split("/")[0]


async def scan_all_exchanges():
    """
    Scans all exchanges and collects signals per base symbol.
    Only fires a signal when ALL configured exchanges agree on direction.
    Handles reversal alerts for active trades.
    """
    daily_remaining = remaining_today()
    if daily_remaining <= 0:
        logger.info(f"Daily limit reached ({MAX_SIGNALS_PER_DAY}) — skipping scan")
        return

    # Structure: {base_symbol: {exchange_id: (Signal, df_15m)}}
    exchange_signals: dict[str, dict[str, tuple[Signal, object]]] = defaultdict(dict)

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

    # Cross-exchange confirmation
    # Signal only qualifies if ALL exchanges agree on same direction
    confirmed_signals: list[tuple[Signal, object]] = []
    reversal_signals: list[tuple[Signal, object]] = []

    num_exchanges = len(Config.EXCHANGES)

    for base, ex_sigs in exchange_signals.items():
        if len(ex_sigs) < num_exchanges:
            # Not all exchanges detected a signal — skip
            logger.debug(
                f"{base}: only {len(ex_sigs)}/{num_exchanges} "
                f"exchanges agree — skipped"
            )
            continue

        # Check all exchanges agree on same direction
        directions = [sig.direction for sig, _ in ex_sigs.values()]
        if len(set(directions)) > 1:
            logger.debug(f"{base}: exchanges disagree on direction — skipped")
            continue

        # All agree — pick the signal with highest confidence
        best_sig, best_df = max(
            ex_sigs.values(),
            key=lambda x: x[0].confidence
        )

        # Remove exchange label from symbol display
        best_sig.exchange = "confirmed"

        # Check if this symbol has an active trade
        full_symbol = best_sig.symbol
        if tracker.is_symbol_active(full_symbol):
            active = tracker.get_active_trade(full_symbol)
            if active and active.signal.direction != best_sig.direction:
                # Opposing signal on active trade — check if strong enough for reversal
                if best_sig.confidence >= 80:
                    reversal_signals.append((best_sig, best_df))
                    logger.info(
                        f"Reversal candidate: {base} "
                        f"{active.signal.direction} → {best_sig.direction} "
                        f"conf={best_sig.confidence}"
                    )
                else:
                    logger.info(
                        f"{base}: opposing signal conf={best_sig.confidence} "
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

    # Send reversal alerts first — these are urgent
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

    # AI filter selects best confirmed signals
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
