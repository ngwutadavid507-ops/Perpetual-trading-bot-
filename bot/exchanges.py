"""
Unified exchange access using ccxt. Handles fetching tradeable perpetual
swap symbols and OHLCV candles for OKX and BingX.
Filters symbols against CoinGecko top list to ensure only high quality
liquid tokens are scanned.
"""

import logging
import ccxt
import pandas as pd

from bot.toplist import get_top_symbols, is_top_symbol

logger = logging.getLogger(__name__)

EXCHANGE_CLASS_MAP = {
    "bybit": (ccxt.bybit, {
        "enableRateLimit": True,
        "options": {
            "defaultType": "linear",
        }
    }),
    "okx": (ccxt.okx, {
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "defaultSubType": "linear",
        }
    }),
    "bingx": (ccxt.bingx, {
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        }
    }),
}


def build_exchange(exchange_id: str):
    """Instantiate a ccxt exchange client configured for USDT perpetual swaps."""
    if exchange_id not in EXCHANGE_CLASS_MAP:
        raise ValueError(f"Unsupported exchange '{exchange_id}'. Add it to EXCHANGE_CLASS_MAP.")
    klass, config = EXCHANGE_CLASS_MAP[exchange_id]
    return klass(config)


def get_liquid_perp_symbols(exchange, min_24h_volume_usdt: float) -> list[str]:
    """
    Return USDT-margined perpetual swap symbols that:
    1. Are in the CoinGecko top 200 by market cap
    2. Clear the 24h volume floor
    3. Are not synthetic/forex/commodity tokens
    """
    # Fetch top list once — cached for 4 hours
    top_symbols = get_top_symbols(limit=200)

    try:
        markets = exchange.load_markets()
    except Exception as e:
        logger.error(f"[{exchange.id}] failed to load markets: {e}")
        return []

    # Junk token filter — blocks BingX synthetics and forex pairs
    junk_prefixes = [
        "NC", "EUR", "GBP", "SGD", "JPY",
        "AUD", "USD2", "2USD", "BVOL", "DVOL"
    ]

    candidates = []
    for symbol, m in markets.items():
        if not (m.get("swap") and m.get("linear") and
                m.get("settle") == "USDT" and m.get("active", True)):
            continue

        base = m.get("base", "").upper()

        # Block junk tokens
        if any(base.startswith(junk) for junk in junk_prefixes):
            continue

        # Only allow top 200 tokens
        if top_symbols and not is_top_symbol(base, top_symbols):
            continue

        candidates.append(symbol)

    if not candidates:
        logger.warning(f"[{exchange.id}] no candidates after top list filter")
        return []

    try:
        tickers = exchange.fetch_tickers(candidates)
    except Exception as e:
        logger.warning(f"[{exchange.id}] bulk fetch_tickers failed ({e}); skipping volume filter")
        return candidates

    liquid = []
    for symbol in candidates:
        t = tickers.get(symbol)
        if not t:
            continue
        quote_vol = t.get("quoteVolume") or t.get("baseVolume") or 0
        if quote_vol >= min_24h_volume_usdt:
            liquid.append(symbol)

    logger.info(
        f"[{exchange.id}] {len(liquid)}/{len(candidates)} top-200 symbols "
        f"pass the {min_24h_volume_usdt:,.0f} USDT volume floor"
    )
    return liquid


def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int = 150) -> pd.DataFrame | None:
    """Fetch OHLCV candles and return as a pandas DataFrame, or None on failure."""
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        logger.debug(f"[{exchange.id}] fetch_ohlcv failed for {symbol}: {e}")
        return None

    if not raw or len(raw) < 50:
        return None

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
