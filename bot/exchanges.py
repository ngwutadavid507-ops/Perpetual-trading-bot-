"""
Unified exchange access using ccxt. Handles fetching tradeable perpetual
swap symbols and OHLCV candles for Bybit, OKX, and BingX (or any other
ccxt-supported exchange you add to EXCHANGES in .env).
"""

import logging
import ccxt
import pandas as pd

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
    Return USDT-margined perpetual swap symbols on this exchange whose
    24h quote volume clears the liquidity floor. This is the filter that
    keeps the bot from firing signals on illiquid, noisy pairs.
    """
    try:
        markets = exchange.load_markets()
    except Exception as e:
        logger.error(f"[{exchange.id}] failed to load markets: {e}")
        return []

    candidates = [
        symbol for symbol, m in markets.items()
        if m.get("swap") and m.get("linear") and m.get("settle") == "USDT" and m.get("active", True)
    ]

    if not candidates:
        return []

    try:
        tickers = exchange.fetch_tickers(candidates)
    except Exception as e:
        logger.warning(f"[{exchange.id}] bulk fetch_tickers failed ({e}); falling back to no-volume-filter list")
        return candidates

    liquid = []
    for symbol in candidates:
        t = tickers.get(symbol)
        if not t:
            continue
        quote_vol = t.get("quoteVolume") or t.get("baseVolume") or 0
        if quote_vol >= min_24h_volume_usdt:
            liquid.append(symbol)

    logger.info(f"[{exchange.id}] {len(liquid)}/{len(candidates)} symbols pass the {min_24h_volume_usdt:,.0f} USDT volume floor")
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
