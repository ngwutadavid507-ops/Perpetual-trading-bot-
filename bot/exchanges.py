"""
Unified exchange access using ccxt. Fetches both 1h and 15m candles
for multi-timeframe signal analysis. Filters out options, futures with
expiry, synthetic/forex tokens, and known junk tokens.
Only scans USDT linear perpetual swaps in the top symbol list.
"""

import logging
import ccxt
import pandas as pd

from bot.toplist import get_top_symbols, is_top_symbol

logger = logging.getLogger(__name__)

EXCHANGE_CLASS_MAP = {
    "bybit": (ccxt.bybit, {
        "enableRateLimit": True,
        "options": {"defaultType": "linear"},
    }),
    "okx": (ccxt.okx, {
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "defaultSubType": "linear",
        },
    }),
    "bingx": (ccxt.bingx, {
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }),
}

# Tokens to always block regardless of top list
JUNK_PREFIXES = [
    "NC", "EUR", "GBP", "SGD", "JPY", "AUD",
    "USD2", "2USD", "BVOL", "DVOL", "PIE", "WLFI",
]

JUNK_EXACT = {
    "PIEVERSE", "WLFI", "EURI", "EURC", "FDUSD",
    "TUSD", "BUSD", "USDP", "GUSD", "HUSD",
}


def build_exchange(exchange_id: str):
    if exchange_id not in EXCHANGE_CLASS_MAP:
        raise ValueError(f"Unsupported exchange '{exchange_id}'.")
    klass, config = EXCHANGE_CLASS_MAP[exchange_id]
    return klass(config)


def get_liquid_perp_symbols(exchange, min_24h_volume_usdt: float) -> list[str]:
    top_symbols = get_top_symbols(limit=200)

    try:
        markets = exchange.load_markets()
    except Exception as e:
        logger.error(f"[{exchange.id}] failed to load markets: {e}")
        return []

    candidates = []
    for symbol, m in markets.items():

        # Must be active linear USDT-settled perpetual swap
        if not m.get("swap"):
            continue
        if not m.get("linear"):
            continue
        if m.get("settle") != "USDT":
            continue
        if not m.get("active", True):
            continue

        # Block options and dated contracts
        if m.get("type") == "option":
            continue
        if m.get("expiry") is not None:
            continue
        if m.get("expiryDatetime") is not None:
            continue

        base = m.get("base", "").upper()

        # Block junk by prefix
        if any(base.startswith(j) for j in JUNK_PREFIXES):
            continue

        # Block junk by exact match
        if base in JUNK_EXACT:
            continue

        # Only allow top symbols
        if top_symbols and not is_top_symbol(base, top_symbols):
            continue

        candidates.append(symbol)

    if not candidates:
        logger.warning(f"[{exchange.id}] no candidates after filtering")
        return []

    try:
        tickers = exchange.fetch_tickers(candidates)
    except Exception as e:
        logger.warning(f"[{exchange.id}] bulk fetch_tickers failed ({e})")
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
        f"[{exchange.id}] {len(liquid)}/{len(candidates)} symbols "
        f"pass the {min_24h_volume_usdt:,.0f} USDT volume floor"
    )
    return liquid


def fetch_ohlcv_df(
    exchange,
    symbol: str,
    timeframe: str,
    limit: int = 150,
) -> pd.DataFrame | None:
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        logger.debug(f"[{exchange.id}] fetch_ohlcv failed for {symbol}: {e}")
        return None

    if not raw or len(raw) < 50:
        return None

    df = pd.DataFrame(
        raw,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def fetch_dual_timeframe(
    exchange,
    symbol: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetches both 1h and 15m candles for a symbol."""
    df_1h = fetch_ohlcv_df(exchange, symbol, "1h", limit=150)
    df_15m = fetch_ohlcv_df(exchange, symbol, "15m", limit=100)
    return df_1h, df_15m
