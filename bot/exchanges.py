"""
Unified exchange access using ccxt.
Fetches 1h for trend and 5m for entry signals.
Includes live price fetching via ticker for signal validation.
Filters out options, synthetics, forex and junk tokens.
Only scans USDT linear perpetual swaps in top symbol list.
"""

import logging
import ccxt
import pandas as pd

from bot.toplist import get_top_symbols, is_top_symbol
from config.settings import Config

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


def build_exchange(exchange_id: str):
    if exchange_id not in EXCHANGE_CLASS_MAP:
        raise ValueError(f"Unsupported exchange '{exchange_id}'.")
    klass, config = EXCHANGE_CLASS_MAP[exchange_id]
    return klass(config)


def get_liquid_perp_symbols(exchange, min_24h_volume_usdt: float) -> list[str]:
    """
    Returns USDT linear perp symbols that:
    1. Pass junk token filters
    2. Are in CoinGecko top 200
    3. Have sufficient 24h volume
    """
    top_symbols = get_top_symbols(limit=200)

    try:
        markets = exchange.load_markets()
    except Exception as e:
        logger.error(f"[{exchange.id}] failed to load markets: {e}")
        return []

    candidates = []
    for symbol, m in markets.items():
        if not m.get("swap"):
            continue
        if not m.get("linear"):
            continue
        if m.get("settle") != "USDT":
            continue
        if not m.get("active", True):
            continue
        if m.get("type") == "option":
            continue
        if m.get("expiry") is not None:
            continue
        if m.get("expiryDatetime") is not None:
            continue

        base = m.get("base", "").upper()

        if any(base.startswith(j) for j in Config.JUNK_PREFIXES):
            continue
        if base in Config.JUNK_EXACT:
            continue
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
    """Fetch OHLCV candles and return as DataFrame."""
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
    """
    Fetches 1h candles for trend direction and S/R levels.
    Fetches 5m candles for entry timing and pattern detection.
    """
    df_1h = fetch_ohlcv_df(exchange, symbol, "1h", limit=150)
    df_5m = fetch_ohlcv_df(exchange, symbol, "5m", limit=100)
    return df_1h, df_5m


def fetch_live_price(exchange, symbol: str) -> float | None:
    """
    Fetches the current live market price for a symbol.
    Used to validate signal entry price before sending.
    Much faster than fetching OHLCV — single ticker call.
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker.get("last")
        if price:
            return float(price)
        # Fallback to bid/ask midpoint
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        if bid and ask:
            return round((float(bid) + float(ask)) / 2, 8)
        return None
    except Exception as e:
        logger.debug(f"[{exchange.id}] live price failed for {symbol}: {e}")
        return None
