"""
Fetches the top N cryptocurrencies by market cap from CoinGecko.
Falls back to a hardcoded top 100 list if CoinGecko is unavailable.
Cache persists for 4 hours to avoid rate limits.
"""

import logging
import time
import requests

logger = logging.getLogger(__name__)

_cache: tuple[float, set[str]] | None = None
CACHE_TTL_SECONDS = 4 * 60 * 60

# Hardcoded fallback — top 100 tokens by market cap
# Used when CoinGecko API is unavailable
FALLBACK_TOP_SYMBOLS = {
    "BTC", "ETH", "USDT", "BNB", "SOL", "XRP", "USDC", "DOGE", "ADA",
    "TRX", "AVAX", "SHIB", "TON", "LINK", "DOT", "BCH", "NEAR", "LTC",
    "UNI", "ICP", "DAI", "APT", "ATOM", "POL", "ETC", "XLM", "OP",
    "HBAR", "FIL", "ARB", "VET", "MKR", "IMX", "AAVE", "ALGO", "STX",
    "TAO", "SUI", "RENDER", "INJ", "GRT", "FTM", "SAND", "MANA", "AXS",
    "THETA", "XTZ", "EGLD", "FLOW", "KAVA", "NEO", "CHZ", "CRV", "ZEC",
    "COMP", "YFI", "SNX", "1INCH", "RUNE", "LDO", "CAKE", "DYDX", "ENJ",
    "BAT", "ZRX", "OCEAN", "BAND", "KNC", "REN", "NMR", "SUSHI", "UMA",
    "CELO", "SKL", "STORJ", "ANKR", "CTSI", "OGN", "PERP", "RARI", "SLP",
    "JASMY", "GALA", "ENS", "APE", "GMT", "LUNC", "LUNA", "HNT", "ROSE",
    "ONE", "ZIL", "IOTA", "XEM", "HOT", "SC", "BTT", "WIN", "FLOKI",
    "PEPE", "WIF", "BONK", "JTO", "PYTH", "JUP", "STRK", "TIA", "HYPE",
}


def get_top_symbols(limit: int = 200) -> set[str]:
    global _cache

    if _cache is not None:
        cached_at, symbols = _cache
        if time.time() - cached_at < CACHE_TTL_SECONDS:
            logger.info(f"[toplist] using cached list ({len(symbols)} symbols)")
            return symbols

    symbols = _fetch_from_coingecko(limit)

    if not symbols:
        if _cache is not None:
            logger.warning("[toplist] fetch failed — using stale cache")
            return _cache[1]
        logger.warning("[toplist] fetch failed — using hardcoded fallback list")
        return FALLBACK_TOP_SYMBOLS

    _cache = (time.time(), symbols)
    logger.info(f"[toplist] refreshed — {len(symbols)} top symbols loaded")
    return symbols


def _fetch_from_coingecko(limit: int) -> set[str]:
    symbols = set()
    per_page = min(limit, 250)
    page = 1

    while len(symbols) < limit:
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": False,
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            for coin in data:
                symbol = coin.get("symbol", "").upper()
                if symbol:
                    symbols.add(symbol)

            if len(data) < per_page:
                break

            page += 1

        except requests.exceptions.RequestException as e:
            logger.error(f"[toplist] CoinGecko request failed: {e}")
            break
        except Exception as e:
            logger.error(f"[toplist] unexpected error: {e}")
            break

    return symbols


def is_top_symbol(base_currency: str, top_symbols: set[str]) -> bool:
    return base_currency.upper() in top_symbols
