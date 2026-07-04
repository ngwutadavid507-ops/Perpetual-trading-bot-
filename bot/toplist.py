"""
Fetches the top N cryptocurrencies by market cap from CoinGecko's
free API (no API key required). Used to filter exchange symbols so
the bot only scans high quality, liquid, well-known tokens.
The list refreshes every 4 hours to stay current without hammering
the free tier rate limits.
"""

import logging
import time
import requests

logger = logging.getLogger(__name__)

_cache: tuple[float, set[str]] | None = None
CACHE_TTL_SECONDS = 4 * 60 * 60


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
            logger.warning("[toplist] fetch failed, using stale cache")
            return _cache[1]
        logger.error("[toplist] fetch failed and no cache available")
        return set()

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
