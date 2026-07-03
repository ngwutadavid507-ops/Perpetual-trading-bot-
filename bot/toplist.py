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

# Cache: (timestamp, set of base currency symbols)
_cache: tuple[float, set[str]] | None = None
CACHE_TTL_SECONDS = 4 * 60 * 60  # refresh every 4 hours


def get_top_symbols(limit: int = 200) -> set[str]:
    """
    Returns a set of base currency symbols for the top N coins
    by market cap e.g. {'BTC', 'ETH', 'SOL', 'BNB', ...}
    """
    global _cache

    # Return cached list if still fresh
    if _cache is not None:
        cached_at, symbols = _cache
        if time.time() - cached_at < CACHE_TTL_SECONDS:
            logger.info(f"[toplist] using cached list ({len(symbols)} symbols)")
            return symbols

    symbols = _fetch_from_coingecko(limit)

    if not symbols:
        # If fetch failed and we have a stale cache, use it rather than blocking all scans
        if _cache is not None:
            logger.warning("[toplist] fetch failed, using stale cache")
            return _cache[1]
        # No cache at all — return empty set so caller can decide what to do
        logger.error("[toplist] fetch failed and no cache available")
        return set()

    _cache = (time.time(), symbols)
    logger.info(f"[toplist] refreshed — {len(symbols)} top symbols loaded")
    return symbols


def _fetch_from_coingecko(limit: int) -> set[str]:
    """
    Calls CoinGecko free API to get top coins by market cap.
    No API key needed for this endpoint.
    Handles pagination since CoinGecko returns max 250 per page.
    """
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
    """Check if a base currency is in the top list."""
    return base_currency.upper() in top_symbols
